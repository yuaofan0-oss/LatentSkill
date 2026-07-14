from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from math import sqrt
from torch.utils.checkpoint import checkpoint

from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithPast,
    Cache,
    CausalLMOutputWithPast,
    DynamicCache,
    FlashAttentionKwargs,
    GenerationMixin,
    GradientCheckpointingLayer,
    PreTrainedModel,
    Qwen3Attention,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    TransformersKwargs,
    Unpack,
    apply_rotary_pos_emb,
    auto_docstring,
    can_return_tuple,
    check_model_inputs,
    create_causal_mask,
    create_sliding_window_causal_mask,
    deprecate_kwarg,
    eager_attention_forward,
)


class LatentSkillLoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, input: Tensor, adapter_state=None) -> Tensor:
        base = F.linear(input, self.weight, self.bias)
        if adapter_state is None:
            return base

        A = adapter_state["A"]              # [Lb, in, r]
        B = adapter_state["B"]              # [Lb, r, out]
        C = adapter_state.get("C", None)    # [Lb, out] or None

        Lb = A.shape[0]
        torch._assert(input.shape[-1] == self.in_features, "last dim must be in_features")
        torch._assert(input.shape[0] % Lb == 0, "input batch must be multiple of lora batch")
        num_beams = input.shape[0] // Lb

        # Flatten all middle dims (e.g., seq_len) into S for faster matmul
        # input: [B, ..., in] -> x: [Lb, beams, S, in]
        x = input.reshape(Lb, num_beams, -1, self.in_features)

        # [Lb, beams, S, in] @ [Lb, in, r] -> [Lb, beams, S, r]
        tmp = torch.matmul(x, A[:, None, :, :])
        # [Lb, beams, S, r] @ [Lb, r, out] -> [Lb, beams, S, out]
        lora_out = torch.matmul(tmp, B[:, None, :, :])

        if self.bias is None:
            torch._assert(C is None, "If bias is None, adapter_state['C'] must also be None")
        else:
            torch._assert(C is not None, "If bias is not None, adapter_state['C'] must also be not None")
            # C: [Lb, out] -> [Lb, 1, 1, out] broadcast across beams and S
            lora_out = lora_out + C[:, None, None, :]

        # Restore original middle dims: [Lb*beams, ..., out]
        lora_out = lora_out.reshape(*input.shape[:-1], self.out_features)

        return base + lora_out

    def adapter_params_numel(self, r):
        if not hasattr(self, "adapter_params_numel_cache"):
            self.adapter_params_numel_cache = (
                self.in_features * r
                + self.out_features * r
                + (self.out_features if self.bias is not None else 0)
            )
        return self.adapter_params_numel_cache

    def configure_adapter_builder(self, method):
        if method == "rl":
            def adapter_builder(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, self.in_features, r) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, r, self.out_features) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        elif method == "rr":
            def adapter_builder(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, self.in_features, r) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, self.out_features, r).transpose(-1, -2) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        elif method == "lr":
            def adapter_builder(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, r, self.in_features).transpose(-1, -2) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, self.out_features, r).transpose(-1, -2) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        elif method == "ll":
            def adapter_builder(r, scale, plain_tensor):
                idx = 0
                A = plain_tensor[:, idx: idx + self.in_features * r].view(-1, r, self.in_features).transpose(-1, -2) * sqrt(scale)
                idx += self.in_features * r
                B = plain_tensor[:, idx: idx + self.out_features * r].view(-1, r, self.out_features) * sqrt(scale)
                idx += self.out_features * r
                C = plain_tensor[:, idx: idx + self.out_features].view(-1, self.out_features) * scale if self.bias is not None else None
                return {"A": A, "B": B, "C": C}

        else:
            raise NotImplementedError(f"LoRA method {method} not implemented")

        self.adapter_builder = adapter_builder

    def build_adapter_state(self, r, scale, plain_tensor):
        torch._assert(
            plain_tensor.shape[-1] == self.adapter_params_numel(r),
            "plain_tensor last dim does not match adapter_params_numel"
        )
        torch._assert(hasattr(self, "adapter_builder"), "adapter_builder not set")
        return self.adapter_builder(r, scale, plain_tensor)

    def init_adapter_state(self, r, scale, device):
        assert r > 0, "r must be positive"
        A = (torch.randn(size=(1, self.in_features, r), device=device) * sqrt(scale)).detach()
        A.requires_grad_()
        B = torch.zeros(size=(1, r, self.out_features), requires_grad=True, device=device)
        C = torch.zeros(size=(1, self.out_features), requires_grad=True, device=device) if self.bias is not None else None
        return {"A": A, "B": B, "C": C}

    def partition_adapter_params(self, r, idx_start):
        A_numel = self.in_features * r
        B_numel = self.out_features * r
        assert self.bias is None
        idx_range = [idx_start, A_numel + idx_start]
        return idx_range, A_numel + B_numel + idx_start


class LatentSkillQwen3MLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        self.gate_proj = LatentSkillLoRALinear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = LatentSkillLoRALinear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = LatentSkillLoRALinear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x, adapter_state=None):
        if adapter_state is None:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x, adapter_state["gate"])) * self.up_proj(x, adapter_state["up"]), adapter_state["down"])
        return down_proj

    def adapter_params_numel(self, r):
        if not hasattr(self, "adapter_params_numel_cache"):
            self.adapter_params_numel_cache = self.gate_proj.adapter_params_numel(r) + self.up_proj.adapter_params_numel(r) + self.down_proj.adapter_params_numel(r)
        return self.adapter_params_numel_cache

    def configure_adapter_builder(self, method):
        self.gate_proj.configure_adapter_builder(method)
        self.up_proj.configure_adapter_builder(method)
        self.down_proj.configure_adapter_builder(method)

    def build_adapter_state(self, r, scale, plain_tensor):
        assert plain_tensor.shape[-1] == self.adapter_params_numel(r), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match adapter_params_numel {self.adapter_params_numel(r)}"
        idx = 0
        gate = self.gate_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.gate_proj.adapter_params_numel(r)])
        idx += self.gate_proj.adapter_params_numel(r)
        up = self.up_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.up_proj.adapter_params_numel(r)])
        idx += self.up_proj.adapter_params_numel(r)
        down = self.down_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.down_proj.adapter_params_numel(r)])
        return {"gate": gate, "up": up, "down": down}

    def init_adapter_state(self, r, scale, device):
        gate = self.gate_proj.init_adapter_state(r, scale, device)
        up = self.up_proj.init_adapter_state(r, scale, device)
        down = self.down_proj.init_adapter_state(r, scale, device)
        return {"gate": gate, "up": up, "down": down}

    def partition_adapter_params(self, r, idx_start):
        idx_range = []
        idx_range_gate, next_start = self.gate_proj.partition_adapter_params(r, idx_start)
        idx_range += idx_range_gate
        idx_range_up, next_start = self.up_proj.partition_adapter_params(r, next_start)
        idx_range += idx_range_up
        idx_range_down, next_start = self.down_proj.partition_adapter_params(r, next_start)
        idx_range += idx_range_down
        return idx_range, next_start

class LatentSkillQwen3Attention(Qwen3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.q_proj = LatentSkillLoRALinear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = LatentSkillLoRALinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = LatentSkillLoRALinear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = LatentSkillLoRALinear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.bias = config.attention_bias

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        adapter_state = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if adapter_state is None:
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_norm(self.q_proj(hidden_states, adapter_state["q"]).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states, adapter_state["k"]).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states, adapter_state["v"]).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if adapter_state is None:
            attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output, adapter_state["o"])
        return attn_output, attn_weights

    def adapter_params_numel(self, r):
        if not hasattr(self, "adapter_params_numel_cache"):
            self.adapter_params_numel_cache = 0
            self.adapter_params_numel_cache += self.q_proj.adapter_params_numel(r)
            self.adapter_params_numel_cache += self.k_proj.adapter_params_numel(r)
            self.adapter_params_numel_cache += self.v_proj.adapter_params_numel(r)
            self.adapter_params_numel_cache += self.o_proj.adapter_params_numel(r)
        return self.adapter_params_numel_cache

    def configure_adapter_builder(self, method):
        self.q_proj.configure_adapter_builder(method)
        self.k_proj.configure_adapter_builder(method)
        self.v_proj.configure_adapter_builder(method)
        self.o_proj.configure_adapter_builder(method)

    def build_adapter_state(self, r, scale, plain_tensor):
        assert plain_tensor.shape[-1] == self.adapter_params_numel(r), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match adapter_params_numel {self.adapter_params_numel(r)}"
        idx = 0
        q = self.q_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.q_proj.adapter_params_numel(r)])
        idx += self.q_proj.adapter_params_numel(r)
        k = self.k_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.k_proj.adapter_params_numel(r)])
        idx += self.k_proj.adapter_params_numel(r)
        v = self.v_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.v_proj.adapter_params_numel(r)])
        idx += self.v_proj.adapter_params_numel(r)
        o = self.o_proj.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.o_proj.adapter_params_numel(r)])
        return {"q": q, "k": k, "v": v, "o": o}

    def init_adapter_state(self, r, scale, device):
        q = self.q_proj.init_adapter_state(r, scale, device)
        k = self.k_proj.init_adapter_state(r, scale, device)
        v = self.v_proj.init_adapter_state(r, scale, device)
        o = self.o_proj.init_adapter_state(r, scale, device)
        return {"q": q, "k": k, "v": v, "o": o}

    def partition_adapter_params(self, r, idx_start):
        idx_range = []
        idx_range_q, next_start = self.q_proj.partition_adapter_params(r, idx_start)
        idx_range += idx_range_q
        idx_range_k, next_start = self.k_proj.partition_adapter_params(r, next_start)
        idx_range += idx_range_k
        idx_range_v, next_start = self.v_proj.partition_adapter_params(r, next_start)
        idx_range += idx_range_v
        idx_range_o, next_start = self.o_proj.partition_adapter_params(r, next_start)
        idx_range += idx_range_o
        return idx_range, next_start

class LatentSkillBaseDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class LatentSkillQwen3DecoderLayer(LatentSkillBaseDecoderLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = LatentSkillQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = LatentSkillQwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        adapter_state: Optional[dict] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            adapter_state=adapter_state['attention'] if adapter_state is not None else None,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_state=adapter_state['mlp'] if adapter_state is not None else None)
        hidden_states = residual + hidden_states
        return hidden_states

    def adapter_params_numel(self, r):
        if not hasattr(self, "adapter_params_numel_cache"):
            self.adapter_params_numel_cache = 0
            self.adapter_params_numel_cache += self.self_attn.adapter_params_numel(r)
            self.adapter_params_numel_cache += self.mlp.adapter_params_numel(r)
        return self.adapter_params_numel_cache

    def configure_adapter_builder(self, method):
        self.self_attn.configure_adapter_builder(method)
        self.mlp.configure_adapter_builder(method)

    def build_adapter_state(self, r, scale, plain_tensor):
        assert plain_tensor.shape[-1] == self.adapter_params_numel(r), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match adapter_params_numel {self.adapter_params_numel(r)}"
        idx = 0
        attention = self.self_attn.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.self_attn.adapter_params_numel(r)])
        idx += self.self_attn.adapter_params_numel(r)
        mlp = self.mlp.build_adapter_state(r, scale, plain_tensor[:, idx: idx + self.mlp.adapter_params_numel(r)])
        return {"attention": attention, "mlp": mlp}

    def init_adapter_state(self, r, scale, device):
        return {"attention": self.self_attn.init_adapter_state(r, scale, device), "mlp": self.mlp.init_adapter_state(r, scale, device)}

    def partition_adapter_params(self, r, idx_start):
        idx_range = []
        idx_range_attn, next_start = self.self_attn.partition_adapter_params(r, idx_start)
        idx_range += idx_range_attn
        idx_range_mlp, next_start = self.mlp.partition_adapter_params(r, next_start)
        idx_range += idx_range_mlp
        return idx_range, next_start

@auto_docstring
class LatentSkillPreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LatentSkillQwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LatentSkillQwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }

@dataclass
class LatentSkillModelOutputWithPast(BaseModelOutputWithPast):
    """Base-model output that also carries per-layer memory-token states."""

    memory_states: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.FloatTensor] = None


@auto_docstring
class LatentSkillQwen3Model(LatentSkillPreTrainedModel):
    """Qwen3 backbone with MetaLoRA injection and optional memory-token states."""

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if config.num_mem_token == -1:
            self.use_mem_token = False
        else:
            self.use_mem_token = True
            self.num_mem_token = config.num_mem_token
            self.mem_tokens = nn.Parameter(torch.zeros((self.num_mem_token, config.hidden_size), requires_grad=True), requires_grad=True)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LatentSkillQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def reset_mem_tokens(self):
        if self.use_mem_token:
            nn.init.zeros_(self.mem_tokens)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        adapter_state: Optional[dict] = None,
        ignore_mem_token: bool = False,
        use_gradient_checkpoint: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        adapter_state (`dict` of `dict` of `torch.FloatTensor`, *optional*):
            A dictionary that maps each layer to its corresponding LoRA parameters. Each layer's LoRA parameters are
            stored in a nested dictionary.
        ignore_mem_token (`bool`, *optional*, defaults to `False`):
            Whether to ignore the memory tokens during the forward pass. If set to `True`, the memory tokens will not be
            used, and the model will behave like a standard transformer without memory tokens.
        use_gradient_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing to save memory during training. If set to `True`, the model will
            recompute certain activations during the backward pass instead of storing them in memory.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            if self.use_mem_token and not ignore_mem_token:
                inputs_embeds = torch.concat((inputs_embeds, self.mem_tokens.unsqueeze(0).repeat(inputs_embeds.shape[0], 1, 1)), dim=-2)
                if attention_mask is not None:
                    attention_mask = torch.concat([attention_mask, torch.ones_like(attention_mask[:, [0]]).repeat(1, self.num_mem_token)], dim=-1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if self.use_mem_token and not ignore_mem_token:
            memory_states = torch.zeros((hidden_states.shape[0], self.config.num_hidden_layers, self.num_mem_token, self.config.hidden_size)).to(self.device)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if use_gradient_checkpoint:
                hidden_states = checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    adapter_state=adapter_state[i] if isinstance(adapter_state, dict) else None,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    adapter_state=adapter_state[i] if isinstance(adapter_state, dict) else None,
                    **kwargs,
                )
            if self.use_mem_token and not ignore_mem_token:
                memory_states[:, i, :, :] = hidden_states[:, -self.num_mem_token:].to(self.device)

        if self.use_mem_token and not ignore_mem_token:
            hidden_states = hidden_states[:, :-self.num_mem_token, :]
        hidden_states = self.norm(hidden_states)
        return LatentSkillModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            memory_states=memory_states if (self.use_mem_token and not ignore_mem_token) else None,
        )

    def adapter_params_numel(self, r):
        if not hasattr(self, "adapter_params_numel_cache"):
            self.adapter_params_numel_cache = 0
            for layer in self.layers:
                self.adapter_params_numel_cache += layer.adapter_params_numel(r)
        return self.adapter_params_numel_cache

    def configure_adapter_builder(self, method):
        for layer in self.layers:
            layer.configure_adapter_builder(method)

    def build_adapter_state(self, r, scale, plain_tensor):
        assert plain_tensor.shape[-1] == self.adapter_params_numel(r), f"plain_tensor's last dimension {plain_tensor.shape[-1]} does not match adapter_params_numel {self.adapter_params_numel(r)}"
        idx = 0
        adapter_state = {}
        for i, layer in enumerate(self.layers):
            layer_adapter_params_numel = layer.adapter_params_numel(r)
            adapter_state[i] = layer.build_adapter_state(r, scale, plain_tensor[:, idx: idx + layer_adapter_params_numel])
            idx += layer_adapter_params_numel
        return adapter_state

    def init_adapter_state(self, r, scale, device):
        adapter_state = {}
        for i, layer in enumerate(self.layers):
            adapter_state[i] = layer.init_adapter_state(r, scale, device)
        return adapter_state

    def partition_adapter_params(self, r, idx_start):
        return self.layers[0].partition_adapter_params(r, idx_start)

@dataclass
class LatentSkillCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal-LM output that also carries memory-token states and regularization loss."""

    memory_states: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.FloatTensor] = None


@auto_docstring
class LatentSkillQwen3ForCausalLM(LatentSkillPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LatentSkillQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def reset_mem_tokens(self):
        self.model.reset_mem_tokens()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        adapter_state: Optional[dict] = None,
        ignore_mem_token: bool = False,
        use_gradient_checkpoint: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        adapter_state (`dict` of `dict` of `torch.FloatTensor`, *optional*):
            A dictionary that maps each layer to its corresponding LoRA parameters. Each layer's LoRA parameters are
            stored in a nested dictionary.
        ignore_mem_token (`bool`, *optional*, defaults to `False`):
            Whether to ignore the memory tokens during the forward pass. If set to `True`, the memory tokens will not be
            used, and the model will behave like a standard transformer without memory tokens.
        use_gradient_checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing to save memory during training. If set to `True`, the model will
            recompute certain activations during the backward pass instead of storing them in memory.
        """
        outputs: LatentSkillModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            adapter_state=adapter_state,
            ignore_mem_token=ignore_mem_token,
            use_gradient_checkpoint=use_gradient_checkpoint,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return LatentSkillCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            memory_states=outputs.memory_states if not ignore_mem_token else None,
        )

    def adapter_params_numel(self, r):
        return self.model.adapter_params_numel(r)

    def configure_adapter_builder(self, method):
        self.model.configure_adapter_builder(method)

    def build_adapter_state(self, r, scale, plain_tensor):
        return self.model.build_adapter_state(r, scale, plain_tensor)

    def init_adapter_state(self, r, scale, device):
        return self.model.init_adapter_state(r, scale, device)

    def partition_adapter_params(self, r, idx_start):
        return self.model.partition_adapter_params(r, idx_start)
