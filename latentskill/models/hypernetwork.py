# LatentSkill hypernetwork module
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_parameter_block_mask(idx_range, couple_hidden_size, couple_num_tokens):
    """Generate the attention mask for coupled LoRA parameter blocks."""
    assert len(idx_range) >= 3 and len(idx_range) % 2 == 1, (
        "idx_range must contain paired block boundaries"
    )
    assert idx_range[-1] == couple_num_tokens * couple_hidden_size, (
        "idx_range does not match couple_num_tokens and couple_hidden_size"
    )

    mask = torch.ones((couple_num_tokens, couple_num_tokens), dtype=torch.bool)
    token_idx_range = []
    for idx in idx_range:
        assert idx % couple_hidden_size == 0, (
            "idx_range must be divisible by couple_hidden_size"
        )
        token_idx_range.append(idx // couple_hidden_size)

    for i in range(0, len(idx_range) - 1, 2):
        left_start = token_idx_range[i]
        left_end = token_idx_range[i + 1]
        right_end = token_idx_range[i + 2]
        mask[left_start:left_end, left_end:right_end] = False
        mask[left_end:right_end, left_start:left_end] = False
    return mask


class SkillHypernetworkTransformer(nn.Module):
    def __init__(self, cfg, idx_range):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size
        self.mean_pool_size = cfg.hypernetwork.transformer_cfg.mean_pool_size
        self.idx_range = idx_range
        self.layer_transformer_first = bool(
            cfg.hypernetwork.transformer_cfg.layer_transformer_first
        )

        self.layer_pe = nn.Parameter(
            torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True
        )
        self.token_pe = nn.Parameter(
            torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True
        )

        transformer_cfg = cfg.hypernetwork.transformer_cfg
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(**transformer_cfg.encoder_cfg)
                for _ in range(transformer_cfg.num_layers)
            ]
        )
        self.couple_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(**transformer_cfg.couple_encoder_cfg)
                for _ in range(transformer_cfg.couple_num_layers)
            ]
        )
        self.couple_hidden_size = transformer_cfg.couple_encoder_cfg.d_model
        self.couple_num_layers = transformer_cfg.couple_num_layers

        assert self.hidden_size % self.couple_hidden_size == 0, (
            "hidden_size must be divisible by couple_hidden_size"
        )
        self.couple_num_tokens = (
            self.num_mem_token * self.hidden_size // self.couple_hidden_size
        )
        couple_mask = build_parameter_block_mask(
            idx_range, self.couple_hidden_size, self.couple_num_tokens
        )
        self.register_buffer("couple_mask", couple_mask, persistent=False)

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_states: tensor with shape
                (batch_size, num_layers, num_mem_token, hidden_size).
        """
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe
        batch_size = memory_states.shape[0]

        # Alternate information exchange across layers and memory tokens.
        for i, layer in enumerate(self.transformer_layers):
            if (i % 2 == 0) == self.layer_transformer_first:
                memory_states = (
                    layer(memory_states.transpose(1, 2).flatten(0, 1))
                    .unflatten(0, (batch_size, self.num_mem_token))
                    .transpose(1, 2)
                )
            else:
                memory_states = layer(memory_states.flatten(0, 1)).unflatten(
                    0, (batch_size, self.num_layers)
                )

        memory_states = torch.mean(
            memory_states.unflatten(
                2, (self.mean_pool_size, self.num_mem_token // self.mean_pool_size)
            ),
            dim=2,
        )

        memory_states = memory_states.view(
            batch_size * self.num_layers, -1, self.couple_hidden_size
        )
        for layer in self.couple_layers:
            memory_states = layer(memory_states, src_mask=self.couple_mask)
        return memory_states.view(batch_size, -1)


class SkillHypernetworkLinear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size

        self.layer_pe = nn.Parameter(
            torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True
        )
        self.token_pe = nn.Parameter(
            torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True
        )

        linear_cfg = cfg.hypernetwork.linear_cfg
        self.dim_list = (
            [self.hidden_size]
            + [linear_cfg.linear_hidden_dim] * (linear_cfg.num_layers - 1)
            + [self.hidden_size]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(self.dim_list[i], self.dim_list[i + 1], bias=linear_cfg.bias)
                for i in range(linear_cfg.num_layers)
            ]
        )

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_states: tensor with shape
                (batch_size, num_layers, num_mem_token, hidden_size).
        """
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe
        batch_size = memory_states.shape[0]
        memory_states = memory_states.flatten(0, 1)
        for layer in self.linear_layers:
            memory_states = F.gelu(layer(memory_states))
        return memory_states.unflatten(0, (batch_size, self.num_layers)).flatten(1, -1)


class SkillHypernetworkLinearGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.num_mem_token = cfg.num_mem_token
        self.hidden_size = cfg.hidden_size

        self.layer_pe = nn.Parameter(
            torch.zeros((self.num_layers, self.hidden_size)), requires_grad=True
        )
        self.token_pe = nn.Parameter(
            torch.zeros((self.num_mem_token, self.hidden_size)), requires_grad=True
        )

        linear_cfg = cfg.hypernetwork.linear_gate_cfg
        self.dim_list = (
            [self.hidden_size]
            + [linear_cfg.linear_hidden_dim] * (linear_cfg.num_layers - 1)
            + [self.hidden_size * 2]
        )
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(self.dim_list[i], self.dim_list[i + 1], bias=linear_cfg.bias)
                for i in range(linear_cfg.num_layers)
            ]
        )

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_states: tensor with shape
                (batch_size, num_layers, num_mem_token, hidden_size).
        """
        memory_states = memory_states + self.layer_pe.unsqueeze(-2) + self.token_pe
        batch_size = memory_states.shape[0]
        memory_states = memory_states.flatten(0, 1)
        for layer in self.linear_layers:
            memory_states = F.gelu(layer(memory_states))
        gate, value = memory_states.chunk(2, dim=-1)
        memory_states = torch.sigmoid(gate) * value
        return memory_states.unflatten(0, (batch_size, self.num_layers)).flatten(1, -1)


class SkillHypernetwork(nn.Module):
    def __init__(self, backbone: nn.Module, cfg, output_dim: int):
        super().__init__()
        self.lora_r = cfg.model.lora_r
        self.output_dim = output_dim
        self.backbone = backbone
        self.idx_range, end = self.backbone.partition_adapter_params(self.lora_r, 0)
        self.idx_range.append(end)
        self.adapter_reg = cfg.optim.adapter_reg if hasattr(cfg, "optim") else 0.0
        self.method = cfg.hypernetwork.method
        self.backbone.configure_adapter_builder(self.method)

        if cfg.hypernetwork.type == "transformer":
            self.generator = SkillHypernetworkTransformer(cfg, self.idx_range)
            self.scale = cfg.hypernetwork.transformer_cfg.scale
        elif cfg.hypernetwork.type == "linear":
            self.generator = SkillHypernetworkLinear(cfg)
            self.scale = cfg.hypernetwork.linear_cfg.scale
        elif cfg.hypernetwork.type == "lineargate":
            self.generator = SkillHypernetworkLinearGate(cfg)
            self.scale = cfg.hypernetwork.linear_gate_cfg.scale
        else:
            raise ValueError(f"Unknown hypernetwork type: {cfg.hypernetwork.type}")

    @property
    def config(self):
        return getattr(self.backbone, "config", None)

    def forward(
        self,
        input_ids,
        input_attention_mask,
        evidence_ids,
        evidence_attention_mask,
        metalora=None,
        labels=None,
        use_generator=True,
        use_gradient_checkpoint=False,
        **kwargs,
    ):
        if use_generator:
            assert metalora is not None, (
                "metalora cannot be None when use_generator is True"
            )
            adapter_state, plain_output = self.build_adapter_state(
                evidence_ids,
                evidence_attention_mask,
                metalora,
                use_gradient_checkpoint=use_gradient_checkpoint,
                return_plain=True,
            )
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                adapter_state=adapter_state,
                labels=labels,
                ignore_mem_token=True,
                use_gradient_checkpoint=use_gradient_checkpoint,
                **kwargs,
            )
            outputs.reg_loss = self.adapter_reg * torch.abs(plain_output).sum()
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                labels=labels,
                ignore_mem_token=True,
                use_gradient_checkpoint=use_gradient_checkpoint,
                **kwargs,
            )
        return outputs

    def build_adapter_state(
        self,
        evidence_ids,
        evidence_attention_mask,
        metalora,
        use_gradient_checkpoint=False,
        return_plain=False,
    ):
        outputs = self.backbone(
            input_ids=evidence_ids,
            attention_mask=evidence_attention_mask,
            adapter_state=metalora,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )
        memory_states = outputs.memory_states

        plain_output = self.generator(memory_states)
        assert plain_output.shape[-1] == self.output_dim, (
            f"SkillHypernetwork output dimension mismatch: got {plain_output.shape[-1]}, "
            f"expected {self.output_dim}"
        )

        adapter_state = self.backbone.build_adapter_state(
            self.lora_r, scale=self.scale, plain_tensor=plain_output
        )
        return adapter_state if not return_plain else (adapter_state, plain_output)
