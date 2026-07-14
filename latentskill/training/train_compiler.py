import os
import math
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
import hydra
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from latentskill.models.hypernetwork import SkillHypernetwork

from transformers import AutoTokenizer

from latentskill.data.datasets import (
    DynamicSkillPretrainDataset,
    SkillPretrainCollator,
    StaticSkillGroupDataset,
    SkillInstructionCollator,
    SkillInstructionDataset,
)
from latentskill.utils.seed import seed_everything
from latentskill.utils.logging import get_logger
from latentskill.training.checkpointing import (
    write_latentskill_checkpoint,
    restore_latentskill_checkpoint,
    write_trainer_state,
    restore_trainer_state,
    find_latest_checkpoint,
)
from latentskill.utils.freeze import freeze_backbone_except_memory
from latentskill.utils.optimize import build_optimizer_and_scheduler
from latentskill.utils.ddp import (
    distributed_requested,
    is_distributed,
    world_size,
    process_rank,
    is_primary_process,
    initialize_distributed,
    cleanup_distributed,
    mean_across_processes,
)
from latentskill.models.lora_ops import (
    freeze_adapter_state,
    iter_trainable_tensors,
    adapter_state_requires_grad,
    merge_adapter_states,
)
from latentskill.utils.init import _resolve_device, _import_class

logger = get_logger("latentskill.train")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def evaluate(
    hypernet_ddp_or_module,
    dataloader,
    device,
    use_amp: bool = False,
    use_generator: bool = True,
    metalora: Optional[torch.Tensor] = None,
    amp_dtype=None,
) -> Dict[str, float]:
    skill_net = hypernet_ddp_or_module.module if isinstance(hypernet_ddp_or_module, DDP) else hypernet_ddp_or_module
    skill_net.eval()

    if use_generator:
        assert metalora is not None, "metalora is required when use_generator=True"

    total_loss = 0.0
    total_reg_loss = 0.0
    n_tokens = 0
    amp_ctx = torch.amp.autocast(
        device_type="cuda",
        dtype=(amp_dtype or torch.bfloat16),
        enabled=(use_amp and device.type == "cuda"),
    )

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)

        with amp_ctx:
            outputs = skill_net(
                input_ids=input_ids,
                input_attention_mask=input_attention_mask,
                evidence_ids=evidence_ids,
                evidence_attention_mask=evidence_attention_mask,
                labels=labels,
                use_generator=use_generator,
                metalora=metalora,
            )
        loss = outputs.loss
        reg_loss = outputs.reg_loss

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        if use_generator:
            total_reg_loss += reg_loss.item() * valid_tokens
        n_tokens += valid_tokens

    # Reduce across ranks
    if is_distributed():
        t = torch.tensor([total_loss, n_tokens, total_reg_loss], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_loss = float(t[0].item())
        n_tokens = int(t[1].item())
        if use_generator:
            total_reg_loss = float(t[2].item())

    avg_loss = total_loss / max(n_tokens, 1)
    avg_reg_loss = total_reg_loss / max(n_tokens, 1) if use_generator else None
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    skill_net.train()
    return {"eval_loss": avg_loss, "perplexity": ppl, "eval_reg_loss": avg_reg_loss}


@hydra.main(version_base=None, config_path="../../configs")
def main(cfg: DictConfig):
    amp_dtype = torch.bfloat16

    torch.set_float32_matmul_precision("high")
    if cfg.run.use_gradient_checkpoint:
        torch._dynamo.config.optimize_ddp = False
    if cfg.mode == "train":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ========= DDP init (safe for single-process) =========
    initialize_distributed()

    if is_primary_process():
        logger.info("Resolved config:")
        logger.info(f"\n\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Seed & device
    # Make seed rank-dependent to vary shuffles but keep reproducibility per rank
    seed_everything(int(cfg.run.seed) + process_rank())
    device = _resolve_device(cfg.run.device)
    torch.backends.cudnn.benchmark = True

    # Load model/tokenizer (supports your local LoRA-wrapped Qwen class)
    if is_primary_process():
        logger.info("Loading model & tokenizer...")
    BackboneModelCls = _import_class(cfg.model.backbone_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    # Infer how many memory tokens are needed for the selected hypernetwork.
    if cfg.hypernetwork.type in ["transformer", "linear", "lineargate"]:
        tmp_model = BackboneModelCls.from_pretrained(cfg.model.model_from, config=config)
        adapter_numel = tmp_model.adapter_params_numel(cfg.model.lora_r)
        assert adapter_numel % (cfg.hidden_size * cfg.num_layers) == 0, \
            "For the transformer hypernetwork, num_mem_token must match the generated LoRA parameter size."
        config.num_mem_token = tmp_model.adapter_params_numel(cfg.model.lora_r) * cfg.hypernetwork.transformer_cfg.mean_pool_size // (cfg.hidden_size * cfg.num_layers)
        cfg.num_mem_token = config.num_mem_token
        del tmp_model
        if is_primary_process():
            logger.info(f"Using {cfg.hypernetwork.type} hypernetwork; num_mem_token={config.num_mem_token}")
    else:
        raise ValueError(f"Unknown hypernetwork type: {cfg.hypernetwork.type}")
    checkpoint_root = str(cfg.paths.get("checkpoint_root", "checkpoints"))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_from, padding_side="left", use_fast=True)
    # Add the special tokens used by the pretraining objectives.
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is not defined or enable_thinking != false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
    # Load the backbone model.
    backbone = BackboneModelCls.from_pretrained(cfg.model.model_from, config=config)
    backbone.reset_mem_tokens()
    backbone.resize_token_embeddings(len(tokenizer))

    # LatentSkill hypernetwork.
    skill_hypernet = SkillHypernetwork(backbone, cfg, backbone.adapter_params_numel(cfg.model.lora_r))
    skill_hypernet.train()
    skill_hypernet.to(device)
    # Freeze the backbone and train only the adapter generator, MetaLoRA, and memory tokens.
    freeze_backbone_except_memory(backbone)
    if is_primary_process():
        logger.info(f"SkillHypernetwork type: {cfg.hypernetwork.type}, Transform method: {cfg.hypernetwork.method}")

    # Training loop scaffolding
    ckpt_root = os.path.join(checkpoint_root, f"{cfg.name}", f"{cfg.mode}")
    if is_primary_process():
        os.makedirs(ckpt_root, exist_ok=True)
    if cfg.resume_global_step == -1:
        resume_dir = None
    elif cfg.resume_global_step == "latest":
        resume_dir = find_latest_checkpoint(ckpt_root)
    elif isinstance(cfg.resume_global_step, int) and cfg.resume_global_step > 0:
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    elif isinstance(cfg.resume_global_step, str) and cfg.resume_global_step.startswith("epoch-"):
        resume_dir = os.path.join(ckpt_root, f"checkpoint-{cfg.resume_global_step}")
        if not os.path.isdir(resume_dir):
            raise ValueError(f"Requested resume dir {resume_dir} does not exist.")
    else:
        raise ValueError(f"Invalid resume_global_step: {cfg.resume_global_step}")

    resume_state = None
    USE_ADDITIONAL_METALORA = bool(cfg.model.ift_additional_metalora_r >= 0 and cfg.mode == "train")
    if is_primary_process():
        logger.info(f"USE_ADDITIONAL_METALORA: {USE_ADDITIONAL_METALORA}, r={cfg.model.ift_additional_metalora_r}")

    def _pretrain_checkpoint_parent(default_name: str) -> str:
        configured_dir = str(cfg.paths.get("pretrain_checkpoint_dir", "")).strip()
        if configured_dir:
            return configured_dir
        configured_name = str(cfg.paths.get("pretrain_checkpoint_name", "")).strip() or default_name
        return os.path.join(checkpoint_root, configured_name, "pretrain")

    if resume_dir is not None:
        # Load model & tokenizer
        if is_primary_process():
            logger.info(f"Resume mode, loading from {resume_dir}...")
        skill_hypernet, metalora, ift_additional_metalora = restore_latentskill_checkpoint(skill_hypernet, resume_dir, device, load_ift_additional_metalora=USE_ADDITIONAL_METALORA, zero_ift_additional_metalora=(cfg.model.ift_additional_metalora_r == 0))
        resume_state = restore_trainer_state(resume_dir)
    else:
        if cfg.mode == "train":
            try:
                pretrain_dir = _pretrain_checkpoint_parent(str(cfg.name))
                pretrain_dir = find_latest_checkpoint(pretrain_dir, only_epoch=True)

                skill_hypernet, metalora, ift_additional_metalora = restore_latentskill_checkpoint(
                    skill_hypernet,
                    pretrain_dir,
                    device,
                    load_ift_additional_metalora=False,
                )

                if USE_ADDITIONAL_METALORA:
                    freeze_adapter_state(metalora)
                    ift_additional_metalora = (
                        skill_hypernet.backbone.init_adapter_state(
                            cfg.model.ift_additional_metalora_r,
                            scale=cfg.hypernetwork.transformer_cfg.scale,
                            device=device,
                        )
                        if cfg.model.ift_additional_metalora_r > 0
                        else None
                    )

                if is_primary_process():
                    logger.info(f"Loaded hypernetwork from pretrain checkpoint: {pretrain_dir}")
                    if USE_ADDITIONAL_METALORA:
                        logger.info(
                            f"Initialized additional IFT metalora with r={cfg.model.ift_additional_metalora_r} from scratch. "
                            f"Freezing pretrain metalora."
                        )
                    else:
                        logger.info("No additional IFT metalora used.")

            except Exception as e:
                if is_primary_process():
                    logger.warning(
                        f"No pretrain checkpoint found in {pretrain_dir}, initializing hypernetwork from scratch."
                    )
                    logger.warning(f"Exception: {e}")
                assert not USE_ADDITIONAL_METALORA, \
                    "IFT additional metalora mustn't be used when no pretrain."
                metalora = skill_hypernet.backbone.init_adapter_state(
                    cfg.model.metalora_r,
                    scale=cfg.hypernetwork.transformer_cfg.scale,
                    device=device,
                )
        elif cfg.mode == "pretrain":
            # Initialize MetaLoRA with the standard LoRA convention: random A and zero B.
            metalora = skill_hypernet.backbone.init_adapter_state(cfg.model.metalora_r, scale=cfg.hypernetwork.transformer_cfg.scale, device=device)
        else:
            raise ValueError(f"Unknown training mode: {cfg.mode}")

    skill_hypernet.backbone.config.use_cache = False

    # ====== Wrap ONLY the trainable module in DDP when applicable ======
    skill_hypernet.to(device)
    if distributed_requested():
        ddp_hypernet = DDP(
            skill_hypernet,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    else:
        ddp_hypernet = skill_hypernet  # no wrapping in single-process run

    # Optimizer & Scheduler
    if is_primary_process():
        logger.info("Setting up optimizer & scheduler...")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "norm1", "norm2"]
    grouped_params = [
        # Group 1: adapter generator and memory-token parameters with weight decay.
        {
            "params": [p for n, p in ddp_hypernet.named_parameters() if (not any(nd in n for nd in no_decay) and not n.startswith("module.backbone"))],
            "weight_decay": cfg.optim.weight_decay,
        },
        # Group 2: adapter generator bias and normalization parameters without weight decay.
        {
            "params": [p for n, p in ddp_hypernet.named_parameters() if (any(nd in n for nd in no_decay) and not n.startswith("module.backbone"))],
            "weight_decay": 0.0,
        },
        # Group 3: MetaLoRA parameters injected into the backbone.
        {
            "params": list(iter_trainable_tensors(metalora) if not USE_ADDITIONAL_METALORA else iter_trainable_tensors(ift_additional_metalora)),
            "weight_decay": cfg.optim.weight_decay,
        }
        # mem_tokens are already part of the hypernetwork module.
    ]

    def validate_trainable_param_groups(grouped_params):
        """
        Assert all params in optimizer param groups have requires_grad=True.
        """
        frozen = []

        for gi, group in enumerate(grouped_params):
            for pi, p in enumerate(group["params"]):
                if not p.requires_grad:
                    frozen.append((gi, pi, tuple(p.shape)))

        if frozen:
            msg = ["Found params with requires_grad=False in grouped_params:"]
            msg += [f"  - group {gi}, param {pi}, shape={shape}"
                    for gi, pi, shape in frozen]
            raise RuntimeError("\n".join(msg))
    if is_primary_process():
        validate_trainable_param_groups(grouped_params)

    # Data
    if is_primary_process():
        logger.info("Preparing data...")
    data_paths = cfg.data.paths
    skill_pretrain_dir = str(data_paths.skill_pretrain_dir)
    skill_ift_train_path = str(data_paths.skill_ift_train)

    if cfg.data.source == "skill-pretrain":
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(data_paths.skill_pretrain_train),
                "validation": str(data_paths.skill_pretrain_val),
            }
        )
        train_texts = dataset["train"]
        val_texts = dataset["validation"]

        train_ds = StaticSkillGroupDataset(
            train_texts["text"], tokenizer, cfg.data.conversation_max_length,
            skill_pretrain_dir, "train"
        )
        val_ds = StaticSkillGroupDataset(
            val_texts["text"], tokenizer, cfg.data.conversation_max_length,
            skill_pretrain_dir, "val"
        )

        # Use the grouped pretraining collator.
        train_collator = SkillPretrainCollator(
            tokenizer=tokenizer,
            cfg=cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
        )
        val_collator = SkillPretrainCollator(
            tokenizer=tokenizer,
            cfg=cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
        )
    elif cfg.data.source == "skill-pretrain-dynamic":
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(data_paths.skill_pretrain_train),
                "validation": str(data_paths.skill_pretrain_val),
            }
        )
        train_texts = dataset["train"]
        val_texts = dataset["validation"]

        train_ds = DynamicSkillPretrainDataset(
            train_texts["text"], tokenizer, cfg, split="train"
        )
        val_ds = DynamicSkillPretrainDataset(
            val_texts["text"], tokenizer, cfg, split="val", force_single_skill=True
        )

        train_collator = SkillPretrainCollator(
            tokenizer=tokenizer,
            cfg=cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
        )
        val_collator = SkillPretrainCollator(
            tokenizer=tokenizer,
            cfg=cfg,
            conversation_max_length=cfg.data.conversation_max_length,
            context_max_length=cfg.data.context_max_length,
        )
    elif cfg.data.source == "skill-ift":
        train_path = skill_ift_train_path
        if is_primary_process():
            logger.info(f"[skill-ift] Loading IFT data from: {train_path}")

        full_ds = SkillInstructionDataset(
            train_path,
            use_exceed=False,
            max_context_len=cfg.data.context_max_length,
            max_conversation_len=cfg.data.conversation_max_length,
        )
        if is_primary_process():
            logger.info(f"[skill-ift] Full dataset size: {len(full_ds)}")

        # Hold out the last few examples for loss monitoring.
        VAL_SIZE = int(data_paths.get("skill_ift_val_size", 200))
        val_indices = list(range(len(full_ds) - VAL_SIZE, len(full_ds)))
        train_indices = list(range(len(full_ds) - VAL_SIZE))

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        val_ds = torch.utils.data.Subset(full_ds, val_indices)

        train_collator = SkillInstructionCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.data.context_max_length,
            conversation_max_length=cfg.data.conversation_max_length,
            cfg=cfg,
        )
        val_collator = SkillInstructionCollator(
            tokenizer=tokenizer,
            context_max_length=cfg.data.context_max_length,
            conversation_max_length=cfg.data.conversation_max_length,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown data source: {cfg.data.source}")



    pin = (device.type == "cuda")

    # Distributed samplers (only if world_size > 1)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size(), rank=process_rank(), shuffle=True, seed=cfg.run.seed) if world_size() > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size(), rank=process_rank(), shuffle=False) if world_size() > 1 else None

    # Use a few workers by default when on GPU
    num_workers_default = 2 if device.type == "cuda" else 0
    num_workers_cfg = getattr(cfg.data, "num_workers", num_workers_default)

    if cfg.data.source == "skill-pretrain-dynamic":
        num_workers_cfg = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=train_collator,
        pin_memory=pin,
        num_workers=num_workers_cfg,
        persistent_workers=pin and num_workers_cfg > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=val_collator,
        pin_memory=pin,
        num_workers=num_workers_cfg,
        persistent_workers=pin and num_workers_cfg > 0,
    )


    optimizer, lr_scheduler = build_optimizer_and_scheduler(grouped_params, train_loader, cfg, device)

    # Only main process writes TB logs
    tb_log_dir = os.path.join("tensorboard", f"{cfg.name}", f"{cfg.mode}")
    writer = SummaryWriter(log_dir=tb_log_dir) if is_primary_process() else None
    if is_primary_process():
        logger.info(f"TensorBoard logs will be written to: {tb_log_dir}")
        logger.info("Starting training loop...")

    # Validate trainable LoRA state before distributed training starts.
    if is_distributed():
        if is_primary_process():
            if USE_ADDITIONAL_METALORA:
                assert adapter_state_requires_grad(metalora, False), "When using additional IFT metalora, the pretrain metalora must be frozen."
                assert adapter_state_requires_grad(ift_additional_metalora, True), "IFT additional metalora must be learnable."
            else:
                assert adapter_state_requires_grad(metalora, True), "Metalora must be learnable."
        dist.barrier()

    global_step = 0
    best_eval_loss = float("inf")
    start_epoch = 0
    start_step_in_epoch = 0
    if resume_state is not None:
        global_step = resume_state["global_step"]
        best_eval_loss = resume_state["best_eval_loss"]
        start_epoch = resume_state["epoch"]
        start_step_in_epoch = resume_state["step_in_epoch"]

    def run_training_epoch(epoch, start_epoch=1, start_step_in_epoch=0):
        nonlocal global_step, best_eval_loss
        epoch_loss = 0.0
        epoch_tokens = 0
        tmp_loss = 0.0
        tmp_tokens = 0
        # Refresh dynamic skill grouping at the start of each epoch.
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if epoch < start_epoch:
            for step, _ in enumerate(train_loader, start=1):
                if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                    lr_scheduler.step()
            return

        pbar = train_loader
        if is_primary_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.optim.num_epochs}")

        for step, batch in enumerate(pbar, start=1):
            if epoch == start_epoch and step <= start_step_in_epoch:
                if step % max(1, cfg.run.gradient_accumulation_steps) == 0:
                    lr_scheduler.step()
                continue
            # Move the batch to the target device.
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            input_attention_mask = batch["input_attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
            evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)

            # Select the active MetaLoRA weights for this step.
            if not USE_ADDITIONAL_METALORA:
                cur_metalora = metalora
            else:
                cur_metalora = merge_adapter_states(metalora, ift_additional_metalora, method=cfg.hypernetwork.method)

            with torch.amp.autocast(enabled=(cfg.run.use_amp and device.type == "cuda"), device_type="cuda", dtype=amp_dtype):
                # Forward through the possibly DDP-wrapped hypernetwork.
                outputs = ddp_hypernet(
                    input_ids=input_ids,
                    input_attention_mask=input_attention_mask,
                    evidence_ids=evidence_ids,
                    evidence_attention_mask=evidence_attention_mask,
                    labels=labels,
                    metalora=cur_metalora,
                    use_gradient_checkpoint=cfg.run.use_gradient_checkpoint,
                )
                loss = (outputs.loss / max(1, cfg.run.gradient_accumulation_steps)).item()
                reg_loss = (outputs.reg_loss / max(1, cfg.run.gradient_accumulation_steps)).item()
                backward_loss = (outputs.loss + outputs.reg_loss) / max(1, cfg.run.gradient_accumulation_steps)

            if writer is not None:
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

            valid_tokens = (labels != -100).sum().item()
            if not math.isinf(loss) and not math.isnan(loss) and valid_tokens > 0:
                backward_loss.backward()
                epoch_loss += loss * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
                tmp_loss += loss * valid_tokens * max(1, cfg.run.gradient_accumulation_steps)
                epoch_tokens += valid_tokens
                tmp_tokens += valid_tokens
            else:
                res = f"NaN/Inf loss detected at epoch {epoch} step {step}!\nBatch:\n{batch}\nloss: {loss}\nvalid tokens: {valid_tokens}\n\n"
                logger.info(res)

            if step % max(1, cfg.run.gradient_accumulation_steps) == 0 or step == len(train_loader):
                if cfg.optim.grad_clip_norm and cfg.optim.grad_clip_norm > 0:
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group["params"], cfg.optim.grad_clip_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                global_step += 1

                # Periodic logging (only on rank 0, with distributed averages)
                if cfg.logging.logging_steps and global_step % cfg.logging.logging_steps == 0:
                    # everyone computes + participates in the reduction
                    avg_loss_local = (epoch_loss / max(epoch_tokens, 1))
                    tmp_loss_local = (tmp_loss / max(tmp_tokens, 1))
                    avg_loss_world = mean_across_processes(avg_loss_local, device)
                    tmp_loss_world = mean_across_processes(tmp_loss_local, device)
                    tmp_loss_reg_local = (reg_loss / max(tmp_tokens, 1))
                    tmp_loss_reg_world = mean_across_processes(tmp_loss_reg_local, device)
                    if is_primary_process():
                        avg_ppl = math.exp(avg_loss_world) if avg_loss_world < 20 else float("inf")
                        tmp_ppl = math.exp(tmp_loss_world) if tmp_loss_world < 20 else float("inf")
                        if writer is not None:
                            writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                            writer.add_scalar("train/epoch_avg_loss", avg_loss_world, global_step)
                            writer.add_scalar("train/epoch_avg_ppl", avg_ppl, global_step)
                            writer.add_scalar("train/tmp_loss", tmp_loss_world, global_step)
                            writer.add_scalar("train/tmp_ppl", tmp_ppl, global_step)
                            writer.add_scalar("train/tmp_reg_loss", tmp_loss_reg_world, global_step)
                        if isinstance(pbar, tqdm):
                            pbar.set_postfix(
                                {
                                    "lr": lr_scheduler.get_last_lr()[0],
                                    "epoch_avg_loss": f"{avg_loss_world:.4f}",
                                    "epoch_avg_ppl": f"{avg_ppl:.2f}",
                                    "tmp_loss": f"{tmp_loss_world:.4f}",
                                    "tmp_ppl": f"{tmp_ppl:.2f}",
                                    "tmp_reg_loss": f"{tmp_loss_reg_world:.8f}",
                                }
                            )
                    tmp_loss = 0.0
                    tmp_tokens = 0

                # ---- Periodic checkpoint (rank 0 only) ----
                if getattr(cfg.save, "save_steps", 0) and global_step % cfg.save.save_steps == 0:
                    if is_distributed():
                        dist.barrier()
                    if is_primary_process():
                        ckpt_dir = os.path.join(ckpt_root, f"checkpoint-{global_step}")
                        logger.info(f"Saving checkpoint to {ckpt_dir}")
                        # Save the unwrapped hypernetwork.
                        write_latentskill_checkpoint(
                            ddp_hypernet.module if isinstance(ddp_hypernet, DDP) else ddp_hypernet,
                            ckpt_dir,
                            extra_state={"global_step": global_step},
                            metalora=metalora,
                            ift_additional_metalora=ift_additional_metalora if USE_ADDITIONAL_METALORA else None,
                        )
                        write_trainer_state(
                            ckpt_dir,
                            global_step,
                            epoch,
                            step,
                            best_eval_loss,
                        )
                    if is_distributed():
                        dist.barrier()

                # ---- Periodic eval ----
                if getattr(cfg.eval, "eval_steps", 0) and global_step % cfg.eval.eval_steps == 0:
                    if not USE_ADDITIONAL_METALORA:
                        cur_metalora = metalora
                    else:
                        cur_metalora = merge_adapter_states(metalora, ift_additional_metalora, method=cfg.hypernetwork.method)
                    eval_metrics = evaluate(ddp_hypernet, val_loader, device, use_amp=cfg.run.use_amp, metalora=cur_metalora, amp_dtype=amp_dtype)
                    if writer is not None:
                        writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
                        writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
                    if is_primary_process():
                        logger.info(f"[Eval @ step {global_step}] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Epoch-end eval/log (averaged)
        avg_epoch_loss_local = (epoch_loss / max(epoch_tokens, 1))
        avg_epoch_loss_world = mean_across_processes(avg_epoch_loss_local, device)
        epoch_ppl = math.exp(avg_epoch_loss_world) if avg_epoch_loss_world < 20 else float("inf")
        if is_primary_process():
            logger.info(f"Epoch {epoch} done. train_loss={avg_epoch_loss_world:.4f} train_ppl={epoch_ppl:.2f}")

        if not USE_ADDITIONAL_METALORA:
            cur_metalora = metalora
        else:
            cur_metalora = merge_adapter_states(metalora, ift_additional_metalora, method=cfg.hypernetwork.method)
        eval_metrics = evaluate(ddp_hypernet, val_loader, device, use_amp=cfg.run.use_amp, metalora=cur_metalora, amp_dtype=amp_dtype)
        if writer is not None:
            writer.add_scalar("eval/loss", eval_metrics["eval_loss"], global_step)
            writer.add_scalar("eval/ppl", eval_metrics["perplexity"], global_step)
        if is_primary_process():
            logger.info(f"[Epoch {epoch} Eval] loss={eval_metrics['eval_loss']:.4f} ppl={eval_metrics['perplexity']:.2f}")
        if is_primary_process():
            ckpt_dir = os.path.join(ckpt_root, f"checkpoint-epoch-{epoch}")
            logger.info(f"Saving checkpoint to {ckpt_dir}")
            # Save the unwrapped hypernetwork.
            write_latentskill_checkpoint(
                ddp_hypernet.module if isinstance(ddp_hypernet, DDP) else ddp_hypernet,
                ckpt_dir,
                extra_state={"global_step": global_step},
                metalora=metalora,
                ift_additional_metalora=ift_additional_metalora if USE_ADDITIONAL_METALORA else None,
            )
            write_trainer_state(
                ckpt_dir,
                global_step,
                epoch,
                step,
                best_eval_loss,
            )

    # Main training epochs
    for epoch in range(1, cfg.optim.num_epochs + 1):
        run_training_epoch(epoch, start_epoch, start_step_in_epoch)


    # Final save (rank 0 only)
    if is_primary_process():
        logger.info("Saving final model...")
        final_dir = os.path.join(ckpt_root, "final")
        write_latentskill_checkpoint(
            ddp_hypernet.module if isinstance(ddp_hypernet, DDP) else ddp_hypernet,
            final_dir,
            extra_state={"global_step": global_step},
            metalora=metalora,
            ift_additional_metalora=ift_additional_metalora if USE_ADDITIONAL_METALORA else None,
        )

        if cfg.paths.output_dir:
            stable_out = cfg.paths.output_dir
            os.makedirs(stable_out, exist_ok=True)
            write_latentskill_checkpoint(
                ddp_hypernet.module if isinstance(ddp_hypernet, DDP) else ddp_hypernet,
                stable_out,
                extra_state={"global_step": global_step},
                metalora=metalora,
                ift_additional_metalora=ift_additional_metalora if USE_ADDITIONAL_METALORA else None,
            )
            logger.info(f"Model saved to {stable_out}")

        logger.info("Complete.")

    if writer is not None:
        writer.close()

    # Cleanup DDP
    cleanup_distributed()


if __name__ == "__main__":
    main()
