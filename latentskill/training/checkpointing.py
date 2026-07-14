import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from latentskill.utils.logging import get_logger
from latentskill.utils.freeze import freeze_backbone_except_memory
from latentskill.models.lora_ops import freeze_adapter_state

logger = get_logger("checkpointing")


def _first_existing_checkpoint_file(in_dir: str, names):
    for name in names:
        path = os.path.join(in_dir, name)
        if os.path.isfile(path):
            return path
    return os.path.join(in_dir, names[0])


def write_latentskill_checkpoint(
    skill_hypernet,
    out_dir: str,
    metalora: Any,
    ift_additional_metalora: Any = None,
    extra_state: Dict[str, Any] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if skill_hypernet.backbone.model.use_mem_token:
        torch.save(skill_hypernet.backbone.model.mem_tokens, os.path.join(out_dir, "mem_tokens.pt"))
    torch.save(skill_hypernet.generator.state_dict(), os.path.join(out_dir, "hypernetwork.pth"))
    torch.save(metalora, os.path.join(out_dir, "metalora.pth"))
    if ift_additional_metalora is not None:
        torch.save(ift_additional_metalora, os.path.join(out_dir, "ift_additional_metalora.pth"))
    if extra_state is not None:
        with open(os.path.join(out_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(extra_state, f, ensure_ascii=False, indent=2)


def restore_latentskill_checkpoint(
    skill_hypernet,
    in_dir,
    device: str,
    load_ift_additional_metalora: bool = False,
    zero_ift_additional_metalora: bool = False,
):
    skill_hypernet.to("cpu")
    if skill_hypernet.backbone.model.use_mem_token:
        saved_mem_tokens = torch.load(os.path.join(in_dir, "mem_tokens.pt"), map_location="cpu", weights_only=False)
        assert saved_mem_tokens.shape == skill_hypernet.backbone.model.mem_tokens.shape, (
            f"Shape mismatch for mem_tokens: saved {saved_mem_tokens.shape}, "
            f"model {skill_hypernet.backbone.model.mem_tokens.shape}"
        )
        skill_hypernet.backbone.model.mem_tokens = saved_mem_tokens
    generator_path = _first_existing_checkpoint_file(
        in_dir, ("hypernetwork.pth", "skill_hypernet.pth")
    )
    skill_hypernet.generator.load_state_dict(
        torch.load(generator_path, weights_only=False, map_location="cpu")
    )
    metalora = torch.load(os.path.join(in_dir, "metalora.pth"), map_location="cpu", weights_only=False)
    skill_hypernet.to(device)
    metalora = materialize_trainable_state(metalora, device)
    freeze_backbone_except_memory(skill_hypernet.backbone)
    ift_additional_metalora_path = os.path.join(in_dir, "ift_additional_metalora.pth")
    if os.path.isfile(ift_additional_metalora_path):
        assert load_ift_additional_metalora and not zero_ift_additional_metalora, "Found ift_additional_metalora.pth but load_ift_additional_metalora is False"
        ift_additional_metalora = torch.load(ift_additional_metalora_path, map_location="cpu", weights_only=False)
        ift_additional_metalora = materialize_trainable_state(ift_additional_metalora, device)
        freeze_adapter_state(metalora)
    else:
        assert not load_ift_additional_metalora or zero_ift_additional_metalora, "ift_additional_metalora.pth not found but load_ift_additional_metalora is True"
        if zero_ift_additional_metalora:
            freeze_adapter_state(metalora)
    return skill_hypernet, metalora, ift_additional_metalora if (load_ift_additional_metalora and not zero_ift_additional_metalora) else None


def _rng_state_dict():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def _set_rng_state(state: Dict[str, Any]):
    if state is None:
        return
    try:
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        logger.warning(f"Could not fully restore RNG states: {e}")


def write_trainer_state(
    out_dir: str,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    best_eval_loss: float,
):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "best_eval_loss": best_eval_loss,
        "rng_state": _rng_state_dict(),
    }
    torch.save(payload, os.path.join(out_dir, "trainer_state.pt"))


def restore_trainer_state(
    in_dir: str,
):
    path = os.path.join(in_dir, "trainer_state.pt")
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _set_rng_state(payload.get("rng_state"))

    return {
        "global_step": payload.get("global_step", 0),
        "epoch": payload.get("epoch", 1),
        "step_in_epoch": payload.get("step_in_epoch", 0),
        "best_eval_loss": payload.get("best_eval_loss", float("inf")),
    }


def find_latest_checkpoint(root_dir: str, only_epoch=False) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None
    cands = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    if only_epoch:
        cands = [d for d in cands if "epoch" in d]
    if not cands:
        return None
    steps = []
    for d in cands:
        try:
            steps.append((int(d.split("-")[-1]), d))
        except Exception:
            pass
    if not steps:
        return None
    steps.sort()
    return os.path.join(root_dir, steps[-1][1])


def materialize_trainable_state(obj, device):
    if torch.is_tensor(obj):
        new_obj = obj.to(device).detach().requires_grad_()
        return new_obj
    elif isinstance(obj, dict):
        return {k: materialize_trainable_state(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [materialize_trainable_state(x, device) for x in obj]
    else:
        return obj
