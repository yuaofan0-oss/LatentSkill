import importlib

import torch

from latentskill.utils.ddp import local_process_rank


def _resolve_device(device_cfg: str) -> torch.device:
    # In DDP we hard-bind to LOCAL_RANK cuda device when available.
    if device_cfg == "auto":
        if torch.cuda.is_available():
            local_rank = local_process_rank()
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    if device_cfg in ("cuda", "cpu"):
        if device_cfg == "cuda" and torch.cuda.is_available():
            local_rank = local_process_rank()
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device(device_cfg)
    raise ValueError(f"Unsupported device setting: {device_cfg}")


def _import_class(path: str):
    if "." not in path:
        raise ValueError("model.class_path must be 'module.ClassName'")
    mod_name, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)
