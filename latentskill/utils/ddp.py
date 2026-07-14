import os

import torch
import torch.distributed as dist


# ========= DDP helpers =========
def distributed_requested() -> bool:
    # If launched with torchrun, WORLD_SIZE will be set (>1 for multi-proc)
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def process_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def local_process_rank() -> int:
    # torchrun sets LOCAL_RANK; default to 0 for single GPU/CPU
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_primary_process() -> bool:
    return process_rank() == 0

def distributed_barrier():
    if is_distributed():
        dist.barrier()

def initialize_distributed():
    # Only initialize if we're truly in a multi-process setting
    if distributed_requested() and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        # Make printing on non-zero ranks quieter
        if not is_primary_process():
            import builtins as __builtin__

            def _suppress_output(*args, **kwargs):
                pass

            __builtin__.print = _suppress_output

def cleanup_distributed():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()

@torch.no_grad()
def mean_across_processes(value: float, device: torch.device) -> float:
    """Average a scalar across processes."""
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= world_size()
    return float(t.item())
# ==============================
