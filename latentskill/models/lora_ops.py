from collections.abc import Mapping, Sequence
import torch

from latentskill.utils.logging import get_logger


logger = get_logger("lora_ops")


def iter_trainable_tensors(tree, prefix="root"):
    """Yield leaf tensors with requires_grad=True from a nested structure."""
    if isinstance(tree, Mapping):
        for k, v in tree.items():
            yield from iter_trainable_tensors(v, prefix=f"{prefix}.{k}")
    elif isinstance(tree, Sequence) and not isinstance(tree, (str, bytes)):
        for i, v in enumerate(tree):
            yield from iter_trainable_tensors(v, prefix=f"{prefix}[{i}]")
    elif torch.is_tensor(tree):
        if tree.requires_grad:
            if tree.is_leaf:
                yield tree
            else:
                raise ValueError(
                    f"Found non-leaf tensor at '{prefix}': "
                    f"shape={tuple(tree.shape)}, grad_fn={tree.grad_fn}"
                )


def merge_adapter_states(lora1: dict, lora2: dict, method: str) -> dict:
    """
    Merge two adapter states by concatenating along the rank dimension.

    Leaf tensors:
      A: [Lb, in, r]   -> concat dim=2  => [Lb, in, r1+r2]
      B: [Lb, r, out]  -> concat dim=1  => [Lb, r1+r2, out]
      C: [Lb, out]     -> sum           => [Lb, out]

    Assumes lora1 and lora2 have identical key structure and same Lb/in/out.
    """

    if method != "rl":
        raise ValueError(f"Unsupported merge method: {method}. Only 'rl' is supported.")
    if lora1 is None:
        return lora2
    if lora2 is None:
        return lora1

    def _merge_leaf(d1, d2, path=""):
        # Leaf node: {"A":..., "B":..., "C":...}
        if isinstance(d1, dict) and "A" in d1 and "B" in d1:
            A1, B1 = d1["A"], d1["B"]
            A2, B2 = d2["A"], d2["B"]

            if A1 is None or A2 is None or B1 is None or B2 is None:
                raise ValueError(f"{path}: A/B cannot be None for rank-concat merge.")

            # Check batch + core dims match
            if A1.shape[0] != A2.shape[0]:
                raise ValueError(f"{path}.A: Lb mismatch {A1.shape[0]} vs {A2.shape[0]}")
            if B1.shape[0] != B2.shape[0]:
                raise ValueError(f"{path}.B: Lb mismatch {B1.shape[0]} vs {B2.shape[0]}")

            # A: [Lb, in, r]
            if A1.shape[1] != A2.shape[1]:
                raise ValueError(f"{path}.A: in_features mismatch {A1.shape[1]} vs {A2.shape[1]}")
            # B: [Lb, r, out]
            if B1.shape[2] != B2.shape[2]:
                raise ValueError(f"{path}.B: out_features mismatch {B1.shape[2]} vs {B2.shape[2]}")

            # Each LoRA pair must be internally rank-consistent.
            if A1.shape[2] != B1.shape[1]:
                raise ValueError(f"{path}: r mismatch inside lora1: A.r={A1.shape[2]} vs B.r={B1.shape[1]}")
            if A2.shape[2] != B2.shape[1]:
                raise ValueError(f"{path}: r mismatch inside lora2: A.r={A2.shape[2]} vs B.r={B2.shape[1]}")

            out = {
                "A": torch.cat([A1, A2], dim=2),  # concat r
                "B": torch.cat([B1, B2], dim=1),  # concat r
            }

            C1 = d1.get("C", None)
            C2 = d2.get("C", None)
            if (C1 is None) != (C2 is None):
                raise ValueError(f"{path}.C: one is None, the other is not.")
            out["C"] = None if C1 is None else (C1 + C2)

            return out

        # Recurse through nested dicts
        if isinstance(d1, dict) and isinstance(d2, dict):
            if d1.keys() != d2.keys():
                raise ValueError(f"{path}: key mismatch {set(d1.keys())} vs {set(d2.keys())}")
            return {
                k: _merge_leaf(
                    d1[k],
                    d2[k],
                    path=f"{path}.{k}" if path else str(k),
                )
                for k in d1.keys()
            }

        raise TypeError(f"{path}: unsupported types {type(d1)} and {type(d2)}")

    return _merge_leaf(lora1, lora2)


def freeze_adapter_state(adapter_state: dict) -> dict:
    """
    Freeze all torch.Tensors inside a nested adapter state in-place
    by setting requires_grad_(False).

    This does not detach tensors or replace tensor objects, so it is safe for
    shared references and cached adapter states.

    Returns the same adapter_state object (mutated).
    """

    def _walk(obj):
        if torch.is_tensor(obj):
            obj.requires_grad_(False)
            return

        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
            return

        if isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
            return

        # Ignore None and scalar values.
        return

    _walk(adapter_state)
    return adapter_state


def adapter_state_requires_grad(adapter_state: dict, expected: bool, *, verbose: bool = True) -> bool:
    """
    Check whether all torch.Tensors inside a nested adapter state
    have requires_grad == expected, and print all mismatches.

    Args:
        adapter_state: nested dict structure containing torch.Tensors.
        expected: True for all trainable tensors, False for all frozen tensors.
        verbose: if True, print all mismatched tensors.

    Returns:
        True if all tensors match, otherwise False.
    """
    if adapter_state is None:
        return True

    wrong = []

    def _walk(obj, path="root"):
        if torch.is_tensor(obj):
            if obj.requires_grad != expected:
                wrong.append((path, obj))
            return obj.requires_grad == expected

        if isinstance(obj, dict):
            ok = True
            for k, v in obj.items():
                ok = _walk(v, f"{path}.{k}") and ok
            return ok

        if isinstance(obj, (list, tuple)):
            ok = True
            for i, v in enumerate(obj):
                ok = _walk(v, f"{path}[{i}]") and ok
            return ok

        return True  # ignore None / non-tensors

    all_ok = _walk(adapter_state)

    if verbose and wrong:
        logger.warning(
            f"[adapter_state_requires_grad] Found {len(wrong)} tensor(s) "
            f"with requires_grad != {expected}:"
        )
        for p, t in wrong:
            logger.warning(
                f" - {p}: requires_grad={t.requires_grad}, "
                f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"
            )

    return all_ok
