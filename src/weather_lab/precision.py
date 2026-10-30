"""Explicit, scoped model-compute policies; checkpoint parameters remain FP32."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode

POLICIES = ("native", "fp32", "bf16")
VERSION = 1


def describe(policy):
    if policy not in POLICIES:
        raise ValueError(f"precision must be one of {POLICIES}")
    return {"policy": policy, "version": VERSION, "weights": "float32",
            "compute": "bfloat16 mixed" if policy == "bf16" else "float32",
            "math": "upstream" if policy == "native" else "float32 maximum",
            "forecast_storage": "float32"}


class FP32Math(TorchDispatchMode):
    """Cap explicit FP64 tensor requests inside one inference call, without patching torch or Timestral."""
    def __init__(self):
        super().__init__()
        self.narrowed_fp64_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Inference mode may expose aten.to.dtype positional arguments rather than _to_copy keywords.
        if any(value is torch.float64 for value in args):
            args = tuple(torch.float32 if value is torch.float64 else value for value in args)
            self.narrowed_fp64_ops += 1
        if kwargs.get("dtype") is torch.float64:
            kwargs = {**kwargs, "dtype": torch.float32}
            self.narrowed_fp64_ops += 1
        return func(*args, **kwargs)


@contextmanager
def inference(policy, device):
    describe(policy)
    if policy == "native":
        yield None
        return
    kind = torch.device(device).type
    with FP32Math() as guard, torch.autocast(kind, dtype=torch.bfloat16, enabled=policy == "bf16"):
        if policy == "bf16" and not torch.is_autocast_enabled(kind):
            raise RuntimeError(f"BF16 autocast is unavailable on {device}; choose FP32 explicitly")
        yield guard
