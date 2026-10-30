"""ZeroGPU integration. Import this module before torch/gradio in the app.

On a ZeroGPU Space (SPACES_ZERO_GPU=true) `spaces` must be imported before CUDA is
initialised; it then patches torch, and functions decorated with `gpu(...)` get a GPU
for their duration. Elsewhere the decorator has no effect and inference stays on CPU.
Explicit Forecast inference runs inside `gpu`; Benchmark reads saved results.
"""

from __future__ import annotations

import os

try:
    import spaces

    gpu = spaces.GPU
except ImportError:  # the app also runs without the `spaces` package
    def gpu(task=None, **_):
        return task if task is not None else (lambda f: f)


def on_zerogpu() -> bool:
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")


def inference_device() -> str:
    """CUDA when a GPU is attached (inside a `gpu` function on ZeroGPU), else CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
