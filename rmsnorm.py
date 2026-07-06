"""Forward-only, float32 RMSNorm reference and lazily built CUDA extension."""

from functools import lru_cache
import math
import os
from pathlib import Path

import torch


def validate(x, weight, eps):
    if x.ndim != 2 or weight.ndim != 1 or x.shape[1] != weight.numel():
        raise ValueError("expected x [rows, width] and weight [width]")
    if min(x.shape) <= 0 or x.shape[0] > 2**31 - 1:
        raise ValueError("dimensions must be positive; rows must fit int32")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("only float32 is supported")
    if x.device != weight.device or x.device.type not in ("cpu", "cuda"):
        raise ValueError("inputs must share a CPU or CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("forward-only operator; detach inputs for inference")
    limits = torch.finfo(torch.float32)
    if not math.isfinite(eps) or not limits.tiny <= eps <= limits.max:
        raise ValueError("eps must be positive, finite, and in the float32 normal range")


def reference(x, weight, eps=1e-5):
    validate(x, weight, eps)
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available, load

    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("CUDA-enabled PyTorch and the CUDA toolkit are required; see README.md")
    if not is_ninja_available():
        raise RuntimeError("Ninja is required: python -m pip install ninja")
    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="cuda_attention_rmsnorm",
        sources=[str(source / "bindings.cpp"), str(source / "rmsnorm.cu")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def cuda_rmsnorm(x, weight, eps=1e-5):
    validate(x, weight, eps)
    if not x.is_cuda:
        raise ValueError("cuda_rmsnorm requires CUDA tensors")
    return extension().rmsnorm(x, weight, eps)
