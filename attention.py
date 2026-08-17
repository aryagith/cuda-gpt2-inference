"""Bounded, forward-only float32 causal attention."""

from functools import lru_cache
import math
import os
from pathlib import Path

import torch


MAX_SEQUENCE = 256
MAX_TILED_SEQUENCE = 1024
MAX_HEAD_DIM = 128
MAX_SCORES = 16 * 1024 * 1024  # 64 MiB float32 score buffer.


def validate(q, k, v, *, max_sequence=MAX_SEQUENCE, scores=True, dtype=torch.float32):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or min(q.shape) <= 0:
        raise ValueError("expected matching nonempty Q/K/V [batch, heads, sequence, head_dim]")
    if q.dtype != dtype or k.dtype != dtype or v.dtype != dtype:
        raise ValueError(f"only {dtype} is supported")
    if q.device != k.device or q.device != v.device or q.device.type not in ("cpu", "cuda"):
        raise ValueError("Q/K/V must share a CPU or CUDA device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Q/K/V must be contiguous")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("forward-only operator; detach inputs for inference")
    batch, heads, sequence, head_dim = q.shape
    if (sequence > max_sequence or head_dim > MAX_HEAD_DIM
            or (scores and batch * heads * sequence**2 > MAX_SCORES) or q.numel() > MAX_SCORES):
        raise ValueError("baseline workload exceeds sequence, head dimension, or buffer limit")


def reference(q, k, v):
    validate(q, k, v, max_sequence=MAX_TILED_SEQUENCE)
    sequence, head_dim = q.shape[-2:]
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    scores = scores.masked_fill(torch.ones(sequence, sequence, device=q.device, dtype=torch.bool).triu(1),
                                -math.inf)
    return torch.softmax(scores, dim=-1) @ v


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available, load

    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("CUDA-enabled PyTorch and the CUDA toolkit are required; see README.md")
    if not is_ninja_available():
        raise RuntimeError("Ninja is required: python -m pip install ninja")
    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="cuda_attention_naive",
        sources=[str(source / name) for name in
                 ("attention_bindings.cpp", "attention.cu", "attention_tensor_core.cu")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def cuda_attention(q, k, v):
    validate(q, k, v)
    if not q.is_cuda:
        raise ValueError("cuda_attention requires CUDA tensors")
    return extension().attention(q, k, v)


def cuda_attention_tiled(q, k, v):
    validate(q, k, v, max_sequence=MAX_TILED_SEQUENCE, scores=False)
    if not q.is_cuda:
        raise ValueError("cuda_attention_tiled requires CUDA tensors")
    return extension().attention_tiled(q, k, v)


def cuda_attention_query_tiled(q, k, v):
    validate(q, k, v, max_sequence=MAX_TILED_SEQUENCE, scores=False)
    if not q.is_cuda:
        raise ValueError("cuda_attention_query_tiled requires CUDA tensors")
    return extension().attention_query_tiled(q, k, v)


def cuda_attention_tensor_core(q, k, v):
    """Optional FP16, 64-wide causal prefill with FP32 Tensor Core accumulation."""
    validate(q, k, v, max_sequence=MAX_TILED_SEQUENCE, scores=False, dtype=torch.float16)
    if not q.is_cuda or q.shape[-1] != 64:
        raise ValueError("Tensor Core attention requires CUDA tensors with head_dim=64")
    return extension().attention_tensor_core(q, k, v)


def validate_decode(q, k, v):
    if (q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or k.shape != v.shape
            or q.shape[:2] != k.shape[:2] or q.shape[2] != 1 or q.shape[3] != k.shape[3]
            or min(q.shape) <= 0 or k.shape[2] <= 0):
        raise ValueError("expected Q [B,H,1,D] and matching nonempty K/V [B,H,T,D]")
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        raise ValueError("only float32 is supported")
    if q.device != k.device or q.device != v.device or q.device.type not in ("cpu", "cuda"):
        raise ValueError("Q/K/V must share a CPU or CUDA device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("Q/K/V must be contiguous")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("forward-only operator; detach inputs for inference")
    if k.shape[2] > 1024 or q.shape[3] > MAX_HEAD_DIM or k.numel() > MAX_SCORES:
        raise ValueError("decode workload exceeds cache, head dimension, or buffer limit")


def reference_decode(q, k, v):
    validate_decode(q, k, v)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
    return torch.softmax(scores, dim=-1) @ v


def cuda_attention_decode(q, k, v):
    validate_decode(q, k, v)
    if not q.is_cuda:
        raise ValueError("cuda_attention_decode requires CUDA tensors")
    return extension().attention_decode(q, k, v)
