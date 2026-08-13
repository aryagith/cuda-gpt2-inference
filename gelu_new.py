"""Forward-only float32/float16 GPT-2 GELU, with float32 CUDA arithmetic."""

from functools import lru_cache
import math
import os
from pathlib import Path

import torch


def reference(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available, load

    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("CUDA-enabled PyTorch and the CUDA toolkit are required; see README.md")
    if not is_ninja_available():
        raise RuntimeError("Ninja is required: python -m pip install ninja")
    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="cuda_gelu_new",
        sources=[str(source / "gelu_bindings.cpp"), str(source / "gelu.cu")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def cuda_gelu_new(x):
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float16) or not x.is_contiguous() or x.numel() == 0:
        raise ValueError("expected a nonempty contiguous float32 or float16 CUDA tensor")
    if x.requires_grad:
        raise ValueError("forward-only operator; detach inputs for inference")
    return extension().gelu_new(x)


class CudaGeluNew(torch.nn.Module):
    def forward(self, x):
        return cuda_gelu_new(x)


def set_gpt2_cuda_gelu(model, enabled):
    """Replace only the pinned GPT-2 MLP activation in an inference model."""
    if (model.config.model_type != "gpt2" or model.config.activation_function != "gelu_new"
            or model.training):
        raise ValueError("expected an eval-mode GPT-2 with gelu_new activation")
    if enabled:
        activation = CudaGeluNew()
    else:
        from transformers.activations import NewGELUActivation
        activation = NewGELUActivation()
    for block in model.transformer.h:
        block.mlp.act = activation
