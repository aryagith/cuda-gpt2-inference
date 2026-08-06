"""Opt-in fused attention residual and ln_2 for pinned float32 GPT-2 inference."""

from functools import lru_cache
import os
from pathlib import Path
from types import MethodType

import torch


def reference(attention, residual, weight, bias, eps):
    summed = attention + residual
    return summed, torch.nn.functional.layer_norm(summed, (768,), weight, bias, eps)


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, is_ninja_available, load

    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError("CUDA-enabled PyTorch and the CUDA toolkit are required; see README.md")
    if not is_ninja_available():
        raise RuntimeError("Ninja is required: python -m pip install ninja")
    source = Path(__file__).resolve().parent / "csrc"
    return load(
        name="cuda_fused_residual_ln",
        sources=[str(source / "fused_ln_bindings.cpp"), str(source / "fused_ln.cu")],
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=True,
    )


def cuda_fused_residual_ln(attention, residual, weight, bias, eps):
    if (attention.ndim != 3 or attention.shape[-1] != 768 or attention.numel() == 0
            or not attention.is_cuda or attention.dtype != torch.float32
            or not attention.is_contiguous() or attention.requires_grad):
        raise ValueError("expected nonempty contiguous [B,S,768] float32 CUDA without grad")
    if (residual.shape != attention.shape or not residual.is_cuda
            or residual.device != attention.device or residual.dtype != torch.float32
            or not residual.is_contiguous() or residual.requires_grad):
        raise ValueError("residual must match attention")
    for param in (weight, bias):
        if (param.shape != (768,) or not param.is_cuda or param.device != attention.device
                or param.dtype != torch.float32 or not param.is_contiguous()
                or (param.requires_grad and torch.is_grad_enabled())):
            raise ValueError("weight and bias must be contiguous [768] float32 CUDA without active autograd")
    return extension().fused_residual_ln(attention, residual, weight, bias, eps)


def _fused_block_forward(
    self, hidden_states, past_key_values=None, cache_position=None,
    attention_mask=None, head_mask=None, encoder_hidden_states=None,
    encoder_attention_mask=None, use_cache=False, output_attentions=False,
    **kwargs,
):
    if encoder_hidden_states is not None or encoder_attention_mask is not None:
        raise ValueError("cross-attention is outside the fused GPT-2 contract")
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_output, attn_weights = self.attn(
        hidden_states, past_key_values=past_key_values,
        cache_position=cache_position, attention_mask=attention_mask,
        head_mask=head_mask, use_cache=use_cache,
        output_attentions=output_attentions, **kwargs,
    )
    residual, normalized = extension().fused_residual_ln(
        attn_output, residual, self.ln_2.weight, self.ln_2.bias, self.ln_2.eps)
    hidden_states = residual + self.mlp(normalized)
    return (hidden_states, attn_weights) if output_attentions else (hidden_states,)


def set_gpt2_fused_ln(model, enabled):
    if (model.config.model_type != "gpt2" or model.config.n_embd != 768
            or model.config.add_cross_attention or model.training):
        raise ValueError("expected eval-mode GPT-2 width 768 without cross-attention")
    for block in model.transformer.h:
        current = block.__dict__.get("forward")
        if current is not None and getattr(current, "__func__", None) is not _fused_block_forward:
            raise ValueError("GPT-2 block already has a custom forward method")
        if enabled and current is None:
            block.forward = MethodType(_fused_block_forward, block)
        elif not enabled and current is not None:
            del block.forward
