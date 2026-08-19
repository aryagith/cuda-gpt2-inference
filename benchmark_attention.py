"""Compare causal attention correctness, CUDA graph latency, and peak allocation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention import (MAX_SEQUENCE, MAX_SCORES, MAX_HEAD_DIM, cuda_attention,
                       cuda_attention_tiled, cuda_attention_query_tiled, cuda_attention_tensor_core, reference)
from benchmark import measure, positive


def sdpa(q, k, v, backend):
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)


def select_fused(q, k, v):
    failures = {}
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            sdpa(q, k, v, backend)
            torch.cuda.synchronize()
            return backend, failures
        except RuntimeError as exc:
            failures[backend.name] = str(exc).splitlines()[0]
    return None, failures


def peak_increment(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    return peak


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=positive, default=4)
    parser.add_argument("--heads", type=positive, default=8)
    parser.add_argument("--head-dim", type=positive, default=64)
    parser.add_argument("--sequences", type=lambda text: [positive(x) for x in text.split(",")],
                        default=[1, 17, 33, 129, 256])
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--repeats", type=positive, default=7)
    parser.add_argument("--iterations", type=positive, default=5)
    parser.add_argument("--graph-ops", type=positive, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensor-core", action="store_true", help="compare FP16 Tensor Core and native FP16 attention")
    args = parser.parse_args()
    if args.tensor_core and args.head_dim != 64:
        parser.error("Tensor Core attention requires --head-dim 64")
    if (args.head_dim > MAX_HEAD_DIM or any(s > 1024 or
            args.batch * args.heads * s * s > MAX_SCORES for s in args.sequences)):
        parser.error("head-dim must be <=128, sequence <=1024, and scores <=16M elements")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)

    shapes = []
    for sequence in args.sequences:
        shape = (args.batch, args.heads, sequence, args.head_dim)
        q, k, v = (torch.randn(shape, device="cuda",
                             dtype=torch.float16 if args.tensor_core else torch.float32) for _ in range(3))
        math_expected = sdpa(q, k, v, SDPBackend.MATH)
        expected = reference(q.float(), k.float(), v.float())
        rtol, atol = (1e-2, 3e-3) if args.tensor_core else (2e-4, 2e-5)
        torch.testing.assert_close(expected, math_expected.float(), rtol=rtol, atol=atol)
        fused_backend, failures = select_fused(q, k, v)
        functions = {"sdpa_math": lambda: sdpa(q, k, v, SDPBackend.MATH)}
        if args.tensor_core:
            functions["tensor_core"] = lambda: cuda_attention_tensor_core(q, k, v)
            functions["sdpa_efficient_attention"] = lambda: sdpa(q, k, v, SDPBackend.EFFICIENT_ATTENTION)
        else:
            functions.update(tiled=lambda: cuda_attention_tiled(q, k, v),
                             query_tiled=lambda: cuda_attention_query_tiled(q, k, v))
        if sequence <= MAX_SEQUENCE and not args.tensor_core:
            functions["naive"] = lambda: cuda_attention(q, k, v)
        if fused_backend is not None:
            functions["sdpa_" + fused_backend.name.lower()] = lambda: sdpa(q, k, v, fused_backend)
        results = {}
        for name, fn in functions.items():
            actual = fn()
            torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol)
            results[name] = {
                "max_abs_error_vs_oracle": (actual - expected).abs().max().item(),
                "peak_allocated_increment_bytes": peak_increment(fn),
                **measure(fn, "cuda", args.warmup, args.repeats, args.iterations, args.graph_ops),
            }
        shapes.append({"shape": shape,
                       "oracle": "explicit reference",
                       "naive_skipped_reason": "FP16 experiment" if args.tensor_core else
                           (None if sequence <= MAX_SEQUENCE else "naive softmax supports at most 256 tokens"),
                       "fused_backend": fused_backend.name if fused_backend else None,
                       "unavailable_fused_backends": failures, "results": results})

    root = Path(__file__).resolve().parent
    sources = [root / name for name in ("attention.py", "benchmark_attention.py", "benchmark.py",
                                       "csrc/attention_bindings.cpp", "csrc/attention.cu",
                                       "csrc/attention_tensor_core.cu")]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "os": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "dtype": "float16" if args.tensor_core else "float32", "causal": True,
        "dropout": 0.0, "tf32_matmul": False, "seed": 42,
        "warmup": args.warmup, "repeats": args.repeats, "iterations": args.iterations,
        "graph_ops": args.graph_ops, "timing": "CUDA graph replay events; milliseconds per operation",
        "memory": "peak torch.cuda.memory_allocated increment for one warmed invocation",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "shapes": shapes,
    }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
