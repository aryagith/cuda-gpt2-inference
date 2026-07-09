"""Compare one-query cache attention with efficient SDPA on resident CUDA tensors."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from attention import cuda_attention_decode, reference_decode
from benchmark import measure, positive


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=positive, default=1)
    parser.add_argument("--heads", type=positive, default=12)
    parser.add_argument("--head-dim", type=positive, default=64)
    parser.add_argument("--lengths", type=lambda text: [positive(x) for x in text.split(",")],
                        default=[1, 17, 33, 129, 256, 512, 1024])
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--repeats", type=positive, default=7)
    parser.add_argument("--iterations", type=positive, default=5)
    parser.add_argument("--graph-ops", type=positive, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is required")
    if args.head_dim > 128 or max(args.lengths) > 1024:
        parser.error("head-dim must be <=128 and cache lengths <=1024")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)

    shapes = []
    for length in args.lengths:
        q = torch.randn(args.batch, args.heads, 1, args.head_dim, device="cuda")
        k, v = (torch.randn(args.batch, args.heads, length, args.head_dim, device="cuda") for _ in range(2))
        expected = reference_decode(q, k, v)
        def sdpa():
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

        results = {}
        for name, fn in (("custom_decode", lambda: cuda_attention_decode(q, k, v)),
                         ("efficient_sdpa", sdpa)):
            actual = fn()
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = fn()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - baseline
            del output
            results[name] = {"max_abs_error_vs_oracle": (actual - expected).abs().max().item(),
                             "peak_allocated_increment_bytes": peak,
                             **measure(fn, "cuda", args.warmup, args.repeats, args.iterations, args.graph_ops)}
        shapes.append({"cache_length": length, "results": results})

    root = Path(__file__).resolve().parent
    sources = ("benchmark_decode.py", "attention.py", "csrc/attention_bindings.cpp", "csrc/attention.cu")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "os": platform.platform(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
              "batch": args.batch, "heads": args.heads, "head_dim": args.head_dim,
              "dtype": "float32", "backend": "PyTorch EFFICIENT_ATTENTION forced",
              "mask": "none; all K/V positions are past or current", "seed": 42, "tf32_matmul": False,
              "warmup": args.warmup, "repeats": args.repeats, "iterations": args.iterations,
              "graph_ops": args.graph_ops, "timing": "CUDA graph replay events; milliseconds per operation",
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
              "shapes": shapes}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
