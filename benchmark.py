"""Compare eager reference, PyTorch RMSNorm, and optionally the custom CUDA kernel."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from rmsnorm import cuda_rmsnorm, reference


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def measure(fn, device, warmup, repeats, iterations, graph_ops=0):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    if graph_ops:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_ops):
                fn()
        graph.replay()
        torch.cuda.synchronize()
    timed_fn = graph.replay if graph_ops else fn
    samples = []
    for _ in range(repeats):
        if device == "cuda":
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                timed_fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop) / (iterations * max(graph_ops, 1)))
        else:
            start = time.perf_counter()
            for _ in range(iterations):
                fn()
            samples.append((time.perf_counter() - start) * 1000 / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--custom", action="store_true", help="also build and benchmark our CUDA kernel")
    parser.add_argument("--rows", type=positive, default=1024)
    parser.add_argument("--width", type=positive, default=1024)
    parser.add_argument("--warmup", type=positive, default=20)
    parser.add_argument("--repeats", type=positive, default=7)
    parser.add_argument("--iterations", type=positive, default=50)
    parser.add_argument("--graph-ops", type=positive, default=0,
                        help="capture this many operations per CUDA graph replay")
    parser.add_argument("--output", type=Path, help="save the report as JSON")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is unavailable; use --device cpu for reference measurements")
    if args.custom and args.device != "cuda":
        parser.error("--custom requires --device cuda")
    if args.graph_ops and args.device != "cuda":
        parser.error("--graph-ops requires --device cuda")

    torch.manual_seed(42)
    x = torch.randn(args.rows, args.width, device=args.device)
    w = torch.randn(args.width, device=args.device)
    functions = {"torch_eager": lambda: reference(x, w),
                 "torch_native": lambda: F.rms_norm(x, (args.width,), w, eps=1e-5)}
    if args.custom:
        functions["custom_cuda"] = lambda: cuda_rmsnorm(x, w)
    expected = functions["torch_native"]()
    eager_expected = functions["torch_eager"]()
    results = {}
    for name, fn in functions.items():
        actual = fn()  # Build, allocate, and check correctness before warmup or timing.
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(actual, eager_expected, rtol=2e-5, atol=3e-6)
        error = (actual - expected).abs().max().item()
        eager_error = (actual - eager_expected).abs().max().item()
        results[name] = {"max_abs_error": error, "max_abs_error_vs_eager": eager_error,
                         **measure(fn, args.device, args.warmup, args.repeats, args.iterations, args.graph_ops)}

    root = Path(__file__).resolve().parent
    sources = [root / name for name in ("rmsnorm.py", "benchmark.py", "csrc/bindings.cpp", "csrc/rmsnorm.cu")]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "os": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "device": args.device, "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "compute_capability": torch.cuda.get_device_capability() if args.device == "cuda" else None,
        "shape": [args.rows, args.width], "dtype": "float32", "eps": 1e-5, "seed": 42,
        "warmup": args.warmup, "repeats": args.repeats, "iterations": args.iterations,
        "graph_ops": args.graph_ops,
        "timing": "CUDA graph events" if args.graph_ops else "CUDA events" if args.device == "cuda" else "wall clock",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "results": results,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
