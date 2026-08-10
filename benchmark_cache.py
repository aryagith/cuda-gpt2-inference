"""Compare fresh dynamic and static GPT-2 caches without graphs or compilation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from benchmark_greedy import PROMPTS
from gelu_new import set_gpt2_cuda_gelu
from model_demo import (MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256,
                        greedy_generate, input_ids, load_model)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/cache-generation.json"))
    args = parser.parse_args()
    if min(args.new_tokens, args.warmup, args.repeats) < 1:
        parser.error("new-tokens, warmup, and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, model = load_model()
    texts = ([line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
              if line.strip()] if args.prompts_file else PROMPTS)
    if not texts:
        parser.error("at least one nonempty prompt is required")
    lengths = [input_ids(tokenizer, text, None).shape[1] for text in texts]
    if max(lengths) + args.new_tokens > model.config.n_positions:
        parser.error("prompt plus generated tokens exceeds GPT-2's position limit")
    set_gpt2_cuda_gelu(model, True)
    routes = (("dynamic", False), ("static", True))

    def run(text, static):
        ids = input_ids(tokenizer, text, None)
        output = greedy_generate(model, ids, args.new_tokens, static_cache=static)
        return output, tokenizer.decode(output[0], skip_special_tokens=True)

    for index in range(args.warmup):
        for _, static in routes:
            run(texts[index % len(texts)], static)
    samples = {name: [] for name, _ in routes}
    peaks = {name: [] for name, _ in routes}
    for index in range(args.repeats * len(texts)):
        outputs = {}
        for name, static in (routes if index % 2 == 0 else tuple(reversed(routes))):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            start = time.perf_counter()
            outputs[name] = run(texts[index % len(texts)], static)
            torch.cuda.synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
            peaks[name].append(torch.cuda.max_memory_allocated() - resident)
        if (not torch.equal(outputs["dynamic"][0], outputs["static"][0])
                or outputs["dynamic"][1] != outputs["static"][1]):
            raise AssertionError(f"cache outputs differ for prompt {index % len(texts)}")
        del outputs

    # Diagnostic profiling is separate from request timing. aten::cat also
    # includes the final output assembly, so record that one remaining call.
    profiles = {}
    for name, static in routes:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profile:
            run(texts[0], static)
            torch.cuda.synchronize()
        profiles[name] = {event.key: event.count for event in profile.key_averages()
                          if event.key in ("aten::cat", "aten::index_copy_", "aten::zeros")}

    root = Path(__file__).resolve().parent
    sources = ("benchmark_cache.py", "benchmark_greedy.py", "model_demo.py", "gelu_new.py",
               "csrc/gelu_bindings.cpp", "csrc/gelu.cu")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256, "dtype": "float32", "tf32_matmul": False,
        "cuda_gelu": True, "fused_ln": False, "new_tokens": args.new_tokens,
        "prompts": list(texts), "prompt_lengths": lengths,
        "warmup": args.warmup, "repeats_per_prompt": args.repeats,
        "timing": "paired synchronized text-to-text wall time, alternating order; fresh cache allocation, tokenization, transfer, prefill, decode, text decoding included; compilation, model load, warmup, profiling excluded; no graphs",
        "generation": "batch one, unpadded, fixed count, no early EOS; static capacity = prompt length + new_tokens",
        "throughput": "output tokens/s = new_tokens / full request seconds; not steady-state decode throughput",
        "token_ids_and_text_equal": True, "profile_operator_counts": profiles,
        "static_paired_wins": sum(b < a for a, b in zip(samples["dynamic"], samples["static"])),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sources},
        "results": {name: {"median_ms": statistics.median(values),
                           "median_output_tokens_per_s": statistics.median(
                               1000 * args.new_tokens / ms for ms in values),
                           "samples_ms": values,
                           "max_incremental_peak_bytes": max(peaks[name])}
                    for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("prompt_lengths", "new_tokens",
                     "token_ids_and_text_equal", "static_paired_wins",
                     "profile_operator_counts", "results")}, indent=2))


if __name__ == "__main__":
    main()
