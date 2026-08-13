"""Paired FP32/FP16 GPT-2 requests with native/custom GELU and separate token diagnostics."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_greedy import PROMPTS
from gelu_new import set_gpt2_cuda_gelu
from model_demo import (MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256,
                        greedy_generate, input_ids, load_model)


def summarize(values):
    """Nearest-rank p95; preserve samples because small sets have coarse tails."""
    ordered = sorted(values)
    return {"median": statistics.median(values),
            "p95": ordered[(95 * len(ordered) + 99) // 100 - 1], "samples": values}


def gpu_state():
    fields = "index,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, timeout=5)
        return {"fields": fields, "values": result.stdout.strip(),
                "error": result.stderr.strip(), "exit_code": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3, help="passes over all prompts")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/precision-generation.json"))
    args = parser.parse_args()
    if min(args.new_tokens, args.warmup, args.repeats) < 1 or not torch.cuda.is_available():
        parser.error("CUDA and positive new-tokens/warmup/repeats are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, model32 = load_model()
    _, model16 = load_model(torch.float16)
    routes = (("fp32_native", model32, False), ("fp32_cuda", model32, True),
              ("fp16_native", model16, False), ("fp16_cuda", model16, True))
    texts = ([line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
              if line.strip()] if args.prompts_file else PROMPTS)
    if not texts:
        parser.error("at least one nonempty prompt is required")
    lengths = [input_ids(tokenizer, text, None).shape[1] for text in texts]
    if max(lengths) + args.new_tokens > model32.config.n_positions:
        parser.error("prompt plus generated tokens exceeds GPT-2's position limit")

    # Teacher-forced logits compare the same input sequence across precision.
    # Cross-precision text equality is recorded, not assumed from tensor closeness.
    correctness = []
    for text in texts:
        ids = input_ids(tokenizer, text, None)
        logits = {}
        for name, model, custom in routes:
            set_gpt2_cuda_gelu(model, custom)
            model.set_attn_implementation("sdpa")
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                logits[name] = model(ids, use_cache=False).logits.float()
            if not torch.isfinite(logits[name]).all():
                raise AssertionError(f"nonfinite logits in {name}")
        gold = logits["fp32_native"]
        correctness.append({name: {
            "max_abs_logit_error_vs_fp32_native": (value - gold).abs().max().item(),
            "mean_abs_logit_error_vs_fp32_native": (value - gold).abs().mean().item(),
            "next_token_argmax_matches_fp32_native": bool(
                value[:, -1].argmax(-1).eq(gold[:, -1].argmax(-1)).all()),
            "teacher_forced_nll": torch.nn.functional.cross_entropy(
                value[:, :-1].reshape(-1, value.shape[-1]), ids[:, 1:].reshape(-1)).item()
                if ids.shape[1] > 1 else None,
        } for name, value in logits.items()})
        torch.testing.assert_close(logits["fp32_cuda"], gold, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(logits["fp16_cuda"], logits["fp16_native"], rtol=0, atol=0)
        del logits

    def run(text, model, token_events=None):
        ids = input_ids(tokenizer, text, None)
        output = greedy_generate(model, ids, args.new_tokens, token_events=token_events)
        return output, tokenizer.decode(output[0], skip_special_tokens=True)

    telemetry = [{"stage": "before_warmup", **gpu_state()}]
    for index in range(args.warmup):
        for _, model, custom in routes:
            set_gpt2_cuda_gelu(model, custom)
            run(texts[index % len(texts)], model)
    samples = {name: [] for name, _, _ in routes}
    peaks = {name: [] for name, _, _ in routes}
    mismatches = {"fp32_native_vs_cuda": [], "fp16_native_vs_cuda": [],
                  "fp32_native_vs_fp16_native": [], "fp32_cuda_vs_fp16_cuda": []}
    for index in range(args.repeats * len(texts)):
        if index % len(texts) == 0:
            telemetry.append({"stage": f"pass_{index // len(texts)}", **gpu_state()})
        outputs = {}
        order = routes[index % len(routes):] + routes[:index % len(routes)]
        if index % 2:
            order = tuple(reversed(order))
        for name, model, custom in order:
            set_gpt2_cuda_gelu(model, custom)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            start = time.perf_counter()
            outputs[name] = run(texts[index % len(texts)], model)
            torch.cuda.synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
            peaks[name].append(torch.cuda.max_memory_allocated() - resident)
        for pair, unequal in mismatches.items():
            left, right = pair.split("_vs_")
            if right == "cuda":
                right = left.split("_")[0] + "_cuda"
            if (not torch.equal(outputs[left][0], outputs[right][0])
                    or outputs[left][1] != outputs[right][1]):
                unequal.append(index % len(texts))
        if mismatches["fp32_native_vs_cuda"]:
            raise AssertionError("float32 custom GELU changed a generated output")
        if mismatches["fp16_native_vs_cuda"]:
            raise AssertionError("float16 custom GELU changed a generated output")
        del outputs
    telemetry.append({"stage": "after_request_timing", **gpu_state()})

    # Event instrumentation is excluded from primary wall-time samples. Events
    # mark GPU argmax completion; they do not represent user-delivered streaming.
    phases = {name: {"first_token_device_ready_ms": [], "inter_token_device_ms": []}
              for name, _, _ in routes}
    for index, text in enumerate(texts):
        order = routes[index % len(routes):] + routes[:index % len(routes)]
        for name, model, custom in order:
            set_gpt2_cuda_gelu(model, custom)
            start = torch.cuda.Event(enable_timing=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(args.new_tokens)]
            torch.cuda.synchronize()
            start.record()
            run(text, model, events)
            torch.cuda.synchronize()
            phases[name]["first_token_device_ready_ms"].append(start.elapsed_time(events[0]))
            phases[name]["inter_token_device_ms"].extend(
                a.elapsed_time(b) for a, b in zip(events, events[1:]))

    root = Path(__file__).resolve().parent
    sources = ("benchmark_precision.py", "benchmark_greedy.py", "model_demo.py", "gelu_new.py",
               "csrc/gelu_bindings.cpp", "csrc/gelu.cu")
    comparisons = (("fp32_native", "fp16_native"), ("fp32_cuda", "fp16_cuda"),
                   ("fp16_native", "fp16_cuda"), ("fp32_native", "fp32_cuda"))
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256, "tf32_matmul": False,
        "prompts": list(texts), "prompt_lengths": lengths, "new_tokens": args.new_tokens,
        "warmup": args.warmup, "repeats_per_prompt": args.repeats,
        "timing": "rotated/reversed paired synchronized text-to-text wall time; tokenization, transfer, dynamic cache setup, prefill, decode, text decoding included; module swaps, model load, compilation, warmup, telemetry, phase diagnostics excluded; no graphs",
        "phase_timing": "separate event-instrumented pass per prompt/route; device readiness at argmax completion, including host enqueue gaps; first interval includes tokenization and input transfer, excludes output detokenization/network/streaming delivery; no per-token host synchronization",
        "throughput": "output count / full-request seconds; batch one, unpadded, fixed count, no early EOS",
        "power_thermal_conditions": "recorded, not controlled", "gpu_telemetry": telemetry,
        "parameter_bytes": {name: sum(p.numel() * p.element_size() for p in model.parameters())
                            for name, model in (("fp32", model32), ("fp16", model16))},
        "memory": "incremental peak above both resident models; excludes model parameter memory listed separately",
        "teacher_forced_correctness": correctness, "mismatched_prompt_indices": mismatches,
        "paired_wins": {f"{right}_over_{left}": sum(b < a for a, b in zip(samples[left], samples[right]))
                        for left, right in comparisons},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sources},
        "results": {name: {"request_ms": summarize(values),
                           "output_tokens_per_s": summarize([1000 * args.new_tokens / ms for ms in values]),
                           "max_incremental_peak_bytes": max(peaks[name]),
                           **{key: summarize(value) if value else None
                              for key, value in phases[name].items()}}
                    for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"median_request_ms": {name: result["request_ms"]["median"]
                                           for name, result in report["results"].items()},
                      "paired_wins": report["paired_wins"], "mismatches": mismatches,
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
