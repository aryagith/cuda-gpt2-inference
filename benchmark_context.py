"""Sweep GPT-2 context lengths with paired attention routes and fresh caches."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark import positive
from benchmark_greedy import PROMPTS
from gelu_new import set_gpt2_cuda_gelu
from model_demo import (ATTENTION_NAME, CALLS, MODEL_ID, MODEL_REVISION,
                        MODEL_WEIGHTS_SHA256, TENSOR_ATTENTION_NAME, TENSOR_CALLS,
                        greedy_generate, input_ids, load_model, measure)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", default="128,256,512,992")
    parser.add_argument("--new-tokens", type=positive, default=32)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/context.json"))
    parser.add_argument("--tensor-core", action="store_true",
                        help="compare optional Tensor Core prefill/native decode against native FP16")
    args = parser.parse_args()
    try:
        lengths = [positive(x) for x in args.lengths.split(",")]
    except (ValueError, argparse.ArgumentTypeError):
        parser.error("lengths must be comma-separated positive integers")
    if any(s + args.new_tokens > 1024 for s in lengths):
        parser.error("each prompt plus output count must fit GPT-2's 1024 positions")
    if not torch.cuda.is_available():
        parser.error("CUDA-enabled PyTorch is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, model = load_model(torch.float16 if args.tensor_core else torch.float32)
    set_gpt2_cuda_gelu(model, True)
    custom_route = TENSOR_ATTENTION_NAME if args.tensor_core else ATTENTION_NAME
    counters = TENSOR_CALLS if args.tensor_core else CALLS
    routes = {"native_sdpa": "sdpa", "custom_attention_adapter": custom_route}
    shapes = []
    for length in lengths:
        ids = input_ids(tokenizer, PROMPTS[0], length, max_length=1024)
        checks, prefill = {}, {}
        baseline = None
        for name, implementation in routes.items():
            model.set_attn_implementation(implementation)
            counters.update({key: 0 for key in counters})
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                output = model(ids, use_cache=True, logits_to_keep=1)
                calls = counters.copy()
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                cached = model(token, past_key_values=output.past_key_values,
                               use_cache=True, logits_to_keep=1)
                logits = (output.logits.clone(), cached.logits.clone())
                if baseline is None:
                    baseline = logits
                else:
                    for actual, expected in zip(logits, baseline):
                        torch.testing.assert_close(actual, expected,
                            rtol=2e-2 if args.tensor_core else 1e-3,
                            atol=0.15 if args.tensor_core else 5e-3)
                checks[name] = {"prefill_calls": calls,
                               "prefill_max_logit_error": (logits[0] - baseline[0]).abs().max().item(),
                               "cached_max_logit_error": (logits[1] - baseline[1]).abs().max().item()}
                if implementation == TENSOR_ATTENTION_NAME:
                    assert calls == {"prefill": 12, "native": 0}, calls
                    assert counters == {"prefill": 12, "native": 12}, counters
                elif implementation == ATTENTION_NAME:
                    assert calls["query_tiled"] == (12 if length >= 64 else 0)
                    assert calls["single_query_tiled"] == (12 if length < 64 else 0)
                    assert calls["sdpa_fallback"] == 0
                    assert CALLS["decode_custom"] == 12
                del output, cached, logits
                prefill[name] = measure(lambda: model(ids, use_cache=True, logits_to_keep=1),
                                        args.warmup, args.repeats * 3)

        def request(text, implementation):
            prompt = input_ids(tokenizer, text, length, max_length=1024)
            tokens = greedy_generate(model, prompt, args.new_tokens, implementation=implementation)
            return tokens, tokenizer.decode(tokens[0], skip_special_tokens=True)

        for _ in range(args.warmup):
            for implementation in routes.values():
                request(PROMPTS[0], implementation)
        samples = {name: [] for name in routes}
        mismatches = []
        dispatch = None
        for index in range(args.repeats * 3):
            outputs = {}
            order = list(routes.items())
            if index % 2:
                order.reverse()
            for name, implementation in order:
                counters.update({key: 0 for key in counters})
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs[name] = request(PROMPTS[index % 3], implementation)
                torch.cuda.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000)
                if implementation == custom_route:
                    dispatch = counters.copy()
                    if args.tensor_core:
                        assert dispatch == {"prefill": 12, "native": 12 * (args.new_tokens - 1)}, dispatch
                    else:
                        assert dispatch["decode_custom"] == 12 * (args.new_tokens - 1)
            native, custom = outputs.values()
            if not torch.equal(native[0], custom[0]) or native[1] != custom[1]:
                mismatches.append({"sample": index, "prompt": index % 3,
                                   "native_output_ids": outputs["native_sdpa"][0][0, length:].tolist(),
                                   "custom_output_ids": outputs["custom_attention_adapter"][0][0, length:].tolist()})
        shapes.append({"prompt_tokens": length, "correctness": checks,
                       "prefill": prefill, "generation_dispatch": dispatch,
                       "generation_token_ids_and_text_equal": not mismatches,
                       "generation_mismatches": mismatches,
                       "generation": {name: {"median_ms": statistics.median(values),
                            "output_tokens_per_s": 1000 * args.new_tokens / statistics.median(values),
                            "samples_ms": values} for name, values in samples.items()}})
        print(json.dumps(shapes[-1]), flush=True)
    root = Path(__file__).resolve().parent
    sources = ("benchmark_context.py", "model_demo.py", "attention.py", "gelu_new.py",
               "csrc/attention.cu", "csrc/attention_tensor_core.cu", "csrc/attention_bindings.cpp", "csrc/gelu.cu")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "model_id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": MODEL_WEIGHTS_SHA256,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "transformers": __import__("transformers").__version__,
              "gpu": torch.cuda.get_device_name(), "dtype": "float16" if args.tensor_core else "float32",
              "tf32_matmul": False,
              "logit_check_tolerance": {"rtol": 0.02, "atol": 0.15} if args.tensor_core else
                                       {"rtol": 0.001, "atol": 0.005},
              "new_tokens": args.new_tokens, "warmup": args.warmup, "repeats_per_prompt": args.repeats,
              "prompts": list(PROMPTS[:3]),
              "prompt_construction": "repeat/truncate three prompt token sequences to exact lengths; synthetic context scaling, not natural long documents",
              "generation_timing": "paired synchronized text-to-text wall time; includes tokenizer, input transfer, repeated/truncated prompt construction, prefill, fixed-count decode, detokenization; no graphs; load/compile/warmup excluded",
              "prefill_timing": "resident input IDs; full model prefill with last-position logits and fresh dynamic cache; sequential route groups; wall and CUDA events include host launch gaps",
              "route": "CUDA GELU on both sides; " +
                  ("FP16 Tensor Core prefill, native cached decode" if args.tensor_core else
                   "FP32 custom tiled prefill and cached decode") +
                  "; dispatch assertions reject silent prefill fallback",
              "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
              "shapes": shapes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
