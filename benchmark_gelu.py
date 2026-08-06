"""Measure fused GELU or residual-LayerNorm in GPT-2 greedy generation."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from benchmark_greedy import PROMPTS
from fused_ln import set_gpt2_fused_ln
from gelu_new import set_gpt2_cuda_gelu
from model_demo import (MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256,
                        greedy_generate, input_ids, load_model)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3, help="passes over all prompts")
    parser.add_argument("--prompts-file", type=Path, help="one unpadded prompt per line")
    parser.add_argument("--fused-ln", action="store_true", help="compare with/without fused attention residual and ln_2, keeping CUDA GELU on both paths")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("results/fused-ln-generation.json" if args.fused_ln
                           else "results/gelu-generation.json")
    if args.new_tokens < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("new-tokens, warmup, and repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, model = load_model()
    prompt_texts = ([line.strip() for line in args.prompts_file.read_text(encoding="utf-8").splitlines()
                     if line.strip()] if args.prompts_file else PROMPTS)
    if not prompt_texts:
        parser.error("prompts file must contain at least one nonempty line")
    prompt_lengths = [input_ids(tokenizer, text, None).shape[1] for text in prompt_texts]
    if max(prompt_lengths) + args.new_tokens > model.config.n_positions:
        parser.error("prompt plus generated tokens exceeds GPT-2's position limit")

    # Check intermediate tensors before timing complete generation.
    ids = input_ids(tokenizer, prompt_texts[0], 129)
    model.set_attn_implementation("sdpa")
    if args.fused_ln:
        set_gpt2_cuda_gelu(model, True)
        set_route = lambda enabled: set_gpt2_fused_ln(model, enabled)
        routes = (("pytorch_ln", False), ("cuda_fused_ln", True))
    else:
        set_route = lambda enabled: set_gpt2_cuda_gelu(model, enabled)
        routes = (("pytorch_gelu", False), ("cuda_gelu", True))
    baseline_name, custom_name = routes[0][0], routes[1][0]
    set_route(False)
    baseline = model(ids, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    set_route(True)
    custom = model(ids, use_cache=True, output_hidden_states=True, logits_to_keep=1)
    hidden_errors = [(a - b).abs().max().item()
                     for a, b in zip(custom.hidden_states, baseline.hidden_states)]
    for actual, expected in zip(custom.hidden_states, baseline.hidden_states):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(custom.logits, baseline.logits, rtol=1e-3, atol=1e-3)
    logit_error = (custom.logits - baseline.logits).abs().max().item()
    token = baseline.logits[:, -1].argmax(dim=-1, keepdim=True)
    set_route(False)
    baseline_next = model(token, past_key_values=baseline.past_key_values,
                          use_cache=True, logits_to_keep=1)
    set_route(True)
    custom_next = model(token, past_key_values=custom.past_key_values,
                        use_cache=True, logits_to_keep=1)
    torch.testing.assert_close(custom_next.logits, baseline_next.logits, rtol=1e-3, atol=1e-3)
    cached_logit_error = (custom_next.logits - baseline_next.logits).abs().max().item()

    def generate_text(text):
        prompt = input_ids(tokenizer, text, None)
        output = greedy_generate(model, prompt, args.new_tokens)
        return output, tokenizer.decode(output[0], skip_special_tokens=True)

    for index in range(args.warmup):
        for _, enabled in routes:
            set_route(enabled)
            generate_text(prompt_texts[index % len(prompt_texts)])

    samples = {name: [] for name, _ in routes}
    for index in range(args.repeats * len(prompt_texts)):
        text = prompt_texts[index % len(prompt_texts)]
        outputs = {}
        order = routes if index % 2 == 0 else tuple(reversed(routes))
        for name, enabled in order:
            set_route(enabled)
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs[name] = generate_text(text)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
        if (not torch.equal(outputs[baseline_name][0], outputs[custom_name][0])
                or outputs[baseline_name][1] != outputs[custom_name][1]):
            raise AssertionError(f"generated output differs for prompt {index % len(prompt_texts)}")

    root = Path(__file__).resolve().parent
    sources = ("benchmark_gelu.py", "gelu_new.py", "csrc/gelu_bindings.cpp",
               "csrc/gelu.cu", "model_demo.py")
    if args.fused_ln:
        sources += ("fused_ln.py", "csrc/fused_ln_bindings.cpp", "csrc/fused_ln.cu")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "weights_sha256": MODEL_WEIGHTS_SHA256,
        "dtype": "float32", "tf32_matmul": False,
        "prompt_lengths": prompt_lengths,
        "fused_ln_comparison": args.fused_ln,
        "new_tokens": args.new_tokens, "warmup": args.warmup,
        "repeats_per_prompt": args.repeats,
        "timing": "paired synchronized text-to-text wall time, alternating order; includes tokenization, transfer, generation, decoding; module swap, model load, warmup excluded",
        "throughput": "end-to-end output tokens/s = new_tokens / complete request seconds; not steady-state decode throughput",
        "intermediate_max_hidden_abs_error": max(hidden_errors),
        "intermediate_max_logit_abs_error": logit_error,
        "cached_max_logit_abs_error": cached_logit_error,
        "token_ids_and_text_equal": True,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sources},
        "results": {name: {"median_ms": statistics.median(values),
                           "median_output_tokens_per_s": statistics.median(
                               1000 * args.new_tokens / ms for ms in values),
                           "samples_ms": values}
                    for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"median_ms": {name: values["median_ms"]
                                    for name, values in report["results"].items()},
                      "median_output_tokens_per_s":
                      {name: values["median_output_tokens_per_s"]
                       for name, values in report["results"].items()},
                      "intermediate_max_logit_abs_error": logit_error,
                      "token_ids_and_text_equal": True,
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
