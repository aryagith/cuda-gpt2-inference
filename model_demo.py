"""Run pinned GPT-2 with custom prompt attention and native cached decoding."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, AttentionInterface, AttentionMaskInterface, StaticCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import sdpa_mask

from attention import cuda_attention_tiled, cuda_attention_query_tiled, cuda_attention_decode


MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "ca2fb2126851846203760056ed55b201030ee9d5"
MODEL_WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "gpt2"
ATTENTION_NAME = "cuda_tiled_prompt"
SINGLE_ATTENTION_NAME = "cuda_single_query_prompt"
CALLS = {"single_query_tiled": 0, "query_tiled": 0, "decode_custom": 0, "sdpa_fallback": 0}


def prompt_attention(module, query, key, value, attention_mask, force_single=False, **kwargs):
    causal = query.shape[-2] == 1 or kwargs.get("is_causal") is True
    eligible = (attention_mask is None and causal and query.is_cuda
                 and query.device == key.device == value.device
                 and query.dtype == key.dtype == value.dtype == torch.float32
                 and query.shape[:2] == key.shape[:2] == value.shape[:2]
                 and query.shape[-1] == key.shape[-1] == value.shape[-1]
                 and query.shape[-1] <= 128
                 and not module.training and not (query.requires_grad or key.requires_grad or value.requires_grad)
                 and kwargs.get("dropout", 0.0) == 0.0 and kwargs.get("head_mask") is None
                 and module.scale_attn_weights and not module.scale_attn_by_inverse_layer_idx
                 and not module.reorder_and_upcast_attn and not module.is_cross_attention)
    prefill = (eligible and query.shape == key.shape == value.shape
               and query.shape[-2] <= 256 and query.numel() <= 16 * 1024 * 1024
               and query.numel() // query.shape[-1] * query.shape[-2] <= 16 * 1024 * 1024)
    decode = (eligible and query.shape[-2] == 1 and key.shape == value.shape
              and 1 < key.shape[-2] <= 1024 and key.numel() <= 16 * 1024 * 1024)
    if prefill:
        use_query_tile = not force_single and query.shape[-2] >= 64
        CALLS["query_tiled" if use_query_tile else "single_query_tiled"] += 1
        kernel = cuda_attention_query_tiled if use_query_tile else cuda_attention_tiled
        output = kernel(query.contiguous(), key.contiguous(), value.contiguous())
        return output.transpose(1, 2).contiguous(), None
    if decode:
        CALLS["decode_custom"] += 1
        output = cuda_attention_decode(query.contiguous(), key.contiguous(), value.contiguous())
        return output.transpose(1, 2).contiguous(), None
    CALLS["sdpa_fallback"] += 1
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def single_prompt_attention(module, query, key, value, attention_mask, **kwargs):
    return prompt_attention(module, query, key, value, attention_mask, force_single=True, **kwargs)


def load_model():
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise RuntimeError("Download the pinned GPT-2 snapshot into models/gpt2; see README.md")
    with (MODEL_DIR / "model.safetensors").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != MODEL_WEIGHTS_SHA256:
            raise RuntimeError("GPT-2 weights differ from the pinned model revision")
    AttentionInterface.register(ATTENTION_NAME, prompt_attention)
    AttentionMaskInterface.register(ATTENTION_NAME, sdpa_mask)
    AttentionInterface.register(SINGLE_ATTENTION_NAME, single_prompt_attention)
    AttentionMaskInterface.register(SINGLE_ATTENTION_NAME, sdpa_mask)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=True, dtype=torch.float32,
        attn_implementation="sdpa").eval().to("cuda")
    config = model.config
    if (config.n_layer, config.n_head, config.n_embd, config.n_positions) != (12, 12, 768, 1024):
        raise RuntimeError("Downloaded model architecture differs from the pinned GPT-2 contract")
    if (config.add_cross_attention or not config.scale_attn_weights
            or config.scale_attn_by_inverse_layer_idx or config.reorder_and_upcast_attn):
        raise RuntimeError("Downloaded model attention configuration is unsupported")
    return tokenizer, model


def input_ids(tokenizer, prompt, length):
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
    if tokens.numel() == 0:
        raise ValueError("prompt must tokenize to at least one token")
    if length:
        tokens = tokens.repeat((length + tokens.numel() - 1) // tokens.numel())[:length]
    if not 1 <= tokens.numel() <= 256:
        raise ValueError("prompt must contain 1 to 256 tokens for custom attention")
    return tokens.unsqueeze(0).to("cuda")


@torch.inference_mode()
def check_outputs(model, ids):
    model.set_attn_implementation("sdpa")
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        baseline = model(ids, use_cache=True, output_hidden_states=True)
    with sdpa_kernel(SDPBackend.MATH):
        math = model(ids, use_cache=True, output_hidden_states=True)
    model.set_attn_implementation(ATTENTION_NAME)
    CALLS.update(single_query_tiled=0, query_tiled=0, decode_custom=0, sdpa_fallback=0)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        custom = model(ids, use_cache=True, output_hidden_states=True)
    expected_calls = {"single_query_tiled": 12 if ids.shape[1] < 64 else 0,
                      "query_tiled": 12 if ids.shape[1] >= 64 else 0,
                      "decode_custom": 0, "sdpa_fallback": 0}
    if CALLS != expected_calls:
        raise AssertionError(f"expected all 12 prompt layers to use tiled attention, got {CALLS}")
    layer_errors = [(a - b).abs().max().item() for a, b in zip(custom.hidden_states, baseline.hidden_states)]
    for actual, expected in zip(custom.hidden_states[:-1], baseline.hidden_states[:-1]):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-4)
    # The final layer norm amplifies FP32 attention-order differences. Math SDPA
    # versus efficient SDPA shows a similar final-state error on these inputs.
    torch.testing.assert_close(custom.hidden_states[-1], baseline.hidden_states[-1], rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(custom.logits, baseline.logits, rtol=1e-3, atol=5e-3)
    token = baseline.logits[:, -1].argmax(dim=-1, keepdim=True)
    model.set_attn_implementation("sdpa")
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        baseline_next = model(token, past_key_values=baseline.past_key_values, use_cache=True)
    model.set_attn_implementation(ATTENTION_NAME)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        custom_next = model(token, past_key_values=custom.past_key_values, use_cache=True)
    expected_calls["decode_custom"] = 12
    if CALLS != expected_calls:
        raise AssertionError(f"expected all 12 cached-token layers to use decode attention, got {CALLS}")
    torch.testing.assert_close(custom_next.logits, baseline_next.logits, rtol=1e-3, atol=5e-3)
    return {"max_hidden_abs_error": max(layer_errors),
            "max_logit_abs_error": (custom.logits - baseline.logits).abs().max().item(),
            "max_cached_logit_abs_error": (custom_next.logits - baseline_next.logits).abs().max().item(),
            "math_sdpa_max_final_hidden_abs_error":
                (math.hidden_states[-1] - baseline.hidden_states[-1]).abs().max().item(),
            "math_sdpa_max_logit_abs_error": (math.logits - baseline.logits).abs().max().item(),
            "single_query_layer_calls": CALLS["single_query_tiled"],
            "query_tiled_layer_calls": CALLS["query_tiled"],
            "decode_custom_calls": CALLS["decode_custom"],
            "decode_fallback_calls": CALLS["sdpa_fallback"]}


@torch.inference_mode()
def generate(model, tokenizer, ids, new_tokens, implementation):
    model.set_attn_implementation(implementation)
    CALLS.update(single_query_tiled=0, query_tiled=0, decode_custom=0, sdpa_fallback=0)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        generated = model.generate(ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    return generated, CALLS.copy()


@torch.inference_mode()
def greedy_generate(model, ids, new_tokens, *, static_cache=False):
    """Fixed-count greedy generation; optionally preallocate a fresh request cache."""
    if new_tokens < 1 or ids.shape[1] + new_tokens > model.config.n_positions:
        raise ValueError("new_tokens must be positive and fit GPT-2's position limit")
    model.set_attn_implementation("sdpa")
    cache_args = {}
    if static_cache:
        positions = torch.arange(ids.shape[1] + new_tokens, device=ids.device)
        cache_args = {"past_key_values": StaticCache(
            config=model.config, max_cache_len=ids.shape[1] + new_tokens),
            "cache_position": positions[:ids.shape[1]]}
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = model(ids, use_cache=True, logits_to_keep=1, **cache_args)
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = [token]
        for index in range(new_tokens - 1):
            position_args = ({"cache_position": positions[ids.shape[1] + index:ids.shape[1] + index + 1]}
                             if static_cache else {})
            output = model(token, past_key_values=cache, use_cache=True, logits_to_keep=1,
                           **position_args)
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
    return torch.cat([ids, *tokens], dim=1)


def measure(run, warmup, repeats):
    for _ in range(warmup):
        output = run()
        del output
    torch.cuda.synchronize()
    samples, events, peaks = [], [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        start = time.perf_counter()
        output = run()
        stop_event.record()
        stop_event.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        events.append(start_event.elapsed_time(stop_event))
        peaks.append(torch.cuda.max_memory_allocated())
        del output
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "median_cuda_event_ms": statistics.median(events), "cuda_event_samples_ms": events,
            "peak_allocated_bytes": max(peaks)}


@torch.inference_mode()
def benchmark_generation(model, tokenizer, ids, new_tokens, warmup, repeats):
    runs = (("baseline_sdpa", lambda: generate(model, tokenizer, ids, new_tokens, "sdpa")),
            ("custom", lambda: generate(model, tokenizer, ids, new_tokens, ATTENTION_NAME)))
    for _ in range(warmup):
        for _, run in runs:
            run()
    samples = {name: [] for name, _ in runs}
    peaks = {name: [] for name, _ in runs}
    for index in range(repeats):
        for name, run in (runs if index % 2 == 0 else reversed(runs)):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = run()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
            peaks[name].append(torch.cuda.max_memory_allocated())
            del output
    return {name: {"median_ms": statistics.median(values), "samples_ms": values,
                   "peak_allocated_bytes": max(peaks[name])} for name, values in samples.items()}


@torch.inference_mode()
def benchmark(model, ids, warmup, repeats, implementation):
    model.set_attn_implementation(implementation)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        prefill = measure(lambda: model(ids, use_cache=True), warmup, repeats)

        def decode_one():
            prompt_output = model(ids, use_cache=True)
            token = prompt_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = prompt_output.past_key_values
            torch.cuda.synchronize()
            return model(token, past_key_values=cache, use_cache=True)

        # A fresh cache is built for each sample, but only the token forward is
        # intended to be timed; measure it separately below.
        for _ in range(warmup):
            warm_output = decode_one()
            del warm_output
        samples, events, peaks = [], [], []
        for _ in range(repeats):
            prompt_output = model(ids, use_cache=True)
            token = prompt_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = prompt_output.past_key_values
            del prompt_output
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            start = time.perf_counter()
            output = model(token, past_key_values=cache, use_cache=True)
            stop_event.record()
            stop_event.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            events.append(start_event.elapsed_time(stop_event))
            peaks.append(torch.cuda.max_memory_allocated())
            del output, cache
    decode = {"median_ms": statistics.median(samples), "samples_ms": samples,
              "median_cuda_event_ms": statistics.median(events), "cuda_event_samples_ms": events,
              "peak_allocated_bytes": max(peaks)}
    return {"prefill": prefill, "cached_token": decode}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The future of GPU computing is")
    parser.add_argument("--lengths", default="17,129,256", help="comma-separated prefill token lengths")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--fast", action="store_true",
                        help="generate once with direct greedy decoding, CUDA GELU, dynamic cache, and PyTorch SDPA")
    parser.add_argument("--generation-length", type=int,
                        help="repeat/truncate prompt tokens to this length for generation (1..256)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if (any(not 1 <= length <= 256 for length in lengths)
            or (args.generation_length is not None and not 1 <= args.generation_length <= 256)
            or args.new_tokens < 1 or args.warmup < 1 or args.repeats < 1):
        parser.error("lengths/generation-length must be 1..256, and new-tokens/warmup/repeats positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, model = load_model()
    loaded_model_allocated = torch.cuda.memory_allocated()
    prompt_ids = input_ids(tokenizer, args.prompt, args.generation_length)
    if prompt_ids.shape[1] + args.new_tokens > model.config.n_positions:
        parser.error("prompt plus generated tokens exceeds GPT-2's 1024 positions")
    if args.fast:
        from gelu_new import set_gpt2_cuda_gelu
        set_gpt2_cuda_gelu(model, True)
        generated = greedy_generate(model, prompt_ids, args.new_tokens)
        print(json.dumps({"route": "direct greedy, CUDA GELU, dynamic cache, efficient SDPA",
                          "prompt_tokens": prompt_ids.shape[1], "new_tokens": args.new_tokens,
                          "text": tokenizer.decode(generated[0], skip_special_tokens=True)}, indent=2))
        return
    generated_baseline, _ = generate(model, tokenizer, prompt_ids, args.new_tokens, "sdpa")
    generated_custom, generation_calls = generate(model, tokenizer, prompt_ids, args.new_tokens, ATTENTION_NAME)
    if generation_calls["single_query_tiled"] + generation_calls["query_tiled"] != 12:
        raise AssertionError(f"generation prefill did not use all 12 tiled layers: {generation_calls}")
    if generation_calls["query_tiled"] != (12 if prompt_ids.shape[1] >= 64 else 0):
        raise AssertionError(f"generation used the wrong prefill kernel: {generation_calls}")
    expected_decode = 12 * (generated_custom.shape[1] - prompt_ids.shape[1] - 1)
    if generation_calls["decode_custom"] != expected_decode or generation_calls["sdpa_fallback"]:
        raise AssertionError(f"generation did not use custom decode in all layers: {generation_calls}")
    generation_benchmark = benchmark_generation(model, tokenizer, prompt_ids, args.new_tokens,
                                                args.warmup, args.repeats)

    shapes = []
    for length in lengths:
        ids = input_ids(tokenizer, args.prompt, length)
        correctness = check_outputs(model, ids)
        baseline = benchmark(model, ids, args.warmup, args.repeats, "sdpa")
        single = benchmark(model, ids, args.warmup, args.repeats, SINGLE_ATTENTION_NAME)
        custom = benchmark(model, ids, args.warmup, args.repeats, ATTENTION_NAME)
        shapes.append({"sequence": length, "correctness": correctness,
                       "baseline_sdpa": baseline, "single_query_prompt": single,
                       "custom_prompt": custom})

    root = Path(__file__).resolve().parent
    sources = [root / path for path in ("model_demo.py", "attention.py", "csrc/attention_bindings.cpp",
                                       "csrc/attention.cu")]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "os": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256,
        "parameters": model.num_parameters(), "dtype": "float32", "heads": 12, "head_dim": 64,
        "layers": 12, "position_limit": 1024, "custom_prompt_limit": 256,
        "loaded_model_allocated_bytes": loaded_model_allocated,
        "baseline_attention": "Transformers SDPA, PyTorch EFFICIENT_ATTENTION forced",
        "custom_prefill_dispatch": "single-query tiled below 64 tokens; four-query tiled at 64..256",
        "decode_attention": "custom float32 cache-attention for unmasked single-token decode; SDPA fallback otherwise",
        "timing": "synchronized wall and CUDA-event time, resident token IDs; cached-token samples exclude cache prefill; CUDA events can include host launch gaps",
        "generation_timing": "paired synchronized wall time, alternating path order; includes prefill and all generated tokens",
        "memory": "peak torch.cuda.memory_allocated including model weights",
        "warmup": args.warmup, "repeats": args.repeats, "tf32_matmul": False,
        "prompt": args.prompt, "generation_prompt_tokens": prompt_ids.shape[1], "new_tokens": args.new_tokens,
        "baseline_text": tokenizer.decode(generated_baseline[0], skip_special_tokens=True),
        "custom_text": tokenizer.decode(generated_custom[0], skip_special_tokens=True),
        "greedy_token_ids_equal": torch.equal(generated_baseline, generated_custom),
        "generation_calls": generation_calls,
        "generation_benchmark": generation_benchmark,
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
