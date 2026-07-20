"""Benchmark fixed-length GPT-2 generation with a captured CUDA decode loop."""

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
from transformers import StaticCache

from model_demo import (ATTENTION_NAME, MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256,
                        generate, input_ids, load_model)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The future of GPU computing is")
    parser.add_argument("--length", type=int, default=129)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, default=Path("results/graph-generation.json"))
    args = parser.parse_args()
    if (not 1 <= args.length <= 256 or not 2 <= args.new_tokens <= 32
            or args.length + args.new_tokens > 1024 or args.warmup < 1 or args.repeats < 1):
        parser.error("length must be 1..256, new-tokens 2..32, total <=1024, and warmup/repeats positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch build is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer, model = load_model()
    prompt_texts = (
        args.prompt,
        "Explain how a GPU warp executes an instruction.",
        "Write a short story about a lighthouse keeper.",
        "What causes the seasons on Earth?",
        "Summarize the benefits of unit tests.",
        "Translate good morning into French.",
        "How does a bicycle gear change speed?",
        "Describe a recipe for vegetable soup.",
        "Why does a rainbow have several colors?",
    )
    prompt_ids = [input_ids(tokenizer, prompt, args.length) for prompt in prompt_texts]
    ids = prompt_ids[0].clone()
    if len({tuple(prompt.flatten().tolist()) for prompt in prompt_ids}) != len(prompt_ids):
        raise RuntimeError("benchmark prompts must have distinct token IDs")
    model.set_attn_implementation("sdpa")
    positions = torch.arange(ids.shape[1], device="cuda")
    cache = StaticCache(config=model.config, max_cache_len=ids.shape[1] + args.new_tokens)
    token_buffer = torch.empty((1, 1), dtype=torch.long, device="cuda")
    position_buffer = torch.tensor([ids.shape[1]], device="cuda")

    def graph_prefill():
        model.set_attn_implementation("sdpa")
        cache.reset()
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            prompt = model(ids, past_key_values=cache, cache_position=positions,
                           use_cache=True, logits_to_keep=1)
        first = prompt.logits[:, -1].argmax(dim=-1, keepdim=True)
        token_buffer.copy_(first)
        position_buffer.fill_(ids.shape[1])
        return first

    graph_prefill()
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        for _ in range(3):
            model(token_buffer, past_key_values=cache, cache_position=position_buffer,
                  use_cache=True, logits_to_keep=1)
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph_prefill()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    graph_tokens = []
    capture_start = time.perf_counter()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION), torch.cuda.graph(graph):
        for _ in range(args.new_tokens - 1):
            result = model(token_buffer, past_key_values=cache, cache_position=position_buffer,
                           use_cache=True, logits_to_keep=1)
            next_token = result.logits[:, -1].argmax(dim=-1, keepdim=True)
            graph_tokens.append(next_token)
            token_buffer.copy_(next_token)
            position_buffer.add_(1)
    torch.cuda.synchronize()
    capture_ms = (time.perf_counter() - capture_start) * 1000

    def graph_generate():
        first = graph_prefill()
        graph.replay()
        return torch.cat([ids, first] + graph_tokens, dim=1)

    expected_by_prompt = []
    for prompt in prompt_ids:
        ids.copy_(prompt)
        expected, _ = generate(model, tokenizer, ids, args.new_tokens, "sdpa")
        if expected.shape[1] != ids.shape[1] + args.new_tokens:
            raise RuntimeError("fixed-length graph requires generation without early EOS")
        expected_by_prompt.append(expected)
    distinct_continuations = len({tuple(output[0, -args.new_tokens:].tolist())
                                  for output in expected_by_prompt})
    if distinct_continuations < 2:
        raise RuntimeError("benchmark prompts did not produce varied continuations")
    ids.copy_(prompt_ids[0])
    expected = expected_by_prompt[0]
    custom, custom_calls = generate(model, tokenizer, ids, args.new_tokens, ATTENTION_NAME)
    actual = graph_generate()
    torch.cuda.synchronize()
    if not torch.equal(actual, expected) or not torch.equal(custom, expected):
        raise AssertionError("graph, custom, and unchanged GPT-2 generated different token IDs")
    if (custom_calls["query_tiled"] != (12 if ids.shape[1] >= 64 else 0)
            or custom_calls["single_query_tiled"] != (12 if ids.shape[1] < 64 else 0)
            or custom_calls["decode_custom"] != 12 * (args.new_tokens - 1)
            or custom_calls["sdpa_fallback"]):
        raise AssertionError(f"custom path unexpectedly fell back: {custom_calls}")

    runs = (("sdpa_dynamic", lambda: generate(model, tokenizer, ids, args.new_tokens, "sdpa")[0]),
            ("custom_dynamic", lambda: generate(model, tokenizer, ids, args.new_tokens, ATTENTION_NAME)[0]),
            ("sdpa_static_graph", graph_generate))
    for index in range(args.warmup):
        ids.copy_(prompt_ids[index % len(prompt_ids)])
        for _, run in runs:
            run()
    samples = {name: [] for name, _ in runs}
    peaks = {name: [] for name, _ in runs}
    for index in range(args.repeats):
        prompt_index = index % len(prompt_ids)
        ids.copy_(prompt_ids[prompt_index])
        order = runs[index % len(runs):] + runs[:index % len(runs)]
        for name, run in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            output = run()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
            peaks[name].append(torch.cuda.max_memory_allocated())
            if not torch.equal(output, expected_by_prompt[prompt_index]):
                raise AssertionError(f"{name} produced wrong token IDs for prompt {prompt_index}")
            del output

    root = Path(__file__).resolve().parent
    sources = ("graph_generation.py", "model_demo.py", "attention.py",
               "csrc/attention_bindings.cpp", "csrc/attention.cu")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "os": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "weights_sha256": MODEL_WEIGHTS_SHA256, "dtype": "float32",
        "prompt_tokens": ids.shape[1], "new_tokens": args.new_tokens,
        "distinct_prompts": len(prompt_ids),
        "distinct_continuations": distinct_continuations,
        "measured_prompt_indices": [index % len(prompt_ids) for index in range(args.repeats)],
        "warmup": args.warmup, "repeats": args.repeats, "tf32_matmul": False,
        "capture_ms": capture_ms,
        "cold_graph_ms_estimate": capture_ms + statistics.median(samples["sdpa_static_graph"]),
        "amortized_graph_ms_estimate": (capture_ms + sum(samples["sdpa_static_graph"])) / args.repeats,
        "graph_attention": "PyTorch EFFICIENT_ATTENTION with StaticCache causal mask",
        "custom_attention": "custom prompt and dynamic-cache decode kernels",
        "timing": "paired synchronized wall time, rotated order, varying token IDs at fixed shape; prefill plus all generated tokens; capture excluded from samples, included in estimates; model load and warmup excluded",
        "token_ids_equal": True, "custom_calls": custom_calls,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "results": {name: {"median_ms": statistics.median(values), "samples_ms": values,
                           "peak_allocated_bytes": max(peaks[name])} for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
