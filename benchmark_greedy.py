"""Compare ordinary GPT-2 generation with a direct greedy KV-cache loop."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from model_demo import (MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS_SHA256,
                        generate, greedy_generate, input_ids, load_model)


PROMPTS = (
    "The future of GPU computing is",
    "Explain how a GPU warp executes an instruction.",
    "Write a short story about a lighthouse keeper.",
    "What causes the seasons on Earth?",
    "Summarize the benefits of unit tests.",
    "Translate good morning into French.",
    "How does a bicycle gear change speed?",
    "Describe a recipe for vegetable soup.",
    "Why does a rainbow have several colors?",
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3, help="passes over all prompts")
    parser.add_argument("--prompts-file", type=Path, help="one unpadded prompt per line")
    parser.add_argument("--output", type=Path, default=Path("results/greedy-generation.json"))
    args = parser.parse_args()
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
    prompts = [input_ids(tokenizer, prompt, None) for prompt in prompt_texts]
    if max(ids.shape[1] for ids in prompts) + args.new_tokens > model.config.n_positions:
        parser.error("prompt plus generated tokens exceeds GPT-2's position limit")

    def ordinary(text):
        ids = input_ids(tokenizer, text, None)
        output, _ = generate(model, tokenizer, ids, args.new_tokens, "sdpa")
        return output, tokenizer.decode(output[0], skip_special_tokens=True)

    def direct(text):
        ids = input_ids(tokenizer, text, None)
        output = greedy_generate(model, ids, args.new_tokens)
        return output, tokenizer.decode(output[0], skip_special_tokens=True)

    runs = (
        ("transformers_generate", ordinary),
        ("direct_greedy", direct),
    )
    for index in range(args.warmup):
        for _, run in runs:
            run(prompt_texts[index % len(prompts)])

    samples = {name: [] for name, _ in runs}
    first_output = None
    for index in range(args.repeats * len(prompts)):
        text = prompt_texts[index % len(prompts)]
        outputs = {}
        order = runs if index % 2 == 0 else tuple(reversed(runs))
        for name, run in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs[name] = run(text)
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - start) * 1000)
        if (not torch.equal(outputs["transformers_generate"][0], outputs["direct_greedy"][0])
                or outputs["transformers_generate"][1] != outputs["direct_greedy"][1]):
            raise AssertionError(f"generated token IDs differ for prompt {index % len(prompts)}")
        if first_output is None:
            first_output = outputs["direct_greedy"][1]

    root = Path(__file__).resolve().parent
    sources = ("benchmark_greedy.py", "model_demo.py")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "weights_sha256": MODEL_WEIGHTS_SHA256,
        "dtype": "float32", "tf32_matmul": False, "new_tokens": args.new_tokens,
        "prompt_count": len(prompts),
        "prompt_lengths": [ids.shape[1] for ids in prompts],
        "warmup": args.warmup, "repeats_per_prompt": args.repeats,
        "timing": "paired synchronized text-to-text wall time, alternating order; includes tokenization, GPU input transfer, generation, and decoding; model load and warmup excluded; no graph capture",
        "token_ids_equal": True, "first_output": first_output,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "results": {name: {"median_ms": statistics.median(values), "samples_ms": values}
                    for name, values in samples.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prompt_lengths": report["prompt_lengths"],
                      "token_ids_equal": True,
                      "median_ms": {name: result["median_ms"]
                                    for name, result in report["results"].items()},
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
