# CUDA Attention Engine

A focused C++/CUDA learning and performance project with verified float32
RMSNorm, causal prompt attention, cached-token attention, and fused GPT-2 GELU
on an RTX 4060 Laptop GPU (8 GB). The custom attention paths run inside pinned
GPT-2; the
unchanged model is the correctness and latency baseline. The local,
Git-ignored `PROJECT_PLAN.md` holds the detailed handoff.

## First operation

For each row, compute `y = x * rsqrt(mean(x*x) + eps) * weight`.

- Inputs: contiguous float32 `x[rows, width]` and `weight[width]` on one device.
- Forward only; no autograd, mixed precision, or arbitrary strides yet.
- Positive dimensions; epsilon must be finite and within the float32 normal range.
- The kernel assigns one 256-thread block per row and reduces squared values
  with warp shuffles and a small shared-memory handoff.
- It uses PyTorch's current CUDA stream and guards the input's device.

## Files

| File | Purpose |
| --- | --- |
| `rmsnorm.py` | Reference, input contract, lazy extension loader |
| `csrc/bindings.cpp` | Python binding and native input validation |
| `csrc/rmsnorm.cu` | First RMSNorm kernel |
| `test_rmsnorm.py` | Reference, CUDA, and stream correctness checks |
| `benchmark.py` | Warmed repeated measurements and JSON metadata |
| `attention.py` | Causal-attention oracle, contract, extension loader |
| `csrc/attention_bindings.cpp` | Independent native attention validation |
| `csrc/attention.cu` | Three-kernel baseline, single-query and four-query prompt tiles, cache decode |
| `test_attention.py` | CPU oracle, CUDA, masking, and stream checks |
| `benchmark_attention.py` | Attention correctness, graph latency, and peak allocation |
| `benchmark_decode.py` | One-query cache-attention correctness and graph latency |
| `model_demo.py` | Pinned GPT-2, prompt/decode adapter, checks, generation, model benchmark |
| `benchmark_greedy.py` | Varied-prompt end-to-end greedy generation comparison |
| `gelu_new.py`, `csrc/gelu*` | Fused CUDA GPT-2 GELU activation and native validation |
| `test_gelu.py`, `benchmark_gelu.py` | Activation correctness and paired model benchmark |
| `graph_generation.py` | Fixed-length CUDA Graph generation and paired model benchmark |

## Engineering report (local measurements, 2026-09-29)

**Machine.** RTX 4060 Laptop GPU (8 GB, compute capability 8.9), NVIDIA
driver 616.56, Windows build 26200, Python 3.11.0, PyTorch 2.13.0+cu130,
Transformers 4.57.6, CUDA Toolkit 13.0.2 (`nvcc` 13.0.88), Ninja 1.13.2,
and Visual Studio Build Tools 2022 17.14.41 (MSVC 19.44.35229). The installed
compiler built and passed these checks, though NVIDIA's CUDA 13.0 table lists
MSVC 193x rather than this MSVC 1944. Source was checked with SHA-256 hashes
in the local JSON reports. Laptop clocks and thermals were not controlled.

**What runs where.** A three-kernel causal baseline exposes the score matrix.
Prompt kernels use online softmax over 32-key tiles and allocate only output;
the four-query version shares a K/V tile across four warps. A separate kernel
handles one new query against cached K/V, which includes the current token.
`model_demo.py` sends supported GPT-2 prefill below 64 tokens to the
single-query tile, 64–256 tokens to the four-query tile, and unmasked one-token
decode through cache length 1024 to the decode kernel. Other calls use
Transformers SDPA. The custom operators are float32, contiguous, forward-only,
and do not cover padding masks, grouped-query attention, dropout, or training.

**Verification.** CPU reference runs passed 3 RMSNorm tests and 4 attention
tests; they explicitly skipped 4 and 6 CUDA tests, respectively. Separate GPU
runs passed all 7 RMSNorm tests and all 10 attention tests, including direct
binding rejection, causal masking, numerical stress, cache lengths 1–1024, and
non-default streams. Attention memcheck reported zero errors; targeted decode
synccheck reported zero errors and racecheck zero hazards. Pinned GPT-2 matched
the unchanged model within the documented tolerances for prompt hidden states,
prompt logits, and next-token logits at lengths 17/129/256. For a 129-token
prompt and eight generated tokens, greedy IDs matched and counters recorded
12 four-query prompt calls, 84 custom decode calls, and zero SDPA fallbacks.

**Selected performance evidence.** Medians below are local observations, not
portable speed guarantees. Kernel values use warmed CUDA graph replay events
with seven samples. The varied-prompt direct and fused-GELU rows each use 27
synchronized text-to-text wall-time samples after three warmups; the other
generation rows have their methods detailed below.

| Workload | Measured path | Comparison | Interpretation |
| --- | ---: | ---: | --- |
| RMSNorm 32×1024 | 2.47 µs | native PyTorch 2.75 µs | Small measured win |
| RMSNorm 1024×1024 | 15.20 µs | native PyTorch 11.77 µs | Custom loses |
| Prompt `[4,8,17,64]` | four-query 19.6 µs | single-query 14.5 µs | Keep short-prompt path |
| Prompt `[4,8,256,64]` | four-query 856.6 µs | single-query 1403.6 µs; efficient SDPA 150.3 µs | 1.64× over old custom, still loses to SDPA |
| Decode `[1,12,1,64]` over 129 keys | 21.3 µs | efficient SDPA 33.0 µs | Kernel-only win |
| Decode `[1,12,1,64]` over 256 keys | 37.1 µs | efficient SDPA 45.7 µs | Kernel-only win |
| GPT-2 cached token after 129-token prefill | 8.18 ms | unchanged model 7.97 ms | No model-level speedup |
| GPT-2 129-token prompt + 8 tokens | 70.47 ms | unchanged model 71.97 ms | Samples overlap; no firm speedup claim |
| GPT-2 text-to-text, varied 6–10-token prompts + 8 tokens | direct greedy 74.04 ms | `model.generate` 86.71 ms | Same token IDs; 26/27 paired wins, no graph |
| GPT-2 direct greedy with fused CUDA GELU, same prompt set | 56.88 ms | direct greedy with PyTorch GELU 70.58 ms | Same token IDs; 24/27 paired wins |
| Nine distinct 129-token prompts + 8 tokens, static-cache graph | 29.75 ms | dynamic SDPA 72.38 ms; custom dynamic 74.23 ms | Replay only; fixed shape |

At prompt `S=256`, the naive attention path increased peak live CUDA
allocation by 10 MiB, while the single- and four-query paths each used 2 MiB.
The decode output-only increment was 3 KiB for `[1,12,1,64]`. Maximum decode
kernel error against the independent float32 oracle among measured cache
lengths was 5.37e-7. At GPT-2 length 256, maximum prompt-logit error was
0.003315 and next-token-logit error was 0.000092; float32 reduction order
explains much of the prompt difference. For the 129-token generation run, both
paths had a 515.57 MiB whole-model peak. At 1024 cached keys, efficient SDPA
samples varied enough that its median comparison is inconclusive.

**Profiling and decisions.** Nsight Compute identified reduction barriers and
small-grid underfill in the original RMSNorm path; the warp-shuffle revision
removed seven block barriers. A device profiler measured ten `[4,8,256,64]`
prompt kernel calls totaling 19.01 ms single-query versus 11.53 ms four-query.
These are kernel times, not whole-model times; no hardware counters were
collected for the four-query or decode changes. Sharing K/V tiles is visible
in the source, but a cache or occupancy explanation would require new counter
measurements. A full cached-token trace found about 258 GPU kernels per forward.
Across five forwards, measured kernel execution summed to 14.82 ms (2.96 ms
per token): cuBLAS matrix-vector work, including the vocabulary projection,
used about 11.08 ms, versus 2.00 ms for efficient SDPA. Ordinary model wall
time was around 8 ms per token, pointing to substantial host/launch gaps as
well as the already optimized matrix operations. CUDA Graph replay addresses
the launch gap; details and limits are below.

**Reproduce.** Use the CUDA setup below in an x64 Native Tools Command Prompt,
activate the environment, and download the pinned GPT-2 snapshot as shown in
the GPT-2 section. Then run:

```text
python test_rmsnorm.py
python test_attention.py
python test_rmsnorm.py --cuda
python test_attention.py --cuda
compute-sanitizer --tool memcheck --error-exitcode 99 python test_attention.py --cuda
compute-sanitizer --tool synccheck --kernel-name kns=decode_kernel --error-exitcode 99 python test_attention.py --cuda --decode-only
compute-sanitizer --tool racecheck --kernel-name kns=decode_kernel --error-exitcode 99 python test_attention.py --cuda --decode-only
python benchmark.py --device cuda --custom --rows 32 --width 1024 --warmup 5 --repeats 7 --iterations 5 --graph-ops 32 --output results/rmsnorm-local.json
python benchmark_attention.py --sequences 17,33,64,129,256 --output results/attention-local.json
python benchmark_decode.py --output results/decode-local.json
python model_demo.py --generation-length 129 --warmup 5 --repeats 9 --output results/model-local.json
python benchmark_greedy.py --output results/greedy-local.json
python test_gelu.py --cuda
python benchmark_gelu.py --output results/gelu-local.json
python graph_generation.py --output results/graph-local.json
```

The benchmark JSON includes raw samples, backend selection, shape, precision,
and source hashes. `results/`, model weights, the project plan, and the private
study guide remain outside Git. The detailed sections below explain algorithms,
earlier measurements, and their limitations.

## CPU quick start

Python 3.11 is a suitable starting point. From this directory on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "torch>=2.7,<3" --index-url https://download.pytorch.org/whl/cpu
python test_rmsnorm.py
python test_attention.py
python benchmark.py --device cpu --rows 32 --width 257 --output results/cpu.json
```

On Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.
Default tests explicitly skip CUDA cases; a CPU pass does not validate CUDA.

## CUDA setup

1. Install a CUDA-enabled PyTorch build using the command from the
   [official installer selector](https://pytorch.org/get-started/locally/).
   If this environment already contains CPU-only PyTorch, replace it with the
   selected CUDA build. Confirm with the check below.
2. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   compatible with `torch.version.cuda`. A GPU driver and a PyTorch wheel alone
   do not provide the CUDA compiler used by this project.
3. Install a host C++ compiler supported by that toolkit. On Windows, use
   Visual Studio C++ Build Tools and an **x64 Native Tools Command Prompt**;
   on Linux, use a supported GCC/G++ version.
4. Activate the Python environment and run `python -m pip install -r requirements.txt`.

```text
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
python test_rmsnorm.py --cuda
python test_attention.py --cuda
python benchmark.py --device cuda --custom --output results/rmsnorm-cuda.json
python benchmark_attention.py --output results/attention-tiled.json
python benchmark_decode.py --output results/attention-decode.json
```

`--cuda` tests require successful compilation and execution; missing tools fail
the run. First use builds the extension through PyTorch and Ninja, outside the
timed region. No separate CMake project is needed. Set `CUDA_HOME` if toolkit
discovery fails. PyTorch selects the visible GPU architecture automatically;
`TORCH_CUDA_ARCH_LIST=8.9` can target this RTX 4060 explicitly. Set `MAX_JOBS=2`
if compilation uses too much host memory.

See [PyTorch extension setup](https://docs.pytorch.org/docs/stable/cpp_extension.html)
for compiler compatibility and architecture settings.

## Benchmark interpretation

Both `torch_eager` (the explicit formula) and `torch_native` (`F.rms_norm`) are
measured. `--custom` adds our kernel; all implementations run on the same device
and inputs. The native PyTorch path is the stronger baseline, though its
implementation depends on the installed PyTorch version. No claim is made that
our kernel beats it.

Reports contain correctness error, individual timing samples, median latency,
shape, seed, precision, runtime versions, GPU identity, and source hashes.
CUDA event measurements use already-resident inputs and include the operation's
execution on the stream; they are not end-to-end inference or host-to-device
transfer measurements. For device-side comparisons, pass `--graph-ops 32` to
capture 32 operations per CUDA graph replay and divide event time by 32. This
reduces Python launch gaps in small workloads. CPU measurements use wall time.
Benchmark outputs are ignored by Git; representative measurements are
summarized here.

## Milestones

1. **RMSNorm:** compiled on the actual GPU, passed correctness tests, recorded a baseline.
2. **Profiling:** inspected the kernel with Nsight Compute, replaced the shared
   tree reduction with warp shuffles, and compared latency across shapes.
3. **Attention:** the bounded causal baseline, tiled online-softmax path, and
   PyTorch oracle are verified; latency and allocation were measured.
4. **Model integration:** pinned GPT-2 runs with custom prompt and cached-token
   attention; intermediate outputs, logits, generation, latency, and memory
   were compared with the unchanged model.
5. **Engineering evidence:** the report above records reproduction commands,
   profiler observations, winning and losing workloads, and limitations.

Keep claims tied to reproducible results. Attribute algorithms and compare with
PyTorch's available optimized attention backend when the attention stage exists.

## Verified local RMSNorm evidence

On 2026-09-29, Windows build 26200 with driver 616.56, Python 3.11.0,
PyTorch 2.13.0+cu130, CUDA Toolkit compiler 13.0.88, Ninja 1.13.2, and
Visual Studio Build Tools 2022 17.14.41 (MSVC 19.44.35229) built and ran
the kernel on an RTX 4060 Laptop GPU (compute capability 8.9). The CPU suite
passed with four CUDA cases skipped; the separate CUDA suite passed all seven
tests. Compute Sanitizer memcheck and synccheck reported zero errors; racecheck
reported zero hazards, including after the reduction change. Nsight Compute
2025.3.1 measured 0.01 waves per multiprocessor at 1×1024 and 0.22 at
32×1024; at 1024×4096, DRAM throughput reached 85.7% of its reported peak.
The original reduction used nine block barriers; the warp-shuffle version uses
two. This targets small-row latency without adding a second kernel path.

CUDA graph event medians in microseconds (seven samples of five replays, with
32 operations per replay):

| Shape | Custom before | Custom after | Native after | Eager after |
| --- | ---: | ---: | ---: | ---: |
| 1×1024 | 2.41 | 2.34 | 2.30 | 7.41 |
| 32×257 | 2.27 | 1.96 | 4.19 | 8.32 |
| 32×1024 | 3.04 | 2.47 | 2.75 | 9.73 |
| 1024×1024 | 19.65 | 15.20 | 11.77 | 38.89 |
| 1024×4096 | 69.54 | 62.43 | 62.91 | 301.36 |

The custom output was checked against both eager and native PyTorch on every
listed shape; maximum absolute error versus eager was 1.91e-6. The custom
kernel still loses to native at 1024×1024. The difference at 1×1024 is small
enough to treat as timing noise. Original Nsight reports and before/after JSON
samples are in the ignored `results/` directory. Laptop power and thermal
conditions were not controlled, so these results are local observations, not
general performance claims. The GitHub Actions workflow still checks the CPU
reference only.

## Causal attention baseline

`attention.py` accepts matching contiguous float32 Q/K/V tensors shaped
`[batch, heads, sequence, head_dim]`, on one CPU or CUDA device, forward only.
The CUDA path uses a QK score kernel, an in-place stable masked softmax, and a
probability-times-V kernel. It allocates one `[batch, heads, sequence, sequence]`
float32 score buffer, so intermediate storage is quadratic in sequence length.
Sequence length is limited to 256, head dimension to 128, and each input and
the score buffer to 16 million float32 elements (64 MiB). This is a correctness
baseline for the 8 GB GPU, not a performance result. Cached decoding, grouped
query attention, dropout, and backward are outside this contract.

The independent PyTorch oracle uses matmul, a causal mask, softmax, and matmul;
tests also select PyTorch's **math SDPA backend** explicitly. GPU tests disable
TF32 matmul and cover sequence length 1, partial lengths 17/33/129, multiple
batches/heads, large finite logits, future-token exclusion, and a non-default
stream. High-logit tests compare with a float64 oracle and allow rtol 1e-3,
atol 3e-3 against float32 oracles because near-tied logits amplify FP32 dot
product ordering differences. Standard cases use rtol 2e-4, atol 2e-5.
On 2026-09-29, all seven CUDA tests passed; Compute Sanitizer memcheck and
synccheck reported zero errors, and racecheck reported zero hazards. The CPU
suite passed three tests with four CUDA tests explicitly skipped. These checks
do not compare against an optimized fused SDPA backend.

## Tiled causal attention

`cuda_attention_tiled` keeps one query per block and streams over 32-key tiles.
For each tile it computes scores only for valid causal positions, then updates
a running maximum `m`, softmax denominator `l`, and weighted-value numerator
`o`. If the new maximum is `m'`, the old state is multiplied by
`exp(m - m')` before adding the tile's `exp(score - m')` contributions.
The output is `o / l`. The first tile uses a zero rescale factor; every tile
has at least one valid key, so masked lanes never enter the reduction. This is
the [online normalizer](https://arxiv.org/abs/1805.02867) recurrence used by
[FlashAttention](https://arxiv.org/abs/2205.14135). This educational kernel
tiles keys and values but does not tile queries or reuse them across blocks.
It allocates only the output tensor, with no `[S,S]` score buffer.

On 2026-09-29, the RTX 4060 Laptop GPU with the toolchain listed above passed
all seven CUDA tests. The separate CPU run passed three reference tests and
skipped four CUDA tests. GPU cases include sequence 1, partial tiles 17/33/129,
the 256 boundary, causal perturbation, multiple batches/heads, large finite
logits, direct binding validation, and a non-default stream. Compute Sanitizer
memcheck and synccheck reported zero errors; racecheck reported zero hazards.
At sequence 256 and head dimension 128 with inputs scaled by 20, both custom
paths can differ from a float64 oracle by about 0.0047 from float32 dot-product
ordering. The high-logit acceptance tests therefore remain at head dimensions
up to 64; the 256×128 boundary is tested at ordinary scale.

Measured medians below are microseconds per operation for float32 `[4,8,S,64]`.
Each path had five warmups, seven samples of five CUDA graph replays, and eight
operations per replay. Peak allocation is the increase in PyTorch's live CUDA
allocation for one warmed call, including output. PyTorch SDPA was forced to
`EFFICIENT_ATTENTION` using
[`sdpa_kernel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
on every shape; the forced `FLASH_ATTENTION` path was unavailable in this build.

| S | Naive µs | Tiled µs | Efficient SDPA µs | Naive peak MiB | Tiled peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 25.0 | 14.5 | 17.6 | 0.168 | 0.133 |
| 33 | 67.8 | 40.8 | 20.1 | 0.391 | 0.258 |
| 129 | 465.0 | 419.5 | 67.7 | 3.039 | 1.008 |
| 256 | 1573.1 | 1521.5 | 152.6 | 10.000 | 2.000 |

At `S=256`, tiled saves 8 MiB of peak allocation versus naive. Its latency
advantage there is small relative to run-to-run variability, and it is about
10× slower than efficient SDPA. A block handles only one query and recomputes
QK dots rather than sharing tiles across queries, which limits throughput.
Maximum absolute error against the explicit PyTorch oracle across measured
shapes was 7.75e-7 for tiled and 2.80e-6 for efficient SDPA. These are local
measurements of resident inputs, not end-to-end inference. The laptop's power
and thermal conditions were uncontrolled; after the run it was 50 °C, 3.11 W,
and 210 MHz SM clock. The exact samples, source SHA-256 hashes, and backend
selection are in the ignored `results/attention-tiled.json`.

### Four-query tile improvement

`cuda_attention_query_tiled` keeps four queries in one block. The four warps
share a 32-key K/V tile in shared memory; K is transposed there so lanes read
different keys without a shared-memory bank conflict. Each query retains its
own running softmax maximum, denominator, and output accumulator. The
single-query path remains available for before/after checks and short prompts.
This change follows the work-partitioning motivation in
[FlashAttention-2](https://arxiv.org/abs/2307.08691); it remains an educational
float32 implementation without Tensor Core matrix multiplication.

On the RTX 4060 Laptop GPU, seven CUDA-graph samples of five replays with eight
operations per replay measured these median microseconds for `[4,8,S,64]`:

| S | Single-query tiled | Four-query tiled | Efficient SDPA |
| ---: | ---: | ---: | ---: |
| 17 | 14.6 | 19.6 | 17.5 |
| 33 | 41.0 | 42.5 | 20.0 |
| 64 | 130.2 | 104.2 | 20.5 |
| 129 | 371.2 | 255.7 | 69.0 |
| 256 | 1418.7 | 855.8 | 110.9 |

At S=256, sharing the tile reduced custom kernel latency by 1.66× while
retaining a 2 MiB peak allocation; the efficient SDPA backend is still much
faster. PyTorch's device profiler measured 10 kernel calls totaling 19.01 ms
single-query versus 11.53 ms four-query for `[4,8,256,64]`, and 7.33 versus
4.48 ms for GPT-2's `[1,12,256,64]` layout. These profiles do not include
Nsight hardware-counter attribution. All attention CUDA tests passed; Compute
Sanitizer memcheck/synccheck reported zero errors and racecheck zero hazards.
The raw comparison, crossover, GPT-2-shape, profiler, and sanitizer reports
are ignored under `results/attention-query-*.json` and `results/query-*.log`.

## GPT-2 prompt and cached-token integration

The demo uses [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
at revision `ca2fb2126851846203760056ed55b201030ee9d5` (MIT license).
The [pinned config](https://huggingface.co/openai-community/gpt2/blob/ca2fb2126851846203760056ed55b201030ee9d5/config.json)
has 12 layers, 12 heads, width 768, and head dimension 64, with a 1024-token
position limit. Its 124,439,808 parameters are loaded in float32, so mixed
precision is unnecessary for this 8 GB GPU. The model allocates 487.47 MiB
after loading; the largest measured prompt peak below is 564.97 MiB. The
downloaded `model.safetensors` must match SHA-256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.

From an x64 Native Tools Command Prompt with the project environment active:

```text
python -m pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai-community/gpt2', revision='ca2fb2126851846203760056ed55b201030ee9d5', allow_patterns=['*.json','*.txt','*.safetensors'], local_dir='models/gpt2')"
python model_demo.py --generation-length 129 --warmup 5 --repeats 9 --output results/model-decode.json
```

`models/` and `results/` are ignored by Git. The local-directory download works
on Windows without requiring symlink privileges. The adapter uses
[Transformers' attention interface](https://huggingface.co/docs/transformers/v4.57.6/attention_interface)
and retains its tokenizer, model layers, cache, and `generate()` method. It
routes unpadded float32 prompt prefill below 64 tokens through the single-query
kernel and 64–256 tokens through the four-query kernel. Supported one-token
decoding uses `cuda_attention_decode` over the updated K/V cache. Masks,
training, and unsupported shapes fall back to Transformers SDPA. PyTorch
`EFFICIENT_ATTENTION` is forced for the unchanged-model comparison.

Before the cache kernel was added, the RTX 4060 Laptop GPU passed prompt
comparisons at 17, 129, and 256 tokens. All 12 prompt layers used the custom
kernel, and the next cached-token forward used SDPA fallback in all 12 layers.
Maximum absolute
prompt logit error against the unchanged model was 0.003315 at 256 tokens;
math SDPA versus efficient SDPA differed by 0.003345 on the same inputs.
The maximum next-token logit error was 0.000092. For the default six-token
prompt and eight generated tokens, greedy token IDs matched in this run;
approximate logit agreement does not guarantee identical text on other inputs.

Before the four-query change, synchronized wall-time medians for device-resident
token IDs, nine samples after five warmups, were below. CUDA-event samples are
also in `results/model-gpt2.json`; they
can include host launch gaps. Peak figures are PyTorch live CUDA allocation
including weights, outputs, and cache. Cached-token timing excludes its cache
prefill. Measurements were sequential on a laptop with uncontrolled clocks and
thermals, so small differences at 17 and 129 tokens are inconclusive.

| Prompt tokens | SDPA prefill ms | Custom prefill ms | SDPA token ms | Custom token ms | Prefill peak MiB, both |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 16.66 | 20.09 | 18.01 | 17.34 | 500.73 |
| 129 | 17.12 | 16.23 | 15.65 | 17.68 | 530.40 |
| 256 | 16.57 | 19.39 | 12.59 | 14.32 | 564.97 |

With all three paths timed in one model run after the change, prefill medians
were:

| Prompt tokens | Efficient SDPA ms | Single-query ms | Selected custom ms |
| ---: | ---: | ---: | ---: |
| 17 | 9.48 | 8.84 | 9.29 (single-query) |
| 129 | 10.44 | 10.85 | 11.12 (four-query) |
| 256 | 12.74 | 18.26 | 15.90 (four-query) |

At 256 tokens, the new path improved full-model prefill versus the old custom
path, though it still loses to efficient SDPA and keeps the same 564.97 MiB
whole-model peak allocation. Small model-level differences vary between runs.
A 129-token generation check at that stage used the four-query path for all 12
prompt layers and SDPA for cached decoding. The new cache kernel is measured
separately in the engineering report above.
`results/model-query-tiled.json` includes raw samples, exact versions, backend,
source hashes, model revision, and weight hash. No weights or local reports are
tracked by Git.

## Cached-token CUDA attention

For one generated token, Q is `[B,H,1,D]` and the updated K/V cache is
`[B,H,T,D]`. Every cached position is past or current, so this operator uses
no causal mask. `cuda_attention_decode` streams over 32-key tiles, maintains
the same stable online-softmax state as prompt attention, and writes one output
per batch/head pair. Python and native bindings separately enforce matching
contiguous float32 tensors, forward-only execution, `T<=1024`, `D<=128`, and a
bounded 16-million-element K/V input. GPT-2's cache update happens in
Transformers before the registered attention callback runs.

Warmed CUDA graph event medians for `[1,12,1,64]` with K/V length `T`, five
warmups and seven samples of five replays with eight operations each:

| Cached keys T | Custom decode µs | Efficient SDPA µs |
| ---: | ---: | ---: |
| 1 | 3.5 | 11.4 |
| 17 | 4.7 | 11.3 |
| 33 | 7.8 | 13.3 |
| 129 | 21.3 | 33.0 |
| 256 | 37.1 | 45.7 |
| 512 | 72.7 | 88.9 |
| 1024 | 143.9 | 170.8 |

These are operation timings, not whole-token latency. SDPA's seven samples at
`T=1024` ranged from about 130 to 196 µs, including samples faster than our
kernel, so that median does not establish a stable win. Correctness against
the explicit oracle and forced efficient SDPA passed at the listed lengths;
direct native rejection, high finite logits against float64, and non-default
stream use also passed. The full attention suite passed 10 CUDA tests.
Memcheck found zero errors; targeted decode synccheck and racecheck found zero
errors and hazards. Raw data are in ignored `results/attention-decode.json`
and `results/decode-*.log`.

The paired whole-model generation measurement with a 129-token prompt and
eight new tokens gave 71.97 ms unchanged GPT-2 versus 70.47 ms custom, with
overlapping samples and equal 515.57 MiB peak live allocation. At a 129-token
cache length, isolated full-model token forwards were 7.97 ms unchanged versus
8.18 ms custom. This shows why a faster attention operation does not imply a
faster model. Greedy IDs matched on the fixed prompt, but that does not promise
identical text for every input. The custom route used 12 four-query prefill
calls and 84 decode calls with no fallback; unsupported model calls still use
Transformers SDPA. `results/model-decode-final.json` holds the exact samples,
settings, model/weight hashes, source hashes, and call counts.

## Varied-prompt generation without capture

`model_demo.greedy_generate` uses GPT-2's existing PyTorch SDPA and dynamic KV
cache, but runs the fixed-count greedy token loop directly. It requests only
the final position's logits on each forward. This path has no graph capture,
fixed prompt shape, or per-shape setup. It accepts unpadded prompt lengths up
to GPT-2's position limit. It always emits the requested number of tokens;
unlike `model.generate`, it does not stop early on EOS or implement sampling.

On the RTX 4060 Laptop GPU, nine distinct natural prompt texts of 6–10 tokens
each were generated three times for eight new tokens. Three warmups preceded
27 paired synchronized wall-time samples with alternating execution order.
The timer includes tokenization, GPU input transfer, generation, and output
decoding. All generated token IDs and text matched `model.generate`; the direct
route was faster in 26 of 27 pairs. Medians were **86.71 ms** for
`model.generate` and **74.04 ms** for the direct loop, a 14.6% latency
reduction. Samples ranged 69.82–123.32 ms and 58.76–92.12 ms, respectively.
A separate nine-prompt check with 16 new tokens also matched IDs and measured
141.51 versus 121.26 ms median; it is a smaller sample. Three longer natural
prompts of 47, 84, and 117 tokens, measured three times each for eight new
tokens, also matched IDs: medians were 72.25 ms ordinary versus 60.28 ms
direct, with nine of nine paired wins.
Model loading and warmup were excluded equally. No CUDA Graph capture is
required. This speedup is from a
simpler generation path, not from the custom attention kernels; the benchmark
does not isolate which removed framework steps account for the gain. An exploratory
FP16 check on nine varied prompts did not improve median latency, so FP16 is
not claimed as a win here. Raw samples and source hashes are in ignored
`results/greedy-generation.json` and `results/greedy-generation-16.json`.
The longer-prompt report is ignored `results/greedy-long-prompts.json`.
To compare your own queries, put one unpadded prompt per line in an ignored
file under `results/` and run
`python benchmark_greedy.py --prompts-file results/your-prompts.txt`.

## Fused GPT-2 GELU: a custom kernel with a model-level win

GPT-2's `gelu_new` applies several PyTorch elementwise operations after each
MLP `c_fc` projection. `csrc/gelu.cu` computes the same tanh-based expression
in one forward-only float32 kernel: one thread reads and writes one adjacent
element, keeping its intermediate values in registers. It uses no shared
memory or inter-thread synchronization. The model's cuBLAS projections remain
unchanged. `set_gpt2_cuda_gelu(model, True)` in `gelu_new.py` replaces the 12
GPT-2 MLP activation modules; passing `False` restores the PyTorch activation.

For a decode-shaped `[1,1,3072]` activation, nine paired wall-time samples of
100 calls each averaged **121.85 µs per PyTorch activation versus 13.99 µs**
per fused CUDA activation at the median. These are launch-inclusive per-call
measurements, not GPU-only kernel times. The more important check uses the
same `model_demo.greedy_generate` path on both sides. For nine naturally
different 6–10-token prompts, three passes each and eight generated tokens,
the complete text-to-text median was **70.58 ms** with PyTorch GELU and
**56.88 ms** with custom GELU, a 19.4% reduction. Custom won 24 of 27 paired
samples; all generated token IDs and text matched. Longer 47/84/117-token
prompts measured 53.17/42.02 ms (nine of nine paired wins), and a nine-prompt
16-new-token check measured 106.74/82.76 ms (eight of nine wins). These latter
two sets are smaller samples. All timing includes tokenization, GPU input
transfer, generation, and text decoding; model load, activation swapping,
extension compilation, and warmup are excluded equally. No graph is captured.

At S129, the largest intermediate hidden-state difference was 0.00213, while
the largest prompt-logit difference was 6.10e-5 and next-token logit difference
was 1.07e-4. The GPU test compares against Transformers' activation at decode
and prompt shapes and checks a non-default stream. CPU reference verification:
one test passed, two CUDA tests skipped. CUDA verification: all three tests
passed. Compute Sanitizer memcheck and synccheck reported zero errors.
The full-suite racecheck completed its tests but stalled during shutdown.
A focused racecheck of the GELU kernel later completed for 1-element,
decode-shaped `[1,1,3072]`, and prompt-shaped `[1,129,3072]` inputs:
**0 hazards, 0 errors, 0 warnings**. GELU uses no shared memory, which is
the memory racecheck examines. Raw samples and source hashes are in
ignored `results/gelu-generation.json`, `results/gelu-long-prompts.json`, and
`results/gelu-generation-16.json`; the ignored focused sanitizer log is
`results/gelu-racecheck-focused.log`.

A fresh same-source rerun on 2026-09-29 reproduced the improvement: the 27
short-prompt pairs measured **54.17/42.77 ms** PyTorch/custom medians (21.0%
lower; 26/27 paired wins), the nine long-prompt pairs **53.12/40.76 ms**
(9/9 wins), and the nine 16-token pairs **105.10/80.06 ms** (9/9 wins).
Every pair produced identical token IDs and text. The isolated activation
rerun measured 110.49/11.33 µs launch-inclusive medians. CPU reference: one
pass, two GPU skips; CUDA: three passes; fresh memcheck and synccheck: zero
errors. Absolute times varied between runs, so the paired comparisons are the
useful evidence. The ignored rerun reports are `results/gelu-*-verification.json`.

## Full-model profiling and CUDA Graph replay

Profiling five complete GPT-2 cached-token forwards after a 129-token prefill
showed **1,290 GPU kernel launches**, or 258 per token. CUDA kernels occupied
about 2.96 ms per forward in that trace; ordinary synchronized model forwards
were around 8 ms without profiler instrumentation. The profiler changes
absolute timing, so these figures identify work categories rather than provide
a precise host/GPU time split. cuBLAS matrix-vector operations accounted for
about 75% of measured GPU kernel time, including the final projection over
GPT-2's vocabulary. Efficient SDPA accounted for about 14%. Replacing the
already optimized cuBLAS operations with another hand-written kernel had no
measured justification. The many small launches were a better target.

`graph_generation.py` uses Transformers' `StaticCache` and PyTorch
`torch.cuda.CUDAGraph` to capture the fixed-count decode loop once. Prefill
still runs normally, then one graph replay executes all cached-token forwards,
greedy `argmax` operations, and device-side cache-position updates. The static
cache retains fixed addresses and masks unused future positions. This route
uses PyTorch `EFFICIENT_ATTENTION` because the static cache supplies a mask;
the custom decode kernel remains the separate dynamic-cache path.

For one RTX 4060 Laptop GPU run with nine **different prompt token sequences**
of the same 129-token length and eight new tokens each, five warmups and nine
paired, rotated-order wall-time samples measured:

| Generation route | Median ms | Observed sample range ms |
| --- | ---: | ---: |
| Dynamic cache, efficient SDPA | 72.38 | 66.69–85.26 |
| Dynamic cache, custom attention | 74.23 | 71.51–84.84 |
| Static cache, captured decode loop | 29.75 | 28.53–34.47 |

All three routes generated the same token IDs for every prompt; the nine
prompts also produced nine distinct continuations. One graph was
captured with the first prompt and replayed after loading each different prompt
into the same-shaped input buffer. Capture took 65.93 ms after model loading
and warmup. Adding it to median replay gives an estimated **95.68 ms first
graph request**, slower than the 72.38 ms ordinary median. Spread over these
nine requests, capture plus replay averaged an estimated 37.96 ms per request,
versus 73.64 ms mean for dynamic SDPA. These estimates exclude model loading,
warmup, and tokenization. Prompt texts are tokenized and repeated to the fixed
length by the existing `input_ids` helper, so this is a controlled same-shape
experiment, not a representative variable-length serving workload. The graph
fixes the batch, prompt length, and number of generated tokens; it does not
handle early EOS, padding, arbitrary lengths, or dynamic branching. Prefill
and result assembly are included in each timed sample. Short 17-token/4-token
and long 256-token/8-token correctness checks also matched ordinary generation.
Raw samples, source hashes, and capture time are in ignored
`results/graph-varying-prompts.json`; profiler traces and summaries are also
under ignored `results/`.
