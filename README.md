# cuda-gpt2-inference

A focused C++/CUDA learning and performance project with verified float32
RMSNorm, causal prompt attention, cached-token attention, fused GPT-2 GELU,
an experimental fused residual plus LayerNorm, and optional FP16 model/GELU and Tensor Core prefill support
on an RTX 4060 Laptop GPU (8 GB). The custom attention paths run inside pinned
GPT-2; the
unchanged model is the correctness and latency baseline. The local,
Git-ignored `PROJECT_PLAN.md` holds the detailed handoff.

## Completed project and recommended route

The bounded learning project is complete as of 2026-09-30. The recommended
generation route uses the direct greedy loop, fused CUDA GELU, dynamic KV
caching, and PyTorch efficient SDPA. From the CUDA environment described below:

```text
python model_demo.py --fast --prompt "The future of GPU computing is" --new-tokens 32
```

The default remains float32. Add `--dtype float16` to use the verified FP16
model and GELU path described in the precision experiment below; it halves
parameter storage, while the latency advantage over optimized FP32 is workload-dependent.

This generates a fixed token count for one unpadded prompt (up to 1023 prompt
tokens, prompt plus output at most 1024). It does not stop early at EOS or
provide sampling. GPT-2 continues text; it is not an instruction-tuned chatbot.
Custom attention and fused LayerNorm remain available as measured experiments.
The project includes independent numerical checks, stream and input-validation
tests, sanitizer evidence, profiler findings, and paired full-request benchmarks.

A final same-source 27-pair eight-token GELU comparison measured **69.88 ms
with PyTorch GELU versus 55.45 ms with CUDA GELU**, a 20.6% reduction; throughput
was **114.5 versus 144.3 output tokens/s**. All generated IDs and text matched.
This is a local paired result on the recorded laptop, not a portable speed
guarantee. Native attention remains faster than our prompt kernel in the
measured large-prompt workload. LayerNorm fusion did not establish a stable
whole-request gain, and fresh static caching was slower in the final experiment.

Original bounded-project verification: CPU suites passed nine reference tests and explicitly
skipped fourteen CUDA cases. Separate CUDA-enabled suites passed all 23 tests,
including those fourteen CUDA cases; the one- and sixteen-token fast demos
ran successfully. The final cache checks also exercised one-, eight-, and
32-token outputs on varied prompt contents and lengths. CUDA sources were
unchanged from their previously documented sanitizer runs. CPU CI now includes
all four operator suites; the updated workflow has not been run on GitHub.

## How the attention kernel improved, iteration by iteration

The implementation progressed through the following measured experiments:

| Iteration | Change | What it established |
| --- | --- | --- |
| Naive baseline | Separate QK, softmax, and weighted-V kernels; full score matrix | A bounded correctness baseline, limited to 256 tokens |
| Online-softmax tiling | Stream 32-key tiles while maintaining running normalization | Removed quadratic score storage; at `[4,8,256,64]`, peak increment fell from 10 to 2 MiB |
| Four-query tiles | Share each K/V tile across four queries | In a later same-run comparison at `[4,8,256,64]`, single/four-query time fell from 1403.6 to 856.6 µs; native still faster |
| Longer-context support | Validate runtime lengths through 1024 and dispatch custom prefill in all layers | Enabled actual custom/native long-prompt comparisons; extending the limit alone gave no speedup |
| Parallel softmax | All warp lanes compute weights; shuffle reductions and register state; fewer block barriers | Reduced the serial softmax work, measured separately below |
| GPT-2 specialization | Compile a 64-wide path with smaller scratch arrays and a fixed dot-product width | Added a further measured gain while retaining generic head-width support |
| Eight-query reuse | Share each K/V tile across eight query warps | Latest paired run reduced 992-token kernel time by 27.9%, prefill by 16.2%, and complete request time by 3.2% versus the specialized four-query version |
| Optional Tensor Core prefill | FP16 QK and PV matrix tiles with FP32 accumulation and online softmax | At 992 tokens, 0.800 ms versus 1.942 ms for the retained FP32 kernel; native FP16 remains faster at 0.101 ms |

The first experiments used different workloads and runs. Their numbers are not
a single cumulative speedup chain. The parallel-softmax experiment's three stages were measured
together on identical `[1,12,992,64]` float32 inputs:

| Stage | Attention time | Reduction from previous custom stage | Gap versus native |
| --- | ---: | ---: | ---: |
| Four-query baseline, serial softmax | 4.375 ms | — | 8.67x slower |
| Parallel softmax | 3.649 ms | 16.6% | 7.23x slower |
| Parallel softmax + 64-wide specialization | 2.748 ms | 24.7% | 5.45x slower |
| Native efficient SDPA | 0.505 ms | — | Baseline |

Together, those changes reduced custom-kernel time by 37.2%. A subsequent
same-run four/eight comparison measured 2.736/1.973 ms versus native 0.505 ms
at the same shape: eight-query reuse adds 27.9% less kernel time, leaving a
3.91x gap to native. These are separate runs; do not multiply rounded ratios
into a claimed same-run cumulative speedup.

### FP32 custom route versus default native prefill

At **992 prompt tokens and 32 outputs**, the paired model comparison measured:

| Measurement | Default native attention | Previous four-query version | Current eight-query version |
| --- | ---: | ---: | ---: |
| Full-model prefill | **27.37 ms** | 54.80 ms | 45.92 ms |
| Complete generation request | **188.00 ms** | 202.24 ms | 195.71 ms |
| End-to-end output tokens/s | **170.2** | 158.2 | 163.5 |

Current custom prefill is **1.68x slower than native**; complete generation is
about **4.1% slower**. It improves on the previous four-query route by 16.2% for
prefill and 3.2% for the complete request, but it has not beaten native attention.
The recommended route therefore continues to use native efficient SDPA.

Here, "default native" means native attention in the same recommended pipeline:
all routes use FP32 GPT-2, CUDA GELU, and fresh dynamic caches. This isolates the
attention change rather than comparing different precision or generation loops.
Full-model prefill includes every transformer layer and last-position logits;
it is not the standalone attention operation in the preceding table. Inputs,
sampling limits, paired timing, synthetic-prompt construction, and uncontrolled
laptop conditions are described in the experiment below.

## Optional Tensor Core prefill experiment

The existing FP32 eight-query kernel remains available. A separate opt-in
`cuda_attention_tensor_core` path accepts contiguous FP16 `[B,H,S,64]` tensors,
with 1..1024 tokens, no gradients, and the existing 16M input-element limit.
It requires a GPU with compute capability 7.0 or newer. GPT-2 uses it for
prefill; cached single-token steps use native FP16 efficient SDPA.

```text
python model_demo.py --fast --dtype float16 --tensor-core --new-tokens 32
python benchmark_attention.py --tensor-core --batch 1 --heads 12 --sequences 67,128,256,512,992,1024 --output results/attention-tensor-core.json
python benchmark_context.py --tensor-core --output results/context-tensor-core.json
```

One warp handles 16 query rows, streaming 16-key tiles. CUDA WMMA computes
QK-transpose and softmax-weights times V with FP16 operands and FP32
accumulators. Scores, running maxima, denominators, and output numerators stay
FP32; weights are rounded to FP16 for the second matrix multiplication, and
the final output is FP16. Causal masks and padded tile entries are applied
before weight multiplication. There is no full sequence-by-sequence score
allocation. The implementation follows the alignment and warp participation
requirements in [NVIDIA's WMMA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).
Disassembly of the compiled kernel contains `HMMA.16816.F32` instructions.

Same-run isolated attention measurements, `[1,12,S,64]`, in microseconds:

| Tokens | Retained eight-query FP32 | Tensor Core FP16 | Native efficient FP16 |
| --- | ---: | ---: | ---: |
| 67 | 25.29 | 27.44 | 8.29 |
| 128 | 67.17 | 49.36 | 8.96 |
| 256 | 206.56 | 113.51 | 17.66 |
| 512 | 555.08 | 256.64 | 38.66 |
| 992 | 1942.19 | 799.92 | 100.99 |
| 1024 | 2075.08 | 851.12 | 103.12 |

At 992 tokens this is 2.43x faster than our FP32 kernel, but 7.92x slower than
native FP16. The comparison with FP32 changes both precision and implementation;
it does not isolate the gain from Tensor Core instructions alone. The FP32 path
receives exact float casts of the same FP16 inputs. Nine samples rotate/reverse
route order; each uses five CUDA graph replays with eight operations. Capture,
compilation, and warmup are excluded. Flash SDPA is unavailable in this Windows
PyTorch build; the available native efficient backend is measured explicitly.

The full-model FP16 experiment used CUDA GELU and fresh dynamic caches on both
sides, 32 generated tokens, three repeated/truncated prompt token sequences,
and three passes per prompt. All 36 native/custom request pairs matched exact
generated IDs and text. Prompt and cached logits had maximum absolute differences
of 0.25; checks used preselected `rtol=0.02, atol=0.15` (combined relative and
absolute tolerance), separately from the tighter FP32 checks. These are numerical
and deterministic-generation checks, not a broad language-quality evaluation.

At 992 tokens, sequential-group prefill medians were 12.74 ms native versus
19.67 ms custom; paired complete requests were 263.51 versus 287.52 ms,
or 121.4 versus 111.3 output tokens/s. Requests include tokenization, input
construction/transfer, prefill, decode, and detokenization; no graphs.
Laptop timings varied substantially, so this run establishes no whole-request
speedup. Native remains the recommended route. This initial WMMA kernel uses
synchronous small tiles and shared-memory staging; further performance work
needs measured changes to tile size, warp utilization, and staging.

CPU verification: five reference/validation tests passed, eight CUDA tests
explicitly skipped. Separate CUDA verification: all 13 attention tests passed,
including FP64 oracles, high finite logits, partial tiles, exact causal guards,
nondefault streams, and native-binding rejection checks. Focused Tensor Core
memcheck and synccheck reported zero errors; racecheck reported zero hazards,
errors, or warnings. Results are recorded under ignored `results/`.
Source-hashed paired kernel/model reports and compiled disassembly also stay
local. The private study guide explains the new arithmetic and precision tradeoff.

## Eight-query reuse experiment

The latest kernel uses eight query warps (256 threads) per block. Each warp
retains its own online-softmax state, while all eight reuse the loaded K/V tile.
Declared shared storage for 64-wide heads rises from 17,920 to 19,456 bytes
per block, but it serves twice as many queries. The generic 128-wide allocation
is 37,888 bytes. Other operations and the short-prompt single-query route are
unchanged. The input, precision, and causal-mask contracts are unchanged.

Same-run four/eight/native attention medians, microseconds, float32:

| Shape `[B,H,S,D]` | Four-query | Eight-query | Efficient SDPA |
| --- | ---: | ---: | ---: |
| `[1,12,17,64]` | 5.94 | 5.89 | 11.24 |
| `[1,12,64,64]` | 28.80 | 23.60 | 13.57 |
| `[1,12,128,64]` | 85.58 | 67.53 | 25.91 |
| `[1,12,256,64]` | 265.65 | 212.97 | 74.57 |
| `[1,12,512,64]` | 766.41 | 555.98 | 156.54 |
| `[1,12,992,64]` | 2736.23 | 1973.09 | 504.76 |
| `[1,1,1024,128]` | 667.26 | 499.00 | 183.76 |

The 17-token difference is small and not treated as a useful win; model dispatch
still uses the single-query path below 64. The measured longer shapes improved
with eight-query reuse, so it is retained. All candidates passed the explicit
oracle before timing. Nine rotated/reversed-order samples each timed five graph
replays of eight operations; capture/loading/compilation/warmup were excluded.

Paired full-model four/eight/native medians used the same CUDA GELU, dynamic
cache, and 32 output tokens on three repeated/truncated prompt sequences:

| Prompt S | Four / eight / native prefill ms | Four / eight / native request ms | Four / eight / native output tokens/s | Eight wins vs four |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 21.40 / 18.97 / 13.90 | 171.55 / 171.58 / 168.02 | 186.5 / 186.5 / 190.5 | 4/9 |
| 992 | 54.80 / 45.92 / 27.37 | 202.24 / 195.71 / 188.00 | 158.2 / 163.5 / 170.2 | 9/9 |

All four/eight/native generations matched exact IDs and text. Custom dispatch
assertions confirmed 12 custom prefill calls, 372 decode calls, and zero native
fallbacks per request. The long case shows a local request gain over four-query;
the 512-token request medians are effectively equal, so no request win is claimed
there. Routes rotated/reversed over nine samples per length; full requests used
synchronized wall time without graphs and included tokenizer, prompt construction,
transfer, prefill, decode, and text decoding. Module switching/loading/compilation/
warmup were excluded. Laptop conditions remained uncontrolled.

Nsight hardware-counter profiling was attempted but failed with
`ERR_NVGPUCTRPERM`; occupancy, bank conflicts, and stall reasons were not measured.
A separate five-invocation CUDA trace confirmed the 64-wide query kernel ran.
Binary resource inspection found 40 registers per thread, 19,456/37,888 bytes
shared storage for the 64/128-wide variants, and zero static local/stack bytes.
These are compiled resource figures, not measured occupancy. Instrumented trace
durations are diagnostic and are not compared with unprofiled benchmark medians.

CPU: four attention references passed, seven CUDA cases explicitly skipped.
GPU: all eleven attention tests passed, including a new `[1,2,67,64]` case that
leaves only three valid query warps in the final eight-query block, plus partial
key tiles, float64 oracles, masks, streams, and native validation. Full-model
hidden-state, prompt-logit, and cached-logit checks passed at 67/513/1023 tokens.
Focused memcheck/synccheck each completed with zero errors and racecheck with
zero hazards/errors/warnings for the eight-query 64- and 128-wide paths.
Ignored evidence: `results/attention-eight-{ablation,model-paired}.json`,
`results/model-eight-final.json`, `results/attention-eight/` source snapshot and
helpers, and `results/attention-eight-{cuda,profile,trace,resources,memcheck,synccheck,racecheck}.log`.

## Parallel softmax and GPT-2 head specialization (four-query stage)

At this stage, the four-query kernel computed each tile's softmax across all 32 lanes of
its query warp. Warp shuffles reduce the maximum and denominator; running
normalization state stays in registers. Query-local weight sharing uses
`__syncwarp()`, while the shared K/V tiles still require block barriers. This
reduces block-wide barriers per key tile from four to two. A compile-time
64-wide specialization reserves only the space GPT-2 needs; other supported
head widths retain the generic 128-wide storage path. Declared shared arrays
fell from 35,888 to 17,920 bytes per specialized block. This is a storage count,
not a measured occupancy claim.

Same-run ablation on the RTX 4060 Laptop, float32, TF32 disabled, `[1,12,S,64]`:

| Prompt S | Original µs | Parallel softmax only µs | Plus 64-wide specialization µs | Efficient SDPA µs |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 131.53 | 113.38 | 85.94 | 26.32 |
| 256 | 333.18 | 282.52 | 212.68 | 57.55 |
| 512 | 1214.00 | 1018.57 | 766.62 | 157.44 |
| 992 | 4374.73 | 3649.43 | 2748.19 | 504.73 |

At 992 tokens, parallel softmax reduced time by 16.6%; specialization reduced
it a further 24.7%, for **37.2% less time (1.59x faster)** than the original
custom kernel. The remaining gap versus native is about 5.45x. The generic
`[1,1,1024,128]` path improved from 769.15 to 670.52 µs; its two parallel-stage
implementations measured nearly identically, as expected without 64-wide
dispatch. All routes passed the explicit oracle; maximum custom error was
1.02e-6 across these shapes.

A separate paired old/new/native model run kept CUDA GELU, fresh dynamic caches,
and fixed-count 32-token decoding identical on all routes:

| Prompt S | Original / new / native prefill ms | Original / new / native request ms | Original / new / native output tokens/s | New wins vs original |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 26.92 / 21.63 / 14.07 | 198.63 / 209.65 / 215.79 | 161.1 / 152.6 / 148.3 | 2/9 |
| 992 | 74.68 / 55.70 / 27.37 | 219.87 / 202.23 / 188.36 | 145.5 / 158.2 / 169.9 | 9/9 |

The 992-token case establishes a local improvement over the old custom path:
prefill time fell 25.4% and complete request time 8.0%. Native attention remains
faster there. The 512-token prefill improvement did not establish a stable
whole-request gain: request samples varied substantially, especially early in
that run. No broader latency claim is made from its median ordering.
All old/new/native pairs matched exact IDs and decoded text; custom routes
asserted 12 custom prefill calls, 372 custom decode calls, and zero fallbacks.
The normal context benchmark also passed all 36 native/new pairs at
128/256/512/992 tokens. Native remains the recommended attention route.

The isolated ablation used nine rotated/reversed-order samples per shape,
each timing five graph replays of eight operations. Capture, compilation, and
warmup are excluded. The paired model run used nine requests per route/length,
three synthetic repeated/truncated prompt sequences, and synchronized wall
time without graphs. Full requests include tokenization, construction/transfer,
prefill, decode, and text decoding; model/module switching, loading, compilation,
and warmup are outside timing. Laptop conditions were uncontrolled. Raw samples,
stage source snapshots, and hashes are ignored under `results/attention-ablation/`,
`results/attention-warp-{ablation,model-paired}.json`, and
`results/context-warp-final.json`. Run the ordinary context/attention benchmarks
below to reproduce the current route; the old stages are local evidence.

CPU verification: four attention reference tests passed; seven CUDA tests were
explicitly skipped. Separate GPU verification: all eleven attention tests
passed, including partial tiles, 992-token GPT-2 head widths, 1024-token
128-wide fallback, float64 comparison, causal masking, and non-default streams.
Focused memcheck/synccheck each reported zero errors and racecheck zero
hazards/errors/warnings for the changed query kernel's 64- and 128-wide paths.
Full GPT-2 hidden-state, prompt-logit, and cached-logit verification also passed
at 513/1023 tokens after the reduction change. Final reports' source hashes,
sample counts, custom dispatch, exact outputs, and sanitizer summaries were checked.

## Longer-context attention measurements before warp optimization

The context sweep covers 128, 256, 512, and 992 prompt tokens with 32 output
tokens, fitting GPT-2's 1024-position limit. Both routes use CUDA GELU and a
fresh dynamic cache. Both tiled prefill kernels now support runtime sequence
lengths through 1024, including partial tiles. The adapter uses custom prefill
and cached-token attention throughout this sweep; dispatch assertions reject
silent fallback. The naive three-kernel baseline retains its actual 256-token
softmax limit. Extending support does not itself improve performance.

```text
python benchmark_context.py --output results/context.json
python benchmark_attention.py --batch 1 --heads 12 --sequences 128,256,512,992 --output results/attention-context.json
python benchmark_decode.py --lengths 128,256,512,1024 --output results/decode-context.json
```

Local RTX 4060 Laptop results on 2026-09-30, float32 with TF32 disabled:

| Prompt tokens | Native / custom full-model prefill ms | Native / custom request ms | Native / custom output tokens/s | Native / four-query attention µs |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 6.72 / 7.44 | 167.22 / 167.24 | 191.4 / 191.3 | 19.48 / 132.53 |
| 256 | 9.17 / 12.74 | 167.19 / 166.83 | 191.4 / 191.8 | 56.04 / 333.06 |
| 512 | 17.00 / 29.32 | 169.04 / 180.26 | 189.3 / 177.5 | 156.49 / 1219.89 |
| 992 | 31.86 / 75.34 | 180.37 / 218.12 | 177.4 / 146.7 | 504.32 / 4372.94 |

All 36 generation pairs matched exact IDs and decoded text. Same-input prompt
and cached-token logits passed rtol 1e-3, atol 5e-3; maximum observed difference
was 0.000122. Dispatch assertions verified all 12 layers used custom prefill
and all cached forwards used custom decode, with zero native fallbacks. The
isolated kernels were checked against the explicit matmul/mask/softmax oracle
at every length; maximum four-query absolute error was 1.02e-6. At 992 tokens,
single-query tiled attention took 7747.12 µs; four-query sharing reduced that
to 4372.94 µs, but efficient SDPA remained much faster. Both custom and native
paths allocated 2.91 MiB above resident inputs for this standalone shape.

Isolated cached-token attention did win: custom/native medians were
19.17/24.47, 37.30/45.95, 74.62/89.42, and 128.51/152.65 µs for caches
of 128, 256, 512, and 1024 tokens. Those savings did not establish a reliable
whole-request win in the sweep above.

These are **controlled context-length experiments**: three prompt token
sequences are repeated/truncated to exact lengths, not natural long documents
or a model-quality test. Request medians use nine alternating-order pairs per
length and include tokenization, input transfer, prompt construction, prefill,
fixed-count decoding, and text decoding. Loading, compilation, and warmup are
excluded; no graph capture is used for requests. Prefill-only measurements use
resident IDs and sequential route groups, so small route differences are not
reliable paired wins. Standalone attention uses CUDA graph replay events to
isolate warmed operations; its timings exclude capture and are not request
latencies. Laptop clocks and thermals were uncontrolled, and request samples
vary substantially; the nonmonotonic request medians are not evidence that
longer prompts cost less. No consistent overall custom-attention win emerged.

CPU verification: four attention reference tests passed; seven CUDA tests were
explicitly skipped. Separate CUDA verification: all eleven attention tests passed,
plus the context sweep's model-logit, dispatch, generation, and isolated
attention oracle checks. Added long-prefill coverage at 257/513/1024 tokens,
odd head width, multiple batches/heads, non-default streams, causal perturbation,
float64 comparison, and rejection above the supported bounds. Native validation
was extended for tiled paths; the tile-processing CUDA arithmetic is unchanged.
Focused Compute Sanitizer memcheck/synccheck each reported zero errors and
racecheck zero hazards/errors/warnings for both tiled kernels at 513 and 1024
tokens. The model CLI also passed complete hidden-state, prompt-logit, and
cached-logit checks at 513/1023 tokens, with a 992-token generation prompt.
Raw samples,
source hashes, model pin, and runtime metadata are saved under ignored
`results/{context-long-custom,attention-long-custom,decode-context-final}.json`.
The earlier `context-final.json` experiment used native fallback above 256 and
does not measure custom long prefill; it is retained only as historical evidence.

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
| `csrc/attention.cu` | Three-kernel baseline, single-query and eight-query prompt tiles, cache decode |
| `csrc/attention_tensor_core.cu` | Optional FP16 WMMA prompt attention with FP32 accumulation, head width 64 |
| `test_attention.py` | CPU oracle, CUDA, masking, and stream checks |
| `benchmark_attention.py` | Attention correctness, graph latency, and peak allocation |
| `benchmark_context.py` | FP32 or optional FP16 Tensor Core/native context sweep and paired request timing |
| `benchmark_decode.py` | One-query cache-attention correctness and graph latency |
| `model_demo.py` | Pinned GPT-2, prompt/decode adapter, checks, generation, model benchmark |
| `benchmark_greedy.py` | Varied-prompt end-to-end greedy generation comparison |
| `gelu_new.py`, `csrc/gelu*` | Fused CUDA GPT-2 GELU activation and native validation |
| `test_gelu.py`, `benchmark_gelu.py` | Activation correctness and paired model benchmark |
| `benchmark_cache.py` | Fresh dynamic/static cache correctness, paired request timing, and profiler counts |
| `benchmark_precision.py` | Four-route FP32/FP16 comparison, latency percentiles, token readiness, memory, and GPU telemetry |
| `fused_ln.py`, `csrc/fused_ln*`, `test_fused_ln.py` | Opt-in residual plus LayerNorm kernel and checks |
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
the current eight-query version shares a K/V tile across eight warps. A separate kernel
handles one new query against cached K/V, which includes the current token.
`model_demo.py` sends supported GPT-2 prefill below 64 tokens to the
single-query tile, 64–1024 tokens to the eight-query tile, and unmasked one-token
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

Both generation benchmarks now also report **end-to-end output tokens/s**:
newly generated tokens divided by synchronized text-to-text request time.
This includes prompt processing and decoding, so it is not steady-state decode
throughput or multi-request server throughput. In a fresh 27-pair eight-token
run, `model.generate` measured 61.13 ms / 130.9 output tokens/s and the direct
loop 51.33 ms / 155.9 output tokens/s; all outputs matched, with 26/27 paired
direct wins. Raw samples are in ignored `results/greedy-throughput.json`.

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

With the same eight-token workload and the new throughput field, a further
27-pair run measured 54.96 ms / **145.5 output tokens/s** for PyTorch GELU and
43.08 ms / **185.7 output tokens/s** for custom GELU. Outputs matched exactly;
custom won 27/27 pairs. Raw samples are in ignored `results/gelu-throughput.json`.

## FP16 GELU and precision comparison (2026-09-30)

`model_demo.py --fast --dtype float16` loads the same pinned GPT-2 weights in
half precision and uses PyTorch efficient SDPA with the direct greedy loop.
RMSNorm, custom attention, and fused LayerNorm retain their float32 contracts.
Only custom GELU adds float16 support. The templated kernel reads/writes half
values, computes each operation in float, and explicitly preserves the half
rounding points of Transformers' native `NewGELUActivation`, including the
rounded square inside the cubic calculation. The float32 route retains its
original fused arithmetic.

An initial round-at-output-only implementation changed one prompt's continuation.
The final implementation matches native half GELU bit-for-bit on **all 63,488
finite FP16 bit patterns** in the tested PyTorch build, including signed zero
and subnormals. Tests also compare against a float64 formula oracle with a
documented tolerance for inherited native-half rounding error; the native-half
comparison itself allows zero error. CUDA stream, odd length, prompt/decode/batch,
extreme finite inputs, and native input rejection checks passed. CPU: one GELU
reference pass, three GPU skips. CUDA-enabled GELU suite: all four tests passed.
Targeted memcheck/synccheck each reported zero errors. The full test-suite
racecheck completed its tests but stalled during shutdown and was terminated;
a focused FP32/FP16 kernel probe then completed with zero hazards, errors,
or warnings. These sanitizer runs target `gelu_new_kernel`; racecheck examines
shared-memory hazards, and this kernel uses no shared memory.

The new benchmark compares four resident routes on identical natural prompts,
rotates/reverses execution order, and includes tokenization, input transfer,
dynamic cache setup, prefill, decoding, and text decoding. Module swaps, loading,
compilation, warmup, GPU-state snapshots, and separate token diagnostics are
outside request timing. There is no graph capture. For nine 6–10-token prompts,
three passes each, and 32 output tokens (27 paired samples per route):

| Route | Median request ms | p95 request ms | Median output tokens/s |
| --- | ---: | ---: | ---: |
| FP32, native GELU | 247.57 | 332.43 | 129.3 |
| FP32, CUDA GELU | 218.81 | 338.55 | 146.2 |
| FP16, native GELU | 250.45 | 361.64 | 127.8 |
| FP16, CUDA GELU | 191.98 | 245.16 | 166.7 |

FP16 CUDA GELU reduced median request time **23.3% versus native FP16 GELU**
and won 26/27 pairs. Eight-token outputs measured 68.97/54.61 ms native/custom
FP16 (20.8% lower, 27/27 wins). Three natural 47/84/117-token prompts with 32
outputs measured 233.82/180.85 ms (22.7% lower, 9/9 wins). The one-token smoke
also passed. Within each precision, native and custom routes matched generated
IDs/text on every measured pair; FP16 teacher-forced logits were exactly equal
between native and custom GELU. Cross-precision equality is not guaranteed:
two of the nine short prompts produced different 32-token continuations in FP16
versus FP32, while all eight-token and longer-prompt comparisons matched.

Parameter storage fell from **474.70 MiB to 237.35 MiB**. For the 32-token short
requests, maximum incremental allocated peak fell from 3,444,736 bytes to
1,731,072 bytes. Both model precisions were resident for the paired comparison;
incremental peaks exclude that resident baseline, and parameter bytes exclude
buffers and allocator reservations. These are not whole-process VRAM totals.

The precision change alone did not establish a reliable latency win: native
FP16 lost 15/27 pairs to native FP32 in the 32-token run. With custom GELU,
FP16 won 17/27 short 32-token pairs but only 15/27 eight-token pairs, and the
long-prompt median was nearly identical to optimized FP32 (180.85/180.53 ms).
Keep FP16 opt-in for memory savings and precision experiments. Clock snapshots
varied substantially (for example, 1890–2565 MHz SM clocks in the 32-token run);
power and thermal conditions were recorded, not controlled. Teacher-forced
logits versus FP32 differed by up to 0.448, and prompt NLL drift by up to 0.0155;
these few prompts are a numerical diagnostic, not a model-quality evaluation.

Separate CUDA-event diagnostics mark GPU argmax completion without synchronizing
after every token. The 32-token FP16 CUDA route measured median first-token
device readiness 7.10 ms and inter-token device gap 5.51 ms. These include host
enqueue gaps and differ from user-delivered streaming TTFT/latency: first-token
detokenization and network delivery are excluded. They come from one separately
instrumented pass per prompt; they should not be summed to reconstruct the
uninstrumented request median. For one output token the inter-token field is
null. p95 uses nearest rank and is coarse at these sample counts; raw samples
are retained alongside telemetry, source hashes, memory, and numerical checks.

```text
python test_gelu.py --cuda
python model_demo.py --fast --dtype float16 --new-tokens 32
python benchmark_precision.py --new-tokens 8 --output results/precision-eight.json
python benchmark_precision.py --new-tokens 32 --output results/precision-thirtytwo.json
```

Use `--prompts-file prompts.txt` for one natural prompt per line, subject to the
existing 256-token prompt limit. Local evidence is ignored under
`results/precision-final-*.json`, `results/precision-final-cuda.log`,
`results/precision-{memcheck,synccheck,racecheck-focused}.log`. The exploratory
`results/precision-8.json` precedes the rounding correction and is not final evidence.

## Fused attention residual and LayerNorm experiment

In each GPT-2 block, the attention output is added to the incoming residual,
then `ln_2` normalizes the 768-feature result. `csrc/fused_ln.cu` performs both
steps in one float32 kernel and returns the sum and normalized tensor, since
the MLP still needs the sum for its own residual. One block handles one token
row; its 256 threads each hold three values in registers and reduce mean and
variance with warp shuffles and shared-memory scratch. `fused_ln.py` can
opt into this GPT-2 block path; the ordinary block method is restored when
disabled. The benchmark keeps the faster CUDA GELU enabled on both sides and
uses PyTorch SDPA, so the comparison isolates this additional fusion.

For a decode-shaped `[1,1,768]` tensor, nine paired batches of 100 calls
measured launch-inclusive medians of 44.60 µs for PyTorch add plus LayerNorm
and 18.75 µs for the direct native fused call. Complete text-to-text results
were less stable. Two 27-pair, eight-token runs after the native-call change
measured **77.20/68.88 ms** PyTorch/fused (23/27 fused wins) and
**71.71/75.19 ms** (11/27 wins). A 27-pair 16-token run measured
103.83/101.22 ms (18/27 wins); three longer prompts repeated three times
measured 46.99/44.28 ms (8/9 wins). Every pair generated identical IDs and
text. The mixed eight-token reruns do not establish a reliable whole-request
speedup, so this path remains opt-in. Compilation, model load, warmup, and
module swapping are outside each timed request.
After a final input-validation change, another 27-pair eight-token run measured
50.34/48.66 ms (18/27 wins), with exact IDs and text; its source hashes match
the committed kernel and adapter. The final ignored report is
`results/fused-ln-final.json`.

A five-forward cached-token trace at S129 confirmed the intended fusion:
native LayerNorm calls fell from 25 to 13 per forward, adds from 25 to 13,
and the new kernel ran 12 times. Cache concatenations stayed at 24, and the
cuBLAS projections were unchanged. The old add and LayerNorm kernels used
about 0.128 ms of GPU work per token in this trace; their remaining calls
plus the fused kernel used about 0.101 ms. This small device-side saving and
uncontrolled laptop timing explain why the full-request result is uncertain.

At S129, maximum hidden/prompt-logit/cached-logit absolute differences were
0.00323/9.16e-5/7.63e-5. CPU: one reference test passed, two CUDA cases
skipped. CUDA: all three tests passed, including a non-default stream and
native input rejection. Compute Sanitizer memcheck and synccheck reported
zero errors; targeted racecheck reported zero hazards, errors, and warnings.
Run `python test_fused_ln.py --cuda` and
`python benchmark_gelu.py --fused-ln --output results/fused-ln-local.json` from
the CUDA setup. Raw reports and sanitizer logs are ignored under `results/`.

## Full-model profiling and CUDA Graph replay

### Final fresh-cache comparison (2026-09-30)

`benchmark_cache.py` compares the same direct loop, CUDA GELU, and efficient
SDPA with either its dynamic cache or a fresh Transformers `StaticCache`.
Static capacity is exactly prompt length plus requested output count. Allocation
and zeroing are included in each request, along with tokenization, transfer,
prefill, decode, and text decoding. There are no graphs or compilation captures;
model loading, extension compilation, warmup, and diagnostic profiling are
outside request timing. Route order alternates, and every pair checks exact
generated token IDs and decoded text.

| Natural prompt lengths | Output tokens | Pairs | Dynamic ms / tokens/s | Static ms / tokens/s | Static paired wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6–10 | 1 | 9 | 7.14 / 140.1 | 10.45 / 95.7 | 0/9 |
| 6–10 | 8 | 27 | 48.09 / 166.3 | 61.23 / 130.7 | 0/27 |
| 6–10 | 32 | 27 | 192.97 / 165.8 | 253.00 / 126.5 | 1/27 |
| 47/84/117 | 32 | 9 | 209.45 / 152.8 | 268.16 / 119.3 | 1/9 |

All pairs matched IDs and text. An earlier run also found static caching slower
for eight-token, 32-token, and longer-prompt workloads. Absolute latency varied
between runs; laptop power and thermal conditions were not controlled. Output
tokens/s here divides output count by complete request time; time to first token
and inter-token latency are not collected by this benchmark.

A separate eight-token profile counted 193 `aten::cat` calls with dynamic
caching versus one final output-assembly call with static caching. Static caching
instead performed 192 indexed writes and 24 zero-buffer allocations. At 32
tokens, the cat counts were 769 versus one, with 768 static indexed writes.
Operator counts confirm the intended change, but do not isolate the cost of
each operation. Extra setup, indexed writes, and masked fixed-capacity attention
are candidate costs; the timings establish the overall regression, not their
individual contributions. Dynamic caching remains the recommended request route.
The experiment does not establish that other preallocated-cache designs or
reused server buffers would lose; a custom cache kernel is not justified by
this comparison alone.

```text
python benchmark_cache.py --new-tokens 8 --output results/cache-eight.json
python benchmark_cache.py --new-tokens 32 --output results/cache-thirtytwo.json
python benchmark_cache.py --new-tokens 32 --prompts-file prompts.txt --output results/cache-varied.json
```

For the last command, create a file containing one natural, unpadded prompt per
line (each at most 256 tokens). Reports retain prompt text, all samples, source
hashes, device/runtime metadata, and incremental peak allocated memory. Final
local evidence remains ignored under `results/cache-final-*.json` and
`results/gelu-final.json`; GPU checks and demo logs are `results/final-*.log`.

### Earlier graph experiment

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

After fused GELU, a new five-forward S129 cached-token profile with
`logits_to_keep=1` found 48 `addmm` calls and one vocabulary `mm` per forward,
about 1.75 ms and 0.72 ms of GPU work per forward in that instrumented run.
Efficient attention used about 0.40 ms. Each forward also invoked 25 native
LayerNorm operations, 25 residual/embedding adds, and 24 cache concatenations.
The residual plus `ln_2` fusion experiment described above removed 12 small
launches per cached token, but its whole-request gain was not stable. The
final cache experiment reported here tested the remaining concatenations and retained
dynamic caching based on full-request measurements.
The ignored profile is `results/profile-gelu-token.json`. Profiler timing is
diagnostic and should not be compared directly with unprofiled latency.

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
