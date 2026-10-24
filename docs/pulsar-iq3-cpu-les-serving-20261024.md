# Pulsar IQ3 CPU LES and cache-aware serving

For the subsequent kernel optimizations, mixed-KV comparison and current serving build, see [the throughput investigation](pulsar-iq3-throughput-20261024.md). The measurements and process identifiers below describe the preceding deployment.

Model: `/mnt/openwebui/AIKAR/Adaptive-bit/models/Lumen-3.1-Pulsar-IQ3-Quality.gguf`, 11,760,518,400 bytes. Source baseline: `36a623af6`, with local uncommitted changes. Host: Xeon E5-2696 v3, 18 cores / 36 threads, AVX2/F16C, 256 KiB L2 per core, 45 MiB L3, 121 GiB RAM. The prior CPU/P620 comparison favored CPU. This session implements and measures the CPU path.

## Implementation and operating settings

LES now performs asynchronous next-layer expert prediction, bounded cache-line preparation, generation/serial cancellation and ready-first execution in generic and repacked CPU expert matrix multiplication. Real router selections, scales, selected output slots and final expert reduction order remain unchanged. Preparation completion is software bookkeeping, not a hardware cache-hit measurement. See [implementation details](pulsar-cpu-les-implementation.md).

CPU paged attention now uses cache-sized K/V tiles, query/GQA reuse, SIMD dot/softmax/accumulation and native F16 conversion through the existing helper. Graph work buffers provide bounded scratch. Scalar attention remains selectable with `GGML_CPU_PAGED_ATTN_REFERENCE=1`. No model weights were modified.

The launcher uses CPU, 18 decode and batch threads, batch 1024, microbatch 512, context 8192, one slot and paged F16 KV with 16-token pages. F16 was modestly faster than Q8_KV at longer contexts; Q8_KV remains an option when KV memory matters more. LES remains opt-in because it slowed the measured short-context workload. SWA-full is disabled, including the inherited environment switch; passing that option to this launcher is rejected. Context shifting is disabled, so prompt plus generated tokens must fit the context. Prompt caching is enabled for serving and disabled for benchmarks.

```bash
./scripts/serve-pulsar-iq3-temporary.sh
```

API base: `http://127.0.0.1:11438/v1`. Alias: `Lumen-3.1-Pulsar-IQ3-Quality`.

To try LES, stop the current server first or choose another port:

```bash
PULSAR_IQ3_LES=1 PULSAR_IQ3_LES_PREFETCH_MIB=4 ./scripts/serve-pulsar-iq3-temporary.sh
```

Additional environment settings are `PULSAR_IQ3_MODEL`, `PULSAR_IQ3_SERVER_BIN`, `PULSAR_IQ3_HOST`, `PULSAR_IQ3_PORT`, `PULSAR_IQ3_CTX`, `PULSAR_IQ3_THREADS`, `PULSAR_IQ3_THREADS_BATCH`, `PULSAR_IQ3_REASONING`, `PULSAR_IQ3_PAGED` and `PULSAR_IQ3_KV_TYPE`. Set `PULSAR_IQ3_KV_TYPE=q8_kv` to use paged Q8_KV; set `PULSAR_IQ3_PAGED=0` for conventional F16 KV. The library/CLI LES budget defaults to 16 MiB; the launcher uses 4 MiB when LES is explicitly enabled. This is a per-job limit; it does not guarantee LLC capacity or residency.

## Matched measurements

Sequential local HTTP requests, native Release build, one slot, context 8192, greedy sampling, fixed uncached input length, 64 generated tokens and startup warmup. Every request asserts the number of evaluated input tokens. These are arithmetic means in tokens/second from a small host-specific screen, not general speed guarantees. Configurations can produce different generated tokens because the attention implementations differ numerically.

At 512 input tokens, three repetitions each:

| Configuration | Prompt processing | Generation |
| --- | ---: | ---: |
| Conventional F16, bounded SWA | 72.23 | 10.29 |
| Paged Q8_KV, scalar reference | 57.28 | 10.10 |
| Paged Q8_KV, optimized | 71.18 | 10.61 |
| Paged F16, optimized with native conversion | 70.35 | 10.65 |
| Paged Q8_KV, LES 4 MiB, 18 threads | 72.12 | 10.18 |
| Paged F16, LES 4 MiB, 18 threads | 70.71 | 10.29 |
| Paged Q8_KV, LES 8 MiB, 16 threads | 71.13 | 10.03 |
| Paged Q8_KV, LES 16 MiB, 18 threads | 71.44 | 9.68 |

Optimized Q8_KV improved prefill by about 24% and generation by about 5% against scalar paged Q8_KV in this screen. Against conventional F16, its generation improvement was about 3%, with slightly lower prefill. F16 native conversion and Q8_KV generation were close. LES with 4 MiB reduced generation by about 4% relative to optimized Q8_KV without LES; larger budgets and 16-thread execution did not recover the loss. LES is implemented but no speedup is claimed. The extra router work and speculative reads compete with inference for CPU/cache/memory resources.

Longer inputs, two repetitions each:

| Input tokens | Configuration | Prompt processing | Generation |
| ---: | --- | ---: | ---: |
| 2048 | Conventional F16 | 68.17 | 9.65 |
| 2048 | Paged Q8_KV, scalar | 37.74 | 8.73 |
| 2048 | Paged Q8_KV, optimized | 66.96 | 9.76 |
| 2048 | Paged F16, optimized | 67.86 | 9.82 |
| 4096 | Conventional F16 | 65.73 | 9.28 |
| 4096 | Paged Q8_KV, scalar | 30.64 | 7.97 |
| 4096 | Paged Q8_KV, optimized | 63.61 | 9.28 |
| 4096 | Paged F16, optimized | 64.53 | 9.42 |

At 4096 tokens, optimized Q8_KV reached about 2.08 times the scalar paged prefill throughput and 16% higher generation throughput. The selected F16 configuration improved generation by about 1.5% over conventional F16 there, with about 1.8% lower prefill throughput. Across the measured input lengths, generation improved about 1.5-3.5% over conventional F16. This is a modest whole-model gain: the large speedup applies to the formerly slower paged implementation, not to the best prior conventional CPU path. The small Q8_KV/F16 speed differences are not a statistical confidence claim.

An earlier optimized F16 implementation without the native conversion helper reached only 9.03 generation tokens/second and was rejected. All measurements here omit SWA-full. Hardware performance counters, including software task-clock, were denied by the host; no host security setting was changed.

## Numerical checks and limits

With optimized Q8_KV attention held fixed, LES off/on produced byte-identical logits across 2,883,584 floats in a probe with 128 fixed input tokens and ten fixed decode steps. The final LES probe recorded 2400 required experts, 2320 predictions, 1279 prediction matches, 897 prepared software hits, 4,348,164,608 prepared bytes and 491 cancelled jobs. These are probe counters, not general hit rates.

Scalar versus optimized attention does not produce identical full-model logits. First-layer Q/K/V and decoded cache data were identical; reduction rounding differences grew through quantization and later MoE routing. Against an independent FP64 first-attention calculation, optimized NMSE was 3.90e-14 (maximum absolute error 5.19e-6), versus scalar NMSE 1.04e-12 (maximum absolute error 2.59e-5). Better accuracy for this operator does not prove whole-model quality equivalence.

A 292-token natural-text teacher-forcing screen covering Korean, coding, arithmetic and technical text gave Q8_KV mean NLL 5.4565 optimized versus 5.5314 scalar, with perplexities 234.28 versus 252.51 and argmax agreement 87.33%. The final F16 build gave NLL 5.4445 optimized versus 5.5094 scalar, perplexities 231.48 versus 247.01 and argmax agreement 88.01%. The F16 LES off/on probe also passed byte comparison of 2,883,584 logits. This tiny sample does not certify language quality or exclude regressions on other inputs. No broad evaluation or CUDA backend equivalence is claimed.

## Verification and reproduction

The final main-checkout build passed 103/103 selected paged-attention cases and 11/11 related CTests. Four additional cases explicitly cover missing pages and an entirely empty key tile. LES tests covered prediction/misses, budgets, duplicate selected IDs and actual changed expert traversal, repacked weights, callback graph reuse, rapid supersession, generation changes and shutdown. The final scheduler unit binary passed ASan/UBSan/LeakSanitizer; the linked CPU libraries were Release libraries, so this is not full-library sanitizer coverage. Earlier sandbox LeakSanitizer could not run under ptrace. Final source review found no remaining critical or important issue in the inspected paths, including the added empty-tile cases. It identified a CPU-only test registration issue, now guarded consistently with the existing direct-backend tests. Stress tests do not exhaust every possible worker interleaving.

```bash
cmake -S . -B build-pulsar-cpu-les -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF
CCACHE_DIR=/tmp/pulsar-iq3-ccache cmake --build build-pulsar-cpu-les --target llama-server test-cpu-les test-backend-ops -j12
```

Raw configurations, per-request timings/outputs, measurement/probe scripts and validation logs are stored in `build-pulsar-cpu-les/results/`. The measurement driver uses an explicit worktree root; change `ROOT` when reproducing against another checkout. Numerical probe scripts contain the model path and the quality probe reads the supplied text fixture from `/tmp/pulsar-quality-text.txt`; adjust those paths for another host. No commit, push or PR submission was performed.

The first final-service validation attempt incorrectly assumed request `cache_prompt=false` overrides a server launched with `--cache-prompt`. Existing `server-schema.cpp` explicitly forces it back to true. That attempt was rejected by its input-count assertion and retained in `serving-check-cache-flag-rejected.log`. A second attempt found that the erase-slot API requires the separately disabled slot-save feature; its HTTP 501 is retained in `serving-check-slot-erase-rejected.log`. The corrected validation replaces the slot with a disjoint one-token prompt before asserting an uncached 1536-token evaluation. Benchmark servers used `--no-cache-prompt` and input-count assertions throughout; their throughput results are unaffected by these validation mistakes.

## Active service verification

The final CPU server runs at `127.0.0.1:11438`, PID 1129065. Its inspected command line has paged F16 KV, no SWA-full and no LES. Health returned `ok`, and `/v1/models` returned the expected alias. Arithmetic returned `42`, Korean returned `서버 준비가 완료되었습니다.`, and the Python example returned a valid `add(a, b)` function. OpenAI streaming included content and the final `[DONE]` event.

Repeated 128-token input reused 127 tokens and evaluated one, with 87.40 ms prompt time. Appending to 1536 tokens reused 129 tokens and evaluated 1407. Repeating the 1536-token input reused 1535 and evaluated one, with 99.13 ms prompt time. After replacing the slot with a disjoint prompt, a fresh request evaluated all 1536 tokens in 22,273 ms. Its four generated tokens matched the cached request. This verifies one bounded-window reuse scenario, not arbitrary-history equivalence or a general latency guarantee.

The runtime reported a 466.26 MiB paged CPU KV pool, 512 full-attention pages and 98 SWA pages. The larger full-attention pool belongs to the model's full-attention layers; SWA-full remains disabled. Observed RSS after checks was about 16.8 GiB, and P620 memory usage was 7 MiB. These are point observations, not peak measurements.

Runtime artifacts: `build-pulsar-cpu-les/results/serving.pid`, `serving.log`, `serving-check.json`, `serving-check.log`, `serving-runtime.json` and `restart.json`. The previous service was stopped after verifying its model and port. Stop this session's server before launching another on the same port:

```bash
kill "$(cat build-pulsar-cpu-les/results/serving.pid)"
```
