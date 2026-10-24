# Pulsar IQ3 CPU throughput investigation

This extends [the CPU LES and paged-attention report](pulsar-iq3-cpu-les-serving-20261024.md). Model weights, selected experts, scales and expert reduction order are preserved. SWA-full remains disabled. All numbers refer to the Xeon E5-2696 v3 host described there, unless explicitly labeled P620. Raw probes, fixtures, configurations and logs are under `build-pulsar-cpu-les/results/phase3/`.

## Measured bottleneck

The evaluation callback timed individual graph operations over 12 fixed decode steps after a 512-token prefill, with 18 CPU threads and paged F16 KV. It splits graph execution and adds overhead; these percentages diagnose operations and are not uninstrumented generation timings.

| Operation group | Share of instrumented operation time |
| --- | ---: |
| Dense Q4_0 matrix multiplication | 38.65% |
| Expert IQ3_XXS matrix multiplication | 14.78% |
| Expert Q4_0 matrix multiplication | 8.14% |
| Expert IQ3_S matrix multiplication | 6.65% |
| Paged attention | 6.74% |
| Dense Q8_0 matrix multiplication | 5.39% |
| Output TANH | 2.69% |

The output matrix alone took about 14.1 ms/token in this instrumented run and spans about 415 MiB. Matrix operands account for 2,107,111,424 logical weight bytes/token, including about 0.642 GB of selected expert weights. This is logical tensor accounting, not measured DRAM traffic. Hardware counters were unavailable; no host security settings were changed.

A separate 1 GiB streaming-read probe over eight passes reached 30.22, 30.46, 30.58, 30.77 and 28.71 GB/s at 6, 12, 18, 24 and 36 threads. Reading every logical weight span once at 30.77 GB/s would allow about 14.6 tokens/s before other work. Cache reuse, traffic, decoding and layout affect the real limit. This estimate does not prove an absolute hardware ceiling, but ordinary 20 or 50 tokens/s cannot be promised from KV compression alone. The same read-once accounting would require about 42.1 or 105.4 GB/s respectively, before attention and other work.

## Exact CPU changes

- Q4_0 GEMV reuses the unsigned-offset dot helper already used by GEMM. Integer corrections and FP32 accumulation order are preserved; CPUs with native signed AVX-VNNI-INT8 retain their existing path. No quantization format changes.
- Large TANH operations partition contiguous elements across threads, including a single long vocabulary row. Strided tensors retain row traversal. The same scalar `tanhf` and conversions are used.
- Paged IQ4_NL decode uses AVX2 nibble lookup, signed extension and scale multiplication using the existing IQ4 table. Other CPUs use the existing scalar decoder.

The existing generic expert path already avoids a per-expert barrier and uses static row partitions for this decode shape. Adding a global work queue would add synchronization without addressing the larger dense-weight cost, so no new queue was introduced. LES remains available but disabled by default because previous measurements found a slowdown.

## KV policy baseline

Before the new changes, sequential uncached HTTP requests used context 8192, one slot, 18 threads, batch 1024, microbatch 512, 64 generated tokens and two repetitions at each input depth. Each request asserted that every input token was evaluated. Arithmetic means below are generation tokens/s; they are a small host-specific screen.

| Global KV | Local KV | 512 input tokens | 4096 input tokens |
| --- | --- | ---: | ---: |
| F16 | F16 | 10.44 | 9.25 |
| Q8_KV | Q8_KV | 10.55 | 9.12 |
| IQ4_NL | Q8_KV | 9.87 | 6.41 |
| IQ4_NL | F16 | 10.03 | 6.42 |

IQ4_NL global plus Q8_KV local used a 202.92 MiB paged pool versus 466.26 MiB for F16. Its scalar reconstruction cost outweighed reduced KV traffic here. The new SIMD decoder is evaluated separately below. F16 is explicitly permitted as the default when faster.

## Validation and candidate measurements

Candidate build: `build-pulsar-throughput`, native Release, CPU only. The live baseline build `build-pulsar-cpu-les` was not rebuilt while loaded by its server.

The build passed 11/11 related CTests and 230/230 selected PAGED_ATTN/TANH backend tests. A direct Q4 GEMV probe passed 64 scalar integer-oracle cases with bitwise FP32 results and output guards, covering activation -128, all weight nibbles, FP16 scales, dimensions 32/256/704/2816 and output tails. A direct IQ4 decoder probe linked against the candidate passed 17,920 bitwise cases. Its extracted helper also passed native, scalar-fallback and ASan/UBSan checks, including signed zero, subnormal scales, large finite scales, unaligned SIMD input and output guards. This is not full-library sanitizer coverage. TANH direct partition tests passed 36 bitwise cases across F32/F16/BF16, strided views, in-place operation and 2/7/64 partitions.

In a small single-thread Q4 microbenchmark with 1000 calls per shape, the new helper reduced call time from 0.656 to 0.486 us for dimension 704 and eight columns, and from 2.609 to 1.908 us for dimension 2816 and eight columns. Wider output shapes also improved in that screen. These cache-resident kernel results are not whole-model speedups.

A paired 64-step fixed-token decode probe measured 10.4355 tokens/s for the previous build and 11.3121 tokens/s for the candidate at 18 threads, about 8.4% faster. Candidate 24 and 30 threads reached 10.7706 and 10.5471 tokens/s; more threads did not help this run. Instrumented TANH time fell from 38.05 to 2.87 ms over 12 steps. These probes use fixed inputs rather than sampled continuations; HTTP generation is measured separately.

The candidate's F16 logits were byte-identical to the previous optimized F16 build across 2,883,584 floats in the fixed prefill/decode probe. A 292-token teacher-forcing screen gave the following values. The small sample is diagnostic and does not certify model quality or establish that a lower-precision policy is better.

| Global/local KV | Mean NLL | Perplexity | Argmax agreement with F16 |
| --- | ---: | ---: | ---: |
| F16/F16 | 5.4445 | 231.48 | 100% |
| Q8_KV/Q8_KV | 5.4565 | 234.28 | 87.67% |
| IQ4_NL/Q8_KV | 5.4920 | 242.75 | 85.62% |
| IQ4_NL/F16 | 5.3904 | 219.29 | 82.53% |

The same uncached HTTP protocol as the baseline, two repetitions per depth, gave:

| Candidate configuration | Input tokens | Prompt tokens/s | Generation tokens/s |
| --- | ---: | ---: | ---: |
| F16/F16, 18 threads | 512 | 70.04 | 11.26 |
| F16/F16, 18 threads | 4096 | 61.26 | 9.85 |
| F16/F16, 24 decode / 18 batch threads | 512 | 71.57 | 10.71 |
| IQ4_NL/Q8_KV, 18 threads | 512 | 72.39 | 11.31 |
| IQ4_NL/Q8_KV, 18 threads | 4096 | 60.33 | 9.75 |

The selected F16 configuration improved generation by about 7.8% at 512 input tokens and 6.6% at 4096 against the paired earlier F16 HTTP screen. IQ4_NL/Q8_KV improved from 6.41 to about 9.75 tokens/s at 4096, about 52%, after the exact decoder and other kernel changes. The very small short-context mixed-KV advantage does not establish superiority; F16 is slightly faster at the longer depth and avoids KV quantization. It remains the default, with mixed KV available for its smaller pool. These are small sequential samples, not confidence intervals or a universal speed claim.

Additional Q4_0 backend checks passed 121/121 MUL_MAT and 87/87 MUL_MAT_ID cases. Two initial overconstrained operation filters selected zero cases and were discarded; the reported counts come from subsequent filters with nonempty results.

## Speculative decoding and P620 screen

All ngram fixtures used conventional F16 KV, bounded SWA, greedy generation and 192 generated tokens. Each configuration ran once per fixture. Outputs were identical to the non-speculative reference on all four fixtures for both ngram implementations, including their truncated continuations. This does not establish equivalence on arbitrary prompts or sampling settings.

| Fixture | No speculation | ngram-mod, max 16 | ngram-map-k, max 16 |
| --- | ---: | ---: | ---: |
| Korean explanation | 11.85 | 11.79 | 11.36 |
| Python code | 11.92 | 11.80 | 10.85 |
| Structured JSON | 11.84 | 12.01 | 15.77 |
| Explicit copy task | 11.15 | 17.68 | 13.46 |

Values are generation tokens/s. Ngram-mod accepted 0/0, 6/16, 34/106 and 162/331 draft tokens in the listed fixture order; ngram-map-k accepted 2/29, 17/128, 87/120 and 102/234. The copy fixture's ngram-mod end-to-end request time, including prefill, fell from 23.57 to 17.20 seconds. Gains depend on repeated text, and neither method becomes the default. The best result here is below 20 tokens/s; 50 was not achieved.

A separate P620 screen used the existing CUDA build `build-qlion-adaptive`, so only comparisons within that build are valid. At 512 input tokens, two repetitions gave 8.59 generation tokens/s for its CPU-only reference, 7.91 with only the tied token-embedding/output matrix assigned to CUDA, and 7.48 with the last dense layer offloaded while keeping MoE on CPU. Prompt processing was 41.30, 51.84 and 52.78 tokens/s respectively. Output offloading improved prefill but reduced generation throughput, so it was not adopted. These are not measurements of a combined new CPU/CUDA build.

## Gemma4 DFlash and DSpark candidates

At the user's request, two compatible-dimension Gemma4 26B-A4B draft candidates were downloaded as data, without executing repository code. Both declare architecture `dflash` and embedding width 2816. The DFlash GGUF has five layers, block size 16 and extraction inputs [2,7,12,18,23,28]. The DSpark GGUF has five layers, block size 7, extraction inputs [3,10,18,25,28] and `sample_from_anchor=false`, leaving six prediction positions. These dimensions/layers fit the 30-layer target; shared dimensions do not guarantee good acceptance for the Lumen fine-tune and IQ3 quantization.

- [DFlash Q8_0 source](https://huggingface.co/williamliao/gemma-4-26B-A4B-it-DFlash-GGUF), pinned revision `d1800ac59f255542ae096018fa696f03918066a6`, 472,432,704 bytes.
- [DSpark Q4_K_M source](https://huggingface.co/williamliao/dspark_gemma4_26b-a4b-it-GGUF), pinned revision `e35d144b11a585cb63b3ba4020ae747fd912adb8`, 728,663,296 bytes.

SHA-256 matched Hugging Face LFS metadata for each complete file. Exact hashes and URLs are in `build-pulsar-throughput/drafts/provenance.json`; files are `dflash-q8.gguf` and `dspark-q4.gguf` in that directory. The Lucebox candidate was excluded because its declared `gemma4-dflash-draft` architecture requires a different loader. Gemma4 31B and 12B drafts were excluded because they are trained for different targets. Native `draft-dflash` and `draft-dspark` paths are used with conventional bounded-SWA F16 KV; no paged rollback guard is removed.

The user's subsequent MTP request added [RachidAR's Gemma4 26B-A4B assistant Q5_K_M](https://huggingface.co/RachidAR/gemma-4-26b-A4B-it-assistant-gguf), pinned revision `91486a329b67bf1a3c42415727fd3f9344df1a44`, 344,490,080 bytes. Its SHA-256 also matched LFS metadata and is preserved in the same provenance file. The runtime recognized `gemma4-assistant.nextn_predict_layers=4` and used native `draft-mtp`; MTP tensors were not invented or inserted into the target GGUF.

First matched 192-token measurements used 18 target/draft threads, draft maximum 3, minimum 1 and probability threshold 0. These are single-fixture measurements, not a broad benchmark:

| Mode | Korean tokens/s | Python tokens/s | Structured JSON tokens/s |
| --- | ---: | ---: | ---: |
| No speculation | 11.85 | 11.92 | 11.84 |
| DFlash Q8_0, max 3 | 6.83 | 15.66 | 16.98 |
| DSpark Q4_K_M, max 3 | 6.13 | 20.82 | 21.81 |
| MTP Q5_K_M, max 3 | 10.61 | 16.23 | 20.21 |

Accepted/proposed draft-token counts, in the same Korean/Python/JSON order, were DFlash 43/440, 128/187, 134/166; DSpark 13/528, 140/150, 143/144; MTP 96/282, 130/181, 143/144. On this target, low Korean acceptance makes both diffusion drafts slower. MTP is less costly there but still slower than the baseline. DSpark is the strongest measured coding candidate, while MTP also passes 20 tokens/s on the repetitive JSON fixture.

Increasing DSpark maximum to 5 gave 19.71 Python and 22.40 JSON tokens/s, accepting 154/184 and 159/159 drafts. Increasing DFlash maximum to 7 gave 13.29 and 19.20 tokens/s, accepting 147/305 and 161/203. Larger blocks do not automatically improve whole-model speed.

Repeating DSpark maximum 3 gave 21.22 Python and 21.61 JSON tokens/s, with identical text and the same 140/150 and 143/144 acceptance counts. Thus the two-run generation averages were 21.02 and 21.71 tokens/s; end-to-end averages including prefill were about 19.44 and 19.71 output tokens/s. The 20-token/s claim applies to generation on these particular fixtures, not arbitrary language or total request latency. No measured configuration reached 50 tokens/s.

MTP maximum 1 gave 11.91 Korean, 13.62 Python and 14.43 JSON tokens/s, with accepted/proposed counts 70/121, 88/103 and 95/95. It nearly matched non-speculative Korean performance, but maximum 3 was better for Python and JSON. The optional MTP profile therefore starts at 3; set maximum 1 explicitly for a latency-sensitive Korean trial. This small screen is insufficient to automate workload selection.

All three max-3 draft modes reproduced the baseline Python and JSON text exactly. Their Korean outputs diverged after 308 common characters and remained nonempty Korean; generated lengths were 386 versus 380 characters. Batching changes floating-point evaluation and can change subsequent quantization/routing and sampled continuations. Consequently no bitwise or general whole-model equivalence is claimed for speculative versus single-token execution. Target verification remains enabled. The default serving mode stays non-speculative F16; optional profiles expose these task-dependent alternatives.

Speculative decoding is evaluated separately with conventional bounded-SWA F16 KV. The server rejects speculative decoding with paged KV; that guard is retained. Natural language, coding, structured output and a labeled copy fixture must be reported separately. Copy gains do not establish general generation throughput.

## Launcher and reproduction

The launcher selects `build-pulsar-throughput/bin/llama-server`, with paged F16 KV and 18 decode/batch threads. It retains context 8192, one slot, batch 1024, microbatch 512, prompt caching and no context shifting. LES is opt-in. SWA-full is disabled in both CLI validation and the inherited environment. A launcher capture check verified that six inherited generic/domain KV variables are cleared before explicit domain flags are passed; an explicit `--swa-full` request exits with status 2.

```bash
./scripts/serve-pulsar-iq3-temporary.sh
```

To use the optional lower-memory mixed KV configuration, choose a different port while the default server is running:

```bash
PULSAR_IQ3_KV_GLOBAL=iq4_nl PULSAR_IQ3_KV_LOCAL=q8_kv PULSAR_IQ3_PORT=11439 ./scripts/serve-pulsar-iq3-temporary.sh
```

Optional CPU draft profiles automatically choose conventional F16 KV and retain bounded SWA. The launcher rejects explicitly combining a profile with paged KV. Choose one profile and an unused port:

```bash
PULSAR_IQ3_SPEC=dspark PULSAR_IQ3_PORT=11439 ./scripts/serve-pulsar-iq3-temporary.sh
PULSAR_IQ3_SPEC=mtp PULSAR_IQ3_PORT=11439 ./scripts/serve-pulsar-iq3-temporary.sh
PULSAR_IQ3_SPEC=dflash PULSAR_IQ3_PORT=11439 ./scripts/serve-pulsar-iq3-temporary.sh
```

The default draft maximum is 3. Override it with `PULSAR_IQ3_DRAFT_MAX`, or supply a compatible existing draft with `PULSAR_IQ3_DRAFT_MODEL`. The downloaded draft data are in the ignored build directory; the launcher reports a missing file if they are removed. Profiles are opt-in and do not select automatically based on prompt language or content.

Build the candidate without modifying libraries used by a live baseline server:

```bash
cmake -S . -B build-pulsar-throughput -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF
CCACHE_DIR=/tmp/pulsar-iq3-ccache cmake --build build-pulsar-throughput --target llama-server test-cpu-les test-backend-ops -j12
build-pulsar-throughput/bin/test-backend-ops test -b CPU -o 'PAGED_ATTN|TANH'
```

The fixed-token profiler and kernel probes require `LD_LIBRARY_PATH` set to the selected build's `bin` directory. Measurement scripts use the current repository directory, local model path and ports 18138/18139. Run them sequentially to avoid CPU contention. The quality probe reads `/tmp/pulsar-quality-text.txt`; its fixture is preserved as `phase3/quality-text.txt` under the raw-results directory. Restore that file before reproducing the screen. No commit, push or PR is part of this private local work.

## Active service and final verification

The previous baseline process 1129065 was stopped after its executable, model and port were verified. The selected candidate now runs as PID 1253152 at `http://127.0.0.1:11438/v1`, alias `Lumen-3.1-Pulsar-IQ3-Quality`. Its inspected process arguments select paged global/local F16 KV, 18 threads, no LES and no SWA-full; its environment explicitly sets `LLAMA_ARG_SWA_FULL=0`. Full-attention pages are 512 and bounded SWA pages are 98. The F16 pool remains 466.26 MiB. Observed RSS after validation was about 16.8 GiB and P620 usage was 7 MiB; these are point observations, not peak values.

Final API checks passed health, model alias, arithmetic `42`, a Korean sentence, Python code and OpenAI streaming termination. A repeated 128-token input reused 127 tokens; the 1536-token repeat reused 1535. After slot replacement with a disjoint prompt, the uncached request evaluated all 1536 tokens, and its four generated tokens matched the cached request. The final uncached API check overlapped the small CTest rerun and is a functional check, not an additional timing benchmark. This checks one bounded-window reuse scenario, not arbitrary-history equivalence.

A final CTest rerun passed 11/11; the earlier candidate operator runs passed 230 paged/TANH, 121 Q4 MUL_MAT and 87 Q4 MUL_MAT_ID cases. `bash -n`, launcher flag captures for all three draft profiles, SWA-full rejection, paged-plus-draft rejection and `git diff --check` passed. The candidate source review found no blocking issue in the inspected throughput changes; no portability/performance claim is made for untested machines. Full-model speculative outputs have the numerical limitation stated above.

Current process, runtime and API artifacts are `build-pulsar-throughput/results/serving.pid`, `serving.log`, `restart.json`, `serving-runtime.json`, `serving-check.json` and `serving-check.log`. Stop this verified process before replacing the default server:

```bash
kill "$(cat build-pulsar-throughput/results/serving.pid)"
```

The default delivers about 11.26 generation tokens/s at the measured short context, with task-specific DSpark generation exceeding 20 tokens/s on coding and repetitive JSON. MTP also exceeded 20 on the JSON fixture. Broad 20-token/s language generation and any 50-token/s result remain unachieved on this hardware/model combination.
