# Lumen CPU kernel optimization measurements

핵심 결과: IQ3 작업 분배와 paged attention의 SIMD 누적을 개선했다. 반복 측정에서 입력 처리량은 512토큰 +9.53%, 4096토큰 +14.23%였고, 단일 요청 디코딩 속도는 사실상 같았다. 실제 HTTP 응답 시간은 중앙값 기준 5.20% 줄었다. 두 고정 입력의 logits와 HTTP 출력은 기준과 일치했다. MTP, 모델 재양자화, expert 생략 없이 얻은 결과이며 운영 서버에는 배포하지 않았다. vllm.cpp의 관련 CPU 기법 대부분은 이 포크에 이미 존재해, 별도 런타임을 가져오기보다 기존 커널을 재사용했다.

Date: 2026-10-24. Workload: Lumen-3.1-Pulsar-IQ3-Quality on Xeon E5-2696 v3, 18 CPU threads, AVX2, F16 paged KV. This experiment follows `lumen-no-spec-optimization-audit-20261024.md`.

## Scope

The changes preserve the model file, expert selection algorithm, KV precision, context semantics, and ordinary autoregressive decoding. MTP, speculative decoding, expert pruning, requantization, and shorter contexts are excluded. The production server remains on its original executable at port 11438. Candidate builds and HTTP measurements use a separate directory and port 18149.

The input worktree already contained substantial server and CPU changes. The baseline includes those changes. `build-lumen-kernels/results/input-worktree.patch` records the starting tracked diff; `*.input.cpp` snapshots separate this experiment from earlier work. Immutable baseline libraries are in `build-lumen-kernels/baseline/bin`.

## Kernel changes

### Tiled MoE output ownership

Previously, 1408 output rows became 22 groups of 64. With 18 workers, four workers received two groups while the others received one. The candidate partitions the 88 existing 16-row microtiles between workers, then processes each worker's range in windows of at most 64 rows. Workers receive 64 or 80 rows for this shape. The expert loop, activation quantization, weight representation, and per-output K accumulation order remain the same.

This change targets routed batches that enter the tiled kernel. It does not accelerate the single-token generic IQ3 dot product. Costs include smaller partial windows and potentially repeated activation unpacking; measurements must cover both IQ3 formats and routed batch sizes.

### Paged-attention PV accumulation

The candidate keeps the existing QK dot products, masks, page decoding, and online softmax. It collects each query/head row's weights for a KV tile and uses the existing `simd_gemm` to accumulate `weights * values` for multiple rows. On AVX2 the existing microkernel keeps six output rows in registers and reuses value vectors. Single-row jobs retain the previous vector MAD path. The extra weights matrix is bounded at 1024 floats per worker; no new allocator or external dependency is introduced.

Zero weights need special care: the original code skips their products. A direct GEMM can produce NaN from `0 * Inf` even when that value should contribute nothing. A dedicated regression reproduced this difference. The final candidate checks decoded value tiles for nonfinite values and falls back to the previous skipped-zero accumulation for those tiles.

## vllm.cpp assessment

Reference inspected: [localai-org/vllm.cpp, commit 7f842ee9b3c7e9b651283de48f42cd76a0ad5d30](https://github.com/localai-org/vllm.cpp/tree/7f842ee9b3c7e9b651283de48f42cd76a0ad5d30). The repository was inspected as source; its marketing benchmarks were not used as evidence for this host.

| Technique | Source | Assessment for this server |
| --- | --- | --- |
| Quantize activations once, compute against quantized weights, partition output rows | [cpu_quant_gemm.cpp](https://github.com/localai-org/vllm.cpp/blob/7f842ee9b3c7e9b651283de48f42cd76a0ad5d30/src/vt/cpu/cpu_quant_gemm.cpp#L1) | Already present in this fork. The reference itself describes its llama.cpp origin. Fine-grained output ownership is compatible with this approach. |
| Repacked quantized GEMM dispatch | [cpu_quant_gemm.cpp](https://github.com/localai-org/vllm.cpp/blob/7f842ee9b3c7e9b651283de48f42cd76a0ad5d30/src/vt/cpu/cpu_quant_gemm.cpp#L153) | Existing AVX2 Q4 repack and tiled IQ3 already cover the relevant path. Copying a second dispatch subsystem has no demonstrated benefit. |
| Hoist KV dtype decisions and widen query rows once | [cpu_paged_attn.cpp](https://github.com/localai-org/vllm.cpp/blob/7f842ee9b3c7e9b651283de48f42cd76a0ad5d30/src/vt/cpu/cpu_paged_attn.cpp#L48) | The local query is already F32 and cached K/V are decoded by row, outside the element reduction. The main remaining reuse opportunity is multirow PV accumulation, implemented with this fork's existing SIMD GEMM. |
| CPU fused SwiGLU reference | [cpu_quant_gemm.cpp](https://github.com/localai-org/vllm.cpp/blob/7f842ee9b3c7e9b651283de48f42cd76a0ad5d30/src/vt/cpu/cpu_quant_gemm.cpp#L275) | The inspected CPU implementation uses two grouped GEMMs and temporary vectors. It is not evidence of a better fused AVX2 kernel for this model's GELU path. |
| ARM NEON/BF16 dot kernels | [cpu_paged_attn.cpp](https://github.com/localai-org/vllm.cpp/blob/7f842ee9b3c7e9b651283de48f42cd76a0ad5d30/src/vt/cpu/cpu_paged_attn.cpp#L40) | This host is AVX2; ARM instructions cannot be adopted directly. |

## Measurement protocol

Release native CPU builds use the same source baseline and compiler. Model timing disables profiling callbacks, uses fixed input token IDs and 64 fixed teacher-forced decode tokens, and runs 512/4096 input tokens with microbatch 512. Full logits are captured separately through the ordinary logits access path. The captured arrays allow direct numerical comparison but do not constitute a general model-quality benchmark.

Sequential baseline/candidate measurements reverse order on the second repeat. The 512-token workload has three final paired repeats and the 4096-token workload two. Builds and operator tests finish before these measurements. Early measurements that overlapped compilation or other tests are excluded. The live production service stays available, so host background activity remains a source of noise.

Operator timing uses the existing backend benchmark harness and production-shape MoE tests. Correctness uses existing tiled/reference checks and the independent double-precision paged-attention oracle. Tests include 3/18-thread MoE tails and 8/9/17 query attention tails, 70-element head tails, causal/noncausal windows, missing pages, and existing cache encodings.

## Results

### Fixed-token model timing

Values are median tokens/s; three paired runs for 512 tokens and two for 4096. Decode uses 64 fixed tokens after each prefill.

| Workload | Baseline | Candidate | Throughput change |
| --- | ---: | ---: | ---: |
| 512-token prefill | 71.6415 | 78.4700 | +9.53% |
| Decode after 512 tokens | 11.2508 | 11.2529 | +0.02% |
| 4096-token prefill | 60.5198 | 69.1308 | +14.23% |
| Decode after 4096 tokens | 10.06795 | 10.04185 | -0.26% |

The improvement is in prefill. These measurements do not establish a single-request decode improvement. Short prefill samples are 71.6415/71.0463/71.6747 before and 79.0291/76.6465/78.4700 after; long samples are 60.6652/60.3744 before and 69.1152/69.1464 after.

For each input depth, the final comparison captures all 262144 vocabulary logits at the final prefill position and all 64 decode positions: 17,039,360 floats per variant. Both depths match the baseline bit for bit, with maximum absolute error 0. This is strong evidence for these fixtures, not a proof for arbitrary inputs or a broad task-quality evaluation.

### Attention operator timing

The existing harness averages hundreds of graph executions for each fixture. These are single harness measurements per shape, not a confidence interval. The table includes the final finite-value scan.

| Head dimension / KV heads / query tokens | Baseline us | Candidate us | Latency reduction |
| --- | ---: | ---: | ---: |
| 512 / 2 / 8 | 9112.69 | 7953.64 | 12.72% |
| 256 / 8 / 8 | 1204.51 | 1290.75 | -7.16% |
| 512 / 2 / 128 | 39697.05 | 32646.39 | 17.76% |
| 256 / 8 / 128 | 8877.47 | 8718.89 | 1.79% |

The global-attention shapes benefit most. The small local-attention fixture regresses; the change is not a universal operator speedup. Full-model prefill still improves in both tested lengths. Further dispatch tuning needs repeated shape-level measurements before claiming that regression is resolved. The harness's printed GB/s is a logical tensor-size calculation, not measured DRAM bandwidth.

### MoE operator timing

Two benchmark invocations per variant, reversing order on the second pair, provide 18 timings per cell. Each fixture uses K=2816, R=1408, two experts selected for every token, and 18 threads. Median latency follows; these cache-resident two-expert fixtures do not reproduce the full 128-expert model's memory traffic.

| Format / routed rows per expert | Baseline us | Candidate us | Latency reduction |
| --- | ---: | ---: | ---: |
| IQ3_XXS / 8 | 1018.0 | 1056.5 | -3.78% |
| IQ3_XXS / 32 | 1786.0 | 1882.5 | -5.40% |
| IQ3_S / 8 | 2156.5 | 945.5 | 56.15% |
| IQ3_S / 32 | 3367.0 | 1551.5 | 53.92% |

The finer ownership is particularly beneficial for the tested IQ3_S fixtures, while IQ3_XXS fixtures regress. Early exploratory runs were noisy and are not the values in this table. The retained change is justified by the repeated full-model prefill measurements, not by an assertion that every quantization format accelerates. A larger MoE kernel-only gain must not be advertised as the server's overall gain.

### HTTP and final verification

Three sequential HTTP requests per variant use two natural-language fixtures in A/B/A order, each exactly 512 input tokens and 64 output tokens. Prompt caching is disabled in the request and server. All six responses report `cache_n=0` and `prompt_n=512`.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Median full HTTP response time | 12.8228 s | 12.1563 s |

This is a 5.20% response-time reduction for this workload. Corresponding response contents match for all three requests. This is a small HTTP screen with baseline measured first; the reversed-order fixed-token experiment provides the stronger repeated performance evidence.

Verification:

- Initial batched-PV candidate: 195/195 paged-attention cases passed the existing independent oracle across F16 and quantized KV encodings. This run preceded the nonfinite-value fallback.
- Final candidate after fallback: 21/21 targeted paged-attention cases passed, including all added tails and missing-page cases.
- Final CTest: tiled matmul, its `OMP_THREAD_LIMIT=7` variant, and the new analytic zero-weight/nonfinite regression passed, 3/3 tests.
- The new regression first passed on the immutable baseline, failed on direct-GEMM PV with NaN output, then passed after the skipped-zero fallback. It has its own small test because the generic scalar online attention reference does not preserve this zero-weight edge case.
- Both fixed-token logits comparisons are bitwise equal, as detailed above.
- SHA256 checks confirm the production server, `libllama.so`, and `libggml-cpu.so` are unchanged. The original PID 1253152 remains healthy at port 11438; the experiment port is closed. Comparison of tracked diff sections confirms unrelated pre-existing edits are preserved.

Raw commands, probes, logs, logits, and JSON results are retained in `build-lumen-kernels/results`. `summary.json` contains the model samples and numerical comparisons. `verification.json` records production checksums, process identity, health, and preservation of unrelated tracked changes.

## Reproduce

```sh
cmake -S . -B build-lumen-kernels -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-lumen-kernels --target llama-server test-tiled-mulmat test-backend-ops test-paged-attn-zero -j12
ctest --test-dir build-lumen-kernels -R 'test-tiled-mulmat|test-paged-attn-zero' --output-on-failure
build-lumen-kernels/bin/test-backend-ops test -b CPU -o PAGED_ATTN -p 'dim=70|missing_pages=1|cache_type=f16.*single_seq=1'
build-lumen-kernels/bin/test-tiled-mulmat --bench-moe
build-lumen-kernels/bin/test-backend-ops perf -b CPU -o PAGED_ATTN -p 'cache_type=f16.*single_seq=1'
python3 build-lumen-kernels/results/measure_models.py
python3 build-lumen-kernels/results/http_kernels.py
```

The retained baseline directory must be preserved for the Python A/B scripts. Do not rebuild the production directory or run competing CPU benchmarks while measuring.

## Conclusion and limits

The retained kernel changes improve this model's measured prefill throughput by approximately 10-14%, with matching logits on both fixed-token fixtures and matching text on the HTTP fixtures. They do not produce a substantial single-request decode acceleration. Some small operator shapes regress, so these results should not be generalized to every CPU, thread count, or model.

vllm.cpp offers useful principles, but most inspected CPU quantization/repacking techniques are already present in this fork. This experiment reuses the local SIMD GEMM and scheduler structure rather than importing a second runtime. A large further decode improvement still requires reducing the profiled dense Q4/Q8 and IQ3 work or improving its memory reuse; prefill-only tiling cannot supply that result. Production deployment remains a separate action; its executable and loaded libraries are unchanged.
