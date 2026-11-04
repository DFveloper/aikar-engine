# QLion2 P620: accumulation, routing, scratch, and frozen statistics

Date: 2026-11-04. This extends [the register study](2026-11-04-qlion2-registers.md). The baseline is that working tree, including its 39-register direct GEMM apply kernel, on top of `95fe8c76a`. These results isolate the subsequent changes; they are not comparisons against upstream llama.cpp.

## Implementation

1. Parallel Q8 packing: dense, sparse-row, and tied accumulators use the existing signed warp maximum and one quantized output per lane. All-zero blocks preserve the original negative-zero scale bits. No floating-point sum order is changed.
2. Stable routing: empty experts return immediately. Eight warp ballots replace the block-wide scan, and a block stops after finding its final route. Input order and route ranks are preserved. The original scan remains the fallback for physical warps other than 32 lanes. Worst-case search work is still proportional to active experts times routes; this is not a general active-only routing algorithm.
3. Short-route path: SM61 updates with at most four total routes reduce directly from the original inputs and apply the update in the same kernel. They do not allocate padded gather matrices, a full F32 expert gradient, or a cuBLAS workspace. Larger route lists retain the existing GEMM path. At 1408 columns, 2816 rows, and an eight-expert workspace, the three avoided scratch buffers total 154 MiB; this is a layout calculation, not an independently sampled peak-VRAM result.
4. Frozen statistics: QLion2 requests an internal threshold-only form of compact statistics. Warmup still computes the original RMS and updates the original state. Once the device-resident sample count reaches the warmup length, CPU/CUDA copy the live parameters and frozen threshold without scanning momentum. The unused RMS slot is zero in this internal mode; the public constructor still returns full statistics by default. Checking device state preserves resumed and reused graphs without host reads.
5. Cooperative Q8 reset copy: each lane copies its own quantized byte; lane zero copies the scale. This reduces the Q8 accumulation specialization from 32 to 15 registers. Both fields are copied without converting their representation.

## Rejected candidates

- Splitting the GEMM into 128-row tiles changed intermediate F32 values in some shapes: for example, 232 of 37,152 outputs at 96 x 129 with 32 nonzero K terms. Large 2816-row GEMMs also became slower in this probe. This path was not added.
- A general scalar direct reduction was too slow for long route lists. The probe reached about 89.8 ms versus 5.16 ms for a 1408 x 2816, K=256 GEMM. An eight-total-route update also slightly regressed for a large Q4_0 case. The installed cutoff is therefore four total routes on SM61.

## Isolated measurements

Quadro P620, driver 580.178.04, CUDA 12.4.131, release SM61 build. Each table entry is the median of three rounds, with the baseline/candidate order alternated. CUDA events measure device execution. The route, statistics, and routed update harnesses exercise CUDA Graph capture explicitly; production P620 training keeps the existing backend graph policy.

| Experiment | Case | Before (ms) | After (ms) | After/before |
| --- | --- | ---: | ---: | ---: |
| acc | Q8 accumulation, 1,048,576 blocks, F32 gradient | 21.384480 | 4.886160 | 0.2285 |
| route | Route builder, 256 experts, 2048 tokens, 8 routes/token | 2.314227 | 0.970233 | 0.4192 |
| host | Full routed update, Q4_0, 1408 x 2816, 2 total routes | 14.045286 | 9.864806 | 0.7024 |
| stats | Frozen compact statistics, 1,048,576 blocks | 12.926694 | 0.004915 | 0.0004 |
| copy | Q8 reset copy, 1,048,576 blocks | 31.957125 | 2.889204 | 0.0904 |

The isolated experiments compare each individual change with its immediate baseline. Their speedups cannot be multiplied to predict total training speed. Small and unchanged copy cases remain near launch or memory-bandwidth limits. In particular, routing uses 30 registers versus 29 before, while sparse-row packing uses 23 versus 22; the selection criterion is measured time plus exact output preservation, not register count alone.

## Correctness and repository checks

- Accumulation and cooperative copy: each of 12 output buffers matches baseline bytes, including nonfinite F32 inputs, zero blocks, signed zeros, reset/non-reset, and a final partial warp group. Three repeated rounds also compare SHA256 hashes.
- Routing: 36 strided-input fixtures check offsets, sorted routes, and route ranks directly against a CPU builder. They include invalid expert IDs, duplicate routes, empty experts, partial tiles, and CUDA Graph capture.
- Routed update: 60 configurations compare all weight, momentum, and residual tensor hashes after ten updates, across Q4_0/Q8_0/MXFP4, broadcast/non-broadcast inputs, small/large shapes, and both sides of the route cutoff.
- Frozen statistics: warmup outputs, frozen thresholds, and state hashes match the original path. The new unused-RMS value is checked separately. Existing trajectory tests compare fused/non-fused updates across warmup and frozen steps.
- `QAT_TEST_BACKEND=CUDA test-qat` passes the CPU and CUDA suites. The routed fixture now uses 11 experts and tests direct and GEMM paths; frozen-state cases are also added to `test-backend-ops`.
- `test-opt` passes. CUDA backend checks pass 14/14 supported cases; six unsupported dense fused accumulation cases are skipped. CPU/CUDA support CSVs include the four new frozen-threshold cases.

## End-to-end training and profiling

### Matched runs

Seven settings use the same seed (1234), Q4_0/Q8_0 selected weights, context/batch/microbatch 256, one parallel sequence, all layers offloaded to CUDA0, four CPU threads, two dataset threads, constant learning rate 0.001, and weight decay 0.01. Native final models and a last-step checkpoint with optimizer state are saved. Accumulation=4 uses four repetitions of the 32-record synthetic dataset. The dense/MoE fixtures have approximately 7M parameters; the larger Gemma 4 fixture has 200,402,386 parameters.

Warmup rows use threshold warmup=2 and two complete epochs. The fixed-tau controls use the default tau=0.001, no threshold warmup, and one complete epoch. These settings have different work counts; compare baseline/candidate within a row, not absolute times between rows.

Each row is the median of three complete runs per variant. Variant order is reversed in the middle round. Wall time includes startup, model/data loading, all training, last-checkpoint serialization, and final-model serialization; output hashing is outside the measured interval. Steady step time excludes the first two optimizer updates. Input files were already cached by calibration. NVML samples whole-device memory every 10 ms. These 42 successful runs are not profiled.

| Setting | Epochs | Updates | Wall before (s) | Wall after (s) | Wall reduction | Step before (s) | Step after (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7M dense, accum=1, warmup=2 | 2 | 12 | 1.036 | 1.062 | -2.6% | 0.038405 | 0.035393 |
| 7M MoE, accum=1, warmup=2 | 2 | 12 | 1.459 | 1.371 | 6.0% | 0.063936 | 0.061407 |
| 7M MoE, accum=4, warmup=2 | 2 | 12 | 5.796 | 5.465 | 5.7% | 0.412864 | 0.383827 |
| 200M, accum=1, warmup=2 | 2 | 12 | 8.677 | 8.135 | 6.2% | 0.613944 | 0.548972 |
| 200M, accum=4, warmup=2 | 2 | 14 | 45.241 | 38.704 | 14.4% | 3.209535 | 2.722767 |
| 200M, accum=1, fixed tau | 1 | 6 | 4.527 | 4.585 | -1.3% | 0.548573 | 0.547295 |
| 200M, accum=4, fixed tau | 1 | 7 | 22.898 | 19.959 | 12.8% | 3.135129 | 2.731959 |

Negative wall reduction means a slower median. Small dense and fixed-tau accumulation=1 wall ranges overlap between variants; no general end-to-end gain is claimed for those settings. Fixed-tau accumulation=1 has essentially unchanged steady step time, because it uses neither gradient accumulation nor warmup statistics. The short-route path is not selected by these normal 256-token MoE windows.

All 21 matched pairs have identical model tensor shapes/types/bytes, optimizer tensor shapes/types/bytes, logged epoch losses, logical steps, and supervised-token counts. Each 200M checkpoint comparison covers 252 model tensors and 696 optimizer tensors. Generated GGUF dumps are removed after hashes are recorded to bound storage; commands, hashes, timings, and logs remain. One earlier run failed checkpoint serialization at the session write quota and is excluded; its log is retained under `train/invalid/checkpoint-write`.

### Memory

| Setting | Host RSS before/after (MiB) | Sampled device peak before/after (MiB) |
| --- | ---: | ---: |
| 200M, accum=1, warmup=2 | 388.8 / 389.0 | 1142.1 / 1142.1 |
| 200M, accum=4, warmup=2 | 389.7 / 390.8 | 1392.1 / 1392.1 |
| 200M, accum=1, fixed tau | 388.8 / 388.6 | 1142.1 / 1142.1 |
| 200M, accum=4, fixed tau | 389.4 / 392.4 | 1392.1 / 1392.1 |

No peak device-memory reduction is observed in these normal-window runs. The short-route scratch bypass has separate isolated coverage; it is not used by these windows.

### GPU profiles

Nsight Systems fails to create its process-waiter semaphore in this environment. `nvprof` collects CUDA activity/API traces and SQLite exports instead. An external interposer starts collection after optimizer step 2 and stops after step 4, including the intervening input staging. Each profile therefore covers two contiguous steady optimizer updates; accumulation=4 includes eight microbatch evaluations. Instrumented wall times are not used as performance results. Six profiled runs also produce model tensors identical to their unprofiled counterparts.

| Workload | GPU activity sum before (ms) | After (ms) | Activity calls before/after |
| --- | ---: | ---: | ---: |
| 7M MoE, accum=1, warmup=2 | 84.591 | 80.020 | 3528 / 3528 |
| 200M, accum=1, warmup=2 | 1115.614 | 979.582 | 7824 / 7824 |
| 200M, accum=4, warmup=2 | 6022.243 | 5070.874 | 32228 / 32228 |

For the 200M accumulation=4 profile:

| Component, two optimizer updates | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Q8 accumulation | 1045.657 | 237.556 |
| QLion statistics | 138.108 | 1.947 |
| GEMM | 1551.626 | 1543.689 |
| Outer-product preparation | 465.293 | 464.953 |
| QLion update | 180.451 | 180.252 |
| cudaStreamSynchronize, 2699 / 2699 calls | 5767.620 | 4836.460 |
| cudaLaunchKernel, 30448 / 30448 calls | 183.509 | 171.853 |

The training trace confirms 32 -> 15 registers for the Q8 accumulator, with 1,856 calls in each variant. Launch counts are unchanged: these changes shorten existing work rather than reducing launch count. Activity durations and API durations can overlap and must not be added to obtain wall time. Profiler start/stop and its explicit synchronization are instrumentation, not application work.

The largest remaining single GPU kernel is F32-to-Q8 gradient conversion: 1,443.048 ms, about 28.5% of the candidate activity sum in this profile. SGEMM variants and outer-product preparation remain substantial. A future conversion/accumulation fusion must preserve the current intermediate Q8 rounding; simply skipping that quantization would change the optimizer trajectory.

### CPU profiles

GNU gprofng samples CPU clocks every 2 ms with synchronization/I/O tracing. Reports select timestamps 5-35 seconds, a 30-second training interval before model saving in both variants. This is an equal wall-time slice, not equal work: the faster candidate completes more work in it. The selected experiments and filtered function reports are under `train/cpu/200m_a4`. Both instrumented runs match their normal model tensors.

Approximately 84.3% of baseline and 85.4% of candidate exclusive CPU samples are attributed to an unresolved static function in libgomp. CUDA runtime work and CPU `ggml_compute_forward_acc` / `ggml_compute_forward_rms_norm_back` also appear. This identifies OpenMP runtime activity as a follow-up investigation, not proof that 85% of wall time can be removed. Runtime symbols need better resolution and wait-policy/thread-count controls before changing those settings.

Full CPU tracing greatly increases save-time overhead, so its wall time is excluded. A separate signal-paused experiment produced no CPU samples and is not used; its files are retained under `train/cpu-paused-empty`. GPU/kernel and CPU profiler data are diagnostic only.

### Reproduction and evidence

```sh
CCACHE_DISABLE=1 cmake --build /tmp/qlion-p620-build --target llama-finetune-qlion test-qat test-opt test-backend-ops -j 4
QAT_TEST_BACKEND=CUDA /tmp/qlion-p620-build/bin/test-qat
/tmp/qlion-p620-build/bin/test-backend-ops test -b CUDA0 -o ACC_QLION_QAT_ID,ACC_QLION_QAT_EXPERTS,QLION_QAT_STATS
ctest --test-dir /tmp/qlion-p620-build -R '^test-opt$' --output-on-failure
```

The local baseline build workflow is stored at `/tmp/qlion-craftsman/baseline.py`; it compiles/relinks only the changed baseline objects. `train/run.py` records exact commands, resource reports, and tensor/file hashes. `train/analyze.py` imports nvprof traces and produces filtered CPU reports. Reproduction uses new output directories because existing profiler experiments are intentionally not overwritten.

Final checks: QAT CPU/CUDA suites pass; 14/14 supported CUDA backend cases pass, with six unsupported variants skipped; `test-opt` passes; source inputs and variant binaries retain their recorded SHA256 hashes. `train/verified.json`, `train/timing-summary.json`, `train/comparisons.json`, and `train/profile-summary.json` summarize the evidence.

## Artifacts and limits

Local harnesses, frozen source copies, compiler logs, raw isolated timings, and `summary.json` are in `/tmp/qlion-craftsman`. Matched runtime variants, input/binary hashes, commands, per-run results, profiler outputs, and training summaries are under `/tmp/qlion-craftsman/train`. Baseline GGML libraries are relinked with the frozen CUDA update/routing source and pre-change `ggml-opt.cpp`/CPU statistics; all other objects and trainer libraries are shared with the candidate build. Explicit library paths and `ldd` records establish variant selection.

Only P620 execution is validated. The workloads are performance and correctness fixtures, not evidence of training quality or speed on the 26B production model.
