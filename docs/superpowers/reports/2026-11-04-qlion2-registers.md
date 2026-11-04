# QLion2 P620 register reduction

Date: 2026-11-04. Baseline: `95fe8c76a`. This is a bounded optimization of `opt_step_qlion_qat_id_apply_gemm`, the direct routed MoE update used by QLion/QLion2. It does not measure the compact Q8 gradient accumulation path or total training throughput.

## Change

- Assign one expert to each CUDA grid row, removing expert-index division and remainder from the kernel.
- Read the GEMM gradient with a linear block index. Compute row and column indices only when routes exceed the GEMM cap.
- Use I32 for route IDs and their divisors; keep tensor offsets and products in I64. Validate route dimensions before multiplying them or narrowing them.
- Extend the existing routed update test from 3 to 11 experts. This exercises a second expert batch, a partial expert batch, empty experts, duplicate routes, and partial warp groups for each weight format.

## Environment and method

Quadro P620 (SM61, 2 GiB), driver 580.178.04, CUDA 12.4.131. Release compiler flags match `/tmp/qlion-p620-build`: `-O3 -DNDEBUG -std=c++17 -use_fast_math`, compute/sm 61. A sampled busy GPU reported 1354 MHz SM and 2504 MHz memory clocks.

The local harness includes a frozen baseline source and the candidate source in separate executables. Both use the same flags and deterministic inputs. CUDA events measure only the apply kernel, after ten warmup updates. Each timed batch contains 100 updates without overflow or 30 with overflow. The baseline and candidate alternate order over three rounds; the table reports the median batch average. No other GPU experiment ran concurrently.

Synthetic precomputed F32 gradients are applied to five experts beginning at expert index 2. Shapes cover 256/1408 columns and 129/2816 rows, route counts 0/256/288, cap 256, and fast state scale off/on. The gradient GEMM, routing, transfers, and file output are outside the timed region.

## Compiler results

| Weight type | Before registers/thread | After registers/thread | Spill loads/stores |
| --- | ---: | ---: | --- |
| Q4_0 | 48 | 39 | 0 / 0 bytes |
| Q8_0 | 48 | 39 | 0 / 0 bytes |
| MXFP4 | 48 | 39 | 0 / 0 bytes |

These counts come from the full source translation unit compiled with `-Xptxas=-v`. No register limit is imposed.

## Kernel timing

Fast state scale is off in this table. Rows show the full GEMM cap and a 32-route overflow. All 24 measured combinations improved; candidate/baseline ratios range from 0.750 to 0.840.

| Columns x rows | Routes/expert | Before (ms) | After (ms) | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 256 x 129 | 256 | 0.1366 | 0.1044 | 23.6% |
| 256 x 129 | 288 | 1.2645 | 1.0627 | 16.0% |
| 256 x 2816 | 256 | 2.8742 | 2.1793 | 24.2% |
| 256 x 2816 | 288 | 27.3654 | 22.9607 | 16.1% |
| 1408 x 129 | 256 | 0.7280 | 0.5527 | 24.1% |
| 1408 x 129 | 288 | 6.9098 | 5.7921 | 16.2% |
| 1408 x 2816 | 256 | 15.7868 | 11.9684 | 24.2% |
| 1408 x 2816 | 288 | 150.5568 | 126.1642 | 16.2% |

## Correctness and validation

- Q4_0: all 72 SHA256 hashes of weight, momentum, and residual outputs match baseline after ten updates across 24 shape/route/fast-scale combinations.
- Edge fixtures: 108 output files match byte-for-byte across Q4_0/Q8_0/MXFP4, broadcast and non-broadcast activations with three used routes, route counts 0/256/288, and fast state scale off/on. These use 96 columns and five rows to exercise partial warp groups.

Project checks:

```sh
CCACHE_DISABLE=1 cmake --build /tmp/qlion-p620-build --target test-qat test-backend-ops -j 4
QAT_TEST_BACKEND=CUDA /tmp/qlion-p620-build/bin/test-qat
/tmp/qlion-p620-build/bin/test-backend-ops test -b CUDA0 -o ACC_QLION_QAT_ID,ACC_QLION_QAT_EXPERTS,QLION_QAT_STATS
```

- Build: passed for `test-qat` and `test-backend-ops` using the current source. Log: `/tmp/qlion-register-study/build-final.log`.
- `QAT_TEST_BACKEND=CUDA test-qat`: passed the full CPU suite and CUDA0 suite, including routed updates for all three weight formats with the expanded 11-expert fixture. Log: `/tmp/qlion-register-study/test-qat.log`.
- Backend operations: 10/10 supported CUDA cases passed against the CPU reference. Six `ACC_QLION_QAT_ID` cases were reported unsupported and skipped; this command does not claim coverage for them. Log: `/tmp/qlion-register-study/test-backend.log`.
- `git diff --check`: passed. Added source lines are ASCII and contain no trailing whitespace.

## Local artifacts

Harnesses, frozen baseline, compiler logs, raw timings, and `summary.json` are in `/tmp/qlion-register-study`. `bench.cu` and `edge.cu` include the selected source via `KERNEL_SOURCE`; `compile.py` takes flags from the existing CMake build. `measure.py` performs output hash verification and the alternating benchmark. Binary dumps from the large fixtures were removed after hashing to stay within the writable storage quota; the small edge dumps remain.

Full training step time, memory usage, training quality, and GPUs other than P620 have not been measured for this change. The speedups above apply only to this direct routed apply kernel.
