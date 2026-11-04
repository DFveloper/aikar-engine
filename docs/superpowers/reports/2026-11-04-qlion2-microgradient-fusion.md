# QLion2 P620: fuse microgradient quantization and accumulation

Date: 2026-11-04. The baseline includes all five changes in the [previous optimization study](2026-11-04-qlion2-craftsman.md), including the 15-register Q8 accumulator. This report measures the additional change, not a comparison against upstream or the original register-study baseline.

## Implementation

The previous dense accumulation graph materialized a Q8 microgradient with F32-to-Q8 CPY, then read it in ACC_QLION_QAT. CUDA now quantizes the F32 gradient inside the accumulator kernel, removing that intermediate tensor and its separate kernel launch. Each warp handles one 32-value block and each lane writes one output byte.

The two quantization stages remain separate calculations. The intermediate signed maximum, division by -128, roundf, byte clamping, and FP16 scale rounding match the existing CUDA CPY quantizer. Non-reset updates dequantize that rounded intermediate before adding the existing accumulator and quantizing again. Reset updates store the intermediate bytes directly. Replacing this with a single quantization would change training results and is not done.

The private ACC_QLION_QAT parameter op_params[2]=1 selects this behavior. Its public constructor and default F32 accumulation semantics remain unchanged. CPU has a reference implementation that calls the existing CPU Q8 quantizer and Q8 accumulation helper. Automatic selection requires a contiguous F32 gradient, a GPU accumulator, and device support. CPU accumulator placement retains the compressed-transfer path; HIP, MUSA, and Vulkan reject the new mode. The GPU device is identified once when allocating accumulator state, rather than querying its type for each gradient callback. Vulkan was not built or run in this study.

## Isolated measurements

Quadro P620, SM61, CUDA 12.4.131, driver 580.178.04, Release build. The harness measures the existing library CPY plus Q8 accumulator against the fused kernel with CUDA events, including CUDA Graph capture. Entries are medians of three alternating rounds after ten warmups. All output buffers match byte for byte.

| Q8 blocks | Reset | Separate (ms) | Fused (ms) |
| ---: | ---: | ---: | ---: |
| 7 | No | 0.012288 | 0.002099 |
| 7 | Yes | 0.011960 | 0.001747 |
| 32,768 | No | 1.241528 | 0.211702 |
| 32,768 | Yes | 1.177261 | 0.139448 |
| 1,048,576 | No | 28.478251 | 6.686085 |
| 1,048,576 | Yes | 26.404812 | 4.408166 |

The fused specialization uses 16 registers with zero spills. The existing Q8 accumulator uses 15. The benefit comes from removing work and memory traffic, not reducing this kernel's register count.

## Matched complete training

All cases use one full epoch, seed 1234, context/batch/microbatch 256, one parallel sequence, Q4_0/Q8_0 selected weights, CUDA0 with all layers offloaded, four CPU threads, two dataset threads, constant learning rate 0.001, and weight decay 0.01. The MoE fixture has approximately 7M parameters; the larger Gemma 4 fixture has 200,402,386 parameters. Accumulation=4 repeats the 32-record synthetic dataset four times. Case names use a4/a1 for accumulation=4/1, fixed for threshold warmup=0, and cpu for CPU accumulator placement. Other rows use threshold warmup=2.

Each setting runs three matched pairs with reversed variant order in the middle round. Wall time includes loading, training, the last-step optimizer checkpoint, and final-model serialization. Hashing is excluded. Steady step time excludes the first two logical updates. NVML samples whole-device memory every 10 ms. Normal timing runs are not profiled.

| Setting | Wall before (s) | Wall after (s) | Wall reduction | Step before (s) | Step after (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| moe_a4_fixed | 3.139 | 2.957 | 5.8% | 0.394857 | 0.360655 |
| 200m_a4 | 20.197 | 15.558 | 23.0% | 2.721270 | 2.043818 |
| 200m_a4_fixed | 20.016 | 15.338 | 23.4% | 2.732482 | 2.037093 |
| 200m_a1_fixed | 4.514 | 4.477 | 0.8% | 0.549230 | 0.544483 |
| moe_a4_cpu | 2.951 | 3.082 | -4.4% | 0.379843 | 0.390408 |

| Setting | Wall range before (s) | Wall range after (s) |
| --- | ---: | ---: |
| moe_a4_fixed | 3.117-3.293 | 2.926-2.978 |
| 200m_a4 | 19.929-20.210 | 15.510-15.568 |
| 200m_a4_fixed | 19.973-20.028 | 15.320-15.409 |
| 200m_a1_fixed | 4.508-4.530 | 4.428-4.522 |
| moe_a4_cpu | 2.909-3.293 | 2.959-3.138 |

The 200M accumulation=4 workloads take 23.0%-23.4% less complete wall time and 24.9%-25.4% less steady step time. The small MoE GPU-accumulation fixture takes 5.8% less median wall time and 8.7% less median steady step time. Accumulation=1 and CPU accumulator placement do not select fusion. Their wall ranges overlap; CPU placement has a 4.4% slower median, so no benefit is claimed for these controls.

All 15 matched pairs have identical model tensor shapes, types and bytes; optimizer tensor shapes, types and bytes; logged epoch losses; logical steps; and supervised-token counts. Each 200M pair checks 252 model tensors and 696 optimizer tensors. Peak sampled GPU memory is unchanged: 1392 MiB for 200M accumulation=4, 1142 MiB for accumulation=1, and 200 MiB for the small MoE GPU accumulator.

## Captured GPU execution

nvprof captures the contiguous interval after optimizer step 2 through step 4: two updates and eight microbatches. Each profiled final model matches its corresponding normal-run model tensor hashes. GPU activity sums are not wall time, and API times must not be added to GPU times.

| Metric | 200M before | 200M after | Small MoE before | Small MoE after |
| --- | ---: | ---: | ---: | ---: |
| GPU activity sum (ms) | 5065.558 | 3701.510 | 445.379 | 408.981 |
| GPU activity calls | 31764 | 29908 | 98980 | 98020 |
| Kernel launches | 29984 | 28128 | 13056 | 12096 |
| Device property queries | 8 | 8 | 8 | 8 |

In the 200M capture, separate F32-to-Q8 CPY takes 1440.657 ms over 1856 calls and the old Q8 accumulator takes 237.664 ms. Fusion takes 311.954 ms over 1856 calls, an 81.4% reduction for their combined work. The separate quantizer has zero remaining calls in the candidate trace; Q8-to-F32 copies elsewhere are unchanged. Total GPU activity falls 26.9%, and kernel launches fall by 1856. The trace confirms 16 registers for fusion and 15 for the baseline accumulator.

The small MoE capture removes 960 quantizer calls: CPY plus accumulation falls from 47.206 ms to 10.590 ms. Total GPU activity falls 8.2%. Its much smaller wall-time improvement reflects other work, including 84,672 device-to-device copies in both captures. The largest remaining individual 200M kernels are GEMMs; fusion is no longer the largest kernel cost.

An initial implementation queried the GPU device type for every gradient tensor. A diagnostic trace showed 1864 cudaGetDeviceProperties calls taking 163.304 ms in the 200M capture and 968 calls taking 98.273 ms in the small MoE capture. This caused a small-model wall-time regression despite less GPU work. The final implementation identifies the GPU once when creating each accumulator state. Initial measurements are retained under pre-cache and excluded from the final tables.

## Correctness and limitations

- The CPU and CUDA QAT suites compare separate CPY plus Q8 accumulation against fused accumulation byte for byte across 32, 96, and 704 columns, five rows, eight updates, and reset/non-reset steps. Inputs cover signed zeros, equal-magnitude signed maxima, rounding ties, FP16 overflow, NaN/Inf, subnormals, and recovery to finite values. Each backend is compared with its own original quantizer; this does not claim CPU/CUDA nonfinite quantization is identical.
- CUDA test-backend-ops checks the six new reset/shape cases against CPU, plus the existing routed and statistics fixtures. All 20 supported cases pass; six unsupported routed dense accumulation cases are skipped. CPU/CUDA support CSVs include the six new cases.
- test-opt passes and git diff --check is clean. Public headers and constructors are unchanged.
- Performance evidence is specific to the P620, these synthetic fixtures, and the existing Q4_0/Q8_0 training configuration. It is not a training-quality study or a benchmark of a full 26B model. HIP, MUSA, Vulkan, and other NVIDIA architectures were not benchmarked.

Artifacts are under /tmp/qlion-micrograd: run.py records exact commands, per-run JSON records model and optimizer tensor hashes, timing-summary.json and comparisons.json contain normal-run results, profile-summary.json and raw nvprof exports contain GPU evidence, and bench.csv contains isolated timings and output hashes. Large generated models are pruned after hashing. The prior five-optimization binary is the immutable baseline in /tmp/qlion-craftsman/train/candidate/bin; candidate binaries are separately snapshotted. Post-run integrity checks confirm both binary snapshots and all four original input files are unchanged.
