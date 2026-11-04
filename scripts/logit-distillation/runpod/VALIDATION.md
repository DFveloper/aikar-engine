# CUDA recovery validation

Date: 2026-11-04. CPU baseline commit: b928ce389. CUDA changes and the Korean first-turn policy are subsequent working-tree changes.

## Tested hardware and builds

The local GPU is a Quadro P620 with 1992 MiB VRAM, compute capability 6.1. Native CUDA 12.4 builds include sm_61 and sm_80. The container targets sm_80 for A100. No A100 is available locally, so full offload, A100 memory use, throughput and recovery quality remain unverified.

The container links against the CUDA driver stub during the build, then relies on NVIDIA Container Toolkit for the real driver at runtime. Driver stubs are not copied into the runtime image. The build directory uses a BuildKit cache for incremental rebuilds.

The default sm_80 Docker image builds successfully as aikar-distill:a100, about 1.57 GB reported image size. Container help, test-logit-distillation and the Python mixture regression test pass. Runtime shared-library resolution has no missing dependencies when the host libcuda.so.1 is mounted read-only. This machine has no NVIDIA Docker runtime configured; this check does not execute container GPU kernels. Native P620 kernels are tested separately below.

## Numerical checks

- CPU test-logit-distillation passes KL reference, finite-difference gradient, mask, temperature and cache checks.
- CUDA DISTILLATION forward matches CPU for vocabularies 8, 64 and 262144, including top-512 at the model's vocabulary size. Finite-difference gradient checks pass for the two small shapes; the harness skips the large shape's finite-difference calculation. Cases include masked rows and temperatures 1 and 2.
- CUDA Q4_1 MUL_MAT_ID_BACK passes five contiguous-weight cases, with input dimensions 32, 96, 256, 288 and 704 and duplicate expert routes. Five strided-weight cases correctly report unsupported and can fall back to CPU. The support gate was tightened after a strided-weight test exposed an assertion.
- CUDA F32-to-FP4_KV and FP4_KV-to-F32 copy cases both match CPU. This enables GPU fake quantization in the full-sequence training graph. Persistent FP4_KV cache SET_ROWS remains CPU-only.
- CUDA quantized microgradient accumulation passes six reset/non-reset cases; combined with the three forward KL cases, that invocation passes nine cases. This does not verify a full GPU optimizer trajectory.
- The broader QAT_TEST_ADAPTIVE_ONLY=CUDA test aborts with an illegal memory access reported in launch_mul_mat_q/mmq.cuh while running compact MoE accumulation trajectories. Its cause has not been established. That suite exercises expert-weight updates, which are outside the selected attention targets. It must not be reported as a passing CUDA regression suite.
- Python mixture regression checks pass deterministic sampling, Korean truncation, reasoning/system preservation, English preservation and cache-policy rejection. Shell syntax, Python compilation and git diff whitespace checks pass.

## Real-model smoke run

The new dataset contains 317 Korean first-turn records and 317 unchanged English conversations sampled with seed 42. Context and microbatch are 256; one training window and one validation window are retained. Teacher top-512 plus tail uses T=1. There is no ground-truth CE. Eight CPU threads are used, with attention targets in blocks 11, 17, 23 and 29.

Teacher extraction with one GPU layer completes: 512 input presentations and 356 supervised rows in 8.727 seconds, or 58.669 input tok/s and 40.793 supervised tok/s. End-to-end process time includes data preparation/model loading and is longer. This short context and partial-offload measurement is not comparable to a sustained A100 production run.

Training with the same partial GPU offload cannot allocate its 10880.60 MiB CUDA compute buffer on the 2 GB P620 and aborts before a training update. CPU training using the exact extracted cache completes one update: 256 input presentations and 191 supervised presentations in 65.478 seconds, or 3.910 input tok/s and 2.917 supervised tok/s. Validation KL is 1.27079320 before the update and 1.29325414 afterward. One update does not establish recovery quality; this smoke run does not improve the held-out loss. CPU compute workspace is 2588866560 bytes.

Local raw logs are /tmp/distill-cuda-kl-grad.log, /tmp/distill-cuda-moe-test-fixed.log, /tmp/distill-cuda-fp4-test.log, /tmp/distill-cuda-optimizer-test.log, /tmp/distill-cuda-qat-test.log and /tmp/distill-firstturn-cpu-train.log. The extraction log and cache are in MODEL_DIR/logit-distillation/benchmark-cuda-firstturn-c256/.

The CPU process exits successfully after writing /tmp/distill-firstturn-smoke-adapter.gguf, about 75 MiB. The adapter contains changed Q weights in all four target blocks and changed output weights in blocks 23 and 29. The Q8 output weights in blocks 11 and 17 do not change quantized payload after this single update. Whole-process wall time is 336.79 seconds, including loading, initial/final validation and adapter generation. Peak RSS is 17952288 KiB. Docker compilation ran concurrently, so these rates are smoke-test measurements under CPU contention.

For production, run preflight and benchmark on the actual A100 before extraction and training. The first-turn context-512 plan has 4493 training windows, 2127247 supervised presentations per epoch, 32 validation windows and a 6609698365-byte cache. The default three epochs retain up to 14 model/state checkpoint pairs and budget about 193.63 GiB after extraction. Recommend an A100 80 GB and a persistent volume with at least 300 GB free, including source models.
