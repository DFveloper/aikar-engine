# GGML Operations

List of GGML operations and backend support status.

CPU `PAGED_ATTN` uses cache-sized tiles and SIMD helpers, with a scalar reference selectable through `GGML_CPU_PAGED_ATTN_REFERENCE=1`. CPU `MUL_MAT_ID` accepts optional expert-priority scheduling metadata. See [CPU LES and paged attention](pulsar-cpu-les-implementation.md) for supported configurations and numerical limits.

CPU `FLASH_ATTN_EXT` accepts paired NVFP4 K/V with FP32 row master scales. It decodes rows within the existing attention tiles and preserves the F16 rounding of the previous full-cache conversion. AVX2/F16C decodes directly to F16; other CPUs use the scalar codec with row-sized scratch. The model graph selects direct attention for CPU-resident NVFP4 caches with at least 512 cells during single-token, single-stream decode. Below 65,536 cache cells, prefill, batched decode and shorter caches retain the previous conversion path because direct attention did not consistently improve their measured throughput. Longer caches use direct attention to bound temporary storage. Mixed NVFP4 cache types are unsupported.

CPU routed QLion accumulation variants are listed as `ACC_QLION_QAT_ID` and `ACC_QLION_QAT_EXPERTS`. They reuse the `ACC_QLION_QAT` and `ACC_QLION_QAT_ROWS` operation codes with variant parameters; the compact expert updater reuses `OPT_STEP_QLION_QAT_SPARSE_ROWS`. CUDA supports compact expert accumulation and its updater; the direct dense fused accumulation variant remains CPU-only. Vulkan rejects these expert variants. See [native QLion training](../examples/qlora_training/README.md) for selection policy and dense fallback.

CUDA `OUT_PROD` bounds conversion scratch to 64 MiB for large contiguous non-Q4 quantized inputs whose decoded row fits that workspace. Q4_0 keeps its existing tiled path. P620 Q6_K/Q8_0 boundary, transposition, broadcasting, and accumulation tests pass against CPU references. Routed CUDA dX and QLion updates emit expert routes in input order. See [P620 validation](superpowers/reports/2026-11-03-qlion2-p620.md) for measurements and limits.

For compact `QLION_QAT_STATS`, internal `op_params[0] = 1` requests only the warmup threshold. CPU and CUDA skip the momentum scan after warmup and set the unused layer RMS output to zero. The public constructor defaults to full statistics. SM61 direct routed QLion updates bypass padded GEMM buffers when the total route count is at most four. See the [P620 optimization study](superpowers/reports/2026-11-04-qlion2-craftsman.md) for matched training timings and profiles.

For dense `ACC_QLION_QAT`, internal `op_params[2] = 1` quantizes the F32 microgradient before accumulation. CUDA fuses the two operations while preserving the intermediate Q8 bytes and FP16 scale rounding; CPU provides a reference implementation. The optimizer selects this mode only for supported GPU accumulators. CPU accumulator placement and unsupported backends retain the separate Q8 cast. See the [microgradient fusion study](superpowers/reports/2026-11-04-qlion2-microgradient-fusion.md).

CUDA `OUT_PROD_ID` and compact expert accumulation gather each expert's activation and gradient vectors in one kernel. Route order and the existing GEMM dimensions are preserved, including broadcast activations and strided token storage. See the [MoE gather and state-kernel study](superpowers/reports/2026-11-04-qlion2-moe-gather.md).

CPU and CUDA `DISTILLATION` and `DISTILLATION_BACK` fuse top-k plus tail KL and its logit gradient. Teacher IDs and log-probabilities are inputs; masked rows produce zero loss and gradient. CUDA uses a shared token-ID hash table and supports k <= 4096. The native trainer uses these operations by default. `GGML_DISTILL_REFERENCE=1` selects the earlier graph-composed loss for numerical and performance comparisons. Other backends can fall back to CPU. See [the recovery workflow](../scripts/logit-distillation/README.md) and [A100 container](../scripts/logit-distillation/runpod/README.md).

## How to add a backend to this table:

1. Run `test-backend-ops support --output csv` with your backend name and redirect output to a csv file in `docs/ops/` (e.g., `docs/ops/CUDA.csv`)
2. Regenerate `/docs/ops.md` via `./scripts/create_ops_docs.py`

Legend:
- ✅ Fully supported by this backend
- 🟡 Partially supported by this backend
- ❌ Not supported by this backend

| Operation | BLAS | CANN | CPU | CUDA | ET | HTP | MTL | OpenCL | SYCL | Vulkan | WebGPU | ZenDNN | zDNN |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|                              ABS | ❌ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              ACC | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|            ACC_QLION_QAT_EXPERTS | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                 ACC_QLION_QAT_ID | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                              ADD | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             ADD1 | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                           ADD_ID | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           ARANGE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           ARGMAX | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          ARGSORT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             CEIL | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            CLAMP | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        COL2IM_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CONCAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                             CONT | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       CONV_1D_DW | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
|                          CONV_2D | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                       CONV_2D_DW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          CONV_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_2D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              COS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      COUNT_EQUAL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              CPY | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                     DISTILLATION | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                DISTILLATION_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|               CROSS_ENTROPY_LOSS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|          CROSS_ENTROPY_LOSS_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CUMSUM | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             DIAG | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    DIAG_MASK_INF | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              DIV | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     DSV4_HC_COMB | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     DSV4_HC_POST | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                      DSV4_HC_PRE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              DUP | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              EXP | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            EXPM1 | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             FILL | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                   FLASH_ATTN_EXT | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                            FLOOR | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                  GATED_DELTA_NET | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                GATED_LINEAR_ATTN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            GEGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        GEGLU_ERF | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      GEGLU_QUICK | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             GELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GELU_ERF | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                       GELU_QUICK | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GET_ROWS | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ❌ | ❌ |
|                    GET_ROWS_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                       GROUP_NORM | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                      HARDSIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        HARDSWISH | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           IM2COL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        IM2COL_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          L2_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       LEAKY_RELU | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                LIGHTNING_INDEXER | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              LOG | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             MEAN | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                              MUL | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          MUL_MAT | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
|                 MUL_MAT_HADAMARD | ❌ | ❌ | ✅ | ❌ | ✅ | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 |
|                       MUL_MAT_ID | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | ❌ |
|                  MUL_MAT_ID_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                  MUL_MAT_ID_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              NEG | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                   OPT_STEP_ADAMW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                     OPT_STEP_SGD | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         OUT_PROD | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                              PAD | ❌ | 🟡 | ✅ | 🟡 | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                   PAD_REFLECT_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                          POOL_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          POOL_2D | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                  QLION_QAT_STATS | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                            REGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             RELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           REPEAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                      REPEAT_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         RMS_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    RMS_NORM_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROLL | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROPE | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        ROPE_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            ROUND | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        RWKV_WKV6 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                        RWKV_WKV7 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            SCALE | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SET | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                         SET_ROWS | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                              SGN | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          SIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SILU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        SILU_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                              SIN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFTPLUS | ❌ | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFT_MAX | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    SOFT_MAX_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                        SOLVE_TRI | ❌ | ❌ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                              SQR | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SQRT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_CONV | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_SCAN | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ❌ | ❌ |
|                             STEP | ❌ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUB | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUM | ❌ | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                         SUM_ROWS | ❌ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                           SWIGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     SWIGLU_CLAMP | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                       SWIGLU_OAI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             TANH | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|               TIMESTEP_EMBEDDING | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            TOP_K | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                              TRI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            TRUNC | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          UPSCALE | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            XIELU | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |

CPU `fp4_kv` is a runtime-only E2M1 cache format with one E4M3 scale per 16 channels and no FP32 master scale. CPU CPY, SET_ROWS and paired FLASH_ATTN_EXT support it. Lumen4Exp selects it for global KV by default. CUDA supports F32/F16 conversion through CPY for training fake quantization; persistent cache writes and direct attention remain CPU operations. Vulkan declines this cache format.
