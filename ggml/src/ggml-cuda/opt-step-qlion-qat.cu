#include "ggml-impl.h"
#include "opt-step-qlion-qat.cuh"
#include "moe-route.cuh"

#include <algorithm>
#include <cstdint>

static constexpr int QLION_QAT_ID_GEMM_CAP   = 128;
static constexpr int QLION_QAT_ID_GEMM_BATCH = 8;

static constexpr int QLION_QAT_WARPS_PER_BLOCK = 4;

static constexpr int QLION_QAT_THREADS =
QLION_QAT_WARPS_PER_BLOCK * 32;

static inline void qlion_qat_id_gemm_batched(
        cublasHandle_t handle,
        int m,
        int n,
        int k,
        const float * a,
        int lda,
        long long stride_a,
        const float * b,
        int ldb,
        long long stride_b,
        float * c,
        int ldc,
        long long stride_c,
        int batch_count) {

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(
        cublasSgemmStridedBatched(
            handle,

            CUBLAS_OP_N,
            CUBLAS_OP_T,

            m,
            n,
            k,

            &alpha,

            a,
            lda,
            stride_a,

            b,
            ldb,
            stride_b,

            &beta,

            c,
            ldc,
            stride_c,

            batch_count
        )
    );
}

static __device__ __forceinline__ float q4_0_value(const block_q4_0 & block, int lane) {
    const uint8_t packed = block.qs[lane & 15];
    const int q = lane < 16 ? packed & 0x0f : packed >> 4;
    return __half2float(block.d) * (q - 8);
}

static __device__ __forceinline__ float mxfp4_value(const block_mxfp4 & block, int lane) {
    const uint8_t packed = block.qs[lane & 15];
    const int q = lane < 16 ? packed & 0x0f : packed >> 4;
    return ggml_cuda_e8m0_to_fp32(block.e) * 0.5f * kvalues_mxfp4[q];
}

static __device__ __forceinline__ float warp_signed_absmax(
        float value) {

    const float av =
        fabsf(value);

    const float amax =
        warp_reduce_max(av);

    const unsigned mask =
        __ballot_sync(
            0xffffffff,
            av == amax
        );

    return __shfl_sync(
        0xffffffff,
        value,
        __ffs(mask) - 1
    );
}

static __device__ __forceinline__ int mxfp4_best_index(
        float value,
        float inv_scale) {

    if (!(inv_scale > 0.0f) || !isfinite(value)) {
        return 0;
    }

    //
    // Normalize once.
    //
    // MXFP4 magnitudes:
    //   0, 1, 2, 3, 4, 6, 8, 12
    //
    const float x =
        fabsf(value) * inv_scale;

    int mag;

    if (x <= 3.5f) {
        if (x <= 1.5f) {
            mag = x <= 0.5f ? 0 : 1;
        } else {
            mag = x <= 2.5f ? 2 : 3;
        }
    } else {
        if (x <= 7.0f) {
            mag = x <= 5.0f ? 4 : 5;
        } else {
            mag = x <= 10.0f ? 6 : 7;
        }
    }

    if (mag == 0) {
        return 0;
    }

    return value < 0.0f
        ? mag + 8
        : mag;
}

template<bool mxfp4>
static __device__ void opt_step_qlion_qat_apply(
        void * __restrict__ weight_data,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t ib,
        int lane,
        float g) {

    const float alpha = pars[0];
    const float beta  = pars[1];
    const float wd    = pars[2];
    const float gclip = pars[3];

    const bool fast_state_scale =
        pars[4] != 0.0f;

    const float momentum_old_scale =
        __half2float(
            momentum[ib].d
        );

    const float momentum_value =
        momentum_old_scale *
        (float)
            momentum[ib].qs[lane];

    const float residual_old_scale =
        __half2float(
            residual[ib].d
        );

    const uint8_t residual_old_packed =
        residual[ib].
            qs[lane & 15];

    const int residual_old_q =
        lane < 16
            ? residual_old_packed &
                0x0f
            : residual_old_packed >>
                4;

    const float residual_value =
        residual_old_scale *
        (
            residual_old_q -
            8
        );

    const float weight_old =
        mxfp4
            ? mxfp4_value(
                ((const block_mxfp4 *)
                    weight_data)[ib],
                lane
            )
            : q4_0_value(
                ((const block_q4_0 *)
                    weight_data)[ib],
                lane
            );
    g = isfinite(g) ? g : 0.0f;
    if (gclip > 0.0f) {
        g = fmaxf(-gclip, fminf(gclip, g));
    }
    float momentum_new = beta * momentum_value + (1.0f - beta) * g;
    momentum_new = isfinite(momentum_new) ? momentum_new : 0.0f;
    momentum_new = fmaxf(-128.0f * 65504.0f, fminf(128.0f * 65504.0f, momentum_new));
    const float direction = momentum_new > 0.0f ? 1.0f : (momentum_new < 0.0f ? -1.0f : 0.0f);
    float target = weight_old + residual_value - alpha * (direction + wd * weight_old);
    target = isfinite(target) ? target : weight_old;
    if (!mxfp4) {
        target = fmaxf(-8.0f * 65504.0f, fminf(8.0f * 65504.0f, target));
    }

    float weight_new;
    if constexpr (mxfp4) {
        const float amax =
            warp_reduce_max(fabsf(target));

        int exponent_i32 = 0;
        float scale = 0.0f;
        float inv_scale = 0.0f;

        if (lane == 0) {
            if (amax > 0.0f) {
                const uint32_t bits =
                    __float_as_uint(amax);

                const int fp32_exponent =
                    (int) ((bits >> 23) & 0xff);

                //
                // For normal FP32:
                //
                // floor(log2(x)) - 2 + 127
                //   == biased_exponent - 2
                //
                // exponent <= 2, including subnormals,
                // clamps to E8M0 exponent 0 anyway.
                //
                exponent_i32 =
                    fp32_exponent > 2
                        ? fp32_exponent - 2
                        : 0;

                if (exponent_i32 > 254) {
                    exponent_i32 = 254;
                }
            } else {
                exponent_i32 = 0;
            }

            scale =
                ggml_cuda_e8m0_to_fp32(
                    (uint8_t) exponent_i32
                ) * 0.5f;

            inv_scale =
                scale != 0.0f
                    ? 1.0f / scale
                    : 0.0f;
        }
        exponent_i32 =
            __shfl_sync(
                0xffffffff,
                exponent_i32,
                0
            );

        scale =
            __shfl_sync(
                0xffffffff,
                scale,
                0
            );

        inv_scale =
            __shfl_sync(
                0xffffffff,
                inv_scale,
                0
            );

        const uint8_t exponent =
            (uint8_t) exponent_i32;
        const int q =
            mxfp4_best_index(
                target,
                inv_scale
            );
        const int q_hi = __shfl_sync(0xffffffff, q, (lane + 16) & 31);
        if (lane == 0) {
            ((block_mxfp4 *) weight_data)[ib].e = exponent;
        }
        if (lane < 16) {
            ((block_mxfp4 *) weight_data)[ib].qs[lane] = q | (q_hi << 4);
        }
        weight_new = scale * kvalues_mxfp4[q];
    } else {
        const float signed_max = warp_signed_absmax(target);
        const float scale = signed_max / -8.0f;
        const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
        const int q = min(15, (int) (target * inverse + 8.5f));
        const int q_hi = __shfl_sync(0xffffffff, q, (lane + 16) & 31);
        if (lane == 0) {
            ((block_q4_0 *) weight_data)[ib].d = __float2half(scale);
        }
        if (lane < 16) {
            ((block_q4_0 *) weight_data)[ib].qs[lane] = q | (q_hi << 4);
        }
        weight_new = __half2float(__float2half(scale)) * (q - 8);
    }

    //
    // ============================================================
    // Residual Q4_0 update
    // ============================================================
    //

    float residual_new =
        target - weight_new;

    residual_new =
        isfinite(residual_new)
            ? residual_new
            : 0.0f;

    residual_new =
        fmaxf(
            -8.0f * 65504.0f,
            fminf(
                8.0f * 65504.0f,
                residual_new
            )
        );


    //
    // 최종적으로 Q4_0에 쓸 값들.
    //
    // residual_scale:
    //   block_q4_0.d 에 저장할 scale
    //
    // residual_inverse:
    //   quantization할 때 사용할 1 / scale
    //
    // residual_q:
    //   현재 lane의 4-bit code
    //
    float residual_scale   = 0.0f;
    float residual_inverse = 0.0f;
    int   residual_q       = 8;


    //
    // fast path가 실제 성공했는지.
    //
    // 이 값은 __all_sync() 결과이기 때문에
    // warp 내 32 lanes 모두 동일한 값을 갖는다.
    //
    bool reused_residual_scale = false;


    //
    // ------------------------------------------------------------
    // Fast path:
    // 이전 step에서 사용한 Q4_0 scale을 그대로 재사용해 본다.
    // ------------------------------------------------------------
    //
    if (fast_state_scale) {

        float old_scale   = 0.0f;
        float old_inverse = 0.0f;

        //
        // scale은 block당 하나뿐이므로
        // lane 0만 half -> float 변환 + reciprocal 수행.
        //
        if (lane == 0) {

            old_scale =
                residual_old_scale;

            if (old_scale != 0.0f &&
                isfinite(old_scale)) {

                old_inverse =
                    1.0f / old_scale;
            }
        }

        //
        // lane 0이 계산한 값을 warp 전체에 broadcast.
        //
        old_scale =
            __shfl_sync(
                0xffffffff,
                old_scale,
                0
            );

        old_inverse =
            __shfl_sync(
                0xffffffff,
                old_inverse,
                0
            );

        //
        // 기존 Q4_0 quantization 공식 그대로.
        //
        // q = int(value / scale + 8.5)
        //
        const float qf =
            residual_new *
            old_inverse +
            8.5f;

        //
        // 중요한 부분.
        //
        // old scale을 그대로 썼을 때
        // 현재 lane의 값이 q=0..15 범위 안에
        // clamp 없이 들어오는가?
        //
        const bool lane_fits =
            old_inverse != 0.0f &&
            isfinite(qf) &&
            qf >= 0.0f &&
            qf < 16.0f;

        //
        // 32개 값 중 단 하나라도 범위를 넘으면
        // 이 block 전체의 scale reuse를 포기한다.
        //
        reused_residual_scale =
            __all_sync(
                0xffffffff,
                lane_fits
            );

        //
        // 32 lanes 모두 representable하면
        // reduction 없이 old scale 재사용.
        //
        if (reused_residual_scale) {

            residual_scale =
                old_scale;

            residual_inverse =
                old_inverse;

            residual_q =
                (int) qf;
        }
    }


    //
    // ------------------------------------------------------------
    // Slow / exact fresh-scale path.
    //
    // --qat-fast-state-scale이 꺼졌거나,
    // scale reuse가 실패했을 때만 실행.
    // ------------------------------------------------------------
    //
    if (!reused_residual_scale) {

        //
        // 이 reduction 하나로
        // absmax + signed max를 동시에 얻는다.
        //
        float residual_signed_max =
            warp_signed_absmax(
                residual_new
            );

        const float residual_absmax =
            fabsf(
                residual_signed_max
            );

        //
        // 기존 subnormal 보호 로직 유지.
    // 너무 작은 residual은 전부 0으로 취급.
    //
        if (residual_absmax > 0.0f &&
            residual_absmax <
                8.0f * 0x1p-24f) {

            residual_new = 0.0f;
            residual_signed_max = 0.0f;
        }

        //
        // 기존 Q4_0 scale 계산.
        //
        residual_scale =
            residual_signed_max /
            -8.0f;

        //
        // reciprocal division은 lane 0 한 번만.
        //
        if (lane == 0 &&
            residual_scale != 0.0f) {

            residual_inverse =
                1.0f /
                residual_scale;
        }

        residual_inverse =
            __shfl_sync(
                0xffffffff,
                residual_inverse,
                0
            );

        //
        // 기존 Q4_0 code 계산.
        //
        residual_q =
            min(
                15,
                (int) (
                    residual_new *
                    residual_inverse +
                    8.5f
                )
            );
    }


    //
    // Q4_0 packing.
    // lane 0..15가 두 개씩 묶어서 byte 하나를 저장.
    //
    const int residual_q_hi =
        __shfl_sync(
            0xffffffff,
            residual_q,
            (lane + 16) & 31
        );


    //
    // scale 저장.
    //
    if (lane == 0) {

        residual[ib].d =
            __float2half(
                residual_scale
            );
    }


    //
    // 32개의 4-bit 값을
    // 16 bytes로 packing.
    //
    if (lane < 16) {

        residual[ib].qs[lane] =
            residual_q |
            (residual_q_hi << 4);
    }


    //
    // ============================================================
    // Momentum Q8_0 update
    // ============================================================
    //

    float momentum_scale   = 0.0f;
    float momentum_inverse = 0.0f;
    int   momentum_q       = 0;

    bool reused_momentum_scale = false;


    //
    // ------------------------------------------------------------
    // Fast path:
    // 이전 Q8_0 scale 재사용 시도.
    // ------------------------------------------------------------
    //
    if (fast_state_scale) {

        float old_scale   = 0.0f;
        float old_inverse = 0.0f;

        if (lane == 0) {

            old_scale =
                momentum_old_scale;

            if (old_scale != 0.0f &&
                isfinite(old_scale)) {

                old_inverse =
                    1.0f /
                    old_scale;
            }
        }

        //
        // warp broadcast
        //
        old_scale =
            __shfl_sync(
                0xffffffff,
                old_scale,
                0
            );

        old_inverse =
            __shfl_sync(
                0xffffffff,
                old_inverse,
                0
            );

        //
        // 기존 momentum quantization은 roundf().
        //
        // 여기서 round까지 한 결과가
        // 실제 int8 범위 안에 있는지 검사.
        //
        const float qf =
            roundf(
                momentum_new *
                old_inverse
            );

        const bool lane_fits =
            old_inverse != 0.0f &&
            isfinite(qf) &&
            qf >= -128.0f &&
            qf <= 127.0f;

        //
        // 모든 32 lanes가 들어갈 때만
        // old Q8 scale 사용.
        //
        reused_momentum_scale =
            __all_sync(
                0xffffffff,
                lane_fits
            );

        if (reused_momentum_scale) {

            momentum_scale =
                old_scale;

            momentum_inverse =
                old_inverse;

            momentum_q =
                (int) qf;
        }
    }


    //
    // ------------------------------------------------------------
    // Fresh Q8 scale path
    // ------------------------------------------------------------
    //
    if (!reused_momentum_scale) {

        //
        // block 전체 max magnitude 한 번 계산.
        //
        const float momentum_signed_max =
            warp_signed_absmax(
                momentum_new
            );

        //
        // 기존 코드와 동일.
    // signed max를 -128에 대응시킨다.
    //
        momentum_scale =
            momentum_signed_max /
            -128.0f;

        //
        // reciprocal division도 lane0만.
        //
        if (lane == 0 &&
            momentum_scale != 0.0f) {

            momentum_inverse =
                1.0f /
                momentum_scale;
        }

        momentum_inverse =
            __shfl_sync(
                0xffffffff,
                momentum_inverse,
                0
            );

        //
        // 기존 Q8_0 quantization.
        //
        momentum_q =
            max(
                -128,
                min(
                    127,
                    (int) roundf(
                        momentum_new *
                        momentum_inverse
                    )
                )
            );
    }


    //
    // Q8_0 scale 저장.
    //
    if (lane == 0) {

        momentum[ib].d =
            __float2half(
                momentum_scale
            );
    }


    //
    // lane 하나 = int8 하나.
    // packing 필요 없음.
    //
    momentum[ib].qs[lane] =
        momentum_q;
}

template<bool mxfp4>
static __global__ void opt_step_qlion_qat(
        void * __restrict__ weight_data,
        const float * __restrict__ grad,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t n_blocks) {

    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int64_t ib =
        (int64_t) blockIdx.x *
            QLION_QAT_WARPS_PER_BLOCK +
        warp;

    if (ib >= n_blocks) {
        return;
    }

    opt_step_qlion_qat_apply<mxfp4>(
        weight_data,
        momentum,
        residual,
        pars,
        ib,
        lane,
        grad[
            ib * 32 +
            lane
        ]
    );
}


static __global__ void qlion_qat_id_gather_expert_batch_padded(
        const float * __restrict__ activations,
        const float * __restrict__ grad,
        const int32_t * __restrict__ expert_offsets,
        const int32_t * __restrict__ routes,
        float * __restrict__ a_gathered,
        float * __restrict__ g_gathered,
        int64_t cols,
        int64_t rows,
        int64_t n_act_used,
        int64_t n_used,
        int64_t expert_base,
        int64_t batch_count,
        int64_t cap) {

    const int64_t per_slot =
        cols + rows;

    const int64_t per_expert =
        cap * per_slot;

    const int64_t total =
        batch_count * per_expert;

    const int64_t i =
        (int64_t) blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i >= total) {
        return;
    }

    const int64_t batch_expert =
        i / per_expert;

    const int64_t local =
        i - batch_expert * per_expert;

    const int64_t slot =
        local / per_slot;

    const int64_t j =
        local - slot * per_slot;

    const int64_t expert =
        expert_base + batch_expert;

    const int32_t begin =
        expert_offsets[expert];

    const int32_t end =
        expert_offsets[expert + 1];

    const int64_t route_pos =
        (int64_t) begin + slot;

    const bool valid =
        route_pos < end;

    int64_t route = 0;

    if (valid) {
        route = routes[route_pos];
    }

    if (j < cols) {
        float value = 0.0f;

        if (valid) {
            const int64_t used =
                route % n_used;

            const int64_t token =
                route / n_used;

            const int64_t act_used =
                used % n_act_used;

            const int64_t act_route =
                act_used +
                n_act_used * token;

            value =
                activations[
                    j +
                    cols * act_route
                ];
        }

        a_gathered[
            batch_expert * cols * cap +
            j +
            cols * slot
        ] = value;

    } else {
        const int64_t row =
            j - cols;

        float value = 0.0f;

        if (valid) {
            value =
                grad[
                    row +
                    rows * route
                ];
        }

        g_gathered[
            batch_expert * rows * cap +
            row +
            rows * slot
        ] = value;
    }
}

template<bool mxfp4>
static __global__ void opt_step_qlion_qat_id_apply_gemm(
        void * __restrict__ weight_data,
        const float * __restrict__ expert_grad,
        const float * __restrict__ activations,
        const float * __restrict__ grad,
        const int32_t * __restrict__ expert_offsets,
        const int32_t * __restrict__ routes,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t n_blocks_row,
        int64_t n_rows,
        int64_t n_act_used,
        int64_t n_used,
        int64_t expert_base,
        int64_t batch_count,
        int64_t cap) {

    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int64_t task =
        (int64_t) blockIdx.x *
            QLION_QAT_WARPS_PER_BLOCK +
        warp;

    const int64_t n_blocks_expert =
        n_blocks_row * n_rows;

    const int64_t total_tasks =
        batch_count *
        n_blocks_expert;

    if (task >= total_tasks) {
        return;
    }

    const int64_t batch_expert =
        task / n_blocks_expert;

    const int64_t local_ib =
        task % n_blocks_expert;

    if (batch_expert >= batch_count) {
        return;
    }

    const int64_t expert =
        expert_base + batch_expert;

    const int64_t block =
        local_ib % n_blocks_row;

    const int64_t row =
        local_ib / n_blocks_row;

    const int64_t cols =
        n_blocks_row * 32;

    //
    // 해당 batch expert의 GEMM 결과.
    //
    const float * current_expert_grad =
        expert_grad +
        batch_expert * cols * n_rows;

    float g =
        current_expert_grad[
            block * 32 +
            lane +
            cols * row
        ];

    //
    // cap을 넘은 route는 scalar fallback으로 정확하게 추가.
    //
    const int32_t begin =
        expert_offsets[expert];

    const int32_t end =
        expert_offsets[expert + 1];

    const int64_t overflow_begin =
        (int64_t) begin + cap;

    if (overflow_begin < end) {
        for (int64_t p = overflow_begin;
             p < end;
             ++p) {

            const int64_t route =
                routes[p];

            const int64_t used =
                route % n_used;

            const int64_t token =
                route / n_used;

            const int64_t act_used =
                used % n_act_used;

            const int64_t act_route =
                act_used +
                n_act_used * token;

            const float a =
                activations[
                    block * 32 +
                    lane +
                    cols * act_route
                ];

            float dg = 0.0f;

            if (lane == 0) {
                dg =
                    grad[
                        row +
                        n_rows * route
                    ];
            }

            dg =
                __shfl_sync(
                    0xffffffff,
                    dg,
                    0
                );

            g =
                fmaf(
                    a,
                    dg,
                    g
                );
        }
    }

    //
    // expert-local → 전체 weight block index.
    //
    const int64_t global_ib =
        expert * n_blocks_expert +
        local_ib;

    opt_step_qlion_qat_apply<mxfp4>(
        weight_data,
        momentum,
        residual,
        pars,
        global_ib,
        lane,
        g
    );
}

template<bool mxfp4>
static __global__ void opt_step_qlion_qat_rows(
        void * __restrict__ weight_data,
        const float * __restrict__ grad,
        const int32_t * __restrict__ ids,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t n_blocks_row,
        int64_t n_rows,
        int64_t n_indices) {
    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int64_t task =
        (int64_t) blockIdx.x *
            QLION_QAT_WARPS_PER_BLOCK +
        warp;

    const int64_t total_tasks =
        n_indices *
        n_blocks_row;

    if (task >= total_tasks) {
        return;
    }

    const int64_t index =
        task /
        n_blocks_row;

    const int64_t block =
        task %
        n_blocks_row;
    if (index >= n_indices) {
        return;
    }
    const int32_t row = ids[index];
    if (row < 0 || row >= n_rows) {
        return;
    }
    for (int64_t i = 0; i < index; ++i) {
        if (ids[i] == row) {
            return;
        }
    }
    float g = 0.0f;
    for (int64_t i = index; i < n_indices; ++i) {
        if (ids[i] == row) {
            g += grad[i * n_blocks_row * 32 + block * 32 + lane];
        }
    }
    const int64_t ib = row * n_blocks_row + block;
    opt_step_qlion_qat_apply<mxfp4>(weight_data, momentum, residual, pars, ib, lane, g);
}

void ggml_cuda_opt_step_qlion_qat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];
    const ggml_tensor * grad = dst->src[1];
    ggml_tensor * momentum = dst->src[2];
    ggml_tensor * residual = dst->src[3];
    const ggml_tensor * pars = dst->src[4];
    GGML_ASSERT(weight->type == GGML_TYPE_MXFP4 || weight->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(grad->type == GGML_TYPE_F32 && momentum->type == GGML_TYPE_Q8_0 && residual->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(pars->type == GGML_TYPE_F32 && ggml_nelements(pars) == 5);
    GGML_ASSERT(ggml_is_contiguous(weight) && ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(momentum) && ggml_is_contiguous(residual) && ggml_is_contiguous(pars));
    const int64_t n_blocks = ggml_nelements(weight) / 32;
    const int64_t grid_blocks =
        (
            n_blocks +
            QLION_QAT_WARPS_PER_BLOCK -
            1
        ) /
    QLION_QAT_WARPS_PER_BLOCK;
    if (weight->type == GGML_TYPE_MXFP4) {
        opt_step_qlion_qat<true><<<grid_blocks, QLION_QAT_THREADS, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (block_q8_0 *) momentum->data,
            (block_q4_0 *) residual->data, (const float *) pars->data, n_blocks);
    } else {
        opt_step_qlion_qat<false><<<grid_blocks, QLION_QAT_THREADS, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (block_q8_0 *) momentum->data,
            (block_q4_0 *) residual->data, (const float *) pars->data, n_blocks);
    }
}

void ggml_cuda_opt_step_qlion_qat_id(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {

    ggml_tensor * weight =
        dst->src[0];

    const ggml_tensor * activations =
        dst->src[1];

    const ggml_tensor * grad =
        dst->src[2];

    const ggml_tensor * ids =
        dst->src[3];

    ggml_tensor * momentum =
        dst->src[4];

    ggml_tensor * residual =
        dst->src[5];

    const ggml_tensor * pars =
        dst->src[6];

    GGML_ASSERT(
        weight->type == GGML_TYPE_MXFP4 ||
        weight->type == GGML_TYPE_Q4_0
    );

    GGML_ASSERT(
        activations->type == GGML_TYPE_F32
    );

    GGML_ASSERT(
        grad->type == GGML_TYPE_F32
    );

    GGML_ASSERT(
        ids->type == GGML_TYPE_I32
    );

    GGML_ASSERT(
        momentum->type == GGML_TYPE_Q8_0
    );

    GGML_ASSERT(
        residual->type == GGML_TYPE_Q4_0
    );

    GGML_ASSERT(
        pars->type == GGML_TYPE_F32 &&
        ggml_nelements(pars) == 5
    );

    //
    // Persistent tensors.
    //
    GGML_ASSERT(
        ggml_is_contiguous(weight)
    );

    GGML_ASSERT(
        ggml_is_contiguous(momentum)
    );

    GGML_ASSERT(
        ggml_is_contiguous(residual)
    );

    GGML_ASSERT(
        ggml_is_contiguous(pars)
    );

    //
    // 현재 backward graph에서는 이 둘이 contiguous인 것이
    // 이미 앞 실행에서 확인됐다.
    //
    GGML_ASSERT(
        ggml_is_contiguous(activations)
    );

    GGML_ASSERT(
        ggml_is_contiguous(grad)
    );

    const int64_t cols =
        weight->ne[0];

    const int64_t rows =
        weight->ne[1];

    const int64_t n_expert =
        weight->ne[2];

    const int64_t n_act_used =
        activations->ne[1];

    const int64_t n_used =
        grad->ne[1];

    const int64_t n_tokens =
        grad->ne[2];

    GGML_ASSERT(
        cols % 32 == 0
    );

    GGML_ASSERT(
        weight->ne[3] == 1
    );

    GGML_ASSERT(
        activations->ne[0] == cols
    );

    GGML_ASSERT(
        grad->ne[0] == rows
    );

    GGML_ASSERT(
        n_act_used > 0
    );

    GGML_ASSERT(
        n_used % n_act_used == 0
    );

    GGML_ASSERT(
        activations->ne[2] == n_tokens
    );

    GGML_ASSERT(
        ids->ne[0] == n_used
    );

    GGML_ASSERT(
        ids->ne[1] == n_tokens
    );

    //
    // ids는 non-contiguous view여도 됨.
    //
    GGML_ASSERT(
        ids->nb[0] % sizeof(int32_t) == 0
    );

    GGML_ASSERT(
        ids->nb[1] % sizeof(int32_t) == 0
    );

    const int64_t ids_s0 =
        ids->nb[0] / sizeof(int32_t);

    const int64_t ids_s1 =
        ids->nb[1] / sizeof(int32_t);

    const int64_t total_routes =
        n_used * n_tokens;

    GGML_ASSERT(
        total_routes > 0
    );

    GGML_ASSERT(
        total_routes <= INT32_MAX
    );

    GGML_ASSERT(
        n_expert > 0
    );

    //
    // stream은 이미 CUDA Graph capture 중일 수 있다.
    // 따라서 여기부터 host synchronization 금지.
    //
    cudaStream_t stream =
        ctx.stream();

    //
    // GPU temporary buffers:
    //
    // expert_offsets: [n_expert + 1]
    // cursor:         [n_expert]
    // routes:         [total_routes]
    //
    // 현재 값이면 대략 수십 KB 수준.
    //
    ggml_cuda_pool_alloc<int32_t>
        expert_offsets(
            ctx.pool(),
            n_expert + 1
        );

    ggml_cuda_pool_alloc<int32_t>
        cursor(
            ctx.pool(),
            n_expert
        );

    ggml_cuda_pool_alloc<int32_t>
        routes(
            ctx.pool(),
            total_routes
        );

    //
    // GPU에서 route grouping.
    //
    ggml_cuda_moe_build_routes(
        stream,

        (const int32_t *)
            ids->data,

        expert_offsets.ptr,
        cursor.ptr,
        routes.ptr,

        //
        // QLion path does not need
        // local route rank.
        //
        nullptr,

        n_expert,
        n_used,
        n_tokens,

        ids_s0,
        ids_s1
    );

    //
    // ============================================================
    // GEMM-based expert gradient path
    // ============================================================
    //

    constexpr int64_t cap =
        QLION_QAT_ID_GEMM_CAP;

    //
    // Reused for every expert.
    //
    // Memory cost for 2816 x 1408, cap=256:
    //
    // A scratch   ~ 2.75 MiB
    // G scratch   ~ 1.38 MiB
    // dW scratch  ~ 15.1 MiB
    //
    // Total       ~ 19.3 MiB
    //
    constexpr int64_t expert_batch =
        QLION_QAT_ID_GEMM_BATCH;

    ggml_cuda_pool_alloc<float> a_gathered(
        ctx.pool(),
        expert_batch * cols * cap
    );

    ggml_cuda_pool_alloc<float> g_gathered(
        ctx.pool(),
        expert_batch * rows * cap
    );

    ggml_cuda_pool_alloc<float> expert_grad(
        ctx.pool(),
        expert_batch * cols * rows
    );

    const int64_t n_blocks_row =
        cols / 32;

    const int64_t n_blocks_expert =
        n_blocks_row * rows;

    cublasHandle_t handle =
        ctx.cublas_handle();

    CUBLAS_CHECK(
        cublasSetStream(
            handle,
            stream
        )
    );

    constexpr int gather_threads =
        256;

    //
    // Fixed expert loop.
    //
    // This loop happens on CPU while building/capturing the graph,
    // but DOES NOT inspect any GPU data.
    //
    // Therefore CUDA Graph capture remains valid.
    //
    for (int64_t expert_base = 0;
        expert_base < n_expert;
        expert_base += expert_batch) {

        const int batch_count =
            (int) std::min<int64_t>(
                expert_batch,
                n_expert - expert_base
            );

        //
        // --------------------------------------------------------
        // 1. Gather this expert's routed activation/grad columns.
        //
        // Invalid slots are explicitly zero-filled.
        // --------------------------------------------------------
        //

        const int64_t gather_elements =
            (int64_t) batch_count *
            cap *
            (cols + rows);

const int64_t gather_blocks =
    (
        gather_elements +
        gather_threads - 1
    ) /
        gather_threads;

        qlion_qat_id_gather_expert_batch_padded
            <<<gather_blocks,
            gather_threads,
            0,
            stream>>>(
                (const float *) activations->data,
                (const float *) grad->data,

                expert_offsets.ptr,
                routes.ptr,

                a_gathered.ptr,
                g_gathered.ptr,

                cols,
                rows,
                n_act_used,
                n_used,

                expert_base,
                batch_count,
                cap
            );

        CUDA_CHECK(cudaGetLastError());


        //
        // --------------------------------------------------------
        // 2. Compute:
        //
        //       dW = A @ G^T
        //
        // A = [cols, cap]
        // G = [rows, cap]
        // dW = [cols, rows]
        //
        // Column-major cuBLAS layout matches GGML's contiguous
        // [cols, rows] layout directly.
        // --------------------------------------------------------
        //

        qlion_qat_id_gemm_batched(
            handle,

            (int) cols,
            (int) rows,
            (int) cap,

            a_gathered.ptr,
            (int) cols,
            (long long) cols * cap,

            g_gathered.ptr,
            (int) rows,
            (long long) rows * cap,

            expert_grad.ptr,
            (int) cols,
            (long long) cols * rows,

            batch_count
        );

        //
        // --------------------------------------------------------
        // 3. Apply QLion/QAT to this expert.
        //
        // Any routes beyond cap are added via scalar overflow
        // fallback in this kernel.
        // --------------------------------------------------------
        //

        const int64_t apply_tasks =
            (int64_t)
                batch_count *
                n_blocks_expert;

        const int64_t apply_blocks =
            (
                apply_tasks +
                QLION_QAT_WARPS_PER_BLOCK -
                1
            ) /
            QLION_QAT_WARPS_PER_BLOCK;
        if (
            weight->type ==
            GGML_TYPE_MXFP4
        ) {
            opt_step_qlion_qat_id_apply_gemm<true>
                <<<apply_blocks, QLION_QAT_THREADS, 0, stream>>>(
                    weight->data,
                    expert_grad.ptr,

                    (const float *) activations->data,
                    (const float *) grad->data,

                    expert_offsets.ptr,
                    routes.ptr,

                    (block_q8_0 *) momentum->data,
                    (block_q4_0 *) residual->data,
                    (const float *) pars->data,

                    n_blocks_row,
                    rows,
                    n_act_used,
                    n_used,

                    expert_base,
                    batch_count,
                    cap
                );
        } else {
            opt_step_qlion_qat_id_apply_gemm<false>
                <<<apply_blocks, QLION_QAT_THREADS, 0, stream>>>(
                    weight->data,
                    expert_grad.ptr,

                    (const float *) activations->data,
                    (const float *) grad->data,

                    expert_offsets.ptr,
                    routes.ptr,

                    (block_q8_0 *) momentum->data,
                    (block_q4_0 *) residual->data,
                    (const float *) pars->data,

                    n_blocks_row,
                    rows,
                    n_act_used,
                    n_used,

                    expert_base,
                    batch_count,
                    cap
                );
        }

        CUDA_CHECK(
            cudaGetLastError()
        );
    }
}
void ggml_cuda_opt_step_qlion_qat_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];
    const ggml_tensor * grad = dst->src[1];
    const ggml_tensor * ids = dst->src[2];
    ggml_tensor * momentum = dst->src[3];
    ggml_tensor * residual = dst->src[4];
    const ggml_tensor * pars = dst->src[5];
    GGML_ASSERT(weight->type == GGML_TYPE_MXFP4 || weight->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(grad->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(momentum->type == GGML_TYPE_Q8_0 && residual->type == GGML_TYPE_Q4_0 && pars->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weight) && ggml_is_contiguous(grad) && ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(momentum) && ggml_is_contiguous(residual));
    const int64_t n_blocks_row = weight->ne[0] / 32;
    const int64_t n_indices = ggml_nelements(ids);
    const int64_t n_blocks = n_indices * n_blocks_row;
    const int64_t grid_blocks =
        (
            n_blocks +
            QLION_QAT_WARPS_PER_BLOCK -
            1
        ) /
        QLION_QAT_WARPS_PER_BLOCK;
    if (weight->type == GGML_TYPE_MXFP4) {
        opt_step_qlion_qat_rows<true><<<grid_blocks, QLION_QAT_THREADS, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (const int32_t *) ids->data,
            (block_q8_0 *) momentum->data, (block_q4_0 *) residual->data, (const float *) pars->data,
            n_blocks_row, weight->ne[1], n_indices);
    } else {
        opt_step_qlion_qat_rows<false><<<grid_blocks, QLION_QAT_THREADS, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (const int32_t *) ids->data,
            (block_q8_0 *) momentum->data, (block_q4_0 *) residual->data, (const float *) pars->data,
            n_blocks_row, weight->ne[1], n_indices);
    }
}
