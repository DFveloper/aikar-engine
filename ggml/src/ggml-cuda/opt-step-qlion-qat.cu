#include "ggml-impl.h"
#include "opt-step-qlion-qat.cuh"

#include <cstdint>

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

static __device__ __forceinline__ float warp_signed_absmax(float value) {
    const float amax = warp_reduce_max(fabsf(value));
    const unsigned mask = __ballot_sync(0xffffffff, fabsf(value) == amax);
    return __shfl_sync(0xffffffff, value, __ffs(mask) - 1);
}

static __device__ __forceinline__ int mxfp4_best_index(float value, float scale) {
    int best = 0;
    float error = fabsf(kvalues_mxfp4[0] * scale - value);
#pragma unroll
    for (int i = 1; i < 16; ++i) {
        const float candidate = fabsf(kvalues_mxfp4[i] * scale - value);
        if (candidate < error) {
            best = i;
            error = candidate;
        }
    }
    return best;
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
    const float beta = pars[1];
    const float wd = pars[2];
    const float gclip = pars[3];
    const block_q8_0 momentum_old = momentum[ib];
    const block_q4_0 residual_old = residual[ib];
    const float weight_old = mxfp4
        ? mxfp4_value(((const block_mxfp4 *) weight_data)[ib], lane)
        : q4_0_value(((const block_q4_0 *) weight_data)[ib], lane);
    const float momentum_value = __half2float(momentum_old.d) * momentum_old.qs[lane];
    const float residual_value = q4_0_value(residual_old, lane);
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
        const float amax = warp_reduce_max(fabsf(target));
        const int exponent_i = amax > 0.0f ? (int) floorf(log2f(amax)) - 2 + 127 : 0;
        const uint8_t exponent = (uint8_t) (exponent_i < 0 ? 0 : (exponent_i > 254 ? 254 : exponent_i));
        const float scale = ggml_cuda_e8m0_to_fp32(exponent) * 0.5f;
        const int q = mxfp4_best_index(target, scale);
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

    float residual_new = target - weight_new;
    residual_new = isfinite(residual_new) ? residual_new : 0.0f;
    residual_new = fmaxf(-8.0f * 65504.0f, fminf(8.0f * 65504.0f, residual_new));
    const float residual_absmax = warp_reduce_max(fabsf(residual_new));
    if (residual_absmax > 0.0f && residual_absmax < 8.0f * 0x1p-24f) {
        residual_new = 0.0f;
    }
    const float residual_signed_max = warp_signed_absmax(residual_new);
    const float residual_scale = residual_signed_max / -8.0f;
    const float residual_inverse = residual_scale != 0.0f ? 1.0f / residual_scale : 0.0f;
    const int residual_q = min(15, (int) (residual_new * residual_inverse + 8.5f));
    const int residual_q_hi = __shfl_sync(0xffffffff, residual_q, (lane + 16) & 31);
    if (lane == 0) {
        residual[ib].d = __float2half(residual_scale);
    }
    if (lane < 16) {
        residual[ib].qs[lane] = residual_q | (residual_q_hi << 4);
    }

    const float momentum_signed_max = warp_signed_absmax(momentum_new);
    const float momentum_scale = momentum_signed_max / -128.0f;
    const float momentum_inverse = momentum_scale != 0.0f ? 1.0f / momentum_scale : 0.0f;
    const int momentum_q = max(-128, min(127, (int) roundf(momentum_new * momentum_inverse)));
    if (lane == 0) {
        momentum[ib].d = __float2half(momentum_scale);
    }
    momentum[ib].qs[lane] = momentum_q;
}

template<bool mxfp4>
static __global__ void opt_step_qlion_qat(
        void * __restrict__ weight_data,
        const float * __restrict__ grad,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t n_blocks) {
    const int64_t ib = blockIdx.x;
    const int lane = threadIdx.x;
    if (ib < n_blocks) {
        opt_step_qlion_qat_apply<mxfp4>(weight_data, momentum, residual, pars, ib, lane, grad[ib * 32 + lane]);
    }
}

template<bool mxfp4>
static __global__ void opt_step_qlion_qat_id(
        void * __restrict__ weight_data,
        const float * __restrict__ activations,
        const float * __restrict__ grad,
        const int32_t * __restrict__ ids,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t n_blocks_row,
        int64_t n_rows,
        int64_t n_expert,
        int64_t n_act_used,
        int64_t n_used,
        int64_t n_tokens) {
    const int64_t ib = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t n_blocks = n_blocks_row * n_rows * n_expert;
    if (ib >= n_blocks) {
        return;
    }
    const int64_t block = ib % n_blocks_row;
    const int64_t row = (ib / n_blocks_row) % n_rows;
    const int64_t expert = ib / (n_blocks_row * n_rows);
    float g = 0.0f;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t used = 0; used < n_used; ++used) {
            const int64_t route = used + n_used * token;

            const int32_t routed_expert =
                ids[
                    used  * ids_s0 +
                    token * ids_s1
                ];

            if (routed_expert == expert) {

            // activations may have either:
            //
            // [K, n_used, T] -> normal routed layout
            // [K, 1,      T] -> broadcast layout
            //
            // For broadcast layout, all routes of a token reuse
            // the same activation vector.
            const int64_t act_route =
                n_act_used == 1
                    ? token
                    : used + n_act_used * token;

            if (ids[route] == expert) {
                const float a =
                    activations[
                        block * 32 +
                        lane +
                        n_blocks_row * 32 * act_route
                    ];

                const float dg =
                    grad[
                        row +
                        n_rows * route
                    ];

                g += a * dg;
            }
        }
    }
    opt_step_qlion_qat_apply<mxfp4>(weight_data, momentum, residual, pars, ib, lane, g);
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
    const int64_t index = blockIdx.x / n_blocks_row;
    const int64_t block = blockIdx.x % n_blocks_row;
    const int lane = threadIdx.x;
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
    GGML_ASSERT(pars->type == GGML_TYPE_F32 && ggml_nelements(pars) == 4);
    GGML_ASSERT(ggml_is_contiguous(weight) && ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(momentum) && ggml_is_contiguous(residual) && ggml_is_contiguous(pars));
    const int64_t n_blocks = ggml_nelements(weight) / 32;
    if (weight->type == GGML_TYPE_MXFP4) {
        opt_step_qlion_qat<true><<<n_blocks, 32, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (block_q8_0 *) momentum->data,
            (block_q4_0 *) residual->data, (const float *) pars->data, n_blocks);
    } else {
        opt_step_qlion_qat<false><<<n_blocks, 32, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (block_q8_0 *) momentum->data,
            (block_q4_0 *) residual->data, (const float *) pars->data, n_blocks);
    }
}

void ggml_cuda_opt_step_qlion_qat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * weight = dst->src[0];
    const ggml_tensor * activations = dst->src[1];
    const ggml_tensor * grad = dst->src[2];
    const ggml_tensor * ids = dst->src[3];
    ggml_tensor * momentum = dst->src[4];
    ggml_tensor * residual = dst->src[5];
    const ggml_tensor * pars = dst->src[6];
    GGML_ASSERT(weight->type == GGML_TYPE_MXFP4 || weight->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(activations->type == GGML_TYPE_F32 && grad->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(momentum->type == GGML_TYPE_Q8_0 && residual->type == GGML_TYPE_Q4_0 && pars->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(weight) && ggml_is_contiguous(activations) && ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_is_contiguous(momentum) && ggml_is_contiguous(residual));
    const int64_t n_act_used = activations->ne[1];
    const int64_t n_used     = grad->ne[1];
    const int64_t n_tokens   = grad->ne[2];

    GGML_ASSERT(
        n_act_used == 1 ||
        n_act_used == n_used
    );

    GGML_ASSERT(activations->ne[2] == n_tokens);

    GGML_ASSERT(ids->ne[0] == n_used);
    GGML_ASSERT(ids->ne[1] == n_tokens);
    GGML_ASSERT(ids->nb[0] % sizeof(int32_t) == 0);
    GGML_ASSERT(ids->nb[1] % sizeof(int32_t) == 0);

    const int64_t ids_s0 =
        ids->nb[0] / sizeof(int32_t);

    const int64_t ids_s1 =
        ids->nb[1] / sizeof(int32_t);
    const int64_t n_blocks_row = weight->ne[0] / 32;
    const int64_t n_blocks = ggml_nelements(weight) / 32;
    if (weight->type == GGML_TYPE_MXFP4) {
        opt_step_qlion_qat_id<true><<<n_blocks, 32, 0, ctx.stream()>>>(
            weight->data,
            (const float *) activations->data,
            (const float *) grad->data,
            (const int32_t *) ids->data,
            (block_q8_0 *) momentum->data,
            (block_q4_0 *) residual->data,
            (const float *) pars->data,
            n_blocks_row,
            weight->ne[1],
            weight->ne[2],
            n_act_used,
            n_used,
            n_tokens,
            ids_s0,
            ids_s1
        );
    } else {
        opt_step_qlion_qat_id<false><<<n_blocks, 32, 0, ctx.stream()>>>(
            weight->data,
            (const float *) activations->data,
            (const float *) grad->data,
            (const int32_t *) ids->data,
            (block_q8_0 *) momentum->data,
            (block_q4_0 *) residual->data,
            (const float *) pars->data,
            n_blocks_row,
            weight->ne[1],
            weight->ne[2],
            n_act_used,
            n_used,
            n_tokens,
            ids_s0,
            ids_s1
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
    if (weight->type == GGML_TYPE_MXFP4) {
        opt_step_qlion_qat_rows<true><<<n_blocks, 32, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (const int32_t *) ids->data,
            (block_q8_0 *) momentum->data, (block_q4_0 *) residual->data, (const float *) pars->data,
            n_blocks_row, weight->ne[1], n_indices);
    } else {
        opt_step_qlion_qat_rows<false><<<n_blocks, 32, 0, ctx.stream()>>>(
            weight->data, (const float *) grad->data, (const int32_t *) ids->data,
            (block_q8_0 *) momentum->data, (block_q4_0 *) residual->data, (const float *) pars->data,
            n_blocks_row, weight->ne[1], n_indices);
    }
}
