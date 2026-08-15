#include "ggml-impl.h"
#include "opt-step-qlion-qat.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

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

static __global__ void qlion_qat_id_gather(
        const float * __restrict__ activations,
        const float * __restrict__ grad,
        const int32_t * __restrict__ routes,
        float * __restrict__ a_gathered,
        float * __restrict__ g_gathered,
        int64_t cols,
        int64_t rows,
        int64_t n_act_used,
        int64_t n_used,
        int64_t n_routes) {

    const int64_t per_route = cols + rows;

    const int64_t i =
        (int64_t) blockIdx.x * blockDim.x +
        threadIdx.x;

    const int64_t total =
        n_routes * per_route;

    if (i >= total) {
        return;
    }

    const int64_t route_index =
        i / per_route;

    const int64_t j =
        i - route_index * per_route;

    const int64_t route =
        routes[route_index];

    const int64_t used =
        route % n_used;

    const int64_t token =
        route / n_used;

    if (j < cols) {
        //
        // Gather activation column.
        //
        // activations may be:
        //
        //   [cols, 1,      tokens]
        // or
        //   [cols, n_used, tokens]
        //

        const int64_t act_route =
            n_act_used == 1
                ? token
                : used + n_act_used * token;

        a_gathered[
            j + cols * route_index
        ] =
            activations[
                j + cols * act_route
            ];
    } else {
        //
        // Gather upstream gradient column.
        //

        const int64_t row =
            j - cols;

        g_gathered[
            row + rows * route_index
        ] =
            grad[
                row + rows * route
            ];
    }
}

template<bool mxfp4>
static __global__ void opt_step_qlion_qat_id_apply_expert(
        void * __restrict__ weight_data,
        const float * __restrict__ expert_grad,
        block_q8_0 * __restrict__ momentum,
        block_q4_0 * __restrict__ residual,
        const float * __restrict__ pars,
        int64_t n_blocks_row,
        int64_t n_rows,
        int64_t expert) {

    const int64_t local_ib =
        blockIdx.x;

    const int64_t n_blocks_expert =
        n_blocks_row * n_rows;

    if (local_ib >= n_blocks_expert) {
        return;
    }

    const int lane =
        threadIdx.x;

    const int64_t block =
        local_ib % n_blocks_row;

    const int64_t row =
        local_ib / n_blocks_row;

    const int64_t cols =
        n_blocks_row * 32;

    //
    // expert_grad layout:
    //
    // [cols, rows]
    //
    // If expert_grad == nullptr, this expert received no
    // routed tokens. g=0, but we still run QLion so momentum
    // decay / weight decay semantics remain unchanged.
    //
    const float g =
        expert_grad != nullptr
            ? expert_grad[
                block * 32 +
                lane +
                cols * row
              ]
            : 0.0f;

    //
    // Weight/momentum/residual are laid out:
    //
    // [cols, rows, expert]
    //
    // opt_step_qlion_qat_apply() expects the GLOBAL
    // quant block index.
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

    //
    // Types
    //
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
        pars->type == GGML_TYPE_F32
    );

    GGML_ASSERT(
        ggml_nelements(pars) == 4
    );

    //
    // Persistent QAT tensors remain contiguous.
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
    // For this fast path we currently require activations
    // and grad to be contiguous.
    //
    // Your current graph already satisfies these.
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
        n_act_used == 1 ||
        n_act_used == n_used
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

    GGML_ASSERT(
        ids->nb[0] % sizeof(int32_t) == 0
    );

    GGML_ASSERT(
        ids->nb[1] % sizeof(int32_t) == 0
    );

    const int64_t total_routes =
        n_used * n_tokens;

    GGML_ASSERT(
        total_routes > 0
    );

    cudaStream_t stream =
        ctx.stream();

    cublasHandle_t handle =
        ctx.cublas_handle();

    CUBLAS_CHECK(
        cublasSetStream(handle, stream)
    );

    //
    // =========================================================
    // 1. Copy routing IDs to host.
    // =========================================================
    //
    // ids can be a non-contiguous view, so we preserve nb[].
    //
    // This follows the same strategy already used by
    // ggml_cuda_out_prod_id().
    //

    const size_t ids_nbytes =
        ggml_nbytes(ids);

    std::vector<char> ids_host(
        ids_nbytes
    );

    if (
        ids->buffer &&
        !ggml_backend_buffer_is_host(ids->buffer)
    ) {
        CUDA_CHECK(
            cudaMemcpyAsync(
                ids_host.data(),
                ids->data,
                ids_nbytes,
                cudaMemcpyDeviceToHost,
                stream
            )
        );

        //
        // We need the routing IDs on CPU before constructing
        // expert route lists.
        //
        CUDA_CHECK(
            cudaStreamSynchronize(stream)
        );
    } else {
        memcpy(
            ids_host.data(),
            ids->data,
            ids_nbytes
        );
    }

    auto read_expert_id =
        [&](int64_t used, int64_t token) -> int32_t {

            return *reinterpret_cast<const int32_t *>(
                ids_host.data() +
                used  * ids->nb[0] +
                token * ids->nb[1]
            );
        };

    //
    // =========================================================
    // 2. Count routes for each expert.
    // =========================================================
    //

    std::vector<int64_t> expert_count(
        n_expert,
        0
    );

    for (int64_t token = 0;
         token < n_tokens;
         ++token) {

        for (int64_t used = 0;
             used < n_used;
             ++used) {

            const int32_t expert =
                read_expert_id(
                    used,
                    token
                );

            GGML_ASSERT(
                expert >= 0 &&
                expert < n_expert
            );

            expert_count[expert]++;
        }
    }

    //
    // Prefix offsets:
    //
    // expert_routes for expert e are:
    //
    //   [expert_offset[e],
    //    expert_offset[e+1])
    //

    std::vector<int64_t> expert_offset(
        n_expert + 1,
        0
    );

    for (int64_t e = 0;
         e < n_expert;
         ++e) {

        expert_offset[e + 1] =
            expert_offset[e] +
            expert_count[e];
    }

    GGML_ASSERT(
        expert_offset[n_expert] ==
        total_routes
    );

    //
    // =========================================================
    // 3. Build route array grouped by expert.
    // =========================================================
    //
    // A route is flattened as:
    //
    //   route = used + n_used * token
    //

    std::vector<int32_t> route_host(
        total_routes
    );

    std::vector<int64_t> cursor =
        expert_offset;

    for (int64_t token = 0;
         token < n_tokens;
         ++token) {

        for (int64_t used = 0;
             used < n_used;
             ++used) {

            const int32_t expert =
                read_expert_id(
                    used,
                    token
                );

            const int64_t route =
                used +
                n_used * token;

            GGML_ASSERT(
                route <= INT32_MAX
            );

            route_host[
                cursor[expert]++
            ] =
                (int32_t) route;
        }
    }

    int64_t max_routes =
        0;

    for (int64_t e = 0;
         e < n_expert;
         ++e) {

        max_routes =
            std::max(
                max_routes,
                expert_count[e]
            );
    }

    GGML_ASSERT(
        max_routes > 0
    );

    //
    // =========================================================
    // 4. GPU scratch buffers.
    // =========================================================
    //

    ggml_cuda_pool_alloc<int32_t>
        routes_gpu(
            ctx.pool(),
            total_routes
        );

    CUDA_CHECK(
        cudaMemcpyAsync(
            routes_gpu.ptr,
            route_host.data(),
            total_routes * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream
        )
    );

    //
    // Gather buffers only need to fit the largest expert.
    //
    ggml_cuda_pool_alloc<float>
        a_gathered(
            ctx.pool(),
            cols * max_routes
        );

    ggml_cuda_pool_alloc<float>
        g_gathered(
            ctx.pool(),
            rows * max_routes
        );

    //
    // One expert gradient only:
    //
    //   [cols, rows]
    //
    // For your 2816 x 1408 matrix:
    // ~15.1 MiB.
    //
    ggml_cuda_pool_alloc<float>
        expert_grad(
            ctx.pool(),
            cols * rows
        );

    const int64_t n_blocks_row =
        cols / 32;

    const int64_t n_blocks_expert =
        n_blocks_row * rows;

    const float alpha =
        1.0f;

    const float beta =
        0.0f;

    constexpr int gather_threads =
        256;

    //
    // =========================================================
    // 5. Process one expert at a time.
    // =========================================================
    //

    for (int64_t expert = 0;
         expert < n_expert;
         ++expert) {

        const int64_t n_routes =
            expert_count[expert];

        const float * expert_grad_ptr =
            nullptr;

        if (n_routes > 0) {

            const int32_t * routes_e =
                routes_gpu.ptr +
                expert_offset[expert];

            //
            // -------------------------------------------------
            // Gather only routes assigned to this expert.
            // -------------------------------------------------
            //

            const int64_t gather_elements =
                n_routes *
                (cols + rows);

            const int64_t gather_blocks =
                (
                    gather_elements +
                    gather_threads -
                    1
                ) /
                gather_threads;

            qlion_qat_id_gather
                <<<gather_blocks,
                   gather_threads,
                   0,
                   stream>>>(
                    (const float *)
                        activations->data,

                    (const float *)
                        grad->data,

                    routes_e,

                    a_gathered.ptr,
                    g_gathered.ptr,

                    cols,
                    rows,

                    n_act_used,
                    n_used,
                    n_routes
                );

            CUDA_CHECK(
                cudaGetLastError()
            );

            //
            // -------------------------------------------------
            // expert_grad =
            //
            //   A_e @ G_e^T
            //
            // A_e:
            //   [cols, n_routes]
            //
            // G_e:
            //   [rows, n_routes]
            //
            // result:
            //   [cols, rows]
            //
            // -------------------------------------------------
            //

            CUBLAS_CHECK(
                cublasSgemm(
                    handle,

                    CUBLAS_OP_N,
                    CUBLAS_OP_T,

                    (int) cols,
                    (int) rows,
                    (int) n_routes,

                    &alpha,

                    a_gathered.ptr,
                    (int) cols,

                    g_gathered.ptr,
                    (int) rows,

                    &beta,

                    expert_grad.ptr,
                    (int) cols
                )
            );

            expert_grad_ptr =
                expert_grad.ptr;
        }

        //
        // -----------------------------------------------------
        // Apply QLion to this expert.
        //
        // If n_routes == 0:
        //
        // expert_grad_ptr == nullptr
        //
        // and the kernel uses g = 0.
        //
        // This preserves momentum/weight-decay behavior.
        // -----------------------------------------------------
        //

        if (
            weight->type ==
            GGML_TYPE_MXFP4
        ) {
            opt_step_qlion_qat_id_apply_expert<true>
                <<<n_blocks_expert,
                   32,
                   0,
                   stream>>>(
                    weight->data,

                    expert_grad_ptr,

                    (block_q8_0 *)
                        momentum->data,

                    (block_q4_0 *)
                        residual->data,

                    (const float *)
                        pars->data,

                    n_blocks_row,
                    rows,
                    expert
                );
        } else {
            opt_step_qlion_qat_id_apply_expert<false>
                <<<n_blocks_expert,
                   32,
                   0,
                   stream>>>(
                    weight->data,

                    expert_grad_ptr,

                    (block_q8_0 *)
                        momentum->data,

                    (block_q4_0 *)
                        residual->data,

                    (const float *)
                        pars->data,

                    n_blocks_row,
                    rows,
                    expert
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
