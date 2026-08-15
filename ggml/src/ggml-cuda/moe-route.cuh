#include "out-prod.cuh"
#include "convert.cuh"
#include "moe-route.cuh"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
```

로.

현재 `q4_0_get()` / `mxfp4_get()`는 **남겨둬.**

그 바로 아래 현재:

```cpp
template<bool mxfp4>
static __global__ void mul_mat_id_back_q4(...)
```

부터 `ggml_cuda_mul_mat_id_back()` 끝까지를 **전부 지우고 아래로 교체**.

```cpp
static constexpr int MUL_MAT_ID_BACK_CAP =
    128;

static constexpr int MUL_MAT_ID_BACK_BATCH =
    8;

static constexpr int MUL_MAT_ID_BACK_WARPS =
    4;


// ============================================================
// Dequantize a batch of experts.
//
// Each warp handles one quant block = 32 columns.
// ============================================================

template<bool mxfp4>
static __global__ void mul_mat_id_back_dequant_batch(
        const void * __restrict__ weight_data,
        float * __restrict__ weight_f32,
        int64_t cols,
        int64_t rows,
        int64_t n_blocks_row,
        int64_t expert_base,
        int64_t batch_count) {

    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int64_t task =
        (int64_t) blockIdx.x *
            MUL_MAT_ID_BACK_WARPS +
        warp;

    const int64_t n_blocks_expert =
        n_blocks_row * rows;

    const int64_t total_tasks =
        batch_count *
        n_blocks_expert;

    if (task >= total_tasks) {
        return;
    }

    const int64_t batch_expert =
        task / n_blocks_expert;

    const int64_t local =
        task -
        batch_expert *
            n_blocks_expert;

    const int64_t block =
        local % n_blocks_row;

    const int64_t row =
        local / n_blocks_row;

    const int64_t expert =
        expert_base +
        batch_expert;

    const int64_t global_ib =
        block +
        n_blocks_row *
            (
                row +
                rows * expert
            );

    const float value =
        mxfp4
            ? mxfp4_get(
                ((const block_mxfp4 *)
                    weight_data)[global_ib],
                lane
            )
            : q4_0_get(
                ((const block_q4_0 *)
                    weight_data)[global_ib],
                lane
            );

    const int64_t col =
        block * 32 +
        lane;

    weight_f32[
        batch_expert *
            cols * rows +
        col +
        cols * row
    ] = value;
}


// ============================================================
// Gather routed gradient columns.
//
// G_e = [rows, CAP]
// ============================================================

static __global__ void mul_mat_id_back_gather_grad(
        const float * __restrict__ grad,
        const int32_t * __restrict__ expert_offsets,
        const int32_t * __restrict__ routes,
        float * __restrict__ grad_gathered,
        int64_t rows,
        int64_t n_used,
        int64_t grad_s0,
        int64_t grad_s1,
        int64_t grad_s2,
        int64_t expert_base,
        int64_t batch_count,
        int64_t cap) {

    const int64_t pair =
        blockIdx.x;

    const int64_t batch_expert =
        pair / cap;

    const int64_t slot =
        pair % cap;

    if (batch_expert >=
        batch_count) {
        return;
    }

    const int64_t expert =
        expert_base +
        batch_expert;

    __shared__ int32_t s_valid;
    __shared__ int32_t s_used;
    __shared__ int32_t s_token;

    if (threadIdx.x == 0) {

        const int32_t begin =
            expert_offsets[expert];

        const int32_t end =
            expert_offsets[expert + 1];

        const int64_t route_pos =
            (int64_t) begin +
            slot;

        if (route_pos < end) {

            const int32_t route =
                routes[route_pos];

            s_valid =
                1;

            s_used =
                route % n_used;

            s_token =
                route / n_used;

        } else {

            s_valid =
                0;

            s_used =
                0;

            s_token =
                0;
        }
    }

    __syncthreads();

    for (int64_t row =
             threadIdx.x;
         row < rows;
         row += blockDim.x) {

        float value =
            0.0f;

        if (s_valid) {
            value =
                grad[
                    row      * grad_s0 +
                    s_used   * grad_s1 +
                    s_token  * grad_s2
                ];
        }

        grad_gathered[
            batch_expert *
                rows * cap +
            row +
            rows * slot
        ] = value;
    }
}


// ============================================================
// Scatter GEMM output back into dst route layout.
//
// C_e = [cols, CAP]
// dst = [cols, n_used, n_tokens]
// ============================================================

static __global__ void mul_mat_id_back_scatter(
        const float * __restrict__ gemm_out,
        const int32_t * __restrict__ expert_offsets,
        const int32_t * __restrict__ routes,
        float * __restrict__ dst,
        int64_t cols,
        int64_t expert_base,
        int64_t batch_count,
        int64_t cap) {

    const int64_t pair =
        blockIdx.x;

    const int64_t batch_expert =
        pair / cap;

    const int64_t slot =
        pair % cap;

    if (batch_expert >=
        batch_count) {
        return;
    }

    const int64_t expert =
        expert_base +
        batch_expert;

    __shared__ int32_t s_route;
    __shared__ int32_t s_valid;

    if (threadIdx.x == 0) {

        const int32_t begin =
            expert_offsets[expert];

        const int32_t end =
            expert_offsets[expert + 1];

        const int64_t route_pos =
            (int64_t) begin +
            slot;

        if (route_pos < end) {
            s_valid =
                1;

            s_route =
                routes[route_pos];

        } else {
            s_valid =
                0;

            s_route =
                0;
        }
    }

    __syncthreads();

    if (!s_valid) {
        return;
    }

    for (int64_t col =
             threadIdx.x;
         col < cols;
         col += blockDim.x) {

        dst[
            col +
            cols *
                (int64_t) s_route
        ] =
            gemm_out[
                batch_expert *
                    cols * cap +
                col +
                cols * slot
            ];
    }
}


// ============================================================
// Exact overflow fallback.
//
// Only routes whose local rank >= CAP do the expensive row loop.
// All normal routes return immediately.
// ============================================================

template<bool mxfp4>
static __global__ void mul_mat_id_back_overflow(
        const void * __restrict__ weight_data,
        const float * __restrict__ grad,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ route_rank,
        float * __restrict__ dst,
        int64_t n_blocks,
        int64_t n_rows,
        int64_t n_expert,
        int64_t n_used,
        int64_t n_tokens,
        int64_t grad_s0,
        int64_t grad_s1,
        int64_t grad_s2,
        int64_t ids_s0,
        int64_t ids_s1,
        int64_t cap) {

    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int64_t task =
        (int64_t) blockIdx.x *
            MUL_MAT_ID_BACK_WARPS +
        warp;

    const int64_t total_routes =
        n_used * n_tokens;

    const int64_t total_tasks =
        n_blocks *
        total_routes;

    if (task >= total_tasks) {
        return;
    }

    const int64_t block =
        task % n_blocks;

    const int64_t route =
        task / n_blocks;

    //
    // Common case: handled by GEMM.
    //
    if (route_rank[route] < cap) {
        return;
    }

    const int64_t used =
        route % n_used;

    const int64_t token =
        route / n_used;

    const int32_t expert =
        ids[
            used  * ids_s0 +
            token * ids_s1
        ];

    if (expert < 0 ||
        expert >= n_expert) {
        return;
    }

    float sum =
        0.0f;

    for (int64_t row = 0;
         row < n_rows;
         ++row) {

        const int64_t ib =
            block +
            n_blocks *
                (
                    row +
                    n_rows *
                        expert
                );

        const float value =
            mxfp4
                ? mxfp4_get(
                    ((const block_mxfp4 *)
                        weight_data)[ib],
                    lane
                )
                : q4_0_get(
                    ((const block_q4_0 *)
                        weight_data)[ib],
                    lane
                );

        sum =
            fmaf(
                value,
                grad[
                    row   * grad_s0 +
                    used  * grad_s1 +
                    token * grad_s2
                ],
                sum
            );
    }

    dst[
        block * 32 +
        lane +
        n_blocks * 32 *
            route
    ] = sum;
}


// ============================================================
// Batched W @ G
// ============================================================

static inline void mul_mat_id_back_gemm_batched(
        cublasHandle_t handle,
        int cols,
        int cap,
        int rows,
        const float * weight,
        const float * grad,
        float * dst,
        int batch_count) {

    const float alpha =
        1.0f;

    const float beta =
        0.0f;

    CUBLAS_CHECK(
        cublasSgemmStridedBatched(
            handle,

            CUBLAS_OP_N,
            CUBLAS_OP_N,

            cols,
            cap,
            rows,

            &alpha,

            weight,
            cols,
            (long long) cols *
                rows,

            grad,
            rows,
            (long long) rows *
                cap,

            &beta,

            dst,
            cols,
            (long long) cols *
                cap,

            batch_count
        )
    );
}


// ============================================================
// New grouped MUL_MAT_ID_BACK launcher
// ============================================================

void ggml_cuda_mul_mat_id_back(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {

    const ggml_tensor * weight =
        dst->src[0];

    const ggml_tensor * grad =
        dst->src[1];

    const ggml_tensor * ids =
        dst->src[2];

    GGML_ASSERT(
        weight->type ==
            GGML_TYPE_MXFP4 ||
        weight->type ==
            GGML_TYPE_Q4_0
    );

    GGML_ASSERT(
        grad->type ==
            GGML_TYPE_F32 &&
        ids->type ==
            GGML_TYPE_I32 &&
        dst->type ==
            GGML_TYPE_F32
    );

    GGML_ASSERT(
        ggml_is_contiguous(weight) &&
        ggml_is_contiguous(dst)
    );

    GGML_ASSERT(
        grad->nb[0] %
            sizeof(float) == 0
    );

    GGML_ASSERT(
        grad->nb[1] %
            sizeof(float) == 0
    );

    GGML_ASSERT(
        grad->nb[2] %
            sizeof(float) == 0
    );

    GGML_ASSERT(
        ids->nb[0] %
            sizeof(int32_t) == 0
    );

    GGML_ASSERT(
        ids->nb[1] %
            sizeof(int32_t) == 0
    );

    const int64_t cols =
        weight->ne[0];

    const int64_t rows =
        weight->ne[1];

    const int64_t n_expert =
        weight->ne[2];

    const int64_t n_used =
        grad->ne[1];

    const int64_t n_tokens =
        grad->ne[2];

    const int64_t total_routes =
        n_used *
        n_tokens;

    const int64_t n_blocks_row =
        cols / 32;

    GGML_ASSERT(
        cols % 32 == 0
    );

    GGML_ASSERT(
        dst->ne[0] ==
            cols
    );

    GGML_ASSERT(
        dst->ne[1] ==
            n_used
    );

    GGML_ASSERT(
        dst->ne[2] ==
            n_tokens
    );

    const int64_t grad_s0 =
        grad->nb[0] /
        sizeof(float);

    const int64_t grad_s1 =
        grad->nb[1] /
        sizeof(float);

    const int64_t grad_s2 =
        grad->nb[2] /
        sizeof(float);

    const int64_t ids_s0 =
        ids->nb[0] /
        sizeof(int32_t);

    const int64_t ids_s1 =
        ids->nb[1] /
        sizeof(int32_t);

    cudaStream_t stream =
        ctx.stream();

    cublasHandle_t handle =
        ctx.cublas_handle();

    CUBLAS_CHECK(
        cublasSetStream(
            handle,
            stream
        )
    );

    //
    // Route metadata.
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

    ggml_cuda_pool_alloc<int32_t>
        route_rank(
            ctx.pool(),
            total_routes
        );

    ggml_cuda_moe_build_routes(
        stream,

        (const int32_t *)
            ids->data,

        expert_offsets.ptr,
        cursor.ptr,
        routes.ptr,
        route_rank.ptr,

        n_expert,
        n_used,
        n_tokens,

        ids_s0,
        ids_s1
    );

    constexpr int64_t cap =
        MUL_MAT_ID_BACK_CAP;

    constexpr int64_t expert_batch =
        MUL_MAT_ID_BACK_BATCH;

    //
    // ~138 MiB for 2816x1408,
    // batch=8, cap=128.
    //
    ggml_cuda_pool_alloc<float>
        weight_f32(
            ctx.pool(),
            expert_batch *
                cols * rows
        );

    ggml_cuda_pool_alloc<float>
        grad_gathered(
            ctx.pool(),
            expert_batch *
                rows * cap
        );

    ggml_cuda_pool_alloc<float>
        gemm_out(
            ctx.pool(),
            expert_batch *
                cols * cap
        );

    constexpr int dequant_threads =
        MUL_MAT_ID_BACK_WARPS *
        32;

    constexpr int gather_threads =
        256;

    for (int64_t expert_base = 0;
         expert_base < n_expert;
         expert_base += expert_batch) {

        const int batch_count =
            (int) std::min<int64_t>(
                expert_batch,
                n_expert -
                    expert_base
            );

        //
        // 1. Dequantize W_e.
        //
        const int64_t dequant_tasks =
            (int64_t) batch_count *
            n_blocks_row *
            rows;

        const int64_t dequant_blocks =
            (
                dequant_tasks +
                MUL_MAT_ID_BACK_WARPS -
                1
            ) /
            MUL_MAT_ID_BACK_WARPS;

        if (weight->type ==
            GGML_TYPE_MXFP4) {

            mul_mat_id_back_dequant_batch<true>
                <<<dequant_blocks,
                   dequant_threads,
                   0,
                   stream>>>(
                    weight->data,
                    weight_f32.ptr,
                    cols,
                    rows,
                    n_blocks_row,
                    expert_base,
                    batch_count
                );

        } else {

            mul_mat_id_back_dequant_batch<false>
                <<<dequant_blocks,
                   dequant_threads,
                   0,
                   stream>>>(
                    weight->data,
                    weight_f32.ptr,
                    cols,
                    rows,
                    n_blocks_row,
                    expert_base,
                    batch_count
                );
        }

        CUDA_CHECK(
            cudaGetLastError()
        );

        //
        // 2. Gather G_e.
        //
        mul_mat_id_back_gather_grad
            <<<batch_count * cap,
               gather_threads,
               0,
               stream>>>(
                (const float *)
                    grad->data,

                expert_offsets.ptr,
                routes.ptr,

                grad_gathered.ptr,

                rows,
                n_used,

                grad_s0,
                grad_s1,
                grad_s2,

                expert_base,
                batch_count,
                cap
            );

        CUDA_CHECK(
            cudaGetLastError()
        );

        //
        // 3. W_e @ G_e.
        //
        mul_mat_id_back_gemm_batched(
            handle,

            (int) cols,
            (int) cap,
            (int) rows,

            weight_f32.ptr,
            grad_gathered.ptr,
            gemm_out.ptr,

            batch_count
        );

        //
        // 4. Scatter back to routes.
        //
        mul_mat_id_back_scatter
            <<<batch_count * cap,
               gather_threads,
               0,
               stream>>>(
                gemm_out.ptr,

                expert_offsets.ptr,
                routes.ptr,

                (float *)
                    dst->data,

                cols,

                expert_base,
                batch_count,
                cap
            );

        CUDA_CHECK(
            cudaGetLastError()
        );
    }

    //
    // Rare >CAP routes.
    //
    const int64_t overflow_tasks =
        total_routes *
        n_blocks_row;

    const int64_t overflow_blocks =
        (
            overflow_tasks +
            MUL_MAT_ID_BACK_WARPS -
            1
        ) /
        MUL_MAT_ID_BACK_WARPS;

    if (weight->type ==
        GGML_TYPE_MXFP4) {

        mul_mat_id_back_overflow<true>
            <<<overflow_blocks,
               MUL_MAT_ID_BACK_WARPS * 32,
               0,
               stream>>>(
                weight->data,

                (const float *)
                    grad->data,

                (const int32_t *)
                    ids->data,

                route_rank.ptr,

                (float *)
                    dst->data,

                n_blocks_row,
                rows,
                n_expert,
                n_used,
                n_tokens,

                grad_s0,
                grad_s1,
                grad_s2,

                ids_s0,
                ids_s1,

                cap
            );

    } else {

        mul_mat_id_back_overflow<false>
            <<<overflow_blocks,
               MUL_MAT_ID_BACK_WARPS * 32,
               0,
               stream>>>(
                weight->data,

                (const float *)
                    grad->data,

                (const int32_t *)
                    ids->data,

                route_rank.ptr,

                (float *)
                    dst->data,

                n_blocks_row,
                rows,
                n_expert,
                n_used,
                n_tokens,

                grad_s0,
                grad_s1,
                grad_s2,

                ids_s0,
                ids_s1,

                cap
            );
    }

    CUDA_CHECK(
        cudaGetLastError()
    );
}
