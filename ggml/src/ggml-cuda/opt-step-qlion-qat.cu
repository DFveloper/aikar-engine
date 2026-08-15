#include "ggml-impl.h"
#include "opt-step-qlion-qat.cuh"
#include "moe-route.cuh"

#include <algorithm>
#include <cstdint>

static constexpr int QLION_QAT_ID_GEMM_CAP   = 256;
static constexpr int QLION_QAT_ID_GEMM_BATCH = 8;

static constexpr int QLION_QAT_WARPS_PER_BLOCK = 4;

static constexpr int QLION_QAT_THREADS = QLION_QAT_WARPS_PER_BLOCK * 32;

static constexpr int64_t QLION_QAT_TIED_TILE_ROWS = 4096;

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
static __global__ void
opt_step_qlion_qat_tied_apply(
        void * __restrict__
            weight_data,

        const float * __restrict__
            tile_grad,

        block_q8_0 * __restrict__
            momentum,

        block_q4_0 * __restrict__
            residual,

        const float * __restrict__
            pars,

        int64_t n_blocks_row,

        int64_t row_base,
        int64_t tile_rows) {

    const int warp =
        threadIdx.x >> 5;

    const int lane =
        threadIdx.x & 31;

    const int64_t task =
        (int64_t) blockIdx.x *
            QLION_QAT_WARPS_PER_BLOCK +
        warp;

    const int64_t total_tasks =
        tile_rows *
        n_blocks_row;

    if (task >=
        total_tasks) {

        return;
    }

    const int64_t local_row =
        task /
        n_blocks_row;

    const int64_t block =
        task %
        n_blocks_row;

    const int64_t global_row =
        row_base +
        local_row;

    const int64_t ib =
        global_row *
            n_blocks_row +
        block;

    const int64_t cols =
        n_blocks_row *
        32;

    const float g =
        tile_grad[
            block * 32 +
            lane +
            cols *
                local_row
        ];

    opt_step_qlion_qat_apply<mxfp4>(
        weight_data,
        momentum,
        residual,
        pars,
        ib,
        lane,
        g
    );
}

static __global__ void
qlion_qat_tied_add_sparse(
        const float * __restrict__
            sparse_grad,

        const int32_t * __restrict__
            ids,

        float * __restrict__
            tile_grad,

        int64_t cols,
        int64_t n_indices,

        int64_t row_base,
        int64_t tile_rows) {

    const int64_t index =
        blockIdx.x;

    if (index >=
        n_indices) {

        return;
    }

    const int32_t row =
        ids[index];

    if (row <
            row_base ||
        row >=
            row_base +
            tile_rows) {

        return;
    }

    const int64_t local_row =
        (int64_t) row -
        row_base;

    for (int64_t col =
             threadIdx.x;

         col < cols;

         col +=
             blockDim.x) {

        //
        // Duplicate token ids are possible,
        // therefore atomicAdd is required.
        //
        atomicAdd(
            &tile_grad[
                col +
                cols *
                    local_row
            ],

            sparse_grad[
                col +
                cols *
                    index
            ]
        );
    }
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

void ggml_cuda_opt_step_qlion_qat_tied(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {

    ggml_tensor * weight =
        dst->src[0];

    const ggml_tensor * dense_a =
        dst->src[1];

    const ggml_tensor * dense_b =
        dst->src[2];

    const ggml_tensor * sparse_grad =
        dst->src[3];

    const ggml_tensor * sparse_ids =
        dst->src[4];

    ggml_tensor * momentum =
        dst->src[5];

    ggml_tensor * residual =
        dst->src[6];

    const ggml_tensor * pars =
        dst->src[7];


    GGML_ASSERT(
        weight->type ==
            GGML_TYPE_MXFP4 ||
        weight->type ==
            GGML_TYPE_Q4_0
    );

    GGML_ASSERT(
        dense_a->type ==
            GGML_TYPE_F32
    );

    GGML_ASSERT(
        dense_b->type ==
            GGML_TYPE_F32
    );

    GGML_ASSERT(
        sparse_grad->type ==
            GGML_TYPE_F32
    );

    GGML_ASSERT(
        sparse_ids->type ==
            GGML_TYPE_I32
    );

    GGML_ASSERT(
        momentum->type ==
            GGML_TYPE_Q8_0
    );

    GGML_ASSERT(
        residual->type ==
            GGML_TYPE_Q4_0
    );

    GGML_ASSERT(
        pars->type ==
            GGML_TYPE_F32 &&
        ggml_nelements(pars) == 5
    );

    GGML_ASSERT(
        ggml_is_contiguous(weight)
    );

    GGML_ASSERT(
        ggml_is_contiguous(dense_a)
    );

    GGML_ASSERT(
        ggml_is_contiguous(dense_b)
    );

    GGML_ASSERT(
        ggml_is_contiguous(sparse_grad)
    );

    GGML_ASSERT(
        ggml_is_contiguous(sparse_ids)
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


    const int64_t cols =
        weight->ne[0];

    const int64_t rows =
        weight->ne[1];

    const int64_t k =
        dense_a->ne[1];

    const int64_t n_indices =
        ggml_nelements(
            sparse_ids
        );


    GGML_ASSERT(
        cols % 32 == 0
    );

    GGML_ASSERT(
        weight->ne[2] == 1 &&
        weight->ne[3] == 1
    );

    GGML_ASSERT(
        dense_a->ne[0] ==
            cols
    );

    GGML_ASSERT(
        dense_b->ne[0] ==
            rows
    );

    GGML_ASSERT(
        dense_b->ne[1] ==
            k
    );

    GGML_ASSERT(
        sparse_grad->ne[0] ==
            cols
    );

    GGML_ASSERT(
        sparse_grad->ne[1] ==
            n_indices
    );


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


    constexpr int64_t tile_capacity =
        QLION_QAT_TIED_TILE_ROWS;


    //
    // Single scratch tile reused across all vocabulary tiles.
    //
    ggml_cuda_pool_alloc<float>
        tile_grad(
            ctx.pool(),
            cols *
                tile_capacity
        );


    const int64_t n_blocks_row =
        cols / 32;


    const float alpha =
        1.0f;

    const float beta =
        0.0f;


    for (int64_t row_base = 0;
         row_base < rows;
         row_base +=
             tile_capacity) {

        const int tile_rows =
            (int) std::min<int64_t>(
                tile_capacity,
                rows -
                    row_base
            );


        // ========================================================
        // Dense LM-head contribution:
        //
        // dense_a [cols, k]
        // dense_b [rows, k]
        //
        // tile =
        //   dense_a @ dense_b[row_base:row_end]^T
        //
        // dense_b is a strided submatrix:
        // pointer += row_base
        // leading dimension remains full `rows`.
        // ========================================================

        CUBLAS_CHECK(
            cublasSgemm(
                handle,

                CUBLAS_OP_N,
                CUBLAS_OP_T,

                (int) cols,
                tile_rows,
                (int) k,

                &alpha,

                (const float *)
                    dense_a->data,

                (int) cols,

                (const float *)
                    dense_b->data +
                    row_base,

                (int) rows,

                &beta,

                tile_grad.ptr,

                (int) cols
            )
        );


        // ========================================================
        // Sparse embedding contribution.
        //
        // tile[row=id] += sparse_grad[:, token]
        // ========================================================

        constexpr int sparse_threads =
            256;

        qlion_qat_tied_add_sparse
            <<<n_indices,
               sparse_threads,
               0,
               stream>>>(
                (const float *)
                    sparse_grad->data,

                (const int32_t *)
                    sparse_ids->data,

                tile_grad.ptr,

                cols,
                n_indices,

                row_base,
                tile_rows
            );

        CUDA_CHECK(
            cudaGetLastError()
        );


        // ========================================================
        // One QLion update using the already-combined gradient.
        // ========================================================

        const int64_t apply_tasks =
            (int64_t)
                tile_rows *
                n_blocks_row;

        const int64_t apply_blocks =
            (
                apply_tasks +
                QLION_QAT_WARPS_PER_BLOCK -
                1
            ) /
            QLION_QAT_WARPS_PER_BLOCK;


        if (weight->type ==
            GGML_TYPE_MXFP4) {

            opt_step_qlion_qat_tied_apply<true>
                <<<apply_blocks,
                   QLION_QAT_THREADS,
                   0,
                   stream>>>(
                    weight->data,

                    tile_grad.ptr,

                    (block_q8_0 *)
                        momentum->data,

                    (block_q4_0 *)
                        residual->data,

                    (const float *)
                        pars->data,

                    n_blocks_row,

                    row_base,
                    tile_rows
                );

        } else {

            opt_step_qlion_qat_tied_apply<false>
                <<<apply_blocks,
                   QLION_QAT_THREADS,
                   0,
                   stream>>>(
                    weight->data,

                    tile_grad.ptr,

                    (block_q8_0 *)
                        momentum->data,

                    (block_q4_0 *)
                        residual->data,

                    (const float *)
                        pars->data,

                    n_blocks_row,

                    row_base,
                    tile_rows
                );
        }

        CUDA_CHECK(
            cudaGetLastError()
        );
    }
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
