#include "common.cuh"
#include "fwht.cuh"

template <int N>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

template <int N>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_set_rows_q8_0_cuda(
        const float * src, block_q8_0 * dst, const int64_t n_rows, const int64_t n_heads,
        const int64_t * row_indices, const int64_t set_rows_stride, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int el_w      = N/warp_size;

    const int64_t r = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }

    src += r*N;
    float reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i*warp_size + lane]*scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h/warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2*step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    ggml_cuda_pdl_sync();
    const int64_t head  = r % n_heads;
    const int64_t token = r / n_heads;
    const int64_t dst_row = row_indices[token];
    block_q8_0 * dst_row_ptr = (block_q8_0 *) ((char *) dst + dst_row*set_rows_stride);
    dst_row_ptr += head*el_w;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const float amax = warp_reduce_max(fabsf(reg[i]));
        const float d = amax/127.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;
        if (lane == 0) {
            dst_row_ptr[i].d = d;
        }
        dst_row_ptr[i].qs[lane] = roundf(reg[i]*id);
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_op_fwht_set_rows_q8_0(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * fwht, ggml_tensor * set_rows) {
    GGML_ASSERT(ggml_are_same_shape(src, fwht));
    if (!ggml_is_contiguous(src) || set_rows->type != GGML_TYPE_Q8_0 ||
            set_rows->src[1]->type != GGML_TYPE_I64 || src->ne[3] != 1) {
        return false;
    }

    const int n = src->ne[0];
    const int64_t rows = ggml_nrows(src);
    const int64_t n_heads = src->ne[1];
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1)/rows_per_block;

    const float * src_d = (const float *) src->data;
    block_q8_0 * dst_d = (block_q8_0 *) set_rows->data;
    const int64_t * row_indices = (const int64_t *) set_rows->src[1]->data;
    const int64_t set_rows_stride = set_rows->nb[1];
    const float scale = 1/sqrtf(n);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const dim3 grid_dims(num_blocks);
    const dim3 block_dims(warp_size, rows_per_block);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params(grid_dims, block_dims, 0, stream);

    switch (n) {
        case 256:
            ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<256>, launch_params,
                src_d, dst_d, rows, n_heads, row_indices, set_rows_stride, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_set_rows_q8_0_cuda<512>, launch_params,
                src_d, dst_d, rows, n_heads, row_indices, set_rows_stride, scale);
            return true;
        default:
            return false;
    }
}
