#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

template <bool inverse>
static __global__ void turbo_wht_f32_cuda(const float * src, float * dst, int64_t groups) {
    const int64_t group = blockIdx.x;
    if (group >= groups) {
        return;
    }

    __shared__ float shared[128];
    const int64_t index = group*128 + threadIdx.x;
    dst[index] = turbo_wht_128_cuda<inverse>(src[index], shared);
}

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src->ne[0] % 128 == 0);

    const int64_t groups = ggml_nelements(src)/128;
    const int inverse = ggml_get_op_params_i32(dst, 0);
    if (groups == 0) {
        return;
    }

    if (inverse) {
        turbo_wht_f32_cuda<true><<<groups, 128, 0, ctx.stream()>>>(
            (const float *) src->data, (float *) dst->data, groups);
    } else {
        turbo_wht_f32_cuda<false><<<groups, 128, 0, ctx.stream()>>>(
            (const float *) src->data, (float *) dst->data, groups);
    }
}
