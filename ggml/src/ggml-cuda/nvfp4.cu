#include "nvfp4.cuh"
#include "convert.cuh"

#include <cfloat>
#include <climits>

struct nvfp4_layout {
    int64_t ne[4];
    size_t nb[4];
};

static nvfp4_layout nvfp4_tensor_layout(const ggml_tensor * tensor) {
    nvfp4_layout result;
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    return result;
}

static __device__ size_t nvfp4_row_offset(int64_t row, const nvfp4_layout & layout) {
    const int64_t i1 = row % layout.ne[1];
    row /= layout.ne[1];
    return i1*layout.nb[1] + (row % layout.ne[2])*layout.nb[2] + (row / layout.ne[2])*layout.nb[3];
}

static __device__ uint8_t nvfp4_scale_code(float value) {
    if (!(value > 0.0f)) {
        return 0;
    }
    const uint32_t bits = __float_as_uint(fminf(value, 448.0f));
    int exponent = int((bits >> 23) & 255) - 127 + 7;
    if (exponent <= 0) {
        return (uint8_t) min(7, int(value*512.0f + 0.5f));
    }
    if (exponent >= 15) {
        return 0x7e;
    }
    int mantissa = ((bits >> 20) & 7) + ((bits >> 19) & 1);
    if (mantissa > 7) {
        mantissa = 0;
        if (++exponent >= 15) {
            return 0x7e;
        }
    }
    return (uint8_t) ((exponent << 3) | mantissa);
}

static __device__ int nvfp4_nearest(float value, float scale) {
    int best = 0;
    float error = fabsf(value);
    for (int i = 1; i < 16; ++i) {
        const float candidate = fabsf(__fsub_rn(__fmul_rn(scale, kvalues_mxfp4[i]), value));
        if (candidate < error) {
            best = i;
            error = candidate;
        }
    }
    return best;
}

template<typename src_t, typename idx_t, bool scatter, bool fp4_kv = false>
static __global__ void nvfp4_quantize_rows(
        const char * src, const char * indices, char * dst,
        nvfp4_layout source, nvfp4_layout index, nvfp4_layout target) {
    const int64_t row = blockIdx.x;
    const char * input = src + nvfp4_row_offset(row, source);
    size_t offset;
    if constexpr (scatter) {
        const int64_t i1 = row % source.ne[1];
        const int64_t i2 = (row / source.ne[1]) % source.ne[2];
        const int64_t i3 = row / (source.ne[1]*source.ne[2]);
        const idx_t selected = *(const idx_t *) (indices + i1*index.nb[0] + (i2 % index.ne[1])*index.nb[1] + (i3 % index.ne[2])*index.nb[2]);
        assert(selected >= 0 && selected < target.ne[1]);
        offset = selected*target.nb[1] + i2*target.nb[2] + i3*target.nb[3];
    } else {
        offset = nvfp4_row_offset(row, target);
    }
    char * output = dst + offset;
    __shared__ float maximum[128];
    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < source.ne[0]; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(ggml_cuda_cast<float>(*(const src_t *) (input + i*source.nb[0]))));
    }
    maximum[threadIdx.x] = amax;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            maximum[threadIdx.x] = fmaxf(maximum[threadIdx.x], maximum[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float master = fp4_kv ? 1.0f : (maximum[0] > 0.0f ? fmaxf(__fdiv_rn(maximum[0], 2688.0f), FLT_MIN) : 1.0f);
    if (!fp4_kv && threadIdx.x == 0) {
        *(float *) output = master;
    }
    auto * blocks = (block_nvfp4 *) (output + (fp4_kv ? 0 : sizeof(float)));
    for (int64_t group = threadIdx.x; group < source.ne[0]/16; group += blockDim.x) {
        float values[16];
        float local_max = 0.0f;
        for (int i = 0; i < 16; ++i) {
            values[i] = __fdiv_rn(ggml_cuda_cast<float>(*(const src_t *) (input + (group*16 + i)*source.nb[0])), master);
            local_max = fmaxf(local_max, fabsf(values[i]));
        }
        const float requested_scale = __fdiv_rn(local_max, 6.0f);
        const uint8_t code = fp4_kv ? (requested_scale >= 448.0f ? 0x7e :
                (requested_scale < 0.015625f ? uint8_t(requested_scale*512.0f + 0.5f) :
                 uint8_t(min(((__float_as_uint(requested_scale) + 0x80000u) >> 20) - 960u, 0x7eu)))) :
                nvfp4_scale_code(requested_scale);
        const float scale = ggml_cuda_ue4m3_to_fp32(code);
        block_nvfp4 & block = blocks[group/4];
        const int sub = group % 4;
        block.d[sub] = code;
        for (int i = 0; i < 8; ++i) {
            block.qs[sub*8 + i] = nvfp4_nearest(values[i], scale) | (nvfp4_nearest(values[i + 8], scale) << 4);
        }
    }
}

template<typename dst_t, bool fp4_kv = false>
static __global__ void nvfp4_dequantize_rows(const char * src, char * dst, nvfp4_layout source, nvfp4_layout target) {
    const int64_t row = blockIdx.x;
    const char * input = src + nvfp4_row_offset(row, source);
    char * output = dst + nvfp4_row_offset(row, target);
    const float master = fp4_kv ? 1.0f : *(const float *) input;
    const auto * blocks = (const block_nvfp4 *) (input + (fp4_kv ? 0 : sizeof(float)));
    for (int64_t i = threadIdx.x; i < source.ne[0]; i += blockDim.x) {
        const block_nvfp4 & block = blocks[i/64];
        const int sub = (i % 64)/16;
        const int j = i % 16;
        const uint8_t packed = block.qs[sub*8 + j % 8];
        const int code = j < 8 ? packed & 15 : packed >> 4;
        const float value = __fmul_rn(__fmul_rn(ggml_cuda_ue4m3_to_fp32(block.d[sub]), kvalues_mxfp4[code]), master);
        *(dst_t *) (output + i*target.nb[0]) = ggml_cuda_cast<dst_t>(value);
    }
}

static __global__ void nvfp4_copy_rows(const char * src, char * dst, nvfp4_layout source, nvfp4_layout target, size_t row_bytes) {
    const char * input = src + nvfp4_row_offset(blockIdx.x, source);
    char * output = dst + nvfp4_row_offset(blockIdx.x, target);
    for (size_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
        output[i] = input[i];
    }
}

void ggml_cuda_cpy_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(src->ne[0] == dst->ne[0] && src->ne[0] % QK_NVFP4 == 0);
    GGML_ASSERT(ggml_nrows(src) == ggml_nrows(dst) && ggml_nrows(src) <= INT_MAX);
    const auto source = nvfp4_tensor_layout(src);
    const auto target = nvfp4_tensor_layout(dst);
    const auto nrows = ggml_nrows(src);
    if (!nrows) {
        return;
    }
    if (src->type == dst->type) {
        nvfp4_copy_rows<<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, (char *) dst->data, source, target, ggml_row_size(src->type, src->ne[0]));
    } else if (dst->type == GGML_TYPE_FP4_KV) {
        if (src->type == GGML_TYPE_F32) {
            nvfp4_quantize_rows<float, int64_t, false, true><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, nullptr, (char *) dst->data, source, {}, target);
        } else {
            GGML_ASSERT(src->type == GGML_TYPE_F16);
            nvfp4_quantize_rows<half, int64_t, false, true><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, nullptr, (char *) dst->data, source, {}, target);
        }
    } else if (src->type == GGML_TYPE_FP4_KV && dst->type == GGML_TYPE_F32) {
        nvfp4_dequantize_rows<float, true><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, (char *) dst->data, source, target);
    } else if (src->type == GGML_TYPE_FP4_KV && dst->type == GGML_TYPE_F16) {
        nvfp4_dequantize_rows<half, true><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, (char *) dst->data, source, target);
    } else if (dst->type == GGML_TYPE_NVFP4) {
        if (src->type == GGML_TYPE_F32) {
            nvfp4_quantize_rows<float, int64_t, false><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, nullptr, (char *) dst->data, source, {}, target);
        } else if (src->type == GGML_TYPE_F16) {
            nvfp4_quantize_rows<half, int64_t, false><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, nullptr, (char *) dst->data, source, {}, target);
        } else {
            GGML_ABORT("unsupported NVFP4 source type");
        }
    } else if (dst->type == GGML_TYPE_F32) {
        nvfp4_dequantize_rows<float><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, (char *) dst->data, source, target);
    } else if (dst->type == GGML_TYPE_F16) {
        nvfp4_dequantize_rows<half><<<nrows, 128, 0, ctx.stream()>>>((const char *) src->data, (char *) dst->data, source, target);
    } else {
        GGML_ABORT("unsupported NVFP4 destination type");
    }
    CUDA_CHECK(cudaGetLastError());
}

template<typename src_t>
static void nvfp4_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * rows, ggml_tensor * dst) {
    const auto source = nvfp4_tensor_layout(src);
    const auto index = nvfp4_tensor_layout(rows);
    const auto target = nvfp4_tensor_layout(dst);
    if (rows->type == GGML_TYPE_I64) {
        nvfp4_quantize_rows<src_t, int64_t, true><<<ggml_nrows(src), 128, 0, ctx.stream()>>>((const char *) src->data, (const char *) rows->data, (char *) dst->data, source, index, target);
    } else {
        GGML_ASSERT(rows->type == GGML_TYPE_I32);
        nvfp4_quantize_rows<src_t, int32_t, true><<<ggml_nrows(src), 128, 0, ctx.stream()>>>((const char *) src->data, (const char *) rows->data, (char *) dst->data, source, index, target);
    }
}

void ggml_cuda_set_rows_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * rows, ggml_tensor * dst) {
    GGML_ASSERT(src->ne[0] == dst->ne[0] && src->ne[0] % QK_NVFP4 == 0 && ggml_nrows(src) <= INT_MAX);
    if (!ggml_nrows(src)) {
        return;
    }
    if (src->type == GGML_TYPE_F32) {
        nvfp4_set_rows<float>(ctx, src, rows, dst);
    } else if (src->type == GGML_TYPE_F16) {
        nvfp4_set_rows<half>(ctx, src, rows, dst);
    } else {
        GGML_ABORT("unsupported NVFP4 SET_ROWS source type");
    }
    CUDA_CHECK(cudaGetLastError());
}
