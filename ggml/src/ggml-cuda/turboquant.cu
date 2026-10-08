#include "turboquant.cuh"
#include "ggml-turboquant.h"

#include <algorithm>
#include <cfloat>

struct turboquant_strides {
    size_t value[4];
};

static turboquant_strides turboquant_tensor_strides(const ggml_tensor * tensor) {
    turboquant_strides strides {};
    if (tensor) {
        for (int axis = 0; axis < 4; ++axis) strides.value[axis] = tensor->nb[axis];
    }
    return strides;
}

static __device__ void turboquant_require(bool condition) {
    if (!condition) {
        printf("TurboQuant: invalid input, norm or cache index\n");
        asm("trap;");
    }
}

static __device__ float turboquant_float(const uint8_t * row) {
    return __uint_as_float(uint32_t(row[0]) | (uint32_t(row[1]) << 8) | (uint32_t(row[2]) << 16) | (uint32_t(row[3]) << 24));
}

static __device__ void turboquant_float_store(uint8_t * row, float value) {
    const uint32_t representation = __float_as_uint(value);
    for (int byte = 0; byte < 4; ++byte) row[byte] = uint8_t(representation >> (8*byte));
}

static __device__ int turboquant_code(const uint8_t * codes, int component, int bits) {
    const int offset = component*bits;
    const int shift = offset % 8;
    unsigned packed = codes[offset/8];
    if (shift + bits > 8) packed |= unsigned(codes[offset/8 + 1]) << 8;
    return (packed >> shift) & ((1 << bits) - 1);
}

template<typename scalar>
static __device__ scalar turboquant_warp_sum(scalar value) {
    for (int offset = 16; offset > 0; offset /= 2) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

template<typename scalar>
static __device__ scalar turboquant_block_sum(scalar value, scalar * workspace) {
    value = turboquant_warp_sum(value);
    if (threadIdx.x % 32 == 0) workspace[threadIdx.x/32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < blockDim.x/32 ? workspace[threadIdx.x] : scalar(0);
        value = turboquant_warp_sum(value);
        if (threadIdx.x == 0) workspace[0] = value;
    }
    __syncthreads();
    const scalar result = workspace[0];
    __syncthreads();
    return result;
}

static __device__ float turboquant_block_max(float value, float * workspace) {
    for (int offset = 16; offset > 0; offset /= 2) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    if (threadIdx.x % 32 == 0) workspace[threadIdx.x/32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < blockDim.x/32 ? workspace[threadIdx.x] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        if (threadIdx.x == 0) workspace[0] = value;
    }
    __syncthreads();
    const float result = workspace[0];
    __syncthreads();
    return result;
}

template<int total_bits, bool key>
static __global__ void turboquant_pack_kernel(const char * input, const char * indices, uint8_t * cache, const float * parameters,
    int head_dim, int heads, int tokens, int capacity, turboquant_strides input_stride, turboquant_strides index_stride, turboquant_strides cache_stride) {
    __shared__ float source[512];
    __shared__ int codes[512];
    __shared__ double residual[512];
    __shared__ double reduction[8];
    const int head = blockIdx.x % heads;
    const int token = blockIdx.x/heads % tokens;
    const int stream = blockIdx.x/(heads*tokens);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x/32;
    const int64_t slot = *(const int64_t *) (indices + token*index_stride.value[0] + stream*index_stride.value[1]);
    turboquant_require(slot >= 0 && slot < capacity);
    if (head == 0) {
        for (int previous = threadIdx.x; previous < token; previous += blockDim.x) {
            const int64_t previous_slot = *(const int64_t *) (indices + previous*index_stride.value[0] + stream*index_stride.value[1]);
            turboquant_require(previous_slot != slot);
        }
    }
    const float * row = (const float *) (input + head*input_stride.value[1] + token*input_stride.value[2] + stream*input_stride.value[3]);
    double norm_sq = 0.0;
    for (int component = threadIdx.x; component < head_dim; component += blockDim.x) {
        const float value = row[component];
        turboquant_require(isfinite(value));
        source[component] = value;
        norm_sq += double(value)*value;
    }
    const double norm = sqrt(turboquant_block_sum(norm_sq, reduction));
    turboquant_require(norm <= FLT_MAX);
    uint8_t * output = cache + head*cache_stride.value[1] + slot*cache_stride.value[2] + stream*cache_stride.value[3];
    constexpr int base_bits = total_bits - int(key);
    const int row_size = (key ? 8 : 4) + head_dim*total_bits/8;
    for (int byte = threadIdx.x; byte < row_size; byte += blockDim.x) output[byte] = 0;
    __syncthreads();
    if (norm == 0.0) return;
    if (threadIdx.x == 0) turboquant_float_store(output, float(norm));
    const float * rotation = parameters + (key ? 0 : head_dim*head_dim);
    const int book_offset = key ? (total_bits == 3 ? 0 : 4) : (total_bits == 3 ? 12 : 20);
    const float * book = parameters + 3*head_dim*head_dim + book_offset;
    for (int rotated_component = warp; rotated_component < head_dim; rotated_component += blockDim.x/32) {
        double product = 0.0;
        for (int component = lane; component < head_dim; component += 32) product += double(rotation[rotated_component*head_dim + component])*source[component]/norm;
        product = turboquant_warp_sum(product);
        if (lane == 0) {
            int code = 0;
            while (code + 1 < (1 << base_bits) && product > 0.5*(double(book[code]) + book[code + 1])) ++code;
            codes[rotated_component] = code;
        }
    }
    __syncthreads();
    uint8_t * packed_codes = output + (key ? 8 : 4);
    for (int byte = threadIdx.x; byte < head_dim*base_bits/8; byte += blockDim.x) {
        unsigned packed = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const int position = 8*byte + bit;
            packed |= ((codes[position/base_bits] >> (position % base_bits)) & 1) << bit;
        }
        packed_codes[byte] = uint8_t(packed);
    }
    if constexpr (!key) return;
    double residual_sq = 0.0;
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        double base = 0.0;
        for (int rotated_component = 0; rotated_component < head_dim; ++rotated_component) base += double(rotation[rotated_component*head_dim + column])*norm*book[codes[rotated_component]];
        const double error = double(source[column]) - base;
        residual[column] = error;
        residual_sq += error*error;
    }
    const double residual_norm = sqrt(turboquant_block_sum(residual_sq, reduction));
    turboquant_require(residual_norm <= FLT_MAX);
    if (threadIdx.x == 0) turboquant_float_store(output + 4, float(residual_norm));
    if (residual_norm == 0.0) return;
    const float * projection = parameters + 2*head_dim*head_dim;
    for (int projected_component = warp; projected_component < head_dim; projected_component += blockDim.x/32) {
        double product = 0.0;
        for (int component = lane; component < head_dim; component += 32) product += double(projection[projected_component*head_dim + component])*residual[component];
        product = turboquant_warp_sum(product);
        if (lane == 0) codes[projected_component] = product >= 0.0;
    }
    __syncthreads();
    uint8_t * packed_signs = packed_codes + head_dim*base_bits/8;
    for (int byte = threadIdx.x; byte < head_dim/8; byte += blockDim.x) {
        unsigned packed = 0;
        for (int bit = 0; bit < 8; ++bit) packed |= codes[8*byte + bit] << bit;
        packed_signs[byte] = uint8_t(packed);
    }
}

static __global__ void turboquant_query_kernel(const char * query, float * transformed, const float * parameters, int head_dim,
    int query_tokens, int query_heads, turboquant_strides query_stride) {
    __shared__ float source[512];
    const int query_token = blockIdx.x % query_tokens;
    const int head = blockIdx.x/query_tokens % query_heads;
    const int stream = blockIdx.x/(query_tokens*query_heads);
    const float * row = (const float *) (query + query_token*query_stride.value[1] + head*query_stride.value[2] + stream*query_stride.value[3]);
    for (int component = threadIdx.x; component < head_dim; component += blockDim.x) {
        source[component] = row[component];
        turboquant_require(isfinite(source[component]));
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x/32;
    for (int output_component = warp; output_component < 2*head_dim; output_component += blockDim.x/32) {
        const int matrix_offset = output_component < head_dim ? 0 : 2*head_dim*head_dim;
        const float * matrix_row = parameters + matrix_offset + (output_component % head_dim)*head_dim;
        float product = 0.0f;
        for (int component = lane; component < head_dim; component += 32) product = fmaf(matrix_row[component], source[component], product);
        product = turboquant_warp_sum(product);
        if (lane == 0) transformed[int64_t(blockIdx.x)*2*head_dim + output_component] = product;
    }
}

static __global__ void turboquant_scores_kernel(const float * transformed, const uint8_t * cache, const char * mask, float * scores,
    const float * parameters, ggml_turboquant_op_params options, int query_tokens, int query_heads, int kv_heads,
    int query_begin, int query_count, int mask_type, int mask_heads, int mask_streams, turboquant_strides cache_stride, turboquant_strides mask_stride) {
    const int local_query = blockIdx.y;
    const int flat_query = query_begin + local_query;
    const int token = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    const int lane = threadIdx.x % 32;
    if (token >= options.n_kv_max || local_query >= query_count) return;
    const int query_token = flat_query % query_tokens;
    const int head = flat_query/query_tokens % query_heads;
    const int stream = flat_query/(query_tokens*query_heads);
    float mask_value = 0.0f;
    if (mask) {
        const char * address = mask + token*mask_stride.value[0] + query_token*mask_stride.value[1] + (head % mask_heads)*mask_stride.value[2] + (stream % mask_streams)*mask_stride.value[3];
        mask_value = mask_type == GGML_TYPE_F16 ? __half2float(*(const half *) address) : *(const float *) address;
    }
    if (mask_value == -INFINITY) {
        if (lane == 0) scores[int64_t(local_query)*options.n_kv_max + token] = -INFINITY;
        return;
    }
    const uint8_t * row = cache + (head/(query_heads/kv_heads))*cache_stride.value[1] + token*cache_stride.value[2] + stream*cache_stride.value[3];
    const int bits = options.bits_k - 1;
    const uint8_t * signs = row + 8 + options.head_dim*bits/8;
    const float * book = parameters + 3*options.head_dim*options.head_dim + (options.bits_k == 3 ? 0 : 4);
    const float * query_row = transformed + int64_t(flat_query)*2*options.head_dim;
    float base = 0.0f, correction = 0.0f;
    for (int component = lane; component < options.head_dim; component += 32) {
        base = fmaf(query_row[component], book[turboquant_code(row + 8, component, bits)], base);
        const int sign = ((signs[component/8] >> (component % 8)) & 1) ? 1 : -1;
        correction = fmaf(query_row[options.head_dim + component], float(sign), correction);
    }
    base = turboquant_warp_sum(base);
    correction = turboquant_warp_sum(correction);
    if (lane == 0) {
        float score = options.scale*(turboquant_float(row)*base + turboquant_float(row + 4)*(1.2533141373155002512f/options.head_dim)*correction);
        if (options.logit_softcap > 0.0f) score = options.logit_softcap*tanhf(score/options.logit_softcap);
        if (options.max_bias > 0.0f) {
            const int head_log2 = 1 << int(floorf(log2f(float(query_heads))));
            const float slope = head < head_log2 ? powf(2.0f, -options.max_bias*(head + 1)/head_log2)
                : powf(2.0f, -0.5f*options.max_bias*(2*(head - head_log2) + 1)/head_log2);
            mask_value *= slope;
        }
        scores[int64_t(local_query)*options.n_kv_max + token] = score + mask_value;
    }
}

static __global__ void turboquant_values_kernel(float * scores, const uint8_t * cache, const float * sinks, char * output,
    const float * parameters, ggml_turboquant_op_params options, int query_tokens, int query_heads, int kv_heads,
    int query_begin, turboquant_strides cache_stride, turboquant_strides output_stride) {
    __shared__ float reduction[8];
    __shared__ float accumulated[512];
    const int local_query = blockIdx.x;
    const int flat_query = query_begin + local_query;
    const int query_token = flat_query % query_tokens;
    const int head = flat_query/query_tokens % query_heads;
    const int stream = flat_query/(query_tokens*query_heads);
    float * row_scores = scores + int64_t(local_query)*options.n_kv_max;
    float maximum = sinks ? sinks[head] : -INFINITY;
    for (int token = threadIdx.x; token < options.n_kv_max; token += blockDim.x) maximum = fmaxf(maximum, row_scores[token]);
    maximum = turboquant_block_max(maximum, reduction);
    float sum = sinks && threadIdx.x == 0 && maximum != -INFINITY ? expf(sinks[head] - maximum) : 0.0f;
    for (int token = threadIdx.x; token < options.n_kv_max; token += blockDim.x) {
        const float weight = row_scores[token] == -INFINITY ? 0.0f : expf(row_scores[token] - maximum);
        row_scores[token] = weight;
        sum += weight;
    }
    const float denominator = turboquant_block_sum(sum, reduction);
    const uint8_t * values = cache + (head/(query_heads/kv_heads))*cache_stride.value[1] + stream*cache_stride.value[3];
    const float * book = parameters + 3*options.head_dim*options.head_dim + (options.bits_v == 3 ? 12 : 20);
    for (int component = threadIdx.x; component < options.head_dim; component += blockDim.x) {
        float value = 0.0f;
        for (int token = 0; token < options.n_kv_max; ++token) {
            const float weight = row_scores[token];
            if (weight == 0.0f) continue;
            const uint8_t * row = values + token*cache_stride.value[2];
            value = fmaf(weight*turboquant_float(row), book[turboquant_code(row + 4, component, options.bits_v)], value);
        }
        accumulated[component] = denominator > 0.0f ? value/denominator : 0.0f;
    }
    __syncthreads();
    const float * rotation = parameters + options.head_dim*options.head_dim;
    float * destination = (float *) (output + head*output_stride.value[1] + query_token*output_stride.value[2] + stream*output_stride.value[3]);
    for (int column = threadIdx.x; column < options.head_dim; column += blockDim.x) {
        float value = 0.0f;
        for (int component = 0; component < options.head_dim; ++component) value = fmaf(rotation[component*options.head_dim + column], accumulated[component], value);
        destination[column] = value;
    }
}

static __global__ void turboquant_softmax_kernel(float * scores, const float * sinks, int n_kv, int query_tokens,
    int query_heads, int query_begin) {
    __shared__ float reduction[8];
    float * row = scores + int64_t(blockIdx.x)*n_kv;
    const int head = (query_begin + blockIdx.x)/query_tokens % query_heads;
    float maximum = sinks ? sinks[head] : -INFINITY;
    for (int token = threadIdx.x; token < n_kv; token += blockDim.x) maximum = fmaxf(maximum, row[token]);
    maximum = turboquant_block_max(maximum, reduction);
    float sum = sinks && threadIdx.x == 0 && maximum != -INFINITY ? expf(sinks[head] - maximum) : 0.0f;
    for (int token = threadIdx.x; token < n_kv; token += blockDim.x) {
        const float weight = row[token] == -INFINITY ? 0.0f : expf(row[token] - maximum);
        row[token] = weight;
        sum += weight;
    }
    const float denominator = turboquant_block_sum(sum, reduction);
    for (int token = threadIdx.x; token < n_kv; token += blockDim.x) row[token] = denominator > 0.0f ? row[token]/denominator : 0.0f;
}

static __global__ void turboquant_values_partial_kernel(const float * scores, const uint8_t * cache, float * partials,
    const float * parameters, ggml_turboquant_op_params options, int query_tokens, int query_heads, int kv_heads,
    int query_begin, int splits, turboquant_strides cache_stride) {
    const int local_query = blockIdx.y;
    const int flat_query = query_begin + local_query;
    const int head = flat_query/query_tokens % query_heads;
    const int stream = flat_query/(query_tokens*query_heads);
    const int token_begin = blockIdx.x*256;
    const int token_end = min(token_begin + 256, options.n_kv_max);
    const float * weights = scores + int64_t(local_query)*options.n_kv_max;
    const uint8_t * values = cache + (head/(query_heads/kv_heads))*cache_stride.value[1] + stream*cache_stride.value[3];
    const float * book = parameters + 3*options.head_dim*options.head_dim + (options.bits_v == 3 ? 12 : 20);
    float * destination = partials + (int64_t(local_query)*splits + blockIdx.x)*options.head_dim;
    for (int component = threadIdx.x; component < options.head_dim; component += blockDim.x) {
        float value = 0.0f;
        for (int token = token_begin; token < token_end; ++token) {
            const float weight = weights[token];
            if (weight == 0.0f) continue;
            const uint8_t * row = values + token*cache_stride.value[2];
            value = fmaf(weight*turboquant_float(row), book[turboquant_code(row + 4, component, options.bits_v)], value);
        }
        destination[component] = value;
    }
}

static __global__ void turboquant_values_merge_kernel(const float * partials, char * output, const float * parameters,
    int head_dim, int query_tokens, int query_heads, int query_begin, int splits, turboquant_strides output_stride) {
    __shared__ float accumulated[512];
    const int flat_query = query_begin + blockIdx.x;
    const int query_token = flat_query % query_tokens;
    const int head = flat_query/query_tokens % query_heads;
    const int stream = flat_query/(query_tokens*query_heads);
    const float * source = partials + int64_t(blockIdx.x)*splits*head_dim;
    for (int component = threadIdx.x; component < head_dim; component += blockDim.x) {
        float value = 0.0f;
        for (int split = 0; split < splits; ++split) value += source[split*head_dim + component];
        accumulated[component] = value;
    }
    __syncthreads();
    const float * rotation = parameters + head_dim*head_dim;
    float * destination = (float *) (output + head*output_stride.value[1] + query_token*output_stride.value[2] + stream*output_stride.value[3]);
    for (int column = threadIdx.x; column < head_dim; column += blockDim.x) {
        float value = 0.0f;
        for (int component = 0; component < head_dim; ++component) value = fmaf(rotation[component*head_dim + column], accumulated[component], value);
        destination[column] = value;
    }
}

void ggml_cuda_op_turboquant_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_turboquant_op_params options;
    memcpy(&options, dst->op_params, sizeof(options));
    const ggml_tensor * input = dst->src[0];
    const ggml_tensor * indices = dst->src[1];
    const float * parameters = (const float *) dst->src[3]->data;
    const int count = input->ne[1]*input->ne[2]*input->ne[3];
    const turboquant_strides input_stride = turboquant_tensor_strides(input);
    const turboquant_strides index_stride = turboquant_tensor_strides(indices);
    const turboquant_strides cache_stride = turboquant_tensor_strides(dst);
    const char * source = (const char *) input->data;
    const char * rows = (const char *) indices->data;
    uint8_t * destination = (uint8_t *) dst->data;
    if (options.key && options.bits_k == 3) {
        turboquant_pack_kernel<3, true><<<count, 256, 0, ctx.stream()>>>(source, rows, destination, parameters, options.head_dim, input->ne[1], input->ne[2], dst->ne[2], input_stride, index_stride, cache_stride);
    } else if (options.key) {
        turboquant_pack_kernel<4, true><<<count, 256, 0, ctx.stream()>>>(source, rows, destination, parameters, options.head_dim, input->ne[1], input->ne[2], dst->ne[2], input_stride, index_stride, cache_stride);
    } else if (options.bits_k == 3) {
        turboquant_pack_kernel<3, false><<<count, 256, 0, ctx.stream()>>>(source, rows, destination, parameters, options.head_dim, input->ne[1], input->ne[2], dst->ne[2], input_stride, index_stride, cache_stride);
    } else {
        turboquant_pack_kernel<4, false><<<count, 256, 0, ctx.stream()>>>(source, rows, destination, parameters, options.head_dim, input->ne[1], input->ne[2], dst->ne[2], input_stride, index_stride, cache_stride);
    }
}

void ggml_cuda_op_turboquant_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_turboquant_op_params options;
    memcpy(&options, dst->op_params, sizeof(options));
    const ggml_tensor * query = dst->src[0];
    const ggml_tensor * cache_k = dst->src[1];
    const ggml_tensor * cache_v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const float * sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    const float * parameters = (const float *) dst->src[5]->data;
    const int query_count = query->ne[1]*query->ne[2]*query->ne[3];
    const int chunk_size = std::min(query_count, 64);
    ggml_cuda_pool_alloc<float> transformed(ctx.pool(), size_t(query_count)*2*options.head_dim);
    ggml_cuda_pool_alloc<float> scores(ctx.pool(), size_t(chunk_size)*options.n_kv_max);
    const int splits = options.n_kv_max >= 512 ? (options.n_kv_max + 255)/256 : 0;
    ggml_cuda_pool_alloc<float> partials(ctx.pool());
    if (splits) partials.alloc(size_t(chunk_size)*splits*options.head_dim);
    turboquant_query_kernel<<<query_count, 256, 0, ctx.stream()>>>((const char *) query->data, transformed.ptr, parameters,
                options.head_dim, query->ne[1], query->ne[2], turboquant_tensor_strides(query));
    for (int begin = 0; begin < query_count; begin += chunk_size) {
        const int count = std::min(chunk_size, query_count - begin);
        const dim3 blocks((options.n_kv_max + 7)/8, count);
        if (options.n_kv_max > 0) {
            turboquant_scores_kernel<<<blocks, 256, 0, ctx.stream()>>>(transformed.ptr, (const uint8_t *) cache_k->data,
                mask ? (const char *) mask->data : nullptr, scores.ptr, parameters, options, query->ne[1], query->ne[2], cache_k->ne[1], begin, count,
                mask ? int(mask->type) : int(GGML_TYPE_F32), mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1,
                turboquant_tensor_strides(cache_k), turboquant_tensor_strides(mask));
        }
        if (splits) {
            turboquant_softmax_kernel<<<count, 256, 0, ctx.stream()>>>(scores.ptr, sinks, options.n_kv_max, query->ne[1], query->ne[2], begin);
            turboquant_values_partial_kernel<<<dim3(splits, count), 256, 0, ctx.stream()>>>(scores.ptr, (const uint8_t *) cache_v->data,
                partials.ptr, parameters, options, query->ne[1], query->ne[2], cache_v->ne[1], begin, splits, turboquant_tensor_strides(cache_v));
            turboquant_values_merge_kernel<<<count, 256, 0, ctx.stream()>>>(partials.ptr, (char *) dst->data, parameters,
                options.head_dim, query->ne[1], query->ne[2], begin, splits, turboquant_tensor_strides(dst));
        } else {
            turboquant_values_kernel<<<count, 256, 0, ctx.stream()>>>(scores.ptr, (const uint8_t *) cache_v->data, sinks, (char *) dst->data,
                parameters, options, query->ne[1], query->ne[2], cache_v->ne[1], begin, turboquant_tensor_strides(cache_v), turboquant_tensor_strides(dst));
        }
    }
}
