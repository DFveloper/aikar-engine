// Adapted from llama.cpp PR #22569 (matiaslin).
#include "pagedattn.cuh"

static __device__ float paged_sum(float value, float * scratch) {
    const int lane = threadIdx.x%32, warp = threadIdx.x/32;
    for (int delta = 16; delta; delta /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    value = threadIdx.x < blockDim.x/32 ? scratch[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int delta = 16; delta; delta /= 2) {
            value += __shfl_down_sync(0xffffffffu, value, delta);
        }
        if (lane == 0) { scratch[0] = value; }
    }
    __syncthreads();
    const float result = scratch[0];
    __syncthreads();
    return result;
}


template<bool quantized>
static __global__ void paged_write(
        const float * k, const float * v, char * cache, const int * slots,
        int hk, int ns, int bs, size_t token_stride, size_t head_stride, size_t page_stride) {
    const int h = blockIdx.x%hk, t = blockIdx.x/hk;
    const int i = threadIdx.x, d = blockDim.x;
    extern __shared__ float scratch[];
    for (int kv = 0; kv < 2; ++kv) {
        const float value = (kv ? v : k)[(t*hk + h)*d + i];
        float scale = 0.0f;
        int8_t quant = 0;
        if constexpr (quantized) {
            float amax = fabsf(value);
            for (int delta = 16; delta; delta /= 2) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
            }
            if (i%32 == 0) { scratch[i/32] = amax; }
            __syncthreads();
            scale = __fdiv_rn(fmaxf(scratch[2*(i/64)], scratch[2*(i/64) + 1]), 127.0f);
            const float inv = scale ? __fdiv_rn(1.0f, scale) : 0.0f;
            quant = (int8_t) max(-127, min(127, (int) roundf(value*inv)));
        }
        for (int seq = 0; seq < ns; ++seq) {
            const int slot = slots[t*ns + seq];
            if (slot < 0) { continue; }
            char * row = cache + (slot/bs)*page_stride + (h + kv*hk)*head_stride + (slot%bs)*token_stride;
            if constexpr (quantized) {
                block_q8_kv * block = (block_q8_kv *) row + i/64;
                if (i%64 == 0) { block->d = __float2half(scale); }
                block->qs[i%64] = quant;
            } else {
                ((half *) row)[i] = __float2half(value);
            }
        }
        if constexpr (quantized) {
            __syncthreads();
        }
    }
}

template<bool quantized>
static __device__ float paged_read(const char * row, int i) {
    if constexpr (quantized) {
        const block_q8_kv * block = (const block_q8_kv *) row + i/64;
        return __half2float(block->d)*block->qs[i%64];
    } else {
        return __half2float(((const half *) row)[i]);
    }
}

template<bool quantized, bool split = false, int dimension = 0, int head_group = 1>
static __global__ void paged_decode(
        const float * __restrict__ q, const char * __restrict__ cache, const int * __restrict__ table, const int * __restrict__ queries,
        float * __restrict__ out, int runtime_d, int nh, int hk, int nt, int np, int bs, int window, float scale, int n_kv, bool causal,
        size_t token_stride, size_t head_stride, size_t page_stride, int nsplits = 1) {
    const int d = dimension ? dimension : runtime_d;
    const int row_id = (blockIdx.x*4 + threadIdx.x/32)*head_group, lane = threadIdx.x%32;
    if (row_id >= nh*nt) { return; }
    const int h = row_id%nh, t = row_id/nh;
    const int hkv = h/(nh/hk), pos = queries[2*t], seq = queries[2*t + 1];
    float query[head_group][(dimension ? dimension : 1024)/32], acc[head_group][(dimension ? dimension : 1024)/32];
    #pragma unroll
    for (int head = 0; head < head_group; ++head) {
        #pragma unroll
        for (int j = 0; j < d/32; ++j) {
            query[head][j] = q[(row_id + head)*d + lane + 32*j]; acc[head][j] = 0.0f;
        }
    }
    float maxima[head_group], sums[head_group];
    #pragma unroll
    for (int head = 0; head < head_group; ++head) { maxima[head] = -INFINITY; sums[head] = 0.0f; }
    const int first = window ? max(0, pos - window + 1) : 0;
    const int span = causal && window ? min(window, n_kv) : n_kv;
    const int chunk = (span - 1)/nsplits + 1;
    const int begin = first + (split ? int(blockIdx.y)*chunk : 0);
    const int last = causal ? pos : n_kv - 1;
    const int end = split ? min(last, begin + chunk - 1) : last;
    for (int p = begin; p <= end; ++p) {
        const int ix = 2*(seq*np + p/bs), physical = table[ix];
        if (physical < 0 || !((uint32_t) table[ix + 1] & (1u << (p%bs)))) { continue; }
        const char * kr = cache + physical*page_stride + hkv*head_stride + (p%bs)*token_stride;
        const char * vr = kr + hk*head_stride;
        float keys[(dimension ? dimension : 1024)/32], values[(dimension ? dimension : 1024)/32];
        #pragma unroll
        for (int component = 0; component < d/32; ++component) {
            keys[component] = paged_read<quantized>(kr, lane + 32*component);
            values[component] = paged_read<quantized>(vr, lane + 32*component);
        }
        #pragma unroll
        for (int head = 0; head < head_group; ++head) {
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < d/32; ++j) { dot += query[head][j]*keys[j]; }
            for (int delta = 16; delta; delta /= 2) { dot += __shfl_xor_sync(0xffffffffu, dot, delta); }
            dot *= scale;
            const float next = fmaxf(maxima[head], dot), old = expf(maxima[head] - next), weight = expf(dot - next);
            #pragma unroll
            for (int j = 0; j < d/32; ++j) { acc[head][j] = acc[head][j]*old + weight*values[j]; }
            sums[head] = sums[head]*old + weight; maxima[head] = next;
        }
    }
    #pragma unroll
    for (int head = 0; head < head_group; ++head) {
        const size_t row = split ? (size_t(row_id + head)*nsplits + blockIdx.y)*(d + 2) : size_t(row_id + head)*d;
        #pragma unroll
        for (int j = 0; j < d/32; ++j) { out[row + lane + 32*j] = split ? acc[head][j] : acc[head][j]/sums[head]; }
        if constexpr (split) {
            if (lane == 0) { out[row + d] = maxima[head]; out[row + d + 1] = sums[head]; }
        }
    }
}

template<bool quantized, bool split = false>
static __global__ void paged_decode_block(
        const float * q, const char * cache, const int * table, const int * queries,
        float * out, int nh, int hk, int np, int bs, int window, float scale, int n_kv, bool causal,
        size_t token_stride, size_t head_stride, size_t page_stride, int nsplits = 1) {
    const int h = blockIdx.x%nh, t = blockIdx.x/nh, i = threadIdx.x, d = blockDim.x;
    const int hkv = h/(nh/hk), pos = queries[2*t], seq = queries[2*t + 1];
    const float query = q[(t*nh + h)*d + i];
    extern __shared__ float scratch[];
    float m = -INFINITY, sum = 0.0f, acc = 0.0f;
    const int first = window ? max(0, pos - window + 1) : 0;
    const int span = causal && window ? min(window, n_kv) : n_kv;
    const int chunk = (span - 1)/nsplits + 1;
    const int begin = first + (split ? int(blockIdx.y)*chunk : 0);
    const int last = causal ? pos : n_kv - 1;
    const int end = split ? min(last, begin + chunk - 1) : last;
    for (int p = begin; p <= end; ++p) {
        const int ix = 2*(seq*np + p/bs), physical = table[ix];
        if (physical < 0 || !((uint32_t) table[ix + 1] & (1u << (p%bs)))) { continue; }
        const char * kr = cache + physical*page_stride + hkv*head_stride + (p%bs)*token_stride;
        const char * vr = kr + hk*head_stride;
        const float dot = paged_sum(query*paged_read<quantized>(kr, i), scratch)*scale;
        const float next = fmaxf(m, dot), old = expf(m - next), weight = expf(dot - next);
        acc = acc*old + weight*paged_read<quantized>(vr, i);
        sum = sum*old + weight; m = next;
    }
    if constexpr (split) {
        const size_t row = (size_t(t*nh + h)*nsplits + blockIdx.y)*(d + 2);
        out[row + i] = acc;
        if (i == 0) { out[row + d] = m; out[row + d + 1] = sum; }
    } else {
        out[(t*nh + h)*d + i] = acc/sum;
    }
}

static __global__ void paged_combine(const float * partial, float * out, int nsplits) {
    const int d = blockDim.x, i = threadIdx.x;
    const size_t base = size_t(blockIdx.x)*nsplits*(d + 2);
    float m = -INFINITY;
    for (int s = 0; s < nsplits; ++s) { m = fmaxf(m, partial[base + size_t(s)*(d + 2) + d]); }
    float acc = 0.0f, sum = 0.0f;
    for (int s = 0; s < nsplits; ++s) {
        const size_t row = base + size_t(s)*(d + 2);
        const float weight = expf(partial[row + d] - m);
        acc += partial[row + i]*weight;
        sum += partial[row + d + 1]*weight;
    }
    out[size_t(blockIdx.x)*d + i] = acc/sum;
}

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    auto * cache = dst->src[3];
    const auto * table = dst->src[4];
    const int * slots = (const int *) dst->src[5]->data;
    const int * queries = (const int *) dst->src[6]->data;
    const int d = q->ne[0], nh = q->ne[1], hk = k->ne[1], nt = q->ne[2], ns = table->ne[2], np = table->ne[1];
    GGML_ASSERT(d <= 1024 && d%32 == 0);
    const int bs = ggml_get_op_params_i32(dst, 1), window = ggml_get_op_params_i32(dst, 2);
    const float scale = ggml_get_op_params_f32(dst, 0);
    const int max_tokens = ggml_get_op_params_i32(dst, 3);
    const bool causal = !ggml_get_op_params_i32(dst, 4);
    const int span = causal && window ? std::min(window, max_tokens) : max_tokens;
    const int nblocks = (nh*nt + 3)/4;
    const int target_blocks = 4*ggml_cuda_info().devices[ctx.device].nsm;
    const int nsplits = std::min((span - 1)/256 + 1, std::max(1, (target_blocks + nblocks - 1)/nblocks));
    const char * fused_setting = getenv("GGML_CUDA_PAGED_FUSED");
    const bool fused = (d == 256 || d == 512) && nt > 8 &&
            (fused_setting ? strcmp(fused_setting, "0") != 0 : nt >= 128);
    ggml_cuda_pool_alloc<float> partial(ctx.pool());
    float * output = (float *) dst->data;
    if (nsplits > 1) { output = partial.alloc(size_t(nt)*nh*nsplits*(d + 2)); }
    if (cache->type == GGML_TYPE_Q8_KV) {
        paged_write<true><<<hk*nt, d, d/32*sizeof(float), ctx.stream()>>>(
            (const float *) k->data, (const float *) v->data, (char *) cache->data, slots,
            hk, ns, bs, cache->nb[1], cache->nb[2], cache->nb[3]);
    } else {
        paged_write<false><<<hk*nt, d, 0, ctx.stream()>>>(
            (const float *) k->data, (const float *) v->data, (char *) cache->data, slots,
            hk, ns, bs, cache->nb[1], cache->nb[2], cache->nb[3]);
    }
#define PAGED_DECODE(quantized, split) \
    paged_decode_block<quantized, split><<<dim3(nh*nt, nsplits), d, d/32*sizeof(float), ctx.stream()>>>( \
        (const float *) q->data, (const char *) cache->data, (const int *) table->data, queries, \
        output, nh, hk, np, bs, window, scale, max_tokens, causal, cache->nb[1], cache->nb[2], cache->nb[3], nsplits)
#define PAGED_PREFILL(quantized, split, dimension) \
    paged_decode<quantized, split, dimension><<<dim3(nblocks, nsplits), 128, 0, ctx.stream()>>>( \
        (const float *) q->data, (const char *) cache->data, (const int *) table->data, queries, \
        output, d, nh, hk, nt, np, bs, window, scale, max_tokens, causal, cache->nb[1], cache->nb[2], cache->nb[3], nsplits)
#define PAGED_FUSED(quantized, split, dimension, head_group) \
    paged_decode<quantized, split, dimension, head_group><<<dim3((nh*nt/head_group + 3)/4, nsplits), 128, 0, ctx.stream()>>>( \
        (const float *) q->data, (const char *) cache->data, (const int *) table->data, queries, \
        output, d, nh, hk, nt, np, bs, window, scale, max_tokens, causal, cache->nb[1], cache->nb[2], cache->nb[3], nsplits)
#define PAGED_DISPATCH(quantized, split) \
    if (fused && d == 256 && (nh/hk)%2 == 0) { PAGED_FUSED(quantized, split, 256, 2); } \
    else if (fused && d == 512 && (nh/hk)%4 == 0) { PAGED_FUSED(quantized, split, 512, 4); } \
    else if (fused && d == 256) { PAGED_FUSED(quantized, split, 256, 1); } \
    else if (fused && d == 512) { PAGED_FUSED(quantized, split, 512, 1); } \
    else if (d == 256) { PAGED_PREFILL(quantized, split, 256); } \
    else if (d == 512) { PAGED_PREFILL(quantized, split, 512); } \
    else if (nt > 8) { PAGED_PREFILL(quantized, split, 0); } \
    else { PAGED_DECODE(quantized, split); }
    if (cache->type == GGML_TYPE_Q8_KV) {
        if (nsplits > 1) { PAGED_DISPATCH(true, true); }
        else { PAGED_DISPATCH(true, false); }
    } else {
        if (nsplits > 1) { PAGED_DISPATCH(false, true); }
        else { PAGED_DISPATCH(false, false); }
    }
#undef PAGED_DISPATCH
#undef PAGED_FUSED
#undef PAGED_PREFILL
#undef PAGED_DECODE
    if (nsplits > 1) { paged_combine<<<nh*nt, d, 0, ctx.stream()>>>(output, (float *) dst->data, nsplits); }
    CUDA_CHECK(cudaGetLastError());
}
