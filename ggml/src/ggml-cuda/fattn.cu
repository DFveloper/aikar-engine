#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

static ggml_tensor fattn_dense_mask_layout(const ggml_tensor * mask, int64_t tokens) {
    ggml_tensor dense = *mask;
    dense.type = GGML_TYPE_F16;
    dense.ne[0] = tokens;
    dense.nb[0] = sizeof(half);
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        dense.nb[dim] = dense.nb[dim - 1]*dense.ne[dim - 1];
    }
    return dense;
}

static __global__ void fattn_unpack_mask(const uint32_t * packed, half * dense, int64_t tokens, int64_t words, int64_t count) {
    const int64_t index = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t token = index % tokens;
    const uint32_t word = packed[index/tokens*words + token/32];
    dense[index] = __float2half_rn((word >> (token % 32)) & 1 ? -INFINITY : 0.0f);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
struct fattn_v_from_k_layout {
    int64_t nbq[4], nbk[4], nbv[4], nbm[4];
    int64_t queries, heads, kv_heads, tokens, mask_heads, mask_streams;
    int prefix, chunks, packed_mask;
    float scale, max_bias, softcap;
};

static __device__ float fattn_fp4_kv_value(const char * row, int d) {
    const auto & block = ((const block_nvfp4 *) row)[d/QK_NVFP4];
    const int sub = d % QK_NVFP4 / QK_NVFP4_SUB;
    const int j = d % QK_NVFP4_SUB;
    const uint8_t packed = block.qs[sub*(QK_NVFP4_SUB/2) + j % (QK_NVFP4_SUB/2)];
    const int code = j < QK_NVFP4_SUB/2 ? packed & 15 : packed >> 4;
    return __fmul_rn(ggml_cuda_ue4m3_to_fp32(block.d[sub]), kvalues_mxfp4[code]);
}

static __global__ void flash_attn_v_from_k(
        const char * q, const char * k, const char * v, const char * mask,
        const float * sinks, const float * scales, float * dst, fattn_v_from_k_layout p) {
    constexpr int dim = 512;
    constexpr int lanes = ggml_cuda_get_physical_warp_size();
    constexpr int values = dim/lanes;
    const int lane = threadIdx.x;
    const int64_t task = blockIdx.x;
    const int64_t head = task % p.heads;
    const int64_t query = task/p.heads % p.queries;
    const int64_t stream = task/(p.heads*p.queries);
    const int64_t kv_head = head/(p.heads/p.kv_heads);
    q += query*p.nbq[1] + head*p.nbq[2] + stream*p.nbq[3];
    k += kv_head*p.nbk[2] + stream*p.nbk[3];
    v += kv_head*p.nbv[2] + stream*p.nbv[3];
    if (mask) mask += query*p.nbm[1] + (head % p.mask_heads)*p.nbm[2] + (stream % p.mask_streams)*p.nbm[3];
    const uint32_t heads_log2 = 1u << uint32_t(floorf(log2f(float(p.heads))));
    const float slope = p.max_bias == 0 ? 1.0f : head < heads_log2 ?
            powf(powf(2.0f, -p.max_bias/heads_log2), head + 1) :
            powf(powf(2.0f, -p.max_bias/(2*heads_log2)), 2*(head - heads_log2) + 1);
    float query_values[values], norm[values], accum[values] = {};
#pragma unroll
    for (int i = 0; i < values; ++i) {
        const int d = lane + i*lanes;
        query_values[i] = __half2float(__float2half_rn(((const float *) q)[d]));
        norm[i] = scales[d];
    }
    float maximum = -INFINITY, sum = 0;
    const int64_t chunk_size = (p.tokens + p.chunks - 1)/p.chunks;
    const int64_t end = min(p.tokens, (blockIdx.y + 1)*chunk_size);
    for (int64_t token = blockIdx.y*chunk_size; token < end; ++token) {
        const float bias = !mask ? 0 : p.packed_mask ?
                ((((const uint32_t *) mask)[token/32] >> (token % 32)) & 1 ? -INFINITY : 0) :
                slope*__half2float(*(const half *) (mask + token*p.nbm[0]));
        if (bias == -INFINITY) continue;
        const char * key = k + token*p.nbk[1];
        const char * value = v + token*p.nbv[1];
        float key_values[values];
        float score = 0;
#pragma unroll
        for (int i = 0; i < values; ++i) {
            key_values[i] = fattn_fp4_kv_value(key, lane + i*lanes);
            score += __half2float(__float2half_rn(key_values[i]))*query_values[i];
        }
        score = warp_reduce_sum(score)*p.scale;
        if (p.softcap != 0) score = p.softcap*tanhf(score/p.softcap);
        score += bias;
        const float next_max = fmaxf(maximum, score);
        const float alpha = expf(maximum - next_max), weight = expf(score - next_max);
#pragma unroll
        for (int i = 0; i < values; ++i) {
            const int d = lane + i*lanes;
            const float reconstructed = d < p.prefix ? fattn_fp4_kv_value(value, d) : __fdiv_rn(key_values[i], norm[i]);
            accum[i] = accum[i]*alpha + weight*__half2float(__float2half_rn(reconstructed));
        }
        sum = sum*alpha + weight;
        maximum = next_max;
    }
    if (sinks && blockIdx.y == 0) {
        const float next_max = fmaxf(maximum, sinks[head]);
        const float alpha = expf(maximum - next_max);
#pragma unroll
        for (int i = 0; i < values; ++i) accum[i] *= alpha;
        sum = sum*alpha + expf(sinks[head] - next_max);
        maximum = next_max;
    }
    float * output = dst + task*(p.chunks == 1 ? dim : p.chunks*(dim + 2)) + blockIdx.y*(dim + 2);
#pragma unroll
    for (int i = 0; i < values; ++i) {
        output[lane + i*lanes] = p.chunks == 1 ? (sum == 0 ? 0 : accum[i]/sum) : accum[i];
    }
    if (p.chunks > 1 && lane == 0) { output[dim] = maximum; output[dim + 1] = sum; }
}

static __global__ void flash_attn_v_from_k_reduce(const float * partials, float * dst, int chunks) {
    constexpr int dim = 512;
    constexpr int lanes = ggml_cuda_get_physical_warp_size();
    const float * input = partials + int64_t(blockIdx.x)*chunks*(dim + 2);
    float maximum = -INFINITY, sum = 0, accum[dim/lanes] = {};
    for (int c = 0; c < chunks; ++c) {
        const float * part = input + c*(dim + 2);
        if (part[dim + 1] == 0) continue;
        const float next_max = fmaxf(maximum, part[dim]);
        const float alpha = expf(maximum - next_max), weight = expf(part[dim] - next_max);
#pragma unroll
        for (int i = 0; i < dim/lanes; ++i) accum[i] = accum[i]*alpha + part[threadIdx.x + i*lanes]*weight;
        sum = sum*alpha + part[dim + 1]*weight;
        maximum = next_max;
    }
#pragma unroll
    for (int i = 0; i < dim/lanes; ++i) dst[int64_t(blockIdx.x)*dim + threadIdx.x + i*lanes] = sum == 0 ? 0 : accum[i]/sum;
}

static void ggml_cuda_flash_attn_v_from_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0];
    const auto * k = dst->src[1];
    const auto * v = dst->src[2];
    const auto * mask = dst->src[3];
    fattn_v_from_k_layout p = {};
    for (int i = 0; i < 4; ++i) {
        p.nbq[i] = q->nb[i]; p.nbk[i] = k->nb[i]; p.nbv[i] = v->nb[i];
        if (mask) p.nbm[i] = mask->nb[i];
    }
    p.queries = q->ne[1]; p.heads = q->ne[2]; p.kv_heads = k->ne[2]; p.tokens = k->ne[1];
    p.mask_heads = mask ? mask->ne[2] : 1; p.mask_streams = mask ? mask->ne[3] : 1;
    p.prefix = v->ne[0];
    p.packed_mask = mask && mask->type == GGML_TYPE_I32;
    p.chunks = q->ne[1] == 1 ? std::min<int64_t>(32, (p.tokens + 255)/256) : 1;
    memcpy(&p.scale, dst->op_params, sizeof(float));
    memcpy(&p.max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&p.softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int64_t tasks = q->ne[1]*q->ne[2]*q->ne[3];
    ggml_cuda_pool_alloc<float> partials(ctx.pool());
    if (p.chunks > 1) partials.alloc(tasks*p.chunks*(512 + 2));
    flash_attn_v_from_k<<<dim3(tasks, p.chunks), WARP_SIZE, 0, ctx.stream()>>>(
            (const char *) q->data, (const char *) k->data, (const char *) v->data, mask ? (const char *) mask->data : nullptr,
            dst->src[4] ? (const float *) dst->src[4]->data : nullptr, (const float *) dst->src[6]->data,
            p.chunks > 1 ? partials.get() : (float *) dst->data, p);
    if (p.chunks > 1) flash_attn_v_from_k_reduce<<<tasks, WARP_SIZE, 0, ctx.stream()>>>(partials.get(), (float *) dst->data, p.chunks);
    CUDA_CHECK(cudaGetLastError());
}
#endif

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// one list per group of ncols1 queries: a column is selected if any query of the group can see it
template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int values_per_lane = 8;
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/warp_size];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        ggml_cuda_thread_mask_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            bool selected = false;
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
            }
            selected_warp[item] = ggml_cuda_ballot((int) selected);
            warp_count += ggml_cuda_popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/warp_size; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const ggml_cuda_thread_mask_t lane_mask = lane == 0 ? 0 : ((ggml_cuda_thread_mask_t)1 << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + ggml_cuda_popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & ((ggml_cuda_thread_mask_t)1 << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += ggml_cuda_popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(mask, indices, counts, n_queries, ncols1, n_kv_max, stream);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    const int64_t s31 = mask->nb[1] / sizeof(half);
    const int64_t s33 = mask->nb[3] / sizeof(half);
    const dim3 blocks_num((n_queries + ncols1 - 1)/ncols1, mask->ne[3], 1);
    const dim3 block_dim(256, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
    // the last group of queries is partial only if ncols1 does not divide n_queries
    GGML_ASSERT(ncols1 == 1 || ncols1 == 8);
    const auto kernel = ncols1 == 1       ? flash_attn_mask_to_sparse_indices<1, false> :
                        n_queries % 8 != 0 ? flash_attn_mask_to_sparse_indices<8, true>  :
                                             flash_attn_mask_to_sparse_indices<8, false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
        (const half *) mask->data, indices, counts, int(mask->ne[0]), n_queries, n_kv_max, s31, s33);
    CUDA_CHECK(cudaGetLastError());
#endif // !defined(GGML_USE_MUSA)
}

static bool ggml_cuda_flash_attn_ext_mma_f16_should_use_sparse(const int device, const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int cc = ggml_cuda_info().devices[device].cc;

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    const bool q8_kv = K->type == GGML_TYPE_Q8_KV && V->type == GGML_TYPE_Q8_KV;
    const bool volta_q8_sparse_shape = Q->ne[1] == 1 && Q->ne[0] == 256 && V->ne[0] == 256 &&
        gqa_ratio == 2 && dst->src[4] == nullptr;
    const bool volta_f16_sparse_shape = Q->ne[1] == 1 &&
        ((Q->ne[0] == 512 && V->ne[0] == 512 && gqa_ratio % 8 == 0) ||
         (Q->ne[0] == 576 && V->ne[0] == 512 && gqa_ratio % 16 == 0));
    const bool mma_sparse_available = q8_kv ? cc == GGML_CUDA_CC_VOLTA && volta_q8_sparse_shape :
        Q->ne[0] != 256 && (turing_mma_available(cc) || (volta_mma_available(cc) && volta_f16_sparse_shape));
    const bool volta_q8_single_token = q8_kv && cc == GGML_CUDA_CC_VOLTA &&
        volta_q8_sparse_shape && Q->ne[1] == 1;
    const int64_t sparse_min_kv = volta_q8_single_token ? 16384 : 4096;
    const bool sparse_kv_length = K->ne[1] >= std::max<int64_t>(sparse_min_kv, 2LL*n_kv_max);
    return GGML_CUDA_CC_IS_NVIDIA(cc) && mma_sparse_available &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        sparse_kv_length;
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#ifdef GGML_USE_MUSA
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // On CDNA only the DKQ=576 kernel variant with ncols2=16 is compiled with sparse support.
    if (amd_mfma_available(cc) && !ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(int(Q->ne[0]), int(V->ne[0]), 1, 16)) {
        return false;
    }

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    if (cc == GGML_CUDA_CC_VOLTA && ncols1 == 1 && ncols2 == 32) {
        return ggml_cuda_flash_attn_ext_mma_f16_should_use_sparse(ggml_cuda_get_device(), dst);
    }
    if (ncols2 == 32) {
        return false;
    }

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

    return ((GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc)) || amd_mfma_available(cc)) && dst->src[5] == nullptr &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, GGML_CUDA_CC_IS_NVIDIA(cc) ? 2*n_gather : 2LL*n_kv_max);
#endif // !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (DKQ == 256 && DV == 256 || DKQ == 512 && DV == 512 || DKQ == 576 && DV == 512) {
        if (cc == GGML_CUDA_CC_VOLTA && ggml_cuda_flash_attn_ext_mma_f16_should_use_sparse(ctx.device, dst)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, 32>(ctx, dst);
            return;
        }
    }
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (!volta_mma_available(cc) && Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = dst->src[5] ? dst->src[5]->ne[0] : K->ne[1];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    // a mask with a head dimension is not shared by the Q heads of a K/V head
    bool use_gqa_opt = mask && mask->ne[2] == 1 && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 40:
            GGML_ASSERT(V->ne[0] == 40);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 40,  40>(ctx, dst);
            break;
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 72:
            GGML_ASSERT(V->ne[0] == 72);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 72,  72>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_GB10) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        if (head_size == (D) && type_K == GGML_TYPE_##type_K_case && type_V == GGML_TYPE_##type_V_case) {                          \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

#define FATTN_VEC_CASES_EXT_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(512, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K_in, const ggml_type type_V_in) {
    const ggml_type type_K = type_K_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_K_in;
    const ggml_type type_V = type_V_in == GGML_TYPE_F32 ? GGML_TYPE_F16 : type_V_in;
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, BF16)

    FATTN_VEC_CASES_ALL_D(F16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q4_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q5_1, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(Q8_0, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(BF16, F8_E4M3)
    FATTN_VEC_CASES_ALL_D(F8_E4M3, F8_E4M3)

    FATTN_VEC_CASES_ALL_D(Q8_KV, Q8_KV)
    FATTN_VEC_CASE(512, Q8_KV, Q8_KV)

    FATTN_VEC_CASES_EXT_D(MXFP4,    MXFP4)
    FATTN_VEC_CASES_EXT_D(MXFP4,    F16)
    FATTN_VEC_CASES_EXT_D(F16,      MXFP4)
    FATTN_VEC_CASES_EXT_D(MXFP4,    TURBO3_0)
    FATTN_VEC_CASES_EXT_D(TURBO3_0, MXFP4)
    FATTN_VEC_CASES_EXT_D(MXFP4,    TURBO4_0)
    FATTN_VEC_CASES_EXT_D(TURBO4_0, MXFP4)
    FATTN_VEC_CASES_EXT_D(TURBO3_0, TURBO3_0)
    FATTN_VEC_CASES_EXT_D(TURBO3_0, TURBO4_0)
    FATTN_VEC_CASES_EXT_D(TURBO3_0, F16)
    FATTN_VEC_CASES_EXT_D(F16,      TURBO3_0)
    FATTN_VEC_CASES_EXT_D(TURBO4_0, TURBO3_0)
    FATTN_VEC_CASES_EXT_D(TURBO4_0, TURBO4_0)
    FATTN_VEC_CASES_EXT_D(TURBO4_0, F16)
    FATTN_VEC_CASES_EXT_D(F16,      TURBO4_0)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_KV:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_F8_E4M3:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
    if (dst->src[3] && dst->src[3]->type == GGML_TYPE_I32) {
        auto mask = fattn_dense_mask_layout(dst->src[3], dst->src[1]->ne[1]);
        ggml_tensor dense = *dst;
        dense.src[3] = &mask;
        return ggml_cuda_get_best_fattn_kernel(device, &dense);
    }
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const auto is_packed_vec_type = [](ggml_type type) {
        return type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0 || type == GGML_TYPE_MXFP4;
    };
    const bool packed_kv = is_packed_vec_type(K->type) || is_packed_vec_type(V->type);
    const bool packed_or_f16_k = is_packed_vec_type(K->type) || K->type == GGML_TYPE_F16;
    const bool packed_or_f16_v = is_packed_vec_type(V->type) || V->type == GGML_TYPE_F16;
    const ggml_tensor * kv_rows = dst->src[5];

    // with kv_rows the kernel iterates over the mask columns
    const int64_t n_kv = kv_rows ? kv_rows->ne[0] : K->ne[1];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, if the mask has a head dimension,
    //     or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && mask->ne[2] == 1 && max_bias == 0.0f && n_kv % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies && !packed_kv && K->type != GGML_TYPE_Q8_KV) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (packed_kv) {
        if (kv_rows || !packed_or_f16_k || !packed_or_f16_v ||
            (Q->ne[0] != 128 && Q->ne[0] != 256 && Q->ne[0] != 512) || K->ne[1] % FATTN_KQ_STRIDE != 0) {
            return BEST_FATTN_KERNEL_NONE;
        }
        return BEST_FATTN_KERNEL_VEC;
    }

    if (mask && mask->ne[2] != 1) {
        // the kv_rows mask is compacted per sequence, it has no head dimension
        if (kv_rows) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // the logit softcap kernels are compiled for these head sizes only
        float logit_softcap;
        memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
        if (logit_softcap != 0.0f && Q->ne[0] != 128 && Q->ne[0] != 256) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }

    // only the MMA kernel reads K/V through kv_rows
    if (kv_rows) {
        return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && mask && Q->ne[0] != 40 && Q->ne[0] != 72 ?
            BEST_FATTN_KERNEL_MMA_F16 : BEST_FATTN_KERNEL_NONE;
    }

    if (K->type == GGML_TYPE_Q8_KV && V->type == GGML_TYPE_Q8_KV) {
        if (ggml_cuda_flash_attn_ext_mma_f16_should_use_sparse(device, dst)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        static const char * disable_q8_kv_tile_env = getenv("GGML_CUDA_DISABLE_Q8_KV_TILE");
        static const bool disable_q8_kv_tile = disable_q8_kv_tile_env != nullptr && std::atoi(disable_q8_kv_tile_env) != 0;
        if (!disable_q8_kv_tile && cc == GGML_CUDA_CC_VOLTA && Q->ne[1] > 1 && (K->ne[0] == 256 || K->ne[0] == 512)) {
            return BEST_FATTN_KERNEL_TILE;
        }
        return BEST_FATTN_KERNEL_VEC;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc)) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (max_bias == 0.0f && (!mask || mask->ne[2] == 1) && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (ggml_cuda_flash_attn_ext_mma_f16_should_use_sparse(device, dst)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        static const char * disable_volta_d512_q8_vec_env = getenv("GGML_CUDA_DISABLE_VOLTA_FATTN_D512_Q8_VEC");
        static const bool disable_volta_d512_q8_vec =
            disable_volta_d512_q8_vec_env != nullptr && std::atoi(disable_volta_d512_q8_vec_env) != 0;
        if (!disable_volta_d512_q8_vec && cc == GGML_CUDA_CC_VOLTA && Q->ne[0] == 512 && Q->ne[1] == 1 &&
                gqa_ratio == 8 && K->ne[1] >= FATTN_KQ_STRIDE && logit_softcap == 0.0f &&
                K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // On CDNA, the sparse MMA kernel is used for DKQ=576 regardless of the head size heuristics.
    if (amd_mfma_available(cc) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);
    if (dst->src[6]) return ggml_nbytes(dst);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = K->type != GGML_TYPE_Q8_KV;
            need_f16_V = V->type != GGML_TYPE_Q8_KV;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (dst->src[6]) { ggml_cuda_flash_attn_v_from_k(ctx, dst); return; }
#endif
    if (dst->src[3] && dst->src[3]->type == GGML_TYPE_I32) {
        const auto * packed = dst->src[3];
        auto mask = fattn_dense_mask_layout(packed, dst->src[1]->ne[1]);
        ggml_cuda_pool_alloc<half> storage(ctx.pool(), ggml_nelements(&mask));
        mask.data = storage.get();
        const int64_t count = ggml_nelements(&mask);
        fattn_unpack_mask<<<(count + 255)/256, 256, 0, ctx.stream()>>>(
            (const uint32_t *) packed->data, storage.get(), mask.ne[0], packed->ne[0], count);
        CUDA_CHECK(cudaGetLastError());
        ggml_tensor dense = *dst;
        dense.src[3] = &mask;
        ggml_cuda_flash_attn_ext(ctx, &dense);
        return;
    }
    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst);
    switch (kernel) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    if (dst->src[6]) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
        return false;
#else
        return dst->src[0]->type == GGML_TYPE_F32 && dst->src[0]->nb[0] == sizeof(float) &&
            dst->src[1]->type == GGML_TYPE_FP4_KV && dst->src[2]->type == GGML_TYPE_FP4_KV &&
            dst->src[1]->ne[0] == 512 && dst->ne[0] == 512 && dst->src[2]->ne[0] < 512 &&
            dst->src[1]->ne[3] == dst->src[0]->ne[3] && dst->src[2]->ne[3] == dst->src[0]->ne[3] &&
            dst->src[6]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[6]) && ggml_nelements(dst->src[6]) == 512;
#endif
    }
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
