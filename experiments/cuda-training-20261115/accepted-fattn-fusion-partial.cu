#include "fattn-back.cuh"
#include "convert.cuh"
#include "turbo-quant.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#endif

template<ggml_type type>
static __device__ __forceinline__ float fattn_back_load(const char * row, const int64_t i) {
    if constexpr (type == GGML_TYPE_F32) {
        return ((const float *) row)[i];
    } else if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[i]);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0 & block = ((const block_q4_0 *) row)[i/QK4_0];
        const int iq = i % QK4_0;
        const uint8_t packed = block.qs[iq % (QK4_0/2)];
        const int q = iq < QK4_0/2 ? packed & 0x0f : packed >> 4;
        return __half2float(block.d)*(q - 8);
    } else if constexpr (type == GGML_TYPE_TURBO3_0) {
        const block_turbo3_0 * blocks = (const block_turbo3_0 *) row;
        return turbo3_dequant_cuda(blocks + i/QK_TURBO3, i % QK_TURBO3);
    } else if constexpr (type == GGML_TYPE_TURBO4_0) {
        const block_turbo4_0 * blocks = (const block_turbo4_0 *) row;
        return turbo4_dequant_cuda(blocks + i/QK_TURBO4, i % QK_TURBO4);
    } else {
        static_assert(type == GGML_TYPE_COUNT, "unsupported FlashAttention backward type");
        return 0.0f;
    }
}

static __device__ __forceinline__ float fattn_back_reduce_sum(float value, float * shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    value = warp_reduce_sum(value);
    if (lane == 0) {
        shared[threadIdx.x/WARP_SIZE] = value;
    }
    __syncthreads();
    value = lane < blockDim.x/WARP_SIZE ? shared[lane] : 0.0f;
    value = warp_reduce_sum(value);
    __syncthreads();
    return value;
}

static __device__ __forceinline__ float fattn_back_reduce_max(float value, float * shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    value = warp_reduce_max(value);
    if (lane == 0) {
        shared[threadIdx.x/WARP_SIZE] = value;
    }
    __syncthreads();
    value = lane < blockDim.x/WARP_SIZE ? shared[lane] : -INFINITY;
    value = warp_reduce_max(value);
    __syncthreads();
    return value;
}

template<typename T, bool cache = false>
static __global__ void flash_attn_back_softmax(
        const float * scores, T * probabilities, const half * mask,
        const float * sinks, float * sink_probabilities, int64_t sink_index,
        int64_t M, int64_t N, uint64_t m_nb1, int causal, float scale) {
    const int64_t iq = blockIdx.x;
    const int64_t key_end = causal ? min(M, M - N + iq + 1) : M;
    const half * pm = mask ? (const half *) ((const char *) mask + iq*m_nb1) : nullptr;
    extern __shared__ float shared[];

    float * cached = shared + blockDim.x;
    float local_max = -INFINITY;
    for (int64_t ik = threadIdx.x; ik < key_end; ik += blockDim.x) {
        const float mv = pm ? __half2float(pm[ik]) : 0.0f;
        const float value = mv == -INFINITY ? -INFINITY : scores[ik + M*iq]*scale + mv;
        if constexpr (cache) {
            cached[ik] = value;
        }
        local_max = fmaxf(local_max, value);
    }
    if (threadIdx.x == 0 && sinks) {
        local_max = fmaxf(local_max, sinks[sink_index]);
    }
    const float max_score = fattn_back_reduce_max(local_max, shared);
    if (max_score == -INFINITY) {
        if (threadIdx.x == 0 && sink_probabilities) {
            sink_probabilities[iq] = 0.0f;
        }
        for (int64_t ik = threadIdx.x; ik < M; ik += blockDim.x) {
            probabilities[ik + M*iq] = ggml_cuda_cast<T>(0.0f);
        }
        return;
    }

    float local_sum = 0.0f;
    for (int64_t ik = threadIdx.x; ik < key_end; ik += blockDim.x) {
        float value;
        if constexpr (cache) {
            value = expf(cached[ik] - max_score);
            cached[ik] = value;
        } else {
            const float mv = pm ? __half2float(pm[ik]) : 0.0f;
            value = mv == -INFINITY ? 0.0f : expf(scores[ik + M*iq]*scale + mv - max_score);
        }
        local_sum += value;
    }
    if (threadIdx.x == 0 && sinks) {
        local_sum += expf(sinks[sink_index] - max_score);
    }
    const float inv_sum = 1.0f/fattn_back_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0 && sink_probabilities) {
        sink_probabilities[iq] = sinks ? expf(sinks[sink_index] - max_score)*inv_sum : 0.0f;
    }

    for (int64_t ik = threadIdx.x; ik < M; ik += blockDim.x) {
        float probability = 0.0f;
        if (ik < key_end) {
            if constexpr (cache) {
                probability = cached[ik]*inv_sum;
            } else {
                const float mv = pm ? __half2float(pm[ik]) : 0.0f;
                if (mv != -INFINITY) {
                    probability = expf(scores[ik + M*iq]*scale + mv - max_score)*inv_sum;
                }
            }
        }
        probabilities[ik + M*iq] = ggml_cuda_cast<T>(probability);
    }
}

template<typename T, int cache = 0>
static __global__ void flash_attn_back_softmax_grad(
        const float * dp, T * probabilities_ds, const float * sink_probabilities,
        float * grad_s, int64_t sink_index, int64_t M, int64_t N, float scale) {
    const int64_t iq = blockIdx.x;
    extern __shared__ float shared[];

    float probabilities[cache ? cache : 1];
    float gradients[cache ? cache : 1];
    float local_mean = 0.0f;
    if constexpr (cache > 0) {
#pragma unroll
        for (int i = 0; i < cache; ++i) {
            const int64_t ik = threadIdx.x + i*blockDim.x;
            if (ik < M) {
                const int64_t index = ik + M*iq;
                probabilities[i] = float(probabilities_ds[index]);
                gradients[i] = dp[index];
                local_mean += probabilities[i]*gradients[i];
            }
        }
    } else {
        for (int64_t ik = threadIdx.x; ik < M; ik += blockDim.x) {
            const int64_t index = ik + M*iq;
            local_mean += float(probabilities_ds[index])*dp[index];
        }
    }
    const float mean = fattn_back_reduce_sum(local_mean, shared);
    if (threadIdx.x == 0 && sink_probabilities && grad_s) {
        atomicAdd(grad_s + sink_index, -sink_probabilities[iq]*mean);
    }

    if constexpr (cache > 0) {
#pragma unroll
        for (int i = 0; i < cache; ++i) {
            const int64_t ik = threadIdx.x + i*blockDim.x;
            if (ik < M) {
                probabilities_ds[ik + M*iq] = ggml_cuda_cast<T>(probabilities[i]*(gradients[i] - mean)*scale);
            }
        }
    } else {
        for (int64_t ik = threadIdx.x; ik < M; ik += blockDim.x) {
            const int64_t index = ik + M*iq;
            const float probability = float(probabilities_ds[index]);
            probabilities_ds[index] = ggml_cuda_cast<T>(probability*(dp[index] - mean)*scale);
        }
    }
}

template<int DK, int DV>
static __global__ void flash_attn_back_fused_rows(
        const half * q, const half * k, const half * v, const half * d,
        const half * mask, const float * sinks, half * probabilities, half * ds,
        int M, int N, int HQ, int q_per_kv, int ldk, int ldv,
        uint64_t k_nb2, uint64_t v_nb2, uint64_t m_nb1, uint64_t m_nb2,
        int mask_heads, int causal, float scale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    constexpr int queries = 16;
    constexpr int threads = 256;
    const int warp = threadIdx.x/WARP_SIZE;
    const int iqh = blockIdx.y;
    const int ikh = iqh/q_per_kv;
    const int query_begin = queries*blockIdx.x;
    const half * pk = (const half *) ((const char *) k + ikh*k_nb2);
    const half * pv = (const half *) ((const char *) v + ikh*v_nb2);
    const half * pq = q + (size_t) DK*(query_begin + N*iqh);
    const half * pd = d + (size_t) DV*(iqh + HQ*query_begin);
    const half * pm = mask ? (const half *) ((const char *) mask + (iqh%mask_heads)*m_nb2) : nullptr;
    half * pp = probabilities + (size_t) M*(query_begin + N*iqh);
    half * pds = ds + (size_t) M*(query_begin + N*iqh);
    extern __shared__ float shared[];
    float * matrix = shared + threads;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    for (int key = 16*warp; key < M; key += 16*(threads/WARP_SIZE)) {
        wmma::fill_fragment(c, 0.0f);
        for (int h = 0; h < DK; h += 16) {
            wmma::load_matrix_sync(a, pk + key*ldk + h, ldk);
            wmma::load_matrix_sync(b, pq + h, DK);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(matrix + key, c, M, wmma::mem_col_major);
    }
    __syncthreads();

    // One warp handles a row while retaining the reference 256-lane sum order.
    for (int j = warp; j < queries; j += threads/WARP_SIZE) {
        const int lane = threadIdx.x%WARP_SIZE;
        const int iq = query_begin + j;
        const int end = causal ? min(M, M - N + iq + 1) : M;
        const half * row_mask = pm ? (const half *) ((const char *) pm + iq*m_nb1) : nullptr;
        float local_max = -INFINITY;
        for (int key = lane; key < end; key += WARP_SIZE) {
            const float mv = row_mask ? __half2float(row_mask[key]) : 0.0f;
            const float value = mv == -INFINITY ? -INFINITY : matrix[key + M*j]*scale + mv;
            matrix[key + M*j] = value;
            local_max = fmaxf(local_max, value);
        }
        if (lane == 0 && sinks) {
            local_max = fmaxf(local_max, sinks[iqh]);
        }
        const float maximum = warp_reduce_max(local_max);
        float local_sum[8] = {};
        if (maximum != -INFINITY) {
#pragma unroll
            for (int group = 0; group < 8; ++group) {
                for (int key = lane + group*WARP_SIZE; key < end; key += threads) {
                    const float value = expf(matrix[key + M*j] - maximum);
                    matrix[key + M*j] = value;
                    local_sum[group] += value;
                }
            }
            if (lane == 0 && sinks) {
                local_sum[0] += expf(sinks[iqh] - maximum);
            }
        }
        float partial = 0.0f;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            const float value = warp_reduce_sum(local_sum[group]);
            if (lane == group) {
                partial = value;
            }
        }
        const float sum = warp_reduce_sum(partial);
        const float inv_sum = sum != 0.0f ? 1.0f/sum : 0.0f;
        for (int key = lane; key < M; key += WARP_SIZE) {
            const float probability = key < end && maximum != -INFINITY ? matrix[key + M*j]*inv_sum : 0.0f;
            pp[key + M*j] = __float2half(probability);
        }
    }
    __syncthreads();

    for (int key = 16*warp; key < M; key += 16*(threads/WARP_SIZE)) {
        wmma::fill_fragment(c, 0.0f);
        for (int h = 0; h < DV; h += 16) {
            wmma::load_matrix_sync(a, pv + key*ldv + h, ldv);
            wmma::load_matrix_sync(b, pd + h, DV*HQ);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(matrix + key, c, M, wmma::mem_col_major);
    }
    __syncthreads();

    for (int j = warp; j < queries; j += threads/WARP_SIZE) {
        const int lane = threadIdx.x%WARP_SIZE;
        float local_mean[8] = {};
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            for (int key = lane + group*WARP_SIZE; key < M; key += threads) {
                local_mean[group] += __half2float(pp[key + M*j])*matrix[key + M*j];
            }
        }
        float partial = 0.0f;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            const float value = warp_reduce_sum(local_mean[group]);
            if (lane == group) {
                partial = value;
            }
        }
        const float mean = warp_reduce_sum(partial);
        for (int key = lane; key < M; key += WARP_SIZE) {
            const float probability = __half2float(pp[key + M*j]);
            pds[key + M*j] = __float2half(probability*(matrix[key + M*j] - mean)*scale);
        }
    }

#else
    GGML_UNUSED(q); GGML_UNUSED(k); GGML_UNUSED(v); GGML_UNUSED(d);
    GGML_UNUSED(mask); GGML_UNUSED(sinks); GGML_UNUSED(probabilities); GGML_UNUSED(ds);
    GGML_UNUSED(M); GGML_UNUSED(N); GGML_UNUSED(HQ); GGML_UNUSED(q_per_kv);
    GGML_UNUSED(ldk); GGML_UNUSED(ldv); GGML_UNUSED(k_nb2); GGML_UNUSED(v_nb2);
    GGML_UNUSED(m_nb1); GGML_UNUSED(m_nb2); GGML_UNUSED(mask_heads); GGML_UNUSED(causal); GGML_UNUSED(scale);
    NO_DEVICE_CODE;
#endif
}

static __global__ void flash_attn_back_q_pointers(
        const half * k, const half * ds, float * dq,
        const half ** keys, const half ** gradients, float ** outputs,
        int HQ, int q_per_kv, int DK, int M, int N, uint64_t k_nb2) {
    const int head = blockIdx.x*blockDim.x + threadIdx.x;
    if (head < HQ) {
        keys[head] = (const half *) ((const char *) k + (head/q_per_kv)*k_nb2);
        gradients[head] = ds + (size_t) M*N*head;
        outputs[head] = dq + (size_t) DK*N*head;
    }
}

static bool use_volta_f16_fattn_back(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];

    float max_bias;
    float logit_softcap;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    return ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_VOLTA &&
        k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
        max_bias == 0.0f && logit_softcap == 0.0f && q->ne[2] % k->ne[2] == 0 &&
        q->ne[0] % 8 == 0 && v->ne[0] % 8 == 0 && q->ne[1] > 0 && k->ne[1] > 0 &&
        k->ne[0] == q->ne[0] && v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2] &&
        k->ne[3] == q->ne[3] && v->ne[3] == q->ne[3] &&
        d->ne[0] == v->ne[0] && d->ne[1] == q->ne[2] && d->ne[2] == q->ne[1] && d->ne[3] == q->ne[3] &&
        (!dst->src[5] || (dst->src[5]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[5]) &&
            dst->src[5]->ne[0] == q->ne[2])) &&
        q->nb[0] == sizeof(float) && d->nb[0] == sizeof(float) &&
        k->nb[0] == sizeof(half) && v->nb[0] == sizeof(half) &&
        k->nb[1] >= q->ne[0]*sizeof(half) && v->nb[1] >= v->ne[0]*sizeof(half) &&
        k->nb[1] % (8*sizeof(half)) == 0 && v->nb[1] % (8*sizeof(half)) == 0 &&
        k->nb[1]/sizeof(half) <= INT_MAX && v->nb[1]/sizeof(half) <= INT_MAX;
}

static bool use_pascal_f32_fattn_back(const ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const float max_bias = ggml_get_op_params_f32(dst, 1);
    const float logit_softcap = ggml_get_op_params_f32(dst, 2);

    return cc >= GGML_CUDA_CC_PASCAL && cc < GGML_CUDA_CC_VOLTA &&
        (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) &&
        (v->type == GGML_TYPE_F16 || v->type == GGML_TYPE_F32) &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        q->ne[0] > 0 && q->ne[0] <= INT_MAX && v->ne[0] > 0 && v->ne[0] <= INT_MAX &&
        q->ne[1] > 0 && q->ne[1] <= INT_MAX && k->ne[1] > 0 && k->ne[1] <= INT_MAX &&
        k->ne[2] > 0 && q->ne[2] % k->ne[2] == 0 && q->ne[2] <= INT_MAX/v->ne[0] &&
        q->ne[3] <= INT_MAX && k->ne[0] == q->ne[0] && v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2] &&
        k->ne[3] == q->ne[3] && v->ne[3] == q->ne[3] &&
        d->ne[0] == v->ne[0] && d->ne[1] == q->ne[2] && d->ne[2] == q->ne[1] && d->ne[3] == q->ne[3] &&
        (!dst->src[5] || (dst->src[5]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[5]) &&
            dst->src[5]->ne[0] == q->ne[2])) &&
        // Two per-head FP32 matrices use at most 128 MiB of scratch.
        q->ne[1] <= (int64_t(16)*1024*1024)/k->ne[1];
}

static __global__ void fattn_back_pack_f32(
        const char * src, float * dst, int64_t ne0, int64_t ne1, int64_t ne2, int64_t count,
        uint64_t nb1, uint64_t nb2, uint64_t nb3) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const int64_t i1 = i/ne0 % ne1;
    const int64_t i2 = i/(ne0*ne1) % ne2;
    const int64_t i3 = i/(ne0*ne1*ne2);
    dst[i] = ((const float *) (src + i1*nb1 + i2*nb2 + i3*nb3))[i % ne0];
}

template<typename T>
static void fattn_back_convert(ggml_backend_cuda_context & ctx, const ggml_tensor * src, T * dst) {
    if constexpr (std::is_same_v<T, float>) {
        if (src->type == GGML_TYPE_F32) {
            if (ggml_is_contiguous(src)) {
                CUDA_CHECK(cudaMemcpyAsync(dst, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, ctx.stream()));
            } else {
                const int64_t count = ggml_nelements(src);
                fattn_back_pack_f32<<<(count + 255)/256, 256, 0, ctx.stream()>>>(
                    (const char *) src->data, dst, src->ne[0], src->ne[1], src->ne[2], count,
                    src->nb[1], src->nb[2], src->nb[3]);
                CUDA_CHECK(cudaGetLastError());
            }
            return;
        }
    }
    to_t_cuda_t<T> convert;
    to_t_nc_cuda_t<T> convert_nc;
    if constexpr (std::is_same_v<T, float>) {
        convert = ggml_get_to_fp32_cuda(src->type);
        convert_nc = ggml_get_to_fp32_nc_cuda(src->type);
    } else {
        convert = ggml_get_to_fp16_cuda(src->type);
        convert_nc = ggml_get_to_fp16_nc_cuda(src->type);
    }
    if (ggml_is_contiguous(src)) {
        GGML_ASSERT(convert);
        convert(src->data, dst, ggml_nelements(src), ctx.stream());
    } else {
        GGML_ASSERT(convert_nc);
        const size_t element_size = ggml_type_size(src->type);
        convert_nc(src->data, dst, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
            src->nb[1]/element_size, src->nb[2]/element_size, src->nb[3]/element_size, ctx.stream());
    }
}

template<typename T>
static void launch_gemm_fattn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];
    const ggml_tensor * mask = dst->src[4];
    const ggml_tensor * sinks = dst->src[5];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int grad_flags = ggml_get_op_params_i32(dst, 5);
    const bool need_q = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_Q);
    const bool need_k = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_K);
    const bool need_v = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_V);

    const int DK = q->ne[0];
    const int DV = v->ne[0];
    const int N  = q->ne[1];
    const int full_M = k->ne[1];
    const int begin_hint = ggml_get_op_params_i32(dst, 6);
    const int count_hint = ggml_get_op_params_i32(dst, 7);
    const bool bounded = !std::is_same_v<T, float> && begin_hint >= 0 && begin_hint%256 == 0 &&
        count_hint > 0 && count_hint <= full_M && begin_hint <= full_M - count_hint;
    const int key_begin = bounded ? begin_hint : 0;
    const int M = bounded ? count_hint : full_M;
    const int HQ = q->ne[2];
    const int HK = k->ne[2];
    const int B  = q->ne[3];
    const int q_per_kv = HQ/HK;
    const bool fp32 = std::is_same_v<T, float>;
    const int ldk = fp32 ? DK : k->nb[1]/sizeof(half);
    const int ldv = fp32 ? DV : v->nb[1]/sizeof(half);
    const cudaDataType_t data_type = fp32 ? CUDA_R_32F : CUDA_R_16F;
    const cublasGemmAlgo_t algorithm = fp32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    ggml_cuda_pool_alloc<T> q_data(ctx.pool(), ggml_nelements(q));
    ggml_cuda_pool_alloc<T> d_data(ctx.pool(), ggml_nelements(d));
    ggml_cuda_pool_alloc<T> k_data(ctx.pool(), fp32 ? ggml_nelements(k) : 0);
    ggml_cuda_pool_alloc<T> v_data(ctx.pool(), fp32 ? ggml_nelements(v) : 0);
    const bool need_sinks = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_SINKS);
    const bool fused_rows = !fp32 && bounded && B == 1 && M <= 1280 && M%16 == 0 && N%16 == 0 &&
        DK == 256 && DV == 256 && (need_q || need_k) && !(sinks && need_sinks) &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_VOLTA;
    const size_t matrix_count = (size_t) M*N;
    ggml_cuda_pool_alloc<float> matrix_f32(ctx.pool(), fused_rows ? 0 : matrix_count);
    ggml_cuda_pool_alloc<T> matrix_p(ctx.pool(), matrix_count*(fused_rows ? HQ : 1));
    ggml_cuda_pool_alloc<T> matrix_ds(ctx.pool(), fused_rows ? matrix_count*HQ : 0);
    ggml_cuda_pool_alloc<const half *> q_keys(ctx.pool(), fused_rows && need_q ? HQ : 0);
    ggml_cuda_pool_alloc<const half *> q_gradients(ctx.pool(), fused_rows && need_q ? HQ : 0);
    ggml_cuda_pool_alloc<float *> q_outputs(ctx.pool(), fused_rows && need_q ? HQ : 0);
    ggml_cuda_pool_alloc<T> matrix_q(ctx.pool(), bounded && need_q && !fused_rows ? (size_t) full_M*N : 0);
    ggml_cuda_pool_alloc<float> sink_probabilities(ctx.pool(), sinks ? N : 0);

    cudaStream_t stream = ctx.stream();
    fattn_back_convert(ctx, q, q_data.ptr);
    fattn_back_convert(ctx, d, d_data.ptr);
    if constexpr (std::is_same_v<T, float>) {
        fattn_back_convert(ctx, k, k_data.ptr);
        fattn_back_convert(ctx, v, v_data.ptr);
    }

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;
    const int threads = 256;
    const size_t offs_k = GGML_PAD(ggml_nelements(q)*sizeof(float), GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + GGML_PAD(ggml_nelements(k)*sizeof(float), GGML_MEM_ALIGN);
    const size_t offs_s = offs_v + GGML_PAD(ggml_nelements(v)*sizeof(float), GGML_MEM_ALIGN);
    float * grad_q = (float *) dst->data;
    float * grad_k = need_k ? (float *) ((char *) dst->data + offs_k) : nullptr;
    float * grad_v = need_v ? (float *) ((char *) dst->data + offs_v) : nullptr;
    float * grad_s = sinks && (!(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) || (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_SINKS)) ?
        (float *) ((char *) dst->data + offs_s) : nullptr;

    if (bounded) {
        if (grad_k) CUDA_CHECK(cudaMemsetAsync(grad_k, 0, ggml_nelements(k)*sizeof(float), stream));
        if (grad_v) CUDA_CHECK(cudaMemsetAsync(grad_v, 0, ggml_nelements(v)*sizeof(float), stream));
    }
    if constexpr (!std::is_same_v<T, float>) {
        if (fused_rows) {
            const int shared_bytes = (threads + 16*M)*sizeof(float);
            const half * pk = (const half *) k->data + key_begin*ldk;
            const half * pv = (const half *) v->data + key_begin*ldv;
            const half * pm = mask ? (const half *) mask->data + key_begin : nullptr;
            const dim3 blocks(N/16, HQ);
            CUDA_CHECK(cudaFuncSetAttribute(flash_attn_back_fused_rows<256, 256>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            flash_attn_back_fused_rows<256, 256><<<blocks, threads, shared_bytes, stream>>>(
                q_data.ptr, pk, pv, d_data.ptr, pm, sinks ? (const float *) sinks->data : nullptr,
                matrix_p.ptr, matrix_ds.ptr, M, N, HQ, q_per_kv, ldk, ldv, k->nb[2], v->nb[2],
                mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->ne[2] : 1,
                ggml_flash_attn_ext_get_causal(dst) ? 1 : 0, scale);
            CUDA_CHECK(cudaGetLastError());
            if (need_q) {
                flash_attn_back_q_pointers<<<(HQ + 31)/32, 32, 0, stream>>>(
                    pk, matrix_ds.ptr, grad_q, q_keys.ptr, q_gradients.ptr, q_outputs.ptr,
                    HQ, q_per_kv, DK, M, N, k->nb[2]);
                CUDA_CHECK(cudaGetLastError());
                CUBLAS_CHECK(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    DK, N, M, &alpha, (const void * const *) q_keys.ptr, data_type, ldk,
                    (const void * const *) q_gradients.ptr, data_type, M,
                    &beta_zero, (void * const *) q_outputs.ptr, CUDA_R_32F, DK,
                    HQ, CUBLAS_COMPUTE_32F, algorithm));
            }
        }
    }
    for (int ib = 0; ib < B; ++ib) {
        for (int ikh = 0; ikh < HK; ++ikh) {
            const T * pk = fp32 ? k_data.ptr + (size_t) DK*M*(ikh + HK*ib) :
                (const T *) ((const char *) k->data + ikh*k->nb[2] + ib*k->nb[3] + key_begin*k->nb[1]);
            const T * pv = fp32 ? v_data.ptr + (size_t) DV*M*(ikh + HK*ib) :
                (const T *) ((const char *) v->data + ikh*v->nb[2] + ib*v->nb[3] + key_begin*v->nb[1]);
            float * pgk = grad_k ? grad_k + (size_t) DK*(full_M*(ikh + HK*ib) + key_begin) : nullptr;
            float * pgv = grad_v ? grad_v + (size_t) DV*(full_M*(ikh + HK*ib) + key_begin) : nullptr;

            for (int ih = 0; ih < q_per_kv; ++ih) {
                const int iqh = ikh*q_per_kv + ih;
                const T * pq = q_data.ptr + (size_t) DK*N*(iqh + HQ*ib);
                const T * pd = d_data.ptr + (size_t) DV*(iqh + HQ*N*ib);
                float * pgq = grad_q + (size_t) DK*N*(iqh + HQ*ib);
                const half * pm = mask ? (const half *) ((const char *) mask->data +
                    (iqh % mask->ne[2])*mask->nb[2] + (ib % mask->ne[3])*mask->nb[3] + key_begin*sizeof(half)) : nullptr;

                T * probabilities = matrix_p.ptr + (fused_rows ? matrix_count*iqh : 0);
                T * ds = fused_rows ? matrix_ds.ptr + matrix_count*iqh : probabilities;
                if (!fused_rows) {
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        M, N, DK, &alpha,
                        pk, data_type, ldk, pq, data_type, DK,
                        &beta_zero, matrix_f32.ptr, CUDA_R_32F, M,
                        CUBLAS_COMPUTE_32F, algorithm));

                    if (M <= 8192) {
                        flash_attn_back_softmax<T, true><<<N, threads, (threads + M)*sizeof(float), stream>>>(
                            matrix_f32.ptr, matrix_p.ptr, pm,
                            sinks ? (const float *) sinks->data : nullptr,
                            sinks ? sink_probabilities.ptr : nullptr, iqh,
                            M, N, mask ? mask->nb[1] : 0,
                            ggml_flash_attn_ext_get_causal(dst) ? 1 : 0, scale);
                    } else {
                        flash_attn_back_softmax<T><<<N, threads, threads*sizeof(float), stream>>>(
                            matrix_f32.ptr, matrix_p.ptr, pm,
                            sinks ? (const float *) sinks->data : nullptr,
                            sinks ? sink_probabilities.ptr : nullptr, iqh,
                            M, N, mask ? mask->nb[1] : 0,
                            ggml_flash_attn_ext_get_causal(dst) ? 1 : 0, scale);
                    }
                    CUDA_CHECK(cudaGetLastError());

                }

                if (need_v) {
                    const float * beta = ih == 0 ? &beta_zero : &beta_one;
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        DV, M, N, &alpha,
                        pd, data_type, DV*HQ, probabilities, data_type, M,
                        beta, pgv, CUDA_R_32F, DV,
                        CUBLAS_COMPUTE_32F, algorithm));
                }

                if (!fused_rows && (need_q || need_k || need_sinks)) {
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        M, N, DV, &alpha,
                        pv, data_type, ldv, pd, data_type, DV*HQ,
                        &beta_zero, matrix_f32.ptr, CUDA_R_32F, M,
                        CUBLAS_COMPUTE_32F, algorithm));

                    if (M <= 16*threads) {
                        flash_attn_back_softmax_grad<T, 16><<<N, threads, threads*sizeof(float), stream>>>(
                            matrix_f32.ptr, ds,
                            sinks ? sink_probabilities.ptr : nullptr,
                            sinks && need_sinks ? grad_s : nullptr,
                            iqh, M, N, scale);
                    } else {
                        flash_attn_back_softmax_grad<T><<<N, threads, threads*sizeof(float), stream>>>(
                            matrix_f32.ptr, ds,
                            sinks ? sink_probabilities.ptr : nullptr,
                            sinks && need_sinks ? grad_s : nullptr,
                            iqh, M, N, scale);
                    }
                    CUDA_CHECK(cudaGetLastError());
                }

                if (need_q && !fused_rows) {
                    const T * q_probabilities = ds;
                    if (bounded) {
                        // Keep the original GEMM reduction length and zero the masked keys.
                        CUDA_CHECK(cudaMemsetAsync(matrix_q.ptr, 0, (size_t) full_M*N*sizeof(T), stream));
                        CUDA_CHECK(cudaMemcpy2DAsync(matrix_q.ptr + key_begin, full_M*sizeof(T),
                            ds, M*sizeof(T), M*sizeof(T), N, cudaMemcpyDeviceToDevice, stream));
                        q_probabilities = matrix_q.ptr;
                    }
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        DK, N, full_M, &alpha,
                        pk - key_begin*ldk, data_type, ldk, q_probabilities, data_type, full_M,
                        &beta_zero, pgq, CUDA_R_32F, DK,
                        CUBLAS_COMPUTE_32F, algorithm));
                }

                if (need_k) {
                    const float * beta = ih == 0 ? &beta_zero : &beta_one;
                    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        DK, M, N, &alpha,
                        pq, data_type, DK, ds, data_type, M,
                        beta, pgk, CUDA_R_32F, DK,
                        CUBLAS_COMPUTE_32F, algorithm));
                }
            }
        }
    }
}

template<ggml_type type_K, ggml_type type_V>
static __global__ void flash_attn_back_kernel(
        const float * q, const char * k, const char * v, const float * d,
        const half * mask, const float * sinks, float * dst,
        int64_t DK, int64_t DV, int64_t N, int64_t M, int64_t HQ, int64_t HK, int64_t B,
        uint64_t q_nb1, uint64_t q_nb2, uint64_t q_nb3,
        uint64_t k_nb1, uint64_t k_nb2, uint64_t k_nb3,
        uint64_t v_nb1, uint64_t v_nb2, uint64_t v_nb3,
        uint64_t d_nb1, uint64_t d_nb2, uint64_t d_nb3,
        uint64_t m_nb1, uint64_t m_nb2, uint64_t m_nb3, int64_t MH, int64_t MB,
        float scale, float max_bias, float logit_softcap, int causal, int grad_flags,
        size_t offs_k, size_t offs_v, size_t offs_s) {
    const int64_t row = blockIdx.x;
    const int64_t iq = row % N;
    const int64_t iqh = (row/N) % HQ;
    const int64_t ib = row/(N*HQ);
    if (ib >= B) {
        return;
    }

    const int64_t ikh = iqh/(HQ/HK);
    const int64_t key_end = causal ? min(M, M - N + iq + 1) : M;
    const float * pq = (const float *) ((const char *) q + iq*q_nb1 + iqh*q_nb2 + ib*q_nb3);
    const float * pd = (const float *) ((const char *) d + iqh*d_nb1 + iq*d_nb2 + ib*d_nb3);
    const half * pm = mask ? (const half *) ((const char *) mask + iq*m_nb1 + (iqh%MH)*m_nb2 + (ib%MB)*m_nb3) : nullptr;

    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) HQ));
    const float m0 = exp2f(-max_bias/n_head_log2);
    const float m1 = exp2f(-(max_bias/2.0f)/n_head_log2);
    const float slope = max_bias > 0.0f ?
        (iqh < n_head_log2 ? powf(m0, iqh + 1) : powf(m1, 2*(iqh - n_head_log2) + 1)) : 1.0f;

    extern __shared__ float shared[];
    float local_max = -INFINITY;
    for (int64_t ik = threadIdx.x; ik < key_end; ik += blockDim.x) {
        const float mv = pm ? slope*__half2float(pm[ik]) : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }
        const char * pk = k + ik*k_nb1 + ikh*k_nb2 + ib*k_nb3;
        float dot = 0.0f;
        for (int64_t id = 0; id < DK; ++id) {
            dot += pq[id]*fattn_back_load<type_K>(pk, id);
        }
        float score = dot*scale;
        if (logit_softcap != 0.0f) {
            score = logit_softcap*tanhf(score/logit_softcap);
        }
        local_max = fmaxf(local_max, score + mv);
    }
    if (threadIdx.x == 0 && sinks) {
        local_max = fmaxf(local_max, sinks[iqh]);
    }
    const float max_score = fattn_back_reduce_max(local_max, shared);
    if (max_score == -INFINITY) {
        return;
    }

    float local_sum = 0.0f;
    float local_weighted_dp = 0.0f;
    for (int64_t ik = threadIdx.x; ik < key_end; ik += blockDim.x) {
        const float mv = pm ? slope*__half2float(pm[ik]) : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }
        const char * pk = k + ik*k_nb1 + ikh*k_nb2 + ib*k_nb3;
        const char * pv = v + ik*v_nb1 + ikh*v_nb2 + ib*v_nb3;
        float dot = 0.0f;
        for (int64_t id = 0; id < DK; ++id) {
            dot += pq[id]*fattn_back_load<type_K>(pk, id);
        }
        float score = dot*scale;
        if (logit_softcap != 0.0f) {
            score = logit_softcap*tanhf(score/logit_softcap);
        }
        const float weight = expf(score + mv - max_score);
        float dp = 0.0f;
        for (int64_t id = 0; id < DV; ++id) {
            dp += pd[id]*fattn_back_load<type_V>(pv, id);
        }
        local_sum += weight;
        local_weighted_dp += weight*dp;
    }
    if (threadIdx.x == 0 && sinks) {
        local_sum += expf(sinks[iqh] - max_score);
    }
    const float sum = fattn_back_reduce_sum(local_sum, shared);
    const float weighted_dp = fattn_back_reduce_sum(local_weighted_dp, shared);
    const float inv_sum = 1.0f/sum;
    const float mean = weighted_dp*inv_sum;

    float * grad_q = dst;
    float * grad_k = (!(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) || (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_K)) ?
        (float *) ((char *) dst + offs_k) : nullptr;
    float * grad_v = (!(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) || (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_V)) ?
        (float *) ((char *) dst + offs_v) : nullptr;
    const bool need_q = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_Q);
    const bool need_k = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_K);
    const bool need_v = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_V);

    for (int64_t ik = threadIdx.x; ik < key_end; ik += blockDim.x) {
        const float mv = pm ? slope*__half2float(pm[ik]) : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }
        const char * pk = k + ik*k_nb1 + ikh*k_nb2 + ib*k_nb3;
        const char * pv = v + ik*v_nb1 + ikh*v_nb2 + ib*v_nb3;
        float dot = 0.0f;
        for (int64_t id = 0; id < DK; ++id) {
            dot += pq[id]*fattn_back_load<type_K>(pk, id);
        }
        float score = dot*scale;
        float derivative = scale;
        if (logit_softcap != 0.0f) {
            const float t = tanhf(score/logit_softcap);
            score = logit_softcap*t;
            derivative *= 1.0f - t*t;
        }
        const float p = expf(score + mv - max_score)*inv_sum;
        float dp = 0.0f;
        for (int64_t id = 0; id < DV; ++id) {
            dp += pd[id]*fattn_back_load<type_V>(pv, id);
            if (need_v) {
                atomicAdd(grad_v + id + DV*(ik + M*(ikh + HK*ib)), p*pd[id]);
            }
        }
        const float ds = p*(dp - mean)*derivative;
        for (int64_t id = 0; id < DK; ++id) {
            if (need_q) {
                atomicAdd(grad_q + id + DK*(iq + N*(iqh + HQ*ib)), ds*fattn_back_load<type_K>(pk, id));
            }
            if (need_k) {
                atomicAdd(grad_k + id + DK*(ik + M*(ikh + HK*ib)), ds*pq[id]);
            }
        }
    }

    const bool need_sinks = !(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) ||
        (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_SINKS);
    if (threadIdx.x == 0 && sinks && need_sinks) {
        float * grad_s = (float *) ((char *) dst + offs_s);
        const float p_sink = expf(sinks[iqh] - max_score)*inv_sum;
        atomicAdd(grad_s + iqh, -p_sink*mean);
    }
}

template<ggml_type type_K, ggml_type type_V>
static void launch_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];
    const ggml_tensor * mask = dst->src[4];
    const ggml_tensor * sinks = dst->src[5];

    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int grad_flags = ggml_get_op_params_i32(dst, 5);

    const size_t offs_k = GGML_PAD(ggml_nelements(q)*sizeof(float), GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + GGML_PAD(ggml_nelements(k)*sizeof(float), GGML_MEM_ALIGN);
    const size_t offs_s = offs_v + GGML_PAD(ggml_nelements(v)*sizeof(float), GGML_MEM_ALIGN);
    cudaStream_t stream = ctx.stream();
    const bool gemm_f16 = use_volta_f16_fattn_back(ctx, dst);
    const bool gemm_f32 = use_pascal_f32_fattn_back(ctx, dst);
    if (gemm_f16 || gemm_f32) {
        // GEMMs overwrite Q/K/V; only the sink gradient uses atomic accumulation.
        if (sinks && (!(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID) || (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_SINKS))) {
            CUDA_CHECK(cudaMemsetAsync((char *) dst->data + offs_s, 0, ggml_nbytes(dst) - offs_s, stream));
        }
        if (gemm_f16) launch_gemm_fattn_back<half>(ctx, dst);
        else launch_gemm_fattn_back<float>(ctx, dst);
        return;
    }
    if (!(grad_flags & GGML_FLASH_ATTN_BACK_GRAD_VALID)) {
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    } else {
        if (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_Q) {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, offs_k, stream));
        }
        if (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_K) {
            CUDA_CHECK(cudaMemsetAsync((char *) dst->data + offs_k, 0, offs_v - offs_k, stream));
        }
        if (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_V) {
            CUDA_CHECK(cudaMemsetAsync((char *) dst->data + offs_v, 0, offs_s - offs_v, stream));
        }
        if (sinks && (grad_flags & GGML_FLASH_ATTN_BACK_GRAD_SINKS)) {
            CUDA_CHECK(cudaMemsetAsync((char *) dst->data + offs_s, 0, ggml_nbytes(dst) - offs_s, stream));
        }
    }

    const int threads = 128;
    const int blocks = q->ne[1]*q->ne[2]*q->ne[3];
    flash_attn_back_kernel<type_K, type_V><<<blocks, threads, threads*sizeof(float), stream>>>(
        (const float *) q->data, (const char *) k->data, (const char *) v->data, (const float *) d->data,
        mask ? (const half *) mask->data : nullptr, sinks ? (const float *) sinks->data : nullptr, (float *) dst->data,
        q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3],
        q->nb[1], q->nb[2], q->nb[3], k->nb[1], k->nb[2], k->nb[3],
        v->nb[1], v->nb[2], v->nb[3], d->nb[1], d->nb[2], d->nb[3],
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1,
        scale, max_bias, logit_softcap, ggml_flash_attn_ext_get_causal(dst) ? 1 : 0, grad_flags,
        offs_k, offs_v, offs_s);
}

static bool fattn_back_type_supported(const ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0;
}

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];
    return q->type == GGML_TYPE_F32 && d->type == GGML_TYPE_F32 &&
        q->nb[0] == sizeof(float) && d->nb[0] == sizeof(float) &&
        fattn_back_type_supported(k->type) && fattn_back_type_supported(v->type) &&
        k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
        k->ne[0] % ggml_blck_size(k->type) == 0 && v->ne[0] % ggml_blck_size(v->type) == 0 &&
        (!dst->src[4] || (dst->src[4]->type == GGML_TYPE_F16 && dst->src[4]->nb[0] == sizeof(half)));
}

template<ggml_type type_K>
static void launch_flash_attn_back_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[2]->type) {
        case GGML_TYPE_F32:      launch_flash_attn_back<type_K, GGML_TYPE_F32>     (ctx, dst); break;
        case GGML_TYPE_F16:      launch_flash_attn_back<type_K, GGML_TYPE_F16>     (ctx, dst); break;
        case GGML_TYPE_Q4_0:     launch_flash_attn_back<type_K, GGML_TYPE_Q4_0>    (ctx, dst); break;
        case GGML_TYPE_TURBO3_0: launch_flash_attn_back<type_K, GGML_TYPE_TURBO3_0>(ctx, dst); break;
        case GGML_TYPE_TURBO4_0: launch_flash_attn_back<type_K, GGML_TYPE_TURBO4_0>(ctx, dst); break;
        default: GGML_ABORT("unsupported FlashAttention backward V type: %s", ggml_type_name(dst->src[2]->type));
    }
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_back_supported(dst));
    switch (dst->src[1]->type) {
        case GGML_TYPE_F32:      launch_flash_attn_back_v<GGML_TYPE_F32>     (ctx, dst); break;
        case GGML_TYPE_F16:      launch_flash_attn_back_v<GGML_TYPE_F16>     (ctx, dst); break;
        case GGML_TYPE_Q4_0:     launch_flash_attn_back_v<GGML_TYPE_Q4_0>    (ctx, dst); break;
        case GGML_TYPE_TURBO3_0: launch_flash_attn_back_v<GGML_TYPE_TURBO3_0>(ctx, dst); break;
        case GGML_TYPE_TURBO4_0: launch_flash_attn_back_v<GGML_TYPE_TURBO4_0>(ctx, dst); break;
        default: GGML_ABORT("unsupported FlashAttention backward K type: %s", ggml_type_name(dst->src[1]->type));
    }
}
