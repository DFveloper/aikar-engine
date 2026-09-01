#include "fattn-back.cuh"
#include "turbo-quant.cuh"

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
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
}

static __device__ __forceinline__ float fattn_back_reduce_max(float value, float * shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float result = shared[0];
    __syncthreads();
    return result;
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
    float * grad_k = (float *) ((char *) dst + offs_k);
    float * grad_v = (float *) ((char *) dst + offs_v);
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
