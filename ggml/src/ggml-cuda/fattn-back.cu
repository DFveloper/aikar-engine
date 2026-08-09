#include "fattn-back.cuh"

template<typename T>
static __device__ __forceinline__ float fattn_back_load(const T * ptr) {
    return (float) *ptr;
}

template<>
__device__ __forceinline__ float fattn_back_load<half>(const half * ptr) {
    return __half2float(*ptr);
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

template<typename K, typename V>
static __global__ void flash_attn_back_kernel(
        const float * q, const K * k, const V * v, const float * d,
        const half * mask, const float * sinks, float * dst,
        int64_t DK, int64_t DV, int64_t N, int64_t M, int64_t HQ, int64_t HK, int64_t B,
        uint64_t q_nb1, uint64_t q_nb2, uint64_t q_nb3,
        uint64_t k_nb0, uint64_t k_nb1, uint64_t k_nb2, uint64_t k_nb3,
        uint64_t v_nb0, uint64_t v_nb1, uint64_t v_nb2, uint64_t v_nb3,
        uint64_t d_nb1, uint64_t d_nb2, uint64_t d_nb3,
        uint64_t m_nb1, uint64_t m_nb2, uint64_t m_nb3, int64_t MH, int64_t MB,
        float scale, float max_bias, float logit_softcap, int causal,
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
        const K * pk = (const K *) ((const char *) k + ik*k_nb1 + ikh*k_nb2 + ib*k_nb3);
        float dot = 0.0f;
        for (int64_t id = 0; id < DK; ++id) {
            dot += pq[id]*fattn_back_load((const K *) ((const char *) pk + id*k_nb0));
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
        const K * pk = (const K *) ((const char *) k + ik*k_nb1 + ikh*k_nb2 + ib*k_nb3);
        const V * pv = (const V *) ((const char *) v + ik*v_nb1 + ikh*v_nb2 + ib*v_nb3);
        float dot = 0.0f;
        for (int64_t id = 0; id < DK; ++id) {
            dot += pq[id]*fattn_back_load((const K *) ((const char *) pk + id*k_nb0));
        }
        float score = dot*scale;
        if (logit_softcap != 0.0f) {
            score = logit_softcap*tanhf(score/logit_softcap);
        }
        const float weight = expf(score + mv - max_score);
        float dp = 0.0f;
        for (int64_t id = 0; id < DV; ++id) {
            dp += pd[id]*fattn_back_load((const V *) ((const char *) pv + id*v_nb0));
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

    for (int64_t ik = threadIdx.x; ik < key_end; ik += blockDim.x) {
        const float mv = pm ? slope*__half2float(pm[ik]) : 0.0f;
        if (mv == -INFINITY) {
            continue;
        }
        const K * pk = (const K *) ((const char *) k + ik*k_nb1 + ikh*k_nb2 + ib*k_nb3);
        const V * pv = (const V *) ((const char *) v + ik*v_nb1 + ikh*v_nb2 + ib*v_nb3);
        float dot = 0.0f;
        for (int64_t id = 0; id < DK; ++id) {
            dot += pq[id]*fattn_back_load((const K *) ((const char *) pk + id*k_nb0));
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
            dp += pd[id]*fattn_back_load((const V *) ((const char *) pv + id*v_nb0));
            atomicAdd(grad_v + id + DV*(ik + M*(ikh + HK*ib)), p*pd[id]);
        }
        const float ds = p*(dp - mean)*derivative;
        for (int64_t id = 0; id < DK; ++id) {
            atomicAdd(grad_q + id + DK*(iq + N*(iqh + HQ*ib)), ds*fattn_back_load((const K *) ((const char *) pk + id*k_nb0)));
            atomicAdd(grad_k + id + DK*(ik + M*(ikh + HK*ib)), ds*pq[id]);
        }
    }

    if (threadIdx.x == 0 && sinks) {
        float * grad_s = (float *) ((char *) dst + offs_s);
        const float p_sink = expf(sinks[iqh] - max_score)*inv_sum;
        atomicAdd(grad_s + iqh, -p_sink*mean);
    }
}

template<typename K, typename V>
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

    const size_t offs_k = GGML_PAD(ggml_nelements(q)*sizeof(float), GGML_MEM_ALIGN);
    const size_t offs_v = offs_k + GGML_PAD(ggml_nelements(k)*sizeof(float), GGML_MEM_ALIGN);
    const size_t offs_s = offs_v + GGML_PAD(ggml_nelements(v)*sizeof(float), GGML_MEM_ALIGN);
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

    const int threads = 128;
    const int blocks = q->ne[1]*q->ne[2]*q->ne[3];
    flash_attn_back_kernel<K, V><<<blocks, threads, threads*sizeof(float), stream>>>(
        (const float *) q->data, (const K *) k->data, (const V *) v->data, (const float *) d->data,
        mask ? (const half *) mask->data : nullptr, sinks ? (const float *) sinks->data : nullptr, (float *) dst->data,
        q->ne[0], v->ne[0], q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3],
        q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3],
        v->nb[0], v->nb[1], v->nb[2], v->nb[3], d->nb[1], d->nb[2], d->nb[3],
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        mask ? mask->ne[2] : 1, mask ? mask->ne[3] : 1,
        scale, max_bias, logit_softcap, ggml_flash_attn_ext_get_causal(dst) ? 1 : 0,
        offs_k, offs_v, offs_s);
}

bool ggml_cuda_flash_attn_back_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * d = dst->src[3];
    return q->type == GGML_TYPE_F32 && d->type == GGML_TYPE_F32 &&
        q->nb[0] == sizeof(float) && d->nb[0] == sizeof(float) &&
        (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16) &&
        (v->type == GGML_TYPE_F32 || v->type == GGML_TYPE_F16) &&
        k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type) &&
        (!dst->src[4] || (dst->src[4]->type == GGML_TYPE_F16 && dst->src[4]->nb[0] == sizeof(half)));
}

void ggml_cuda_flash_attn_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_back_supported(dst));
    if (dst->src[1]->type == GGML_TYPE_F16) {
        if (dst->src[2]->type == GGML_TYPE_F16) launch_flash_attn_back<half, half>(ctx, dst);
        else launch_flash_attn_back<half, float>(ctx, dst);
    } else {
        if (dst->src[2]->type == GGML_TYPE_F16) launch_flash_attn_back<float, half>(ctx, dst);
        else launch_flash_attn_back<float, float>(ctx, dst);
    }
}
