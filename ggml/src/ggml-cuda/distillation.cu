#include "distillation.cuh"

static __device__ bool distillation_selected(const int32_t * table, int capacity, int32_t token) {
    unsigned slot = (unsigned(token)*2654435761u) & (capacity - 1);
    while (table[slot] != -1) {
        if (table[slot] == token) return true;
        slot = (slot + 1) & (capacity - 1);
    }
    return false;
}

template<bool backward>
static __global__ void distillation_f32(const float * logits, const int32_t * ids,
        const float * logp, const float * weights, const float * upstream,
        float * output, int vocab, int top_k, int capacity, float temperature) {
    extern __shared__ int32_t selected[];
    const int row = blockIdx.x;
    logits += int64_t(row)*vocab;
    ids += int64_t(row)*top_k;
    logp += int64_t(row)*(top_k + 1);
    output += backward ? int64_t(row)*vocab : row;
    if (weights[row] == 0) {
        for (int token = threadIdx.x; token < (backward ? vocab : 1); token += WARP_SIZE) output[token] = 0;
        return;
    }
    for (int slot = threadIdx.x; slot < capacity; slot += WARP_SIZE) selected[slot] = -1;
    __syncthreads();
    for (int index = threadIdx.x; index < top_k; index += WARP_SIZE) {
        const int32_t token = ids[index];
        assert(token >= 0 && token < vocab);
        unsigned slot = (unsigned(token)*2654435761u) & (capacity - 1);
        while (atomicCAS(selected + slot, -1, token) != -1) slot = (slot + 1) & (capacity - 1);
    }
    __syncthreads();
    float maximum = -INFINITY, tail_maximum = -INFINITY;
    for (int token = threadIdx.x; token < vocab; token += WARP_SIZE) {
        const float value = logits[token]/temperature;
        maximum = fmaxf(maximum, value);
        if (!distillation_selected(selected, capacity, token)) tail_maximum = fmaxf(tail_maximum, value);
    }
    maximum = warp_reduce_max(maximum);
    tail_maximum = warp_reduce_max(tail_maximum);
    float total = 0, tail = 0;
    for (int token = threadIdx.x; token < vocab; token += WARP_SIZE) {
        const float value = logits[token]/temperature;
        total += expf(value - maximum);
        if (!distillation_selected(selected, capacity, token)) tail += expf(value - tail_maximum);
    }
    total = warp_reduce_sum(total);
    tail = warp_reduce_sum(tail);
    const float teacher_tail = expf(logp[top_k]);
    if constexpr (backward) {
        float mass = 0;
        for (int index = threadIdx.x; index < top_k; index += WARP_SIZE) mass += expf(logp[index]);
        mass = warp_reduce_sum(mass) + teacher_tail;
        const float factor = weights[row]*upstream[row]/temperature;
        for (int token = threadIdx.x; token < vocab; token += WARP_SIZE) {
            const float value = logits[token]/temperature;
            const float conditional = distillation_selected(selected, capacity, token) ? 0 : expf(value - tail_maximum)/tail;
            output[token] = factor*(mass*expf(value - maximum)/total - teacher_tail*conditional);
        }
        __syncwarp();
        for (int index = threadIdx.x; index < top_k; index += WARP_SIZE) output[ids[index]] -= factor*expf(logp[index]);
    } else {
        const float normalizer = maximum + logf(total);
        float loss = 0;
        for (int index = threadIdx.x; index < top_k; index += WARP_SIZE) {
            const float student = logits[ids[index]]/temperature - normalizer;
            loss += expf(logp[index])*(logp[index] - student);
        }
        loss = warp_reduce_sum(loss);
        if (threadIdx.x == 0) output[0] = weights[row]*(loss + teacher_tail*(logp[top_k] - tail_maximum - logf(tail) + normalizer));
    }
}

void ggml_cuda_distillation(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * logits = dst->src[0];
    const int top_k = dst->src[1]->ne[0];
    int capacity = 1;
    while (capacity < 2*top_k) capacity *= 2;
    const float temperature = ggml_get_op_params_f32(dst, 0);
    const bool backward = dst->op == GGML_OP_DISTILLATION_BACK;
    const auto kernel = backward ? distillation_f32<true> : distillation_f32<false>;
    kernel<<<logits->ne[1], WARP_SIZE, capacity*sizeof(int32_t), ctx.stream()>>>(
            static_cast<const float *>(logits->data), static_cast<const int32_t *>(dst->src[1]->data),
            static_cast<const float *>(dst->src[2]->data), static_cast<const float *>(dst->src[3]->data),
            backward ? static_cast<const float *>(dst->src[4]->data) : nullptr,
            static_cast<float *>(dst->data), logits->ne[0], top_k, capacity, temperature);
    CUDA_CHECK(cudaGetLastError());
}
