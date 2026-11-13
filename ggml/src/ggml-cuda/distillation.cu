#include "distillation.cuh"

static constexpr int head_tile = 4096;
static constexpr int head_threads = 256;

static __device__ bool head_better(float a, int ai, float b, int bi) {
    return a > b || (a == b && ai < bi);
}

static __device__ void head_sort(float * values, int * ids, int count) {
    for (int width = 2; width <= count; width *= 2) {
        for (int stride = width/2; stride > 0; stride /= 2) {
            for (int i = threadIdx.x; i < count; i += blockDim.x) {
                const int j = i ^ stride;
                if (j <= i) continue;
                const bool swap = (i & width) == 0 ?
                    head_better(values[j], ids[j], values[i], ids[i]) :
                    head_better(values[i], ids[i], values[j], ids[j]);
                if (swap) {
                    const float value = values[i];
                    const int id = ids[i];
                    values[i] = values[j]; ids[i] = ids[j];
                    values[j] = value; ids[j] = id;
                }
            }
            __syncthreads();
        }
    }
}

static __device__ double head_sum(double value, double * work) {
    work[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) work[threadIdx.x] += work[threadIdx.x + stride];
        __syncthreads();
    }
    const double result = work[0];
    __syncthreads();
    return result;
}

template<ggml_type type>
static __device__ float head_weight(const void * row, int col) {
    if constexpr (type == GGML_TYPE_F32) {
        return ((const float *) row)[col];
    } else if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[col]);
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const auto & block = ((const block_q4_0 *) row)[col/QK4_0];
        const int offset = col % QK4_0;
        const int q = (block.qs[offset % (QK4_0/2)] >> (offset >= QK4_0/2 ? 4 : 0)) & 15;
        return __half2float(block.d)*(q - 8);
    } else {
        const auto & block = ((const block_q8_0 *) row)[col/QK8_0];
        return __half2float(block.d)*block.qs[col % QK8_0];
    }
}

template<ggml_type type>
static __global__ void head_project_top_k(const char * weights, const float * hidden,
        double * output, int dim, int vocab, size_t weight_stride, int k,
        float temperature, float softcap, int tiles) {
    __shared__ float values[head_tile];
    __shared__ int ids[head_tile];
    __shared__ double work[head_threads];
    const int row = blockIdx.y;
    const int tile = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    hidden += int64_t(row)*dim;
    for (int i = warp; i < head_tile; i += head_threads/WARP_SIZE) {
        const int token = tile*head_tile + i;
        float dot = 0;
        if (token < vocab) {
            const void * weight = weights + size_t(token)*weight_stride;
            for (int col = lane; col < dim; col += WARP_SIZE) {
                dot = fmaf(head_weight<type>(weight, col), hidden[col], dot);
            }
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            if (softcap > 0) dot = softcap*tanhf(dot/softcap);
            values[i] = token < vocab ? dot : -INFINITY;
            ids[i] = token < vocab ? token : INT_MAX;
        }
    }
    __syncthreads();
    head_sort(values, ids, head_tile);
    const double maximum = values[0];
    double total = 0, tail = 0;
    for (int i = threadIdx.x; i < head_tile; i += blockDim.x) {
        const double p = exp((double(values[i]) - maximum)/temperature);
        total += p;
        if (i >= k) tail += p;
    }
    total = head_sum(total, work);
    tail = head_sum(tail, work);
    output += (int64_t(row)*tiles + tile)*(2*k + 3);
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        output[i] = ids[i];
        output[k + i] = values[i];
    }
    if (threadIdx.x == 0) {
        output[2*k] = maximum;
        output[2*k + 1] = log(total);
        output[2*k + 2] = log(tiles == 1 ? fmax(tail, 1e-300) : tail) - log(total);
    }
}

static __global__ void head_merge_top_k(const double * input, double * output,
        int groups, int k, int sort_count, float temperature) {
    __shared__ float values[1024];
    __shared__ int ids[1024];
    __shared__ double work[head_threads];
    const int group = blockIdx.x, row = blockIdx.y;
    const int next_groups = (groups + 1)/2;
    const int width = 2*k + 3;
    const double * a = input + (int64_t(row)*groups + 2*group)*width;
    const double * b = 2*group + 1 < groups ? a + width : nullptr;
    for (int i = threadIdx.x; i < sort_count; i += blockDim.x) {
        const double * src = i < k ? a : b;
        const int j = i < k ? i : i - k;
        values[i] = src && i < 2*k ? src[k + j] : -INFINITY;
        ids[i] = src && i < 2*k ? int(src[j]) : INT_MAX;
    }
    __syncthreads();
    head_sort(values, ids, sort_count);
    const double maximum = values[0];
    double discarded = 0;
    for (int i = k + threadIdx.x; i < sort_count; i += blockDim.x) {
        discarded += exp((double(values[i]) - maximum)/temperature);
    }
    discarded = head_sum(discarded, work);
    output += (int64_t(row)*next_groups + group)*width;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        output[i] = ids[i];
        output[k + i] = values[i];
    }
    if (threadIdx.x == 0) {
        const double a_scale = (a[2*k] - maximum)/temperature + a[2*k + 1];
        const double b_scale = b ? (b[2*k] - maximum)/temperature + b[2*k + 1] : -INFINITY;
        const double total = exp(a_scale) + exp(b_scale);
        const double tail = discarded + exp(a_scale + a[2*k + 2]) + (b ? exp(b_scale + b[2*k + 2]) : 0);
        output[2*k] = maximum;
        output[2*k + 1] = log(total);
        // Accumulate discarded mass directly; total minus selected loses small tails.
        output[2*k + 2] = log(next_groups == 1 ? fmax(tail, 1e-300) : tail) - log(total);
    }
}

void ggml_cuda_mul_mat_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * w = dst->src[0];
    const auto * h = dst->src[1];
    const int k = ggml_get_op_params_i32(dst, 0);
    const float temperature = ggml_get_op_params_f32(dst, 1);
    const float softcap = ggml_get_op_params_f32(dst, 2);
    const int tiles = (w->ne[1] + head_tile - 1)/head_tile;
    const size_t width = 2*k + 3;
    const int chunk_rows = std::min<int64_t>(h->ne[1], 8);
    ggml_cuda_pool_alloc<double> first(ctx.pool()), second(ctx.pool());
    double * result = (double *) dst->data;
    if (tiles > 1) first.alloc(size_t(chunk_rows)*tiles*width);
    if (tiles > 2) second.alloc(size_t(chunk_rows)*((tiles + 1)/2)*width);
    auto kernel = head_project_top_k<GGML_TYPE_F32>;
    switch (w->type) {
        case GGML_TYPE_F32: break;
        case GGML_TYPE_F16: kernel = head_project_top_k<GGML_TYPE_F16>; break;
        case GGML_TYPE_Q4_0: kernel = head_project_top_k<GGML_TYPE_Q4_0>; break;
        case GGML_TYPE_Q8_0: kernel = head_project_top_k<GGML_TYPE_Q8_0>; break;
        default: GGML_ABORT("unsupported fused head type");
    }
    int sort_count = 1;
    while (sort_count < 2*k) sort_count *= 2;
    for (int64_t row = 0; row < h->ne[1]; row += chunk_rows) {
        const int rows = std::min<int64_t>(chunk_rows, h->ne[1] - row);
        double * final = result + row*width;
        double * projected = tiles == 1 ? final : first.get();
        const float * hidden = (const float *) h->data + row*h->ne[0];
        const dim3 grid(tiles, rows);
        kernel<<<grid, head_threads, 0, ctx.stream()>>>(
                (const char *) w->data, hidden, projected, w->ne[0], w->ne[1], w->nb[1], k, temperature, softcap, tiles);
        int groups = tiles;
        bool use_second = true;
        while (groups > 1) {
            const int next_groups = (groups + 1)/2;
            double * target = next_groups == 1 ? final : use_second ? second.get() : first.get();
            head_merge_top_k<<<dim3(next_groups, rows), head_threads, 0, ctx.stream()>>>(
                    projected, target, groups, k, sort_count, temperature);
            projected = target;
            groups = next_groups;
            use_second = !use_second;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

template<bool maximum>
static __device__ double top_k_stats_reduce(double value, double * shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = maximum ? fmax(shared[threadIdx.x], shared[threadIdx.x + stride]) :
                    shared[threadIdx.x] + shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

static __global__ void top_k_stats_f32(const float * logits, const int32_t * ids,
        double * output, int vocab, int top_k, float temperature) {
    extern __shared__ uint32_t top_k_selected[];
    __shared__ double reductions[3][256];
    const int row = blockIdx.x;
    logits += int64_t(row)*vocab;
    ids += int64_t(row)*top_k;
    output += int64_t(row)*(top_k + 3);
    for (int slot = threadIdx.x; slot < (vocab + 31)/32; slot += blockDim.x) top_k_selected[slot] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        const int token = ids[i];
        assert(token >= 0 && token < vocab);
        atomicOr(top_k_selected + token/32, 1u << (token%32));
        output[i] = logits[token];
    }
    __syncthreads();
    double maximum = -INFINITY;
    for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
        maximum = fmax(maximum, isfinite(logits[token]) ? double(logits[token]) : INFINITY);
    }
    maximum = top_k_stats_reduce<true>(maximum, reductions[0]);
    double total = 0, tail = 0;
    for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
        const double p = exp((double(logits[token]) - maximum)/temperature);
        total += p;
        if (!(top_k_selected[token/32] & (1u << (token%32)))) tail += p;
    }
    total = top_k_stats_reduce<false>(total, reductions[1]);
    tail = top_k_stats_reduce<false>(tail, reductions[2]);
    if (threadIdx.x == 0) {
        output[top_k] = maximum;
        output[top_k + 1] = log(total);
        output[top_k + 2] = log(fmax(tail, 1e-300)) - log(total);
    }
}

void ggml_cuda_top_k_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * logits = dst->src[0];
    const auto * ids = dst->src[1];
    top_k_stats_f32<<<logits->ne[1], 256, size_t((logits->ne[0] + 31)/32)*sizeof(uint32_t), ctx.stream()>>>(
            static_cast<const float *>(logits->data), static_cast<const int32_t *>(ids->data),
            static_cast<double *>(dst->data), logits->ne[0], ids->ne[0], ggml_get_op_params_f32(dst, 0));
}

template<bool use_bitmap>
static __device__ bool distillation_selected(const int32_t * table, int capacity, int32_t token) {
    if constexpr (use_bitmap) return uint32_t(table[token / 32]) & (1u << (token % 32));
    unsigned slot = (unsigned(token)*2654435761u) & (capacity - 1);
    while (table[slot] != -1) {
        if (table[slot] == token) return true;
        slot = (slot + 1) & (capacity - 1);
    }
    return false;
}

template<bool backward, bool use_bitmap>
static __global__ void distillation_f32(const float * logits, const int32_t * ids,
        const float * logp, const float * weights, const float * upstream,
        float * output, int vocab, int top_k, int capacity, float temperature) {
    extern __shared__ int32_t selected[];
    // Keep separate buffers so consecutive block reductions cannot overwrite pending reads.
    __shared__ float reductions[5][WARP_SIZE];
    const int row = blockIdx.x;
    logits += int64_t(row)*vocab;
    ids += int64_t(row)*top_k;
    logp += int64_t(row)*(top_k + 1);
    output += backward ? int64_t(row)*vocab : row;
    if (weights[row] == 0) {
        for (int token = threadIdx.x; token < (backward ? vocab : 1); token += blockDim.x) output[token] = 0;
        return;
    }
    for (int slot = threadIdx.x; slot < capacity; slot += blockDim.x) selected[slot] = use_bitmap ? 0 : -1;
    __syncthreads();
    for (int index = threadIdx.x; index < top_k; index += blockDim.x) {
        const int32_t token = ids[index];
        assert(token >= 0 && token < vocab);
        if constexpr (use_bitmap) {
            atomicOr(reinterpret_cast<unsigned *>(selected) + token / 32, 1u << (token % 32));
        } else {
            unsigned slot = (unsigned(token)*2654435761u) & (capacity - 1);
            while (atomicCAS(selected + slot, -1, token) != -1) slot = (slot + 1) & (capacity - 1);
        }
    }
    __syncthreads();
    float maximum = -INFINITY, tail_maximum = -INFINITY;
    for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
        const float value = logits[token]/temperature;
        maximum = fmaxf(maximum, value);
        if (!distillation_selected<use_bitmap>(selected, capacity, token)) tail_maximum = fmaxf(tail_maximum, value);
    }
    maximum = block_reduce<block_reduce_method::MAX>(maximum, reductions[0]);
    tail_maximum = block_reduce<block_reduce_method::MAX>(tail_maximum, reductions[1]);
    float total = 0, tail = 0;
    for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
        const float value = logits[token]/temperature;
        total += expf(value - maximum);
        if (!distillation_selected<use_bitmap>(selected, capacity, token)) tail += expf(value - tail_maximum);
    }
    total = block_reduce<block_reduce_method::SUM>(total, reductions[2]);
    tail = block_reduce<block_reduce_method::SUM>(tail, reductions[3]);
    const float teacher_tail = expf(logp[top_k]);
    if constexpr (backward) {
        float mass = 0;
        for (int index = threadIdx.x; index < top_k; index += blockDim.x) mass += expf(logp[index]);
        mass = block_reduce<block_reduce_method::SUM>(mass, reductions[4]) + teacher_tail;
        const float factor = weights[row]*upstream[row]/temperature;
        for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
            const float value = logits[token]/temperature;
            const float conditional = distillation_selected<use_bitmap>(selected, capacity, token) ? 0 : expf(value - tail_maximum)/tail;
            output[token] = factor*(mass*expf(value - maximum)/total - teacher_tail*conditional);
        }
        __syncthreads();
        for (int index = threadIdx.x; index < top_k; index += blockDim.x) output[ids[index]] -= factor*expf(logp[index]);
    } else {
        const float normalizer = maximum + logf(total);
        float loss = 0;
        for (int index = threadIdx.x; index < top_k; index += blockDim.x) {
            const float student = logits[ids[index]]/temperature - normalizer;
            loss += expf(logp[index])*(logp[index] - student);
        }
        loss = block_reduce<block_reduce_method::SUM>(loss, reductions[4]);
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
    const int bitmap_words = (logits->ne[0] + 31)/32;
    // Bound the bitmap at 32 KiB; larger vocabularies keep the sparse hash table.
    const bool use_bitmap = bitmap_words <= 8192;
    if (use_bitmap) capacity = bitmap_words;
    const auto kernel = backward ? (use_bitmap ? distillation_f32<true, true> : distillation_f32<true, false>) :
                                   (use_bitmap ? distillation_f32<false, true> : distillation_f32<false, false>);
    const int threads = logits->ne[0] < 4096 ? WARP_SIZE : 16*WARP_SIZE;
    kernel<<<logits->ne[1], threads, capacity*sizeof(int32_t), ctx.stream()>>>(
            static_cast<const float *>(logits->data), static_cast<const int32_t *>(dst->src[1]->data),
            static_cast<const float *>(dst->src[2]->data), static_cast<const float *>(dst->src[3]->data),
            backward ? static_cast<const float *>(dst->src[4]->data) : nullptr,
            static_cast<float *>(dst->data), logits->ne[0], top_k, capacity, temperature);
    CUDA_CHECK(cudaGetLastError());
}
