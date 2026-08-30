#pragma once

#include "common.cuh"

// Algorithm constants follow TheTom/llama-cpp-turboquant commit df7f5472949ce37cdc6a2155ef6b8836a8c10bac (MIT).

static __device__ __constant__ float turbo_centroids_3_cuda[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f,
};

static __device__ __constant__ float turbo_centroids_4_cuda[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f,
};

static __device__ __constant__ int8_t turbo_signs_1_cuda[128] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
};

static __device__ __constant__ int8_t turbo_signs_2_cuda[128] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
};

template <bool inverse>
static __device__ __forceinline__ float turbo_wht_128_cuda(float value, float * shared) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    value *= inverse ? turbo_signs_2_cuda[tid] : turbo_signs_1_cuda[tid];

#pragma unroll
    for (int step = 1; step < 32; step <<= 1) {
        const float other = __shfl_xor_sync(0xffffffff, value, step);
        value = lane & step ? other - value : value + other;
    }

    shared[tid] = value;
    __syncthreads();
    if ((tid & 63) < 32) {
        const float a = shared[tid];
        const float b = shared[tid + 32];
        shared[tid] = a + b;
        shared[tid + 32] = a - b;
    }
    __syncthreads();
    if (tid < 64) {
        const float a = shared[tid];
        const float b = shared[tid + 64];
        shared[tid] = a + b;
        shared[tid + 64] = a - b;
    }
    __syncthreads();

    value = shared[tid] * 0.08838834764831845f;
    value *= inverse ? turbo_signs_1_cuda[tid] : turbo_signs_2_cuda[tid];
    return value;
}

static __device__ __forceinline__ int turbo_nearest_3_cuda(float x) {
    if (x < -0.154496f) return 0;
    if (x < -0.092804f) return 1;
    if (x < -0.044243f) return 2;
    if (x <  0.000000f) return 3;
    if (x <  0.044243f) return 4;
    if (x <  0.092804f) return 5;
    if (x <  0.154496f) return 6;
    return 7;
}

static __device__ __forceinline__ int turbo_nearest_4_cuda(float x) {
    if (x < -0.212203f) return 0;
    if (x < -0.162947f) return 1;
    if (x < -0.127026f) return 2;
    if (x < -0.097164f) return 3;
    if (x < -0.070671f) return 4;
    if (x < -0.046174f) return 5;
    if (x < -0.022824f) return 6;
    if (x <  0.000000f) return 7;
    if (x <  0.022824f) return 8;
    if (x <  0.046174f) return 9;
    if (x <  0.070671f) return 10;
    if (x <  0.097164f) return 11;
    if (x <  0.127026f) return 12;
    if (x <  0.162947f) return 13;
    if (x <  0.212203f) return 14;
    return 15;
}

static __device__ __forceinline__ float turbo3_dequant_cuda(const block_turbo3_0 * block, int i) {
    const int ql = (block->qs[i/4] >> (2*(i % 4))) & 3;
    const int qh = (block->signs[i/8] >> (i % 8)) & 1;
    return __half2float(block->norm)*turbo_centroids_3_cuda[ql | (qh << 2)];
}

static __device__ __forceinline__ float turbo4_dequant_cuda(const block_turbo4_0 * block, int i) {
    const int q = (block->qs[i/2] >> (4*(i % 2))) & 15;
    return __half2float(block->norm)*turbo_centroids_4_cuda[q];
}
