#pragma once

#include <cuda_runtime.h>

#include <cstdint>

//
// Parallel, CUDA-graph-safe MoE route grouping.
//
// Final routes inside each expert are sorted by original route index,
// so the ordering matches the old deterministic single-thread builder.
//

static __global__ void ggml_cuda_moe_route_count(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ counts,
        int64_t total_routes,
        int64_t n_expert,
        int64_t n_used,
        int64_t ids_s0,
        int64_t ids_s1) {

    const int64_t route =
        (int64_t) blockIdx.x * blockDim.x +
        threadIdx.x;

    if (route >= total_routes) {
        return;
    }

    const int64_t used =
        route % n_used;

    const int64_t token =
        route / n_used;

    const int32_t expert =
        ids[
            used  * ids_s0 +
            token * ids_s1
        ];

    if (expert >= 0 &&
        expert < n_expert) {

        atomicAdd(
            &counts[expert],
            1
        );
    }
}


static __global__ void ggml_cuda_moe_route_prefix(
        int32_t * __restrict__ counts_cursor,
        int32_t * __restrict__ expert_offsets,
        int64_t n_expert) {

    if (blockIdx.x != 0 ||
        threadIdx.x != 0) {
        return;
    }

    int32_t sum = 0;

    expert_offsets[0] = 0;

    for (int64_t e = 0;
         e < n_expert;
         ++e) {

        const int32_t count =
            counts_cursor[e];

        //
        // Scatter cursor starts at expert begin.
        //
        counts_cursor[e] =
            sum;

        sum += count;

        expert_offsets[e + 1] =
            sum;
    }
}


static constexpr int ggml_cuda_moe_route_threads = 256;

static __global__ void ggml_cuda_moe_route_scatter(
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_offsets,
        int32_t * __restrict__ routes,
        int32_t * __restrict__ route_rank,
        int64_t total_routes,
        int64_t n_used,
        int64_t ids_s0,
        int64_t ids_s1) {
    if constexpr (ggml_cuda_get_physical_warp_size() != 32) {
        const int64_t expert = blockIdx.x;
        const int lane = threadIdx.x;
        __shared__ int32_t prefix[ggml_cuda_moe_route_threads];
        __shared__ int32_t written;
        if (lane == 0) written = expert_offsets[expert];
        __syncthreads();

        // A block writes each expert's routes in input order without atomic scatter.
        for (int64_t base = 0; base < total_routes; base += blockDim.x) {
            const int64_t route = base + lane;
            const bool match = route < total_routes &&
                ids[(route % n_used)*ids_s0 + (route / n_used)*ids_s1] == expert;
            prefix[lane] = match;
            __syncthreads();
            for (int stride = 1; stride < ggml_cuda_moe_route_threads; stride *= 2) {
                const int32_t add = lane >= stride ? prefix[lane - stride] : 0;
                __syncthreads();
                prefix[lane] += add;
                __syncthreads();
            }
            if (match) {
                const int32_t position = written + prefix[lane] - 1;
                routes[position] = (int32_t) route;
                if (route_rank) route_rank[route] = position - expert_offsets[expert];
            }
            __syncthreads();
            if (lane == 0) written += prefix[ggml_cuda_moe_route_threads - 1];
            __syncthreads();
        }
        return;
    }

    const int64_t expert = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int32_t end = expert_offsets[expert + 1];
    int32_t written = expert_offsets[expert];
    if (written == end) {
        return;
    }
    __shared__ int32_t warp_counts[ggml_cuda_moe_route_threads / 32];

    // Ballot ranks preserve input order without a block-wide prefix scan.
    for (int64_t base = 0; base < total_routes && written < end; base += blockDim.x) {
        const int64_t route = base + threadIdx.x;
        const bool match = route < total_routes &&
            ids[(route % n_used)*ids_s0 + (route / n_used)*ids_s1] == expert;
        const unsigned mask = __ballot_sync(0xffffffff, match);
        if (lane == 0) {
            warp_counts[warp] = __popc(mask);
        }
        __syncthreads();
        int32_t before = 0;
        int32_t count = 0;
        for (int i = 0; i < ggml_cuda_moe_route_threads / 32; ++i) {
            const int32_t n = warp_counts[i];
            if (i < warp) {
                before += n;
            }
            count += n;
        }
        __syncthreads();
        if (match) {
            const int32_t position = written + before + __popc(mask & ((1u << lane) - 1));
            routes[position] = (int32_t) route;
            if (route_rank) {
                route_rank[route] = position - expert_offsets[expert];
            }
        }
        written += count;
    }
}


static inline void ggml_cuda_moe_build_routes(
        cudaStream_t stream,
        const int32_t * ids,
        int32_t * expert_offsets,
        int32_t * cursor,
        int32_t * routes,
        int32_t * route_rank,
        int64_t n_expert,
        int64_t n_used,
        int64_t n_tokens,
        int64_t ids_s0,
        int64_t ids_s1) {

    const int64_t total_routes =
        n_used * n_tokens;

    constexpr int threads =
        ggml_cuda_moe_route_threads;

    const int64_t blocks =
        (
            total_routes +
            threads - 1
        ) /
        threads;

    CUDA_CHECK(
        cudaMemsetAsync(
            cursor,
            0,
            n_expert *
                sizeof(int32_t),
            stream
        )
    );

    if (route_rank) {
        CUDA_CHECK(
            cudaMemsetAsync(
                route_rank,
                0xff,
                total_routes *
                    sizeof(int32_t),
                stream
            )
        );
    }

    ggml_cuda_moe_route_count
        <<<blocks, threads, 0, stream>>>(
            ids,
            cursor,
            total_routes,
            n_expert,
            n_used,
            ids_s0,
            ids_s1
        );

    CUDA_CHECK(
        cudaGetLastError()
    );

    ggml_cuda_moe_route_prefix
        <<<1, 1, 0, stream>>>(
            cursor,
            expert_offsets,
            n_expert
        );

    CUDA_CHECK(
        cudaGetLastError()
    );

    ggml_cuda_moe_route_scatter
        <<<n_expert, threads, 0, stream>>>(
            ids,
            expert_offsets,
            routes,
            route_rank,
            total_routes,
            n_used,
            ids_s0,
            ids_s1
        );

    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaGetLastError()
    );
}
