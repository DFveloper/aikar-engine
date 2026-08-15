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


static __global__ void ggml_cuda_moe_route_scatter(
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_offsets,
        int32_t * __restrict__ cursor,
        int32_t * __restrict__ routes,
        int32_t * __restrict__ route_rank,
        int64_t total_routes,
        int64_t n_expert,
        int64_t n_used,
        int64_t ids_s0,
        int64_t ids_s1) {

    const int64_t route =
        (int64_t) blockIdx.x *
            blockDim.x +
        threadIdx.x;

    if (route >=
        total_routes) {

        return;
    }

    const int64_t used =
        route %
        n_used;

    const int64_t token =
        route /
        n_used;

    const int32_t expert =
        ids[
            used  * ids_s0 +
            token * ids_s1
        ];

    if (expert < 0 ||
        expert >= n_expert) {

        return;
    }

    const int32_t pos =
        atomicAdd(
            &cursor[expert],
            1
        );

    routes[pos] =
        (int32_t) route;

    if (route_rank) {

        route_rank[route] =
            pos -
            expert_offsets[expert];
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
        256;

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
        <<<blocks, threads, 0, stream>>>(
            ids,
            expert_offsets,
            cursor,
            routes,
            total_routes,
            n_expert,
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