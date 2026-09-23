#pragma once

#include <cstdint>
#include <vector>

namespace den2moee {

struct SvdFactor {
    int rows = 0;
    int cols = 0;
    int logical_rank = 0;
    int storage_rank = 0;
    std::vector<float> u;
    std::vector<float> sigma;
    std::vector<float> v;
    float relative_error = 0.0f;
};

SvdFactor truncated_svd(const std::vector<float> & matrix, int rows, int cols,
                        int logical_rank, int storage_rank, uint64_t seed);

} // namespace den2moee
