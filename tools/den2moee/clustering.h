#pragma once

#include <cstdint>
#include <vector>

namespace den2moee {

struct ClusterResult {
    std::vector<std::vector<int>> neuron_ids;
    std::vector<std::vector<float>> centers;
};

struct ExpertRoles {
    std::vector<int> ordered_clusters;
    std::vector<int> shared_clusters;
    std::vector<int> routed_clusters;
    std::vector<std::vector<int>> coverage_vectors;
    std::vector<int> coverage_scores;
    std::vector<float> mean_norms;
};

ClusterResult balanced_cluster(const std::vector<std::vector<float>> & features, int k, uint64_t seed,
                               int max_iter = 100, float tolerance = 1e-5f);

ExpertRoles rank_expert_roles(const std::vector<std::vector<float>> & features,
                              const ClusterResult & clusters, float coverage_ratio,
                              float coverage_percentile, int shared_experts);

} // namespace den2moee
