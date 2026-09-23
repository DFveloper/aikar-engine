#include "clustering.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>

namespace den2moee {

static float cosine_distance(const std::vector<float> & a, const std::vector<float> & b) {
    float dot = 0.0f;
    float na = 0.0f;
    float nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na == 0.0f || nb == 0.0f) return 1.0f;
    return std::max(0.0f, 1.0f - dot / std::sqrt(na * nb));
}

static std::vector<float> normalized(const std::vector<float> & input) {
    float norm = 0.0f;
    for (float value : input) norm += value * value;
    norm = std::sqrt(norm);
    std::vector<float> result = input;
    if (norm > 0.0f) {
        for (float & value : result) value /= norm;
    }
    return result;
}

static std::vector<std::vector<float>> init_centers(const std::vector<std::vector<float>> & features,
                                                    int k, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::vector<float>> centers;
    centers.reserve(k);
    centers.push_back(features[rng() % features.size()]);

    while (static_cast<int>(centers.size()) < k) {
        std::vector<float> distances(features.size());
        float total = 0.0f;
        for (size_t i = 0; i < features.size(); ++i) {
            float best = std::numeric_limits<float>::max();
            for (const auto & center : centers) {
                best = std::min(best, cosine_distance(features[i], center));
            }
            distances[i] = best * best;
            total += distances[i];
        }
        if (total <= 0.0f) {
            centers.push_back(features[centers.size() % features.size()]);
            continue;
        }
        std::uniform_real_distribution<float> dist(0.0f, total);
        const float target = dist(rng);
        float cumulative = 0.0f;
        size_t chosen = features.size() - 1;
        for (size_t i = 0; i < distances.size(); ++i) {
            cumulative += distances[i];
            if (cumulative >= target) {
                chosen = i;
                break;
            }
        }
        centers.push_back(features[chosen]);
    }
    for (auto & center : centers) center = normalized(center);
    return centers;
}

ClusterResult balanced_cluster(const std::vector<std::vector<float>> & features, int k, uint64_t seed,
                               int max_iter, float tolerance) {
    if (features.empty() || k <= 0 || static_cast<int>(features.size()) % k != 0) {
        throw std::invalid_argument("balanced clustering requires n divisible by k");
    }
    const size_t dimension = features.front().size();
    if (dimension == 0) throw std::invalid_argument("clustering feature dimension is empty");
    for (const auto & feature : features) {
        if (feature.size() != dimension) throw std::invalid_argument("feature dimensions differ");
    }

    const int capacity = static_cast<int>(features.size()) / k;
    auto centers = init_centers(features, k, seed);
    std::vector<std::vector<int>> assignments(k);

    for (int iter = 0; iter < max_iter; ++iter) {
        struct Candidate {
            float distance;
            int neuron;
            int cluster;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(features.size() * k);
        for (int neuron = 0; neuron < static_cast<int>(features.size()); ++neuron) {
            for (int cluster = 0; cluster < k; ++cluster) {
                candidates.push_back({cosine_distance(features[neuron], centers[cluster]), neuron, cluster});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
            return std::tie(a.distance, a.neuron, a.cluster) < std::tie(b.distance, b.neuron, b.cluster);
        });

        assignments.assign(k, {});
        std::vector<bool> assigned(features.size(), false);
        for (const Candidate & candidate : candidates) {
            if (assigned[candidate.neuron] || static_cast<int>(assignments[candidate.cluster].size()) >= capacity) {
                continue;
            }
            assignments[candidate.cluster].push_back(candidate.neuron);
            assigned[candidate.neuron] = true;
        }
        for (int neuron = 0; neuron < static_cast<int>(features.size()); ++neuron) {
            if (assigned[neuron]) continue;
            for (int cluster = 0; cluster < k; ++cluster) {
                if (static_cast<int>(assignments[cluster].size()) < capacity) {
                    assignments[cluster].push_back(neuron);
                    assigned[neuron] = true;
                    break;
                }
            }
        }
        for (auto & ids : assignments) std::sort(ids.begin(), ids.end());

        float movement = 0.0f;
        for (int cluster = 0; cluster < k; ++cluster) {
            std::vector<float> mean(dimension, 0.0f);
            for (int neuron : assignments[cluster]) {
                for (size_t d = 0; d < dimension; ++d) mean[d] += features[neuron][d];
            }
            for (float & value : mean) value /= static_cast<float>(capacity);
            mean = normalized(mean);
            movement += cosine_distance(centers[cluster], mean);
            centers[cluster] = std::move(mean);
        }
        if (movement <= tolerance) break;
    }

    return {std::move(assignments), std::move(centers)};
}

static float percentile(std::vector<float> values, float percent) {
    if (values.empty()) throw std::invalid_argument("percentile input is empty");
    std::sort(values.begin(), values.end());
    const float position = (percent / 100.0f) * static_cast<float>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) return values[lower];
    const float fraction = position - static_cast<float>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

ExpertRoles rank_expert_roles(const std::vector<std::vector<float>> & features,
                              const ClusterResult & clusters, float coverage_ratio,
                              float coverage_percentile, int shared_experts) {
    if (features.empty() || clusters.neuron_ids.empty() || shared_experts <= 0 ||
        shared_experts >= static_cast<int>(clusters.neuron_ids.size()) ||
        coverage_ratio <= 0.0f || coverage_ratio > 1.0f ||
        coverage_percentile < 0.0f || coverage_percentile > 100.0f) {
        throw std::invalid_argument("invalid expert role configuration");
    }
    const size_t dimension = features.front().size();
    const int n_clusters = static_cast<int>(clusters.neuron_ids.size());
    for (const auto & feature : features) {
        if (feature.size() != dimension) throw std::invalid_argument("feature dimensions differ");
    }

    std::vector<std::vector<float>> means(n_clusters, std::vector<float>(dimension, 0.0f));
    std::vector<float> mean_norms(n_clusters, 0.0f);
    for (int cluster = 0; cluster < n_clusters; ++cluster) {
        if (clusters.neuron_ids[cluster].empty()) throw std::invalid_argument("empty cluster");
        for (int neuron : clusters.neuron_ids[cluster]) {
            if (neuron < 0 || neuron >= static_cast<int>(features.size())) {
                throw std::invalid_argument("cluster neuron index is out of range");
            }
            for (size_t d = 0; d < dimension; ++d) means[cluster][d] += features[neuron][d];
        }
        for (float & value : means[cluster]) value /= static_cast<float>(clusters.neuron_ids[cluster].size());
        for (float value : means[cluster]) mean_norms[cluster] += value * value;
        mean_norms[cluster] = std::sqrt(mean_norms[cluster]);
    }

    ExpertRoles result;
    result.coverage_vectors.assign(n_clusters, std::vector<int>(dimension, 0));
    result.coverage_scores.assign(n_clusters, 0);
    result.mean_norms = mean_norms;
    for (size_t domain = 0; domain < dimension; ++domain) {
        std::vector<float> values(n_clusters);
        for (int cluster = 0; cluster < n_clusters; ++cluster) values[cluster] = means[cluster][domain];
        const float threshold = percentile(values, coverage_percentile);
        for (int cluster = 0; cluster < n_clusters; ++cluster) {
            if (means[cluster][domain] > threshold) {
                result.coverage_vectors[cluster][domain] = 1;
                ++result.coverage_scores[cluster];
            }
        }
    }

    result.ordered_clusters.resize(n_clusters);
    std::iota(result.ordered_clusters.begin(), result.ordered_clusters.end(), 0);
    std::stable_sort(result.ordered_clusters.begin(), result.ordered_clusters.end(), [&](int a, int b) {
        if (result.coverage_scores[a] != result.coverage_scores[b]) {
            return result.coverage_scores[a] > result.coverage_scores[b];
        }
        if (mean_norms[a] != mean_norms[b]) return mean_norms[a] > mean_norms[b];
        return a < b;
    });

    const int ratio_shared = std::max(1, static_cast<int>(n_clusters * coverage_ratio));
    if (ratio_shared != shared_experts) {
        throw std::invalid_argument("shared expert count does not match coverage ratio");
    }
    result.shared_clusters.assign(result.ordered_clusters.begin(), result.ordered_clusters.begin() + shared_experts);
    result.routed_clusters.assign(result.ordered_clusters.begin() + shared_experts, result.ordered_clusters.end());
    return result;
}

} // namespace den2moee
