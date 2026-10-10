#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct aikar_ream_groups {
    std::vector<int32_t> centers;
    std::vector<int32_t> labels;
    std::vector<std::vector<int32_t>> members;
};

struct aikar_ream_expert {
    int32_t embedding = 0;
    int32_t hidden = 0;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> down;
};

void aikar_ream_set_backend(ggml_backend_t backend);

std::vector<double> aikar_ream_saliency_weights(const std::vector<double> & saliency);
aikar_ream_groups aikar_ream_pseudo_group(const std::vector<double> & saliency, const std::vector<float> & distances,
                                        int32_t target, int32_t group_size);
void aikar_ream_validate_groups(const aikar_ream_groups & groups, int32_t experts, int32_t target);
std::vector<int32_t> aikar_ream_hungarian(const std::vector<float> & costs, int32_t neurons);
aikar_ream_expert aikar_ream_permute(const aikar_ream_expert & expert, const std::vector<int32_t> & permutation);
void aikar_ream_accumulate(aikar_ream_expert & accumulator, const aikar_ream_expert & expert, double weight);
std::vector<float> aikar_ream_weight_features(const aikar_ream_expert & expert);
std::vector<float> aikar_ream_normalize_features(const std::vector<float> & values, int32_t rows, int32_t features);
std::vector<float> aikar_ream_distances(const std::vector<float> & a, const std::vector<float> & b,
                                      int32_t rows, int32_t features, int32_t threads, bool normalized = true);
void aikar_ream_forward(const aikar_ream_expert & expert, const std::vector<float> & inputs, int32_t threads,
                       std::vector<float> & hidden, std::vector<float> & outputs);

std::vector<float> aikar_ream_project_mean(const aikar_ream_expert & expert, const std::vector<double> & hidden_mean);

class aikar_ream_forward_runner {
public:
    aikar_ream_forward_runner(const aikar_ream_expert & expert, int32_t chunk, int32_t threads, bool hidden_only = false, ggml_type precision = GGML_TYPE_F32, bool summarize = false);
    ~aikar_ream_forward_runner();
    void run(const std::vector<float> & inputs, std::vector<float> & hidden, std::vector<float> & outputs);
    void run_device(const ggml_tensor * inputs, uint64_t offset, int32_t tokens, std::vector<float> & hidden, std::vector<float> & outputs);
    void run_summary(const ggml_tensor * cached, uint64_t offset, const std::vector<float> & inputs, const std::vector<float> & probabilities, const std::vector<int32_t> & samples, std::vector<float> & hidden, std::vector<float> & weighted_sum);
private:
    void set_input(const std::vector<float> & inputs);
    void set_device_input(const ggml_tensor * inputs, uint64_t offset, int32_t tokens);
    void compute(int32_t tokens, std::vector<float> & hidden, std::vector<float> & outputs);
    struct impl;
    std::unique_ptr<impl> state;
};

struct aikar_ream_layout {
    int32_t experts = 0;
    int32_t top_k = 0;
    int32_t embedding = 0;
    int32_t hidden = 0;
    std::vector<int32_t> layers;
    uint64_t source_bytes = 0;
    uint64_t expert_bytes = 0;
    uint64_t workspace_bytes = 0;
};

aikar_ream_layout aikar_ream_inspect(const std::string & source, int32_t target, int32_t samples, int32_t chunk);
aikar_ream_expert aikar_ream_read_expert(const std::string & source, int32_t layer, int32_t expert);
std::vector<uint8_t> aikar_ream_encode(const std::vector<float> & values, ggml_type type, int64_t row);

using aikar_ream_slice_reader = std::function<std::vector<uint8_t>(const std::string &, int32_t, size_t)>;
void aikar_ream_export(const std::string & source, const std::string & output,
                       const std::map<int32_t, aikar_ream_groups> & groups, const std::string & provenance,
                       const aikar_ream_slice_reader & reader);
