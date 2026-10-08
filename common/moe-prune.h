#pragma once

#include "llama.h"

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

constexpr int32_t COMMON_MOE_PRUNE_STATS_VERSION = 3;

struct common_moe_prune_layer {
    std::vector<int32_t> disabled_experts;
};

struct common_moe_prune_profile {
    int32_t version = 1;
    std::string mode = "soft";
    std::string architecture;
    std::string model_hash;
    std::string expert_tensor_hash;
    int32_t expert_count = 0;
    int32_t experts_used = 0;
    std::string dataset_hash;
    std::string ppl_mask;
    std::string metric;
    int64_t evaluated_tokens = 0;
    int64_t calibration_tokens = 0;
    int32_t calibration_collector_version = 0;
    std::string calibration_tokenized_hash;
    std::string calibration_execution_hash;
    std::string calibration_fingerprint;
    int32_t calibration_seed = -1;
    int32_t calibration_context = 0;
    int32_t calibration_batch = 0;
    int32_t calibration_ubatch = 0;
    double requested_ratio = 0.0;
    double actual_ratio = 0.0;
    std::map<int32_t, common_moe_prune_layer> layers;
};

struct common_moe_prune_model_info {
    std::string architecture;
    std::string model_hash;
    std::string expert_tensor_hash;
    int32_t layer_count = 0;
    int32_t expert_count = 0;
    int32_t experts_used = 0;
    std::vector<int32_t> moe_layers;
    uint64_t expert_bytes = 0;
};

struct common_moe_prune_expert_stats {
    uint64_t selection_count = 0;
    double probability_sum = 0.0;
    double output_norm_sum = 0.0;
    double weighted_output_sum = 0.0;
    uint64_t reap_count = 0;
    double reap_output_norm_sum = 0.0;
    double reap_sum = 0.0;
    uint64_t reap_selection_count = 0;

    void record_selection(double gate, bool collect_reap);
    double mean_probability() const;
    double mean_output_norm() const;
    double importance() const;
    double reap_score() const;
    double mean_reap_output_norm() const;
};

using common_moe_prune_stats = std::map<int32_t, std::vector<common_moe_prune_expert_stats>>;

std::vector<int32_t> common_moe_prune_selected_ids(const ggml_tensor * ids);

void common_moe_prune_collect_output(
        const ggml_tensor * output,
        const std::vector<int32_t> & ids,
        const std::vector<float> & weights,
        std::vector<common_moe_prune_expert_stats> & stats,
        bool collect_legacy);

std::string common_moe_prune_sha256_data(const void * data, size_t size);
std::string common_moe_prune_sha256_file(const std::string & path);
common_moe_prune_model_info common_moe_prune_inspect_model(const std::string & path);
common_moe_prune_model_info common_moe_prune_inspect_model_cached(const std::string & path, const std::string & cache_path, bool * cache_hit = nullptr);
common_moe_prune_profile common_moe_prune_profile_load(const std::string & path);
void common_moe_prune_profile_write(const common_moe_prune_profile & profile, const std::string & path);
void common_moe_prune_profile_validate(const common_moe_prune_profile & profile, const common_moe_prune_model_info & model);
void common_moe_prune_profile_apply(llama_model * model, const common_moe_prune_profile & profile);

std::vector<common_moe_prune_profile> common_moe_prune_make_profiles(
        const common_moe_prune_model_info & model,
        const common_moe_prune_stats & stats,
        const std::vector<double> & ratios,
        double max_layer_ratio,
        const std::string & dataset_hash,
        const std::string & ppl_mask,
        const std::string & metric,
        int64_t evaluated_tokens);
