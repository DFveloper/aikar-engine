#pragma once

#include "dataset.h"
#include "llama.h"
#include "scorer.h"

#include <cstdint>
#include <string>
#include <vector>

struct heretic_reference_position {
    std::vector<float> probabilities;
    llama_token token = -1;
};

struct heretic_cached_entry {
    std::vector<llama_token> prompt;
    std::vector<llama_token> target_tokens;
    std::string group;
    std::string desired;
    float desired_score = 0.0f;
    std::vector<heretic_reference_position> reference;
};

struct heretic_evaluation_config {
    int32_t kl_tokens = 8;
    int32_t behavior_tokens = 32;
    int32_t ctx = 512;
};

struct heretic_metrics {
    double behavior_score = 0.0;
    double behavior_loss = 0.0;
    double general_kl = 0.0;
    double same_top1 = 0.0;
    double mean_probability_deviation = 0.0;
    double total_loss = 0.0;
    double evaluation_seconds = 0.0;
    double tokens_per_second = 0.0;
    int64_t evaluated_tokens = 0;
    int64_t target_count = 0;
    int64_t control_count = 0;
    bool valid = true;
    std::string rejection_reason;
    std::vector<std::string> response_samples;
};

std::vector<heretic_dataset_entry> heretic_format_dataset(
        const llama_model * model,
        const std::vector<heretic_dataset_entry> & dataset,
        bool apply_chat_template);

std::vector<heretic_cached_entry> heretic_prepare_cache(
        const llama_model * model,
        const std::vector<heretic_dataset_entry> & dataset,
        int32_t ctx);

heretic_metrics heretic_evaluate(
        llama_context * context,
        std::vector<heretic_cached_entry> & dataset,
        bool cache_reference,
        const heretic_prefix_scorer & scorer,
        const heretic_evaluation_config & config);
