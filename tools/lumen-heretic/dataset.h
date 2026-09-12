#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct heretic_dataset_entry {
    std::string prompt;
    std::string group;
    std::string desired;
    std::vector<std::string> target_tokens;
    float desired_score = 0.0f;
    int64_t line = 0;
};

struct heretic_dataset_summary {
    int64_t target_count = 0;
    int64_t control_count = 0;
    int64_t duplicate_count = 0;
    int64_t overlap_count = 0;
    int64_t long_prompt_count = 0;
    bool statistically_weak = false;
};

std::vector<heretic_dataset_entry> heretic_load_dataset(const std::string & path);

heretic_dataset_summary heretic_validate_dataset(
        const std::vector<heretic_dataset_entry> & dataset,
        int64_t minimum_target,
        int64_t minimum_control,
        size_t long_prompt_bytes);

std::pair<std::vector<heretic_dataset_entry>, std::vector<heretic_dataset_entry>> heretic_stratified_split(
        const std::vector<heretic_dataset_entry> & dataset,
        double eval_fraction,
        uint32_t seed);

