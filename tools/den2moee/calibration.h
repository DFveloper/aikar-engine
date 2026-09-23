#pragma once

#include "chat.h"
#include "config.h"
#include "json.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

namespace den2moee {

struct CalibrationSample {
    std::string domain;
    std::string text;
    std::vector<int32_t> tokens;
};

struct CalibrationRecord {
    std::string domain;
    std::string text;
    std::vector<common_chat_msg> messages;
};

CalibrationRecord parse_calibration_record(const common_json & item, CalibrationInputFormat format);

struct CalibrationResult {
    int hidden_size = 0;
    std::vector<int> intermediate_sizes;
    std::vector<std::string> domains;
    std::vector<std::vector<std::vector<float>>> layer_features;
    std::vector<std::vector<float>> mean_inputs;
    size_t forward_count = 0;
    size_t rss_forward_count = 0;
    size_t peak_cpu_scratch = 0;
    size_t peak_vram = 0;
};

std::vector<CalibrationSample> load_calibration_jsonl(const std::string & path,
                                                       const llama_model * model,
                                                       int max_seq_len,
                                                       size_t max_samples = 0,
                                                       CalibrationInputFormat format = CalibrationInputFormat::Auto,
                                                       const std::string & chat_template = {});

CalibrationResult run_streaming_calibration(const std::string & model_path,
                                             const std::vector<CalibrationSample> & samples,
                                             const std::vector<int> & intermediate_sizes,
                                             const Options & options);

} // namespace den2moee
