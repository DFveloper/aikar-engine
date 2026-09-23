#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"

namespace den2moee {

enum class TokenScoreMode {
    ScsRss,
    ScsOnly,
    Uniform,
};

enum class ScoreActivation {
    Gemma4Gelu,
    UpstreamSilu,
};

enum class CalibrationInputFormat {
    Auto,
    Raw,
    Chat,
};

struct Options {
    int experts = 8;
    int shared_experts = 1;
    int routed_experts = 7;
    int null_experts = 1;
    int top_k = 2;
    float coverage_threshold_ratio = 0.125f;
    float coverage_percentile = 50.0f;
    float rank_ratio = 0.40f;
    uint64_t seed = 12345;
    int span_size = 64;
    int rss_ngram = 8;
    int rss_stride = 4;
    int perturb_token_id = 0;
    int max_seq_len = 2048;
    size_t max_samples = 0;
    int gpu_layers = -1;
    int n_threads = 1;
    std::string device;
    int layer_start = 0;
    int layer_end = -1;
    size_t vram_limit_mib = 14336;
    TokenScoreMode token_score = TokenScoreMode::ScsRss;
    ScoreActivation score_activation = ScoreActivation::Gemma4Gelu;
    CalibrationInputFormat input_format = CalibrationInputFormat::Auto;
    std::string chat_template;
};

struct LayerConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int expert_intermediate_size = 0;
    int logical_rank = 0;
    int storage_rank = 0;
};

void validate_options(const Options & options);
LayerConfig make_layer_config(int hidden_size, int intermediate_size, const Options & options);

const char * token_score_mode_name(TokenScoreMode mode);
const char * score_activation_name(ScoreActivation activation);
TokenScoreMode parse_token_score_mode(const std::string & value);
ScoreActivation parse_score_activation(const std::string & value);
const char * calibration_input_format_name(CalibrationInputFormat format);
CalibrationInputFormat parse_calibration_input_format(const std::string & value);

std::vector<ggml_backend_dev_t> parse_device_spec(const std::string & value);
std::string device_spec_name(const std::vector<ggml_backend_dev_t> & devices);

} // namespace den2moee
