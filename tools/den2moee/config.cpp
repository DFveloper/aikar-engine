#include "config.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace den2moee {

void validate_options(const Options & options) {
    if (options.experts <= 0 || options.shared_experts < 0 || options.routed_experts < 0 ||
        options.shared_experts + options.routed_experts != options.experts) {
        throw std::invalid_argument("experts must equal shared_experts + routed_experts");
    }
    if (options.null_experts < 0) {
        throw std::invalid_argument("null_experts must be non-negative");
    }
    if (options.routed_experts + options.null_experts <= 0 || options.top_k <= 0 ||
        options.top_k > options.routed_experts + options.null_experts) {
        throw std::invalid_argument("top_k must select at least one valid router candidate");
    }
    if (!(options.coverage_threshold_ratio >= 0.0f && options.coverage_threshold_ratio <= 1.0f) ||
        !(options.coverage_percentile >= 0.0f && options.coverage_percentile <= 100.0f)) {
        throw std::invalid_argument("coverage values must be within their ranges");
    }
    if (!(options.rank_ratio > 0.0f && options.rank_ratio <= 1.0f)) {
        throw std::invalid_argument("rank_ratio must be in (0, 1]");
    }
    if (options.span_size <= 0 || options.rss_ngram <= 0 || options.rss_stride <= 0 ||
        options.max_seq_len <= 0) {
        throw std::invalid_argument("calibration sizes must be positive");
    }
}

LayerConfig make_layer_config(int hidden_size, int intermediate_size, const Options & options) {
    validate_options(options);
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("hidden and intermediate sizes must be positive");
    }
    if (intermediate_size % options.experts != 0) {
        throw std::invalid_argument("intermediate size is not divisible by expert count");
    }

    const int min_dim = std::min(hidden_size, intermediate_size / options.experts);
    const int logical_rank = static_cast<int>(std::floor(min_dim * options.rank_ratio));
    if (logical_rank <= 0) {
        throw std::invalid_argument("rank ratio produces a zero logical rank");
    }

    const int storage_rank = ((logical_rank + 31) / 32) * 32;
    return {
        hidden_size,
        intermediate_size,
        intermediate_size / options.experts,
        logical_rank,
        storage_rank,
    };
}

const char * token_score_mode_name(TokenScoreMode mode) {
    switch (mode) {
        case TokenScoreMode::ScsRss: return "scs-rss";
        case TokenScoreMode::ScsOnly: return "scs-only";
        case TokenScoreMode::Uniform: return "uniform";
    }
    return "unknown";
}

const char * score_activation_name(ScoreActivation activation) {
    switch (activation) {
        case ScoreActivation::Gemma4Gelu: return "gemma4-gelu";
        case ScoreActivation::UpstreamSilu: return "upstream-silu";
    }
    return "unknown";
}

TokenScoreMode parse_token_score_mode(const std::string & value) {
    if (value == "scs-rss") return TokenScoreMode::ScsRss;
    if (value == "scs-only") return TokenScoreMode::ScsOnly;
    if (value == "uniform") return TokenScoreMode::Uniform;
    throw std::invalid_argument("unknown token score mode: " + value);
}

ScoreActivation parse_score_activation(const std::string & value) {
    if (value == "gemma4-gelu") return ScoreActivation::Gemma4Gelu;
    if (value == "upstream-silu") return ScoreActivation::UpstreamSilu;
    throw std::invalid_argument("unknown score activation: " + value);
}

const char * calibration_input_format_name(CalibrationInputFormat format) {
    switch (format) {
        case CalibrationInputFormat::Auto: return "auto";
        case CalibrationInputFormat::Raw:  return "raw";
        case CalibrationInputFormat::Chat: return "chat";
    }
    return "unknown";
}

CalibrationInputFormat parse_calibration_input_format(const std::string & value) {
    if (value == "auto") return CalibrationInputFormat::Auto;
    if (value == "raw" || value == "text") return CalibrationInputFormat::Raw;
    if (value == "chat" || value == "messages" || value == "instruct") return CalibrationInputFormat::Chat;
    throw std::invalid_argument("unknown calibration input format: " + value);
}

std::vector<ggml_backend_dev_t> parse_device_spec(const std::string & value) {
    if (value.empty() || value == "auto") {
        return {};
    }

    if (value == "none") {
        return { nullptr };
    }

    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    std::stringstream stream(value);
    std::string name;
    while (std::getline(stream, name, ',')) {
        const auto first = name.find_first_not_of(" \t");
        const auto last = name.find_last_not_of(" \t");
        if (first == std::string::npos) {
            throw std::invalid_argument("invalid empty device in --device");
        }
        name = name.substr(first, last - first + 1);
        ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
        if (!device || ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            throw std::invalid_argument("invalid device: " + name);
        }
        devices.push_back(device);
    }
    if (devices.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    devices.push_back(nullptr);
    return devices;
}

std::string device_spec_name(const std::vector<ggml_backend_dev_t> & devices) {
    if (devices.empty()) {
        return "auto";
    }
    if (devices.size() == 1 && devices[0] == nullptr) {
        return "none";
    }
    std::string result;
    for (ggml_backend_dev_t device : devices) {
        if (device == nullptr) {
            break;
        }
        if (!result.empty()) {
            result += ',';
        }
        result += ggml_backend_dev_name(device);
    }
    return result;
}

} // namespace den2moee
