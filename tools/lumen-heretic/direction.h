#pragma once

#include "dataset.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct heretic_layer_diagnostic {
    double target_mean_norm = 0.0;
    double control_mean_norm = 0.0;
    double difference_norm = 0.0;
    double mean_cosine = 0.0;
    double target_variance_trace = 0.0;
    double control_variance_trace = 0.0;
    double fisher_separation = 0.0;
    int64_t target_count = 0;
    int64_t control_count = 0;
};

struct heretic_router_diagnostic {
    std::vector<double> target_frequency;
    std::vector<double> control_frequency;
    double js_divergence = 0.0;
};

struct heretic_analysis {
    std::string model_fingerprint;
    std::string dataset_fingerprint;
    std::vector<std::vector<float>> directions;
    std::vector<float> direction_norms;
    std::vector<heretic_layer_diagnostic> layer_diagnostics;
    std::vector<heretic_router_diagnostic> router_diagnostics;
    int64_t target_count = 0;
    int64_t control_count = 0;
    double analysis_seconds = 0.0;
};

std::vector<heretic_dataset_entry> heretic_load_dataset(const std::string & path);

bool heretic_extract_directions(
        llama_model * model,
        llama_context_params context_params,
        const std::vector<heretic_dataset_entry> & dataset,
        int32_t residual_token,
        bool orthogonalize_control,
        heretic_analysis & analysis,
        std::string & error);

bool heretic_save_analysis(const std::string & path, const heretic_analysis & analysis, std::string & error);
bool heretic_load_analysis(const std::string & path, heretic_analysis & analysis, std::string & error);
