#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum qat_weight_format {
    QAT_WEIGHT_MXFP4,
    QAT_WEIGHT_Q4_0,
};

struct qat_tensor_state {
    enum qat_weight_format format = QAT_WEIGHT_MXFP4;
    int64_t ne = 0;
    std::vector<uint8_t> weight;
    std::vector<uint8_t> momentum;
    std::vector<uint8_t> residual;
};

struct qat_qlion_params {
    float learning_rate = 1.0e-4f;
    float beta = 0.9f;
    float weight_decay = 0.0f;
    float gradient_clip = 0.0f;
    bool enable_residual = true;
};

struct qat_step_stats {
    int64_t ne = 0;
    int64_t nonfinite_gradients = 0;
    int64_t clipped_gradients = 0;
    int64_t clipped_targets = 0;
    int64_t residual_zero_blocks = 0;
    int64_t residual_tiny_blocks = 0;
    int64_t residual_saturated = 0;
    int64_t weight_codes_changed = 0;
    int64_t feedback_code_transitions = 0;

    double residual_rmse = 0.0;
    double residual_cosine_similarity = 1.0;
    double residual_zero_rate = 1.0;
    double residual_saturation_rate = 0.0;
    double residual_scale_min = 0.0;
    double residual_scale_max = 0.0;
    double residual_scale_mean = 0.0;
    double residual_norm = 0.0;
    double residual_norm_over_weight_norm = 0.0;
    double residual_norm_over_update_norm = 0.0;
    double weight_code_change_rate = 0.0;

    size_t weight_bytes = 0;
    size_t momentum_bytes = 0;
    size_t residual_bytes = 0;
    size_t temporary_bytes = 0;
};

enum ggml_type qat_weight_ggml_type(enum qat_weight_format format);

bool qat_tensor_state_init(
        struct qat_tensor_state & state,
        enum qat_weight_format format,
        const float * weights,
        int64_t ne,
        std::string & error);

bool qat_tensor_state_init_quantized(
        struct qat_tensor_state & state,
        enum qat_weight_format format,
        const void * weights,
        size_t weight_bytes,
        int64_t ne,
        std::string & error);

bool qat_tensor_state_validate(const struct qat_tensor_state & state, std::string & error);

bool qat_tensor_state_decode_weight(const struct qat_tensor_state & state, std::vector<float> & values, std::string & error);
bool qat_tensor_state_decode_momentum(const struct qat_tensor_state & state, std::vector<float> & values, std::string & error);
bool qat_tensor_state_decode_residual(const struct qat_tensor_state & state, std::vector<float> & values, std::string & error);

bool qat_tensor_state_step(
        struct qat_tensor_state & state,
        const float * gradient,
        const struct qat_qlion_params & params,
        struct qat_step_stats & stats,
        std::string & error);

