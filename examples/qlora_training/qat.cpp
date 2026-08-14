#include "qat.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>

static constexpr int64_t QAT_BLOCK_SIZE = 32;
static constexpr float QAT_Q4_0_MIN_ABSMAX = 8.0f * 0x1p-24f;
static constexpr float QAT_Q4_0_MAX_ABS = 8.0f * 65504.0f;
static constexpr float QAT_Q8_0_MAX_ABS = 128.0f * 65504.0f;

enum ggml_type qat_weight_ggml_type(enum qat_weight_format format) {
    switch (format) {
        case QAT_WEIGHT_MXFP4: return GGML_TYPE_MXFP4;
        case QAT_WEIGHT_Q4_0:  return GGML_TYPE_Q4_0;
    }
    return GGML_TYPE_COUNT;
}

static bool qat_type_codec(enum ggml_type type, const struct ggml_type_traits * & traits, std::string & error) {
    traits = ggml_get_type_traits(type);
    if (!traits || !traits->to_float || !traits->from_float_ref) {
        error = std::string("missing GGML codec for ") + ggml_type_name(type);
        return false;
    }
    if (traits->blck_size != QAT_BLOCK_SIZE) {
        error = std::string("unsupported GGML block size for ") + ggml_type_name(type);
        return false;
    }
    return true;
}

static bool qat_shape_valid(int64_t ne, std::string & error) {
    if (ne <= 0 || ne % QAT_BLOCK_SIZE != 0) {
        error = "QAT tensor element count must be a positive multiple of 32";
        return false;
    }
    return true;
}

static bool qat_decode(
        enum ggml_type type,
        const std::vector<uint8_t> & data,
        int64_t ne,
        std::vector<float> & values,
        std::string & error) {
    const struct ggml_type_traits * traits = nullptr;
    if (!qat_type_codec(type, traits, error)) {
        return false;
    }
    const size_t expected = ggml_row_size(type, ne);
    if (data.size() != expected) {
        error = std::string("invalid ") + ggml_type_name(type) + " state byte count";
        return false;
    }
    if (!ggml_validate_row_data(type, data.data(), data.size())) {
        error = std::string("invalid ") + ggml_type_name(type) + " state data";
        return false;
    }
    values.resize(ne);
    traits->to_float(data.data(), values.data(), ne);
    for (int64_t i = 0; i < ne; ++i) {
        if (!std::isfinite(values[i])) {
            error = std::string("nonfinite value decoded from ") + ggml_type_name(type);
            return false;
        }
    }
    return true;
}

static bool qat_encode(
        enum ggml_type type,
        const std::vector<float> & values,
        std::vector<uint8_t> & data,
        std::string & error) {
    const struct ggml_type_traits * traits = nullptr;
    if (!qat_type_codec(type, traits, error)) {
        return false;
    }
    if (!qat_shape_valid(values.size(), error)) {
        return false;
    }
    data.resize(ggml_row_size(type, values.size()));
    traits->from_float_ref(values.data(), data.data(), values.size());
    if (!ggml_validate_row_data(type, data.data(), data.size())) {
        error = std::string("GGML produced invalid ") + ggml_type_name(type) + " state data";
        return false;
    }
    return true;
}

static void qat_zero_state(enum ggml_type type, int64_t ne, std::vector<uint8_t> & data) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> zero(ne, 0.0f);
    data.resize(ggml_row_size(type, ne));
    traits->from_float_ref(zero.data(), data.data(), ne);
}

bool qat_tensor_state_init(
        struct qat_tensor_state & state,
        enum qat_weight_format format,
        const float * weights,
        int64_t ne,
        std::string & error) {
    error.clear();
    if (!weights || !qat_shape_valid(ne, error)) {
        if (!weights && error.empty()) {
            error = "QAT weight input is null";
        }
        return false;
    }
    std::vector<float> finite_weights(ne);
    for (int64_t i = 0; i < ne; ++i) {
        if (!std::isfinite(weights[i])) {
            error = "QAT weight input contains NaN or Inf";
            return false;
        }
        finite_weights[i] = weights[i];
    }
    state = {};
    state.format = format;
    state.ne = ne;
    if (!qat_encode(qat_weight_ggml_type(format), finite_weights, state.weight, error)) {
        return false;
    }
    qat_zero_state(GGML_TYPE_Q8_0, ne, state.momentum);
    qat_zero_state(GGML_TYPE_Q4_0, ne, state.residual);
    return qat_tensor_state_validate(state, error);
}

bool qat_tensor_state_init_quantized(
        struct qat_tensor_state & state,
        enum qat_weight_format format,
        const void * weights,
        size_t weight_bytes,
        int64_t ne,
        std::string & error) {
    error.clear();
    if (!weights || !qat_shape_valid(ne, error)) {
        if (!weights && error.empty()) {
            error = "QAT quantized weight input is null";
        }
        return false;
    }
    const enum ggml_type weight_type = qat_weight_ggml_type(format);
    if (weight_bytes != ggml_row_size(weight_type, ne)) {
        error = "QAT quantized weight byte count does not match tensor shape";
        return false;
    }
    state = {};
    state.format = format;
    state.ne = ne;
    state.weight.resize(weight_bytes);
    memcpy(state.weight.data(), weights, weight_bytes);
    qat_zero_state(GGML_TYPE_Q8_0, ne, state.momentum);
    qat_zero_state(GGML_TYPE_Q4_0, ne, state.residual);
    return qat_tensor_state_validate(state, error);
}

bool qat_tensor_state_validate(const struct qat_tensor_state & state, std::string & error) {
    error.clear();
    if (!qat_shape_valid(state.ne, error)) {
        return false;
    }
    const enum ggml_type weight_type = qat_weight_ggml_type(state.format);
    if (weight_type == GGML_TYPE_COUNT) {
        error = "invalid QAT weight format";
        return false;
    }
    struct item {
        enum ggml_type type;
        const std::vector<uint8_t> * data;
    } items[] = {
        { weight_type, &state.weight },
        { GGML_TYPE_Q8_0, &state.momentum },
        { GGML_TYPE_Q4_0, &state.residual },
    };
    for (const item & value : items) {
        if (value.data->size() != ggml_row_size(value.type, state.ne)) {
            error = std::string("invalid byte count for ") + ggml_type_name(value.type);
            return false;
        }
        if (!ggml_validate_row_data(value.type, value.data->data(), value.data->size())) {
            error = std::string("invalid state data for ") + ggml_type_name(value.type);
            return false;
        }
    }
    return true;
}

bool qat_tensor_state_decode_weight(const struct qat_tensor_state & state, std::vector<float> & values, std::string & error) {
    return qat_decode(qat_weight_ggml_type(state.format), state.weight, state.ne, values, error);
}

bool qat_tensor_state_decode_momentum(const struct qat_tensor_state & state, std::vector<float> & values, std::string & error) {
    return qat_decode(GGML_TYPE_Q8_0, state.momentum, state.ne, values, error);
}

bool qat_tensor_state_decode_residual(const struct qat_tensor_state & state, std::vector<float> & values, std::string & error) {
    return qat_decode(GGML_TYPE_Q4_0, state.residual, state.ne, values, error);
}

static uint8_t qat_weight_code(const std::vector<uint8_t> & data, enum ggml_type type, int64_t index) {
    const size_t block_size = ggml_type_size(type);
    const size_t scale_size = type == GGML_TYPE_MXFP4 ? 1 : sizeof(ggml_fp16_t);
    const size_t block = index / QAT_BLOCK_SIZE;
    const size_t in_block = index % QAT_BLOCK_SIZE;
    const uint8_t packed = data[block * block_size + scale_size + in_block % (QAT_BLOCK_SIZE / 2)];
    return in_block < QAT_BLOCK_SIZE / 2 ? packed & 0x0f : packed >> 4;
}

static void qat_residual_stats(
        const std::vector<float> & residual_true,
        const std::vector<float> & residual_stored,
        const std::vector<uint8_t> & encoded,
        const std::vector<float> & weight,
        const std::vector<float> & update,
        struct qat_step_stats & stats) {
    double error2 = 0.0;
    double dot = 0.0;
    double true2 = 0.0;
    double stored2 = 0.0;
    double weight2 = 0.0;
    double update2 = 0.0;
    int64_t zeros = 0;
    double scale_sum = 0.0;
    double scale_min = std::numeric_limits<double>::infinity();
    double scale_max = 0.0;
    int64_t nonzero_scales = 0;
    const size_t q4_block_size = ggml_type_size(GGML_TYPE_Q4_0);

    for (int64_t ib = 0; ib < stats.ne / QAT_BLOCK_SIZE; ++ib) {
        ggml_fp16_t scale_fp16;
        memcpy(&scale_fp16, encoded.data() + ib * q4_block_size, sizeof(scale_fp16));
        const double scale = std::abs((double) ggml_fp16_to_fp32(scale_fp16));
        if (scale == 0.0) {
            stats.residual_zero_blocks++;
        } else {
            scale_min = std::min(scale_min, scale);
            scale_max = std::max(scale_max, scale);
            scale_sum += scale;
            nonzero_scales++;
        }
    }

    for (int64_t i = 0; i < stats.ne; ++i) {
        const double a = residual_true[i];
        const double b = residual_stored[i];
        const double e = a - b;
        error2 += e * e;
        dot += a * b;
        true2 += a * a;
        stored2 += b * b;
        weight2 += (double) weight[i] * weight[i];
        update2 += (double) update[i] * update[i];
        zeros += b == 0.0f;

        const uint8_t code = qat_weight_code(encoded, GGML_TYPE_Q4_0, i);
        if ((code == 0 || code == 15) && a != 0.0) {
            stats.residual_saturated++;
        }
    }

    stats.residual_rmse = std::sqrt(error2 / stats.ne);
    if (true2 > 0.0 && stored2 > 0.0) {
        stats.residual_cosine_similarity = dot / std::sqrt(true2 * stored2);
    } else {
        stats.residual_cosine_similarity = true2 == 0.0 && stored2 == 0.0 ? 1.0 : 0.0;
    }
    stats.residual_zero_rate = (double) zeros / stats.ne;
    stats.residual_saturation_rate = (double) stats.residual_saturated / stats.ne;
    stats.residual_scale_min = nonzero_scales ? scale_min : 0.0;
    stats.residual_scale_max = scale_max;
    stats.residual_scale_mean = nonzero_scales ? scale_sum / nonzero_scales : 0.0;
    stats.residual_norm = std::sqrt(stored2);
    stats.residual_norm_over_weight_norm = weight2 > 0.0 ? std::sqrt(stored2 / weight2) : 0.0;
    stats.residual_norm_over_update_norm = update2 > 0.0 ? std::sqrt(stored2 / update2) : 0.0;
}

bool qat_tensor_state_step(
        struct qat_tensor_state & state,
        const float * gradient,
        const struct qat_qlion_params & params,
        struct qat_step_stats & stats,
        std::string & error) {
    stats = {};
    error.clear();
    if (!gradient) {
        error = "QAT gradient input is null";
        return false;
    }
    if (!(params.learning_rate > 0.0f) || !std::isfinite(params.learning_rate)) {
        error = "QAT learning rate must be finite and positive";
        return false;
    }
    if (!(params.beta >= 0.0f && params.beta < 1.0f) || !std::isfinite(params.beta)) {
        error = "QAT beta must be finite and in [0, 1)";
        return false;
    }
    if (!(params.weight_decay >= 0.0f) || !std::isfinite(params.weight_decay)) {
        error = "QAT weight decay must be finite and nonnegative";
        return false;
    }
    if (!(params.gradient_clip >= 0.0f) || !std::isfinite(params.gradient_clip)) {
        error = "QAT gradient clip must be finite and nonnegative";
        return false;
    }
    if (!qat_tensor_state_validate(state, error)) {
        return false;
    }

    std::vector<float> weight;
    std::vector<float> momentum;
    std::vector<float> residual;
    if (!qat_tensor_state_decode_weight(state, weight, error) ||
        !qat_tensor_state_decode_momentum(state, momentum, error) ||
        !qat_tensor_state_decode_residual(state, residual, error)) {
        return false;
    }

    stats.ne = state.ne;
    stats.weight_bytes = state.weight.size();
    stats.momentum_bytes = state.momentum.size();
    stats.residual_bytes = state.residual.size();

    std::vector<float> momentum_new(state.ne);
    std::vector<float> update(state.ne);
    std::vector<float> target(state.ne);
    std::vector<float> target_no_feedback(state.ne);
    for (int64_t i = 0; i < state.ne; ++i) {
        float g = gradient[i];
        if (!std::isfinite(g)) {
            g = 0.0f;
            stats.nonfinite_gradients++;
        }
        if (params.gradient_clip > 0.0f) {
            const float clipped = std::max(-params.gradient_clip, std::min(params.gradient_clip, g));
            stats.clipped_gradients += clipped != g;
            g = clipped;
        }
        float m = params.beta * momentum[i] + (1.0f - params.beta) * g;
        if (!std::isfinite(m)) {
            m = 0.0f;
        }
        m = std::max(-QAT_Q8_0_MAX_ABS, std::min(QAT_Q8_0_MAX_ABS, m));
        momentum_new[i] = m;
        const float direction = m > 0.0f ? 1.0f : (m < 0.0f ? -1.0f : 0.0f);
        update[i] = -params.learning_rate * (direction + params.weight_decay * weight[i]);
        target_no_feedback[i] = weight[i] + update[i];
        target[i] = target_no_feedback[i] + (params.enable_residual ? residual[i] : 0.0f);
        if (state.format == QAT_WEIGHT_Q4_0) {
            const float clipped = std::max(-QAT_Q4_0_MAX_ABS, std::min(QAT_Q4_0_MAX_ABS, target[i]));
            stats.clipped_targets += clipped != target[i];
            target[i] = clipped;
            target_no_feedback[i] = std::max(-QAT_Q4_0_MAX_ABS, std::min(QAT_Q4_0_MAX_ABS, target_no_feedback[i]));
        } else if (!std::isfinite(target[i])) {
            target[i] = weight[i];
            stats.clipped_targets++;
        }
    }

    std::vector<uint8_t> weight_new;
    std::vector<uint8_t> weight_no_feedback;
    const enum ggml_type weight_type = qat_weight_ggml_type(state.format);
    if (!qat_encode(weight_type, target, weight_new, error) ||
        !qat_encode(weight_type, target_no_feedback, weight_no_feedback, error)) {
        return false;
    }

    std::vector<float> weight_new_f32;
    if (!qat_decode(weight_type, weight_new, state.ne, weight_new_f32, error)) {
        return false;
    }

    std::vector<float> residual_true(state.ne);
    std::vector<float> residual_safe(state.ne);
    for (int64_t ib = 0; ib < state.ne / QAT_BLOCK_SIZE; ++ib) {
        float absmax = 0.0f;
        for (int64_t j = 0; j < QAT_BLOCK_SIZE; ++j) {
            const int64_t i = ib * QAT_BLOCK_SIZE + j;
            float value = target[i] - weight_new_f32[i];
            if (!std::isfinite(value)) {
                value = 0.0f;
            }
            value = std::max(-QAT_Q4_0_MAX_ABS, std::min(QAT_Q4_0_MAX_ABS, value));
            residual_true[i] = value;
            absmax = std::max(absmax, std::abs(value));
        }
        const bool tiny = absmax > 0.0f && absmax < QAT_Q4_0_MIN_ABSMAX;
        stats.residual_tiny_blocks += tiny;
        for (int64_t j = 0; j < QAT_BLOCK_SIZE; ++j) {
            const int64_t i = ib * QAT_BLOCK_SIZE + j;
            residual_safe[i] = tiny || !params.enable_residual ? 0.0f : residual_true[i];
        }
    }

    std::vector<uint8_t> residual_new;
    std::vector<uint8_t> momentum_encoded;
    if (!qat_encode(GGML_TYPE_Q4_0, residual_safe, residual_new, error) ||
        !qat_encode(GGML_TYPE_Q8_0, momentum_new, momentum_encoded, error)) {
        return false;
    }

    std::vector<float> residual_stored;
    if (!qat_decode(GGML_TYPE_Q4_0, residual_new, state.ne, residual_stored, error)) {
        return false;
    }

    for (int64_t i = 0; i < state.ne; ++i) {
        const uint8_t old_code = qat_weight_code(state.weight, weight_type, i);
        const uint8_t new_code = qat_weight_code(weight_new, weight_type, i);
        const uint8_t no_feedback_code = qat_weight_code(weight_no_feedback, weight_type, i);
        stats.weight_codes_changed += old_code != new_code;
        stats.feedback_code_transitions += params.enable_residual && old_code != new_code && old_code == no_feedback_code;
    }
    stats.weight_code_change_rate = (double) stats.weight_codes_changed / state.ne;
    qat_residual_stats(residual_true, residual_stored, residual_new, weight_new_f32, update, stats);
    stats.temporary_bytes = (weight.size() + momentum.size() + residual.size() + momentum_new.size() + update.size() +
        target.size() + target_no_feedback.size() + weight_new_f32.size() + residual_true.size() + residual_safe.size() +
        residual_stored.size()) * sizeof(float) + weight_new.size() + weight_no_feedback.size() + residual_new.size() + momentum_encoded.size();

    state.weight = std::move(weight_new);
    state.momentum = std::move(momentum_encoded);
    state.residual = std::move(residual_new);
    return qat_tensor_state_validate(state, error);
}

