#include "qat.h"

#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-opt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static double rmse(const std::vector<float> & a, const std::vector<float> & b) {
    check(a.size() == b.size(), "RMSE shape mismatch");
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum / a.size());
}

static double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    check(a.size() == b.size(), "cosine shape mismatch");
    double dot = 0.0;
    double a2 = 0.0;
    double b2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * b[i];
        a2 += (double) a[i] * a[i];
        b2 += (double) b[i] * b[i];
    }
    return a2 > 0.0 && b2 > 0.0 ? dot / std::sqrt(a2 * b2) : (a2 == 0.0 && b2 == 0.0 ? 1.0 : 0.0);
}

static qat_tensor_state make_state(enum qat_weight_format format, int64_t ne = 32) {
    std::vector<float> weight(ne, 0.5f);
    weight[0] = 1.0f;
    for (int64_t i = 1; i < ne; ++i) {
        weight[i] += 0.125f * std::sin((float) i);
    }
    qat_tensor_state state;
    std::string error;
    check(qat_tensor_state_init(state, format, weight.data(), weight.size(), error), error.c_str());
    return state;
}

static void test_zero_gradient(enum qat_weight_format format) {
    qat_tensor_state state = make_state(format);
    const std::vector<uint8_t> before = state.weight;
    std::vector<float> gradient(state.ne, 0.0f);
    qat_qlion_params params;
    qat_step_stats stats;
    std::string error;
    check(qat_tensor_state_step(state, gradient.data(), params, stats, error), error.c_str());
    check(state.weight == before, "zero gradient changed quantized weight");
    check(stats.weight_codes_changed == 0, "zero gradient changed weight codes");
    check(stats.residual_norm == 0.0, "zero gradient created a residual");
}

static void test_q4_0_residual_round_trip() {
    qat_tensor_state state = make_state(QAT_WEIGHT_MXFP4);
    std::vector<float> gradient(state.ne, 0.0f);
    gradient[1] = 1.0f;
    qat_qlion_params params;
    params.learning_rate = 0.007f;
    params.beta = 0.0f;
    qat_step_stats stats;
    std::string error;
    check(qat_tensor_state_step(state, gradient.data(), params, stats, error), error.c_str());
    std::vector<float> residual;
    check(qat_tensor_state_decode_residual(state, residual, error), error.c_str());
    check(ggml_validate_row_data(GGML_TYPE_Q4_0, state.residual.data(), state.residual.size()), "Q4_0 residual failed validation");
    check(std::isfinite(stats.residual_rmse), "residual RMSE is not finite");
    check(stats.residual_cosine_similarity > 0.99, "Q4_0 residual direction was not preserved");
    check(std::abs(residual[1]) > 0.0f, "Q4_0 residual lost a representable update");
}

static int run_tiny_updates(enum qat_weight_format format, bool feedback, int64_t & feedback_transitions) {
    qat_tensor_state state = make_state(format);
    const std::vector<uint8_t> before = state.weight;
    std::vector<float> gradient(state.ne, 0.0f);
    gradient[1] = 1.0f;
    qat_qlion_params params;
    params.learning_rate = 0.001f;
    params.beta = 0.0f;
    params.enable_residual = feedback;
    std::string error;
    feedback_transitions = 0;
    int first_change = -1;
    for (int step = 1; step <= 256; ++step) {
        qat_step_stats stats;
        check(qat_tensor_state_step(state, gradient.data(), params, stats, error), error.c_str());
        feedback_transitions += stats.feedback_code_transitions;
        if (first_change < 0 && state.weight != before) {
            first_change = step;
        }
        std::vector<float> residual;
        check(qat_tensor_state_decode_residual(state, residual, error), error.c_str());
        for (float value : residual) {
            check(std::isfinite(value), "residual became nonfinite");
            check(std::abs(value) < 1.0f, "residual became unbounded");
        }
    }
    return first_change;
}

static void test_error_feedback(enum qat_weight_format format) {
    int64_t transitions_feedback = 0;
    int64_t transitions_disabled = 0;
    const int change_feedback = run_tiny_updates(format, true, transitions_feedback);
    const int change_disabled = run_tiny_updates(format, false, transitions_disabled);
    check(change_feedback > 1, "tiny update changed the weight without accumulation");
    check(change_feedback <= 256, "error feedback did not produce a code transition");
    check(change_disabled < 0, "weight changed after residual feedback was disabled");
    check(transitions_feedback > 0, "feedback transition metric did not record a transition");
    check(transitions_disabled == 0, "disabled feedback recorded a transition");
}

static void test_q8_0_momentum_cycles() {
    qat_tensor_state state = make_state(QAT_WEIGHT_MXFP4);
    std::vector<float> gradient(state.ne);
    for (int i = 0; i < state.ne; ++i) {
        gradient[i] = 0.03f * std::sin((float) i);
    }
    qat_qlion_params params;
    params.learning_rate = 1.0e-5f;
    params.beta = 0.95f;
    std::vector<float> reference(state.ne, 0.0f);
    std::string error;
    for (int step = 0; step < 100; ++step) {
        qat_step_stats stats;
        check(qat_tensor_state_step(state, gradient.data(), params, stats, error), error.c_str());
        for (int64_t i = 0; i < state.ne; ++i) {
            reference[i] = params.beta * reference[i] + (1.0f - params.beta) * gradient[i];
        }
    }
    std::vector<float> momentum;
    check(qat_tensor_state_decode_momentum(state, momentum, error), error.c_str());
    check(rmse(reference, momentum) < 5.0e-4, "Q8_0 momentum drift exceeded tolerance");
}

static void test_zero_residual_blocks() {
    qat_tensor_state state = make_state(QAT_WEIGHT_Q4_0);
    std::vector<float> residual;
    std::string error;
    check(qat_tensor_state_decode_residual(state, residual, error), error.c_str());
    for (float value : residual) {
        check(value == 0.0f, "all-zero Q4_0 residual block decoded nonzero");
    }
    check(ggml_validate_row_data(GGML_TYPE_Q4_0, state.residual.data(), state.residual.size()), "all-zero Q4_0 residual block failed validation");
}

static void test_output_formats() {
    for (enum qat_weight_format format : { QAT_WEIGHT_MXFP4, QAT_WEIGHT_Q4_0 }) {
        qat_tensor_state state = make_state(format);
        std::vector<float> gradient(state.ne, 0.0f);
        gradient[3] = -1.0f;
        qat_qlion_params params;
        qat_step_stats stats;
        std::string error;
        check(qat_tensor_state_step(state, gradient.data(), params, stats, error), error.c_str());
        const enum ggml_type type = qat_weight_ggml_type(format);
        check(state.weight.size() == ggml_row_size(type, state.ne), "output format byte size changed");
        check(ggml_validate_row_data(type, state.weight.data(), state.weight.size()), "output format failed GGML validation");
    }
}

static void test_nonfinite_gradient_guard() {
    qat_tensor_state state = make_state(QAT_WEIGHT_MXFP4);
    std::vector<float> gradient(state.ne, 0.0f);
    gradient[1] = NAN;
    gradient[2] = INFINITY;
    qat_qlion_params params;
    qat_step_stats stats;
    std::string error;
    check(qat_tensor_state_step(state, gradient.data(), params, stats, error), error.c_str());
    check(stats.nonfinite_gradients == 2, "nonfinite gradient count is incorrect");
    check(qat_tensor_state_validate(state, error), error.c_str());
}

static void test_reference_trajectory(enum qat_weight_format format) {
    qat_tensor_state state = make_state(format, 128);
    std::string error;
    std::vector<float> reference_weight;
    check(qat_tensor_state_decode_weight(state, reference_weight, error), error.c_str());
    std::vector<float> reference_momentum(state.ne, 0.0f);
    std::vector<float> gradient(state.ne);
    qat_qlion_params params;
    params.learning_rate = 0.0007f;
    params.beta = 0.9f;
    params.weight_decay = 0.01f;
    int64_t feedback_transitions = 0;
    qat_step_stats last_stats;
    for (int step = 0; step < 96; ++step) {
        for (int64_t i = 0; i < state.ne; ++i) {
            gradient[i] = 0.03f * std::sin(0.17f * (float) i + 0.09f * step);
            reference_momentum[i] = params.beta * reference_momentum[i] + (1.0f - params.beta) * gradient[i];
            const float direction = reference_momentum[i] > 0.0f ? 1.0f : (reference_momentum[i] < 0.0f ? -1.0f : 0.0f);
            reference_weight[i] -= params.learning_rate * (direction + params.weight_decay * reference_weight[i]);
        }
        check(qat_tensor_state_step(state, gradient.data(), params, last_stats, error), error.c_str());
        feedback_transitions += last_stats.feedback_code_transitions;
    }
    std::vector<float> actual_weight;
    std::vector<float> actual_momentum;
    check(qat_tensor_state_decode_weight(state, actual_weight, error), error.c_str());
    check(qat_tensor_state_decode_momentum(state, actual_momentum, error), error.c_str());
    const double weight_cosine = cosine_similarity(reference_weight, actual_weight);
    const double weight_rmse = rmse(reference_weight, actual_weight);
    const double momentum_cosine = cosine_similarity(reference_momentum, actual_momentum);
    const double momentum_rmse = rmse(reference_momentum, actual_momentum);
    check(std::isfinite(weight_rmse) && weight_cosine > 0.95, "quantized weight trajectory diverged from reference");
    check(std::isfinite(momentum_rmse) && momentum_cosine > 0.99, "Q8_0 momentum trajectory diverged from reference");
    check(std::isfinite(last_stats.residual_rmse) && std::isfinite(last_stats.residual_norm), "residual quality metrics are nonfinite");
    printf("QAT trajectory %s: weight_rmse=%.8g weight_cosine=%.8f momentum_rmse=%.8g momentum_cosine=%.8f residual_rmse=%.8g residual_cosine=%.8f residual_zero_rate=%.8f residual_saturation_rate=%.8f residual_norm=%.8g feedback_transitions=%ld\n",
        ggml_type_name(qat_weight_ggml_type(format)), weight_rmse, weight_cosine, momentum_rmse, momentum_cosine,
        last_stats.residual_rmse, last_stats.residual_cosine_similarity, last_stats.residual_zero_rate,
        last_stats.residual_saturation_rate, last_stats.residual_norm, (long) feedback_transitions);
}

static void test_state_resume_trajectory(enum qat_weight_format format) {
    qat_tensor_state uninterrupted = make_state(format, 128);
    qat_tensor_state resumed;
    qat_qlion_params params;
    params.learning_rate = 0.001f;
    params.beta = 0.91f;
    params.weight_decay = 0.02f;
    std::vector<float> gradient(uninterrupted.ne);
    std::string error;
    for (int step = 0; step < 64; ++step) {
        for (int64_t i = 0; i < uninterrupted.ne; ++i) {
            gradient[i] = 0.04f * std::cos(0.11f * (float) i + 0.07f * step);
        }
        qat_step_stats stats;
        check(qat_tensor_state_step(uninterrupted, gradient.data(), params, stats, error), error.c_str());
        if (step == 31) {
            resumed.format = uninterrupted.format;
            resumed.ne = uninterrupted.ne;
            resumed.weight = uninterrupted.weight;
            resumed.momentum = uninterrupted.momentum;
            resumed.residual = uninterrupted.residual;
            check(qat_tensor_state_validate(resumed, error), error.c_str());
        } else if (step > 31) {
            check(qat_tensor_state_step(resumed, gradient.data(), params, stats, error), error.c_str());
        }
    }
    check(uninterrupted.weight == resumed.weight, "resumed QAT weight differs from uninterrupted trajectory");
    check(uninterrupted.momentum == resumed.momentum, "resumed QAT momentum differs from uninterrupted trajectory");
    check(uninterrupted.residual == resumed.residual, "resumed QAT residual differs from uninterrupted trajectory");
}

static void test_native_backend_step(
        enum qat_weight_format format,
        ggml_backend_dev_t device,
        bool required) {
    qat_tensor_state reference = make_state(format);
    const std::vector<uint8_t> weight_before = reference.weight;
    const std::vector<uint8_t> momentum_before = reference.momentum;
    const std::vector<uint8_t> residual_before = reference.residual;
    std::vector<float> gradient(reference.ne);
    for (int64_t i = 0; i < reference.ne; ++i) {
        gradient[i] = 0.05f * std::sin((float) i);
    }
    qat_qlion_params params;
    params.learning_rate = 0.003f;
    params.beta = 0.85f;
    params.weight_decay = 0.01f;
    params.gradient_clip = 0.04f;
    qat_step_stats stats;
    std::string error;
    check(qat_tensor_state_step(reference, gradient.data(), params, stats, error), error.c_str());

    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend && !required) {
        printf("QAT backend test skipped: backend unavailable\n");
        return;
    }
    check(backend != nullptr, "failed to create QAT test backend");
    ggml_init_params init_params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    check(ctx != nullptr, "failed to create native QAT test context");
    ggml_tensor * weight = ggml_new_tensor_1d(ctx, qat_weight_ggml_type(format), reference.ne);
    ggml_set_param(weight);
    ggml_tensor * grad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, reference.ne);
    ggml_tensor * momentum = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, reference.ne);
    ggml_tensor * residual = ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_0, reference.ne);
    ggml_tensor * opt_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * result = ggml_opt_step_qlion_qat(ctx, weight, grad, momentum, residual, opt_params);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "failed to allocate native QAT test tensors");

    const float native_params[] = { params.learning_rate, params.beta, params.weight_decay, params.gradient_clip };
    ggml_backend_tensor_set(weight, weight_before.data(), 0, weight_before.size());
    ggml_backend_tensor_set(grad, gradient.data(), 0, gradient.size() * sizeof(float));
    ggml_backend_tensor_set(momentum, momentum_before.data(), 0, momentum_before.size());
    ggml_backend_tensor_set(residual, residual_before.data(), 0, residual_before.size());
    ggml_backend_tensor_set(opt_params, native_params, 0, sizeof(native_params));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "native backend QAT step failed");

    std::vector<uint8_t> weight_after(reference.weight.size());
    std::vector<uint8_t> momentum_after(reference.momentum.size());
    std::vector<uint8_t> residual_after(reference.residual.size());
    ggml_backend_tensor_get(weight, weight_after.data(), 0, weight_after.size());
    ggml_backend_tensor_get(momentum, momentum_after.data(), 0, momentum_after.size());
    ggml_backend_tensor_get(residual, residual_after.data(), 0, residual_after.size());
    check(weight_after == reference.weight, "native backend weight differs from reference");
    check(momentum_after == reference.momentum, "native backend momentum differs from reference");
    check(residual_after == reference.residual, "native backend residual differs from reference");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static struct ggml_opt_optimizer_params test_qlion_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);
    struct ggml_opt_optimizer_params params = ggml_opt_get_default_optimizer_params(nullptr);
    params.qlion_qat.alpha = 0.01f;
    params.qlion_qat.beta = 0.0f;
    return params;
}

static void test_native_optimizer_graph(enum qat_weight_format format) {
    qat_tensor_state initial = make_state(format);
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(backend != nullptr, "failed to create optimizer CPU backend");
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    check(sched != nullptr, "failed to create optimizer scheduler");

    ggml_init_params static_params = {
        /*.mem_size   =*/ 3 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(static_params);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx_static, qat_weight_ggml_type(format), 32, 1);
    ggml_set_name(weight, "tied.weight");
    ggml_set_param(weight);
    ggml_tensor * weight_alias = ggml_new_tensor_2d(ctx_static, qat_weight_ggml_type(format), 32, 1);
    ggml_set_name(weight_alias, "tied.weight");
    ggml_set_param(weight_alias);
    ggml_tensor * input = ggml_new_tensor_2d(ctx_static, GGML_TYPE_F32, 32, 1);
    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    check(static_buffer != nullptr, "failed to allocate optimizer static tensors");
    std::vector<float> input_values(32, 1.0f);
    ggml_backend_tensor_set(weight, initial.weight.data(), 0, initial.weight.size());
    ggml_backend_tensor_set(weight_alias, initial.weight.data(), 0, initial.weight.size());
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));

    ggml_init_params compute_params = {
        /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 3 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(compute_params);
    ggml_tensor * output = ggml_add(ctx_compute,
        ggml_mul_mat(ctx_compute, weight, input), ggml_mul_mat(ctx_compute, weight_alias, input));
    struct ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    opt_params.ctx_compute = ctx_compute;
    opt_params.inputs = input;
    opt_params.outputs = output;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_QLION_QAT;
    opt_params.get_opt_pars = test_qlion_optimizer_params;
    ggml_opt_context_t opt = ggml_opt_init(opt_params);
    check(ggml_opt_qat_state_count(opt) == 1, "optimizer did not create exactly one QAT state");
    check(ggml_opt_qat_state_param(opt, 0) == weight, "optimizer QAT state is attached to the wrong parameter");
    check(ggml_opt_qat_state_momentum(opt, 0)->type == GGML_TYPE_Q8_0, "optimizer momentum is not Q8_0");
    check(ggml_opt_qat_state_residual(opt, 0)->type == GGML_TYPE_Q4_0, "optimizer residual is not Q4_0");
    check(ggml_opt_step(opt) == 0, "optimizer initial step is incorrect");
    ggml_opt_alloc(opt, true);
    ggml_set_zero(ggml_opt_labels(opt));
    check(ggml_opt_grad_acc(opt, weight) == nullptr, "QLion QAT allocated a persistent full-size gradient accumulator");

    ggml_opt_result_t result = ggml_opt_result_init();
    ggml_opt_eval(opt, result);
    check(ggml_opt_step(opt) == 1, "optimizer step did not advance");
    std::vector<uint8_t> weight_after(initial.weight.size());
    std::vector<uint8_t> alias_after(initial.weight.size());
    ggml_backend_tensor_get(weight, weight_after.data(), 0, weight_after.size());
    ggml_backend_tensor_get(weight_alias, alias_after.data(), 0, alias_after.size());
    check(weight_after != initial.weight, "optimizer graph did not update native quantized weight");
    check(alias_after == weight_after, "tied parameter did not receive the native quantized update");
    check(ggml_validate_row_data(qat_weight_ggml_type(format), weight_after.data(), weight_after.size()),
        "optimizer graph produced an invalid native quantized weight");

    ggml_opt_result_free(result);
    ggml_opt_free(opt);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
}

static std::vector<uint8_t> quantize_rows(enum ggml_type type, const std::vector<float> & values, int64_t row_size) {
    check(values.size() % row_size == 0, "quantize rows shape mismatch");
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    check(traits && traits->from_float_ref, "missing row quantizer");
    const int64_t rows = values.size() / row_size;
    const size_t row_bytes = ggml_row_size(type, row_size);
    std::vector<uint8_t> result(rows * row_bytes);
    for (int64_t row = 0; row < rows; ++row) {
        traits->from_float_ref(values.data() + row * row_size, result.data() + row * row_bytes, row_size);
    }
    return result;
}

static void test_native_moe_dx(enum qat_weight_format format, ggml_backend_dev_t device, bool required) {
    constexpr int64_t cols = 32;
    constexpr int64_t rows = 7;
    constexpr int64_t experts = 4;
    constexpr int64_t used = 2;
    constexpr int64_t tokens = 3;
    const enum ggml_type weight_type = qat_weight_ggml_type(format);
    const ggml_type_traits * traits = ggml_get_type_traits(weight_type);
    std::vector<float> weight_values(cols * rows * experts);
    for (size_t i = 0; i < weight_values.size(); ++i) {
        weight_values[i] = 0.2f * std::sin(0.17f * (float) i);
    }
    const std::vector<uint8_t> weight_data = quantize_rows(weight_type, weight_values, cols);
    std::vector<float> decoded(weight_values.size());
    const size_t row_bytes = ggml_row_size(weight_type, cols);
    for (int64_t row = 0; row < rows * experts; ++row) {
        traits->to_float(weight_data.data() + row * row_bytes, decoded.data() + row * cols, cols);
    }
    std::vector<float> grad(rows * used * tokens);
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = 0.1f * std::cos(0.11f * (float) i);
    }
    const int32_t route_ids[used * tokens] = { 0, 2, 3, 2, 1, 3 };
    std::vector<float> expected(cols * used * tokens, 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t route = 0; route < used; ++route) {
            const int32_t expert = route_ids[route + used * token];
            for (int64_t col = 0; col < cols; ++col) {
                for (int64_t row = 0; row < rows; ++row) {
                    expected[col + cols * (route + used * token)] +=
                        decoded[col + cols * (row + rows * expert)] * grad[row + rows * (route + used * token)];
                }
            }
        }
    }

    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend && !required) {
        return;
    }
    check(backend != nullptr, "failed to create MoE dX test backend");
    ggml_init_params params = {
        /*.mem_size   =*/ 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx, weight_type, cols, rows, experts);
    ggml_tensor * grad_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, used, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    ggml_tensor * dx = ggml_mul_mat_id_back(ctx, weight, grad_tensor, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, dx);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "failed to allocate MoE dX tensors");
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(grad_tensor, grad.data(), 0, grad.size() * sizeof(float));
    ggml_backend_tensor_set(ids, route_ids, 0, sizeof(route_ids));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "native quantized MoE dX failed");
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(dx, actual.data(), 0, actual.size() * sizeof(float));
    check(rmse(actual, expected) < 1.0e-6, "native quantized MoE dX differs from reference");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

struct moe_op_trace {
    bool routed_update = false;
    bool full_gradient = false;
};

static bool trace_moe_ops(struct ggml_tensor * tensor, bool ask, void * userdata) {
    GGML_UNUSED(ask);
    moe_op_trace & trace = *(moe_op_trace *) userdata;
    trace.routed_update = trace.routed_update || tensor->op == GGML_OP_OPT_STEP_QLION_QAT_ID;
    trace.full_gradient = trace.full_gradient || tensor->op == GGML_OP_OUT_PROD_ID;
    return false;
}

static void test_native_routed_update(enum qat_weight_format format, ggml_backend_dev_t device, bool required) {
    constexpr int64_t cols = 32;
    constexpr int64_t rows = 5;
    constexpr int64_t experts = 3;
    constexpr int64_t used = 2;
    constexpr int64_t tokens = 2;
    const enum ggml_type weight_type = qat_weight_ggml_type(format);
    std::vector<float> initial_values(cols * rows * experts);
    for (size_t i = 0; i < initial_values.size(); ++i) {
        initial_values[i] = 0.4f + 0.1f * std::sin(0.07f * (float) i);
    }
    const std::vector<uint8_t> initial_weight = quantize_rows(weight_type, initial_values, cols);
    std::vector<float> activations(cols * used * tokens);
    std::vector<float> upstream(rows * used * tokens);
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = 0.2f * std::cos(0.13f * (float) i);
    }
    for (size_t i = 0; i < upstream.size(); ++i) {
        upstream[i] = 0.15f * std::sin(0.19f * (float) i);
    }
    const int32_t route_ids[used * tokens] = { 0, 2, 2, 0 };
    std::vector<float> full_gradient(initial_values.size(), 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t route = 0; route < used; ++route) {
            const int64_t expert = route_ids[route + used * token];
            for (int64_t row = 0; row < rows; ++row) {
                for (int64_t col = 0; col < cols; ++col) {
                    full_gradient[col + cols * (row + rows * expert)] +=
                        activations[col + cols * (route + used * token)] * upstream[row + rows * (route + used * token)];
                }
            }
        }
    }
    qat_tensor_state reference;
    std::string error;
    check(qat_tensor_state_init_quantized(reference, format, initial_weight.data(), initial_weight.size(), initial_values.size(), error), error.c_str());
    qat_qlion_params params;
    params.learning_rate = 0.004f;
    params.beta = 0.8f;
    params.weight_decay = 0.01f;
    params.gradient_clip = 0.03f;
    qat_step_stats stats;
    check(qat_tensor_state_step(reference, full_gradient.data(), params, stats, error), error.c_str());

    ggml_backend_t backend = device ? ggml_backend_dev_init(device, nullptr) : nullptr;
    if (!backend && !required) {
        return;
    }
    check(backend != nullptr, "failed to create routed QAT backend");
    ggml_init_params init_params = {
        /*.mem_size   =*/ 12 * ggml_tensor_overhead() + ggml_graph_overhead_custom(12, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx, weight_type, cols, rows, experts);
    ggml_set_param(weight);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, cols, used, tokens);
    ggml_tensor * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rows, used, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    ggml_tensor * momentum = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, cols, rows, experts);
    ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, cols, rows, experts);
    ggml_tensor * opt_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * result = ggml_opt_step_qlion_qat_id(ctx, weight, a, g, ids, momentum, residual, opt_params);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 12, false);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "failed to allocate routed QAT tensors");
    const float native_params[] = { params.learning_rate, params.beta, params.weight_decay, params.gradient_clip };
    ggml_backend_tensor_set(weight, initial_weight.data(), 0, initial_weight.size());
    ggml_backend_tensor_set(a, activations.data(), 0, activations.size() * sizeof(float));
    ggml_backend_tensor_set(g, upstream.data(), 0, upstream.size() * sizeof(float));
    ggml_backend_tensor_set(ids, route_ids, 0, sizeof(route_ids));
    std::vector<uint8_t> momentum_zero(ggml_nbytes(momentum), 0);
    std::vector<uint8_t> residual_zero(ggml_nbytes(residual), 0);
    ggml_backend_tensor_set(momentum, momentum_zero.data(), 0, momentum_zero.size());
    ggml_backend_tensor_set(residual, residual_zero.data(), 0, residual_zero.size());
    ggml_backend_tensor_set(opt_params, native_params, 0, sizeof(native_params));
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "native routed QAT update failed");
    std::vector<uint8_t> weight_after(reference.weight.size());
    std::vector<uint8_t> momentum_after(reference.momentum.size());
    std::vector<uint8_t> residual_after(reference.residual.size());
    ggml_backend_tensor_get(weight, weight_after.data(), 0, weight_after.size());
    ggml_backend_tensor_get(momentum, momentum_after.data(), 0, momentum_after.size());
    ggml_backend_tensor_get(residual, residual_after.data(), 0, residual_after.size());
    check(weight_after == reference.weight, "native routed QAT weight differs from reference");
    check(momentum_after == reference.momentum, "native routed QAT momentum differs from reference");
    check(residual_after == reference.residual, "native routed QAT residual differs from reference");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_moe_sparse_qat(enum qat_weight_format format) {
    constexpr int64_t cols = 32;
    constexpr int64_t rows = 32;
    constexpr int64_t experts = 4;
    constexpr int64_t used = 1;
    constexpr int64_t tokens = 2;
    const enum ggml_type weight_type = qat_weight_ggml_type(format);
    std::vector<float> weight_values(cols * rows * experts);
    for (int64_t expert = 0; expert < experts; ++expert) {
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t col = 0; col < cols; ++col) {
                weight_values[col + cols * (row + rows * expert)] =
                    0.125f * (expert + 1) + 0.01f * std::sin((float) (col + 3 * row));
            }
        }
    }
    const std::vector<uint8_t> weight_data = quantize_rows(weight_type, weight_values, cols);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[] = { backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_init_params static_params = {
        /*.mem_size   =*/ 6 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_static = ggml_init(static_params);
    ggml_tensor * weight = ggml_new_tensor_3d(ctx_static, weight_type, cols, rows, experts);
    ggml_set_name(weight, "moe.weight");
    ggml_set_param(weight);
    ggml_tensor * input = ggml_new_tensor_3d(ctx_static, GGML_TYPE_F32, cols, used, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx_static, GGML_TYPE_I32, used, tokens);
    ggml_backend_buffer_t static_buffer = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    check(static_buffer != nullptr, "failed to allocate MoE test tensors");
    std::vector<float> input_values(cols * used * tokens);
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = 0.25f + 0.02f * std::cos((float) i);
    }
    const int32_t route_ids[tokens] = { 1, 3 };
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size());
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, route_ids, 0, sizeof(route_ids));

    ggml_init_params compute_params = {
        /*.mem_size   =*/ GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 3 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_compute = ggml_init(compute_params);
    ggml_tensor * output = ggml_mul_mat_id(ctx_compute, weight, input, ids);
    ggml_opt_params opt_params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR);
    opt_params.ctx_compute = ctx_compute;
    opt_params.inputs = input;
    opt_params.outputs = output;
    opt_params.optimizer = GGML_OPT_OPTIMIZER_TYPE_QLION_QAT;
    opt_params.get_opt_pars = test_qlion_optimizer_params;
    ggml_opt_context_t opt = ggml_opt_init(opt_params);
    ggml_opt_alloc(opt, true);
    ggml_set_zero(ggml_opt_labels(opt));
    check(ggml_opt_grad_acc(opt, weight) == nullptr, "MoE QAT allocated a persistent full gradient accumulator");
    moe_op_trace trace;
    ggml_backend_sched_set_eval_callback(sched, trace_moe_ops, &trace);
    ggml_opt_result_t result = ggml_opt_result_init();
    ggml_opt_eval(opt, result);
    check(trace.routed_update, "MoE QAT did not use the routed optimizer update");
    check(!trace.full_gradient, "MoE QAT materialized a full expert gradient tensor");

    std::vector<uint8_t> weight_after(weight_data.size());
    ggml_backend_tensor_get(weight, weight_after.data(), 0, weight_after.size());
    const size_t expert_bytes = ggml_row_size(weight_type, cols) * rows;
    check(memcmp(weight_after.data(), weight_data.data(), expert_bytes) == 0, "inactive MoE expert 0 changed");
    check(memcmp(weight_after.data() + 2 * expert_bytes, weight_data.data() + 2 * expert_bytes, expert_bytes) == 0,
        "inactive MoE expert 2 changed");
    check(memcmp(weight_after.data() + expert_bytes, weight_data.data() + expert_bytes, expert_bytes) != 0,
        "active MoE expert 1 did not change");
    check(memcmp(weight_after.data() + 3 * expert_bytes, weight_data.data() + 3 * expert_bytes, expert_bytes) != 0,
        "active MoE expert 3 did not change");

    ggml_opt_result_free(result);
    ggml_opt_free(opt);
    ggml_backend_buffer_free(static_buffer);
    ggml_free(ctx_static);
    ggml_free(ctx_compute);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
}

int main() {
    ggml_backend_load_all();
    test_zero_gradient(QAT_WEIGHT_MXFP4);
    test_zero_gradient(QAT_WEIGHT_Q4_0);
    test_q4_0_residual_round_trip();
    test_error_feedback(QAT_WEIGHT_MXFP4);
    test_error_feedback(QAT_WEIGHT_Q4_0);
    test_q8_0_momentum_cycles();
    test_zero_residual_blocks();
    test_output_formats();
    test_nonfinite_gradient_guard();
    test_reference_trajectory(QAT_WEIGHT_MXFP4);
    test_reference_trajectory(QAT_WEIGHT_Q4_0);
    test_state_resume_trajectory(QAT_WEIGHT_MXFP4);
    test_state_resume_trajectory(QAT_WEIGHT_Q4_0);
    ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    test_native_backend_step(QAT_WEIGHT_MXFP4, cpu_device, true);
    test_native_backend_step(QAT_WEIGHT_Q4_0, cpu_device, true);
    test_native_moe_dx(QAT_WEIGHT_MXFP4, cpu_device, true);
    test_native_moe_dx(QAT_WEIGHT_Q4_0, cpu_device, true);
    test_native_routed_update(QAT_WEIGHT_MXFP4, cpu_device, true);
    test_native_routed_update(QAT_WEIGHT_Q4_0, cpu_device, true);
    const char * backend_filter = getenv("QAT_TEST_BACKEND");
    for (size_t i = 0; backend_filter && i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (!strstr(ggml_backend_dev_name(device), backend_filter)) {
            continue;
        }
        printf("Testing QAT backend: %s\n", ggml_backend_dev_name(device));
        test_native_backend_step(QAT_WEIGHT_MXFP4, device, false);
        test_native_backend_step(QAT_WEIGHT_Q4_0, device, false);
        test_native_moe_dx(QAT_WEIGHT_MXFP4, device, false);
        test_native_moe_dx(QAT_WEIGHT_Q4_0, device, false);
        test_native_routed_update(QAT_WEIGHT_MXFP4, device, false);
        test_native_routed_update(QAT_WEIGHT_Q4_0, device, false);
    }
    test_native_optimizer_graph(QAT_WEIGHT_MXFP4);
    test_native_optimizer_graph(QAT_WEIGHT_Q4_0);
    test_moe_sparse_qat(QAT_WEIGHT_MXFP4);
    test_moe_sparse_qat(QAT_WEIGHT_Q4_0);
    printf("QAT reference tests passed\n");
    return 0;
}
