#include "mixed-cache.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "llama-hparams.h"
#include "llama-paired-forward.h"
#include "llama-graph.h"
#include "llama-model.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <random>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void no_cache_mask_checks() {
    constexpr int rows = 17, columns = 32;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (bool causal : {false, true}) for (bool alibi : {false, true}) {
            for (bool irregular : {false, true}) for (bool multiple : {false, true}) {
                for (auto swa : {LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD, LLAMA_SWA_TYPE_CHUNKED, LLAMA_SWA_TYPE_SYMMETRIC}) {
                    llama_hparams hparams = {};
                    hparams.use_alibi = alibi;
                    hparams.swa_type = swa;
                    hparams.n_swa = 3;
                    llama_cparams cparams = {};
                    cparams.causal_attn = causal;
                    llm_graph_input_attn_no_cache input(hparams, cparams);
                    ggml_context_ptr ctx(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
                    input.self_attn_mask.raw = ggml_new_tensor_2d(ctx.get(), type, columns, rows);
                    input.self_attn_mask_swa.raw = ggml_new_tensor_2d(ctx.get(), type, columns, rows);
                    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
                    std::vector<llama_pos> positions(rows);
                    std::vector<llama_seq_id> sequences(rows);
                    std::vector<llama_seq_id *> ids(rows);
                    for (int i = 0; i < rows; ++i) {
                        positions[i] = 7 + i*(irregular ? 2 : 1);
                        sequences[i] = multiple ? i%2 : 0;
                        ids[i] = &sequences[i];
                    }
                    llama_ubatch batch = {};
                    batch.n_tokens = rows;
                    batch.pos = positions.data();
                    batch.seq_id = ids.data();
                    input.set_input(&batch);
                    for (bool sliding : {false, true}) {
                        auto * tensor = sliding && swa != LLAMA_SWA_TYPE_NONE ? input.self_attn_mask_swa.raw : input.self_attn_mask.raw;
                        std::vector<float> values(rows*columns);
                        if (type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> packed(values.size());
                            ggml_backend_tensor_get(tensor, packed.data(), 0, ggml_nbytes(tensor));
                            ggml_fp16_to_fp32_row(packed.data(), values.data(), values.size());
                        } else {
                            ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
                        }
                        for (int q = 0; q < rows; ++q) for (int k = 0; k < columns; ++k) {
                            float expected = -INFINITY;
                            if (k < rows && sequences[k] == sequences[q] && (!causal || positions[k] <= positions[q]) &&
                                (!sliding || !llama_hparams::is_masked_swa(hparams.n_swa, swa, positions[k], positions[q]))) {
                                expected = alibi ? -std::abs(positions[k] - positions[q]) : 0.0f;
                            }
                            check(values[q*columns + k] == expected, "no-cache attention mask differs from position reference");
                        }
                    }
                }
            }
        }
    }
    std::puts("attention masks: contiguous, irregular, sequence, ALiBi and SWA checks passed");
}

struct paired_result { std::vector<float> train, validation, gradient; };

static std::vector<float> fake_quant_gradient(ggml_backend_t backend, bool reference) {
    llm_graph_result result(1024);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    llm_graph_params params = {};
    params.model = model.get();
    model->hparams.n_layer_all = 1;
    model->hparams.n_embd = 64;
    model->hparams.n_head_arr[0] = 2;
    model->hparams.n_head_kv_arr[0] = 1;
    model->hparams.n_embd_head_k_full = 32;
    model->hparams.n_embd_head_v_full = 32;
    params.res = &result;
    llm_graph_context builder(params);
    auto * ctx = result.get_ctx();
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 8);
    auto * upstream = ggml_dup_tensor(ctx, input);
    ggml_set_param(input);
    ggml_tensor * output;
    if (reference) {
        auto * packed = ggml_cast(ctx, input, GGML_TYPE_FP4_KV);
        auto * restored = ggml_cast(ctx, packed, GGML_TYPE_F32);
        auto * stopped = ggml_cpy_no_grad(ctx, restored, ggml_dup_tensor(ctx, restored));
        auto * master = ggml_cpy_no_grad(ctx, input, ggml_dup_tensor(ctx, input));
        output = ggml_add(ctx, input, ggml_sub(ctx, stopped, master));
    } else {
        output = builder.build_kv_fake_quant(input, GGML_TYPE_FP4_KV);
    }
    ggml_set_output(output);
    auto * loss = ggml_sum(ctx, ggml_mul(ctx, output, upstream));
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_t backends[] = {backend, cpu.get()};
    const bool on_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    auto * sched = ggml_backend_sched_new(backends, nullptr, on_cpu ? 1 : 2, 1024, false, true);
    check(ggml_backend_sched_alloc_graph(sched, graph), "FP4 STE allocation failed");
    std::vector<float> values(2048), expected(2048);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(float(i)*.17f)*7.0f;
        expected[i] = float(int(i%19) - 9)/19;
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
    ggml_backend_tensor_set(upstream, expected.data(), 0, expected.size()*sizeof(float));
    ggml_graph_reset(graph);
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "FP4 STE compute failed");
    std::vector<float> actual(values.size()*2);
    ggml_backend_tensor_get(output, actual.data(), 0, values.size()*sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_grad(graph, input), actual.data() + values.size(), 0, values.size()*sizeof(float));
    check(std::memcmp(actual.data() + values.size(), expected.data(), expected.size()*sizeof(float)) == 0, "FP4 STE lost its identity gradient");
    ggml_backend_sched_free(sched);
    return actual;
}

static void fake_quant_checks(ggml_backend_t backend) {
    const auto reference = fake_quant_gradient(backend, true);
    const auto candidate = fake_quant_gradient(backend, false);
    check(reference.size() == candidate.size() && std::memcmp(reference.data(), candidate.data(), reference.size()*sizeof(float)) == 0, "FP4 stop-gradient copy removal changed values or gradients");
    std::printf("FP4 STE: copy removal preserves exact values and identity gradients on %s\n", ggml_backend_name(backend));
}

static paired_result paired_forward(ggml_backend_t backend, bool paired, bool routed, bool broadcast, bool zero_copy = false, bool crop = false) {
    ggml_context_ptr ctx(ggml_init({ 32*1024*1024, nullptr, true }));
    ggml_context_ptr inputs(ggml_init({ 8*ggml_tensor_overhead(), nullptr, true }));
    const int dim = 64, rows = 64, experts = routed ? 4 : 1, used = routed ? 2 : 1;
    auto * weight = ggml_new_tensor_3d(inputs.get(), GGML_TYPE_Q4_0, dim, rows, experts);
    ggml_set_name(weight, "left");
    auto * train = routed ? ggml_new_tensor_3d(inputs.get(), GGML_TYPE_F32, dim, broadcast ? 1 : used, 8) :
                            ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, dim, 8);
    auto * validation = routed ? ggml_new_tensor_3d(inputs.get(), GGML_TYPE_F32, dim, broadcast ? 1 : used, 5) :
                                 ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, dim, 5);
    ggml_set_name(train, "right");
    ggml_set_name(validation, "right");
    ggml_set_param(train);
    auto * ids_t = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, used, 8);
    auto * ids_v = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, used, 5);
    auto * crop_ids = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_I32, 5);
    auto * state = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_F32, 64);
    auto * patch = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_F32, 64);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), backend));
    check(bool(buffer), "paired inputs allocation failed");
    auto * train_input = crop ? ggml_get_rows(ctx.get(), train, crop_ids) : train;
    auto * out_t = routed ? ggml_mul_mat_id(ctx.get(), weight, train_input, ids_t) : ggml_mul_mat(ctx.get(), weight, train_input);
    auto * out_v = routed ? ggml_mul_mat_id(ctx.get(), weight, validation, ids_v) : ggml_mul_mat(ctx.get(), weight, validation);
    ggml_set_output(out_t);
    ggml_set_output(out_v);
    auto * read_t = ggml_tanh(ctx.get(), out_t);
    auto * read_v = ggml_tanh(ctx.get(), out_v);
    ggml_set_output(read_t);
    ggml_set_output(read_v);
    auto * loss = ggml_add(ctx.get(), ggml_sum(ctx.get(), ggml_sqr(ctx.get(), out_t)),
                                   ggml_sum(ctx.get(), ggml_sqr(ctx.get(), out_v)));
    ggml_set_loss(loss);
    auto * train_graph = ggml_new_graph_custom(ctx.get(), 1024, true);
    auto * val_graph = ggml_new_graph_custom(ctx.get(), 1024, false);
    auto * inactive_t = ggml_cpy(ctx.get(), patch, state);
    auto * inactive_v = ggml_cpy(ctx.get(), patch, state);
    ggml_build_forward_order(train_graph, inactive_t);
    ggml_build_forward_order(val_graph, inactive_v);
    ggml_build_forward_expand(train_graph, loss);
    ggml_build_forward_expand(val_graph, out_v);
    ggml_cgraph * paired_graph = nullptr;
    if (paired) {
        weight->flags |= GGML_TENSOR_FLAG_PARAM;
        check(llama_pair_forward_graphs(ctx.get(), train_graph, val_graph).pairs == 0, "mutable weights must not be paired");
        weight->flags &= ~GGML_TENSOR_FLAG_PARAM;
        const auto combined = llama_pair_forward_graphs(ctx.get(), train_graph, val_graph, zero_copy ? weight : nullptr);
        check(combined.pairs == 1, "shared projection was not paired");
        if (zero_copy) check(out_v->view_src == out_t->view_src, "detached projection allocated another buffer");
        paired_graph = combined.graph;
        check(!((inactive_t->flags | inactive_v->flags) & GGML_TENSOR_FLAG_COMPUTE), "pairing activated an inactive embedding branch");
    }
    auto * graph = paired_graph ? paired_graph : ggml_new_graph_custom(ctx.get(), 4096, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_forward_expand(graph, out_v);
    ggml_build_forward_expand(graph, read_t);
    ggml_build_forward_expand(graph, read_v);
    ggml_set_fused_backward(ctx.get(), true);
    ggml_build_backward_expand(ctx.get(), graph, nullptr);
    if (paired && zero_copy && !routed) check(ggml_graph_get_grad(graph, out_t->src[0]) == nullptr, "unused projection suffix acquired a gradient");
    check(ggml_graph_get_grad(graph, validation) == nullptr, "validation input acquired a gradient");
    if (paired) check(ggml_graph_get_grad(graph, out_v) == nullptr, "validation branch was not detached");
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_t backends[] = {backend, cpu.get()};
    const int backend_count = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU ? 1 : 2;
    auto * sched = ggml_backend_sched_new(backends, nullptr, backend_count, 4096, false, true);
    check(ggml_backend_sched_alloc_graph(sched, graph), "paired graph allocation failed");
    std::vector<float> weights(ggml_nelements(weight)), x(ggml_nelements(train)), v(ggml_nelements(validation));
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = .1f*std::cos(float(i)*.11f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = .2f*std::sin(float(i)*.07f);
    for (size_t i = 0; i < v.size(); ++i) v[i] = .3f*std::cos(float(i)*.09f);
    std::vector<uint8_t> quantized(ggml_nbytes(weight));
    ggml_quantize_chunk(weight->type, weights.data(), quantized.data(), 0, rows*experts, dim, nullptr);
    ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(train, x.data(), 0, x.size()*sizeof(float));
    ggml_backend_tensor_set(validation, v.data(), 0, v.size()*sizeof(float));
    const int32_t selected[] = {0, 2, 4, 6, 7};
    ggml_backend_tensor_set(crop_ids, selected, 0, sizeof(selected));
    std::vector<float> untouched(64, 3.0f), replacement(64, 7.0f);
    ggml_backend_tensor_set(state, untouched.data(), 0, 64*sizeof(float));
    ggml_backend_tensor_set(patch, replacement.data(), 0, 64*sizeof(float));
    for (auto * ids : { ids_t, ids_v }) {
        std::vector<int32_t> routes(ggml_nelements(ids));
        for (size_t i = 0; i < routes.size(); ++i) routes[i] = (i/used + i%used)%experts;
        ggml_backend_tensor_set(ids, routes.data(), 0, routes.size()*sizeof(int32_t));
    }
    ggml_graph_reset(graph);
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "paired graph compute failed");
    paired_result result;
    result.train.resize(ggml_nelements(out_t)); result.validation.resize(ggml_nelements(out_v)); result.gradient.resize(x.size());
    ggml_backend_tensor_get(read_t, result.train.data(), 0, result.train.size()*sizeof(float));
    ggml_backend_tensor_get(read_v, result.validation.data(), 0, result.validation.size()*sizeof(float));
    ggml_backend_tensor_get(ggml_graph_get_grad(graph, train), result.gradient.data(), 0, result.gradient.size()*sizeof(float));
    ggml_backend_tensor_get(state, replacement.data(), 0, 64*sizeof(float));
    check(replacement == untouched, "inactive branch changed external memory");
    ggml_backend_sched_free(sched);
    return result;
}

static void paired_forward_checks(ggml_backend_t backend) {
    for (bool routed : {false, true}) for (bool broadcast : {false, true}) for (bool zero_copy : {false, true}) for (bool crop : {false, true}) {
        if ((routed || crop) && ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_META) continue;
        if (crop && routed) continue;
        const auto serial = paired_forward(backend, false, routed, broadcast, false, crop);
        const auto paired = paired_forward(backend, true, routed, broadcast, zero_copy, crop);
        const std::pair<const std::vector<float> *, const std::vector<float> *> components[] = {
            { &serial.train, &paired.train }, { &serial.validation, &paired.validation }, { &serial.gradient, &paired.gradient }
        };
        for (const auto & values : components) {
            double error = 0, norm = 0;
            for (size_t i = 0; i < values.first->size(); ++i) {
                const double a = (*values.first)[i], b = (*values.second)[i];
                error += (a-b)*(a-b); norm += a*a;
            }
            const double relative = std::sqrt(error/std::max(norm, 1e-30));
            if (relative >= .02) {
                std::fprintf(stderr, "paired mismatch backend=%s routed=%d broadcast=%d zero_copy=%d component=%s relative=%g first=%g/%g\n",
                    ggml_backend_name(backend), routed, broadcast, zero_copy, values.first == &serial.train ? "train" :
                    values.first == &serial.validation ? "validation" : "gradient", relative, (*values.first)[0], (*values.second)[0]);
            }
            check(relative < .02, "paired projection changed outputs or training gradients");
        }
    }
    std::printf("paired forward: dense/routed outputs, training gradients, detached validation and mutable-weight exclusion passed on %s\n", ggml_backend_name(backend));
}

static double reference(const std::vector<float> & logits, const std::vector<int32_t> & ids,
        const std::vector<float> & logp, const std::vector<float> & weights, float temperature) {
    const int nr = weights.size(), nv = logits.size()/nr, k = ids.size()/nr;
    double loss = 0;
    for (int r = 0; r < nr; ++r) {
        double total = 0;
        const double maximum = *std::max_element(logits.begin() + r*nv, logits.begin() + (r+1)*nv);
        for (int j = 0; j < nv; ++j) total += std::exp((logits[r*nv+j] - maximum) / temperature);
        std::vector<uint8_t> selected(nv, 0);
        for (int j = 0; j < k; ++j) {
            selected[ids[r*k+j]] = true;
            const double student_logp = (logits[r*nv+ids[r*k+j]] - maximum)/temperature - std::log(total);
            loss += weights[r]*std::exp(double(logp[r*(k+1)+j]))*(logp[r*(k+1)+j] - student_logp);
        }
        double tail_max = -INFINITY, tail = 0;
        for (int j = 0; j < nv; ++j) if (!selected[j]) tail_max = std::max(tail_max, double(logits[r*nv+j]));
        for (int j = 0; j < nv; ++j) if (!selected[j]) tail += std::exp((logits[r*nv+j] - tail_max)/temperature);
        const double tail_logp = (tail_max - maximum)/temperature + std::log(tail) - std::log(total);
        loss += weights[r]*std::exp(double(logp[r*(k+1)+k]))*(logp[r*(k+1)+k] - tail_logp);
    }
    return loss*temperature*temperature/nr;
}

static void loss_and_gradient(float temperature, bool masked, bool extreme = false, bool reference_mode = false,
        ggml_backend_t backend = nullptr, int nv = 8) {
    ggml_init_params params = {16*1024*1024, nullptr, backend != nullptr};
    auto * ctx = ggml_init(params);
    std::vector<float> values = {1.3f, -.2f, .5f, -1.f, .8f, -.8f, .1f, -.4f,
        -.1f, .4f, -.3f, 1.1f, .2f, -.5f, .7f, -.9f};
    std::vector<int32_t> indices = {4, 0, 2, 6, 3, 1};
    if (nv != 8) {
        values.resize(size_t(nv)*2);
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i)*0.07f);
        indices[2] = nv - 1;
        indices[5] = nv - 2;
    }
    if (extreme) values[0] = 1000.f;
    const std::vector<float> weights = {1.5f, masked ? 0.f : .5f};
    std::vector<float> targets = {.3f, .25f, .2f, .25f, .4f, .3f, .1f, .2f};
    for (auto & p : targets) p = std::log(p);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nv, 2);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 2);
    auto * logp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto * row_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_param(logits);
    auto loss_function = reference_mode ? ggml_opt_distillation_loss_reference : ggml_opt_distillation_loss;
    auto * loss = loss_function(ctx, logits, ids, logp, row_weights, temperature);
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    auto * buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    check(!backend || buffer, "cannot allocate GPU KL tensors");
    const auto set = [&](ggml_tensor * tensor, const void * data, size_t size) {
        if (backend) ggml_backend_tensor_set(tensor, data, 0, size);
        else std::memcpy(tensor->data, data, size);
    };
    set(logits, values.data(), values.size()*4);
    set(ids, indices.data(), indices.size()*4);
    set(logp, targets.data(), targets.size()*4);
    set(row_weights, weights.data(), weights.size()*4);
    ggml_graph_reset(graph);
    check((backend ? ggml_backend_graph_compute(backend, graph) : ggml_graph_compute_with_ctx(ctx, graph, 2)) ==
            GGML_STATUS_SUCCESS, "loss graph failed");
    float actual_loss;
    std::vector<float> gradient(values.size());
    if (backend) {
        ggml_backend_tensor_get(loss, &actual_loss, 0, 4);
        ggml_backend_tensor_get(ggml_graph_get_grad(graph, logits), gradient.data(), 0, gradient.size()*4);
    } else {
        actual_loss = *static_cast<float *>(loss->data);
        std::memcpy(gradient.data(), ggml_graph_get_grad(graph, logits)->data, gradient.size()*4);
    }
    const double expected = reference(values, indices, targets, weights, temperature);
    check(std::abs(actual_loss - expected) < 2e-5 + 1e-6*std::abs(expected), "KL differs from reference");
    for (size_t test = 0; test < std::min<size_t>(32, values.size()); ++test) {
        const size_t i = values.size() <= 32 ? test : test < indices.size() ?
                (test/3)*nv + indices[test] : (test - 6)*(values.size() - 1)/25;
        const float original = values[i];
        const float eps = 0.001f;
        values[i] = original + eps;
        const double upper = values[i];
        const double plus = reference(values, indices, targets, weights, temperature);
        values[i] = original - eps;
        const double lower = values[i];
        const double minus = reference(values, indices, targets, weights, temperature);
        values[i] = original;
        check(std::abs(gradient[i] - (plus-minus)/(upper-lower)) < 2e-4, "KL gradient differs from finite difference");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void encoder_checks() {
    std::mt19937 random(42);
    for (uint32_t vocab : {13u, 257u, 262144u}) for (uint32_t k : {1u, 7u, std::min(512u, vocab - 1)}) {
        logit_distillation::header h;
        h.vocab = vocab;
        h.top_k = k;
        for (float temperature : {0.5f, 1.f, 2.f}) {
            h.temperature = temperature;
            logit_distillation::row_encoder encoder(h);
            std::vector<float> logits(vocab);
            std::vector<uint8_t> encoded(h.row_bytes());
            std::vector<int32_t> expected(vocab), actual(k);
            std::vector<uint8_t> selected(vocab);
            // Reuse the encoder across ties, ordered values and changing top-k sets.
            for (int pattern = 0; pattern < 5; ++pattern) {
                for (uint32_t i = 0; i < vocab; ++i) {
                    logits[i] = pattern == 0 ? 0 : pattern == 1 ? float(i)*0.001f :
                            pattern == 2 ? -float(i)*0.001f : float(int(random()%2001) - 1000)*0.01f;
                }
                std::iota(expected.begin(), expected.end(), 0);
                std::sort(expected.begin(), expected.end(), [&](int32_t a, int32_t b) {
                    return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
                });
                std::fill(selected.begin(), selected.end(), 0);
                for (uint32_t i = 0; i < k; ++i) selected[expected[i]] = 1;
                const double maximum = logits[expected[0]];
                double total = 0, tail = 0;
                for (uint32_t i = 0; i < vocab; ++i) {
                    const double p = std::exp((double(logits[i]) - maximum)/temperature);
                    total += p;
                    if (!selected[i]) tail += p;
                }
                check(encoder.encode(encoded.data(), logits.data()) == tail/total, "teacher tail differs from full-sort reference");
                std::memcpy(actual.data(), encoded.data(), k*4);
                check(std::equal(actual.begin(), actual.end(), expected.begin()), "bounded heap changed top-k IDs or tie order");
                for (uint32_t i = 0; i < k; ++i) {
                    ggml_fp16_t actual_logp;
                    std::memcpy(&actual_logp, encoded.data() + k*4 + i*2, 2);
                    const float value = (double(logits[expected[i]]) - maximum)/temperature - std::log(total);
                    check(actual_logp == ggml_fp32_to_fp16(value), "encoded top-k probability changed");
                }
                float actual_tail;
                std::memcpy(&actual_tail, encoded.data() + k*6, 4);
                check(actual_tail == float(std::log(std::max(tail, 1e-300)) - std::log(total)), "encoded tail probability changed");
            }
            for (uint32_t index : {0u, vocab - 1}) for (float invalid : {INFINITY, -INFINITY, NAN}) {
                logits[index] = invalid;
                bool rejected = false;
                try { encoder.encode(encoded.data(), logits.data()); }
                catch (const std::exception &) { rejected = true; }
                check(rejected, "non-finite teacher logit accepted");
                logits[index] = 0;
            }
        }
    }
}

static std::pair<double, size_t> tiled_distillation_check(bool tiled, bool backward = true, bool nested = false) {
    constexpr int64_t rows = 513, vocab = 1024, dim = 32;
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    ggml_backend_t devices[] = {backend.get()};
    auto * sched = ggml_backend_sched_new(devices, nullptr, 1, 8192, false, false);
    ggml_context_ptr data(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
    auto * input = ggml_new_tensor_2d(data.get(), GGML_TYPE_F32, dim, rows);
    auto * weight = ggml_new_tensor_2d(data.get(), GGML_TYPE_Q4_0, dim, vocab);
    ggml_set_name(weight, "tiled_head.weight");
    ggml_set_param(weight);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(data.get(), backend.get()));
    std::vector<float> x(dim*rows), w(dim*vocab);
    for (size_t j = 0; j < x.size(); ++j) x[j] = .1f*std::sin(float(j)*.13f);
    for (size_t j = 0; j < w.size(); ++j) w[j] = .1f*std::cos(float(j)*.07f);
    std::vector<uint8_t> packed(ggml_nbytes(weight));
    ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), packed.data(), 0, vocab, dim, nullptr);
    ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
    ggml_context_ptr compute(ggml_init({32*1024*1024, nullptr, true}));
    auto * graph = ggml_new_graph_custom(compute.get(), 8192, true);
    ggml_tensor * output = nullptr;
    for (int64_t start = 0; start < rows; start += tiled ? 256 : rows) {
        const int64_t count = std::min<int64_t>(tiled ? 256 : rows, rows - start);
        auto * view = ggml_view_2d(compute.get(), input, dim, count, input->nb[1], start*input->nb[1]);
        view->flags |= GGML_TENSOR_FLAG_RECOMPUTE_INPUT;
        auto * inner = nested ? ggml_view_2d(compute.get(), view, dim, count, view->nb[1], 0) : view;
        if (nested) inner->flags |= GGML_TENSOR_FLAG_RECOMPUTE_INPUT;
        auto * logits = ggml_tanh(compute.get(), ggml_mul_mat(compute.get(), weight, inner));
        logits->flags |= GGML_TENSOR_FLAG_RECOMPUTE;
        if (nested) {
            logits = ggml_scale(compute.get(), logits, 0.5f);
            logits->flags |= GGML_TENSOR_FLAG_RECOMPUTE;
            logits = ggml_scale(compute.get(), logits, 2.0f);
        }
        output = output ? ggml_concat(compute.get(), output, logits, 1) : logits;
    }
    ggml_build_forward_expand(graph, output);
    auto params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_DISTILLATION);
    params.optimizer = GGML_OPT_OPTIMIZER_TYPE_QLION_QAT;
    params.activation_recompute = true;
    params.distillation_top_k = 3;
    auto * opt = ggml_opt_init(params);
    ggml_opt_qat_register_param(opt, weight);
    ggml_opt_prepare_alloc(opt, compute.get(), graph, input, output);
    ggml_opt_alloc(opt, backward);
    std::vector<int32_t> ids(3*rows), targets(rows);
    std::vector<float> logp(4*rows), weights(rows);
    for (int64_t row = 0; row < rows; ++row) {
        for (int k = 0; k < 3; ++k) ids[3*row + k] = (row + k)%vocab;
        for (int k = 0; k < 4; ++k) logp[4*row + k] = std::log(k == 3 ? .4f : .2f);
        weights[row] = row%7 == 0 ? 0.0f : 1.0f + .25f*(row%3);
        targets[row] = weights[row] > 0 ? row%vocab : -1;
    }
    const auto set = [](ggml_tensor * tensor, const auto & values) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(values[0]));
    };
    set(ggml_opt_distillation_ids(opt), ids);
    set(ggml_opt_distillation_logp(opt), logp);
    set(ggml_opt_sparse_weights(opt), weights);
    set(ggml_opt_sparse_targets(opt), targets);
    auto * result = ggml_opt_result_init();
    ggml_opt_eval(opt, result);
    double loss = 0;
    ggml_opt_result_loss(result, &loss, nullptr);
    const size_t bytes = ggml_backend_sched_get_buffer_size(sched, backend.get());
    if (backward) check(ggml_opt_recompute_regions(opt) == (tiled ? 3 : 1)*(nested ? 2 : 1), "head replay regions changed");
    ggml_opt_result_free(result);
    ggml_opt_free(opt);
    ggml_backend_sched_free(sched);
    return {loss, bytes};
}

static void tiled_distillation_checks() {
    const auto full = tiled_distillation_check(false);
    const auto tiled = tiled_distillation_check(true);
    check(std::isfinite(tiled.first) && std::abs(full.first - tiled.first) < 1e-5, "tiled KL normalization changed with masks or a partial tile");
    check(tiled.second < full.second, "tiled head did not reduce workspace");
    const auto nested = tiled_distillation_check(true, true, true);
    check(std::abs(tiled.first - nested.first) < 1e-7, "nested replay changed KL loss");
    const auto validation = tiled_distillation_check(true, false);
    check(std::abs(full.first - validation.first) < 1e-5, "tiled validation changed KL normalization");
    check(validation.second <= tiled.second, "validation retained backward workspace");
}

static void mixed_cache_checks() {
    const auto root = std::filesystem::temp_directory_path() /
        ("qlion-mixed-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    nlohmann::json manifest = {{"format", "lumen-mixed-cache-v1"}, {"caches", nlohmann::json::array()}};
    const float logits[] = {4, 3, 2, -20, -21, -22, -23, -24};
    for (int source = 0; source < 2; ++source) {
        logit_distillation::header h;
        h.context = 2; h.vocab = 8; h.top_k = 3;
        h.windows = source == 0 ? 3 : 2;
        h.train_windows = h.windows - 1;
        const auto path = root / (std::to_string(source) + ".dat");
        std::ofstream file(path, std::ios::binary);
        h.save(file);
        for (uint64_t window = 0; window < h.windows; ++window) {
            const int32_t tokens[] = {int32_t(source*3 + window), 1};
            const int32_t labels[] = {1, -1};
            logit_distillation::write(file, tokens, sizeof(tokens));
            logit_distillation::write(file, labels, sizeof(labels));
            logit_distillation::encode_row(file, logits, h);
        }
        manifest["caches"].push_back({{"path", path.string()}, {"windows", h.windows}, {"train_windows", h.train_windows}});
    }
    const auto path = root / "mixed.json";
    { std::ofstream file(path); file << manifest; }
    for (int limit : {0, 1}) {
        mixed_cache cache(path.string());
        int64_t split = 0;
        auto * dataset = cache.dataset(2, split, limit);
        check(split == (limit ? 1 : 3), "mixed training partition changed");
        check(ggml_opt_dataset_ndata(dataset) == (limit ? 2 : 5), "mixed window limit changed");
        const int32_t expected[] = {0, 3, 1, 2, 4};
        auto * tokens = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
        const int32_t limited[] = {0, 2};
        for (int64_t i = 0; i < ggml_opt_dataset_ndata(dataset); ++i) {
            check(tokens[2*i] == (limit ? limited[i] : expected[i]), "mixed interleaving changed packed tokens");
        }
        // dataset() also exposes the original row offsets for limited partitions.
        std::vector<uint64_t> offsets(2*ggml_opt_dataset_ndata(dataset));
        ggml_opt_dataset_get_batch_host_aux(dataset, offsets.data(), offsets.size()*sizeof(uint64_t), 0);
        for (size_t i = 0; i < offsets.size(); i += 2) {
            int32_t ids[3];
            float logp[4];
            mixed_cache::fetch(&cache, offsets[i], ids, logp);
            check(ids[0] == 0 && ids[1] == 1 && ids[2] == 2 && offsets[i + 1] == 0, "mixed teacher offsets changed");
        }
        ggml_opt_dataset_free(dataset);
    }
    std::filesystem::remove_all(root);
}

static void cache_checks() {
    logit_distillation::header h;
    h.context = 2; h.vocab = 8; h.top_k = 3; h.windows = 1; h.train_windows = 1;
    const float logits[] = {4, 3, 2, -20, -21, -22, -23, -24};
    std::ostringstream out(std::ios::binary);
    h.save(out);
    const double tail = logit_distillation::encode_row(out, logits, h);
    check(tail < 1e-8 && tail > 0, "export clipped or discarded the teacher tail");
    const std::string content = out.str();
    std::ostringstream serial(std::ios::binary), parallel(std::ios::binary);
    std::vector<const float *> batch(5, logits);
    for (const auto * row : batch) logit_distillation::encode_row(serial, row, h);
    logit_distillation::encode_rows(parallel, batch, h, 3);
    check(serial.str() == parallel.str(), "parallel extraction changed record order or values");
    const auto path = std::filesystem::temp_directory_path() /
            ("qlion-distillation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".dat");
    {
        std::ofstream file(path, std::ios::binary);
        file.write(content.data(), content.size());
    }
    {
        // A torn first window must be removed before extraction resumes.
        logit_distillation::header resume_header = h;
        const int32_t tokens[2] = {0, 1}, labels[2] = {1, -1};
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        resume_header.save(file);
        const uint64_t header_size = file.tellp();
        file.write(reinterpret_cast<const char *>(tokens), sizeof(tokens));
        file.close();
        uint64_t rows = 99;
        check(logit_distillation::resume_cache(path.string(), resume_header, tokens, labels, rows) == 0 && rows == 0 &&
                std::filesystem::file_size(path) == header_size, "torn window was not truncated");
        file.open(path, std::ios::binary | std::ios::app);
        file.write(reinterpret_cast<const char *>(tokens), sizeof(tokens));
        file.write(reinterpret_cast<const char *>(labels), sizeof(labels));
        logit_distillation::encode_row(file, logits, h);
        file.close();
        check(logit_distillation::resume_cache(path.string(), resume_header, tokens, labels, rows) == 1 && rows == 1,
                "completed window was not retained");
        std::ofstream restore(path, std::ios::binary | std::ios::trunc);
        restore.write(content.data(), content.size());
    }
    {
        logit_distillation::cache loaded(path.string());
        const uint64_t offset = loaded.in.tellg();
        int32_t ids[3];
        float probabilities[4];
        loaded.row(offset, ids, probabilities);
        check(ids[0] == 0 && ids[1] == 1 && ids[2] == 2, "top-k IDs differ after reload");
        double sum = 0;
        for (float p : probabilities) sum += std::exp(double(p));
        check(std::abs(sum - 1) < 1e-6, "FP16 teacher distribution was not renormalized");
        for (int32_t invalid : {ids[0], -1, int32_t(h.vocab), INT32_MAX}) {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(offset + 4);
            file.write(reinterpret_cast<const char *>(&invalid), 4);
            file.close();
            bool rejected = false;
            try {
                logit_distillation::cache corrupted(path.string());
                corrupted.row(offset, ids, probabilities);
            } catch (const std::exception &) { rejected = true; }
            check(rejected, "invalid or duplicate cache ID accepted");
        }
    }
    {
        logit_distillation::header boundary = h;
        boundary.vocab = 130;
        boundary.top_k = 4;
        std::vector<float> boundary_logits(130, -10);
        const int32_t expected[] = {0, 63, 64, 129};
        for (int i = 0; i < 4; ++i) boundary_logits[expected[i]] = 4 - i;
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        boundary.save(file);
        const uint64_t offset = file.tellp();
        logit_distillation::encode_row(file, boundary_logits.data(), boundary);
        file.close();
        logit_distillation::cache loaded(path.string());
        int32_t ids[4];
        float logp[5];
        for (int repeat = 0; repeat < 2; ++repeat) {
            loaded.row(offset, ids, logp);
            check(std::equal(ids, ids + 4, expected), "cache bitmap failed across word boundaries or row reuse");
        }
    }
    std::filesystem::remove(path);
    std::istringstream in(content, std::ios::binary);
    logit_distillation::header restored;
    restored.load(in);
    check(restored.top_k == 3 && restored.temperature == 1, "header roundtrip failed");
    bool rejected = false;
    try {
        std::istringstream broken(content.substr(0, 20), std::ios::binary);
        restored.load(broken);
    } catch (const std::exception &) { rejected = true; }
    check(rejected, "truncated header accepted");
    rejected = false;
    try {
        std::istringstream broken("badmagic", std::ios::binary);
        restored.load(broken);
    } catch (const std::exception &) { rejected = true; }
    check(rejected, "invalid cache version accepted");
}

static void benchmark(bool reference_mode) {
    constexpr int nv = 262144, nr = 16, k = 512;
    ggml_init_params params = {size_t(1024)*1024*1024, nullptr, false};
    auto * ctx = ggml_init(params);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nv, nr);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, nr);
    auto * logp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k+1, nr);
    auto * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nr);
    for (int r = 0; r < nr; ++r) {
        static_cast<float *>(weights->data)[r] = 1;
        for (int i = 0; i < nv; ++i) static_cast<float *>(logits->data)[r*nv+i] = std::sin(float(i%109)*.07f);
        for (int i = 0; i < k; ++i) {
            static_cast<int32_t *>(ids->data)[r*k+i] = (i*487+r*13)%nv;
            static_cast<float *>(logp->data)[r*(k+1)+i] = std::log(.5f/k);
        }
        static_cast<float *>(logp->data)[r*(k+1)+k] = std::log(.5f);
    }
    ggml_set_param(logits);
    auto fn = reference_mode ? ggml_opt_distillation_loss_reference : ggml_opt_distillation_loss;
    auto * loss = fn(ctx, logits, ids, logp, weights, 1);
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_graph_reset(graph);
    check(ggml_graph_compute_with_ctx(ctx, graph, 8) == GGML_STATUS_SUCCESS, "benchmark warmup failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) check(ggml_graph_compute_with_ctx(ctx, graph, 8) == GGML_STATUS_SUCCESS, "benchmark failed");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()/3;
    std::printf("distill_bench: implementation=%s vocab=%d rows=%d k=%d threads=8 ms=%.3f tensor_bytes=%zu loss=%.8g\n",
            reference_mode ? "reference" : "fused", nv, nr, k, ms, ggml_used_mem(ctx), *static_cast<float *>(loss->data));
    ggml_free(ctx);
}

static void frozen_q41_moe_gradient() {
    auto * ctx = ggml_init({1024*1024, nullptr, false});
    auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 96, 3, 2);
    auto * grad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
    std::vector<float> original(96*3*2), decoded(original.size());
    for (size_t index = 0; index < original.size(); ++index) original[index] = std::sin(float(index)*0.1f);
    ggml_quantize_chunk(GGML_TYPE_Q4_1, original.data(), weight->data, 0, 6, 96, nullptr);
    ggml_get_type_traits(GGML_TYPE_Q4_1)->to_float(weight->data, decoded.data(), decoded.size());
    const int32_t routes[] = {0, 1, 1, 0};
    std::memcpy(ids->data, routes, sizeof(routes));
    auto * gradient = static_cast<float *>(grad->data);
    for (int index = 0; index < 12; ++index) gradient[index] = float(index - 5)*0.2f;
    auto * output = ggml_mul_mat_id_back(ctx, weight, grad, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    for (int threads : {1, 4}) {
        check(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "Q4_1 MoE backward failed");
        for (int route = 0; route < 4; ++route) for (int column = 0; column < 96; ++column) {
            double expected = 0;
            for (int row = 0; row < 3; ++row) expected += double(decoded[(routes[route]*3 + row)*96 + column])*gradient[route*3 + row];
            check(std::abs(static_cast<float *>(output->data)[route*96 + column] - expected) < 1e-5,
                    "Q4_1 MoE gradient differs from dequantized matrix reference");
        }
    }
    ggml_free(ctx);
}

static ggml_backend_meta_split_state test_split_state(const ggml_tensor * tensor, void * userdata) {
    const auto * axes = static_cast<const int *>(userdata);
    int axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (std::strcmp(tensor->name, "left") == 0) axis = axes[0];
    if (std::strcmp(tensor->name, "right") == 0) axis = axes[1];
    if (std::strcmp(tensor->name, "q") == 0 || std::strcmp(tensor->name, "k") == 0 || std::strcmp(tensor->name, "v") == 0) axis = 2;
    if (std::strcmp(tensor->name, "upstream") == 0) axis = 1;
    ggml_backend_meta_split_state state = {ggml_backend_meta_split_axis(axis), {0}, {1}, 1};
    if (axis >= 0 && axis < GGML_MAX_DIMS) {
        state.ne[0] = tensor->ne[axis]/2;
        state.ne[1] = tensor->ne[axis] - state.ne[0];
    }
    return state;
}

static void tensor_parallel_accumulate(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int mode = 0; mode < 14; ++mode) {
        const bool subset = mode == 6 || mode == 7;
        const bool qkv = mode >= 8;
        const bool zero_qkv = mode >= 11;
        const int part = qkv ? (mode - 8)%3 : 0;
        int axes[] = {mode == 1 ? 1 : 0, mode == 0 || subset || qkv ? 1 : 0};
        auto split = [](const ggml_tensor * tensor, void * userdata) {
            auto state = test_split_state(tensor, userdata);
            if (std::strcmp(tensor->name, "cache") == 0) {
                state.axis = GGML_BACKEND_SPLIT_AXIS_0; state.ne[0] = state.ne[1] = tensor->ne[0]/2;
            }
            if (std::strcmp(tensor->name, "left") == 0 && tensor->ne[0] == 64 && tensor->ne[1] == 4) {
                state.nr[0] = 2;
                state.ne[0] /= 2;
                state.ne[1] /= 2;
            }
            if (std::strcmp(tensor->name, "qkv") == 0) {
                state.axis = tensor->ne[0] == 32 ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
                state.n_segments = 2;
                state.nr[1] = 2;
                state.ne[0] = state.ne[1] = 32;
                state.ne[2] = state.ne[3] = 16;
            }
            return state;
        };
        auto * dev = ggml_backend_meta_device(devices, 2, split, axes);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * a = mode == 1 ? ggml_new_tensor_3d(data, GGML_TYPE_F32, 16, 4, 4) :
            ggml_new_tensor_2d(data, GGML_TYPE_F32, qkv ? 128 : 64, subset ? 16 : 4);
        auto * b = mode == 0 || subset || qkv ? ggml_new_tensor_3d(data, GGML_TYPE_F32, 16, qkv && part != 0 ? 2 : 4, 4) :
            ggml_new_tensor_2d(data, GGML_TYPE_F32, mode == 1 ? 64 : 32, 4);
        ggml_set_name(a, "left"); ggml_set_name(b, "right");
        // Whole-cache reshapes use one segment; fused gate/up gradients use two.
        if (mode == 0 || mode >= 4) ggml_set_name(a, "cache");
        if (qkv) ggml_set_name(a, "qkv");
        auto * weight = zero_qkv ? ggml_new_tensor_2d(data, GGML_TYPE_F32, 32, 128) : nullptr;
        auto * input = zero_qkv ? ggml_new_tensor_2d(data, GGML_TYPE_F32, 32, 4) : nullptr;
        if (weight) ggml_set_name(weight, "qkv");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta accumulation allocation failed");
        std::vector<float> av(ggml_nelements(a)), bv(ggml_nelements(b));
        for (size_t i = 0; i < av.size(); ++i) av[i] = float(i % 13)/13;
        for (size_t i = 0; i < bv.size(); ++i) bv[i] = float(i % 7)/7;
        ggml_backend_tensor_set(a, av.data(), 0, av.size()*sizeof(float));
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size()*sizeof(float));
        if (zero_qkv) {
            std::vector<float> zeros(ggml_nelements(weight), 0.0f);
            ggml_backend_tensor_set(weight, zeros.data(), 0, ggml_nbytes(weight));
            ggml_backend_tensor_set(input, zeros.data(), 0, ggml_nbytes(input));
            a = ggml_scale(compute, ggml_mul_mat(compute, weight, input), 0.0f);
            std::fill(av.begin(), av.end(), 0.0f);
        }
        if (mode == 4 || mode == 5) {
            a = ggml_scale(compute, a, 0.0f);
            std::fill(av.begin(), av.end(), 0.0f);
        }
        const size_t offset = qkv ? (part == 0 ? 0 : part == 1 ? 64 : 96)*sizeof(float) :
            mode == 6 ? 3*a->nb[1] : (mode == 3 || mode == 5 ? 32*sizeof(float) : 0);
        auto * result = ggml_acc(compute, a, b, mode < 2 || subset || qkv ? b->ne[0]*sizeof(float) : a->nb[1],
            qkv ? a->nb[1] : mode < 2 || subset ? b->ne[0]*b->ne[1]*sizeof(float) : a->nb[2], ggml_nbytes(a), offset);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, result);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta accumulation graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta accumulation failed");
        std::vector<float> actual(av.size());
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size()*sizeof(float));
        for (size_t i = 0; i < bv.size(); ++i) {
            const size_t width = b->ne[0]*b->ne[1];
            const size_t index = qkv ? (i/width)*128 + i%width : mode < 2 || subset ? i : (i/32)*64 + i%32;
            av[index + offset/sizeof(float)] += bv[i];
        }
        for (size_t i = 0; i < av.size(); ++i) check(std::abs(actual[i] - av[i]) < 1e-6, "meta accumulation differs from reference");
        ggml_gallocr_free(allocator); ggml_backend_buffer_free(buffer);
        ggml_free(compute); ggml_free(data); ggml_backend_free(backend);
    }
}

static void tensor_parallel_copy(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int axis : {0, 1}) for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        int axes[] = {axis, GGML_BACKEND_SPLIT_AXIS_MIRRORED};
        auto * dev = ggml_backend_meta_device(devices, 2, test_split_state, axes);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * source = ggml_new_tensor_2d(data, GGML_TYPE_F32, 128, 8);
        auto * destination = ggml_new_tensor_2d(data, type, 256, 4);
        ggml_set_name(source, "left");
        ggml_set_name(destination, "right");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta copy input allocation failed");
        std::vector<float> values(ggml_nelements(source));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 37) - 18)/19;
        ggml_backend_tensor_set(source, values.data(), 0, values.size()*sizeof(float));
        auto * result = ggml_cpy(compute, source, destination);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, result);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta copy graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta copy compute failed");
        std::vector<uint8_t> packed(ggml_nbytes(result));
        std::vector<float> actual(values.size());
        ggml_backend_tensor_get(result, packed.data(), 0, packed.size());
        for (size_t offset : {size_t(3), result->nb[1] - 7, packed.size()/2 - 13}) {
            std::vector<uint8_t> part(67);
            ggml_backend_tensor_get(result, part.data(), offset, part.size());
            check(std::memcmp(part.data(), packed.data() + offset, part.size()) == 0, "meta partial copy read differs from full read");
        }
        if (type == GGML_TYPE_F32) {
            std::memcpy(actual.data(), packed.data(), packed.size());
        } else {
            ggml_get_type_traits(type)->to_float(packed.data(), actual.data(), actual.size());
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            check(std::abs(actual[i] - values[i]) < (type == GGML_TYPE_F32 ? 1e-6f : 0.008f), "meta copy differs from reference");
        }
        ggml_gallocr_free(allocator); ggml_backend_buffer_free(buffer);
        ggml_free(compute); ggml_free(data); ggml_backend_free(backend);
    }
}

static void tensor_parallel_partial_write(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int axis : {0, 1, 2}) for (int mode : {0, 1, 2}) for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
        if (axis == 2 && mode != 0) continue;
        int config[] = {axis, mode};
        auto split = [](const ggml_tensor * tensor, void * userdata) {
            const auto * config = static_cast<const int *>(userdata);
            const int axis = config[0];
            ggml_backend_meta_split_state state = {ggml_backend_meta_split_axis(axis), {0}, {1}, 1};
            state.ne[0] = state.ne[1] = tensor->ne[axis]/2;
            if (config[1] == 1) {
                state.nr[0] = 2;
                state.ne[0] /= 2;
                state.ne[1] /= 2;
            } else if (config[1] == 2) {
                state.n_segments = 2;
                state.nr[1] = 1;
                state.ne[0] = state.ne[1] = tensor->ne[axis]/8;
                state.ne[2] = state.ne[3] = 3*tensor->ne[axis]/8;
            }
            return state;
        };
        auto * dev = ggml_backend_meta_device(devices, 2, split, config);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        auto * tensor = ggml_new_tensor_3d(ctx, type, 256, 8, axis == 2 ? 8 : 1);
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
        check(buffer != nullptr, "meta partial write allocation failed");
        std::vector<uint8_t> expected(ggml_nbytes(tensor)), actual(expected.size());
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = uint8_t(i*37 + 11);
        ggml_backend_tensor_set(tensor, expected.data(), 0, expected.size());
        ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
        check(actual == expected, "meta initial write differs from reference");
        for (size_t offset : {size_t(3), tensor->nb[1]/2 - 13, tensor->nb[1] - 7, expected.size()/2 - 13, expected.size() - 1, expected.size()}) {
            const size_t count = std::min(size_t(67), expected.size() - offset);
            std::vector<uint8_t> patch(count);
            for (size_t i = 0; i < count; ++i) patch[i] = uint8_t(offset + i*19);
            std::copy(patch.begin(), patch.end(), expected.begin() + offset);
            ggml_backend_tensor_set(tensor, patch.data(), offset, count);
            ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
            check(actual == expected, "meta partial write changes unexpected bytes");
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    }
}

static void tensor_parallel_async(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    const char * mode = std::getenv("GGML_CUDA_ALLREDUCE");
    if (mode && std::strcmp(mode, "none") == 0) return;
    ggml_backend_ptr first_backend(ggml_backend_dev_init(first, nullptr));
    ggml_backend_ptr second_backend(ggml_backend_dev_init(second, nullptr));
    ggml_backend_t backends[] = {first_backend.get(), second_backend.get()};
    auto * reg = ggml_backend_dev_backend_reg(first);
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto reduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    auto destroy = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    check(init && reduce && destroy, "CUDA collective interface is missing");
    void * comm = init(backends, 2);
    check(comm != nullptr, "CUDA collective initialization failed");
    for (int64_t count : {1024, 1048576, 1032, 9437184, 1024}) {
        ggml_context_ptr contexts[2];
        ggml_backend_buffer_ptr buffers[2];
        ggml_tensor * tensors[2];
        ggml_cgraph * graphs[2];
        std::vector<float> values(count);
        for (int j = 0; j < 2; ++j) {
            contexts[j].reset(ggml_init({1024*1024, nullptr, true}));
            tensors[j] = ggml_new_tensor_1d(contexts[j].get(), GGML_TYPE_F32, count);
            tensors[j]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            auto * scaled = ggml_scale_inplace(contexts[j].get(), tensors[j], 0.5f);
            graphs[j] = ggml_new_graph(contexts[j].get());
            ggml_build_forward_expand(graphs[j], scaled);
            buffers[j].reset(ggml_backend_alloc_ctx_tensors(contexts[j].get(), backends[j]));
            check(buffers[j] != nullptr, "CUDA collective allocation failed");
            std::fill(values.begin(), values.end(), float(j + 1));
            ggml_backend_tensor_set(tensors[j], values.data(), 0, ggml_nbytes(tensors[j]));
        }
        if (mode && std::strcmp(mode, "internal") == 0) {
            const char * threshold_env = std::getenv("GGML_CUDA_AR_BF16_THRESHOLD");
            const size_t threshold = threshold_env ? std::strtoull(threshold_env, nullptr, 10) : 1;
            const bool bf16 = threshold > 0 && count*sizeof(float) >= threshold;
            std::vector<float> expected(count, 0.0f);
            for (int active : {1, 2}) {
                std::fill(expected.begin(), expected.end(), 0.0f);
                for (int j = 0; j < 2; ++j) {
                    tensors[j]->flags = j < active ? GGML_TENSOR_FLAG_COMPUTE : 0;
                    for (int64_t k = 0; k < count; ++k) {
                        values[k] = std::sin(float(k%7919)*0.013f + j)*7.0f;
                        if (j < active) expected[k] += bf16 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(values[k])) : values[k];
                    }
                    ggml_backend_tensor_set(tensors[j], values.data(), 0, ggml_nbytes(tensors[j]));
                }
                check(reduce(comm, tensors), "CUDA collective reference submission failed");
                for (int j = 0; j < 2; ++j) {
                    ggml_backend_synchronize(backends[j]);
                    ggml_backend_tensor_get(tensors[j], values.data(), 0, ggml_nbytes(tensors[j]));
                    check(std::memcmp(values.data(), expected.data(), count*sizeof(float)) == 0, "CUDA collective differs from exact rounded-input reference");
                }
            }
            for (int j = 0; j < 2; ++j) {
                tensors[j]->flags = GGML_TENSOR_FLAG_COMPUTE;
                std::fill(values.begin(), values.end(), float(j + 1));
                ggml_backend_tensor_set(tensors[j], values.data(), 0, ggml_nbytes(tensors[j]));
            }
        }
        // Queue enough reductions to reuse event slots without a host synchronization.
        for (int i = 0; i < 16; ++i) {
            check(reduce(comm, tensors), "CUDA collective submission failed");
            for (int j = 0; j < 2; ++j) check(ggml_backend_graph_compute_async(backends[j], graphs[j]) == GGML_STATUS_SUCCESS, "CUDA collective scale failed");
        }
        for (int j = 0; j < 2; ++j) {
            ggml_backend_synchronize(backends[j]);
            ggml_backend_tensor_get(tensors[j], values.data(), 0, ggml_nbytes(tensors[j]));
            for (float value : values) check(value == 1.5f, "queued CUDA collective changed a value");
        }
    }
    destroy(comm);
    std::puts("CUDA collective: queued mixed-size reductions and event reuse passed");
}

static void tensor_parallel_host_copy(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int axis : {0, 1, 2}) {
        auto split = [](const ggml_tensor * tensor, void * userdata) {
            const int axis = *static_cast<int *>(userdata);
            ggml_backend_meta_split_state state = {ggml_backend_meta_split_axis(axis), {0}, {1}, 1};
            state.ne[0] = state.ne[1] = tensor->ne[axis]/2;
            return state;
        };
        ggml_backend_ptr meta(ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, split, &axis), nullptr));
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        ggml_context_ptr data(ggml_init({1024*1024, nullptr, true}));
        ggml_context_ptr compute(ggml_init({1024*1024, nullptr, true}));
        auto * input = ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, 32, 16, 4);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(data.get(), meta.get()));
        std::vector<float> values(ggml_nelements(input));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i%31) - 15);
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        auto * scaled = ggml_scale(compute.get(), input, 2.0f);
        auto * host = ggml_cont(compute.get(), scaled);
        host->flags |= GGML_TENSOR_FLAG_HOST;
        auto * graph = ggml_new_graph(compute.get());
        ggml_build_forward_expand(graph, host);
        ggml_backend_t backends[] = {meta.get(), cpu.get()};
        auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
        ggml_backend_sched_set_tensor_backend(sched, scaled, meta.get());
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "Meta host copy failed");
        std::vector<float> actual(values.size());
        ggml_backend_tensor_get(host, actual.data(), 0, ggml_nbytes(host));
        for (size_t i = 0; i < values.size(); ++i) check(actual[i] == 2*values[i], "Meta host copy changed a shard");
        ggml_backend_sched_free(sched);
    }
}

static void host_gradient_dependency(ggml_backend_dev_t device) {
    ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    ggml_context_ptr host_data(ggml_init({1024*1024, nullptr, true}));
    ggml_context_ptr gpu_data(ggml_init({1024*1024, nullptr, true}));
    ggml_context_ptr compute(ggml_init({1024*1024, nullptr, true}));
    auto * a = ggml_new_tensor_2d(host_data.get(), GGML_TYPE_F32, 32, 128);
    auto * b = ggml_new_tensor_2d(host_data.get(), GGML_TYPE_F32, 16, 128);
    auto * acc = ggml_new_tensor_2d(host_data.get(), GGML_TYPE_F32, 32, 16);
    auto * input = ggml_new_tensor_2d(gpu_data.get(), GGML_TYPE_F32, 32, 16);
    ggml_backend_buffer_ptr host_buffer(ggml_backend_alloc_ctx_tensors(host_data.get(), cpu.get()));
    ggml_backend_buffer_ptr gpu_buffer(ggml_backend_alloc_ctx_tensors(gpu_data.get(), gpu.get()));
    auto * product = ggml_out_prod(compute.get(), a, b);
    auto * gradient = ggml_add_inplace(compute.get(), acc, product);
    auto * independent = ggml_scale(compute.get(), input, 2.0f);
    auto * output = ggml_add(compute.get(), independent, gradient);
    auto * graph = ggml_new_graph(compute.get());
    ggml_build_forward_expand(graph, gradient);
    ggml_build_forward_expand(graph, independent);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {gpu.get(), cpu.get()};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    ggml_backend_sched_set_tensor_backend(sched, product, cpu.get());
    ggml_backend_sched_set_tensor_backend(sched, gradient, cpu.get());
    ggml_backend_sched_set_tensor_backend(sched, independent, gpu.get());
    ggml_backend_sched_set_tensor_backend(sched, output, gpu.get());
    check(ggml_backend_sched_alloc_graph(sched, graph), "host gradient allocation failed");
    for (int iteration = 0; iteration < 32; ++iteration) {
        for (auto * tensor : {a, b, acc, input}) {
            const float value = tensor == a ? float(iteration + 1) : tensor == b ? .25f : tensor == acc ? 1.0f : float(iteration);
            std::vector<float> values(ggml_nelements(tensor), value);
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "host gradient execution failed");
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        const float expected = 32.0f*(iteration + 1) + 1.0f + 2.0f*iteration;
        for (float value : actual) check(value == expected, "GPU consumed an incomplete host gradient");
    }
    ggml_backend_sched_free(sched);
    std::puts("host gradient: repeated in-place accumulation and GPU dependency checks passed");
}

static void tensor_parallel_sum(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int axis : {0, 1}) {
        int axes[] = {axis, axis};
        auto * dev = ggml_backend_meta_device(devices, 2, test_split_state, axes);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * input = ggml_new_tensor_2d(data, GGML_TYPE_F32, 32, 16);
        auto * other = ggml_new_tensor_2d(data, GGML_TYPE_F32, 32, 16);
        ggml_set_name(input, "left");
        ggml_set_name(other, "right");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta sum input allocation failed");
        std::vector<float> values(ggml_nelements(input));
        float expected = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float(int(i % 7) - 3);
            expected += values[i]*2;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
        ggml_tensor metadata_copy = *input;
        std::vector<float> copied(values.size());
        ggml_backend_tensor_get(&metadata_copy, copied.data(), 0, ggml_nbytes(input));
        check(copied == values, "meta copied model metadata read differs from reference");
        std::fill(copied.begin(), copied.end(), -13.0f);
        ggml_backend_tensor_set(other, copied.data(), 0, ggml_nbytes(other));
        metadata_copy = *other;
        ggml_backend_tensor_get(&metadata_copy, copied.data(), 0, ggml_nbytes(other));
        for (float value : copied) check(value == -13.0f, "meta metadata copy read the wrong model tensor");
        auto * scaled = ggml_scale(compute, input, 2.0f);
        check(ggml_backend_buft_get_alloc_size(buft, scaled) == ggml_nbytes(scaled)/2,
                "meta workspace must use local split size");
        auto * sum = ggml_scale(compute, ggml_sum(compute, scaled), 1.0f);
        auto * transposed = ggml_transpose(compute, scaled);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, sum);
        ggml_build_forward_expand(graph, transposed);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta sum graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta sum compute failed");
        float actual = 0;
        ggml_backend_tensor_get(sum, &actual, 0, sizeof(actual));
        check(actual == expected, "meta scalar sum differs from reference");
        std::vector<float> transposed_storage(values.size());
        ggml_backend_tensor_get(transposed, transposed_storage.data(), 0, ggml_nbytes(transposed));
        for (size_t i = 0; i < values.size(); ++i) {
            check(transposed_storage[i] == values[i]*2, "meta permuted storage read differs from reference");
        }
        std::vector<uint8_t> part(67);
        ggml_backend_tensor_get(transposed, part.data(), 13, part.size());
        check(std::memcmp(part.data(), reinterpret_cast<const uint8_t *>(transposed_storage.data()) + 13, part.size()) == 0,
                "meta partial permuted read differs from full read");
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(compute);
        ggml_free(data);
    }
}

static void tensor_parallel_backward(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    const int mirrored = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    static int axes[][2] = {{1, 1}, {0, mirrored}, {mirrored, 0}};
    static int routed_axes[][2] = {{1, 0}, {0, mirrored}, {mirrored, 0}};
    for (bool routed : {false, true}) for (size_t index = 0; index < 3; ++index) {
        auto & split = axes[index];
        if (routed && split[0] == mirrored) continue;
        auto * dev = ggml_backend_meta_device(devices, 2, test_split_state, routed ? routed_axes[index] : split);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * left = ggml_new_tensor_3d(data, routed ? GGML_TYPE_Q4_0 : GGML_TYPE_F32, routed ? 64 : 32, 64, routed ? 4 : 1);
        auto * right = routed ? ggml_new_tensor_3d(data, GGML_TYPE_F32, 64, 2, 3) :
                ggml_new_tensor_2d(data, GGML_TYPE_F32, 16, 64);
        auto * ids = routed ? ggml_new_tensor_2d(data, GGML_TYPE_I32, 2, 3) : nullptr;
        ggml_set_name(left, "left");
        ggml_set_name(right, "right");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta input allocation failed");
        std::vector<float> a(ggml_nelements(left)), b(ggml_nelements(right));
        for (size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 17) - 8)/17;
        for (size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 11) - 5)/11;
        if (routed) {
            std::vector<uint8_t> packed(ggml_nbytes(left));
            ggml_quantize_chunk(GGML_TYPE_Q4_0, a.data(), packed.data(), 0, ggml_nrows(left), left->ne[0], nullptr);
            ggml_backend_tensor_set(left, packed.data(), 0, packed.size());
            ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(packed.data(), a.data(), a.size());
        } else {
            ggml_backend_tensor_set(left, a.data(), 0, a.size()*sizeof(float));
        }
        ggml_backend_tensor_set(right, b.data(), 0, b.size()*sizeof(float));
        const int32_t routes[] = {0, 3, 2, 1, 3, 2};
        if (ids) ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
        auto * result = routed ? ggml_mul_mat_id_back(compute, left, right, ids) : ggml_out_prod(compute, left, right);
        result = ggml_scale(compute, result, 1.0f);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, result);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta backward failed");
        std::vector<float> actual(ggml_nelements(result));
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size()*sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            const size_t width = left->ne[0];
            const size_t column = i % width;
            const size_t row = i / width;
            const size_t expert = routed ? routes[row] : 0;
            double expected = 0;
            for (size_t k = 0; k < 64; ++k) {
                expected += double(a[expert*width*64 + k*width + column]) * b[routed ? row*64 + k : k*16 + row];
            }
            check(std::abs(actual[i] - expected) < 2e-5, "tensor parallel backward differs from reference");
        }
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_free(compute);
        ggml_free(data);
        ggml_backend_free(backend);
    }
}

static std::vector<float> attention_backward(bool parallel, bool q_only = false) {
    auto * cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    ggml_backend_dev_t devices[] = {cpu, cpu};
    static int axes[] = {2, 2};
    auto * dev = parallel ? ggml_backend_meta_device(devices, 2, test_split_state, axes) : cpu;
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(dev);
    auto * data = ggml_init({1024*1024, nullptr, true});
    auto * compute = ggml_init({1024*1024, nullptr, true});
    ggml_tensor * inputs[] = {
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 8, 4),
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 8, 2),
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 8, 2),
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 4, 8),
    };
    const char * names[] = {"q", "k", "v", "upstream"};
    for (int i = 0; i < 4; ++i) ggml_set_name(inputs[i], names[i]);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
    for (int i = 0; i < 4; ++i) {
        std::vector<float> values(ggml_nelements(inputs[i]));
        for (size_t j = 0; j < values.size(); ++j) values[j] = float(int((j + i) % 19) - 9)/19;
        ggml_backend_tensor_set(inputs[i], values.data(), 0, values.size()*sizeof(float));
    }
    auto * forward = ggml_flash_attn_ext(compute, inputs[0], inputs[1], inputs[2], nullptr, 0.125f, 0, 0);
    auto * back = ggml_flash_attn_ext_back(compute, forward, inputs[3]);
    if (q_only) {
        back->ne[0] = ggml_nelements(inputs[0]);
        back->nb[1] = back->nb[2] = back->nb[3] = back->ne[0]*sizeof(float);
        const int flags = GGML_FLASH_ATTN_BACK_GRAD_VALID | GGML_FLASH_ATTN_BACK_GRAD_Q;
        std::memcpy(back->op_params + 5, &flags, sizeof(flags));
        if (parallel) {
            check(ggml_backend_buft_get_alloc_size(buft, back) == ggml_nbytes(back)/2,
                    "meta Q-only backward reserved unused K/V gradient segments");
        }
    }
    auto * graph = ggml_new_graph(compute);
    std::vector<ggml_tensor *> views;
    size_t offset = 0;
    for (int i = 0; i < (q_only ? 1 : 3); ++i) {
        auto * input = inputs[i];
        auto * view = ggml_view_4d(compute, back, input->ne[0], input->ne[1], input->ne[2], 1,
                input->ne[0]*4, input->ne[0]*input->ne[1]*4, ggml_nelements(input)*4, offset);
        views.push_back(view);
        ggml_build_forward_expand(graph, view);
        offset += GGML_PAD(ggml_nelements(input)*4, GGML_MEM_ALIGN);
    }
    auto * allocator = ggml_gallocr_new(buft);
    check(ggml_gallocr_alloc_graph(allocator, graph), "attention gradient allocation failed");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention gradient compute failed");
    std::vector<float> result(offset/4);
    offset = 0;
    for (auto * view : views) {
        ggml_backend_tensor_get(view, result.data() + offset, 0, ggml_nbytes(view));
        offset += ggml_nelements(view);
    }
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(buffer);
    ggml_free(compute);
    ggml_free(data);
    ggml_backend_free(backend);
    return result;
}

static std::vector<float> full_context_backward(uint32_t tile, bool flash, bool last_only, ggml_backend_t backend, bool q_only = false, bool host = false, bool swa = false, bool crop = false) {
    llm_graph_result result(8192);
    llama_adapter_loras loras;
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    llm_graph_params params = {};
    params.model = model.get();
    model->hparams.n_layer_all = 2;
    model->hparams.n_embd = 64;
    model->hparams.n_head_arr[0] = 2;
    model->hparams.n_head_kv_arr[0] = 1;
    model->hparams.n_embd_head_k_full = 32;
    model->hparams.n_embd_head_v_full = 32;
    params.cparams.n_ctx = 1024;
    params.cparams.n_ubatch = tile;
    params.cparams.flash_attn = flash;
    params.cparams.offload_kqv = true;
    params.cparams.op_offload = true;
    params.cparams.causal_attn = true;
    params.ubatch.n_tokens = 1024;
    std::vector<llama_pos> positions(1024);
    for (int i = 0; i < 1024; ++i) positions[i] = i;
    if (swa) {
        model->hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        model->hparams.n_swa = 128;
        model->hparams.is_swa_impl[0] = 1;
        if (crop) params.ubatch.pos = positions.data();
    }
    params.training = true;
    params.full_sequence = true;
    params.loras = &loras;
    params.res = &result;
    llm_graph_context builder(params);
    auto * ctx = result.get_ctx();
    ggml_context_ptr data(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
    ggml_tensor * inputs[] = {
        ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, 32, 2, 1024),
        ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, 32, 1, 1024),
        ggml_new_tensor_3d(data.get(), GGML_TYPE_F32, 32, 1, 1024),
        ggml_new_tensor_2d(data.get(), flash ? GGML_TYPE_F16 : GGML_TYPE_F32, 1024, 1024),
    };
    ggml_backend_buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(data.get(), backend));
    check(input_buffer != nullptr, "full-context input allocation failed");
    ggml_set_param(inputs[0]);
    if (!q_only) {
        ggml_set_param(inputs[1]);
        ggml_set_param(inputs[2]);
    }
    auto * output = builder.build_attn_mha(inputs[0], inputs[1], inputs[2], nullptr, inputs[3], nullptr, nullptr, nullptr, 0, .1767767f, 0);
    if (tile == 1024 && flash && !last_only && !q_only && !host && !swa) {
        builder.build_attn_mha(inputs[0], inputs[1], inputs[2], nullptr, inputs[3], nullptr, nullptr, nullptr, 0, .1767767f, 1);
        std::set<const ggml_tensor *> masks;
        int attention_nodes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(result.get_gf()); ++i) {
            const auto * node = ggml_graph_node(result.get_gf(), i);
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                masks.insert(node->src[3]);
                ++attention_nodes;
            }
        }
        check(attention_nodes == 8 && masks.size() == 4, "training layers must share fixed mask tiles");
    }
    int64_t largest_query = 0;
    int64_t largest_kv = 0;
    int32_t largest_backward_kv = 0;
    for (int i = 0; i < ggml_graph_n_nodes(result.get_gf()); ++i) {
        const auto * node = ggml_graph_node(result.get_gf(), i);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            largest_query = std::max(largest_query, node->src[0]->ne[1]);
            largest_kv = std::max(largest_kv, node->src[1]->ne[1]);
            int32_t backward_kv;
            std::memcpy(&backward_kv, node->op_params + 7, sizeof(backward_kv));
            largest_backward_kv = std::max(largest_backward_kv, backward_kv);
        }
        if (node->op == GGML_OP_SOFT_MAX) largest_query = std::max(largest_query, node->ne[1]);
    }
    check(largest_query == LLM_TRAINING_TILE_SIZE, "attention did not use the fixed training tile");
    if (swa && crop) check(largest_kv == 1024 && largest_backward_kv > 0 && largest_backward_kv <= 512, "SWA backward range is missing or changes forward");
    if (host) {
        output = ggml_cont(ctx, output);
        output->flags |= GGML_TENSOR_FLAG_HOST;
    }
    auto * last = ggml_view_1d(ctx, output, 64, 1023*output->nb[1]);
    auto * loss = ggml_sum(ctx, last_only ? last : ggml_sqr(ctx, output));
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 8192, true);
    ggml_build_forward_expand(graph, loss);
    ggml_set_fused_backward(ctx, true);
    ggml_build_backward_expand(ctx, graph, nullptr);
    if (flash && q_only) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_FLASH_ATTN_BACK) {
                check(ggml_nbytes(node) == ggml_nelements(node->src[0])*sizeof(float), "Q-only backward reserved unused full-context K/V gradients");
                auto * vulkan = ggml_backend_dev_by_name("Vulkan0");
                if (vulkan) check(!ggml_backend_dev_supports_op(vulkan, node), "Vulkan accepted compact gradients without selective writes");
            }
        }
    }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    ggml_backend_t backends[] = {backend, cpu.get()};
    const bool on_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    auto * sched = ggml_backend_sched_new(backends, nullptr, on_cpu ? 1 : 2, 8192, false, true);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (!(node->flags & GGML_TENSOR_FLAG_HOST) && !ggml_is_view(node) && ggml_backend_supports_op(backend, node)) {
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
        }
    }
    check(ggml_backend_sched_alloc_graph(sched, graph), "full-context graph allocation failed");
    if (host) check(ggml_backend_dev_type(ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched, output))) == GGML_BACKEND_DEVICE_TYPE_CPU, "host boundary stayed on GPU");
    for (int i = 0; i < 3; ++i) {
        std::vector<float> values(ggml_nelements(inputs[i]));
        for (size_t j = 0; j < values.size(); ++j) values[j] = .2f*std::sin(float(j + i*17)*.07f);
        ggml_backend_tensor_set(inputs[i], values.data(), 0, values.size()*sizeof(float));
    }
    std::vector<float> mask(1024*1024);
    for (int q = 0; q < 1024; ++q) for (int k = 0; k < 1024; ++k) mask[q*1024 + k] = k > q || (swa && q - k >= 128) ? -INFINITY : 0.0f;
    if (flash) {
        std::vector<ggml_fp16_t> half(mask.size());
        ggml_fp32_to_fp16_row(mask.data(), half.data(), mask.size());
        ggml_backend_tensor_set(inputs[3], half.data(), 0, half.size()*sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(inputs[3], mask.data(), 0, mask.size()*sizeof(float));
    }
    ggml_graph_reset(graph);
    check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "full-context backward failed");
    std::vector<float> gradients(4*32*1024);
    size_t offset = 0;
    for (int i = 0; i < (q_only ? 1 : 3); ++i) {
        auto * grad = ggml_graph_get_grad(graph, inputs[i]);
        check(grad != nullptr, "Q/K/V gradient is disconnected");
        ggml_backend_tensor_get(grad, gradients.data() + offset, 0, ggml_nelements(inputs[i])*sizeof(float));
        offset += ggml_nelements(inputs[i]);
    }
    ggml_backend_sched_free(sched);
    return gradients;
}

static void full_context_checks(ggml_backend_t backend) {
    for (bool last_only : {false, true}) {
        const auto reference = full_context_backward(1024, true, last_only, backend, false, false, true, false);
        const auto cropped = full_context_backward(1024, true, last_only, backend, false, false, true, true);
        double error = 0, norm = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            check(std::isfinite(cropped[i]), "non-finite cropped SWA gradient");
            error += std::pow(double(cropped[i]) - reference[i], 2);
            norm += double(reference[i])*reference[i];
        }
        const double relative = std::sqrt(error/std::max(norm, 1e-30));
        std::printf("cropped SWA: last_only=%d gradient_relative_l2=%.9g\n", last_only, relative);
        check(relative < 1e-5, "cropped SWA changed full-context gradients");
    }
    for (bool flash : {false, true}) for (bool last_only : {false, true}) {
        const auto reference = full_context_backward(1024, flash, last_only, backend);
        const auto host = full_context_backward(1024, flash, last_only, backend, false, true);
        double host_error = 0, host_norm = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            host_error += std::pow(double(reference[i]) - host[i], 2);
            host_norm += double(reference[i])*reference[i];
        }
        check(std::sqrt(host_error/std::max(host_norm, 1e-30)) < 1e-5, "RAM offload changed full-context gradients");
        if (flash) {
            const auto q_only = full_context_backward(768, flash, last_only, backend, true);
            double error = 0, norm = 0;
            for (size_t i = 0; i < 2*32*1024; ++i) {
                error += std::pow(double(reference[i]) - q_only[i], 2);
                norm += double(reference[i])*reference[i];
            }
            check(std::sqrt(error/std::max(norm, 1e-30)) < 1e-5, "compact Q-only backward changed Q gradients");
        }
        for (uint32_t tile : {768u, 512u, 256u}) {
            const auto host_tiled = full_context_backward(tile, flash, last_only, backend, false, true);
            check(std::memcmp(host.data(), host_tiled.data(), host.size()*sizeof(float)) == 0, "ubatch changed RAM-offloaded gradient bits");
            const auto tiled = full_context_backward(tile, flash, last_only, backend);
            check(std::memcmp(reference.data(), tiled.data(), reference.size()*sizeof(float)) == 0, "ubatch changed full-context gradient bits");
            for (int qkv = 0; qkv < 3; ++qkv) {
                double prefix_norm = 0, error = 0, norm = 0;
                const size_t begin = qkv == 0 ? 0 : (qkv + 1)*32*1024;
                const size_t count = qkv == 0 ? 2*32*1024 : 32*1024;
                for (size_t i = begin; i < begin + count; ++i) {
                    check(std::isfinite(tiled[i]), "non-finite full-context gradient");
                    error += std::pow(double(tiled[i]) - reference[i], 2);
                    norm += double(reference[i])*reference[i];
                    if (i - begin < 32*512) prefix_norm += double(tiled[i])*tiled[i];
                }
                if (qkv > 0) check(prefix_norm > 1e-12, "loss did not reach the first K/V tile");
                const double relative = std::sqrt(error/std::max(norm, 1e-30));
                std::printf("full-context gradient: backend=%s flash=%d last_only=%d qkv=%d relative_l2=%.9g prefix_l2=%.9g\n",
                        ggml_backend_name(backend), flash, last_only, qkv, relative, std::sqrt(prefix_norm));
                check(relative == 0, "full-context gradients differ");
            }
        }
    }
    std::printf("full-context: ctx1024 ub256/512/768/1024 bit-exact K/V gradients and last-token-to-prefix backward passed on %s\n", ggml_backend_name(backend));
}

static void compact_encoding_checks(ggml_backend_t backend = nullptr) {
    for (int nv : {13, 257, 4097, 262144}) for (int nr : {1, 3}) for (int k : {1, std::min(nv - 1, 512)})
            for (int mode : {0, 1, 2, 3, 4}) for (float t : {.5f, 1.f, 2.f}) {
        ggml_init_params params = {32*1024*1024, nullptr, backend != nullptr};
        auto * ctx = ggml_init(params);
        auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nv, nr);
        auto * ids = ggml_top_k(ctx, logits, k);
        auto * stats = ggml_top_k_stats(ctx, logits, ids, t);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, stats);
        auto * buffer = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
        check(!backend || buffer, "cannot allocate compact tensors");
        std::vector<float> values(size_t(nv)*nr);
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> random(-20.f, 20.f);
        for (auto & value : values) value = mode == 0 ? random(generator) : mode == 1 ? std::round(random(generator)) :
                mode == 2 ? (generator()%2 ? 0.f : -0.f) : -1000.f;
        if (mode == 3) for (int r = 0; r < nr; ++r) values[size_t(r)*nv] = 1000.f;
        if (mode == 4) for (int r = 0; r < nr; ++r) values[size_t(r)*nv + 7] = NAN;
        if (backend) ggml_backend_tensor_set(logits, values.data(), 0, values.size()*sizeof(float));
        else std::memcpy(logits->data, values.data(), values.size()*sizeof(float));
        check((backend ? ggml_backend_graph_compute(backend, graph) : ggml_graph_compute_with_ctx(ctx, graph, 3)) ==
                GGML_STATUS_SUCCESS, "compact graph failed");
        std::vector<int32_t> actual_ids(size_t(k)*nr);
        std::vector<double> actual_stats(size_t(k + 3)*nr);
        if (backend) {
            ggml_backend_tensor_get(ids, actual_ids.data(), 0, actual_ids.size()*sizeof(int32_t));
            ggml_backend_tensor_get(stats, actual_stats.data(), 0, actual_stats.size()*sizeof(double));
        } else {
            std::memcpy(actual_ids.data(), ids->data, actual_ids.size()*sizeof(int32_t));
            std::memcpy(actual_stats.data(), stats->data, actual_stats.size()*sizeof(double));
        }
        logit_distillation::header h;
        h.vocab = nv; h.top_k = k; h.temperature = t;
        std::ostringstream compact(std::ios::binary), reference(std::ios::binary);
        bool rejected = false;
        try {
            logit_distillation::encode_compact_rows(compact, actual_ids.data(), actual_stats.data(), nr, h);
        } catch (const std::runtime_error &) { rejected = true; }
        check(rejected == (mode == 4), "compact non-finite validation failed");
        if (mode != 4) {
            for (int r = 0; r < nr; ++r) logit_distillation::encode_row(reference, values.data() + size_t(r)*nv, h);
            const auto a = compact.str(), b = reference.str();
            for (int r = 0; r < nr; ++r) {
                const size_t offset = size_t(r)*h.row_bytes();
                if (std::memcmp(a.data() + offset, b.data() + offset, size_t(k)*6) != 0) {
                    std::fprintf(stderr, "compact mismatch: nv=%d mode=%d temperature=%g row=%d backend=%s\n", nv, mode, t, r,
                            backend ? ggml_backend_name(backend) : "CPU");
                    check(false, "compact IDs or FP16 logp differ");
                }
                float ta, tb;
                std::memcpy(&ta, a.data() + offset + size_t(k)*6, 4);
                std::memcpy(&tb, b.data() + offset + size_t(k)*6, 4);
                check(std::abs(ta - tb) <= 1e-6f, "compact tail differs");
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
}

static void swa_fused_backward_checks(ggml_backend_t backend) {
    constexpr int dim = 256, keys = 1536, queries = 32, heads = 4, kv_heads = 2;
    for (bool q_only : {false, true}) for (bool masked : {false, true}) for (bool with_sink : {false, true}) {
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
        auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, queries, heads);
        auto * k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, keys, kv_heads);
        auto * v = ggml_dup_tensor(ctx.get(), k);
        auto * d = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, heads, queries);
        auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, keys, queries);
        auto * sink = with_sink ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, heads) : nullptr;
        auto * forward = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/16, 0, 0);
        ggml_flash_attn_ext_set_causal(forward, true);
        ggml_flash_attn_ext_add_sinks(forward, sink);
        auto * reference = ggml_flash_attn_ext_back(ctx.get(), forward, d);
        auto * fused = ggml_flash_attn_ext_back(ctx.get(), forward, d);
        const int range_begin = keys - 1280, range_count = 1280;
        std::memcpy(fused->op_params + 6, &range_begin, sizeof(range_begin));
        std::memcpy(fused->op_params + 7, &range_count, sizeof(range_count));
        const int flags = GGML_FLASH_ATTN_BACK_GRAD_VALID | GGML_FLASH_ATTN_BACK_GRAD_Q |
            (q_only ? 0 : GGML_FLASH_ATTN_BACK_GRAD_K | GGML_FLASH_ATTN_BACK_GRAD_V);
        for (auto * tensor : {reference, fused}) {
            std::memcpy(tensor->op_params + 5, &flags, sizeof(flags));
            tensor->ne[0] = ggml_nelements(q) + (q_only ? 0 : ggml_nelements(k) + ggml_nelements(v));
            tensor->nb[1] = tensor->nb[2] = tensor->nb[3] = tensor->ne[0]*sizeof(float);
        }
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, reference);
        ggml_build_forward_expand(graph, fused);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        check(buffer != nullptr, "SWA fused backward allocation failed");
        std::mt19937 random(19);
        std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
        for (auto * tensor : {q, k, v, d, sink}) {
            if (!tensor) continue;
            std::vector<float> values(ggml_nelements(tensor));
            for (auto & value : values) value = distribution(random);
            if (tensor->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> packed(values.size());
                ggml_fp32_to_fp16_row(values.data(), packed.data(), values.size());
                ggml_backend_tensor_set(tensor, packed.data(), 0, ggml_nbytes(tensor));
            } else {
                ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
            }
        }
        std::vector<float> values(keys*queries, -INFINITY);
        for (int j = 0; j < queries; ++j) for (int i = 0; i < keys; ++i) {
            const int position = keys - queries + j;
            if (!masked && i <= position && i > position - 1024) values[j*keys + i] = distribution(random)*0.25f;
        }
        std::vector<ggml_fp16_t> packed(values.size());
        ggml_fp32_to_fp16_row(values.data(), packed.data(), values.size());
        ggml_backend_tensor_set(mask, packed.data(), 0, ggml_nbytes(mask));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "SWA fused backward compute failed");
        std::vector<float> expected(ggml_nelements(reference)), actual(expected.size());
        ggml_backend_tensor_get(reference, expected.data(), 0, ggml_nbytes(reference));
        ggml_backend_tensor_get(fused, actual.data(), 0, ggml_nbytes(fused));
        double error = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            check(std::isfinite(actual[i]), "SWA fused backward produced non-finite gradient");
            error += double(actual[i] - expected[i])*(actual[i] - expected[i]);
            norm += double(expected[i])*expected[i];
        }
        const double relative = std::sqrt(error/std::max(norm, 1e-30));
        std::printf("SWA fused backward: q_only=%d masked=%d sink=%d relative_l2=%.9g\n", q_only, masked, with_sink, relative);
        check(relative < 1e-4, "SWA fused backward differs from reference");
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--swa-fused-cuda") == 0) {
            ggml_backend_load_all();
            auto * device = ggml_backend_dev_by_name("CUDA0");
            check(device != nullptr, "SWA fused backward regression requires CUDA0");
            ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
            check(backend != nullptr, "cannot initialize SWA fused backend");
            swa_fused_backward_checks(backend.get());
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--host-gradient") == 0) {
            ggml_backend_load_all();
            auto * device = ggml_backend_dev_by_name("CUDA0");
            check(device != nullptr, "host gradient regression requires CUDA");
            host_gradient_dependency(device);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--async-cuda") == 0) {
            ggml_backend_load_all();
            auto * first = ggml_backend_dev_by_name("CUDA0");
            auto * second = ggml_backend_dev_by_name("CUDA1");
            check(first && second, "async regression requires two CUDA devices");
            tensor_parallel_async(first, second);
            tensor_parallel_host_copy(first, second);
            host_gradient_dependency(first);
            return 0;
        }
        if (argc == 2 && (std::strcmp(argv[1], "--full-context") == 0 || std::strcmp(argv[1], "--full-context-cuda") == 0)) {
            ggml_backend_ptr backend;
            if (std::strcmp(argv[1], "--full-context-cuda") == 0) {
                ggml_backend_load_all();
                auto * device = ggml_backend_dev_by_name("CUDA0");
                check(device != nullptr, "full-context regression requires CUDA0");
                backend.reset(ggml_backend_dev_init(device, nullptr));
            } else {
                backend.reset(ggml_backend_cpu_init());
            }
            check(backend != nullptr, "cannot initialize full-context backend");
            full_context_checks(backend.get());
            return 0;
        }
        llama_hparams hparams;
        hparams.n_layer_all = 30;
        hparams.n_layer_kv_from_start = 11;
        hparams.kv_global_only = true;
        for (uint32_t i = 0; i < 30; ++i) hparams.is_swa_impl[i] = i % 6 != 5;
        for (uint32_t i = 0; i < 30; ++i) {
            const int32_t expected = i % 6 != 5 ? int32_t(i) : 5;
            check(hparams.kv_source(i) == expected, "Lumen global KV source differs from recovery architecture");
            check(hparams.has_kv(i) == (i % 6 != 5 || i == 5), "Lumen must have one global KV owner");
        }
        if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
            benchmark(true);
            benchmark(false);
            return 0;
        }
        for (bool reference_mode : {false, true}) for (float t : {1.f, 2.f}) {
            loss_and_gradient(t, false, false, reference_mode);
            loss_and_gradient(t, true, false, reference_mode);
            loss_and_gradient(t, false, true, reference_mode);
        }
        compact_encoding_checks();
        std::puts("compact encoding: CPU IDs, FP16 logp, tails, ties and non-finite checks passed");
        encoder_checks();
        cache_checks();
        mixed_cache_checks();
        no_cache_mask_checks();
        tiled_distillation_checks();
        frozen_q41_moe_gradient();
        auto * cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        auto * paired_cpu = ggml_backend_dev_init(cpu, nullptr);
        fake_quant_checks(paired_cpu);
        full_context_checks(paired_cpu);
        paired_forward_checks(paired_cpu);
        ggml_backend_free(paired_cpu);
        tensor_parallel_backward(cpu, cpu);
        tensor_parallel_sum(cpu, cpu);
        tensor_parallel_accumulate(cpu, cpu);
        tensor_parallel_copy(cpu, cpu);
        tensor_parallel_partial_write(cpu, cpu);
        if (argc == 2 && std::strcmp(argv[1], "--cuda") == 0) {
            ggml_backend_load_all();
            auto * first = ggml_backend_dev_by_name("CUDA0");
            auto * second = ggml_backend_dev_by_name("CUDA1");
            check(first && second, "copy regression requires two CUDA devices");
            for (auto * device : {first, second}) {
                auto * backend = ggml_backend_dev_init(device, nullptr);
                check(backend != nullptr, "cannot initialize CUDA KL backend");
                fake_quant_checks(backend);
                full_context_checks(backend);
                compact_encoding_checks(backend);
                paired_forward_checks(backend);
                for (int nv : {8, 4097, 262144, 262145}) for (float t : {1.f, 2.f}) {
                    loss_and_gradient(t, false, false, false, backend, nv);
                    loss_and_gradient(t, true, true, false, backend, nv);
                }
                ggml_backend_free(backend);
            }
            // The internal collective can round inputs to BF16; this reference checks F32 peer reduction.
            const char * allreduce = std::getenv("GGML_CUDA_ALLREDUCE");
            if (allreduce && std::strcmp(allreduce, "none") == 0) tensor_parallel_backward(first, second);
            tensor_parallel_async(first, second);
            tensor_parallel_host_copy(first, second);
            host_gradient_dependency(first);
            tensor_parallel_accumulate(first, second);
            tensor_parallel_sum(first, second);
            tensor_parallel_copy(first, second);
            tensor_parallel_partial_write(first, second);
            ggml_backend_dev_t devices[] = {first, second};
            int axes[] = {0, 0};
            auto * paired_meta = ggml_backend_dev_init(ggml_backend_meta_device(devices, 2, test_split_state, axes), nullptr);
            paired_forward_checks(paired_meta);
            ggml_backend_free(paired_meta);
        }
        for (bool q_only : {false, true}) {
            const auto serial = attention_backward(false, q_only);
            const auto parallel = attention_backward(true, q_only);
            for (size_t i = 0; i < serial.size(); ++i) {
                check(std::abs(serial[i] - parallel[i]) < 2e-5, "tensor parallel attention gradient differs from reference");
            }
        }
        std::puts("distillation: KL reference, finite-difference gradients, masks, temperatures and cache checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
