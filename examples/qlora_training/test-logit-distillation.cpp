#include "logit-distillation.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <sstream>
#include <filesystem>
#include <chrono>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static double reference(const std::vector<float> & logits, const std::vector<int32_t> & ids,
        const std::vector<float> & logp, const std::vector<float> & weights, float temperature) {
    constexpr int nv = 8, nr = 2, k = 3;
    double loss = 0;
    for (int r = 0; r < nr; ++r) {
        double total = 0;
        const double maximum = *std::max_element(logits.begin() + r*nv, logits.begin() + (r+1)*nv);
        for (int j = 0; j < nv; ++j) total += std::exp((logits[r*nv+j] - maximum) / temperature);
        bool selected[nv] = {};
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

static void loss_and_gradient(float temperature, bool masked, bool extreme = false, bool reference_mode = false) {
    ggml_init_params params = {16*1024*1024, nullptr, false};
    auto * ctx = ggml_init(params);
    std::vector<float> values = {1.3f, -.2f, .5f, -1.f, .8f, -.8f, .1f, -.4f,
        -.1f, .4f, -.3f, 1.1f, .2f, -.5f, .7f, -.9f};
    if (extreme) values[0] = 1000.f;
    const std::vector<int32_t> indices = {4, 0, 2, 6, 3, 1};
    const std::vector<float> weights = {1.5f, masked ? 0.f : .5f};
    std::vector<float> targets = {.3f, .25f, .2f, .25f, .4f, .3f, .1f, .2f};
    for (auto & p : targets) p = std::log(p);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 2);
    auto * logp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto * row_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    std::memcpy(logits->data, values.data(), values.size()*4);
    std::memcpy(ids->data, indices.data(), indices.size()*4);
    std::memcpy(logp->data, targets.data(), targets.size()*4);
    std::memcpy(row_weights->data, weights.data(), weights.size()*4);
    ggml_set_param(logits);
    auto loss_function = reference_mode ? ggml_opt_distillation_loss_reference : ggml_opt_distillation_loss;
    auto * loss = loss_function(ctx, logits, ids, logp, row_weights, temperature);
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_graph_reset(graph);
    check(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS, "loss graph failed");
    const double expected = reference(values, indices, targets, weights, temperature);
    check(std::abs(*static_cast<float *>(loss->data) - expected) < 2e-5 + 1e-6*std::abs(expected), "KL differs from reference");
    const auto * gradient = static_cast<const float *>(ggml_graph_get_grad(graph, logits)->data);
    for (size_t i = 0; i < values.size(); ++i) {
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
    ggml_free(ctx);
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
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(offset + 4);
        file.write(reinterpret_cast<const char *>(ids), 4);
        file.close();
    }
    bool duplicate_rejected = false;
    try {
        logit_distillation::cache loaded(path.string());
        int32_t ids[3];
        float probabilities[4];
        loaded.row(loaded.in.tellg(), ids, probabilities);
    } catch (const std::exception &) { duplicate_rejected = true; }
    std::filesystem::remove(path);
    check(duplicate_rejected, "duplicate cache IDs accepted");
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

int main(int argc, char ** argv) {
    try {
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
        cache_checks();
        frozen_q41_moe_gradient();
        std::puts("distillation: KL reference, finite-difference gradients, masks, temperatures and cache checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
