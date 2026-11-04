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

static void loss_and_gradient(float temperature, bool masked, bool extreme = false) {
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
    auto * loss = ggml_opt_distillation_loss(ctx, logits, ids, logp, row_weights, temperature);
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
    const auto path = std::filesystem::temp_directory_path() /
            ("qlion-distillation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".dat");
    {
        std::ofstream file(path, std::ios::binary);
        file.write(content.data(), content.size());
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

int main() {
    try {
        for (float t : {1.f, 2.f}) {
            loss_and_gradient(t, false);
            loss_and_gradient(t, true);
            loss_and_gradient(t, false, true);
        }
        cache_checks();
        std::puts("distillation: KL reference, finite-difference gradients, masks, temperatures and cache checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
