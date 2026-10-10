#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include "optimize.h"
#include "global-loss.h"
#include "candidate-store.h"
#include "native-global.h"
#include "hash/hash.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

using json = nlohmann::ordered_json;
using namespace llama_opt;

static void check(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void read_bytes(std::istream & f, void * p, size_t n) {
    f.read(static_cast<char *>(p), n);
    check(bool(f), "short input read");
}

static void write_bytes(std::ostream & f, const void * p, size_t n) {
    f.write(static_cast<const char *>(p), n);
    check(bool(f), "output write failed");
}

static double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void save_json(const std::string & path, const json & j) {
    check(!std::filesystem::exists(path), "refusing to overwrite report: " + path);
    std::ofstream f(path);
    f << j.dump(2) << '\n';
    check(bool(f), "cannot save report");
}

static double dot(const std::vector<double> & a, const std::vector<double> & b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

struct adam_graph {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    ggml_tensor *         p = nullptr, *g = nullptr, *m = nullptr, *v = nullptr, *params = nullptr;
    ggml_cgraph *         graph = nullptr;

    explicit adam_graph(size_t count) {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, 1);
        ctx = ggml_init({ 1024 * 1024, nullptr, true });
        check(ctx != nullptr, "Adam graph allocation failed");
        p = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
        ggml_set_param(p);
        g      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
        m      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
        v      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
        params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        graph  = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, ggml_opt_step_adamw(ctx, p, g, m, v, params));
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr, "Adam buffer allocation failed");
    }

    ~adam_graph() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    void step(std::vector<double> &       alpha,
              const std::vector<double> & gradient,
              std::vector<double> &       first,
              std::vector<double> &       second,
              int                         iteration,
              float                       lr) {
        auto put = [](ggml_tensor * t, const std::vector<double> & x) {
            check(x.size() == size_t(ggml_nelements(t)), "Adam shape mismatch");
            std::vector<float> f(x.begin(), x.end());
            for (float z : f) {
                check(std::isfinite(z), "nonfinite Adam input");
            }
            ggml_backend_tensor_set(t, f.data(), 0, f.size() * 4);
        };
        put(p, alpha);
        put(g, gradient);
        put(m, first);
        put(v, second);
        const float configuration[8] = { lr,
                                         .9f,
                                         .999f,
                                         1e-8f,
                                         0,
                                         float(1 / (1 - std::pow(.9f, iteration))),
                                         float(1 / (1 - std::pow(.999f, iteration))),
                                         0 };
        ggml_backend_tensor_set(params, configuration, 0, sizeof(configuration));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Adam update failed");
        auto get = [](ggml_tensor * t, std::vector<double> & x) {
            std::vector<float> f(x.size());
            ggml_backend_tensor_get(t, f.data(), 0, f.size() * 4);
            x.assign(f.begin(), f.end());
        };
        get(p, alpha);
        get(m, first);
        get(v, second);
    }
};

static json global_loss_check() {
    const std::vector<double> alpha = { -.4, .3, .2, .1, -.7, .4 }, noise = { .2, -.1, .3, -.2, .4, .1 };
    const double tau = .7;
    const std::vector<float> inputs = { .8f, -.5f, -.3f, .9f, .4f, .6f };
    const std::vector<std::vector<float>> candidates = {
        { .5f, -.4f, .2f, .7f, .6f, -.2f, -.1f, .4f, .3f, -.6f, .4f, .8f },
        { .7f, -.3f, .2f, .5f, .4f, -.5f, .6f, .2f, .9f, -.1f, -.2f, .8f }
    };
    const std::vector<int32_t> targets = { 1, 0, 1 };
    const std::vector<float> teacher = { float(std::log(.3)), float(std::log(.7)), float(std::log(.6)),
                                          float(std::log(.4)), float(std::log(.2)), float(std::log(.8)) };
    auto noisy = alpha;
    for (size_t j = 0; j < noisy.size(); ++j) {
        noisy[j] = (alpha[j] + noise[j]) / tau;
    }
    const auto p = probabilities(noisy, 3);
    json results = json::array();
    for (bool hard : { false, true }) {
        for (bool kl : { false, true }) {
            auto * ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
            check(ctx != nullptr, "global test context allocation failed");
            auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
            auto * w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
            auto * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
            ggml_set_param(w1);
            ggml_set_param(w2);
            auto * z = ggml_mul_mat(ctx, w2, ggml_tanh(ctx, ggml_mul_mat(ctx, w1, x)));
            auto * seed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
            auto * objective = ggml_sum(ctx, ggml_mul(ctx, z, seed));
            ggml_set_loss(objective);
            auto * forward = ggml_new_graph_custom(ctx, 1024, true);
            ggml_build_forward_expand(forward, objective);
            auto * backward = ggml_graph_dup(ctx, forward, true);
            ggml_build_backward_expand(ctx, backward, nullptr);
            auto backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(backend, 1);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            check(buffer != nullptr, "global test buffer allocation failed");
            std::vector<size_t> assignment = { 1, 2 };
            for (size_t group = 0; group < 2; ++group) {
                std::vector<float> mixed(4, 0);
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t j = 0; j < 4; ++j) {
                        mixed[j] += float(hard ? k == assignment[group] : p[group * 3 + k]) * candidates[group][k * 4 + j];
                    }
                }
                ggml_backend_tensor_set(group == 0 ? w1 : w2, mixed.data(), 0, 16);
            }
            ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
            std::vector<float> zeros(6, 0), logits(6);
            ggml_backend_tensor_set(seed, zeros.data(), 0, 24);
            check(ggml_backend_graph_compute(backend, forward) == GGML_STATUS_SUCCESS, "global test forward failed");
            ggml_backend_tensor_get(z, logits.data(), 0, 24);
            const auto loss = token_loss(logits, targets, teacher, 2, kl);
            ggml_backend_tensor_set(seed, loss.logits_gradient.data(), 0, 24);
            ggml_graph_reset(backward);
            check(ggml_backend_graph_compute(backend, backward) == GGML_STATUS_SUCCESS, "global test backward failed");
            std::vector<double> gradient;
            for (size_t group = 0; group < 2; ++group) {
                std::vector<float> dw(4);
                ggml_backend_tensor_get(ggml_graph_get_grad(backward, group == 0 ? w1 : w2), dw.data(), 0, 16);
                std::vector<double> dp(3, 0), soft(p.begin() + group * 3, p.begin() + group * 3 + 3);
                for (size_t k = 0; k < 3; ++k) {
                    for (size_t j = 0; j < 4; ++j) {
                        dp[k] += double(dw[j]) * candidates[group][k * 4 + j];
                    }
                }
                const auto d = precision_vjp(soft, dp, tau);
                gradient.insert(gradient.end(), d.begin(), d.end());
            }
            results.push_back({ { "alpha", alpha }, { "noise", noise }, { "temperature", tau },
                               { "inputs", inputs }, { "candidates", candidates }, { "targets", targets },
                               { "teacher_log_probs", teacher }, { "hard", hard }, { "objective", kl ? "kl" : "ce" },
                               { "assignment", assignment }, { "loss", loss.value }, { "gradient", gradient } });
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            ggml_free(ctx);
        }
    }
    return results;
}

static void shared_permute_gradient_check(bool shared) {
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    ggml_set_param(x);
    auto * y = ggml_rms_norm(ctx, ggml_scale(ctx, x, 2), 1.f);
    auto * a = ggml_sum(ctx, ggml_transpose(ctx, y));
    auto * b = ggml_sum(ctx, ggml_transpose(ctx, y));
    auto * loss = shared ? ggml_add(ctx, a, b) : a;
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    auto backend = ggml_backend_cpu_init();
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_clear(buffer, 0);
    ggml_graph_reset(graph);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "shared permute backward failed");
    std::vector<float> dx(6);
    ggml_backend_tensor_get(ggml_graph_get_grad(graph, x), dx.data(), 0, dx.size() * sizeof(float));
    check(std::all_of(dx.begin(), dx.end(), [shared](float d) { return d == (shared ? 4 : 2); }), "shared permute gradient mismatch");
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void self_test() {
    shared_permute_gradient_check(true);
    shared_permute_gradient_check(false);
    {
        auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 65536, 1);
        auto * inputs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 65536, 1);
        auto * output = ggml_mul_mat(ctx, weights, inputs);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_use_ref(backend, true);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> w(65536, 1), x(65536, 1);
        w.front() = 1e8f; w.back() = -1e8f;
        ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(inputs, x.data(), 0, x.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "long reference matmul failed");
        float actual;
        ggml_backend_tensor_get(output, &actual, 0, sizeof(actual));
        check(actual == 65534, "reference matmul loses small terms during cancellation");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    {
        auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 65536);
        auto * adjoints = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 65536);
        auto * dx = ggml_out_prod(ctx, weights, adjoints);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, dx);
        auto backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_use_ref(backend, true);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> w(2 * 65536, .01f), g(65536, 1e-8f);
        g[0] = -1;
        ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
        ggml_backend_tensor_set(adjoints, g.data(), 0, g.size() * sizeof(float));
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "long vocabulary adjoint failed");
        float actual[2];
        ggml_backend_tensor_get(dx, actual, 0, sizeof(actual));
        const float expected = float(double(w[0]) * (-1 + 65535 * double(g[1])));
        check(std::abs(actual[0] - expected) < 1e-8f && std::abs(actual[1] - expected) < 1e-8f,
              "long vocabulary reference adjoint loses small probabilities");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    std::vector<float> w(128), dy(128);
    for (size_t i = 0; i < w.size(); ++i) {
        w[i]  = float(std::sin(i * .13));
        dy[i] = float(std::cos(i * .21));
    }
    auto               s = initialize(w, 64, 2);
    auto               v = sample(s, 123, 0.7f, 2.3f);
    std::vector<float> dl, ds;
    backward(s, v, dy, .7f, 2.3f, dl, ds);
    json j = {
        { "logits",      s.logits  },
        { "scales",      s.scales  },
        { "noise",       v.noise   },
        { "dy",          dy        },
        { "forward",     v.weights },
        { "grad_logits", dl        },
        { "grad_scales", ds        }
    };
    j["replay_equal"]    = v.noise == sample(s, 123, .7f, 2.3f).noise;
    std::vector<float> p = { 1, -2, .4f }, m = { .2f, -.3f, .01f }, g = { -.5f, .7f, .2f };
    j["lion_before"]   = p;
    j["moment_before"] = m;
    j["lion_grad"]     = g;
    lion(p, m, g, .01f, .9f, .95f, .2f);
    j["lion_after"]        = p;
    j["moment_after"]      = m;
    std::vector<double> ap = { .2, -.4, .1 }, ag = { .7, -.3, .9 }, am(3, 0), av(3, 0);
    j["adam_before"]   = ap;
    j["adam_gradient"] = ag;
    adam_graph optimizer(3);
    optimizer.step(ap, ag, am, av, 1, .01f);
    j["adam_after"]          = ap;
    j["adam_first"]          = am;
    j["adam_second"]         = av;
    auto               bytes = pack_q2(s);
    std::vector<float> decoded(w.size());
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(bytes.data(), decoded.data(), decoded.size());
    double error = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        const float * l = s.logits.data() + 4 * i;
        const int     k = int(std::max_element(l, l + 4) - l);
        error           = std::max(error, std::abs(double(decoded[i] - (k - 2) * s.scales[i / 64])));
    }
    j["packing_error"] = error;
    auto mapping       = s;
    mapping.scales     = { .375f, -.125f };
    for (size_t i = 0; i < 128; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            mapping.logits[i * 4 + k] = k == i % 4 ? 1.f : -1.f;
        }
    }
    const auto mapping_bytes = pack_q2(mapping);
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(mapping_bytes.data(), decoded.data(), decoded.size());
    for (size_t i = 0; i < 128; ++i) {
        check(decoded[i] == float(int(i % 4) - 2) * mapping.scales[i / 64], "signed Q2_0 code mapping failed");
    }
    j["signed_codebook_mapping"] = true;
    bool alignment_rejected      = false;
    try {
        initialize(std::vector<float>(65, 0), 65, 1);
    } catch (const std::runtime_error &) {
        alignment_rejected = true;
    }
    check(alignment_rejected, "incomplete Q2_0 block accepted");
    std::vector<std::vector<uint64_t>> costs = {
        { 18, 36,  68  },
        { 36, 72,  136 },
        { 54, 108, 204 }
    };
    std::vector<std::vector<double>> scores = {
        { -4, -2, -1 },
        { -7, -3, -1 },
        { -9, -4, -1 }
    };
    std::vector<double> a = { -.4, .3, .2, .1, -.7, .4, .8, .1, -.2 }, vg = { .4, .1, -.3, .7, -.2, .1, .2, .5, .3 };
    j["rco_alpha_before"]    = a;
    j["rco_gradient_before"] = vg;
    auto n                   = normal(a, costs);
    project(vg, n);
    j["projection_dot"] = dot(vg, n);
    j["rco_projected"]  = vg;
    j["target_budget"]  = 216.;
    j["budget_after"]   = retract(a, costs, 216);
    j["rco_retracted"]  = a;
    n                   = normal(a, costs);
    project(vg, n);
    j["transport_dot"]   = dot(vg, n);
    j["rco_transported"] = vg;
    std::vector<double> ra(9, 0), rm(9, 0), rvv(9, 0);
    retract(ra, costs, 216);
    j["synthetic_alpha_before"] = ra;
    adam_graph riemannian(9);
    double     residual = 0;
    for (int iteration = 1; iteration <= 20; ++iteration) {
        const auto          probs = probabilities(ra, 3);
        std::vector<double> grad(9, 0);
        double              loss = 0;
        for (size_t i = 0; i < 3; ++i) {
            double mean = 0;
            for (size_t k = 0; k < 3; ++k) {
                const double difference = probs[i * 3 + k] - (k == 1 ? 1. : 0.);
                loss += difference * difference;
                mean += probs[i * 3 + k] * 2 * difference;
            }
            for (size_t k = 0; k < 3; ++k) {
                grad[i * 3 + k] = probs[i * 3 + k] * (2 * (probs[i * 3 + k] - (k == 1 ? 1. : 0.)) - mean);
            }
        }
        if (iteration == 1) {
            j["synthetic_initial_loss"] = loss;
        }
        project(grad, normal(ra, costs));
        riemannian.step(ra, grad, rm, rvv, iteration, .05f);
        residual = std::max(residual, std::abs(retract(ra, costs, 216) - 216));
        project(rm, normal(ra, costs));
    }
    j["synthetic_alpha_after"]     = ra;
    j["synthetic_budget_residual"] = residual;
    const auto choice              = assign(costs, scores, 216);
    uint64_t   used                = 0;
    double     score               = 0;
    for (size_t i = 0; i < choice.size(); ++i) {
        used += costs[i][choice[i]];
        score += scores[i][choice[i]];
    }
    j["costs"]            = costs;
    j["scores"]           = scores;
    j["byte_budget"]      = 216;
    j["assignment_score"] = score;
    j["assigned_bytes"]   = used;
    bool rejected         = false;
    try {
        assign(costs, scores, 1);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "infeasible budget accepted");
    j["global_loss"] = global_loss_check();
    std::cout << j.dump() << '\n';
}

struct capture_data {
    uint64_t           columns = 0, rows = 0, samples = 0;
    std::vector<float> weights, inputs;
    std::string        tensor;
};

static bool collect(ggml_tensor * t, bool ask, void * ud) {
    auto & d = *static_cast<capture_data *>(ud);
    if (t->op != GGML_OP_MUL_MAT || !t->src[0] || d.tensor != t->src[0]->name) {
        return false;
    }
    if (ask) {
        return true;
    }
    const auto * w = t->src[0];
    const auto * x = t->src[1];
    check(w->ne[2] == 1 && w->ne[3] == 1 && x->type == GGML_TYPE_F32, "capture supports dense F32 inputs only");
    if (d.weights.empty()) {
        d.columns = w->ne[0];
        d.rows    = w->ne[1];
        std::vector<uint8_t> raw(ggml_nbytes(w));
        ggml_backend_tensor_get(w, raw.data(), 0, raw.size());
        d.weights.resize(d.columns * d.rows);
        if (w->type == GGML_TYPE_F32) {
            std::memcpy(d.weights.data(), raw.data(), raw.size());
        } else {
            const auto * codec = ggml_get_type_traits(w->type);
            check(codec->to_float != nullptr, "missing weight decoder");
            codec->to_float(raw.data(), d.weights.data(), d.weights.size());
        }
    }
    std::vector<uint8_t> raw(ggml_nbytes(x));
    ggml_backend_tensor_get(x, raw.data(), 0, raw.size());
    for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < x->ne[1]; ++i1) {
                const auto * row =
                    reinterpret_cast<const float *>(raw.data() + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);
                d.inputs.insert(d.inputs.end(), row, row + d.columns);
                ++d.samples;
            }
        }
    }
    return true;
}

static capture_data load_capture(const std::string & path) {
    capture_data  d;
    std::ifstream f(path, std::ios::binary);
    uint32_t      magic = 0;
    read_bytes(f, &magic, 4);
    check(magic == 0x31515347, "invalid capture format");
    read_bytes(f, &d.columns, 8);
    read_bytes(f, &d.rows, 8);
    read_bytes(f, &d.samples, 8);
    check(d.columns > 0 && d.columns <= 65536 && d.rows > 0 && d.rows <= 65536 && d.samples > 0 && d.samples <= 4096,
          "capture shape limit exceeded");
    d.weights.resize(d.columns * d.rows);
    d.inputs.resize(d.columns * d.samples);
    read_bytes(f, d.weights.data(), d.weights.size() * 4);
    read_bytes(f, d.inputs.data(), d.inputs.size() * 4);
    return d;
}

struct model_session {
    llama_model *      model           = nullptr;
    llama_context *    ctx             = nullptr;
    ggml_backend_dev_t devices[2]      = { nullptr, nullptr };
    double             prefill_seconds = 0;

    ~model_session() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }
};

static std::vector<llama_token> run_model(model_session &     m,
                                          const std::string & path,
                                          const std::string & text,
                                          capture_data *      capture,
                                          int                 gpu,
                                          const std::vector<llama_token> * supplied = nullptr) {
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = gpu;
    if (gpu != 0) {
        m.devices[0] = ggml_backend_dev_by_name("CUDA0");
        check(m.devices[0] != nullptr, "CUDA0 is required for GPU inference; use GPU_LAYERS=0 for CPU");
        mp.devices = m.devices;
    }
    m.model = llama_model_load_from_file(path.c_str(), mp);
    check(m.model != nullptr, "model load failed");
    const auto *             vocab = llama_model_get_vocab(m.model);
    std::vector<llama_token> tokens(4096);
    int n;
    if (supplied) {
        check(supplied->size() > 1 && supplied->size() <= 512, "supplied prompt must be 2..512 tokens");
        tokens = *supplied;
        n = tokens.size();
    } else {
        n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
    }
    check(n > 1 && n <= 512, "PoC requires 2..512 tokens");
    tokens.resize(n);
    auto cp            = llama_context_default_params();
    cp.n_ctx           = std::max(128, n + 32);
    cp.n_batch         = n;
    cp.n_ubatch        = n;
    cp.n_threads       = 8;
    cp.n_threads_batch = 8;
    cp.no_perf         = false;
    cp.op_offload      = gpu != 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (capture) {
        cp.cb_eval           = collect;
        cp.cb_eval_user_data = capture;
    }
    m.ctx = llama_init_from_model(m.model, cp);
    check(m.ctx != nullptr, "context creation failed");
    auto b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        b.token[i]     = tokens[i];
        b.pos[i]       = i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = true;
    }
    b.n_tokens                 = n;
    const double prefill_start = seconds();
    const int    status        = llama_decode(m.ctx, b);
    llama_batch_free(b);
    check(status == 0, "decode failed");
    llama_synchronize(m.ctx);
    for (int i = 0; i < n; ++i) {
        check(llama_get_logits_ith(m.ctx, i) != nullptr, "prefill logits unavailable");
    }
    m.prefill_seconds = seconds() - prefill_start;
    return tokens;
}

static std::string text_file(const std::string & path) {
    std::ifstream f(path);
    check(bool(f), "cannot open text");
    return std::string(std::istreambuf_iterator<char>(f), {});
}

static void capture_model(int argc, char ** argv) {
    check(argc == 7, "capture MODEL TEXT TENSOR OUTPUT GPU_LAYERS");
    capture_data d;
    d.tensor        = argv[4];
    auto * metadata = gguf_init_from_file(argv[2], { true, nullptr });
    check(metadata != nullptr, "capture source metadata unavailable");
    const auto id = gguf_find_tensor(metadata, d.tensor.c_str());
    check(id >= 0, "capture tensor not found in GGUF");
    const auto * shape = gguf_get_tensor_ne(metadata, id);
    const bool   dense = shape[2] == 1 && shape[3] == 1;
    gguf_free(metadata);
    check(dense, "capture supports dense 2D tensors only; expert tensors are unsupported");
    model_session m;
    const double  start  = seconds();
    auto          tokens = run_model(m, argv[2], text_file(argv[3]), &d, std::stoi(argv[6]));
    check(!d.weights.empty() && d.samples > 0, "requested tensor was not captured");
    check(!std::filesystem::exists(argv[5]), "capture exists");
    std::ofstream f(argv[5], std::ios::binary);
    uint32_t      magic = 0x31515347;
    write_bytes(f, &magic, 4);
    write_bytes(f, &d.columns, 8);
    write_bytes(f, &d.rows, 8);
    write_bytes(f, &d.samples, 8);
    write_bytes(f, d.weights.data(), d.weights.size() * 4);
    write_bytes(f, d.inputs.data(), d.inputs.size() * 4);
    save_json(std::string(argv[5]) + ".json", {
                                                  { "tensor",  d.tensor          },
                                                  { "columns", d.columns         },
                                                  { "rows",    d.rows            },
                                                  { "samples", d.samples         },
                                                  { "tokens",  tokens            },
                                                  { "seconds", seconds() - start },
                                                  { "source",  argv[2]           }
    });
}

struct capture_store_state {
    std::map<std::string, size_t> groups;
    std::vector<json> captures;
    std::string directory;
};

static bool collect_store(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<capture_store_state *>(user_data);
    if (tensor->op != GGML_OP_MUL_MAT || !tensor->src[0]) { return false; }
    const auto found = state.groups.find(tensor->src[0]->name);
    if (found == state.groups.end()) { return false; }
    if (ask) { return true; }
    check(state.captures[found->second].is_null(), "capture tensor executes more than once; use one complete window");
    capture_data data;
    data.tensor = found->first;
    collect(tensor, false, &data);
    const auto path = state.directory + "/" + std::to_string(found->second) + ".gsq";
    check(!std::filesystem::exists(path), "capture payload exists");
    std::ofstream file(path, std::ios::binary);
    const uint32_t magic = 0x31515347;
    write_bytes(file, &magic, 4);
    write_bytes(file, &data.columns, 8);
    write_bytes(file, &data.rows, 8);
    write_bytes(file, &data.samples, 8);
    write_bytes(file, data.weights.data(), data.weights.size() * sizeof(float));
    write_bytes(file, data.inputs.data(), data.inputs.size() * sizeof(float));
    file.close();
    check(bool(file), "capture payload write failed");
    state.captures[found->second] = { { "tensor", data.tensor }, { "file", path },
        { "columns", data.columns }, { "rows", data.rows }, { "samples", data.samples } };
    return true;
}

static void capture_store(int argc, char ** argv) {
    check(argc == 7, "capture-store MODEL MANIFEST TOKENS DIRECTORY REPORT");
    check(!std::filesystem::exists(argv[5]) && !std::filesystem::exists(argv[6]), "capture output exists");
    candidate_store store(argv[2], argv[3]);
    std::ifstream input(argv[4]);
    json ids;
    input >> ids;
    const auto tokens = ids.get<std::vector<llama_token>>();
    check(tokens.size() >= 2 && tokens.size() <= 512, "capture window must be 2..512 tokens");
    capture_store_state state;
    state.directory = argv[5];
    state.captures.resize(store.groups.size());
    for (size_t i = 0; i < store.groups.size(); ++i) { state.groups.emplace(store.groups[i].tensor, i); }
    std::filesystem::create_directories(state.directory);
    model_session model;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    model.model = llama_model_load_from_file(argv[2], mp);
    check(model.model != nullptr, "capture source load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max<size_t>(128, tokens.size());
    cp.n_batch = cp.n_ubatch = tokens.size();
    cp.n_threads = cp.n_threads_batch = 8;
    cp.op_offload = false;
    cp.cb_eval = collect_store;
    cp.cb_eval_user_data = &state;
    model.ctx = llama_init_from_model(model.model, cp);
    check(model.ctx != nullptr, "capture context creation failed");
    auto batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = tokens.size();
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
    }
    const double start = seconds();
    const int status = llama_decode(model.ctx, batch);
    llama_batch_free(batch);
    check(status == 0, "capture forward failed");
    llama_synchronize(model.ctx);
    check(std::all_of(state.captures.begin(), state.captures.end(), [](const json & item) { return !item.is_null(); }),
          "candidate tensor not captured by model graph");
    save_json(argv[6], { { "tokens", tokens }, { "source_identity", store.source_identity },
        { "captures", state.captures }, { "seconds", seconds() - start },
        { "calibration_scored_tokens", tokens.size() - 1 }, { "execution", "original BF16 CPU forward" } });
}

struct linear_graph {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;
    ggml_backend_buffer_t buffer  = nullptr;
    ggml_tensor *         w = nullptr, *x = nullptr, *dy = nullptr, *y = nullptr, *dw = nullptr;
    ggml_cgraph *         forward = nullptr, *back = nullptr;

    linear_graph(const capture_data & d, const std::string & device) {
        backend = ggml_backend_init_by_name(device.c_str(), nullptr);
        check(backend != nullptr, "backend not available: " + device);
        if (device == "CPU") {
            ggml_backend_cpu_set_n_threads(backend, 8);
        }
        ctx = ggml_init({ size_t(16 * 1024 * 1024), nullptr, true });
        check(ctx != nullptr, "graph context allocation failed");
        w       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.columns, d.rows);
        x       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.columns, d.samples);
        dy      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.rows, d.samples);
        y       = ggml_mul_mat(ctx, w, x);
        dw      = ggml_out_prod(ctx, x, dy);
        forward = ggml_new_graph(ctx);
        ggml_build_forward_expand(forward, y);
        back = ggml_new_graph(ctx);
        ggml_build_forward_expand(back, dw);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        check(buffer != nullptr, "linear buffer allocation failed");
        ggml_backend_tensor_set(x, d.inputs.data(), 0, d.inputs.size() * 4);
    }

    ~linear_graph() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    std::vector<float> evaluate(const std::vector<float> & weights) {
        ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * 4);
        check(ggml_backend_graph_compute(backend, forward) == GGML_STATUS_SUCCESS, "linear forward failed");
        std::vector<float> out(ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * 4);
        return out;
    }

    std::vector<float> gradient(const std::vector<float> & gradient) {
        ggml_backend_tensor_set(dy, gradient.data(), 0, gradient.size() * 4);
        check(ggml_backend_graph_compute(backend, back) == GGML_STATUS_SUCCESS, "linear backward failed");
        std::vector<float> out(ggml_nelements(dw));
        ggml_backend_tensor_get(dw, out.data(), 0, out.size() * 4);
        return out;
    }
};

static double mse(const std::vector<float> & a, const std::vector<float> & b) {
    check(a.size() == b.size(), "MSE shape mismatch");
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double e = double(a[i]) - b[i];
        sum += e * e;
    }
    return sum / a.size();
}

static void linear_test(int argc, char ** argv) {
    check(argc == 5, "linear-test CAPTURE BACKEND OUTPUT");
    const auto d = load_capture(argv[2]);
    check(!std::filesystem::exists(argv[4]), "linear test output exists");
    linear_graph graph(d, argv[3]);
    const auto   out = graph.evaluate(d.weights);
    auto         dy  = out;
    for (size_t i = 0; i < dy.size(); ++i) {
        dy[i] = float(std::sin(i * .017) * .1);
    }
    const auto    dw = graph.gradient(dy);
    std::ofstream f(argv[4], std::ios::binary);
    write_bytes(f, out.data(), out.size() * 4);
    write_bytes(f, dw.data(), dw.size() * 4);
}

struct replacement {
    ggml_type            type;
    std::vector<uint8_t> bytes;
    std::string file = {};
};

static void materialize(const std::string &                        source,
                        const std::string &                        output,
                        const std::map<std::string, replacement> & changes,
                        uint64_t expected_bytes = UINT64_MAX,
                        uint64_t byte_budget = UINT64_MAX) {
    check(source != output && !std::filesystem::exists(output) && !std::filesystem::exists(output + ".partial"),
          "output already exists or equals source");
    ggml_context * tensors = nullptr;
    auto *         src     = gguf_init_from_file(source.c_str(), { true, &tensors });
    check(src != nullptr, "GGUF metadata read failed");
    auto * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    for (int64_t i = 0; i < gguf_get_n_tensors(src); ++i) {
        gguf_add_tensor(dst, ggml_get_tensor(tensors, gguf_get_tensor_name(src, i)));
    }
    for (const auto & r : changes) {
        check(gguf_find_tensor(src, r.first.c_str()) >= 0, "replacement tensor missing");
        gguf_set_tensor_type(dst, r.first.c_str(), r.second.type);
        const uint64_t size = r.second.file.empty() ? r.second.bytes.size() : std::filesystem::file_size(r.second.file);
        check(gguf_get_tensor_size(dst, gguf_find_tensor(dst, r.first.c_str())) == size,
              "replacement byte size mismatch");
    }
    check(gguf_write_to_file(dst, (output + ".partial").c_str(), true), "GGUF header write failed");
    std::ifstream     input(source, std::ios::binary);
    std::fstream      out(output + ".partial", std::ios::binary | std::ios::in | std::ios::out);
    std::vector<char> chunk(1024 * 1024), zero(32, 0);
    check(bool(input) && bool(out), "GGUF data streams unavailable");
    const size_t data_offset = gguf_get_meta_size(dst);
    for (int64_t i = 0; i < gguf_get_n_tensors(src); ++i) {
        const std::string name   = gguf_get_tensor_name(src, i);
        const size_t      offset = data_offset + gguf_get_tensor_offset(dst, i);
        out.seekp(offset);
        const auto it = changes.find(name);
        if (it != changes.end() && it->second.file.empty()) {
            write_bytes(out, it->second.bytes.data(), it->second.bytes.size());
        } else {
            std::ifstream candidate;
            std::istream * payload = &input;
            size_t remaining;
            if (it != changes.end()) {
                candidate.open(it->second.file, std::ios::binary);
                check(bool(candidate), "packed candidate unavailable");
                payload = &candidate;
                remaining = std::filesystem::file_size(it->second.file);
            } else {
                input.seekg(gguf_get_data_offset(src) + gguf_get_tensor_offset(src, i));
                remaining = gguf_get_tensor_size(src, i);
            }
            while (remaining) {
                const size_t n = std::min(remaining, chunk.size());
                read_bytes(*payload, chunk.data(), n);
                write_bytes(out, chunk.data(), n);
                remaining -= n;
            }
        }
        const size_t padding = (gguf_get_alignment(dst) - gguf_get_tensor_size(dst, i) % gguf_get_alignment(dst)) %
                               gguf_get_alignment(dst);
        if (padding) {
            zero.resize(padding, 0);
            write_bytes(out, zero.data(), padding);
        }
    }
    out.flush();
    check(bool(out), "GGUF flush failed");
    out.close();
    gguf_free(dst);
    gguf_free(src);
    ggml_free(tensors);
    auto * verify = gguf_init_from_file((output + ".partial").c_str(), { true, nullptr });
    check(verify != nullptr, "GGUF output metadata verification failed");
    gguf_free(verify);
    const auto actual = std::filesystem::file_size(output + ".partial");
    check((expected_bytes == UINT64_MAX || actual == expected_bytes) && actual <= byte_budget,
          "GGUF output violates expected size or byte budget");
    std::filesystem::rename(output + ".partial", output);
}

static void candidate_publish(int argc, char ** argv) {
    check(argc == 8, "candidate-publish MODEL MANIFEST ASSIGNMENT OUTPUT BUDGET REPORT");
    candidate_store store(argv[2], argv[3]);
    std::ifstream file(argv[4]);
    json value;
    file >> value;
    const auto selected = value.get<std::vector<size_t>>();
    check(selected.size() == store.groups.size(), "assignment shape mismatch");
    std::map<std::string, replacement> changes;
    uint64_t expected = store.fixed;
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto & group = store.groups[i];
        const auto & choice = group.choices.at(selected[i]);
        check(expected <= UINT64_MAX - choice.aligned_bytes, "assignment byte overflow");
        expected += choice.aligned_bytes;
        changes.emplace(group.tensor, replacement{ choice.type, {}, choice.file });
    }
    const uint64_t limit = std::stoull(argv[6]);
    check(expected <= limit, "assignment exceeds byte budget");
    materialize(argv[2], argv[5], changes, expected, limit);
    save_json(argv[7], { { "actual_file_bytes", expected }, { "total_byte_budget", limit },
                        { "remaining_bytes", limit - expected }, { "assignment", selected } });
}

static void candidate_verify(int argc, char ** argv) {
    check(argc == 7, "candidate-verify MODEL MANIFEST ASSIGNMENT OUTPUT REPORT");
    check(!std::filesystem::exists(argv[6]), "verification report exists");
    candidate_store store(argv[2], argv[3]);
    json assignment;
    std::ifstream input(argv[4]);
    input >> assignment;
    const auto selected = assignment.get<std::vector<size_t>>();
    check(selected.size() == store.groups.size(), "verification assignment shape mismatch");
    std::map<std::string, const packed_choice *> choices;
    uint64_t expected = store.fixed;
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto & choice = store.groups[i].choices.at(selected[i]);
        choices.emplace(store.groups[i].tensor, &choice);
        expected += choice.aligned_bytes;
    }
    auto * source = gguf_init_from_file(argv[2], { true, nullptr });
    auto * output = gguf_init_from_file(argv[5], { true, nullptr });
    check(source && output, "verification GGUF metadata unavailable");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> source_owner(source, gguf_free), output_owner(output, gguf_free);
    check(gguf_get_n_tensors(source) == gguf_get_n_tensors(output), "output tensor count differs");
    check(std::filesystem::file_size(argv[5]) == expected, "output file size differs from assignment");
    auto metadata_bytes = [](const gguf_context * context) {
        auto * normalized = gguf_init_empty();
        gguf_set_kv(normalized, context);
        std::vector<uint8_t> bytes(gguf_get_meta_size(normalized));
        gguf_get_meta_data(normalized, bytes.data());
        gguf_free(normalized);
        return bytes;
    };
    check(metadata_bytes(source) == metadata_bytes(output), "output metadata values differ");
    std::ifstream original(argv[2], std::ios::binary), actual(argv[5], std::ios::binary);
    std::vector<char> a(1024 * 1024), b(a.size());
    uint64_t compared = 0;
    json verified = json::array();
    for (int64_t i = 0; i < gguf_get_n_tensors(source); ++i) {
        const std::string name = gguf_get_tensor_name(source, i);
        const int64_t id = gguf_find_tensor(output, name.c_str());
        check(id >= 0, "output tensor missing");
        check(std::equal(gguf_get_tensor_ne(source, i), gguf_get_tensor_ne(source, i) + 4, gguf_get_tensor_ne(output, id)),
              "output tensor shape differs");
        const auto choice = choices.find(name);
        const auto type = choice == choices.end() ? gguf_get_tensor_type(source, i) : choice->second->type;
        const uint64_t size = choice == choices.end() ? gguf_get_tensor_size(source, i) : choice->second->bytes;
        check(gguf_get_tensor_type(output, id) == type && gguf_get_tensor_size(output, id) == size, "output tensor type or size differs");
        std::ifstream packed;
        std::istream * expected_data = &original;
        if (choice == choices.end()) {
            original.seekg(gguf_get_data_offset(source) + gguf_get_tensor_offset(source, i));
        } else {
            packed.open(choice->second->file, std::ios::binary);
            expected_data = &packed;
            verified.push_back({ { "tensor", name }, { "producer", choice->second->producer }, { "sha256", choice->second->sha256 } });
        }
        actual.seekg(gguf_get_data_offset(output) + gguf_get_tensor_offset(output, id));
        for (uint64_t remaining = size; remaining > 0;) {
            const size_t count = std::min<uint64_t>(remaining, a.size());
            read_bytes(*expected_data, a.data(), count);
            read_bytes(actual, b.data(), count);
            check(std::equal(a.begin(), a.begin() + count, b.begin()), "output tensor payload differs");
            remaining -= count;
            compared += count;
        }
    }
    save_json(argv[6], { { "passed", true }, { "actual_file_bytes", expected }, { "compared_tensor_bytes", compared },
        { "metadata_preserved", true }, { "fixed_tensors_preserved", true }, { "candidates", verified } });
}

static void train(int argc, char ** argv) {
    check(argc == 9 || argc == 10, "gsq MODEL CAPTURE TENSOR OUTPUT STEPS BACKEND REPORT [PRIOR]");
    const auto d     = load_capture(argv[3]);
    const int  steps = std::stoi(argv[6]);
    check(steps >= 2 && steps <= 10000, "steps must be 2..10000");
    linear_graph graph(d, argv[7]);
    auto         target = graph.evaluate(d.weights);
    auto         s      = initialize(d.weights, d.columns, d.rows);
    if (argc == 10) {
        std::ifstream prior(argv[9], std::ios::binary);
        uint32_t      magic   = 0;
        uint64_t      columns = 0, rows = 0;
        read_bytes(prior, &magic, 4);
        read_bytes(prior, &columns, 8);
        read_bytes(prior, &rows, 8);
        check(magic == 0x31525047 && columns == d.columns && rows == d.rows, "invalid external prior");
        std::vector<float> qw(d.weights.size()), qs(d.weights.size() / 64);
        read_bytes(prior, qw.data(), qw.size() * 4);
        read_bytes(prior, qs.data(), qs.size() * 4);
        s = initialize_prior(qw, qs, d.columns, d.rows);
    }
    const auto         initial_bytes = pack_q2(s);
    std::vector<float> initial(d.weights.size());
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(initial_bytes.data(), initial.data(), initial.size());
    json report = {
        { "algorithm", argc == 10 ? "GSQ external prior linear reconstruction" : "GSQ-RTN linear reconstruction" },
        { "seed", 42 },
        { "group_size", 64 },
        { "backend", argv[7] },
        { "steps", steps },
        { "initial_mse", mse(graph.evaluate(initial), target) }
    };
    if (argc == 10) {
        report["external_prior"] = argv[9];
    }
    const double       start = seconds();
    std::vector<float> dl, ds;
    double             best       = report["initial_mse"];
    gsq_state          best_state = s;
    for (int step = 0; step < steps; ++step) {
        const float  progress = float(step) / (steps - 1), tau = 2.0f + (0.05f - 2.0f) * progress,
                     kappa = 100.0f + 400.0f * progress;
        const auto   v     = sample(s, 42 + step, tau, kappa);
        auto         out   = graph.evaluate(v.weights);
        const double loss  = mse(out, target);
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = float(2.0 * (out[i] - target[i]) / out.size());
        }
        const auto dw = graph.gradient(out);
        backward(s, v, dw, tau, kappa, dl, ds);
        lion(s.logits, s.momentum, dl, 1e-4f, .9f, .95f, 1);
        lion(s.scales, s.scale_momentum, ds, 5e-5f, .9f, .95f, 0);
        if (step % 5 == 0 || step == steps - 1) {
            auto               packed = pack_q2(s);
            std::vector<float> hard(d.weights.size());
            ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(packed.data(), hard.data(), hard.size());
            const double hard_loss = mse(graph.evaluate(hard), target);
            if (hard_loss < best) {
                best       = hard_loss;
                best_state = s;
            }
            std::cerr << "step " << step << " soft_mse " << loss << " hard_mse " << hard_loss << '\n';
            report["history"].push_back({
                { "step",        step      },
                { "soft_mse",    loss      },
                { "hard_mse",    hard_loss },
                { "temperature", tau       },
                { "logit_scale", kappa     }
            });
        }
    }
    report["training_seconds"] = seconds() - start;
    const auto   packed        = pack_q2(best_state);
    const double writing       = seconds();
    if (std::string(argv[1]) == "gsq-pack") {
        check(!std::filesystem::exists(argv[5]), "packed GSQ output already exists");
        std::ofstream payload(argv[5], std::ios::binary);
        write_bytes(payload, packed.data(), packed.size());
        report["output_format"] = "packed q2_0 tensor";
    } else {
        materialize(argv[2], argv[5], { { argv[4], { GGML_TYPE_Q2_0, packed } } });
        report["output_format"] = "GGUF";
    }
    report["materialization_seconds"] = seconds() - writing;
    report["best_hard_mse"]           = best;
    report["output_bytes"]            = std::filesystem::file_size(argv[5]);
    report["optimized_parameters"]    = d.weights.size();
    report["tensor"]                  = argv[4];
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    report["peak_cpu_rss_kib"] = usage.ru_maxrss;
    save_json(argv[8], report);
}

static void ptq(int argc, char ** argv) {
    check(argc == 7, "ptq MODEL CAPTURE TENSOR OUTPUT TYPE");
    auto              d    = load_capture(argv[3]);
    const std::string type = argv[6];
    ggml_type         q    = GGML_TYPE_COUNT;
    if (type == "Q2_0") {
        q = GGML_TYPE_Q2_0;
    }
    if (type == "Q4_0") {
        q = GGML_TYPE_Q4_0;
    }
    if (type == "Q8_0") {
        q = GGML_TYPE_Q8_0;
    }
    check(q != GGML_TYPE_COUNT, "unsupported quantization type");
    check(d.columns % ggml_blck_size(q) == 0, "unaligned row");
    std::vector<uint8_t> packed(ggml_row_size(q, d.columns) * d.rows);
    ggml_get_type_traits(q)->from_float_ref(d.weights.data(), packed.data(), d.weights.size());
    materialize(argv[2], argv[5],
                {
                    { argv[4], { q, std::move(packed) } }
    });
}

static std::vector<uint8_t> tensor_bytes(const std::string & path, const std::string & name, ggml_type expected) {
    auto * meta = gguf_init_from_file(path.c_str(), { true, nullptr });
    check(meta != nullptr, "candidate GGUF read failed");
    const auto id = gguf_find_tensor(meta, name.c_str());
    check(id >= 0 && gguf_get_tensor_type(meta, id) == expected, "candidate tensor type mismatch");
    std::vector<uint8_t> bytes(gguf_get_tensor_size(meta, id));
    std::ifstream        f(path, std::ios::binary);
    f.seekg(gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id));
    read_bytes(f, bytes.data(), bytes.size());
    gguf_free(meta);
    return bytes;
}

struct candidate_group {
    std::string                     tensor;
    capture_data                    data;
    std::vector<replacement>        candidates;
    std::vector<std::vector<float>> weights;
    std::vector<float>              target;
    std::unique_ptr<linear_graph>   graph;
};

static void rco_proxy(int argc, char ** argv) {
    check(argc == 9, "rco-proxy MODEL CONFIG OUTPUT TENSOR_BYTE_BUDGET STEPS BACKEND REPORT");
    std::ifstream cf(argv[3]);
    json          config;
    cf >> config;
    const uint64_t limit = std::stoull(argv[5]);
    const int      steps = std::stoi(argv[6]);
    check(steps >= 2 && steps <= 10000, "invalid RCO steps");
    const std::filesystem::path store = config.at("candidate_directory").get<std::string>();
    check(!std::filesystem::exists(store), "candidate directory already exists");
    std::filesystem::create_directories(store);
    auto * metadata = gguf_init_from_file(argv[2], { true, nullptr });
    check(metadata != nullptr, "source GGUF unavailable");
    const uint64_t alignment = gguf_get_alignment(metadata);
    auto           aligned   = [&](uint64_t n) {
        return ((n + alignment - 1) / alignment) * alignment;
    };
    uint64_t fixed = gguf_get_meta_size(metadata);
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
        fixed += aligned(gguf_get_tensor_size(metadata, i));
    }
    std::vector<candidate_group>       groups;
    std::vector<std::vector<uint64_t>> costs;
    json                               report = {
        { "algorithm",             "RCO linear reconstruction proxy" },
        { "original_global_ce_kl", false                             },
        { "seed",                  42                                },
        { "steps",                 steps                             },
        { "backend",               argv[7]                           }
    };
    for (const auto & entry : config.at("tensors")) {
        candidate_group group;
        group.tensor = entry.at("tensor").get<std::string>();
        group.data   = load_capture(entry.at("capture").get<std::string>());
        for (const auto & previous : groups) {
            check(previous.tensor != group.tensor, "duplicate RCO tensor");
        }
        const auto id = gguf_find_tensor(metadata, group.tensor.c_str());
        check(id >= 0, "RCO source tensor missing");
        const auto * shape = gguf_get_tensor_ne(metadata, id);
        check(shape[0] == int64_t(group.data.columns) && shape[1] == int64_t(group.data.rows) && shape[2] == 1 &&
                  shape[3] == 1,
              "RCO source shape mismatch");
        fixed -= aligned(gguf_get_tensor_size(metadata, id));
        group.graph  = std::make_unique<linear_graph>(group.data, argv[7]);
        group.target = group.graph->evaluate(group.data.weights);
        std::vector<uint64_t> row_costs;
        json                  choices = json::array();
        for (auto type : { GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 }) {
            check(group.data.columns % ggml_blck_size(type) == 0, "candidate row alignment mismatch");
            replacement r{ type, std::vector<uint8_t>(ggml_row_size(type, group.data.columns) * group.data.rows) };
            if (type == GGML_TYPE_Q2_0 && entry.contains("gsq_model")) {
                r.bytes = tensor_bytes(entry.at("gsq_model").get<std::string>(), group.tensor, type);
            } else {
                ggml_get_type_traits(type)->from_float_ref(group.data.weights.data(), r.bytes.data(),
                                                           group.data.weights.size());
            }
            check(r.bytes.size() == ggml_row_size(type, group.data.columns) * group.data.rows &&
                      ggml_validate_row_data(type, r.bytes.data(), r.bytes.size()),
                  "invalid quantization candidate");
            std::vector<float> weights(group.data.weights.size());
            ggml_get_type_traits(type)->to_float(r.bytes.data(), weights.data(), weights.size());
            const std::string label = ggml_type_name(type);
            const auto        file  = store / (group.tensor + "." + label + ".bin");
            std::ofstream     f(file, std::ios::binary);
            write_bytes(f, r.bytes.data(), r.bytes.size());
            choices.push_back({
                { "type", label },
                { "bytes", r.bytes.size() },
                { "aligned_bytes", aligned(r.bytes.size()) },
                { "reconstruction_mse", mse(group.graph->evaluate(weights), group.target) },
                { "file", file.string() }
            });
            row_costs.push_back(aligned(r.bytes.size()));
            group.weights.push_back(std::move(weights));
            group.candidates.push_back(std::move(r));
        }
        costs.push_back(std::move(row_costs));
        report["candidates"].push_back({
            { "tensor",  group.tensor },
            { "choices", choices      }
        });
        groups.push_back(std::move(group));
    }
    gguf_free(metadata);
    check(!groups.empty(), "no RCO tensors");
    std::vector<double> alpha(groups.size() * 3, 0), first(alpha.size(), 0), second(alpha.size(), 0);
    retract(alpha, costs, double(limit));
    adam_graph      optimizer(alpha.size());
    std::mt19937_64 rng(42);
    const double    start            = seconds();
    double          largest_residual = 0, largest_tangent = 0;
    for (int step = 0; step < steps; ++step) {
        const double tau   = std::pow(.05, double(step) / (steps - 1));
        auto         noisy = alpha;
        for (double & z : noisy) {
            const double u = (double(rng() >> 11) + .5) / 9007199254740992.;
            z              = (z - std::log(-std::log(u))) / tau;
        }
        const auto                       soft = probabilities(noisy, 3);
        std::vector<std::vector<double>> scores(groups.size(), std::vector<double>(3));
        for (size_t i = 0; i < groups.size(); ++i) {
            for (size_t k = 0; k < 3; ++k) {
                scores[i][k] = noisy[i * 3 + k];
            }
        }
        const auto          hard = assign(costs, scores, limit);
        std::vector<double> grad(alpha.size(), 0);
        double              loss = 0;
        for (size_t i = 0; i < groups.size(); ++i) {
            auto & group  = groups[i];
            auto   output = group.graph->evaluate(group.weights[hard[i]]);
            loss += mse(output, group.target) / groups.size();
            for (size_t j = 0; j < output.size(); ++j) {
                output[j] = float(2 * (output[j] - group.target[j]) / (output.size() * groups.size()));
            }
            const auto dw    = group.graph->gradient(output);
            double     dp[3] = { 0, 0, 0 }, mean = 0;
            for (size_t k = 0; k < 3; ++k) {
                for (size_t j = 0; j < dw.size(); ++j) {
                    dp[k] += double(dw[j]) * group.weights[k][j];
                }
                mean += soft[i * 3 + k] * dp[k];
            }
            for (size_t k = 0; k < 3; ++k) {
                grad[i * 3 + k] = soft[i * 3 + k] * (dp[k] - mean) / tau;
            }
        }
        auto n = normal(alpha, costs);
        project(grad, n);
        largest_tangent = std::max(largest_tangent, std::abs(dot(grad, n)));
        optimizer.step(alpha, grad, first, second, step + 1, .05f);
        const double achieved = retract(alpha, costs, double(limit));
        largest_residual      = std::max(largest_residual, std::abs(achieved - limit));
        n                     = normal(alpha, costs);
        project(first, n);
        if (step % 10 == 0 || step == steps - 1) {
            report["history"].push_back({
                { "step", step },
                { "reconstruction_mse", loss },
                { "expected_tensor_bytes", achieved },
                { "temperature", tau },
                { "transport_dot", dot(first, n) }
            });
            std::cerr << "rco step " << step << " mse " << loss << " bytes " << achieved << '\n';
        }
    }
    report["training_seconds"] = seconds() - start;
    std::vector<std::vector<double>> scores(groups.size(), std::vector<double>(3));
    for (size_t i = 0; i < groups.size(); ++i) {
        for (size_t k = 0; k < 3; ++k) {
            scores[i][k] = alpha[i * 3 + k];
        }
    }
    const auto                         selected = assign(costs, scores, limit);
    std::map<std::string, replacement> changes;
    uint64_t                           used = 0;
    for (size_t i = 0; i < groups.size(); ++i) {
        used += costs[i][selected[i]];
        report["assignment"].push_back({
            { "tensor",        groups[i].tensor                                       },
            { "type",          ggml_type_name(groups[i].candidates[selected[i]].type) },
            { "aligned_bytes", costs[i][selected[i]]                                  }
        });
        changes.emplace(groups[i].tensor, std::move(groups[i].candidates[selected[i]]));
    }
    materialize(argv[2], argv[4], changes);
    const uint64_t actual = std::filesystem::file_size(argv[4]);
    check(actual == fixed + used && actual <= fixed + limit, "materialized GGUF violates exact byte budget");
    report["precision_logits"]           = alpha;
    report["fixed_gguf_bytes"]           = fixed;
    report["tensor_byte_budget"]         = limit;
    report["total_byte_budget"]          = fixed + limit;
    report["actual_file_bytes"]          = actual;
    report["max_budget_residual_bytes"]  = largest_residual;
    report["max_projected_gradient_dot"] = largest_tangent;
    save_json(argv[8], report);
    save_json((store / "manifest.json").string(), report.at("candidates"));
}

static std::vector<float> log_probs(const std::vector<float> & logits, size_t vocabulary) {
    check(vocabulary > 0 && logits.size() % vocabulary == 0, "invalid logits shape");
    auto result = logits;
    for (size_t offset = 0; offset < logits.size(); offset += vocabulary) {
        const float high = *std::max_element(logits.begin() + offset, logits.begin() + offset + vocabulary);
        double sum = 0;
        for (size_t k = 0; k < vocabulary; ++k) { sum += std::exp(double(logits[offset + k]) - high); }
        const double norm = high + std::log(sum);
        for (size_t k = 0; k < vocabulary; ++k) { result[offset + k] = float(logits[offset + k] - norm); }
    }
    return result;
}

static std::vector<float> cached_teacher(native_global & model, const candidate_store & store,
    const std::vector<int32_t> & tokens, const std::string & directory, size_t index) {
    std::filesystem::create_directories(directory);
    const std::string path = (std::filesystem::path(directory) / (std::to_string(index) + ".f32")).string();
    if (std::filesystem::exists(path + ".json")) {
        json metadata;
        std::ifstream metadata_file(path + ".json");
        metadata_file >> metadata;
        check(metadata.at("source_identity") == store.source_identity && metadata.at("tokens") == tokens,
              "teacher cache source or tokens differ");
        const size_t elements = metadata.at("elements");
        check(elements > 0 && elements % (tokens.size() - 1) == 0 &&
              std::filesystem::file_size(path) == elements * sizeof(float), "teacher cache shape differs");
        std::vector<float> teacher(elements);
        std::ifstream input(path, std::ios::binary);
        read_bytes(input, teacher.data(), elements * sizeof(float));
        check(hash_sha256_hex(teacher.data(), elements * sizeof(float)) == metadata.at("sha256").get<std::string>(),
              "teacher cache hash differs");
        return teacher;
    }
    check(!std::filesystem::exists(path), "teacher cache payload has no manifest");
    auto logits = model.reference_logits(tokens, true);
    const size_t vocabulary = logits.size() / tokens.size();
    logits.resize(logits.size() - vocabulary);
    auto teacher = log_probs(logits, vocabulary);
    std::ofstream output(path, std::ios::binary);
    write_bytes(output, teacher.data(), teacher.size() * sizeof(float));
    output.close();
    save_json(path + ".json", { { "source_identity", store.source_identity }, { "tokens", tokens },
        { "elements", teacher.size() }, { "sha256", hash_sha256_hex(teacher.data(), teacher.size() * sizeof(float)) } });
    return teacher;
}

static void trim_teacher_cache(const candidate_store & store, const std::string & directory, size_t keep, uint64_t limit) {
    if (limit == 0) { return; }
    std::vector<std::filesystem::directory_entry> payloads;
    uint64_t bytes = 0;
    for (const auto & entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".f32") {
            bytes += entry.file_size();
            if (entry.path().stem() != std::to_string(keep)) { payloads.push_back(entry); }
        }
    }
    std::sort(payloads.begin(), payloads.end(), [](const auto & a, const auto & b) { return a.last_write_time() < b.last_write_time(); });
    for (const auto & entry : payloads) {
        if (bytes <= limit) { break; }
        check(std::filesystem::exists(entry.path().string() + ".json"), "teacher cache payload has no manifest");
        json metadata;
        std::ifstream input(entry.path().string() + ".json");
        input >> metadata;
        check(metadata.at("source_identity") == store.source_identity, "teacher cache directory contains another source");
        bytes -= entry.file_size();
        std::filesystem::remove(entry.path().string() + ".json");
        std::filesystem::remove(entry.path());
    }
    check(bytes <= limit, "teacher cache budget cannot hold one window");
}

static void rco_global(int argc, char ** argv) {
    check(argc == 11 || argc == 12, "rco-global MODEL MANIFEST TOKENS CACHE OUTPUT TOTAL_BYTE_BUDGET STEPS BACKEND REPORT [CONFIG]");
    check(!std::filesystem::exists(argv[10]), "RCO report already exists");
    candidate_store store(argv[2], argv[3]);
    json token_json, config = json::object();
    std::ifstream token_file(argv[4]);
    token_file >> token_json;
    if (argc == 12) { std::ifstream file(argv[11]); file >> config; }
    const auto windows = token_json.get<std::vector<std::vector<int32_t>>>();
    check(!windows.empty(), "empty RCO corpus");
    size_t context = 0, corpus_tokens = 0;
    for (const auto & window : windows) {
        check(window.size() >= 2 && window.size() <= 512, "invalid RCO token window");
        context = std::max(context, window.size());
        corpus_tokens += window.size() - 1;
    }
    const auto costs = store.costs();
    const size_t options = costs.front().size();
    check(std::all_of(costs.begin(), costs.end(), [options](const auto & row) { return row.size() == options; }),
          "global RCO requires equal candidate counts");
    const uint64_t total_budget = std::stoull(argv[7]);
    check(total_budget > store.fixed && total_budget <= (1ULL << 53), "total budget is outside fixed model or exact arithmetic range");
    const uint64_t limit = total_budget - store.fixed;
    const int steps = std::stoi(argv[8]);
    check(steps >= 2 && steps <= 10000, "RCO steps must be 2..10000");
    const int stop_after = config.value("stop_after", steps);
    check(stop_after > 0 && stop_after <= steps, "invalid RCO stop step");
    const std::string objective = config.value("objective", std::string("kl"));
    check(objective == "ce" || objective == "kl", "global objective must be ce or kl");
    const bool float32_math = config.value("float32_math", true);
    check(float32_math, "rco-global requires validated float32_math; rounded BF16 activation backward is unsupported");
    const double lr = config.value("learning_rate", .05);
    const uint64_t seed = config.value("seed", uint64_t(42));
    size_t batches = config.value("batches_per_step", size_t(1));
    if (batches == 0 || batches > windows.size()) { batches = windows.size(); }
    const int samples = config.value("gumbel_samples", 1);
    check(std::isfinite(lr) && lr > 0 && samples >= 1 && samples <= 16, "invalid RCO optimizer settings");
    const auto manifest_text = text_file(argv[3]), tokens_text = text_file(argv[4]);
    json fingerprint = { { "source_identity", store.source_identity },
        { "manifest_sha256", hash_sha256_hex(manifest_text.data(), manifest_text.size()) },
        { "tokens_sha256", hash_sha256_hex(tokens_text.data(), tokens_text.size()) },
        { "total_byte_budget", total_budget }, { "steps", steps }, { "objective", objective },
        { "float32_math", float32_math }, { "learning_rate", lr }, { "seed", seed },
        { "batches_per_step", batches }, { "gumbel_samples", samples }, { "backend", argv[9] } };
    std::vector<double> alpha(store.groups.size() * options, 0), first(alpha.size(), 0), second(alpha.size(), 0);
    json history = json::array();
    json monitor = nullptr;
    const int monitor_index = config.value("monitor_window", -1);
    check(monitor_index >= -1 && (monitor_index < 0 || size_t(monitor_index) < windows.size()), "invalid monitor window");
    if (monitor_index >= 0) { fingerprint["monitor_window"] = monitor_index; }
    std::mt19937_64 rng(seed);
    int begin = 0;
    if (config.contains("resume")) {
        json checkpoint;
        std::ifstream file(config.at("resume").get<std::string>());
        file >> checkpoint;
        check(checkpoint.at("fingerprint") == fingerprint, "stale RCO checkpoint");
        alpha = checkpoint.at("alpha").get<std::vector<double>>();
        first = checkpoint.at("first").get<std::vector<double>>();
        second = checkpoint.at("second").get<std::vector<double>>();
        begin = checkpoint.at("step");
        history = checkpoint.at("history");
        if (monitor_index >= 0) { monitor = checkpoint.at("monitor"); }
        std::istringstream state(checkpoint.at("rng").get<std::string>());
        state >> rng;
        check(bool(state) && alpha.size() == store.groups.size() * options && first.size() == alpha.size() && second.size() == alpha.size() &&
              begin >= 0 && begin < stop_after && history.size() == size_t(begin), "invalid RCO checkpoint state");
        check(std::all_of(alpha.begin(), alpha.end(), [](double x) { return std::isfinite(x); }) &&
              std::all_of(first.begin(), first.end(), [](double x) { return std::isfinite(x); }) &&
              std::all_of(second.begin(), second.end(), [](double x) { return std::isfinite(x) && x >= 0; }), "non-finite RCO checkpoint state");
    } else {
        retract(alpha, costs, double(limit));
    }
    native_global model(store, context, argv[9], float32_math);
    adam_graph optimizer(alpha.size());
    const double started = seconds();
    auto monitored_loss = [&](const std::vector<size_t> & hard) {
        const auto & tokens = windows[monitor_index];
        const auto teacher = objective == "kl" ? cached_teacher(model, store, tokens, argv[5], monitor_index) : std::vector<float>{};
        if (objective == "kl") { trim_teacher_cache(store, argv[5], monitor_index, config.value("teacher_cache_max_bytes", uint64_t(0))); }
        return model.evaluate(tokens, teacher, objective == "kl", probabilities(alpha, options), hard, 1, false).loss;
    };
    if (monitor_index >= 0 && begin == 0) {
        std::vector<std::vector<double>> scores(store.groups.size(), std::vector<double>(options));
        for (size_t i = 0; i < store.groups.size(); ++i) { std::copy_n(alpha.begin() + i * options, options, scores[i].begin()); }
        monitor = { { "window", monitor_index }, { "scored_tokens", windows[monitor_index].size() - 1 },
            { "initial_loss", monitored_loss(assign(costs, scores, limit)) } };
    }
    std::string checkpoint_path;
    for (int step = begin; step < stop_after; ++step) {
        const double temperature = std::pow(.05, double(step) / (steps - 1));
        std::vector<size_t> order(windows.size());
        std::iota(order.begin(), order.end(), 0);
        if (batches < windows.size()) { std::shuffle(order.begin(), order.end(), rng); order.resize(batches); }
        size_t batch_tokens = 0;
        for (size_t index : order) { batch_tokens += windows[index].size() - 1; }
        std::vector<double> gradient(alpha.size(), 0);
        double loss = 0;
        uint64_t peak_compute = 0;
        for (int sample = 0; sample < samples; ++sample) {
            auto noisy = alpha;
            for (double & z : noisy) {
                const double u = (double(rng() >> 11) + .5) / 9007199254740992.;
                z = (z - std::log(-std::log(u))) / temperature;
            }
            const auto soft = probabilities(noisy, options);
            std::vector<std::vector<double>> scores(store.groups.size(), std::vector<double>(options));
            for (size_t i = 0; i < store.groups.size(); ++i) {
                std::copy_n(noisy.begin() + i * options, options, scores[i].begin());
            }
            const auto hard = assign(costs, scores, limit);
            for (size_t index : order) {
                const auto teacher = objective == "kl" ? cached_teacher(model, store, windows[index], argv[5], index) : std::vector<float>{};
                if (objective == "kl") { trim_teacher_cache(store, argv[5], index, config.value("teacher_cache_max_bytes", uint64_t(0))); }
                const auto result = model.evaluate(windows[index], teacher, objective == "kl", soft, hard, temperature, true);
                const double weight = double(windows[index].size() - 1) / (batch_tokens * samples);
                loss += result.loss * weight;
                for (size_t j = 0; j < gradient.size(); ++j) { gradient[j] += result.gradient[j] * weight; }
                peak_compute = std::max(peak_compute, result.compute_bytes);
                std::cerr << "rco global step " << step << " window " << index << " loss " << result.loss << '\n';
            }
        }
        const double raw_norm = std::sqrt(dot(gradient, gradient));
        auto n = normal(alpha, costs);
        project(gradient, n);
        const double projected_dot = dot(gradient, n);
        optimizer.step(alpha, gradient, first, second, step + 1, float(lr));
        const double expected = retract(alpha, costs, double(limit));
        n = normal(alpha, costs);
        project(first, n);
        history.push_back({ { "step", step }, { "loss", loss }, { "temperature", temperature },
            { "windows", order }, { "scored_tokens", batch_tokens }, { "raw_gradient_norm", raw_norm },
            { "projected_gradient_dot", projected_dot }, { "transport_dot", dot(first, n) },
            { "expected_tensor_bytes", expected }, { "compute_bytes", peak_compute } });
        std::ostringstream state;
        state << rng;
        checkpoint_path = std::string(argv[10]) + ".step." + std::to_string(step + 1) + ".json";
        save_json(checkpoint_path, { { "fingerprint", fingerprint }, { "step", step + 1 },
            { "alpha", alpha }, { "first", first }, { "second", second }, { "rng", state.str() }, { "history", history }, { "monitor", monitor } });
    }
    std::vector<std::vector<double>> scores(store.groups.size(), std::vector<double>(options));
    for (size_t i = 0; i < store.groups.size(); ++i) { std::copy_n(alpha.begin() + i * options, options, scores[i].begin()); }
    const auto selected = assign(costs, scores, limit);
    std::map<std::string, replacement> changes;
    uint64_t expected = store.fixed;
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto & choice = store.groups[i].choices[selected[i]];
        expected += choice.aligned_bytes;
        changes.emplace(store.groups[i].tensor, replacement{ choice.type, {}, choice.file });
    }
    check(expected <= total_budget, "final RCO assignment exceeds total bytes");
    const bool complete = stop_after == steps;
    if (monitor_index >= 0 && complete) { monitor["final_loss"] = monitored_loss(selected); }
    json report = { { "algorithm", "RCO global model CE/KL with budget-constrained Gumbel STE" }, { "objective", objective },
        { "fingerprint", fingerprint }, { "complete", complete }, { "completed_steps", stop_after },
        { "history", history }, { "monitor", monitor }, { "checkpoint", checkpoint_path }, { "precision_logits", alpha },
        { "first_moment", first }, { "second_moment", second }, { "assignment", selected },
        { "fixed_gguf_bytes", store.fixed }, { "expected_file_bytes", expected }, { "total_byte_budget", total_budget },
        { "remaining_bytes", total_budget - expected }, { "training_seconds", seconds() - started },
        { "calibration_windows", windows.size() }, { "calibration_scored_tokens", corpus_tokens },
        { "teacher_execution", "CPU native inference from source BF16 GGUF; cached full vocabulary log probabilities" } };
    if (complete && config.value("publish", true)) {
        materialize(argv[2], argv[6], changes, expected, total_budget);
        report["output"] = argv[6];
        report["actual_file_bytes"] = std::filesystem::file_size(argv[6]);
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    report["peak_cpu_rss_kib"] = usage.ru_maxrss;
    save_json(argv[10], report);
}

static void global_check(int argc, char ** argv) {
    check(argc == 7, "global-check MODEL TOKENS CONFIG REPORT GPU_LAYERS");
    check(!std::filesystem::exists(argv[5]), "global-check report already exists");
    std::ifstream tf(argv[3]), cf(argv[4]);
    json token_json, config;
    tf >> token_json; cf >> config;
    const auto tokens = token_json.get<std::vector<int32_t>>();
    const std::string manifest = config.value("manifest", std::string(argv[5]) + ".candidates/manifest.json");
    if (!config.contains("manifest")) {
        create_candidate_store(argv[2], argv[4], std::string(argv[5]) + ".candidates", std::string(argv[5]) + ".store.json");
    }
    candidate_store store(argv[2], manifest);
    const bool float32_math = config.value("float32_math", false);
    native_global model(store, tokens.size(), std::stoi(argv[6]) == 0 ? "CPU" : "CUDA0", float32_math);
    const auto cached = model.reference_logits(tokens, true);
    const auto uncached = model.reference_logits(tokens, false);
    double max_error = 0;
    for (size_t i = 0; i < cached.size(); ++i) { max_error = std::max(max_error, std::abs(double(cached[i]) - uncached[i])); }
    const size_t vocabulary = cached.size() / tokens.size();
    const std::vector<float> reference(cached.begin(), cached.end() - vocabulary);
    const auto teacher = log_probs(reference, vocabulary);
    const size_t options = store.groups[0].choices.size();
    check(std::all_of(store.groups.begin(), store.groups.end(), [options](const auto & g) { return g.choices.size() == options; }),
          "global-check requires equal candidate counts");
    std::vector<double> alpha(store.groups.size() * options, 0);
    json report = { { "mode", "global CE/KL gradient validation" }, { "cached_vs_training_logits_max_error", max_error },
                    { "tokens", tokens }, { "float32_math", float32_math },
                    { "candidate_manifest", manifest }, { "gradient_checks", json::array() } };
    const bool finite_difference = config.value("finite_difference", true);
    report["finite_difference_enabled"] = finite_difference;
    bool passed = max_error < 5e-3 && finite_difference;
    const auto fd_steps = config.value("finite_difference_steps", std::vector<double>{ .1, .05, .025 });
    std::vector<size_t> fd_groups;
    for (size_t i = 0; i < store.groups.size(); ++i) { fd_groups.push_back(i); }
    fd_groups = config.value("finite_difference_groups", fd_groups);
    check(!fd_groups.empty() && std::all_of(fd_groups.begin(), fd_groups.end(), [&](size_t i) { return i < store.groups.size(); }),
          "invalid finite difference groups");
    report["finite_difference_groups"] = fd_groups;
    report["finite_difference_policy"] = "each group must match a two-point or five-point centered stencil; all estimates use unchanged tolerances";
    check(!fd_steps.empty() && std::all_of(fd_steps.begin(), fd_steps.end(), [](double h) { return std::isfinite(h) && h > 0 && h <= .25; }),
          "invalid finite difference steps");
    for (bool kl : { false, true }) {
        const auto soft = probabilities(alpha, options);
        const auto result = model.evaluate(tokens, teacher, kl, soft, {}, 1, true);
        std::cerr << "global " << (kl ? "kl" : "ce") << " loss " << result.loss << " gradient " << json(result.gradient).dump() << '\n';
        json test = { { "objective", kl ? "kl" : "ce" }, { "loss", result.loss }, { "gradient", result.gradient },
                      { "forward_recompute_max_error", result.forward_recompute_max_error },
                      { "compute_bytes", result.compute_bytes }, { "finite_differences", json::array() } };
        if (finite_difference) {
            for (size_t group : fd_groups) {
                const size_t index = group * options;
                bool group_match = false;
                std::vector<std::pair<double, double>> differences;
                for (double h : fd_steps) {
                    auto high = alpha, low = alpha;
                    high[index] += h; low[index] -= h;
                    const double hi = model.evaluate(tokens, teacher, kl, probabilities(high, options), {}, 1, false).loss;
                    const double lo = model.evaluate(tokens, teacher, kl, probabilities(low, options), {}, 1, false).loss;
                    const double fd = (hi - lo) / (2 * h), analytic = result.gradient[index];
                    const double error = std::abs(fd - analytic);
                    differences.emplace_back(h, fd);
                    test["finite_differences"].push_back({ { "group", group }, { "step", h }, { "analytic", analytic },
                        { "finite_difference", fd }, { "absolute_error", error },
                        { "tolerance", std::max(1e-5, std::abs(analytic) * 1e-3) } });
                    const bool match = error < std::max(1e-5, std::abs(analytic) * 1e-3);
                    group_match = group_match || match;
                    std::cerr << "global finite difference group " << group << " h " << h << " analytic " << analytic << " numeric " << fd << " pass " << match << '\n';
                }
                for (const auto & inner : differences) {
                    for (const auto & outer : differences) {
                        if (std::abs(2 * inner.first - outer.first) > 1e-12) { continue; }
                        const double fd = (4 * inner.second - outer.second) / 3;
                        const double analytic = result.gradient[index];
                        const double error = std::abs(fd - analytic);
                        const double tolerance = std::max(1e-5, std::abs(analytic) * 1e-3);
                        test["finite_differences"].push_back({ { "group", group }, { "method", "five-point centered stencil" },
                            { "step", inner.first }, { "outer_step", outer.first }, { "analytic", analytic },
                            { "finite_difference", fd }, { "absolute_error", error }, { "tolerance", tolerance } });
                        group_match = group_match || error < tolerance;
                        std::cerr << "global five-point difference group " << group << " h " << inner.first << " error " << error << " pass " << (error < tolerance) << '\n';
                    }
                }
                passed = passed && group_match;
            }
        }
        report["gradient_checks"].push_back(std::move(test));
    }
    report["passed"] = passed;
    save_json(argv[5], report);
    std::cout << report.dump(2) << '\n';
    check(passed, "Gemma global gradient validation failed; inspect report");
}

static void evaluate(int argc, char ** argv) {
    check(argc == 7, "eval MODEL TEXT REFERENCE REPORT GPU_LAYERS (REFERENCE='-' writes REPORT.logits)");
    model_session  m;
    const double   start  = seconds();
    const auto     tokens = run_model(m, argv[2], text_file(argv[3]), nullptr, std::stoi(argv[6]));
    const uint64_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(m.model)), count = tokens.size() - 1;
    std::ifstream  ref;
    uint64_t       rv = 0, rn = 0;
    if (std::string(argv[4]) != "-") {
        ref.open(argv[4], std::ios::binary);
        read_bytes(ref, &rv, 8);
        read_bytes(ref, &rn, 8);
        check(rv == vocab && rn == count, "reference shape mismatch");
        std::vector<llama_token> rt(tokens.size());
        read_bytes(ref, rt.data(), rt.size() * sizeof(llama_token));
        check(rt == tokens, "reference token mismatch");
    }
    std::ofstream cache;
    if (!ref.is_open()) {
        check(!std::filesystem::exists(std::string(argv[5]) + ".logits"), "reference exists");
        cache.open(std::string(argv[5]) + ".logits", std::ios::binary);
        write_bytes(cache, &vocab, 8);
        write_bytes(cache, &count, 8);
        write_bytes(cache, tokens.data(), tokens.size() * sizeof(llama_token));
    }
    double             nll = 0, kl = 0, error = 0, max_error = 0;
    std::vector<float> teacher(vocab);
    for (size_t t = 0; t < count; ++t) {
        const float * l = llama_get_logits_ith(m.ctx, t);
        check(l != nullptr, "missing logits");
        const double max = *std::max_element(l, l + vocab);
        double       sum = 0;
        for (size_t k = 0; k < vocab; ++k) {
            sum += std::exp(l[k] - max);
        }
        const double logz = max + std::log(sum);
        nll += logz - l[tokens[t + 1]];
        if (ref.is_open()) {
            read_bytes(ref, teacher.data(), vocab * 4);
            const double tm = *std::max_element(teacher.begin(), teacher.end());
            double       ts = 0;
            for (float z : teacher) {
                ts += std::exp(z - tm);
            }
            const double tz = tm + std::log(ts);
            for (size_t k = 0; k < vocab; ++k) {
                const double lp = teacher[k] - tz;
                kl += std::exp(lp) * (lp - (l[k] - logz));
                const double e = double(l[k]) - teacher[k];
                error += e * e;
                max_error = std::max(max_error, std::abs(e));
            }
        } else {
            write_bytes(cache, l, vocab * 4);
        }
    }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    json j = {
        { "model",                  argv[2]                                                       },
        { "tokens",                 tokens                                                        },
        { "evaluation_tokens",      count                                                         },
        { "context",                llama_n_ctx(m.ctx)                                            },
        { "nll",                    nll / count                                                   },
        { "ppl",                    std::exp(nll / count)                                         },
        { "teacher_kl",             ref.is_open() ? json(kl / count) : json(nullptr)              },
        { "logits_mse",             ref.is_open() ? json(error / (count * vocab)) : json(nullptr) },
        { "max_logits_error",       ref.is_open() ? json(max_error) : json(nullptr)               },
        { "seconds_including_load", seconds() - start                                             },
        { "file_bytes",             std::filesystem::file_size(argv[2])                           },
        { "peak_cpu_rss_kib",       usage.ru_maxrss                                               },
        { "gpu_layers",             std::stoi(argv[6])                                            }
    };
    save_json(argv[5], j);
    std::cout << j.dump(2) << '\n';
}

static void corpus_evaluate(int argc, char ** argv) {
    check(argc == 8, "corpus-eval MODEL WINDOWS TEACHER_CACHE REPORT GPU_LAYERS MODE(write/read)");
    check(!std::filesystem::exists(argv[5]), "corpus report exists");
    const bool write = std::string(argv[7]) == "write";
    check(write || std::string(argv[7]) == "read", "invalid teacher cache mode");
    const std::filesystem::path cache(argv[4]);
    check(!write || !std::filesystem::exists(cache), "teacher cache exists");
    std::ifstream input(argv[3]);
    json ids;
    input >> ids;
    const auto windows = ids.get<std::vector<std::vector<int32_t>>>();
    check(!windows.empty(), "empty evaluation corpus");
    size_t context = 0;
    for (const auto & window : windows) {
        check(window.size() >= 2 && window.size() <= 512, "invalid evaluation window");
        context = std::max(context, window.size());
    }
    const int gpu = std::stoi(argv[6]);
    json teacher_metadata;
    if (write) {
        std::filesystem::create_directories(cache);
        teacher_metadata = { { "source", argv[2] }, { "sha256", file_sha256(argv[2]) },
            { "windows", ids }, { "gpu_layers", gpu }, { "context", context } };
    } else {
        std::ifstream metadata(cache / "manifest.json");
        metadata >> teacher_metadata;
        check(teacher_metadata.at("windows") == ids && teacher_metadata.at("gpu_layers") == gpu &&
              teacher_metadata.at("context") == context, "teacher evaluation conditions differ");
        check(file_sha256(teacher_metadata.at("source").get<std::string>()) == teacher_metadata.at("sha256"),
              "teacher source changed");
    }
    const double start = seconds();
    model_session model;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = gpu;
    if (gpu != 0) {
        model.devices[0] = ggml_backend_dev_by_name("CUDA0");
        check(model.devices[0] != nullptr, "CUDA0 unavailable");
        mp.devices = model.devices;
    }
    model.model = llama_model_load_from_file(argv[2], mp);
    check(model.model != nullptr, "evaluation model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max<size_t>(128, context);
    cp.n_batch = cp.n_ubatch = context;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.op_offload = gpu != 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    model.ctx = llama_init_from_model(model.model, cp);
    check(model.ctx != nullptr, "evaluation context creation failed");
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.model));
    double ce = 0, kl = 0, error = 0, prefill = 0;
    size_t scored = 0, input_tokens = 0;
    json history = json::array();
    for (size_t index = 0; index < windows.size(); ++index) {
        const auto & tokens = windows[index];
        llama_memory_clear(llama_get_memory(model.ctx), true);
        auto batch = llama_batch_init(tokens.size(), 0, 1);
        batch.n_tokens = tokens.size();
        for (size_t j = 0; j < tokens.size(); ++j) {
            batch.token[j] = tokens[j]; batch.pos[j] = j;
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = true;
        }
        const double begin = seconds();
        const int status = llama_decode(model.ctx, batch);
        llama_batch_free(batch);
        check(status == 0, "evaluation decode failed");
        llama_synchronize(model.ctx);
        const double elapsed = seconds() - begin;
        prefill += elapsed;
        const size_t count = tokens.size() - 1;
        std::vector<float> logits(count * vocab);
        for (size_t j = 0; j < count; ++j) {
            const auto * row = llama_get_logits_ith(model.ctx, j);
            check(row != nullptr, "evaluation logits unavailable");
            std::copy(row, row + vocab, logits.begin() + j * vocab);
        }
        const auto path = cache / (std::to_string(index) + ".f32");
        std::vector<float> teacher;
        if (write) {
            std::ofstream output(path, std::ios::binary);
            write_bytes(output, logits.data(), logits.size() * sizeof(float));
            output.close();
            check(bool(output), "teacher payload write failed");
            save_json((cache / (std::to_string(index) + ".json")).string(), {
                { "tokens", tokens }, { "vocabulary", vocab }, { "elements", logits.size() },
                { "sha256", hash_sha256_hex(logits.data(), logits.size() * sizeof(float)) } });
            teacher = logits;
        } else {
            json metadata;
            std::ifstream file(cache / (std::to_string(index) + ".json"));
            file >> metadata;
            check(metadata.at("tokens") == tokens && metadata.at("vocabulary") == vocab &&
                  metadata.at("elements") == logits.size() &&
                  std::filesystem::file_size(path) == logits.size() * sizeof(float), "teacher token or shape mismatch");
            teacher.resize(logits.size());
            std::ifstream payload(path, std::ios::binary);
            read_bytes(payload, teacher.data(), teacher.size() * sizeof(float));
            check(hash_sha256_hex(teacher.data(), teacher.size() * sizeof(float)) == metadata.at("sha256"),
                  "teacher payload hash mismatch");
        }
        double window_ce = 0, window_kl = 0, window_error = 0;
        check(std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }) &&
              std::all_of(teacher.begin(), teacher.end(), [](float x) { return std::isfinite(x); }),
              "nonfinite evaluation logits");
        for (size_t row = 0; row < count; ++row) {
            const auto offset = row * vocab;
            const double a = *std::max_element(logits.begin() + offset, logits.begin() + offset + vocab);
            const double b = *std::max_element(teacher.begin() + offset, teacher.begin() + offset + vocab);
            double sa = 0, sb = 0;
            for (size_t j = 0; j < vocab; ++j) {
                sa += std::exp(double(logits[offset + j]) - a);
                sb += std::exp(double(teacher[offset + j]) - b);
            }
            const double za = a + std::log(sa), zb = b + std::log(sb);
            check(tokens[row + 1] >= 0 && size_t(tokens[row + 1]) < vocab, "evaluation target outside vocabulary");
            window_ce += za - logits[offset + tokens[row + 1]];
            for (size_t j = 0; j < vocab; ++j) {
                const double q = double(teacher[offset + j]) - zb;
                window_kl += std::exp(q) * (q - (double(logits[offset + j]) - za));
                const double difference = double(logits[offset + j]) - teacher[offset + j];
                window_error += difference * difference;
            }
        }
        ce += window_ce; kl += window_kl; error += window_error;
        scored += count; input_tokens += tokens.size();
        history.push_back({ { "index", index }, { "scored_tokens", count },
            { "ce_loss", window_ce / count }, { "teacher_kl", window_kl / count }, { "prefill_seconds", elapsed } });
        std::cerr << "evaluation window " << index << " ce " << window_ce / count << " kl " << window_kl / count << '\n';
    }
    if (write) { save_json((cache / "manifest.json").string(), teacher_metadata); }
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    save_json(argv[5], { { "model", argv[2] }, { "teacher", teacher_metadata.at("source") },
        { "evaluation_tokens", scored }, { "input_tokens", input_tokens }, { "context", context },
        { "ce_loss", ce / scored }, { "ppl", std::exp(ce / scored) }, { "teacher_kl", kl / scored },
        { "logits_mse", error / (double(scored) * vocab) }, { "windows", history },
        { "file_bytes", std::filesystem::file_size(argv[2]) }, { "gpu_layers", gpu },
        { "prefill_seconds", prefill }, { "prefill_tokens_per_second", input_tokens / prefill },
        { "seconds_including_load", seconds() - start }, { "peak_cpu_rss_kib", usage.ru_maxrss } });
}

static void infer(int argc, char ** argv) {
    check(argc == 7, "infer MODEL TEXT GENERATION_TOKENS REPORT GPU_LAYERS");
    const int count = std::stoi(argv[4]);
    check(count > 0 && count <= 32, "generation limit is 1..32 tokens");
    model_session model;
    std::vector<llama_token> ids;
    const bool supplied = std::string(argv[1]) == "infer-tokens";
    if (supplied) {
        json input;
        std::ifstream file(argv[3]);
        file >> input;
        ids = input.get<std::vector<llama_token>>();
    }
    const auto tokens = run_model(model, argv[2], supplied ? std::string() : text_file(argv[3]), nullptr,
                                  std::stoi(argv[6]), supplied ? &ids : nullptr);
    llama_perf_context_reset(model.ctx);
    const auto *             vocab = llama_model_get_vocab(model.model);
    const size_t             size  = llama_vocab_n_tokens(vocab);
    std::string              text;
    std::vector<llama_token> generated;
    const double             start = seconds();
    for (int step = 0; step < count; ++step) {
        const float * logits = llama_get_logits_ith(model.ctx, -1);
        check(logits != nullptr, "generation logits unavailable");
        const auto token = llama_token(std::max_element(logits, logits + size) - logits);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        char      piece[512];
        const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        check(n >= 0, "token piece exceeds PoC buffer");
        text.append(piece, n);
        generated.push_back(token);
        auto batch         = llama_batch_init(1, 0, 1);
        batch.token[0]     = token;
        batch.pos[0]       = tokens.size() + step;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        batch.n_tokens     = 1;
        const int result   = llama_decode(model.ctx, batch);
        llama_batch_free(batch);
        check(result == 0, "generation decode failed");
        llama_synchronize(model.ctx);
        check(llama_get_logits_ith(model.ctx, -1) != nullptr, "decode logits unavailable");
    }
    const double elapsed = seconds() - start;

    save_json(argv[5], {
                           { "model",                     argv[2]                                      },
                           { "greedy",                    true                                         },
                           { "generated_text",            text                                         },
                           { "generated_tokens",          generated                                    },
                           { "prompt_tokens",             tokens.size()                                },
                           { "prompt_token_ids",          tokens                                       },
                           { "prefill_ms",                model.prefill_seconds * 1000                 },
                           { "prefill_tokens_per_second",
                            model.prefill_seconds > 0 ? tokens.size() / model.prefill_seconds : 0      },
                           { "decode_ms",                 elapsed * 1000                               },
                           { "decode_tokens_per_second",  elapsed > 0 ? generated.size() / elapsed : 0 },
                           { "decode_wall_seconds",       elapsed                                      },
                           { "context",                   llama_n_ctx(model.ctx)                       }
    });
    std::cout << text << '\n';
}

int main(int argc, char ** argv) {
    try {
        check(argc >= 2, "commands: self-test, capture, gsq, ptq, rco-proxy, linear-test, eval, infer");
        llama_backend_init();
        const std::string command = argv[1];
        if (command == "self-test") {
            self_test();
        } else if (command == "candidate-store") {
            check(argc == 6, "candidate-store MODEL CONFIG DIRECTORY REPORT");
            create_candidate_store(argv[2], argv[3], argv[4], argv[5]);
        } else if (command == "global-check") {
            global_check(argc, argv);
        } else if (command == "rco-global") {
            rco_global(argc, argv);
        } else if (command == "candidate-validate") {
            check(argc == 4, "candidate-validate MODEL MANIFEST");
            candidate_store store(argv[2], argv[3]);
            for (size_t i = 0; i < store.groups.size(); ++i) {
                for (size_t k = 0; k < store.groups[i].choices.size(); ++k) {
                    store.decode_tile(i, k, 0, 1);
                }
            }
        } else if (command == "candidate-publish") {
            candidate_publish(argc, argv);
        } else if (command == "candidate-verify") {
            candidate_verify(argc, argv);
        } else if (command == "capture") {
            capture_model(argc, argv);
        } else if (command == "capture-store") {
            capture_store(argc, argv);
        } else if (command == "gsq" || command == "gsq-pack") {
            train(argc, argv);
        } else if (command == "ptq") {
            ptq(argc, argv);
        } else if (command == "rco-proxy") {
            rco_proxy(argc, argv);
        } else if (command == "linear-test") {
            linear_test(argc, argv);
        } else if (command == "eval") {
            evaluate(argc, argv);
        } else if (command == "corpus-eval") {
            corpus_evaluate(argc, argv);
        } else if (command == "infer" || command == "infer-tokens") {
            infer(argc, argv);
        } else {
            throw std::runtime_error("unknown command");
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
