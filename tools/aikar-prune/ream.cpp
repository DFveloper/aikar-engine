#include "ream.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "hard-prune.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>

namespace {
ggml_backend_t compute_backend = nullptr;
void check(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("REAM: " + message);
}
void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(std::string("REAM: ") + message);
}
void finite(const std::vector<float> & values) {
    for (float x : values) check(std::isfinite(x), "non-finite weight, activation or cost");
}
void validate_expert(const aikar_ream_expert & e) {
    check(e.embedding > 0 && e.hidden > 0, "invalid expert dimensions");
    const size_t size = (size_t) e.embedding * e.hidden;
    check(e.gate.size() == size && e.up.size() == size && e.down.size() == size, "expert weight shapes differ");
    finite(e.gate); finite(e.up); finite(e.down);
}
struct ggml_delete { void operator()(ggml_context * p) const { ggml_free(p); } };
struct gguf_delete { void operator()(gguf_context * p) const { gguf_free(p); } };
using ctx_ptr = std::unique_ptr<ggml_context, ggml_delete>;
using file_ptr = std::unique_ptr<gguf_context, gguf_delete>;
ctx_ptr context(size_t bytes) {
    ctx_ptr ctx(ggml_init({bytes + 4 * 1024 * 1024, nullptr, false}));
    check(ctx != nullptr, "cannot allocate CPU GGML workspace of " + std::to_string(bytes) + " bytes");
    return ctx;
}
void compute(ggml_context * ctx, ggml_tensor * out, int32_t threads, ggml_tensor * hidden = nullptr) {
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (!compute_backend) {
        check(ggml_graph_compute_with_ctx(ctx, graph, std::max(1, threads)) == GGML_STATUS_SUCCESS, "CPU expert graph failed");
        return;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) check(ggml_backend_supports_op(compute_backend, ggml_graph_node(graph, i)), "compute device does not support expert graph operation");
    size_t free_bytes = 0, total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(compute_backend), &free_bytes, &total_bytes);
    check(free_bytes > ggml_get_mem_size(ctx), "compute device out of memory: free=" + std::to_string(free_bytes) + ", graph upper bound=" + std::to_string(ggml_get_mem_size(ctx)));
    auto host_buffer = ggml_backend_cpu_buffer_from_ptr(ggml_get_mem_buffer(ctx), ggml_get_mem_size(ctx));
    check(host_buffer != nullptr, "cannot register CPU graph workspace");
    struct buffer_guard {
        ggml_backend_buffer_t buffer;
        ~buffer_guard() { ggml_backend_buffer_free(buffer); }
    } host_guard {host_buffer};
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) tensor->buffer = host_buffer;
    auto copy = ggml_backend_graph_copy(compute_backend, graph);
    check(copy.buffer && copy.graph, "cannot allocate expert graph on compute device");
    struct copy_guard {
        struct ggml_backend_graph_copy copy;
        ~copy_guard() { ggml_backend_graph_copy_free(copy); }
    } guard {copy};
    check(ggml_backend_graph_compute(compute_backend, copy.graph) == GGML_STATUS_SUCCESS, "device expert graph failed");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * original = ggml_graph_node(graph, i);
        if (original == out || original == hidden) ggml_backend_tensor_get(ggml_graph_node(copy.graph, i), original->data, 0, ggml_nbytes(original));
    }
}
std::vector<float> normalize(const std::vector<float> & x, int32_t rows, int32_t columns) {
    check(rows > 0 && columns > 0 && x.size() == (size_t) rows * columns, "invalid feature matrix");
    finite(x);
    std::vector<float> y(x.size());
    for (int32_t r = 0; r < rows; ++r) {
        double norm = 0;
        for (int32_t c = 0; c < columns; ++c) norm += (double) x[r * columns + c] * x[r * columns + c];
        norm = std::sqrt(norm) + 1e-8;
        for (int32_t c = 0; c < columns; ++c) y[r * columns + c] = x[r * columns + c] / norm;
    }
    return y;
}
}

void aikar_ream_set_backend(ggml_backend_t backend) { compute_backend = backend; }

std::vector<float> aikar_ream_normalize_features(const std::vector<float> & values, int32_t rows, int32_t features) {
    return normalize(values, rows, features);
}

std::vector<double> aikar_ream_saliency_weights(const std::vector<double> & saliency) {
    check(!saliency.empty(), "empty saliency");
    double minimum = INFINITY;
    for (double x : saliency) {
        check(std::isfinite(x) && x >= 0, "invalid saliency");
        if (x > 0) minimum = std::min(minimum, x);
    }
    const double fallback = std::isfinite(minimum) ? std::min(0.5, minimum) : 1.0;
    auto result = saliency;
    for (auto & x : result) if (x == 0) x = fallback;
    return result;
}

void aikar_ream_validate_groups(const aikar_ream_groups & g, int32_t experts, int32_t target) {
    check(experts > 0 && target > 0 && target <= experts, "invalid expert count");
    check(g.centers.size() == (size_t) target && g.members.size() == (size_t) target && g.labels.size() == (size_t) experts,
          "group count or mapping size differs");
    std::vector<int32_t> seen(experts, 0);
    std::set<int32_t> centers;
    for (int32_t group = 0; group < target; ++group) {
        check(!g.members[group].empty() && g.members[group][0] == g.centers[group], "missing group center");
        check(centers.insert(g.centers[group]).second, "duplicate group center");
        for (int32_t e : g.members[group]) {
            check(e >= 0 && e < experts, "invalid expert ID");
            check(++seen[e] == 1 && g.labels[e] == group, "duplicate expert or inconsistent label");
        }
    }
    for (int32_t count : seen) check(count == 1, "missing expert");
}

aikar_ream_groups aikar_ream_pseudo_group(const std::vector<double> & saliency, const std::vector<float> & d,
                                        int32_t target, int32_t group_size) {
    const int32_t n = saliency.size();
    check(n > 0 && target > 0 && target <= n && group_size > 0 && (int64_t) target * group_size >= n,
          "target and group size cannot cover all experts");
    check(d.size() == (size_t) n * n, "invalid expert distance matrix");
    finite(d);
    aikar_ream_saliency_weights(saliency);
    aikar_ream_groups g;
    g.centers.resize(n);
    std::iota(g.centers.begin(), g.centers.end(), 0);
    std::stable_sort(g.centers.begin(), g.centers.end(), [&](int32_t a, int32_t b) {
        return saliency[a] == saliency[b] ? a < b : saliency[a] > saliency[b];
    });
    if (target == n) std::sort(g.centers.begin(), g.centers.end());
    g.centers.resize(target);
    g.labels.assign(n, -1);
    g.members.resize(target);
    for (int32_t j = 0; j < target; ++j) {
        g.labels[g.centers[j]] = j;
        g.members[j].push_back(g.centers[j]);
    }
    for (int32_t j = 0; j < target; ++j) {
        std::vector<int32_t> closest(n);
        std::iota(closest.begin(), closest.end(), 0);
        const int32_t center = g.centers[j];
        std::stable_sort(closest.begin(), closest.end(), [&](int32_t a, int32_t b) {
            return d[center * n + a] == d[center * n + b] ? a < b : d[center * n + a] < d[center * n + b];
        });
        for (int32_t e : closest) {
            if (g.members[j].size() >= (size_t) group_size) break;
            if (g.labels[e] < 0) { g.labels[e] = j; g.members[j].push_back(e); }
        }
    }
    aikar_ream_validate_groups(g, n, target);
    return g;
}

std::vector<int32_t> aikar_ream_hungarian(const std::vector<float> & cost, int32_t n) {
    check(n > 0 && cost.size() == (size_t) n * n, "invalid assignment matrix");
    finite(cost);
    std::vector<double> u(n + 1), v(n + 1), minv(n + 1);
    std::vector<int32_t> p(n + 1), way(n + 1);
    std::vector<uint8_t> used(n + 1);
    for (int32_t i = 1; i <= n; ++i) {
        p[0] = i;
        int32_t j0 = 0;
        std::fill(minv.begin(), minv.end(), INFINITY);
        std::fill(used.begin(), used.end(), false);
        do {
            used[j0] = true;
            int32_t i0 = p[j0], j1 = 0;
            double delta = INFINITY;
            for (int32_t j = 1; j <= n; ++j) if (!used[j]) {
                const double cur = cost[(i0 - 1) * n + j - 1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int32_t j = 0; j <= n; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { const int32_t j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int32_t> permutation(n);
    for (int32_t j = 1; j <= n; ++j) permutation[p[j] - 1] = j - 1;
    return permutation;
}

aikar_ream_expert aikar_ream_permute(const aikar_ream_expert & e, const std::vector<int32_t> & perm) {
    validate_expert(e);
    check(perm.size() == (size_t) e.hidden, "invalid neuron permutation size");
    std::vector<bool> seen(e.hidden, false);
    auto result = e;
    for (int32_t i = 0; i < e.hidden; ++i) {
        const int32_t j = perm[i];
        check(j >= 0 && j < e.hidden && !seen[j], "invalid or duplicate neuron permutation");
        seen[j] = true;
        std::copy_n(e.gate.data() + j * e.embedding, e.embedding, result.gate.data() + i * e.embedding);
        std::copy_n(e.up.data() + j * e.embedding, e.embedding, result.up.data() + i * e.embedding);
        for (int32_t d = 0; d < e.embedding; ++d) result.down[d * e.hidden + i] = e.down[d * e.hidden + j];
    }
    return result;
}

void aikar_ream_accumulate(aikar_ream_expert & a, const aikar_ream_expert & e, double weight) {
    validate_expert(e); validate_expert(a);
    check(a.embedding == e.embedding && a.hidden == e.hidden && std::isfinite(weight), "invalid merge operands");
    for (size_t i = 0; i < a.gate.size(); ++i) {
        a.gate[i] += (float) (weight * e.gate[i]);
        a.up[i] += (float) (weight * e.up[i]);
        a.down[i] += (float) (weight * e.down[i]);
    }
    validate_expert(a);
}

std::vector<float> aikar_ream_weight_features(const aikar_ream_expert & e) {
    validate_expert(e);
    std::vector<float> result((size_t) e.hidden * e.embedding * 3);
    for (int32_t i = 0; i < e.hidden; ++i) for (int32_t d = 0; d < e.embedding; ++d) {
        result[i * e.embedding * 3 + d] = e.gate[i * e.embedding + d];
        result[i * e.embedding * 3 + e.embedding + d] = e.up[i * e.embedding + d];
        result[i * e.embedding * 3 + 2 * e.embedding + d] = e.down[d * e.hidden + i];
    }
    return result;
}

std::vector<float> aikar_ream_distances(const std::vector<float> & a, const std::vector<float> & b,
                                      int32_t rows, int32_t columns, int32_t threads, bool normalized) {
    check(rows > 0 && columns > 0 && a.size() == (size_t) rows * columns && b.size() == a.size(), "invalid distance features");
    finite(a); finite(b);
    const auto normalized_a = normalized ? normalize(a, rows, columns) : std::vector<float>();
    const auto normalized_b = normalized ? normalize(b, rows, columns) : std::vector<float>();
    const auto & x = normalized ? normalized_a : a;
    const auto & y = normalized ? normalized_b : b;
    auto ctx = context((x.size() + y.size() + (size_t) rows * rows) * sizeof(float));
    auto * ta = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, rows);
    auto * tb = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, rows);
    memcpy(ta->data, x.data(), x.size() * sizeof(float));
    memcpy(tb->data, y.data(), y.size() * sizeof(float));
    auto * dots = ggml_mul_mat(ctx.get(), tb, ta);
    ggml_prec_set_acc(dots, GGML_PREC_F32);
    ggml_prec_set_src(dots, GGML_PREC_F32, 0);
    ggml_prec_set_src(dots, GGML_PREC_F32, 1);
    compute(ctx.get(), dots, threads);
    std::vector<double> nx(rows), ny(rows);
    for (int32_t i = 0; i < rows; ++i) for (int32_t c = 0; c < columns; ++c) {
        nx[i] += (double) x[i * columns + c] * x[i * columns + c];
        ny[i] += (double) y[i * columns + c] * y[i * columns + c];
    }
    std::vector<float> costs((size_t) rows * rows);
    for (int32_t i = 0; i < rows; ++i) for (int32_t j = 0; j < rows; ++j) {
        costs[i * rows + j] = std::sqrt(std::max(0.0, nx[i] + ny[j] - 2.0 * ((float *) dots->data)[i * rows + j]));
    }
    finite(costs);
    return costs;
}

void aikar_ream_forward(const aikar_ream_expert & e, const std::vector<float> & inputs, int32_t threads,
                       std::vector<float> & hidden, std::vector<float> & outputs) {
    validate_expert(e); finite(inputs);
    check(!inputs.empty() && inputs.size() % e.embedding == 0, "invalid expert input shape");
    const size_t tokens = inputs.size() / e.embedding;
    auto ctx = context((e.gate.size() * 3 + inputs.size() * 2 + (size_t) e.hidden * tokens * 5) * sizeof(float));
    auto * gate = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, e.embedding, e.hidden);
    auto * up = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, e.embedding, e.hidden);
    auto * down = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, e.hidden, e.embedding);
    auto * inp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, e.embedding, tokens);
    memcpy(gate->data, e.gate.data(), e.gate.size() * sizeof(float));
    memcpy(up->data, e.up.data(), e.up.size() * sizeof(float));
    memcpy(down->data, e.down.data(), e.down.size() * sizeof(float));
    memcpy(inp->data, inputs.data(), inputs.size() * sizeof(float));
    auto * gate_out = ggml_mul_mat(ctx.get(), gate, inp);
    auto * up_out = ggml_mul_mat(ctx.get(), up, inp);
    for (auto * projection : {gate_out, up_out}) {
        ggml_prec_set_acc(projection, GGML_PREC_F32);
        ggml_prec_set_src(projection, GGML_PREC_F32, 0);
        ggml_prec_set_src(projection, GGML_PREC_F32, 1);
    }
    auto * act = ggml_mul(ctx.get(), ggml_gelu(ctx.get(), gate_out), up_out);
    auto * out = ggml_mul_mat(ctx.get(), down, act);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_prec_set_src(out, GGML_PREC_F32, 0);
    ggml_prec_set_src(out, GGML_PREC_F32, 1);
    compute(ctx.get(), out, threads, act);
    hidden.assign((float *) act->data, (float *) act->data + e.hidden * tokens);
    outputs.assign((float *) out->data, (float *) out->data + e.embedding * tokens);
    finite(hidden); finite(outputs);
}

struct aikar_ream_forward_runner::impl {
    ctx_ptr ctx;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * output = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t host = nullptr;
    struct ggml_backend_graph_copy device = {};
    ggml_tensor * device_input = nullptr;
    ggml_tensor * device_hidden = nullptr;
    ggml_tensor * device_output = nullptr;
    int32_t embedding, neurons, chunk, threads;
    bool hidden_only = false;
    ggml_tensor * weights = nullptr;
    ggml_tensor * sample_ids = nullptr;
    ggml_tensor * weighted_sum = nullptr;
    ggml_tensor * sampled = nullptr;
    ggml_tensor * device_weights = nullptr;
    ggml_tensor * device_sample_ids = nullptr;
    ggml_tensor * device_weighted_sum = nullptr;
    ggml_tensor * device_sampled = nullptr;
    ~impl() {
        if (device.buffer) ggml_backend_graph_copy_free(device);
        if (host) ggml_backend_buffer_free(host);
    }
};

aikar_ream_forward_runner::aikar_ream_forward_runner(const aikar_ream_expert & e, int32_t chunk, int32_t threads, bool hidden_only, ggml_type precision, bool summarize)
    : state(new impl) {
    validate_expert(e);
    check(chunk > 0, "invalid forward chunk size");
    auto & s = *state;
    s.embedding = e.embedding; s.neurons = e.hidden; s.chunk = chunk; s.threads = std::max(1, threads);
    s.backend = compute_backend; s.hidden_only = hidden_only;
    check(!summarize || (hidden_only && s.backend), "Expert summary requires a hidden-only GPU graph");
    check(precision == GGML_TYPE_F32 || (precision == GGML_TYPE_F16 && s.backend), "F16 Expert features require a GPU backend");
    s.ctx = context((e.gate.size() * 3 + (size_t) chunk * (e.embedding * 2 + e.hidden * 9 + 4)) * sizeof(float));
    auto * gate = ggml_new_tensor_2d(s.ctx.get(), precision, e.embedding, e.hidden);
    auto * up = ggml_new_tensor_2d(s.ctx.get(), precision, e.embedding, e.hidden);
    auto * down = hidden_only ? nullptr : ggml_new_tensor_2d(s.ctx.get(), precision, e.hidden, e.embedding);
    s.input = ggml_new_tensor_2d(s.ctx.get(), precision, e.embedding, chunk);
    ggml_set_name(s.input, "ream_input");
    auto set_weights = [&](ggml_tensor * tensor, const std::vector<float> & values) {
        if (precision == GGML_TYPE_F16) {
            for (float value : values) check(std::abs(value) <= 65504, "Expert weight overflows F16; use f32 features");
            ggml_fp32_to_fp16_row(values.data(), (ggml_fp16_t *) tensor->data, values.size());
        } else memcpy(tensor->data, values.data(), values.size() * sizeof(float));
    };
    set_weights(gate, e.gate); set_weights(up, e.up); if (down) set_weights(down, e.down);
    memset(s.input->data, 0, ggml_nbytes(s.input));
    auto * gate_out = ggml_mul_mat(s.ctx.get(), gate, s.input);
    auto * up_out = ggml_mul_mat(s.ctx.get(), up, s.input);
    s.hidden = ggml_mul(s.ctx.get(), ggml_gelu(s.ctx.get(), gate_out), up_out);
    s.output = hidden_only ? nullptr : ggml_mul_mat(s.ctx.get(), down, s.hidden);
    for (auto * projection : {gate_out, up_out, s.output}) {
        if (!projection) continue;
        ggml_prec_set_acc(projection, GGML_PREC_F32);
        ggml_prec_set_src(projection, precision == GGML_TYPE_F16 ? GGML_PREC_F16 : GGML_PREC_F32, 0);
        ggml_prec_set_src(projection, precision == GGML_TYPE_F16 ? GGML_PREC_F16 : GGML_PREC_F32, 1);
    }
    s.graph = ggml_new_graph(s.ctx.get());
    if (summarize) {
        s.weights = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, 1, chunk);
        s.sample_ids = ggml_new_tensor_1d(s.ctx.get(), GGML_TYPE_I32, chunk);
        ggml_set_name(s.weights, "ream_probabilities"); ggml_set_name(s.sample_ids, "ream_sample_ids");
        memset(s.weights->data, 0, ggml_nbytes(s.weights)); memset(s.sample_ids->data, 0, ggml_nbytes(s.sample_ids));
        auto * weighted = ggml_mul(s.ctx.get(), s.hidden, s.weights);
        s.weighted_sum = ggml_sum_rows(s.ctx.get(), ggml_cont(s.ctx.get(), ggml_transpose(s.ctx.get(), weighted)));
        s.sampled = ggml_get_rows(s.ctx.get(), s.hidden, s.sample_ids);
        ggml_build_forward_expand(s.graph, s.weighted_sum);
        ggml_build_forward_expand(s.graph, s.sampled);
    } else ggml_build_forward_expand(s.graph, hidden_only ? s.hidden : s.output);
    if (s.backend) {
        for (int i = 0; i < ggml_graph_n_nodes(s.graph); ++i) {
            check(ggml_backend_supports_op(s.backend, ggml_graph_node(s.graph, i)), "compute device does not support expert graph operation");
        }
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(s.backend), &free_bytes, &total_bytes);
        check(free_bytes > ggml_get_mem_size(s.ctx.get()), "insufficient device memory for reusable expert graph");
        s.host = ggml_backend_cpu_buffer_from_ptr(ggml_get_mem_buffer(s.ctx.get()), ggml_get_mem_size(s.ctx.get()));
        check(s.host != nullptr, "cannot register expert graph host buffer");
        for (auto * t = ggml_get_first_tensor(s.ctx.get()); t; t = ggml_get_next_tensor(s.ctx.get(), t)) t->buffer = s.host;
        s.device = ggml_backend_graph_copy(s.backend, s.graph);
        check(s.device.buffer && s.device.graph, "cannot allocate reusable expert graph");
        s.device_input = ggml_get_tensor(s.device.ctx_allocated, "ream_input");
        if (summarize) {
            s.device_weights = ggml_get_tensor(s.device.ctx_allocated, "ream_probabilities");
            s.device_sample_ids = ggml_get_tensor(s.device.ctx_allocated, "ream_sample_ids");
        }
        for (int i = 0; i < ggml_graph_n_nodes(s.graph); ++i) {
            if (ggml_graph_node(s.graph, i) == s.hidden) s.device_hidden = ggml_graph_node(s.device.graph, i);
            if (summarize && ggml_graph_node(s.graph, i) == s.weighted_sum) s.device_weighted_sum = ggml_graph_node(s.device.graph, i);
            if (summarize && ggml_graph_node(s.graph, i) == s.sampled) s.device_sampled = ggml_graph_node(s.device.graph, i);
            if (ggml_graph_node(s.graph, i) == s.output) s.device_output = ggml_graph_node(s.device.graph, i);
        }
        check(s.device_input && s.device_hidden && (hidden_only || s.device_output), "missing reusable graph tensors");
    }
}

aikar_ream_forward_runner::~aikar_ream_forward_runner() = default;

void aikar_ream_forward_runner::run(const std::vector<float> & inputs, std::vector<float> & hidden, std::vector<float> & outputs) {
    set_input(inputs);
    compute(inputs.size() / state->embedding, hidden, outputs);
}

void aikar_ream_forward_runner::set_input(const std::vector<float> & inputs) {
    auto & s = *state;
    finite(inputs);
    check(!inputs.empty() && inputs.size() % s.embedding == 0 && inputs.size() <= (size_t) s.embedding * s.chunk, "invalid reusable expert input shape");
    const size_t input_bytes = inputs.size() * ggml_type_size(s.input->type);
    if (s.input->type == GGML_TYPE_F16) {
        for (float value : inputs) check(std::abs(value) <= 65504, "Expert input overflows F16; use f32 features");
        ggml_fp32_to_fp16_row(inputs.data(), (ggml_fp16_t *) s.input->data, inputs.size());
    } else memcpy(s.input->data, inputs.data(), input_bytes);
    memset((char *) s.input->data + input_bytes, 0, ggml_nbytes(s.input) - input_bytes);
    if (s.backend) ggml_backend_tensor_set(s.device_input, s.input->data, 0, ggml_nbytes(s.input));
}

void aikar_ream_forward_runner::run_device(const ggml_tensor * inputs, uint64_t offset, int32_t tokens, std::vector<float> & hidden, std::vector<float> & outputs) {
    set_device_input(inputs, offset, tokens);
    compute(tokens, hidden, outputs);
}

void aikar_ream_forward_runner::set_device_input(const ggml_tensor * inputs, uint64_t offset, int32_t tokens) {
    auto & s = *state;
    check(s.backend && inputs && inputs->buffer && inputs->type == s.input->type && inputs->ne[0] == s.embedding, "invalid cached Expert input");
    check(tokens > 0 && tokens <= s.chunk && offset <= (uint64_t) inputs->ne[1] && (uint64_t) s.chunk <= (uint64_t) inputs->ne[1] - offset, "invalid cached Expert chunk");
    ggml_tensor view = *s.device_input;
    view.buffer = inputs->buffer;
    view.data = (char *) inputs->data + offset * inputs->nb[1];
    ggml_backend_tensor_copy(&view, s.device_input);
}

void aikar_ream_forward_runner::run_summary(const ggml_tensor * cached, uint64_t offset, const std::vector<float> & inputs, const std::vector<float> & probabilities, const std::vector<int32_t> & samples, std::vector<float> & hidden, std::vector<float> & weighted_sum) {
    auto & s = *state;
    check(s.backend && s.device_weighted_sum && s.device_sampled, "Expert summary graph is unavailable");
    check(!probabilities.empty() && probabilities.size() <= (size_t) s.chunk && samples.size() <= (size_t) s.chunk, "invalid Expert summary size");
    for (float p : probabilities) check(std::isfinite(p) && p >= 0 && p <= 1, "invalid Expert summary probability");
    for (int32_t id : samples) check(id >= 0 && (size_t) id < probabilities.size(), "invalid Expert sample index");
    if (cached) set_device_input(cached, offset, probabilities.size());
    else {
        check(inputs.size() == probabilities.size() * s.embedding, "Expert summary input size differs");
        set_input(inputs);
    }
    auto * weights = (float *) s.weights->data;
    std::copy(probabilities.begin(), probabilities.end(), weights);
    std::fill(weights + probabilities.size(), weights + s.chunk, 0);
    auto * ids = (int32_t *) s.sample_ids->data;
    std::copy(samples.begin(), samples.end(), ids);
    std::fill(ids + samples.size(), ids + s.chunk, 0);
    ggml_backend_tensor_set(s.device_weights, weights, 0, ggml_nbytes(s.weights));
    ggml_backend_tensor_set(s.device_sample_ids, ids, 0, ggml_nbytes(s.sample_ids));
    check(ggml_backend_graph_compute(s.backend, s.device.graph) == GGML_STATUS_SUCCESS, "Expert summary graph failed");
    weighted_sum.resize(s.neurons); hidden.resize(samples.size() * s.neurons);
    ggml_backend_tensor_get(s.device_weighted_sum, weighted_sum.data(), 0, weighted_sum.size() * sizeof(float));
    if (!hidden.empty()) ggml_backend_tensor_get(s.device_sampled, hidden.data(), 0, hidden.size() * sizeof(float));
    finite(weighted_sum); finite(hidden);
}

void aikar_ream_forward_runner::compute(int32_t tokens, std::vector<float> & hidden, std::vector<float> & outputs) {
    auto & s = *state;
    hidden.resize((size_t) tokens * s.neurons); outputs.resize(s.hidden_only ? 0 : (size_t) tokens * s.embedding);
    if (s.backend) {
        check(ggml_backend_graph_compute(s.backend, s.device.graph) == GGML_STATUS_SUCCESS, "reusable device expert graph failed");
        ggml_backend_tensor_get(s.device_hidden, hidden.data(), 0, hidden.size() * sizeof(float));
        if (!s.hidden_only) ggml_backend_tensor_get(s.device_output, outputs.data(), 0, outputs.size() * sizeof(float));
    } else {
        check(ggml_graph_compute_with_ctx(s.ctx.get(), s.graph, s.threads) == GGML_STATUS_SUCCESS, "reusable CPU expert graph failed");
        memcpy(hidden.data(), s.hidden->data, hidden.size() * sizeof(float));
        if (!s.hidden_only) memcpy(outputs.data(), s.output->data, outputs.size() * sizeof(float));
    }
    finite(hidden); finite(outputs);
}

std::vector<float> aikar_ream_project_mean(const aikar_ream_expert & e, const std::vector<double> & hidden_mean) {
    check(hidden_mean.size() == (size_t) e.hidden && e.down.size() == (size_t) e.hidden * e.embedding, "invalid hidden mean shape");
    for (double x : hidden_mean) check(std::isfinite(x), "non-finite hidden mean");
    std::vector<float> output(e.embedding);
    for (int32_t d = 0; d < e.embedding; ++d) {
        double sum = 0;
        for (int32_t h = 0; h < e.hidden; ++h) sum += (double) e.down[d * e.hidden + h] * hidden_mean[h];
        output[d] = sum;
    }
    finite(output);
    return output;
}

namespace {
bool suffix(const std::string & s, const std::string & end) {
    return s.size() >= end.size() && s.compare(s.size() - end.size(), end.size(), end) == 0;
}
int32_t layer_of(const std::string & name) {
    int32_t layer = -1;
    return sscanf(name.c_str(), "blk.%d.", &layer) == 1 ? layer : -1;
}
bool expert_weight(const std::string & n) {
    return suffix(n, ".ffn_gate_up_exps.weight") || suffix(n, ".ffn_gate_exps.weight") ||
           suffix(n, ".ffn_up_exps.weight") || suffix(n, ".ffn_down_exps.weight");
}
bool supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
            return true;
        default: return false;
    }
}
struct source_file {
    ctx_ptr tensors;
    file_ptr meta;
    std::ifstream input;
    explicit source_file(const std::string & path) : input(path, std::ios::binary) {
        ggml_context * raw = nullptr;
        meta.reset(gguf_init_from_file(path.c_str(), {true, &raw}));
        tensors.reset(raw);
        check(meta != nullptr && input.good(), "cannot read source GGUF: " + path);
        const auto split = gguf_find_key(meta.get(), "split.count");
        check(split < 0 || gguf_get_val_u16(meta.get(), split) == 1, "split GGUF is unsupported");
    }
    uint32_t integer(const char * key) {
        const auto id = gguf_find_key(meta.get(), key);
        check(id >= 0 && gguf_get_kv_type(meta.get(), id) == GGUF_TYPE_UINT32, std::string("missing or invalid metadata: ") + key);
        return gguf_get_val_u32(meta.get(), id);
    }
    ggml_tensor * tensor(const std::string & name) { return ggml_get_tensor(tensors.get(), name.c_str()); }
    std::vector<uint8_t> bytes(const std::string & name, int32_t slice, size_t size) {
        const auto id = gguf_find_tensor(meta.get(), name.c_str());
        check(id >= 0 && slice >= 0 && size > 0, "missing tensor: " + name);
        const size_t total = gguf_get_tensor_size(meta.get(), id);
        check((uint64_t) (slice + 1) * size <= total, "tensor slice out of bounds: " + name);
        std::vector<uint8_t> result(size);
        input.clear();
        input.seekg(gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), id) + (uint64_t) slice * size);
        input.read((char *) result.data(), size);
        check((size_t) input.gcount() == size, "short source tensor read");
        return result;
    }
    std::vector<float> floats(const std::string & name, int32_t expert) {
        auto * t = tensor(name);
        check(t != nullptr && supported(t->type), "unsupported tensor: " + name);
        const size_t size = ggml_row_size(t->type, t->ne[0]) * t->ne[1];
        const auto data = bytes(name, expert, size);
        std::vector<float> values(t->ne[0] * t->ne[1]);
        if (t->type == GGML_TYPE_F32) memcpy(values.data(), data.data(), data.size());
        else {
            const auto * traits = ggml_get_type_traits(t->type);
            check(traits->to_float != nullptr, "missing dequantizer");
            for (int64_t row = 0; row < t->ne[1]; ++row) {
                traits->to_float(data.data() + row * ggml_row_size(t->type, t->ne[0]), values.data() + row * t->ne[0], t->ne[0]);
            }
        }
        finite(values);
        return values;
    }
};
source_file & cached_source(const std::string & path) {
    static std::string current;
    static std::unique_ptr<source_file> source;
    if (!source || current != path) { source.reset(new source_file(path)); current = path; }
    return *source;
}
}

aikar_ream_layout aikar_ream_inspect(const std::string & path, int32_t target, int32_t samples, int32_t chunk) {
    source_file s(path);
    const auto arch = gguf_find_key(s.meta.get(), "general.architecture");
    check(arch >= 0 && std::string(gguf_get_val_str(s.meta.get(), arch)) == "gemma4", "only Gemma4 GELU routed experts are supported");
    aikar_ream_layout layout;
    layout.experts = s.integer("gemma4.expert_count");
    layout.top_k = s.integer("gemma4.expert_used_count");
    const int32_t blocks = s.integer("gemma4.block_count");
    const auto mtp = gguf_find_key(s.meta.get(), "gemma4.nextn_predict_layers");
    check(mtp < 0 || (gguf_get_kv_type(s.meta.get(), mtp) == GGUF_TYPE_UINT32 && gguf_get_val_u32(s.meta.get(), mtp) == 0),
          "MTP metadata requires an unsupported forward path");
    check(target >= layout.top_k && target <= layout.experts && layout.top_k > 0, "target must be between Top-K and source expert count");
    check(samples > 0 && chunk > 0, "activation sample and chunk counts must be positive");
    std::set<int32_t> layers;
    for (int64_t i = 0; i < gguf_get_n_tensors(s.meta.get()); ++i) {
        const std::string name = gguf_get_tensor_name(s.meta.get(), i);
        const auto * t = s.tensor(name);
        check(name.find("mtp") == std::string::npos && name.find("nextn") == std::string::npos,
              "MTP tensor requires an unsupported forward path: " + name);
        if (name.find("_exps.") != std::string::npos || suffix(name, ".ffn_gate_inp.weight")) {
            check(supported(t->type) && t->ne[0] % ggml_blck_size(t->type) == 0, "unsupported format or unaligned row: " + name);
            check(expert_weight(name) || suffix(name, ".ffn_down_exps.scale") || suffix(name, ".ffn_gate_inp.weight"),
                  "unsupported expert tensor or bias: " + name);
            if (expert_weight(name)) {
                check(t->ne[2] == layout.experts && t->ne[3] == 1 && ggml_is_contiguous(t), "invalid expert axis: " + name);
                layout.expert_bytes += ggml_nbytes(t);
            }
            if (suffix(name, ".ffn_down_exps.scale")) {
                check(t->ne[0] == layout.experts && t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1 && !ggml_is_quantized(t->type), "invalid expert scale");
            }
            if (suffix(name, ".ffn_gate_inp.weight")) {
                check(t->ne[1] == layout.experts && t->ne[2] == 1 && t->ne[3] == 1, "invalid router shape");
                layers.insert(layer_of(name));
            }
        }
        check(!suffix(name, ".ffn_gate_inp.bias") && name.find("ffn_exp_probs_b") == std::string::npos,
              "Gemma4 router bias or correction bias is unsupported: " + name);
    }
    check(!layers.empty(), "no routed MoE layers");
    for (int32_t layer : layers) {
        check(layer >= 0 && layer < blocks, "expert layer outside model blocks; MTP is unsupported");
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const auto * router = s.tensor(prefix + "ffn_gate_inp.weight");
        const auto * fused = s.tensor(prefix + "ffn_gate_up_exps.weight");
        const auto * gate = s.tensor(prefix + "ffn_gate_exps.weight");
        const auto * up = s.tensor(prefix + "ffn_up_exps.weight");
        const auto * down = s.tensor(prefix + "ffn_down_exps.weight");
        check(down && (fused || (gate && up)), "missing FFN projections");
        const int32_t embedding = router->ne[0], hidden = down->ne[0];
        check(down->ne[1] == embedding, "down projection shape differs");
        if (fused) {
            check(!gate && !up && fused->ne[0] == embedding && fused->ne[1] == hidden * 2, "invalid fused gate/up shape");
        } else {
            check(gate->ne[0] == embedding && up->ne[0] == embedding && gate->ne[1] == hidden && up->ne[1] == hidden,
                  "invalid separate gate/up shape");
        }
        check(layout.embedding == 0 || (layout.embedding == embedding && layout.hidden == hidden), "heterogeneous FFN dimensions are unsupported");
        layout.embedding = embedding; layout.hidden = hidden;
    }
    layout.layers.assign(layers.begin(), layers.end());
    layout.source_bytes = std::filesystem::file_size(path);
    layout.workspace_bytes = sizeof(float) * ((uint64_t) layout.hidden * layout.embedding * 30 +
        (uint64_t) layout.hidden * samples * 6 + (uint64_t) layout.hidden * layout.hidden * 4 +
        (uint64_t) chunk * (layout.embedding * 4 + layout.hidden * 10 + 4) + (uint64_t) layout.experts * layout.experts * 4) + 16 * 1024 * 1024;
    return layout;
}

aikar_ream_expert aikar_ream_read_expert(const std::string & source, int32_t layer, int32_t expert) {
    auto & s = cached_source(source);
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto * down = s.tensor(prefix + "ffn_down_exps.weight");
    check(down && expert >= 0 && expert < down->ne[2], "invalid expert read");
    aikar_ream_expert e;
    e.embedding = down->ne[1]; e.hidden = down->ne[0];
    if (s.tensor(prefix + "ffn_gate_up_exps.weight")) {
        const auto fused = s.floats(prefix + "ffn_gate_up_exps.weight", expert);
        const size_t half = (size_t) e.hidden * e.embedding;
        check(fused.size() == half * 2, "invalid fused projection");
        e.gate.assign(fused.begin(), fused.begin() + half);
        e.up.assign(fused.begin() + half, fused.end());
    } else {
        e.gate = s.floats(prefix + "ffn_gate_exps.weight", expert);
        e.up = s.floats(prefix + "ffn_up_exps.weight", expert);
    }
    e.down = s.floats(prefix + "ffn_down_exps.weight", expert);
    if (s.tensor(prefix + "ffn_down_exps.scale")) {
        const auto scales = s.floats(prefix + "ffn_down_exps.scale", 0);
        for (auto & x : e.down) x *= scales.at(expert);
    }
    validate_expert(e);
    return e;
}

std::vector<uint8_t> aikar_ream_encode(const std::vector<float> & values, ggml_type type, int64_t row) {
    finite(values);
    check(supported(type) && row > 0 && row % ggml_blck_size(type) == 0 && values.size() % row == 0, "unsupported output format or row shape");
    std::vector<uint8_t> bytes(ggml_row_size(type, row) * (values.size() / row));
    if (type == GGML_TYPE_F32) memcpy(bytes.data(), values.data(), bytes.size());
    else {
        const auto * traits = ggml_get_type_traits(type);
        check(traits->from_float_ref && traits->to_float, "quantization format requires an unsupported importance matrix");
        std::vector<float> decoded(row);
        for (size_t i = 0; i < values.size() / row; ++i) {
            auto * dest = bytes.data() + i * ggml_row_size(type, row);
            traits->from_float_ref(values.data() + i * row, dest, row);
            traits->to_float(dest, decoded.data(), row);
            finite(decoded);
        }
    }
    return bytes;
}

void aikar_ream_export(const std::string & source, const std::string & path,
                       const std::map<int32_t, aikar_ream_groups> & groups, const std::string & provenance,
                       const aikar_ream_slice_reader & reader) {
    check(!groups.empty(), "no groups to export");
    for (const auto & p : {path, path + ".ream.tmp", path + ".ream.tmp.report.json", path + ".report.json"}) {
        check(!std::filesystem::exists(p) || !std::filesystem::equivalent(source, p), "output aliases the source GGUF");
        check(!std::filesystem::exists(std::filesystem::symlink_status(p)), "output or staging path already exists: " + p);
    }
    source_file s(source);
    const int32_t experts = s.integer("gemma4.expert_count");
    const int32_t target = groups.begin()->second.centers.size();
    aikar_ream_inspect(source, target, 1, 1);
    nlohmann::ordered_json mapping = nlohmann::ordered_json::object();
    for (const auto & layer : groups) {
        aikar_ream_validate_groups(layer.second, experts, target);
        auto centers = layer.second.centers;
        std::sort(centers.begin(), centers.end());
        std::vector<int32_t> old_to_new(experts);
        for (int32_t e = 0; e < experts; ++e) {
            const int32_t center = layer.second.centers[layer.second.labels[e]];
            old_to_new[e] = std::lower_bound(centers.begin(), centers.end(), center) - centers.begin();
        }
        mapping[std::to_string(layer.first)] = {{"centers", centers}, {"original_to_new", old_to_new}, {"groups", layer.second.members}};
    }
    file_ptr output(gguf_init_empty());
    gguf_set_kv(output.get(), s.meta.get());
    gguf_set_val_u32(output.get(), "gemma4.expert_count", target);
    gguf_set_val_str(output.get(), "aikar.ream.mapping", mapping.dump().c_str());
    gguf_set_val_str(output.get(), "aikar.ream.provenance", provenance.c_str());
    ctx_ptr tensors(ggml_init({std::max<size_t>(4 * 1024 * 1024, gguf_get_n_tensors(s.meta.get()) * ggml_tensor_overhead() * 2), nullptr, true}));
    check(tensors != nullptr, "cannot allocate GGUF directory");
    struct plan { std::string name; std::vector<int32_t> centers; size_t slice = 0; };
    std::vector<plan> plans;
    std::set<int32_t> routed;
    for (int64_t i = 0; i < gguf_get_n_tensors(s.meta.get()); ++i) {
        const std::string name = gguf_get_tensor_name(s.meta.get(), i);
        const auto * old = s.tensor(name);
        int64_t ne[4] = {old->ne[0], old->ne[1], old->ne[2], old->ne[3]};
        const int32_t layer = layer_of(name);
        const auto found = groups.find(layer);
        plan p {name, {}, 0};
        if (expert_weight(name) || suffix(name, ".ffn_gate_inp.weight") || suffix(name, ".ffn_down_exps.scale")) {
            check(found != groups.end(), "missing routed layer group mapping");
            p.centers = found->second.centers;
            std::sort(p.centers.begin(), p.centers.end());
            if (expert_weight(name)) { p.slice = ggml_row_size(old->type, ne[0]) * ne[1]; ne[2] = target; }
            else if (suffix(name, ".ffn_gate_inp.weight")) {
                p.slice = ggml_row_size(old->type, ne[0]); ne[1] = target; routed.insert(layer);
            } else { p.slice = ggml_type_size(old->type); ne[0] = target; }
        }
        auto * t = ggml_new_tensor(tensors.get(), old->type, ggml_n_dims(old), ne);
        ggml_set_name(t, name.c_str()); gguf_add_tensor(output.get(), t);
        plans.push_back(std::move(p));
    }
    check(routed.size() == groups.size(), "group mapping contains nonexistent layers");
    const std::string staging = path + ".ream.tmp";
    struct cleanup {
        std::string path;
        ~cleanup() { std::remove(path.c_str()); std::remove((path + ".report.json").c_str()); }
    } guard {staging};
    check(gguf_write_to_file(output.get(), staging.c_str(), true), "cannot write GGUF metadata");
    std::ofstream out(staging, std::ios::binary | std::ios::app);
    uint64_t cursor = 0;
    std::vector<char> buffer(4 * 1024 * 1024);
    for (size_t i = 0; i < plans.size(); ++i) {
        const uint64_t offset = gguf_get_tensor_offset(output.get(), i);
        check(offset >= cursor, "GGUF offsets overlap");
        std::vector<char> pad(offset - cursor, 0); out.write(pad.data(), pad.size());
        const auto & p = plans[i];
        if (p.centers.empty()) {
            s.input.clear(); s.input.seekg(gguf_get_data_offset(s.meta.get()) + gguf_get_tensor_offset(s.meta.get(), i));
            uint64_t remaining = gguf_get_tensor_size(s.meta.get(), i);
            while (remaining) {
                const size_t count = std::min<uint64_t>(remaining, buffer.size());
                s.input.read(buffer.data(), count); check((size_t) s.input.gcount() == count, "short unchanged tensor read");
                out.write(buffer.data(), count); remaining -= count;
            }
        } else for (int32_t e : p.centers) {
            const auto bytes = reader ? reader(p.name, e, p.slice) : s.bytes(p.name, e, p.slice);
            check(bytes.size() == p.slice, "replacement slice size differs");
            out.write((const char *) bytes.data(), bytes.size());
        }
        if (!out.good()) {
            const int error = errno;
            throw std::runtime_error("REAM: output write failed at tensor " + p.name + ", offset=" + std::to_string(offset) +
                ", errno=" + std::to_string(error) + " (" + std::strerror(error) + ")");
        }
        cursor = offset + gguf_get_tensor_size(output.get(), i);
    }
    std::vector<char> padding((gguf_get_alignment(output.get()) - cursor % gguf_get_alignment(output.get())) % gguf_get_alignment(output.get()), 0);
    out.write(padding.data(), padding.size()); out.close(); check(out.good(), "failed to flush output GGUF");
    source_file reload(staging);
    check(reload.integer("gemma4.expert_count") == (uint32_t) target && reload.integer("gemma4.expert_used_count") == (uint32_t) s.integer("gemma4.expert_used_count"),
          "saved expert count or Top-K differs");
    const uint64_t data_end = gguf_get_data_offset(reload.meta.get()) + cursor;
    check(std::filesystem::file_size(staging) >= data_end, "truncated output GGUF");
    nlohmann::ordered_json report = {{"format", "aikar-ream-report"}, {"version", 1}, {"source", source},
        {"output", path}, {"source_experts", experts}, {"target_experts", target}, {"top_k", s.integer("gemma4.expert_used_count")},
        {"source_bytes", std::filesystem::file_size(source)}, {"output_bytes", std::filesystem::file_size(staging)},
        {"layers", mapping}, {"provenance", nlohmann::ordered_json::parse(provenance)}};
    std::ofstream report_out(staging + ".report.json"); report_out << report.dump(2) << '\n'; report_out.close();
    check(report_out.good(), "cannot write merge report");
    aikar_hard_prune_publish(staging, path);
}
