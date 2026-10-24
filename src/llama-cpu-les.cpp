#include "llama-cpu-les.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

static bool llama_cpu_les_host_tensor(const ggml_tensor * tensor) {
    if (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer)) { return true; }
    auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
    return dev && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CPU") == 0;
}

struct llama_cpu_les::impl {
    struct layer_state {
        std::vector<uint8_t> ready;
        std::vector<uint8_t> predicted;
    };
    struct job {
        size_t layer;
        uint64_t generation;
        uint64_t serial;
        bool prediction;
        std::vector<int32_t> experts;
        std::vector<float> hidden;
    };
    struct callback_data {
        llama_cpu_les * owner;
        size_t layer;
    };

    std::vector<llama_cpu_les_layer> layers;
    std::vector<layer_state> states;
    std::vector<callback_data> callbacks;
    int n_selected;
    size_t budget;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable drained;
    std::deque<job> jobs;
    std::thread worker;
    std::atomic<uint64_t> serial{0};
    uint64_t generation = 0;
    bool stopping = false;
    bool busy = false;
    llama_cpu_les_stats counters;

    impl(std::vector<llama_cpu_les_layer> input, int selected, size_t bytes) :
        layers(std::move(input)), n_selected(selected), budget(bytes) {
        if (layers.empty() || layers.size() > 4096 || !budget || n_selected <= 0) {
            throw std::invalid_argument("invalid CPU LES configuration");
        }
        for (const auto & layer : layers) {
            const auto * router = layer.router;
            if (!router || router->type != GGML_TYPE_F32 || !ggml_is_contiguous(router) ||
                    router->ne[0] <= 0 || router->ne[0] > 65536 || router->ne[1] < n_selected ||
                    router->ne[1] > 4096 || router->ne[2] != 1 || router->ne[3] != 1 ||
                    !router->data || !llama_cpu_les_host_tensor(router) ||
                    !std::isfinite(layer.epsilon) || layer.epsilon <= 0 ||
                    !std::isfinite(layer.input_scale) || layer.input_scale <= 0) {
                throw std::invalid_argument("CPU LES requires contiguous host F32 routers");
            }
            if (layer.router_scale && (layer.router_scale->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(layer.router_scale) ||
                    ggml_nelements(layer.router_scale) != router->ne[0] || !layer.router_scale->data ||
                    !llama_cpu_les_host_tensor(layer.router_scale))) {
                throw std::invalid_argument("CPU LES requires host F32 router scales");
            }
            if (layer.experts.empty()) {
                throw std::invalid_argument("CPU LES requires expert weights");
            }
            for (const auto * tensor : layer.experts) {
                if (!tensor || !tensor->data || !ggml_is_contiguous(tensor) ||
                        tensor->ne[2] != router->ne[1] || tensor->ne[3] != 1 ||
                        !llama_cpu_les_host_tensor(tensor) ||
                        tensor->nb[2] == 0 || ggml_nbytes(tensor)/tensor->nb[2] < size_t(router->ne[1])) {
                    throw std::invalid_argument(std::string("CPU LES requires contiguous CPU expert weights: ") + (tensor ? ggml_get_name(tensor) : "missing"));
                }
            }
            states.push_back({std::vector<uint8_t>(router->ne[1]), std::vector<uint8_t>(router->ne[1])});
        }
        callbacks.resize(layers.size());
        worker = std::thread([this] { run(); });
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ++serial;
            jobs.clear();
        }
        wake.notify_one();
        worker.join();
    }

    bool current(const job & work) const {
        return serial.load(std::memory_order_relaxed) == work.serial;
    }

    std::vector<int32_t> predict(const job & work) {
        const auto & layer = layers[work.layer];
        const size_t width = layer.router->ne[0];
        if (work.hidden.size() != width) {
            return {};
        }
        double squares = 0;
        for (float x : work.hidden) { squares += double(x)*x; }
        if (!std::isfinite(squares)) { return {}; }
        const float norm = layer.input_scale/std::sqrt(squares/width + layer.epsilon);
        const float * scale = layer.router_scale ? (const float *) layer.router_scale->data : nullptr;
        std::vector<float> input(width);
        for (size_t i = 0; i < width; ++i) {
            input[i] = work.hidden[i]*norm*(scale ? scale[i] : 1.0f);
        }
        std::vector<float> logits(layer.router->ne[1]);
        std::vector<int32_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0);
        for (size_t expert = 0; expert < logits.size(); ++expert) {
            if (!current(work)) { return {}; }
            const auto * row = (const float *) ((const char *) layer.router->data + expert*layer.router->nb[1]);
            float dot = 0;
#ifdef GGML_USE_CPU
            const auto * traits = ggml_get_type_traits_cpu(GGML_TYPE_F32);
            traits->vec_dot(int(width), &dot, 0, input.data(), 0, row, 0, 1);
#else
            for (size_t i = 0; i < width; ++i) { dot += input[i]*row[i]; }
#endif
            if (!std::isfinite(dot)) { return {}; }
            logits[expert] = dot;
        }
        std::partial_sort(order.begin(), order.begin() + n_selected, order.end(), [&](int a, int b) {
            return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
        });
        order.resize(n_selected);
        return order;
    }

    void process(job & work) {
        if (!current(work)) { return; }
        if (work.prediction) {
            work.experts = predict(work);
            std::lock_guard<std::mutex> lock(mutex);
            if (!current(work) || generation != work.generation) { return; }
            for (int expert : work.experts) { states[work.layer].predicted[expert] = 1; }
            counters.predicted_experts += work.experts.size();
        }
        size_t remaining = budget;
        for (int expert : work.experts) {
            if (!current(work)) { break; }
            size_t required = 0;
            bool fits = true;
            for (const auto * tensor : layers[work.layer].experts) {
                if (tensor->nb[2] > remaining - required) { fits = false; break; }
                required += tensor->nb[2];
            }
            if (!fits) { continue; }
            size_t touched = 0;
            bool complete = true;
            for (const auto * tensor : layers[work.layer].experts) {
                const auto * data = (const volatile uint8_t *) tensor->data + size_t(expert)*tensor->nb[2];
                for (size_t offset = 0; offset < tensor->nb[2]; offset += 64) {
                    if ((offset % 65536) == 0 && !current(work)) { complete = false; break; }
                    (void) data[offset];
                    touched += std::min(size_t(64), tensor->nb[2] - offset);
                }
                if (!complete) { break; }
            }
            remaining -= touched;
            std::lock_guard<std::mutex> lock(mutex);
            counters.prepared_bytes += touched;
            if (complete && current(work) && generation == work.generation) {
                states[work.layer].ready[expert] = 1;
            }
        }
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            wake.wait(lock, [&] { return stopping || !jobs.empty(); });
            if (stopping) { return; }
            job work = std::move(jobs.front());
            jobs.pop_front();
            busy = true;
            lock.unlock();
            process(work);
            lock.lock();
            if (!current(work)) { ++counters.cancelled_jobs; }
            busy = false;
            if (jobs.empty()) { drained.notify_all(); }
        }
    }

    static void callback(ggml_tensor * dst, const ggml_tensor *, const ggml_tensor * ids,
            const ggml_tensor * hidden, int ith, int, void * userdata) {
        if (ith != 0) { return; }
        const auto & data = *(const callback_data *) userdata;
        data.owner->prepare(data.layer, (const int32_t *) ids->data, ids->ne[0],
                (const float *) hidden->data, hidden->ne[0], (float *) dst->data);
    }
};

llama_cpu_les::llama_cpu_les(std::vector<llama_cpu_les_layer> layers, int n_selected, size_t prefetch_bytes) :
    pimpl(std::make_unique<impl>(std::move(layers), n_selected, prefetch_bytes)) {
    for (size_t layer = 0; layer < pimpl->callbacks.size(); ++layer) {
        pimpl->callbacks[layer] = {this, layer};
    }
}

llama_cpu_les::~llama_cpu_les() = default;

void llama_cpu_les::prepare(size_t layer, const int32_t * selected, size_t n_selected,
        const float * hidden, size_t n_hidden, float * priority) {
    if (layer >= pimpl->layers.size() || !selected || !hidden || !priority ||
            n_hidden != size_t(pimpl->layers[layer].router->ne[0]) || n_selected > size_t(pimpl->layers[layer].router->ne[1])) {
        throw std::invalid_argument("invalid CPU LES routing input");
    }
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (layer == 0) {
        ++pimpl->generation;
        for (auto & state : pimpl->states) {
            std::fill(state.ready.begin(), state.ready.end(), 0);
            std::fill(state.predicted.begin(), state.predicted.end(), 0);
        }
    }
    const auto & state = pimpl->states[layer];
    std::copy(state.ready.begin(), state.ready.end(), priority);
    std::vector<int32_t> demand;
    std::vector<uint8_t> seen(state.ready.size());
    for (size_t i = 0; i < n_selected; ++i) {
        const int32_t expert = selected[i];
        if (expert < 0 || size_t(expert) >= state.ready.size()) {
            throw std::invalid_argument("CPU LES expert index out of bounds");
        }
        if (seen[expert]) { continue; }
        seen[expert] = 1;
        ++pimpl->counters.required_experts;
        pimpl->counters.prediction_matches += state.predicted[expert] != 0;
        pimpl->counters.ready_hits += state.ready[expert] != 0;
        if (!state.ready[expert]) { demand.push_back(expert); }
    }
    std::sort(demand.begin(), demand.end());
    pimpl->counters.cancelled_jobs += pimpl->jobs.size();
    pimpl->jobs.clear();
    const uint64_t serial = ++pimpl->serial;
    if (layer + 1 < pimpl->layers.size()) {
        pimpl->jobs.push_back({layer + 1, pimpl->generation, serial, true, {}, std::vector<float>(hidden, hidden + n_hidden)});
    }
    if (!demand.empty()) {
        pimpl->jobs.push_back({layer, pimpl->generation, serial, false, std::move(demand), {}});
    }
    pimpl->wake.notify_one();
}

void llama_cpu_les::synchronize() {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    pimpl->drained.wait(lock, [&] { return !pimpl->busy && pimpl->jobs.empty(); });
}

llama_cpu_les_stats llama_cpu_les::stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}

ggml_tensor * llama_cpu_les::build_priority(ggml_context * ctx, ggml_tensor * logits,
        ggml_tensor * selected, ggml_tensor * hidden, size_t layer) {
    if (layer >= pimpl->layers.size() || hidden->type != GGML_TYPE_F32 ||
            logits->type != GGML_TYPE_F32 || selected->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(logits) || !ggml_is_contiguous(selected) || !ggml_is_contiguous(hidden) ||
            hidden->ne[0] != pimpl->layers[layer].router->ne[0] ||
            logits->ne[0] != pimpl->layers[layer].router->ne[1] ||
            ggml_nelements(hidden) != hidden->ne[0] || ggml_nelements(logits) != logits->ne[0] ||
            ggml_nelements(selected) != selected->ne[0]) {
        return nullptr;
    }
    return ggml_map_custom3(ctx, logits, selected, hidden, impl::callback, 1, &pimpl->callbacks[layer]);
}
