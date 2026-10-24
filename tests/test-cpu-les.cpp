#include "llama-cpu-les.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-cpu/ggml-cpu-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>

static void require(bool value) {
    if (!value) {
        std::fprintf(stderr, "CPU LES test failed\n");
        std::abort();
    }
}

struct fixture {
    ggml_context * ctx = ggml_init({1024*1024, nullptr, false});
    std::vector<llama_cpu_les_layer> layers;

    fixture() {
        require(ctx != nullptr);
        for (int i = 0; i < 2; ++i) {
            auto * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
            const float weights[] = {1, 0, 0, 1, -1, 0, 0, -1};
            std::copy(weights, weights + 8, (float *) router->data);
            auto * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 4);
            ggml_set_f32(experts, 1);
            layers.push_back({router, nullptr, {experts}, 1e-6f, 1.0f});
        }
    }

    ~fixture() { ggml_free(ctx); }
};

static void test_prediction_and_demand_fallback() {
    fixture f;
    llama_cpu_les les(f.layers, 1, 64);
    const int32_t first[] = {0};
    const int32_t predicted[] = {1};
    const int32_t missed[] = {2};
    const float hidden[] = {0, 2};
    float priority[4];
    les.prepare(0, first, 1, hidden, 2, priority);
    les.synchronize();
    les.prepare(1, predicted, 1, hidden, 2, priority);
    require(priority[1] == 1 && priority[2] == 0);
    les.synchronize();
    les.prepare(1, missed, 1, hidden, 2, priority);
    require(priority[2] == 0);
    les.synchronize();
    les.prepare(1, missed, 1, hidden, 2, priority);
    require(priority[2] == 1);
    const auto stats = les.stats();
    require(stats.prediction_matches >= 1 && stats.ready_hits >= 1);
    require(stats.prepared_bytes > 0);
}

static void test_generation_and_budget() {
    fixture f;
    const int32_t ids[] = {1};
    const float first[] = {0, 2};
    const float next[] = {2, 0};
    float priority[4];
    llama_cpu_les les(f.layers, 1, 64);
    les.prepare(0, ids, 1, first, 2, priority);
    les.synchronize();
    les.prepare(0, ids, 1, next, 2, priority);
    les.synchronize();
    les.prepare(1, ids, 1, next, 2, priority);
    require(priority[0] == 1 && priority[1] == 0);
    les.synchronize();
    llama_cpu_les small(f.layers, 1, 8);
    small.prepare(0, ids, 1, first, 2, priority);
    small.synchronize();
    small.prepare(1, ids, 1, first, 2, priority);
    small.synchronize();
    require(std::all_of(priority, priority + 4, [](float x) { return x == 0; }));
    require(small.stats().prepared_bytes == 0);
    bool rejected = false;
    try { llama_cpu_les invalid(f.layers, 1, 0); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected);
}

static void test_priority_preserves_selected_slots() {
    auto * backend = ggml_backend_cpu_init();
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    require(backend && ctx);
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 2, 4);
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 1);
    auto * priority = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * output = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_mul_mat_id_set_expert_priority(output, priority);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr);
    const float w[] = {1,2,3,4, 0,0,0,0, 5,6,7,8, 0,0,0,0};
    const float x[] = {2,3};
    const int32_t selected[] = {2,0,2};
    const float ready[] = {0,0,1,0};
    const int visits[] = {-1,-1,2,-1,0,1,-1,3};
    for (int i = 0; i < 8; ++i) { require(ggml_cpu_moe_expert_for_visit(ready, 4, i) == visits[i]); }
    ggml_backend_tensor_set(weights, w, 0, sizeof(w));
    ggml_backend_tensor_set(input, x, 0, sizeof(x));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    ggml_backend_tensor_set(priority, ready, 0, sizeof(ready));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_cpu_set_n_threads(backend, 2);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    float actual[6];
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    const float expected[] = {28,38,8,18,28,38};
    require(std::equal(actual, actual + 6, expected));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

static void test_repacked_experts_and_priority() {
    auto * reg = ggml_backend_cpu_reg();
    auto get_buffers = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t repack = nullptr;
    if (get_buffers) {
        for (auto * types = get_buffers(ggml_backend_reg_dev_get(reg, 0)); *types; ++types) {
            if (std::string(ggml_backend_buft_name(*types)) == "CPU_REPACK") { repack = *types; break; }
        }
    }
    if (!repack) { std::puts("CPU_REPACK unavailable; skipped repack case"); return; }
    fixture f;
    auto * weights_ctx = ggml_init({1024*1024, nullptr, true});
    auto * weights = ggml_new_tensor_3d(weights_ctx, GGML_TYPE_Q4_0, 256, 8, 4);
    auto * weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx, repack);
    require(weights_buffer != nullptr);
    std::vector<float> original(256*8*4);
    for (int expert = 0; expert < 4; ++expert) {
        std::fill(original.begin() + expert*256*8, original.begin() + (expert + 1)*256*8, float(expert + 1));
    }
    std::vector<uint8_t> quantized(ggml_nbytes(weights));
    ggml_quantize_chunk(GGML_TYPE_Q4_0, original.data(), quantized.data(), 0, 8*4, 256, nullptr);
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    for (auto & layer : f.layers) { layer.experts = {weights}; }
    {
        llama_cpu_les les(f.layers, 1, 16384);
        const int32_t selected[] = {2,0,2};
        const float hidden[] = {0,2};
        float ready[4];
        les.prepare(0, selected, 3, hidden, 2, ready);
        les.synchronize();
        require(les.stats().prepared_bytes > 0);
    }
    auto * backend = ggml_backend_cpu_init();
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 1);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 1);
    auto * priority = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * baseline = ggml_mul_mat_id(ctx, weights, input, ids);
    auto * scheduled = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_mul_mat_id_set_expert_priority(scheduled, priority);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr);
    std::vector<float> values(256, 1.0f/256);
    const int32_t selected[] = {2,0,2};
    const float ready[] = {0,0,1,0};
    ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
    ggml_backend_tensor_set(ids, selected, 0, sizeof(selected));
    ggml_backend_tensor_set(priority, ready, 0, sizeof(ready));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, baseline);
    ggml_build_forward_expand(graph, scheduled);
    ggml_backend_cpu_set_n_threads(backend, 2);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    float a[24], b[24];
    ggml_backend_tensor_get(baseline, a, 0, sizeof(a));
    ggml_backend_tensor_get(scheduled, b, 0, sizeof(b));
    require(std::equal(a, a + 24, b));
    for (int i = 0; i < 24; ++i) { require(std::abs(a[i] - (i/8 == 1 ? 1.0f : 3.0f)) < 0.01f); }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(weights_ctx);
}

static void test_rapid_generation_replacement() {
    fixture f;
    auto * storage = ggml_init({32*1024*1024, nullptr, false});
    require(storage != nullptr);
    for (auto & layer : f.layers) {
        auto * experts = ggml_new_tensor_3d(storage, GGML_TYPE_F32, 256, 1024, 4);
        ggml_set_f32(experts, 1);
        layer.experts = {experts};
    }
    {
        llama_cpu_les les(f.layers, 1, 4*1024*1024);
        const int32_t ids[] = {1};
        const float first[] = {0,2}, next[] = {2,0};
        float priority[4];
        for (int round = 0; round < 8; ++round) {
            for (int i = 0; i < 64; ++i) {
                les.prepare(0, ids, 1, (i%2) ? next : first, 2, priority);
            }
            les.synchronize();
            les.prepare(1, ids, 1, next, 2, priority);
            require(priority[0] == 1 && priority[1] == 0);
        }
    }
    ggml_free(storage);
}

static void test_callback_graph_reuse() {
    fixture f;
    llama_cpu_les les(f.layers, 1, 64);
    auto * logits = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 4);
    auto * selected = ggml_new_tensor_1d(f.ctx, GGML_TYPE_I32, 1);
    auto * hidden = ggml_new_tensor_1d(f.ctx, GGML_TYPE_F32, 2);
    ggml_set_f32(logits, 0);
    auto * priority = les.build_priority(f.ctx, logits, selected, hidden, 1);
    require(priority != nullptr);
    auto * graph = ggml_new_graph(f.ctx);
    ggml_build_forward_expand(graph, priority);
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);
    for (int iteration = 0; iteration < 2; ++iteration) {
        const int32_t expert = iteration ? 0 : 1;
        const float input[] = {iteration ? 2.0f : 1.0f, iteration ? 0.0f : 2.0f};
        float ready[4];
        les.prepare(0, &expert, 1, input, 2, ready);
        les.synchronize();
        ggml_set_i32(selected, expert);
        std::copy(input, input + 2, (float *) hidden->data);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        require(((float *) priority->data)[expert] == 1);
        require(((float *) priority->data)[1 - expert] == 0);
        les.synchronize();
    }
    ggml_backend_free(backend);
}

static void test_shutdown_with_pending_work() {
    fixture f;
    for (int i = 0; i < 20; ++i) {
        llama_cpu_les les(f.layers, 1, 64);
        const int32_t ids[] = {0};
        const float hidden[] = {0, 2};
        float priority[4];
        les.prepare(0, ids, 1, hidden, 2, priority);
    }
}

int main() {
    test_prediction_and_demand_fallback();
    test_generation_and_budget();
    test_priority_preserves_selected_slots();
    test_shutdown_with_pending_work();
    test_repacked_experts_and_priority();
    test_rapid_generation_replacement();
    test_callback_graph_reuse();
    std::puts("CPU LES tests passed");
}
