#ifndef META_SOURCE
#define META_SOURCE "/home/user/aikar-engine/ggml/src/ggml-backend-meta.cpp"
#endif
#include META_SOURCE
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>

static ggml_backend_meta_split_state split_input(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    state.ne[0] = state.ne[1] = tensor->ne[0]/2;
    return state;
}

int main() {
    auto * cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    ggml_backend_dev_t devices[] = {cpu, cpu};
    auto * dev = ggml_backend_meta_device(devices, 2, split_input, nullptr);
    auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 1024);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    auto * context = ggml_init({64*1024*1024, nullptr, true});
    std::vector<ggml_tensor *> nodes;
    for (int chain = 0; chain < 200; ++chain) {
        auto * tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, 32);
        tensor->buffer = buffer;
        nodes.push_back(tensor);
        for (int depth = 0; depth < 100; ++depth) {
            tensor = ggml_scale(context, tensor, 0.5f);
            tensor->buffer = buffer;
            nodes.push_back(tensor);
        }
    }
    auto * buf_ctx = static_cast<ggml_backend_meta_buffer_context *>(buffer->context);
    auto & stc = buf_ctx->stc_compute[0];
    for (int round = 0; round < 8; ++round) {
        const auto start = std::chrono::steady_clock::now();
        uint64_t checksum = 0;
        for (int repeat = 0; repeat < 20; ++repeat) {
            for (auto * tensor : nodes) {
                for (bool sync : {false, true}) {
                    auto state = ggml_backend_meta_get_split_state(stc, tensor, sync);
                    if (state.axis != 0 || state.ne[0] != 16 || state.ne[1] != 16) return 2;
                    checksum += state.ne[0] + state.ne[1];
                }
            }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("round=%d ms=%.3f checksum=%llu\n", round, ms, static_cast<unsigned long long>(checksum));
    }
    nodes[nodes.size()/2]->op_params[0] ^= 1;
    auto state = ggml_backend_meta_get_split_state(stc, nodes.back(), true);
    state = ggml_backend_meta_get_split_state(stc, nodes[nodes.size()/2], false);
    if (state.axis != 0 || state.ne[0] != 16) return 3;
    std::puts("metadata invalidation: passed");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
}
