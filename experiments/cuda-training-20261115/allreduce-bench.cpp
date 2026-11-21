#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
int main() {
    ggml_backend_load_all();
    ggml_backend_t backends[] = {ggml_backend_init_by_name("CUDA0", nullptr), ggml_backend_init_by_name("CUDA1", nullptr)};
    if (!backends[0] || !backends[1]) return 1;
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[0]));
    auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
    auto reduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
    auto destroy = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
    if (!init || !reduce || !destroy) return 2;
    void * comm = init(backends, 2);
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) for (int64_t ne : {1024, 1048576, 1032, 20971520, 1024}) {
        ggml_context * contexts[2]; ggml_backend_buffer_t buffers[2]; ggml_tensor * tensors[2]; ggml_cgraph * graphs[2];
        std::vector<float> values(ne);
        for (int j = 0; j < 2; ++j) {
            contexts[j] = ggml_init({1024*1024, nullptr, true});
            tensors[j] = ggml_new_tensor_1d(contexts[j], type, ne);
            tensors[j]->flags |= GGML_TENSOR_FLAG_COMPUTE;
            // The scale is in place and leaves 1.5 on each rank after each reduction.
            auto * input = type == GGML_TYPE_F32 ? tensors[j] : ggml_cast(contexts[j], tensors[j], GGML_TYPE_F32);
            auto * scaled = ggml_scale_inplace(contexts[j], input, 0.5f);
            if (type != GGML_TYPE_F32) scaled = ggml_cpy(contexts[j], scaled, tensors[j]);
            graphs[j] = ggml_new_graph(contexts[j]); ggml_build_forward_expand(graphs[j], scaled);
            buffers[j] = ggml_backend_alloc_ctx_tensors(contexts[j], backends[j]);
            std::fill(values.begin(), values.end(), float(j + 1));
            if (type == GGML_TYPE_F32) ggml_backend_tensor_set(tensors[j], values.data(), 0, ne*sizeof(float));
            else {
                std::vector<ggml_fp16_t> packed(ne);
                if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(values.data(), packed.data(), ne);
                else ggml_fp32_to_bf16_row(values.data(), (ggml_bf16_t *) packed.data(), ne);
                ggml_backend_tensor_set(tensors[j], packed.data(), 0, ne*2);
            }
        }
        const int rounds = ne > 1048576 ? 8 : 32;
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < rounds; ++r) {
            if (!reduce(comm, tensors)) return 3;
            for (int j = 0; j < 2; ++j) if (ggml_backend_graph_compute_async(backends[j], graphs[j]) != GGML_STATUS_SUCCESS) return 4;
        }
        const double submit = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (int j = 0; j < 2; ++j) ggml_backend_synchronize(backends[j]);
        const double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (int j = 0; j < 2; ++j) {
            if (type == GGML_TYPE_F32) ggml_backend_tensor_get(tensors[j], values.data(), 0, ne*sizeof(float));
            else {
                std::vector<ggml_fp16_t> packed(ne); ggml_backend_tensor_get(tensors[j], packed.data(), 0, ne*2);
                if (type == GGML_TYPE_F16) ggml_fp16_to_fp32_row(packed.data(), values.data(), ne);
                else ggml_bf16_to_fp32_row((ggml_bf16_t *) packed.data(), values.data(), ne);
            }
            for (float value : values) if (value != 1.5f) {std::fprintf(stderr, "bad reduction type=%d ne=%lld rank=%d got=%g\n", type, (long long)ne, j, value); return 5;}
            ggml_backend_buffer_free(buffers[j]); ggml_free(contexts[j]);
        }
        std::printf("type=%s ne=%lld rounds=%d submit_ms=%.3f total_ms=%.3f OK\n", ggml_type_name(type), (long long)ne, rounds, submit, total); std::fflush(stdout);
    }
    destroy(comm); for (auto * b : backends) ggml_backend_free(b);
}
