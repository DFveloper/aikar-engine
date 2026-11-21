#include "ggml.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <vector>
int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend || argc != 2) return 1;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 512, 32);
    auto * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 256, 32);
    auto * result = ggml_concat(ctx, a, b, 1);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 2;
    for (auto * tensor : {a, b}) {
        std::vector<float> data(ggml_nelements(tensor));
        for (size_t i = 0; i < data.size(); ++i) data[i] = float(int(i%113) - 56);
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }
    for (int i = 0; i < 20; ++i) {
        auto start = std::chrono::steady_clock::now();
        for (int j = 0; j < 50; ++j) if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 3;
        ggml_backend_synchronize(backend);
        std::printf("iter=%d ms=%.5f\n", i, std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/50);
    }
    std::vector<float> data(ggml_nelements(result));
    ggml_backend_tensor_get(result, data.data(), 0, ggml_nbytes(result));
    auto * file = fopen(argv[1], "wb");
    if (!file) return 4;
    fwrite(data.data(), sizeof(float), data.size(), file);
    fclose(file);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}
