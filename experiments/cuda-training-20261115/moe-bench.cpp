#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    const int64_t cols = argc > 1 ? atoi(argv[1]) : 2816, rows = argc > 2 ? atoi(argv[2]) : 352;
    const bool forward = argc > 4;
    const int64_t experts = argc > 5 ? atoi(argv[5]) : 116, used = 8, tokens = argc > 6 ? atoi(argv[6]) : 256;
    ggml_backend_load_all_from_path(std::getenv("AIKAR_BENCH_BACKEND_PATH")); auto * backend = ggml_backend_init_by_name("CUDA0", nullptr); if (!backend) return 1;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, cols, rows, experts);
    auto * g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, forward ? cols : rows, forward && cols > rows ? 1 : used, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    auto * result = forward ? ggml_mul_mat_id(ctx, w, g, ids) : ggml_mul_mat_id_back(ctx, w, g, ids);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, result);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) return 2;
    {
        std::vector<float> weights(ggml_nelements(w));
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int((i*37)%257) - 128)/1024;
        std::vector<char> packed(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), packed.data(), 0, ggml_nrows(w), cols, nullptr);
        ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    }
    std::vector<float> gradient(ggml_nelements(g)); for (size_t i = 0; i < gradient.size(); ++i) gradient[i] = float(int((i*17)%113) - 56)*(forward ? 0.015625f : 1e-8f);
    std::vector<int32_t> routes(used*tokens); for (size_t i = 0; i < routes.size(); ++i) routes[i] = ((i/used)*7 + i%used*13)%experts;
    ggml_backend_tensor_set(g, gradient.data(), 0, ggml_nbytes(g)); ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    for (int i = 0; i < 12; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 3;
        ggml_backend_synchronize(backend);
        std::printf("cols=%lld rows=%lld iter=%d ms=%.5f\n", (long long)cols, (long long)rows, i, std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    if (argc > 3) {std::vector<float> data(ggml_nelements(result));ggml_backend_tensor_get(result, data.data(), 0, ggml_nbytes(result)); auto * file=fopen(argv[3],"wb"); if (!file) return 4; fwrite(data.data(), sizeof(float), data.size(),file); fclose(file);}
    ggml_gallocr_free(alloc); ggml_free(ctx); ggml_backend_free(backend);
}
