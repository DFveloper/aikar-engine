#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    const int64_t m = argc > 1 ? atoi(argv[1]) : 8192;
    const int64_t n = argc > 2 ? atoi(argv[2]) : 2816;
    const int64_t k = std::getenv("BENCH_K") ? atoi(std::getenv("BENCH_K")) : 256;
    ggml_backend_load_all();
    auto * cuda = std::getenv("BENCH_CUDA") ? ggml_backend_init_by_name(std::getenv("BENCH_CUDA"), nullptr) : nullptr;
    if (std::getenv("BENCH_CUDA") && !cuda) return 4;
    auto * cuda_second = std::getenv("BENCH_CUDA_SECOND") ? ggml_backend_init_by_name(std::getenv("BENCH_CUDA_SECOND"), nullptr) : nullptr;
    if (std::getenv("BENCH_CUDA_SECOND") && !cuda_second) return 4;
    auto * backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, std::getenv("BENCH_THREADS") ? atoi(std::getenv("BENCH_THREADS")) : 18);
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    const auto type = argc > 4 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
    auto * a = ggml_new_tensor_2d(ctx, type, m, k);
    auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, k);
    ggml_set_input(a);
    ggml_set_input(b);
    auto * out = ggml_out_prod(ctx, a, b);
    ggml_tensor * acc = nullptr;
    if (std::getenv("BENCH_ACCUMULATE")) {
        acc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, n);
        ggml_set_input(acc);
        out = ggml_add_inplace(ctx, acc, out);
    }
    auto * sink = std::getenv("BENCH_Q8_SINK") ? ggml_cast(ctx, out, GGML_TYPE_Q8_0) : out;
    ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sink);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 1;
    for (auto * t : {a, b}) {
        std::vector<float> values(ggml_nelements(t));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i*17)%113) - 56)*0.001f;
        if (t->type == GGML_TYPE_F32) ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        else {
            std::vector<char> packed(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, values.data(), packed.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
            ggml_backend_tensor_set(t, packed.data(), 0, packed.size());
        }
    }
    if (acc) {
        std::vector<float> seed(ggml_nelements(acc));
        for (size_t i = 0; i < seed.size(); ++i) seed[i] = float(int(i%37) - 18)*0.013f;
        ggml_backend_tensor_set(acc, seed.data(), 0, ggml_nbytes(acc));
    }
    for (int i = 0; i < 5; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        std::printf("iter=%d ms=%.5f\n", i, std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    if (argc > 3) {
        std::vector<float> data(ggml_nelements(out));
        ggml_backend_tensor_get(out, data.data(), 0, ggml_nbytes(out));
        auto * f = fopen(argv[3], "wb");
        if (!f) return 3;
        fwrite(data.data(), sizeof(float), data.size(), f);
        fclose(f);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    if (cuda) ggml_backend_free(cuda);
    if (cuda_second) ggml_backend_free(cuda_second);
}
