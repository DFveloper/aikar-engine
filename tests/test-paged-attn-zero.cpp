#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::fprintf(stderr, "CPU backend initialization failed\n");
        return 1;
    }
    ggml_init_params parameters = {1024*1024, nullptr, true};
    ggml_context * context = ggml_init(parameters);
    const int dim = 64, heads = 16, heads_kv = 2, tokens = 8, block = 16;
    auto * query = ggml_new_tensor_3d(context, GGML_TYPE_F32, dim, heads, tokens);
    auto * key = ggml_new_tensor_3d(context, GGML_TYPE_F32, dim, heads_kv, tokens);
    auto * value = ggml_new_tensor_3d(context, GGML_TYPE_F32, dim, heads_kv, tokens);
    auto * cache = ggml_new_tensor_4d(context, GGML_TYPE_F16, dim, block, 2*heads_kv, 1);
    auto * table = ggml_new_tensor_3d(context, GGML_TYPE_I32, 2, 1, 1);
    auto * slots = ggml_new_tensor_2d(context, GGML_TYPE_I32, 1, tokens);
    auto * positions = ggml_new_tensor_2d(context, GGML_TYPE_I32, 2, tokens);
    auto * output = ggml_paged_attn(context, query, key, value, cache, table, slots, positions, 1.0f, block, 0);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    std::vector<float> queries(dim*heads*tokens, 1.0f), zeros(dim*heads_kv*tokens, 0.0f);
    std::vector<ggml_fp16_t> data(dim*block*2*heads_kv);
    for (int head = 0; head < heads_kv; ++head) {
        for (int pos = 0; pos < block; ++pos) {
            for (int column = 0; column < dim; ++column) {
                data[(head*block + pos)*dim + column] = ggml_fp32_to_fp16(pos == 0 ? -1000.0f : 0.0f);
                data[((head + heads_kv)*block + pos)*dim + column] = ggml_fp32_to_fp16(pos == 0 ? INFINITY : 1.0f);
            }
        }
    }
    std::vector<int32_t> writes(tokens, -1), info(2*tokens, 0);
    for (int token = 0; token < tokens; ++token) { info[2*token] = 15; }
    const int32_t pages[] = {0, 65535};
    ggml_backend_tensor_set(query, queries.data(), 0, ggml_nbytes(query));
    ggml_backend_tensor_set(key, zeros.data(), 0, ggml_nbytes(key));
    ggml_backend_tensor_set(value, zeros.data(), 0, ggml_nbytes(value));
    ggml_backend_tensor_set(cache, data.data(), 0, ggml_nbytes(cache));
    ggml_backend_tensor_set(table, pages, 0, sizeof(pages));
    ggml_backend_tensor_set(slots, writes.data(), 0, ggml_nbytes(slots));
    ggml_backend_tensor_set(positions, info.data(), 0, ggml_nbytes(positions));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
    bool passed = true;
    for (float result : actual) { passed = passed && std::isfinite(result) && std::abs(result - 1.0f) < 1e-6f; }
    std::printf("zero-weight nonfinite values: %s first=%g\n", passed ? "PASS" : "FAIL", actual[0]);
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
