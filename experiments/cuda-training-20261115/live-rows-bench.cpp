#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char ** argv) {
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name(argc > 1 ? argv[1] : "CUDA2", nullptr);
    if (!backend) return 1;
    const int64_t cols = argc > 2 ? atoll(argv[2]) : 2816;
    const int64_t count = argc > 3 ? atoll(argv[3]) : 16384;
    const int64_t selected = argc > 4 ? atoll(argv[4]) : count;
    const int mode = argc > 5 ? atoi(argv[5]) : 0;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, count);
    auto * grad = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, selected);
    auto * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, selected);
    auto * output = ggml_get_rows_back(ctx, grad, rows, input);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) return 2;
    std::vector<float> values(cols*selected);
    for (size_t i=0; i<values.size(); ++i) values[i] = float(int(i*17%257)-128)*0.001f;
    std::vector<int32_t> ids(selected);
    for (int64_t i=0;i<selected;++i) ids[i] = mode ? (i*17)%(count/4) : i%count;
    std::vector<float> expected(cols*count,0),actual(cols*count);
    for (int64_t i=0;i<selected;++i) for (int64_t c=0;c<cols;++c) expected[ids[i]*cols+c] += values[i*cols+c];
    ggml_backend_tensor_set(grad,values.data(),0,ggml_nbytes(grad));
    ggml_backend_tensor_set(rows,ids.data(),0,ggml_nbytes(rows));
    for (int iter=0;iter<4;++iter) {
        auto start=std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend,graph) != GGML_STATUS_SUCCESS) return 3;
        ggml_backend_synchronize(backend);
        printf("iter=%d ms=%.4f\n",iter,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());fflush(stdout);
    }
    ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
    size_t different=0;for(size_t i=0;i<actual.size();++i) different+=memcmp(&actual[i],&expected[i],sizeof(float))!=0;
    printf("floats=%zu different=%zu\n",actual.size(),different);
    for (int64_t i=0;i<selected;++i) ids[i] = (count-1-ids[i]);
    std::fill(expected.begin(),expected.end(),0.0f);
    for (int64_t i=0;i<selected;++i) for (int64_t c=0;c<cols;++c) expected[ids[i]*cols+c] += values[i*cols+c];
    ggml_backend_tensor_set(rows,ids.data(),0,ggml_nbytes(rows));
    if (ggml_backend_graph_compute(backend,graph) != GGML_STATUS_SUCCESS) return 5;
    ggml_backend_tensor_get(output,actual.data(),0,ggml_nbytes(output));
    for(size_t i=0;i<actual.size();++i) different+=memcmp(&actual[i],&expected[i],sizeof(float))!=0;
    printf("updated_rows_different=%zu\n",different);
    ggml_gallocr_free(alloc);ggml_free(ctx);ggml_backend_free(backend);
    return different ? 4 : 0;
}
