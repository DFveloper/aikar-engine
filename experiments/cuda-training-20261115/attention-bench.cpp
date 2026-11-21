#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
int main(int argc, char ** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 16384;
    const int N = 256, D = 512, HQ = 16, HK = 4;
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 1;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, HQ, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, M, HK, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, M, HK, 1);
    auto * d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, HQ, N, 1);
    auto * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, M, N);
    auto * f = ggml_flash_attn_ext(ctx, q, k, v, m, 1/sqrtf(D), 0, 0);
    auto * b = ggml_flash_attn_ext_back(ctx, f, d);
    b->op_params[5] = GGML_FLASH_ATTN_BACK_GRAD_VALID | GGML_FLASH_ATTN_BACK_GRAD_Q;
    b->ne[0] = ggml_nelements(q);
    b->nb[1] = b->nb[2] = b->nb[3] = ggml_nbytes(q);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, b);
    auto * alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) return 2;
    for (auto * t : {q,k,v,d,m}) {
        std::vector<float> data(ggml_nelements(t));
        for (size_t i=0;i<data.size();++i) data[i]=t==m ? 0.f : 0.125f*sinf(float(i%997));
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> packed(data.size());
            ggml_fp32_to_fp16_row(data.data(),packed.data(),data.size());
            ggml_backend_tensor_set(t,packed.data(),0,ggml_nbytes(t));
        } else ggml_backend_tensor_set(t,data.data(),0,ggml_nbytes(t));
    }
    for (int i=0;i<6;++i) {
        const auto start=std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph)!=GGML_STATUS_SUCCESS) return 3;
        ggml_backend_synchronize(backend);
        printf("M=%d iteration=%d ms=%.4f\n",M,i,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        fflush(stdout);
    }
    std::vector<float> result(ggml_nelements(b));
    ggml_backend_tensor_get(b,result.data(),0,ggml_nbytes(b));
    if (argc>2) { auto * file=fopen(argv[2],"wb"); fwrite(result.data(),sizeof(float),result.size(),file); fclose(file); }
    ggml_gallocr_free(alloc); ggml_free(ctx); ggml_backend_free(backend);
}
