#include "ggml.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) return 1;
    const int d = atoi(argv[1]), kv = atoi(argv[2]), nq = 256, nh = d == 256 ? 8 : 2, gqa = d == 256 ? 2 : 8;
    const int window = d == 512 ? 0 : 1024;
    const int mode = std::getenv("AIKAR_BENCH_MODE") ? atoi(std::getenv("AIKAR_BENCH_MODE")) : 0;
    const bool crop = argc > 4 && std::strcmp(argv[4], "crop") == 0;
    ggml_backend_load_all_from_path(std::getenv("AIKAR_BENCH_BACKEND_PATH"));
    auto * backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 2;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, nq, nh*gqa);
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, kv, nh);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, kv, nh);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nq);
    const int begin = crop ? kv - 1280 : 0;
    const int count = kv - begin;
    auto * kc = ggml_view_3d(ctx, k, d, count, nh, k->nb[1], k->nb[2], begin*k->nb[1]);
    auto * vc = ggml_view_3d(ctx, v, d, count, nh, v->nb[1], v->nb[2], begin*v->nb[1]);
    auto * mc = ggml_view_2d(ctx, mask, count, nq, mask->nb[1], begin*mask->nb[0]);
    if (crop) mc = ggml_cont(ctx, mc);
    auto * forward = ggml_flash_attn_ext(ctx, q, kc, vc, mc, 1.0f/sqrtf(d), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_causal(forward, true);
    if (argc > 4 && !crop) {
        const int32_t start = kv - 1280, size = 1280;
        memcpy(forward->op_params + 6, &start, sizeof(start));
        memcpy(forward->op_params + 7, &size, sizeof(size));
    }
    if (mode == 3 || mode == 4) {
        auto * sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nh*gqa);
        ggml_flash_attn_ext_add_sinks(forward, sinks);
    }
    auto * grad = ggml_dup_tensor(ctx, forward);
    auto * result = ggml_flash_attn_ext_back(ctx, forward, grad);
    if (mode == 2 || mode == 3) {
        const int32_t flags = GGML_FLASH_ATTN_BACK_GRAD_VALID | GGML_FLASH_ATTN_BACK_GRAD_Q;
        memcpy(result->op_params + 5, &flags, sizeof(flags));
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 3;
    if (forward->src[4]) {
        std::vector<float> values(nh*gqa, 0.25f);
        ggml_backend_tensor_set(forward->src[4], values.data(), 0, values.size()*sizeof(float));
    }
    {
        std::vector<float> data(ggml_nelements(q));
        for (size_t i=0; i<data.size(); ++i) data[i] = float(int((i*17)%113)-56)/128;
        ggml_backend_tensor_set(q,data.data(),0,ggml_nbytes(q));
    }
    for (auto * t : {k,v}) {
        std::vector<float> data(ggml_nelements(t));
        std::vector<ggml_fp16_t> packed(data.size());
        for (size_t i=0; i<data.size(); ++i) data[i] = float(int((i*37+(t==v?13:0))%257)-128)/256;
        if (d == 512) {
            const auto * traits = ggml_get_type_traits(GGML_TYPE_FP4_KV);
            std::vector<unsigned char> fp4(ggml_row_size(GGML_TYPE_FP4_KV, data.size()));
            traits->from_float_ref(data.data(), fp4.data(), data.size());
            std::vector<float> restored(data.size());
            traits->to_float(fp4.data(), restored.data(), data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = data[i] + (restored[i] - data[i]);
            }
        }
        ggml_fp32_to_fp16_row(data.data(),packed.data(),data.size());
        ggml_backend_tensor_set(t,packed.data(),0,ggml_nbytes(t));
    }
    {
        std::vector<float> data(kv*nq);
        std::vector<ggml_fp16_t> packed(data.size());
        for(int j=0;j<nq;++j) for(int i=0;i<kv;++i) {
            const int pos=kv-nq+j;
            data[j*kv+i]=(mode != 1 && i<=pos && (window==0 || i>pos-window)) ?
                (mode == 5 ? float((i+j)%11 - 5)/16 : 0.0f) : -std::numeric_limits<float>::infinity();
        }
        ggml_fp32_to_fp16_row(data.data(),packed.data(),data.size());
        ggml_backend_tensor_set(mask,packed.data(),0,ggml_nbytes(mask));
    }
    {
        std::vector<float> data(ggml_nelements(grad));
        for (size_t i = 0; i < data.size(); ++i) data[i] = float(int((i*29)%97)-48)/128;
        ggml_backend_tensor_set(grad, data.data(), 0, ggml_nbytes(grad));
    }
    for(int i=0;i<2;++i) {
        const auto start=std::chrono::steady_clock::now();
        for(int j=0;j<1;++j) if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 4;
        ggml_backend_synchronize(backend);
        printf("d=%d kv=%d window=%d iter=%d ms=%.6f\n",d,kv,window,i,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::vector<float> data(ggml_nelements(result));
    ggml_backend_tensor_get(result,data.data(),0,ggml_nbytes(result));
    auto * file=fopen(argv[3],"wb");if(!file)return 5;
    fwrite(data.data(),sizeof(float),data.size(),file);fclose(file);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
