#include "ggml.h"
#include "ggml-backend.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 4) return 1;
    const int d = atoi(argv[1]), kv = atoi(argv[2]), nq = 256, nh = d == 256 ? 8 : 2, gqa = d == 256 ? 2 : 8;
    const int window = argc > 4 ? atoi(argv[4]) : 1024;
    const bool ste = argc > 5;
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name("CUDA0", nullptr);
    if (!backend) return 2;
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, nq, nh*gqa);
    auto * k_input = ggml_new_tensor_3d(ctx, ste ? GGML_TYPE_F32 : GGML_TYPE_F16, d, kv, nh);
    auto * v_input = ggml_new_tensor_3d(ctx, ste ? GGML_TYPE_F32 : GGML_TYPE_F16, d, kv, nh);
    auto fake_quant = [&](ggml_tensor * input) {
        auto * quant = ggml_cast(ctx, input, GGML_TYPE_FP4_KV);
        auto * restored = ggml_cast(ctx, quant, GGML_TYPE_F32);
        auto * stop = ggml_cpy_no_grad(ctx, restored, ggml_dup_tensor(ctx, restored));
        auto * master_stop = ggml_cpy_no_grad(ctx, input, ggml_dup_tensor(ctx, input));
        return ggml_cast(ctx, ggml_add(ctx, input, ggml_sub(ctx, stop, master_stop)), GGML_TYPE_F16);
    };
    auto * k = ste ? fake_quant(k_input) : k_input;
    auto * v = ste ? fake_quant(v_input) : v_input;
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nq);
    auto * result = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(d), 0.0f, 0.0f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 3;
    {
        std::vector<float> data(ggml_nelements(q));
        for (size_t i=0; i<data.size(); ++i) data[i] = float(int((i*17)%113)-56)/128;
        ggml_backend_tensor_set(q,data.data(),0,ggml_nbytes(q));
    }
    for (auto * t : {k_input,v_input}) {
        std::vector<float> data(ggml_nelements(t));
        std::vector<ggml_fp16_t> packed(data.size());
        for (size_t i=0; i<data.size(); ++i) data[i] = float(int((i*37+(t==v_input?13:0))%257)-128)/256;
        if (d == 512 && !ste) {
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
        ggml_backend_tensor_set(t,ste ? static_cast<void *>(data.data()) : static_cast<void *>(packed.data()),0,ggml_nbytes(t));
    }
    {
        std::vector<float> data(kv*nq);
        std::vector<ggml_fp16_t> packed(data.size());
        for(int j=0;j<nq;++j) for(int i=0;i<kv;++i) {
            const int pos=kv-nq+j;
            data[j*kv+i]=(i<=pos && (window==0 || i>pos-window))?0.0f:-std::numeric_limits<float>::infinity();
        }
        ggml_fp32_to_fp16_row(data.data(),packed.data(),data.size());
        ggml_backend_tensor_set(mask,packed.data(),0,ggml_nbytes(mask));
    }
    for(int i=0;i<12;++i) {
        const auto start=std::chrono::steady_clock::now();
        for(int j=0;j<5;++j) if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 4;
        ggml_backend_synchronize(backend);
        printf("d=%d kv=%d window=%d iter=%d ms=%.6f\n",d,kv,window,i,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/5);
    }
    std::vector<float> data(ggml_nelements(result));
    ggml_backend_tensor_get(result,data.data(),0,ggml_nbytes(result));
    auto * file=fopen(argv[3],"wb");if(!file)return 5;
    fwrite(data.data(),sizeof(float),data.size(),file);fclose(file);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
}
