#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.load_mode = strcmp(argv[3], "mmap") == 0 ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    params.lazy_mode = strcmp(argv[4], "on") == 0 ? LLAMA_LAZY_MODE_ON : LLAMA_LAZY_MODE_OFF;
    params.use_extra_bufts = strcmp(argv[5], "on") == 0;
    const char * paths[] = { argv[2], nullptr };
    if (strcmp(argv[2], "-") != 0) params.qlion_adapters = paths;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], params), llama_model_free);
    if (!model) return 1;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 256;
    context_params.n_batch = 64;
    context_params.n_ubatch = 64;
    context_params.n_threads = 4;
    context_params.n_threads_batch = 4;
    context_params.op_offload = false;
    context_params.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), context_params), llama_free);
    if (!ctx) return 1;
    std::vector<llama_token> tokens(32);
    const auto * vocab = llama_model_get_vocab(model.get());
    const int count = llama_tokenize(vocab, "hi", 2, tokens.data(), tokens.size(), true, true);
    if (count <= 0 || count > (int) tokens.size()) return 1;
    auto batch = llama_batch_get_one(tokens.data(), count);
    if (llama_decode(ctx.get(), batch) != 0) return 1;
    const auto * logits = llama_get_logits(ctx.get());
    const size_t size = llama_vocab_n_tokens(vocab);
    if (!logits || fwrite(logits, sizeof(float), size, stdout) != size) return 1;
    ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
}
