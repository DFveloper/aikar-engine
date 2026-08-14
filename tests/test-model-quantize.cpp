#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void write_test_model(const std::string & path) {
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_u32(gguf, "general.file_type", LLAMA_FTYPE_MOSTLY_BF16);
    gguf_set_val_u32(gguf, "llama.block_count", 1);
    gguf_set_val_u32(gguf, "llama.context_length", 128);
    gguf_set_val_u32(gguf, "llama.embedding_length", 256);
    gguf_set_val_u32(gguf, "llama.feed_forward_length", 256);
    gguf_set_val_u32(gguf, "llama.attention.head_count", 2);
    gguf_set_val_u32(gguf, "llama.attention.head_count_kv", 2);
    gguf_set_val_u32(gguf, "llama.attention.key_length", 128);
    gguf_set_val_u32(gguf, "llama.attention.value_length", 128);
    gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "test");
    const char * tokens[] = { "token" };
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, 1);

    ggml_init_params params = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "failed to create tensor context");

    ggml_tensor * embedding = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 256, 65537);
    ggml_set_name(embedding, "token_embd.weight");
    for (int64_t i = 0; i < ggml_nelements(embedding); ++i) {
        ((ggml_bf16_t *) embedding->data)[i] = ggml_fp32_to_bf16((float) (i % 31) / 31.0f);
    }
    gguf_add_tensor(gguf, embedding);

    ggml_tensor * projection = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, 256, 257);
    ggml_set_name(projection, "per_layer_model_proj.weight");
    for (int64_t i = 0; i < ggml_nelements(projection); ++i) {
        ((ggml_bf16_t *) projection->data)[i] = ggml_fp32_to_bf16((float) (i % 29) / 29.0f);
    }
    gguf_add_tensor(gguf, projection);

    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, 256);
    ggml_set_name(norm, "output_norm.weight");
    for (int64_t i = 0; i < ggml_nelements(norm); ++i) {
        ((ggml_bf16_t *) norm->data)[i] = ggml_fp32_to_bf16(1.0f);
    }
    gguf_add_tensor(gguf, norm);

    check(gguf_write_to_file(gguf, path.c_str(), false), "failed to write test GGUF");
    ggml_free(ctx);
    gguf_free(gguf);
}

int main() {
    static_assert(LLAMA_FTYPE_MOSTLY_MXFP4_MOE == LLAMA_FTYPE_MOSTLY_MXFP4);

    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("llama-quantize-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string input = (dir / "input.gguf").string();
    const std::string output = (dir / "output.gguf").string();
    write_test_model(input);

    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.ftype = LLAMA_FTYPE_MOSTLY_MXFP4;
    params.nthread = 2;
    check(llama_model_quantize(input.c_str(), output.c_str(), &params) == 0, "MXFP4 quantization failed");

    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };
    gguf_context * result = gguf_init_from_file(output.c_str(), gguf_params);
    check(result != nullptr, "failed to load quantized GGUF");
    check(gguf_get_tensor_type(result, gguf_find_tensor(result, "token_embd.weight")) == GGML_TYPE_MXFP4,
            "dense embedding was not quantized to MXFP4");
    check(gguf_get_tensor_type(result, gguf_find_tensor(result, "per_layer_model_proj.weight")) == GGML_TYPE_MXFP4,
            "per-layer projection was not quantized to MXFP4");
    check(gguf_get_tensor_type(result, gguf_find_tensor(result, "output_norm.weight")) == GGML_TYPE_BF16,
            "normalization tensor type changed");

    gguf_free(result);
    std::filesystem::remove(output);
    std::filesystem::remove(input);
    std::filesystem::remove(dir);
    return 0;
}
