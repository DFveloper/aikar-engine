#pragma once

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct llama_model;

struct heretic_tensor_shape {
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
};

void heretic_surgery_f32(
        float * values,
        int64_t n_input,
        int64_t n_output,
        int64_t n_matrices,
        const float * direction,
        float alpha);

bool heretic_surgery_quantized(
        const uint8_t * source,
        size_t source_size,
        const heretic_tensor_shape & shape,
        const std::vector<float> & direction,
        float alpha,
        std::vector<uint8_t> & output,
        std::string & error);

bool heretic_is_target_tensor(const std::string & name, bool attn, bool mlp, bool experts);

bool heretic_rewrite_gguf(
        const std::string & source_path,
        const std::string & output_path,
        const std::vector<std::vector<float>> & directions,
        const std::vector<float> & alpha_attn,
        const std::vector<float> & alpha_mlp,
        bool attn,
        bool mlp,
        bool experts,
        std::string & error);

struct heretic_runtime_editor {
    heretic_runtime_editor(const std::string & source_path, llama_model * model);
    ~heretic_runtime_editor();

    heretic_runtime_editor(const heretic_runtime_editor &) = delete;
    heretic_runtime_editor & operator=(const heretic_runtime_editor &) = delete;

    bool apply(
            const std::vector<std::vector<float>> & directions,
            const std::vector<float> & alpha_attn,
            const std::vector<float> & alpha_mlp,
            bool attn,
            bool mlp,
            bool experts,
            std::string & error);

private:
    struct impl;
    impl * pimpl;
};
