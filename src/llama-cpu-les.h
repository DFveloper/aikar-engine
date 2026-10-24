#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_cpu_les_layer {
    const ggml_tensor * router;
    const ggml_tensor * router_scale;
    std::vector<const ggml_tensor *> experts;
    float epsilon;
    float input_scale;
};

struct llama_cpu_les_stats {
    uint64_t required_experts = 0;
    uint64_t predicted_experts = 0;
    uint64_t prediction_matches = 0;
    uint64_t ready_hits = 0;
    uint64_t prepared_bytes = 0;
    uint64_t cancelled_jobs = 0;
};

class llama_cpu_les {
public:
    llama_cpu_les(std::vector<llama_cpu_les_layer> layers, int n_selected, size_t prefetch_bytes);
    ~llama_cpu_les();

    void prepare(size_t layer, const int32_t * selected, size_t n_selected,
            const float * hidden, size_t n_hidden, float * priority);
    void synchronize();
    llama_cpu_les_stats stats() const;
    ggml_tensor * build_priority(ggml_context * ctx, ggml_tensor * logits,
            ggml_tensor * selected, ggml_tensor * hidden, size_t layer);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
