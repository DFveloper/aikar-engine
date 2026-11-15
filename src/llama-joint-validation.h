#pragma once

#include "llama-context.h"

struct llama_opt_joint_validation {
    llama_context * owner;
    llama_context * validation;
    bool fuse_all = true;
    ggml_opt_dataset_t dataset = nullptr;
    llama_joint_validation_status status;
    int64_t first = 0, next = 0, end = 0;
    uint64_t microbatches = 0;
    double loss_sum = 0;
    int64_t supervised = 0;
    llama_batch batch {};
    std::unique_ptr<llama_memory_context_i> memory;
    llama_ubatch ubatch {};
    std::vector<int32_t> labels;
    std::vector<uint64_t> teacher_rows;
    std::vector<uint32_t> output_rows;
    ggml_context_ptr snapshot_ctx;
    std::vector<uint8_t> paired_meta;
    ggml_context_ptr paired_ctx;
    std::vector<ggml_backend_buffer_ptr> snapshot_buffers;
    std::map<const ggml_tensor *, ggml_tensor *> weights;
    llm_graph_result * result = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * logp = nullptr;
    ggml_tensor * row_weights = nullptr;
    ggml_tensor * loss = nullptr;
    ggml_tensor * targets = nullptr;
    ggml_tensor * ncorrect = nullptr;

    llama_opt_joint_validation(llama_context * owner, llama_context * validation);
    ~llama_opt_joint_validation();
    void start(ggml_opt_dataset_t data, int64_t first, int64_t end, int64_t step);
    ggml_cgraph * build();
    void set_head_precision(ggml_cgraph * graph);
    ggml_context * pairing_context(ggml_cgraph * train, ggml_cgraph * validation);
    void set_inputs();
    void collect();
};
