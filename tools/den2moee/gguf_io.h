#pragma once

#include "config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"

namespace den2moee {

struct GgufTensorInfo {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    size_t size = 0;
    size_t offset = 0;
};

struct DenseModelInfo {
    std::string architecture;
    int layer_count = 0;
    int hidden_size = 0;
    std::vector<int> intermediate_sizes;
    bool tied_output = false;
    bool has_ple = false;
    bool den2moee = false;
    std::vector<GgufTensorInfo> tensors;
};

class MappedGguf {
public:
    explicit MappedGguf(const std::string & path);
    ~MappedGguf();

    MappedGguf(const MappedGguf &) = delete;
    MappedGguf & operator=(const MappedGguf &) = delete;

    const gguf_context * context() const { return context_; }
    const DenseModelInfo & info() const { return info_; }
    const GgufTensorInfo & tensor(const std::string & name) const;
    const uint8_t * tensor_data(const GgufTensorInfo & tensor) const;
    const uint8_t * tensor_data(const std::string & name) const;
    size_t data_offset() const { return data_offset_; }

private:
    void inspect();

    std::string path_;
    int fd_ = -1;
    size_t mapped_size_ = 0;
    const uint8_t * mapped_ = nullptr;
    gguf_context * context_ = nullptr;
    ggml_context * tensor_context_ = nullptr;
    size_t data_offset_ = 0;
    DenseModelInfo info_;
};

void print_model_inspection(const MappedGguf & model);

} // namespace den2moee
