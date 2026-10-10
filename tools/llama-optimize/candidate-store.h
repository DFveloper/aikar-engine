#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llama_opt {
struct packed_choice {
    ggml_type type;
    std::string producer, file, sha256;
    uint64_t bytes = 0, aligned_bytes = 0;
};

struct candidate_group_info {
    std::string tensor, operation;
    std::vector<int64_t> shape;
    std::vector<packed_choice> choices;
};

struct candidate_store {
    std::string source_model, source_identity;
    uint64_t fixed = 0, alignment = 0;
    std::vector<candidate_group_info> groups;

    candidate_store(const std::string & source, const std::string & manifest);
    std::vector<uint8_t> read_packed(size_t group, size_t option) const;
    std::vector<float> decode_tile(size_t group, size_t option, int64_t first_row, int64_t row_count) const;
    std::vector<std::vector<uint64_t>> costs() const;
};

void create_candidate_store(const std::string & source, const std::string & config,
                            const std::string & directory, const std::string & report);
std::string file_sha256(const std::string & file);
} // namespace llama_opt
