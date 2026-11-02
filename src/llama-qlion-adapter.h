#pragma once

#include "llama.h"
#include "ggml.h"
#include <memory>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_qlion_snapshot {
    std::string path;
    std::string file_hash;
    std::vector<std::string> names;
    std::vector<std::string> hashes;
};

// Capture the on-disk base before the first optimizer update.
LLAMA_API llama_qlion_snapshot llama_qlion_capture(const std::string & base, const std::vector<ggml_tensor *> & tensors);
LLAMA_API void llama_qlion_save(const llama_qlion_snapshot & base, const std::vector<ggml_tensor *> & tensors,
        const std::string & output, const std::string & units = "full");
LLAMA_API void llama_qlion_merge(const std::string & base, const std::vector<std::string> & adapters,
        const std::string & output, bool allow_compatible = false);
// Destinations must use resident, unpacked buffers. Discard the model if application fails.
LLAMA_API void llama_qlion_apply(const std::string & base, const std::vector<std::string> & adapters,
        const std::vector<ggml_tensor *> & tensors, bool allow_compatible = false);

class llama_qlion_overlay {
public:
    llama_qlion_overlay(const std::string & base, const std::vector<std::string> & paths, bool compatible);
    ~llama_qlion_overlay();
    bool has(const char * name) const;
    void patch_mapping(void * data, size_t size) const;
    void read_tensor(const char * name, void * data, size_t size) const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
