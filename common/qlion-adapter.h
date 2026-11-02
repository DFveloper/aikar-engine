#pragma once

#include "../src/llama-qlion-adapter.h"

using common_qlion_snapshot = llama_qlion_snapshot;
inline auto common_qlion_capture(const std::string & base, const std::vector<ggml_tensor *> & tensors) {
    return llama_qlion_capture(base, tensors);
}
inline void common_qlion_save(const common_qlion_snapshot & base, const std::vector<ggml_tensor *> & tensors,
        const std::string & output, const std::string & units = "full") {
    llama_qlion_save(base, tensors, output, units);
}
inline void common_qlion_merge(const std::string & base, const std::vector<std::string> & adapters,
        const std::string & output, bool compatible = false) {
    llama_qlion_merge(base, adapters, output, compatible);
}
inline void common_qlion_apply(const std::string & base, const std::vector<std::string> & adapters,
        const std::vector<ggml_tensor *> & tensors, bool compatible = false) {
    llama_qlion_apply(base, adapters, tensors, compatible);
}
