#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// The caller must finish device updates before capturing a checkpoint.
class qat_tensor_snapshot {
public:
    explicit qat_tensor_snapshot(const std::vector<ggml_tensor *> & sources) {
        const size_t count = std::max<size_t>(1, sources.size());
        if (count > std::numeric_limits<size_t>::max()/ggml_tensor_overhead()) {
            throw std::runtime_error("checkpoint tensor count overflow");
        }
        ctx.reset(ggml_init({ count*ggml_tensor_overhead(), nullptr, true }));
        if (!ctx) throw std::runtime_error("cannot allocate checkpoint metadata");
        for (auto * source : sources) {
            if (!source || !ggml_is_contiguous(source)) {
                throw std::runtime_error("checkpoint tensors must be contiguous");
            }
            if (copies.count(source)) continue;
            auto * copy = ggml_dup_tensor(ctx.get(), source);
            ggml_set_name(copy, source->name);
            copies.emplace(source, copy);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        if (!buffer) throw std::runtime_error("cannot allocate checkpoint snapshot");
        for (const auto & entry : copies) {
            ggml_backend_tensor_get(entry.first, entry.second->data, 0, ggml_nbytes(entry.first));
        }
    }

    ggml_tensor * get(const ggml_tensor * source) const {
        return copies.at(source);
    }

    std::vector<ggml_tensor *> get(const std::vector<ggml_tensor *> & sources) const {
        std::vector<ggml_tensor *> result;
        result.reserve(sources.size());
        for (const auto * source : sources) result.push_back(get(source));
        return result;
    }

private:
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> copies;
};
