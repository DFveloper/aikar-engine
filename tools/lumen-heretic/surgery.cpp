#include "surgery.h"

#include "gguf.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};

struct ggml_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int32_t tensor_layer(const std::string & name) {
    int32_t layer = -1;
    return sscanf(name.c_str(), "blk.%d.", &layer) == 1 ? layer : -1;
}

void copy_bytes(std::ifstream & input, std::ofstream & output, uint64_t offset, uint64_t size) {
    input.clear();
    input.seekg((std::streamoff) offset);
    if (!input) throw std::runtime_error("failed to seek source GGUF");
    std::vector<char> buffer(4 * 1024 * 1024);
    while (size > 0) {
        const size_t chunk = (size_t) std::min<uint64_t>(size, buffer.size());
        input.read(buffer.data(), chunk);
        if ((size_t) input.gcount() != chunk) throw std::runtime_error("failed to read source GGUF tensor data");
        output.write(buffer.data(), chunk);
        if (!output) throw std::runtime_error("failed to write output GGUF tensor data");
        size -= chunk;
    }
}

std::vector<uint8_t> read_bytes(std::ifstream & input, uint64_t offset, size_t size) {
    std::vector<uint8_t> result(size);
    input.clear();
    input.seekg((std::streamoff) offset);
    if (!input) throw std::runtime_error("failed to seek source GGUF tensor");
    input.read((char *) result.data(), result.size());
    if ((size_t) input.gcount() != result.size()) throw std::runtime_error("failed to read source GGUF tensor");
    return result;
}

}

struct heretic_runtime_editor::impl {
    std::string source_path;
    llama_model * model;
    gguf_ptr gguf;
    ggml_ptr tensor_ctx;
    std::ifstream input;
    std::unordered_set<std::string> modified;

    impl(const std::string & path, llama_model * model) : source_path(path), model(model) {
        ggml_context * tensor_ctx_raw = nullptr;
        gguf.reset(gguf_init_from_file(path.c_str(), { true, &tensor_ctx_raw }));
        tensor_ctx.reset(tensor_ctx_raw);
        input.open(path, std::ios::binary);
        if (!gguf || !tensor_ctx || !input) throw std::runtime_error("failed to initialize quantized runtime editor");
    }
};

void heretic_surgery_f32(
        float * values,
        int64_t n_input,
        int64_t n_output,
        int64_t n_matrices,
        const float * direction,
        float alpha) {
    std::vector<float> projection(n_input);
    for (int64_t matrix = 0; matrix < n_matrices; ++matrix) {
        std::fill(projection.begin(), projection.end(), 0.0f);
        float * base = values + matrix * n_input * n_output;
        for (int64_t row = 0; row < n_output; ++row) {
            const float scale = direction[row];
            const float * source = base + row * n_input;
            for (int64_t col = 0; col < n_input; ++col) projection[col] += scale * source[col];
        }
        for (int64_t row = 0; row < n_output; ++row) {
            const float scale = alpha * direction[row];
            float * target = base + row * n_input;
            for (int64_t col = 0; col < n_input; ++col) target[col] -= scale * projection[col];
        }
    }
}

bool heretic_surgery_quantized(
        const uint8_t * source,
        size_t source_size,
        const heretic_tensor_shape & shape,
        const std::vector<float> & direction,
        float alpha,
        std::vector<uint8_t> & output,
        std::string & error) {
    if (shape.type != GGML_TYPE_Q4_0) {
        error = "prototype surgery requires Q4_0 target tensors";
        return false;
    }
    const int64_t n_input = shape.ne[0];
    const int64_t n_output = shape.ne[1];
    const int64_t n_matrices = shape.ne[2] * shape.ne[3];
    if (n_input <= 0 || n_output <= 0 || n_matrices <= 0 || n_input % ggml_blck_size(shape.type) != 0) {
        error = "invalid or non-block-aligned target tensor shape";
        return false;
    }
    if ((int64_t) direction.size() != n_output) {
        error = "direction length does not match tensor residual-output dimension";
        return false;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(shape.type);
    if (!traits || !traits->to_float) {
        error = "Q4_0 decoder is unavailable";
        return false;
    }
    const size_t row_size = ggml_row_size(shape.type, n_input);
    const size_t matrix_size = row_size * n_output;
    if (source_size != matrix_size * n_matrices) {
        error = "quantized byte count does not match target tensor shape";
        return false;
    }
    output.resize(source_size);
    auto quantize_matrix = [&](int64_t matrix) {
        std::vector<float> row_values(n_input);
        std::vector<float> projection(n_input);
        std::fill(projection.begin(), projection.end(), 0.0f);
        const uint8_t * matrix_source = source + matrix * matrix_size;
        uint8_t * matrix_output = output.data() + matrix * matrix_size;
        for (int64_t row = 0; row < n_output; ++row) {
            traits->to_float(matrix_source + row * row_size, row_values.data(), n_input);
            const float scale = direction[row];
            for (int64_t col = 0; col < n_input; ++col) projection[col] += scale * row_values[col];
        }
        for (int64_t row = 0; row < n_output; ++row) {
            traits->to_float(matrix_source + row * row_size, row_values.data(), n_input);
            const float scale = alpha * direction[row];
            for (int64_t col = 0; col < n_input; ++col) row_values[col] -= scale * projection[col];
            const size_t written = ggml_quantize_chunk(shape.type, row_values.data(), matrix_output + row * row_size, 0, 1, n_input, nullptr);
            if (written != row_size) return false;
        }
        return true;
    };
    const int64_t n_threads = std::min<int64_t>(n_matrices, std::min<unsigned>(16, std::max(1u, std::thread::hardware_concurrency())));
    if (n_threads == 1) {
        if (!quantize_matrix(0)) {
            error = "Q4_0 quantizer returned an unexpected row size";
            return false;
        }
    } else {
        std::atomic<int64_t> next_matrix(0);
        std::atomic<bool> failed(false);
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        for (int64_t thread = 0; thread < n_threads; ++thread) {
            workers.emplace_back([&]() {
                while (!failed.load(std::memory_order_relaxed)) {
                    const int64_t matrix = next_matrix.fetch_add(1, std::memory_order_relaxed);
                    if (matrix >= n_matrices) break;
                    if (!quantize_matrix(matrix)) failed.store(true, std::memory_order_relaxed);
                }
            });
        }
        for (std::thread & worker : workers) worker.join();
        if (failed.load(std::memory_order_relaxed)) {
            error = "Q4_0 quantizer returned an unexpected row size";
            return false;
        }
    }
    return true;
}

bool heretic_is_target_tensor(const std::string & name, bool attn, bool mlp, bool experts) {
    return (attn && ends_with(name, ".attn_output.weight")) ||
           (mlp && ends_with(name, ".ffn_down.weight")) ||
           (experts && ends_with(name, ".ffn_down_exps.weight"));
}

bool heretic_rewrite_gguf(
        const std::string & source_path,
        const std::string & output_path,
        const std::vector<std::vector<float>> & directions,
        const std::vector<float> & alpha_attn,
        const std::vector<float> & alpha_mlp,
        bool attn,
        bool mlp,
        bool experts,
        std::string & error) {
    const std::string tmp_path = output_path + ".tmp";
    try {
        ggml_context * tensor_ctx_raw = nullptr;
        gguf_ptr source(gguf_init_from_file(source_path.c_str(), { true, &tensor_ctx_raw }));
        ggml_ptr tensor_ctx(tensor_ctx_raw);
        if (!source || !tensor_ctx) throw std::runtime_error("failed to read source GGUF metadata");
        gguf_ptr output(gguf_init_empty());
        gguf_set_kv(output.get(), source.get());
        const int64_t n_tensors = gguf_get_n_tensors(source.get());
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(source.get(), i);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx.get(), name);
            if (!tensor) throw std::runtime_error(std::string(name) + ": missing tensor metadata");
            gguf_add_tensor(output.get(), tensor);
        }
        if (!gguf_write_to_file(output.get(), tmp_path.c_str(), true)) throw std::runtime_error("failed to write output GGUF metadata");

        std::ifstream input(source_path, std::ios::binary);
        std::ofstream out(tmp_path, std::ios::binary | std::ios::app);
        if (!input || !out) throw std::runtime_error("failed to open GGUF tensor streams");
        const uint64_t data_offset = gguf_get_data_offset(source.get());
        uint64_t output_cursor = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const std::string name = gguf_get_tensor_name(source.get(), i);
            const uint64_t tensor_offset = gguf_get_tensor_offset(source.get(), i);
            const size_t tensor_size = gguf_get_tensor_size(source.get(), i);
            if (tensor_offset > output_cursor) {
                std::vector<char> padding((size_t) (tensor_offset - output_cursor), 0);
                out.write(padding.data(), padding.size());
            }
            const int32_t layer = tensor_layer(name);
            const bool is_target = heretic_is_target_tensor(name, attn, mlp, experts);
            const float alpha = layer >= 0 && (size_t) layer < directions.size() ?
                    (ends_with(name, ".attn_output.weight") ? alpha_attn.at(layer) : alpha_mlp.at(layer)) : 0.0f;
            if (!is_target || alpha == 0.0f) {
                copy_bytes(input, out, data_offset + tensor_offset, tensor_size);
            } else {
                if (layer < 0 || (size_t) layer >= directions.size()) throw std::runtime_error(name + ": invalid layer index");
                const ggml_tensor * tensor = ggml_get_tensor(tensor_ctx.get(), name.c_str());
                heretic_tensor_shape shape;
                shape.type = tensor->type;
                memcpy(shape.ne, tensor->ne, sizeof(shape.ne));
                std::vector<uint8_t> source_bytes = read_bytes(input, data_offset + tensor_offset, tensor_size);
                std::vector<uint8_t> edited;
                std::string tensor_error;
                if (!heretic_surgery_quantized(source_bytes.data(), source_bytes.size(), shape, directions[layer], alpha, edited, tensor_error)) {
                    throw std::runtime_error(name + ": " + tensor_error);
                }
                out.write((const char *) edited.data(), edited.size());
            }
            output_cursor = tensor_offset + tensor_size;
        }
        out.close();
        if (!out) throw std::runtime_error("failed to finish output GGUF");
        if (std::rename(tmp_path.c_str(), output_path.c_str()) != 0) throw std::runtime_error("failed to atomically replace output GGUF");
        return true;
    } catch (const std::exception & e) {
        std::remove(tmp_path.c_str());
        error = e.what();
        return false;
    }
}

heretic_runtime_editor::heretic_runtime_editor(const std::string & source_path, llama_model * model) :
        pimpl(new impl(source_path, model)) {
}

heretic_runtime_editor::~heretic_runtime_editor() {
    delete pimpl;
}

bool heretic_runtime_editor::apply(
        const std::vector<std::vector<float>> & directions,
        const std::vector<float> & alpha_attn,
        const std::vector<float> & alpha_mlp,
        bool attn,
        bool mlp,
        bool experts,
        std::string & error) {
    try {
        const uint64_t data_offset = gguf_get_data_offset(pimpl->gguf.get());
        const auto & runtime_tensors = llama_internal_get_tensor_map(pimpl->model);
        for (const auto & item : runtime_tensors) {
            const std::string & name = item.first;
            if (!heretic_is_target_tensor(name, attn, mlp, experts)) continue;
            ggml_tensor * runtime = item.second;
            const int32_t layer = tensor_layer(name);
            if (layer < 0 || (size_t) layer >= directions.size()) throw std::runtime_error(name + ": invalid layer index");
            const float alpha = ends_with(name, ".attn_output.weight") ? alpha_attn.at(layer) : alpha_mlp.at(layer);
            if (alpha == 0.0f && pimpl->modified.count(name) == 0) continue;
            const int64_t tensor_id = gguf_find_tensor(pimpl->gguf.get(), name.c_str());
            if (tensor_id < 0) throw std::runtime_error(name + ": tensor is missing from source GGUF");
            const size_t tensor_size = gguf_get_tensor_size(pimpl->gguf.get(), tensor_id);
            if (ggml_nbytes(runtime) != tensor_size) throw std::runtime_error(name + ": runtime tensor is repacked and cannot be edited safely");
            std::vector<uint8_t> source = read_bytes(pimpl->input,
                    data_offset + gguf_get_tensor_offset(pimpl->gguf.get(), tensor_id), tensor_size);
            if (alpha == 0.0f) {
                ggml_backend_tensor_set(runtime, source.data(), 0, source.size());
                pimpl->modified.erase(name);
                continue;
            }
            heretic_tensor_shape shape;
            shape.type = runtime->type;
            memcpy(shape.ne, runtime->ne, sizeof(shape.ne));
            std::vector<uint8_t> edited;
            std::string tensor_error;
            if (!heretic_surgery_quantized(source.data(), source.size(), shape, directions[layer], alpha, edited, tensor_error)) {
                throw std::runtime_error(name + ": " + tensor_error);
            }
            ggml_backend_tensor_set(runtime, edited.data(), 0, edited.size());
            pimpl->modified.insert(name);
        }
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
