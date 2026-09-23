#include "gguf_io.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace den2moee {

namespace {

std::string required_string(const gguf_context * context, const char * key) {
    const int64_t id = gguf_find_key(context, key);
    if (id < 0 || gguf_get_kv_type(context, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("missing GGUF string metadata: ") + key);
    }
    return gguf_get_val_str(context, id);
}

uint32_t required_u32(const gguf_context * context, const char * key) {
    const int64_t id = gguf_find_key(context, key);
    if (id < 0) throw std::runtime_error(std::string("missing GGUF integer metadata: ") + key);
    switch (gguf_get_kv_type(context, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(context, id);
        case GGUF_TYPE_INT32: return static_cast<uint32_t>(gguf_get_val_i32(context, id));
        default: throw std::runtime_error(std::string("GGUF metadata is not an integer: ") + key);
    }
}

std::vector<int> feed_forward_lengths(const gguf_context * context, int layers) {
    const int64_t id = gguf_find_key(context, "gemma4.feed_forward_length");
    if (id < 0) throw std::runtime_error("missing Gemma4 feed_forward_length metadata");
    std::vector<int> result(layers);
    if (gguf_get_kv_type(context, id) == GGUF_TYPE_ARRAY) {
        if (gguf_get_arr_n(context, id) != static_cast<size_t>(layers)) {
            throw std::runtime_error("Gemma4 feed_forward_length array has wrong size");
        }
        const enum gguf_type type = gguf_get_arr_type(context, id);
        if (type != GGUF_TYPE_UINT32 && type != GGUF_TYPE_INT32) {
            throw std::runtime_error("Gemma4 feed_forward_length array is not integer");
        }
        if (type == GGUF_TYPE_UINT32) {
            const auto * values = static_cast<const uint32_t *>(gguf_get_arr_data(context, id));
            for (int i = 0; i < layers; ++i) result[i] = static_cast<int>(values[i]);
        } else {
            const auto * values = static_cast<const int32_t *>(gguf_get_arr_data(context, id));
            for (int i = 0; i < layers; ++i) result[i] = values[i];
        }
    } else {
        const int value = static_cast<int>(required_u32(context, "gemma4.feed_forward_length"));
        std::fill(result.begin(), result.end(), value);
    }
    return result;
}

} // namespace

MappedGguf::MappedGguf(const std::string & path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("failed to open GGUF: " + path + ": " + std::strerror(errno));
    struct stat st = {};
    if (fstat(fd_, &st) != 0 || st.st_size <= 0) throw std::runtime_error("failed to stat GGUF: " + path);
    mapped_size_ = static_cast<size_t>(st.st_size);
    mapped_ = static_cast<const uint8_t *>(mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        throw std::runtime_error("failed to mmap GGUF: " + path + ": " + std::strerror(errno));
    }
    gguf_init_params params = { true, &tensor_context_ };
    context_ = gguf_init_from_file(path.c_str(), params);
    if (context_ == nullptr) throw std::runtime_error("failed to parse GGUF: " + path);
    data_offset_ = gguf_get_data_offset(context_);
    inspect();
}

MappedGguf::~MappedGguf() {
    if (context_ != nullptr) gguf_free(context_);
    if (tensor_context_ != nullptr) ggml_free(tensor_context_);
    if (mapped_ != nullptr) munmap(const_cast<uint8_t *>(mapped_), mapped_size_);
    if (fd_ >= 0) close(fd_);
}

void MappedGguf::inspect() {
    info_.architecture = required_string(context_, "general.architecture");
    if (info_.architecture != "gemma4") throw std::runtime_error("Den2MoEE converter requires general.architecture=gemma4");
    info_.layer_count = static_cast<int>(required_u32(context_, "gemma4.block_count"));
    info_.hidden_size = static_cast<int>(required_u32(context_, "gemma4.embedding_length"));
    info_.intermediate_sizes = feed_forward_lengths(context_, info_.layer_count);
    info_.tied_output = gguf_find_tensor(context_, "output.weight") < 0 &&
                        gguf_find_tensor(context_, "token_embd.weight") >= 0;
    info_.has_ple = gguf_find_tensor(context_, "per_layer_token_embd.weight") >= 0 ||
                    gguf_find_tensor(context_, "per_layer_model_proj.weight") >= 0;
    const int64_t den2moee_key = gguf_find_key(context_, "gemma4.den2moee.enabled");
    info_.den2moee = den2moee_key >= 0 && gguf_get_kv_type(context_, den2moee_key) == GGUF_TYPE_BOOL &&
                     gguf_get_val_bool(context_, den2moee_key);

    const int64_t tensor_count = gguf_get_n_tensors(context_);
    info_.tensors.reserve(static_cast<size_t>(tensor_count));
    for (int64_t i = 0; i < tensor_count; ++i) {
        GgufTensorInfo tensor;
        tensor.name = gguf_get_tensor_name(context_, i);
        tensor.type = gguf_get_tensor_type(context_, i);
        std::copy_n(gguf_get_tensor_ne(context_, i), GGML_MAX_DIMS, tensor.ne.begin());
        tensor.size = gguf_get_tensor_size(context_, i);
        tensor.offset = gguf_get_tensor_offset(context_, i);
        info_.tensors.push_back(std::move(tensor));
    }

    const int64_t enabled_key = gguf_find_key(context_, "gemma4.den2moee.layer_enabled");
    const uint32_t * enabled = enabled_key >= 0 ? static_cast<const uint32_t *>(gguf_get_arr_data(context_, enabled_key)) : nullptr;
    for (int layer = 0; layer < info_.layer_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        if (info_.den2moee && enabled != nullptr && enabled[layer] != 0) {
            const GgufTensorInfo & shared_gate = tensor(prefix + "ffn_gate_shexp.weight");
            const GgufTensorInfo & shared_up = tensor(prefix + "ffn_up_shexp.weight");
            const GgufTensorInfo & shared_down = tensor(prefix + "ffn_down_shexp.weight");
            const GgufTensorInfo & router = tensor(prefix + "den2moee.router.weight");
            const GgufTensorInfo & gate_v = tensor(prefix + "den2moee.gate_v.weight");
            const GgufTensorInfo & gate_u = tensor(prefix + "den2moee.gate_u.weight");
            const GgufTensorInfo & down = tensor(prefix + "den2moee.down.weight");
            if (shared_gate.type != GGML_TYPE_Q4_0 || shared_up.type != GGML_TYPE_Q4_0 ||
                shared_down.type != GGML_TYPE_Q4_0 || gate_v.type != GGML_TYPE_Q4_0 ||
                gate_u.type != GGML_TYPE_Q4_0 || down.type != GGML_TYPE_Q4_0 || router.type != GGML_TYPE_F32) {
                throw std::runtime_error(prefix + "Den2MoEE tensor type mismatch");
            }
            continue;
        }
        const GgufTensorInfo & gate = tensor(prefix + "ffn_gate.weight");
        const GgufTensorInfo & up = tensor(prefix + "ffn_up.weight");
        const GgufTensorInfo & down = tensor(prefix + "ffn_down.weight");
        if (gate.type != GGML_TYPE_Q4_0 || up.type != GGML_TYPE_Q4_0 || down.type != GGML_TYPE_Q4_0) {
            throw std::runtime_error(prefix + "FFN tensors must all be Q4_0");
        }
        if (gate.ne[0] != info_.hidden_size || gate.ne[1] != info_.intermediate_sizes[layer] ||
            up.ne[0] != info_.hidden_size || up.ne[1] != info_.intermediate_sizes[layer] ||
            down.ne[0] != info_.intermediate_sizes[layer] || down.ne[1] != info_.hidden_size) {
            throw std::runtime_error(prefix + "FFN tensor shape does not match metadata");
        }
    }
}

const GgufTensorInfo & MappedGguf::tensor(const std::string & name) const {
    const auto it = std::find_if(info_.tensors.begin(), info_.tensors.end(), [&](const GgufTensorInfo & value) {
        return value.name == name;
    });
    if (it == info_.tensors.end()) throw std::runtime_error("missing GGUF tensor: " + name);
    return *it;
}

const uint8_t * MappedGguf::tensor_data(const GgufTensorInfo & tensor) const {
    if (data_offset_ > mapped_size_ || tensor.offset > mapped_size_ - data_offset_ ||
        tensor.size > mapped_size_ - data_offset_ - tensor.offset) {
        throw std::runtime_error("GGUF tensor range is outside mapped file: " + tensor.name);
    }
    return mapped_ + data_offset_ + tensor.offset;
}

const uint8_t * MappedGguf::tensor_data(const std::string & name) const {
    return tensor_data(tensor(name));
}

void print_model_inspection(const MappedGguf & model) {
    const DenseModelInfo & info = model.info();
    std::fprintf(stderr, "[den2moee] architecture: %s\n", info.architecture.c_str());
    std::fprintf(stderr, "[den2moee] layers=%d hidden=%d tied_output=%s ple=%s den2moee=%s\n",
                 info.layer_count, info.hidden_size, info.tied_output ? "yes" : "no", info.has_ple ? "yes" : "no",
                 info.den2moee ? "yes" : "no");
    for (int layer = 0; layer < info.layer_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const int64_t enabled_key = gguf_find_key(model.context(), "gemma4.den2moee.layer_enabled");
        const uint32_t * enabled = enabled_key >= 0 ? static_cast<const uint32_t *>(gguf_get_arr_data(model.context(), enabled_key)) : nullptr;
        if (info.den2moee && enabled != nullptr && enabled[layer] != 0) {
            const auto & gate = model.tensor(prefix + "ffn_gate_shexp.weight");
            const auto & routed = model.tensor(prefix + "den2moee.gate_v.weight");
            std::fprintf(stderr, "[den2moee] layer=%d converted shared_gate=%s[%lld,%lld] routed_gate_v=%s[%lld,%lld,%lld]\n",
                         layer, ggml_type_name(gate.type), (long long) gate.ne[0], (long long) gate.ne[1],
                         ggml_type_name(routed.type), (long long) routed.ne[0], (long long) routed.ne[1], (long long) routed.ne[2]);
            continue;
        }
        const auto & gate = model.tensor(prefix + "ffn_gate.weight");
        const auto & up = model.tensor(prefix + "ffn_up.weight");
        const auto & down = model.tensor(prefix + "ffn_down.weight");
        std::fprintf(stderr, "[den2moee] layer=%d ffn=%d gate=%s[%lld,%lld] up=%s[%lld,%lld] down=%s[%lld,%lld]\n",
                     layer, info.intermediate_sizes[layer], ggml_type_name(gate.type),
                     (long long) gate.ne[0], (long long) gate.ne[1], ggml_type_name(up.type),
                     (long long) up.ne[0], (long long) up.ne[1], ggml_type_name(down.type),
                     (long long) down.ne[0], (long long) down.ne[1]);
    }
    std::fprintf(stderr, "[den2moee] tensors=%zu\n", info.tensors.size());
}

} // namespace den2moee
