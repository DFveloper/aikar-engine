#include "moe-prune.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
extern "C" {
#include "hash/sha256/sha256.h"
}

#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <sys/stat.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

namespace {

struct gguf_deleter {
    void operator()(gguf_context * ctx) const { gguf_free(ctx); }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;

std::string digest_hex(const unsigned char digest[SHA256_DIGEST_SIZE]) {
    static const char hex[] = "0123456789abcdef";
    std::string result(SHA256_DIGEST_SIZE * 2, '0');
    for (size_t i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return "sha256:" + result;
}

void hash_bytes(sha256_t & hash, const void * data, size_t size) {
    sha256_update(&hash, static_cast<const unsigned char *>(data), size);
}

bool is_expert_identity_tensor(const std::string & name) {
    if (name.find(".ffn_gate_inp.weight") != std::string::npos || name.find(".ffn_gate_inp.scale") != std::string::npos) {
        return true;
    }
    return name.find(".ffn_gate_up_exps.") != std::string::npos ||
           name.find(".ffn_gate_exps.") != std::string::npos ||
           name.find(".ffn_up_exps.") != std::string::npos ||
           name.find(".ffn_down_exps.") != std::string::npos;
}

int32_t metadata_i32(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        throw std::runtime_error("missing GGUF metadata: " + key);
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (int32_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(ctx, id);
        default: throw std::runtime_error("GGUF metadata is not an integer: " + key);
    }
}

void write_json_atomic(const json & value, const std::string & path) {
    static std::atomic<uint64_t> next_tmp_id { 0 };
#if defined(_WIN32)
    const int pid = _getpid();
#else
    const int pid = getpid();
#endif
    const std::string tmp = path + "." + std::to_string(pid) + "." + std::to_string(next_tmp_id++) + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to open output: " + tmp);
        }
        out << value.dump(2) << '\n';
        if (!out) {
            throw std::runtime_error("failed to write output: " + tmp);
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to replace output: " + path);
    }
}

json model_file_identity(const std::string & path) {
    const std::filesystem::path canonical = std::filesystem::canonical(path);
    json identity = {
        { "path", canonical.string() },
        { "size", std::filesystem::file_size(canonical) },
        { "mtime", std::filesystem::last_write_time(canonical).time_since_epoch().count() },
    };
#if defined(__linux__)
    struct stat st;
    if (stat(canonical.c_str(), &st) != 0) {
        throw std::runtime_error("failed to stat GGUF model: " + path);
    }
    identity["device"] = st.st_dev;
    identity["inode"] = st.st_ino;
    identity["ctime_sec"] = st.st_ctim.tv_sec;
    identity["ctime_nsec"] = st.st_ctim.tv_nsec;
#endif
    return identity;
}

json model_info_json(const common_moe_prune_model_info & model) {
    return { { "architecture", model.architecture }, { "model_hash", model.model_hash },
             { "expert_tensor_hash", model.expert_tensor_hash }, { "layer_count", model.layer_count },
             { "expert_count", model.expert_count }, { "experts_used", model.experts_used },
             { "moe_layers", model.moe_layers }, { "expert_bytes", model.expert_bytes } };
}

}

void common_moe_prune_expert_stats::record_selection(double gate, bool collect_reap) {
    if (!collect_reap) return;
    if (!std::isfinite(gate) || gate < 0.0 || !std::isfinite(probability_sum + gate) ||
        selection_count == UINT64_MAX || reap_selection_count == UINT64_MAX) {
        throw std::runtime_error("invalid routing gate or selection accumulator overflow");
    }
    ++selection_count;
    probability_sum += gate;
    ++reap_selection_count;
}

double common_moe_prune_expert_stats::mean_probability() const {
    return selection_count == 0 ? 0.0 : probability_sum / selection_count;
}

double common_moe_prune_expert_stats::mean_output_norm() const {
    return selection_count == 0 ? 0.0 : output_norm_sum / selection_count;
}

double common_moe_prune_expert_stats::importance() const {
    return selection_count == 0 ? 0.0 : weighted_output_sum / selection_count;
}

double common_moe_prune_expert_stats::reap_score() const {
    return reap_count == 0 ? 0.0 : reap_sum / reap_count;
}

double common_moe_prune_expert_stats::mean_reap_output_norm() const {
    return reap_count == 0 ? 0.0 : reap_output_norm_sum / reap_count;
}

std::vector<int32_t> common_moe_prune_selected_ids(const ggml_tensor * ids) {
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) {
        throw std::runtime_error("unsupported Top-K expert ID tensor");
    }
    std::vector<int32_t> result(ids->ne[0] * ids->ne[1]);
    const size_t row_bytes = ids->ne[0] * sizeof(int32_t);
    const bool host = !ids->buffer || ggml_backend_buffer_is_host(ids->buffer);
    std::vector<uint8_t> download;
    const uint8_t * source = static_cast<const uint8_t *>(ids->data);
    if (!host) {
        download.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, download.data(), 0, download.size());
        source = download.data();
    }
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        std::memcpy(result.data() + token * ids->ne[0], source + token * ids->nb[1], row_bytes);
    }
    return result;
}

void common_moe_prune_collect_output(
        const ggml_tensor * output,
        const std::vector<int32_t> & ids,
        const std::vector<float> & weights,
        std::vector<common_moe_prune_expert_stats> & stats,
        bool collect_legacy) {
    if (output->type != GGML_TYPE_F32 && output->type != GGML_TYPE_F16 && output->type != GGML_TYPE_BF16) {
        throw std::runtime_error("unsupported REAP activation type");
    }
    const size_t element_size = ggml_type_size(output->type);
    if (output->nb[0] != element_size || output->ne[3] != 1 ||
        ids.size() != (size_t) (output->ne[1] * output->ne[2]) || weights.size() != ids.size()) {
        throw std::runtime_error("REAP output and routing shapes differ");
    }
    // Read bounded chunks; do not retain expert activations.
    std::array<float, 4096> scratch;
    const bool host = !output->buffer || ggml_backend_buffer_is_host(output->buffer);
    for (int64_t token = 0; token < output->ne[2]; ++token) {
        for (int64_t slot = 0; slot < output->ne[1]; ++slot) {
            const size_t route = token * output->ne[1] + slot;
            const int32_t expert = ids[route];
            const double gate = weights[route];
            if (expert < 0 || (size_t) expert >= stats.size() || !std::isfinite(gate) || gate < 0.0) {
                throw std::runtime_error("invalid REAP route or gate weight");
            }
            double sum_sq = 0.0;
            const size_t row_offset = token * output->nb[2] + slot * output->nb[1];
            for (int64_t start = 0; start < output->ne[0];) {
                const size_t count = std::min<size_t>(output->ne[0] - start, sizeof(scratch) / element_size);
                const size_t offset = row_offset + start * element_size;
                const void * data;
                if (host) {
                    data = (const char *) output->data + offset;
                } else {
                    ggml_backend_tensor_get(output, scratch.data(), offset, count * element_size);
                    data = scratch.data();
                }
                for (size_t i = 0; i < count; ++i) {
                    const double value = output->type == GGML_TYPE_F32 ? ((const float *) data)[i] :
                        output->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) data)[i]) :
                        ggml_bf16_to_fp32(((const ggml_bf16_t *) data)[i]);
                    sum_sq += value * value;
                }
                start += count;
            }
            const double norm = std::sqrt(sum_sq);
            auto & stat = stats[expert];
            if (!std::isfinite(norm) || !std::isfinite(stat.reap_sum + gate * norm) ||
                !std::isfinite(stat.reap_output_norm_sum + norm) || stat.reap_count == UINT64_MAX ||
                (collect_legacy && (!std::isfinite(stat.output_norm_sum + norm) || !std::isfinite(stat.weighted_output_sum + gate * norm)))) {
                throw std::runtime_error("non-finite REAP activation or accumulator overflow");
            }
            ++stat.reap_count;
            stat.reap_output_norm_sum += norm;
            stat.reap_sum += gate * norm;
            if (collect_legacy) {
                stat.output_norm_sum += norm;
                stat.weighted_output_sum += gate * norm;
            }
        }
    }
}

std::string common_moe_prune_sha256_data(const void * data, size_t size) {
    sha256_t hash;
    sha256_init(&hash);
    if (size > 0) hash_bytes(hash, data, size);
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    return digest_hex(digest);
}

std::string common_moe_prune_sha256_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file for hashing: " + path);
    }
    sha256_t hash;
    sha256_init(&hash);
    std::vector<char> buffer(4 * 1024 * 1024);
    while (in) {
        in.read(buffer.data(), buffer.size());
        const std::streamsize n = in.gcount();
        if (n > 0) {
            hash_bytes(hash, buffer.data(), n);
        }
    }
    if (!in.eof()) {
        throw std::runtime_error("failed while hashing file: " + path);
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&hash, digest);
    return digest_hex(digest);
}

common_moe_prune_model_info common_moe_prune_inspect_model(const std::string & path) {
    gguf_ptr ctx(gguf_init_from_file(path.c_str(), { true, nullptr }));
    if (!ctx) {
        throw std::runtime_error("failed to read GGUF model: " + path);
    }

    const int64_t arch_id = gguf_find_key(ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        throw std::runtime_error("missing GGUF metadata: general.architecture");
    }

    common_moe_prune_model_info result;
    result.architecture = gguf_get_val_str(ctx.get(), arch_id);
    if (result.architecture != "gemma4") {
        throw std::runtime_error("unsupported architecture: MoE pruning supports Gemma 4 26B A4B only");
    }
    result.layer_count = metadata_i32(ctx.get(), "gemma4.block_count");
    if (result.layer_count != 30) {
        throw std::runtime_error("unsupported Gemma 4 variant: expected 30 layers for 26B A4B");
    }
    result.expert_count = metadata_i32(ctx.get(), "gemma4.expert_count");
    result.experts_used = metadata_i32(ctx.get(), "gemma4.expert_used_count");
    if (result.expert_count <= 0 || result.experts_used <= 0 || result.experts_used > result.expert_count) {
        throw std::runtime_error("invalid Gemma 4 expert metadata");
    }

    struct expert_tensor_span {
        std::string name;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        int32_t type;
        uint64_t offset;
        size_t size;
    };
    std::vector<expert_tensor_span> expert_tensors;
    sha256_t expert_hash;
    sha256_init(&expert_hash);
    const size_t data_offset = gguf_get_data_offset(ctx.get());
    std::set<int32_t> moe_layers;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
        const std::string name = gguf_get_tensor_name(ctx.get(), i);
        if (!is_expert_identity_tensor(name)) {
            continue;
        }
        int32_t layer = -1;
        if (sscanf(name.c_str(), "blk.%d.", &layer) == 1) {
            moe_layers.insert(layer);
        }
        const int64_t * ne = gguf_get_tensor_ne(ctx.get(), i);
        const int32_t type = (int32_t) gguf_get_tensor_type(ctx.get(), i);
        const size_t size = gguf_get_tensor_size(ctx.get(), i);
        expert_tensor_span span { name, {}, type, data_offset + gguf_get_tensor_offset(ctx.get(), i), size };
        std::copy(ne, ne + GGML_MAX_DIMS, span.ne.begin());
        expert_tensors.push_back(std::move(span));
        result.expert_bytes += size;
    }
    if (moe_layers.empty()) {
        throw std::runtime_error("Gemma 4 model has no routed expert tensors");
    }
    result.moe_layers.assign(moe_layers.begin(), moe_layers.end());

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open GGUF tensor data: " + path);
    }
    sha256_t model_hash;
    sha256_init(&model_hash);
    std::vector<unsigned char> buffer(4 * 1024 * 1024);
    uint64_t cursor = 0;
    auto read_range = [&](uint64_t size, bool is_expert) {
        while (size > 0) {
            const size_t chunk = (size_t) std::min<uint64_t>(size, buffer.size());
            file.read(reinterpret_cast<char *>(buffer.data()), chunk);
            if ((size_t) file.gcount() != chunk) {
                throw std::runtime_error("failed while hashing model: " + path);
            }
            hash_bytes(model_hash, buffer.data(), chunk);
            if (is_expert) hash_bytes(expert_hash, buffer.data(), chunk);
            cursor += chunk;
            size -= chunk;
        }
    };
    for (const expert_tensor_span & tensor : expert_tensors) {
        if (tensor.offset < cursor) {
            throw std::runtime_error("GGUF expert tensors are not stored in tensor order");
        }
        read_range(tensor.offset - cursor, false);
        hash_bytes(expert_hash, tensor.name.data(), tensor.name.size());
        hash_bytes(expert_hash, tensor.ne.data(), sizeof(int64_t) * tensor.ne.size());
        hash_bytes(expert_hash, &tensor.type, sizeof(tensor.type));
        read_range(tensor.size, true);
    }
    while (file) {
        file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        const std::streamsize n = file.gcount();
        if (n > 0) hash_bytes(model_hash, buffer.data(), n);
    }
    if (!file.eof()) {
        throw std::runtime_error("failed while hashing model: " + path);
    }

    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&expert_hash, digest);
    result.expert_tensor_hash = digest_hex(digest);
    sha256_final(&model_hash, digest);
    result.model_hash = digest_hex(digest);
    return result;
}

common_moe_prune_model_info common_moe_prune_inspect_model_cached(
        const std::string & path, const std::string & cache_path, bool * cache_hit) {
    if (cache_hit) *cache_hit = false;
    const json identity = model_file_identity(path);
    {
        std::ifstream in(cache_path);
        if (in) {
            try {
                json cache;
                in >> cache;
                if (cache.at("format") == "aikar-moe-prune-model-cache" && cache.at("version") == 1 &&
                    cache.at("source") == identity) {
                    const json & saved = cache.at("model");
                    common_moe_prune_model_info result;
                    result.architecture = saved.at("architecture").get<std::string>();
                    result.model_hash = saved.at("model_hash").get<std::string>();
                    result.expert_tensor_hash = saved.at("expert_tensor_hash").get<std::string>();
                    result.layer_count = saved.at("layer_count").get<int32_t>();
                    result.expert_count = saved.at("expert_count").get<int32_t>();
                    result.experts_used = saved.at("experts_used").get<int32_t>();
                    result.moe_layers = saved.at("moe_layers").get<std::vector<int32_t>>();
                    result.expert_bytes = saved.at("expert_bytes").get<uint64_t>();
                    if (result.architecture == "gemma4" && result.layer_count == 30 &&
                        result.model_hash.rfind("sha256:", 0) == 0 &&
                        result.expert_tensor_hash.rfind("sha256:", 0) == 0) {
                        if (cache_hit) *cache_hit = true;
                        return result;
                    }
                }
            } catch (const json::exception &) {
            }
        }
    }
    const common_moe_prune_model_info result = common_moe_prune_inspect_model(path);
    if (model_file_identity(path) != identity) {
        throw std::runtime_error("GGUF model changed while hashing: " + path);
    }
    write_json_atomic({ { "format", "aikar-moe-prune-model-cache" }, { "version", 1 },
                        { "source", identity }, { "model", model_info_json(result) } }, cache_path);
    return result;
}

common_moe_prune_profile common_moe_prune_profile_load(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open pruning profile: " + path);
    }
    json root;
    try {
        in >> root;
    } catch (const std::exception & e) {
        throw std::runtime_error("invalid pruning profile JSON: " + std::string(e.what()));
    }
    if (!root.is_object() || root.value("format", "") != "aikar-moe-prune") {
        throw std::runtime_error("invalid pruning profile format");
    }
    common_moe_prune_profile profile;
    profile.version = root.at("version").get<int32_t>();
    if (profile.version != 1) {
        throw std::runtime_error("unsupported pruning profile version: " + std::to_string(profile.version));
    }
    profile.mode = root.at("mode").get<std::string>();
    if (profile.mode != "soft") {
        throw std::runtime_error("unsupported pruning profile mode: " + profile.mode);
    }
    const json & model = root.at("model");
    profile.architecture = model.at("architecture").get<std::string>();
    profile.model_hash = model.at("model_hash").get<std::string>();
    profile.expert_tensor_hash = model.at("expert_tensor_hash").get<std::string>();
    profile.expert_count = model.at("expert_count").get<int32_t>();
    profile.experts_used = model.at("experts_used").get<int32_t>();
    const json & calibration = root.at("calibration");
    profile.dataset_hash = calibration.at("dataset_hash").get<std::string>();
    profile.ppl_mask = calibration.at("ppl_mask").get<std::string>();
    profile.metric = calibration.at("metric").get<std::string>();
    profile.evaluated_tokens = calibration.at("evaluated_tokens").get<int64_t>();
    profile.calibration_tokens = calibration.value("processed_tokens", int64_t(0));
    profile.calibration_collector_version = calibration.value("collector_version", int32_t(0));
    profile.calibration_tokenized_hash = calibration.value("tokenized_hash", "");
    profile.calibration_execution_hash = calibration.value("execution_hash", "");
    profile.calibration_fingerprint = calibration.value("fingerprint", "");
    profile.calibration_seed = calibration.value("seed", int32_t(-1));
    profile.calibration_context = calibration.value("ctx_size", int32_t(0));
    profile.calibration_batch = calibration.value("batch_size", int32_t(0));
    profile.calibration_ubatch = calibration.value("ubatch_size", int32_t(0));
    const json & pruning = root.at("pruning");
    profile.requested_ratio = pruning.at("requested_ratio").get<double>();
    profile.actual_ratio = pruning.at("actual_ratio").get<double>();
    for (auto it = pruning.at("layers").begin(); it != pruning.at("layers").end(); ++it) {
        size_t used = 0;
        int32_t layer = std::stoi(it.key(), &used);
        if (used != it.key().size()) {
            throw std::runtime_error("invalid layer key in pruning profile: " + it.key());
        }
        profile.layers[layer].disabled_experts = it.value().at("disabled_experts").get<std::vector<int32_t>>();
    }
    return profile;
}

void common_moe_prune_profile_write(const common_moe_prune_profile & profile, const std::string & path) {
    json layers = json::object();
    for (const auto & item : profile.layers) {
        layers[std::to_string(item.first)] = { { "disabled_experts", item.second.disabled_experts } };
    }
    json root = {
        { "format", "aikar-moe-prune" },
        { "version", profile.version },
        { "mode", profile.mode },
        { "model", {
            { "architecture", profile.architecture },
            { "model_hash", profile.model_hash },
            { "expert_tensor_hash", profile.expert_tensor_hash },
            { "expert_count", profile.expert_count },
            { "experts_used", profile.experts_used },
        } },
        { "calibration", {
            { "dataset_hash", profile.dataset_hash },
            { "ppl_mask", profile.ppl_mask },
            { "metric", profile.metric },
            { "evaluated_tokens", profile.evaluated_tokens },
            { "processed_tokens", profile.calibration_tokens },
            { "collector_version", profile.calibration_collector_version },
            { "tokenized_hash", profile.calibration_tokenized_hash },
            { "execution_hash", profile.calibration_execution_hash },
            { "fingerprint", profile.calibration_fingerprint },
            { "seed", profile.calibration_seed },
            { "ctx_size", profile.calibration_context },
            { "batch_size", profile.calibration_batch },
            { "ubatch_size", profile.calibration_ubatch },
        } },
        { "pruning", {
            { "requested_ratio", profile.requested_ratio },
            { "actual_ratio", profile.actual_ratio },
            { "layers", layers },
        } },
    };
    write_json_atomic(root, path);
}

void common_moe_prune_profile_validate(const common_moe_prune_profile & profile, const common_moe_prune_model_info & model) {
    if (profile.architecture != model.architecture) throw std::runtime_error("pruning profile architecture mismatch");
    if (profile.model_hash != model.model_hash) throw std::runtime_error("pruning profile model hash mismatch");
    if (profile.expert_tensor_hash != model.expert_tensor_hash) throw std::runtime_error("pruning profile expert tensor hash mismatch");
    if (profile.expert_count != model.expert_count) throw std::runtime_error("pruning profile expert count mismatch");
    if (profile.experts_used != model.experts_used) throw std::runtime_error("pruning profile router Top-K mismatch");
    if (profile.layers.size() != model.moe_layers.size()) throw std::runtime_error("pruning profile MoE layer count mismatch");
    if (!profile.metric.empty() && (profile.calibration_collector_version != COMMON_MOE_PRUNE_STATS_VERSION ||
        profile.calibration_fingerprint.empty() || profile.calibration_tokenized_hash.empty() || profile.calibration_execution_hash.empty())) {
        throw std::runtime_error("profile has obsolete pruning statistics; regenerate from a new calibration cache");
    }
    size_t expected_disabled = SIZE_MAX;
    for (int32_t layer : model.moe_layers) {
        auto it = profile.layers.find(layer);
        if (it == profile.layers.end()) throw std::runtime_error("pruning profile is missing MoE layer " + std::to_string(layer));
        const auto & disabled = it->second.disabled_experts;
        if (disabled.empty()) throw std::runtime_error("pruning profile disables no experts in layer " + std::to_string(layer));
        if (expected_disabled == SIZE_MAX) expected_disabled = disabled.size();
        if (disabled.size() != expected_disabled) throw std::runtime_error("heterogeneous surviving expert counts are unsupported");
        std::set<int32_t> unique;
        for (int32_t expert : disabled) {
            if (expert < 0 || expert >= model.expert_count) throw std::runtime_error("invalid expert ID in layer " + std::to_string(layer));
            if (!unique.insert(expert).second) throw std::runtime_error("duplicate expert ID in layer " + std::to_string(layer));
        }
        if (model.expert_count - (int32_t) disabled.size() < model.experts_used) throw std::runtime_error("pruning profile violates router Top-K safety");
    }
}

void common_moe_prune_profile_apply(llama_model * model, const common_moe_prune_profile & profile) {
    std::vector<std::vector<int32_t>> storage;
    std::vector<llama_moe_prune_layer> layers;
    storage.reserve(profile.layers.size());
    layers.reserve(profile.layers.size());
    for (const auto & item : profile.layers) {
        storage.push_back(item.second.disabled_experts);
        layers.push_back({ item.first, storage.back().data(), storage.back().size() });
    }
    char error[512];
    if (!llama_model_set_moe_prune(model, layers.data(), layers.size(), error, sizeof(error))) {
        throw std::runtime_error(error);
    }
}

std::vector<common_moe_prune_profile> common_moe_prune_make_profiles(
        const common_moe_prune_model_info & model,
        const common_moe_prune_stats & stats,
        const std::vector<double> & ratios,
        double max_layer_ratio,
        const std::string & dataset_hash,
        const std::string & ppl_mask,
        const std::string & metric,
        int64_t evaluated_tokens) {
    if (metric != "router-output" && metric != "reap" && metric != "frequency") {
        throw std::runtime_error("unsupported importance metric: " + metric);
    }
    if (ratios.empty()) throw std::runtime_error("no pruning ratios were requested");
    if (!std::isfinite(max_layer_ratio) || max_layer_ratio < 0.0 || max_layer_ratio >= 1.0) throw std::runtime_error("max layer ratio must be in [0, 1)");
    std::vector<double> sorted_ratios = ratios;
    std::sort(sorted_ratios.begin(), sorted_ratios.end());
    if (std::adjacent_find(sorted_ratios.begin(), sorted_ratios.end()) != sorted_ratios.end()) throw std::runtime_error("duplicate pruning ratio");
    for (double ratio : sorted_ratios) {
        if (!std::isfinite(ratio) || ratio <= 0.0 || ratio > max_layer_ratio) throw std::runtime_error("pruning ratio must be positive and no greater than max layer ratio");
    }

    std::map<int32_t, std::vector<int32_t>> ranking;
    for (int32_t layer : model.moe_layers) {
        auto found = stats.find(layer);
        if (found == stats.end() || found->second.size() != (size_t) model.expert_count) throw std::runtime_error("missing expert statistics for layer " + std::to_string(layer));
        auto & ids = ranking[layer];
        auto score = [&](int32_t expert) {
            const auto & stat = found->second[expert];
            if (metric == "frequency") return (double) (stat.reap_selection_count > 0 ? stat.reap_selection_count : stat.selection_count);
            if (metric == "router-output") return stat.importance();
            if (stat.reap_count != stat.reap_selection_count || !std::isfinite(stat.reap_sum) || stat.reap_sum < 0.0) {
                throw std::runtime_error("missing or invalid REAP statistics for layer " + std::to_string(layer));
            }
            return stat.reap_score();
        };
        ids.resize(model.expert_count);
        for (int32_t i = 0; i < model.expert_count; ++i) { ids[i] = i; score(i); }
        std::stable_sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
            const double ia = score(a);
            const double ib = score(b);
            return ia == ib ? a < b : ia < ib;
        });
    }

    std::vector<common_moe_prune_profile> result;
    for (double ratio : sorted_ratios) {
        int32_t count = (int32_t) std::floor(model.expert_count * ratio + 1e-12);
        count = std::min(count, model.expert_count - model.experts_used);
        if (count == 0) throw std::runtime_error("pruning ratio is too small to remove an expert");
        common_moe_prune_profile profile;
        profile.architecture = model.architecture;
        profile.model_hash = model.model_hash;
        profile.expert_tensor_hash = model.expert_tensor_hash;
        profile.expert_count = model.expert_count;
        profile.experts_used = model.experts_used;
        profile.dataset_hash = dataset_hash;
        profile.ppl_mask = ppl_mask;
        profile.metric = metric;
        profile.evaluated_tokens = evaluated_tokens;
        profile.requested_ratio = ratio;
        profile.actual_ratio = model.expert_count == 0 ? 0.0 : (double) count / model.expert_count;
        for (int32_t layer : model.moe_layers) {
            auto disabled = std::vector<int32_t>(ranking[layer].begin(), ranking[layer].begin() + count);
            std::sort(disabled.begin(), disabled.end());
            profile.layers[layer].disabled_experts = std::move(disabled);
        }
        result.push_back(std::move(profile));
    }
    return result;
}
