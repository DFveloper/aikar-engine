#include "converter.h"

#include "calibration.h"
#include "clustering.h"
#include "gguf_io.h"
#include "quantize.h"
#include "svd.h"

#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace den2moee {

namespace {

struct gguf_deleter {
    void operator()(gguf_context * context) const { gguf_free(context); }
};

struct ggml_deleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

using gguf_ptr = std::unique_ptr<gguf_context, gguf_deleter>;
using ggml_ptr = std::unique_ptr<ggml_context, ggml_deleter>;

struct OutputTensor {
    enum class Source { Original, Spool };

    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    size_t size = 0;
    Source source = Source::Original;
    int64_t source_id = -1;
    size_t spool_offset = 0;
};

struct SpoolFile {
    std::string path;
    std::ofstream stream;
    size_t bytes = 0;

    explicit SpoolFile(std::string path_) : path(std::move(path_)), stream(path, std::ios::binary | std::ios::trunc) {
        if (!stream) throw std::runtime_error("failed to create converter scratch spool: " + path);
    }

    ~SpoolFile() {
        stream.close();
        std::remove(path.c_str());
    }

    size_t append(const std::vector<uint8_t> & payload) {
        const size_t offset = bytes;
        if (!payload.empty()) stream.write(reinterpret_cast<const char *>(payload.data()), payload.size());
        if (!stream) throw std::runtime_error("failed to write converter scratch spool");
        bytes += payload.size();
        return offset;
    }
};

int layer_from_name(const std::string & name) {
    int layer = -1;
    return std::sscanf(name.c_str(), "blk.%d.", &layer) == 1 ? layer : -1;
}

bool ends_with(const std::string & value, const char * suffix) {
    const size_t length = std::strlen(suffix);
    return value.size() >= length && value.compare(value.size() - length, length, suffix) == 0;
}

bool is_dense_ffn(const std::string & name) {
    return ends_with(name, ".ffn_gate.weight") || ends_with(name, ".ffn_up.weight") ||
           ends_with(name, ".ffn_down.weight");
}

std::vector<uint8_t> as_bytes(const std::vector<float> & values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

std::vector<float> router_random(int rows, int cols, uint64_t seed, float stddev) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, stddev);
    std::vector<float> result(static_cast<size_t>(rows) * cols);
    for (float & value : result) value = normal(rng);
    return result;
}

std::vector<float> router_init_for_cluster(const MappedGguf & source, const std::string & name,
                                           int source_cols, const std::vector<int> & ids) {
    const auto & tensor = source.tensor(name);
    const std::vector<float> rows = dequantize_q4_0_rows(source.tensor_data(tensor), source_cols, ids);
    std::vector<float> result(source_cols, 0.0f);
    for (size_t row = 0; row < ids.size(); ++row) {
        for (int col = 0; col < source_cols; ++col) result[col] += rows[row * source_cols + col];
    }
    for (float & value : result) value /= static_cast<float>(ids.size());
    return result;
}

std::vector<uint8_t> make_f32_tensor(const std::vector<float> & values) {
    return as_bytes(values);
}

std::vector<float> transpose_v(const SvdFactor & factor) {
    std::vector<float> result(static_cast<size_t>(factor.storage_rank) * factor.cols, 0.0f);
    for (int col = 0; col < factor.cols; ++col) {
        for (int rank = 0; rank < factor.storage_rank; ++rank) {
            result[rank * factor.cols + col] = factor.v[col * factor.storage_rank + rank];
        }
    }
    return result;
}

void log_scratch(size_t scratch, int layer, int expert, size_t & peak) {
    peak = std::max(peak, scratch);
    std::fprintf(stderr, "[den2moee] layer=%d expert=%d CPU scratch=%zu MiB\n",
                 layer, expert, (scratch + (1u << 20) - 1) / (1u << 20));
}

} // namespace

void convert_dense_gemma4(const std::string & model_path, const std::string & calibration_path,
                          const std::string & output_path, const Options & options) {
    if (model_path == output_path) throw std::invalid_argument("converter never modifies the source GGUF in place");
    validate_options(options);
    MappedGguf source(model_path);
    const DenseModelInfo & model = source.info();
    if (model.hidden_size <= 0 || model.layer_count <= 0) throw std::runtime_error("invalid Gemma4 model dimensions");
    const int first_layer = std::max(0, options.layer_start);
    const int last_layer = options.layer_end < 0 ? model.layer_count - 1 : options.layer_end;
    if (first_layer > last_layer || last_layer >= model.layer_count) {
        throw std::invalid_argument("layer range is outside the model");
    }

    std::vector<LayerConfig> layer_configs;
    layer_configs.reserve(model.layer_count);
    for (int intermediate : model.intermediate_sizes) {
        layer_configs.push_back(make_layer_config(model.hidden_size, intermediate, options));
    }
    for (int layer = first_layer; layer <= last_layer; ++layer) {
        if (model.intermediate_sizes[layer] % options.experts != 0) {
            throw std::runtime_error("FFN intermediate size is not divisible by --experts at layer " + std::to_string(layer));
        }
    }

    common_params model_params;
    model_params.model.path = model_path;
    model_params.n_gpu_layers = options.gpu_layers;
    const std::vector<ggml_backend_dev_t> devices = parse_device_spec(options.device);
    model_params.devices = devices;
    model_params.n_ctx = options.max_seq_len;
    model_params.n_batch = options.max_seq_len;
    model_params.n_ubatch = std::min(512, options.max_seq_len);
    model_params.warmup = false;
    common_init_result_ptr vocab_init = common_init_from_params(model_params, true);
    if (!vocab_init || vocab_init->model() == nullptr) throw std::runtime_error("failed to load source model vocabulary");
    const std::vector<CalibrationSample> samples = load_calibration_jsonl(
        calibration_path, vocab_init->model(), options.max_seq_len, options.max_samples,
        options.input_format, options.chat_template);
    vocab_init.reset();

    std::vector<int> intermediate_sizes = model.intermediate_sizes;
    CalibrationResult calibration = run_streaming_calibration(model_path, samples, intermediate_sizes, options);
    if (calibration.hidden_size != model.hidden_size ||
        calibration.layer_features.size() != static_cast<size_t>(model.layer_count)) {
        throw std::runtime_error("calibration dimensions do not match GGUF metadata");
    }
    std::fprintf(stderr, "[den2moee] peak VRAM: %zu MiB / limit %zu MiB\n",
                 (calibration.peak_vram + (1u << 20) - 1) / (1u << 20), options.vram_limit_mib);
    std::fprintf(stderr, "[den2moee] peak CPU scratch: %zu MiB\n",
                 (calibration.peak_cpu_scratch + (1u << 20) - 1) / (1u << 20));
    if (calibration.peak_vram > options.vram_limit_mib * (1u << 20)) {
        throw std::runtime_error("calibration exceeded --vram-limit-mib");
    }

    const std::string spool_path = output_path + ".den2moee-spool.tmp";
    SpoolFile spool(spool_path);
    std::vector<OutputTensor> plans;
    plans.reserve(model.tensors.size() + static_cast<size_t>(last_layer - first_layer + 1) * 12);
    size_t peak_scratch = calibration.peak_cpu_scratch;
    auto add_original = [&](int64_t source_id) {
        const auto & tensor = model.tensors[static_cast<size_t>(source_id)];
        OutputTensor plan;
        plan.name = tensor.name;
        plan.type = tensor.type;
        plan.ne = tensor.ne;
        plan.size = tensor.size;
        plan.source = OutputTensor::Source::Original;
        plan.source_id = source_id;
        plans.push_back(std::move(plan));
    };
    auto add_spooled = [&](const std::string & name, ggml_type type,
                           std::array<int64_t, GGML_MAX_DIMS> ne, std::vector<uint8_t> payload) {
        OutputTensor plan;
        plan.name = name;
        plan.type = type;
        plan.ne = ne;
        plan.size = payload.size();
        plan.source = OutputTensor::Source::Spool;
        plan.spool_offset = spool.append(payload);
        plans.push_back(std::move(plan));
    };

    bool routed_layer_seen = false;
    for (int64_t source_id = 0; source_id < static_cast<int64_t>(model.tensors.size()); ++source_id) {
        const std::string & name = model.tensors[static_cast<size_t>(source_id)].name;
        const int layer = layer_from_name(name);
        const bool convert_layer = layer >= first_layer && layer <= last_layer;
        if (convert_layer && is_dense_ffn(name)) {
            if (ends_with(name, ".ffn_gate.weight")) {
                routed_layer_seen = true;
                const LayerConfig & cfg = layer_configs[layer];
                const auto & clusters = balanced_cluster(calibration.layer_features[layer], options.experts,
                                                         options.seed + static_cast<uint64_t>(layer));
                const ExpertRoles roles = rank_expert_roles(calibration.layer_features[layer], clusters,
                    options.coverage_threshold_ratio, options.coverage_percentile, options.shared_experts);
                const std::string prefix = "blk." + std::to_string(layer) + ".";
                const std::string gate_name = prefix + "ffn_gate.weight";
                const std::string up_name = prefix + "ffn_up.weight";
                const std::string down_name = prefix + "ffn_down.weight";
                std::vector<int> shared_ids;
                for (int cluster : roles.shared_clusters) {
                    shared_ids.insert(shared_ids.end(), clusters.neuron_ids[cluster].begin(), clusters.neuron_ids[cluster].end());
                }
                std::sort(shared_ids.begin(), shared_ids.end());
                const int shared_width = static_cast<int>(shared_ids.size());
                add_spooled(prefix + "ffn_gate_shexp.weight", GGML_TYPE_Q4_0,
                    { model.hidden_size, shared_width, 1, 1 },
                    copy_q4_0_rows(source.tensor_data(gate_name), model.hidden_size, shared_ids));
                add_spooled(prefix + "ffn_up_shexp.weight", GGML_TYPE_Q4_0,
                    { model.hidden_size, shared_width, 1, 1 },
                    copy_q4_0_rows(source.tensor_data(up_name), model.hidden_size, shared_ids));
                add_spooled(prefix + "ffn_down_shexp.weight", GGML_TYPE_Q4_0,
                    { shared_width, model.hidden_size, 1, 1 },
                    gather_q4_0_columns(source.tensor_data(down_name), model.hidden_size,
                                        model.intermediate_sizes[layer], shared_ids));

                const int candidates = options.routed_experts + options.null_experts;
                const float router_std = 1e-2f / std::sqrt(static_cast<float>(model.hidden_size));
                std::vector<float> router = router_random(candidates, model.hidden_size,
                                                           options.seed + 0x100000u + layer, router_std);
                std::vector<float> router_init;
                router_init.reserve(static_cast<size_t>(options.routed_experts) * model.hidden_size);
                std::vector<float> correct_bias(candidates, 0.0f);
                std::vector<float> budget_score(candidates, 0.0f);
                float budget_sum = 0.0f;
                for (int expert = 0; expert < options.routed_experts; ++expert) {
                    const int cluster = roles.routed_clusters[expert];
                    const std::vector<int> & ids = clusters.neuron_ids[cluster];
                    const std::vector<float> init = router_init_for_cluster(source, up_name, model.hidden_size, ids);
                    router_init.insert(router_init.end(), init.begin(), init.end());
                    float score = static_cast<float>(roles.coverage_scores[cluster]) /
                                  static_cast<float>(std::max<size_t>(1, roles.coverage_vectors[cluster].size()));
                    budget_score[expert] = score;
                    budget_sum += score;
                }
                if (budget_sum > 0.0f) {
                    for (int expert = 0; expert < options.routed_experts; ++expert) budget_score[expert] /= budget_sum;
                } else {
                    for (int expert = 0; expert < options.routed_experts; ++expert) budget_score[expert] =
                        1.0f / static_cast<float>(options.routed_experts);
                }
                for (int expert = 0; expert < options.routed_experts; ++expert) {
                    double mean = 0.0;
                    for (int hidden = 0; hidden < model.hidden_size; ++hidden) {
                        mean += static_cast<double>(calibration.mean_inputs[layer][hidden]) *
                                router[expert * model.hidden_size + hidden];
                    }
                    correct_bias[expert] = -static_cast<float>(mean);
                }
                add_spooled(prefix + "den2moee.router.weight", GGML_TYPE_F32,
                    { model.hidden_size, candidates, 1, 1 }, make_f32_tensor(router));
                add_spooled(prefix + "den2moee.router_init.weight", GGML_TYPE_F32,
                    { model.hidden_size, options.routed_experts, 1, 1 }, make_f32_tensor(router_init));
                add_spooled(prefix + "den2moee.correct_bias.weight", GGML_TYPE_F32,
                    { candidates, 1, 1, 1 }, make_f32_tensor(correct_bias));
                add_spooled(prefix + "den2moee.budget_score.weight", GGML_TYPE_F32,
                    { candidates, 1, 1, 1 }, make_f32_tensor(budget_score));

                std::vector<float> gate_sigma;
                std::vector<float> up_sigma;
                std::vector<uint8_t> gate_v_payload;
                std::vector<uint8_t> gate_u_payload;
                std::vector<uint8_t> up_v_payload;
                std::vector<uint8_t> up_u_payload;
                std::vector<uint8_t> down_payload;
                for (int expert = 0; expert < options.routed_experts; ++expert) {
                    const int cluster = roles.routed_clusters[expert];
                    const std::vector<int> & ids = clusters.neuron_ids[cluster];
                    const std::vector<float> gate_matrix = dequantize_q4_0_rows(source.tensor_data(gate_name),
                                                                                 model.hidden_size, ids);
                    const std::vector<float> up_matrix = dequantize_q4_0_rows(source.tensor_data(up_name),
                                                                               model.hidden_size, ids);
                    const SvdFactor gate_svd = truncated_svd(gate_matrix, cfg.expert_intermediate_size,
                        model.hidden_size, cfg.logical_rank, cfg.storage_rank,
                        options.seed + static_cast<uint64_t>(layer) * 4096u + expert * 2u);
                    const SvdFactor up_svd = truncated_svd(up_matrix, cfg.expert_intermediate_size,
                        model.hidden_size, cfg.logical_rank, cfg.storage_rank,
                        options.seed + static_cast<uint64_t>(layer) * 4096u + expert * 2u + 1u);
                    log_scratch(gate_matrix.size() * sizeof(float) + up_matrix.size() * sizeof(float) +
                                gate_svd.u.size() * sizeof(float) + gate_svd.v.size() * sizeof(float) +
                                up_svd.u.size() * sizeof(float) + up_svd.v.size() * sizeof(float),
                                layer, expert, peak_scratch);
                    std::fprintf(stderr, "[den2moee] layer=%d routed=%d logical_rank=%d storage_rank=%d gate_error=%.6f up_error=%.6f\n",
                                 layer, expert, cfg.logical_rank, cfg.storage_rank,
                                 gate_svd.relative_error, up_svd.relative_error);
                    gate_sigma.insert(gate_sigma.end(), gate_svd.sigma.begin(), gate_svd.sigma.end());
                    up_sigma.insert(up_sigma.end(), up_svd.sigma.begin(), up_svd.sigma.end());
                    std::vector<uint8_t> expert_down_payload = gather_q4_0_columns(
                        source.tensor_data(down_name), model.hidden_size,
                        model.intermediate_sizes[layer], ids);
                    const std::vector<uint8_t> expert_gate_v = quantize_q4_0(transpose_v(gate_svd), cfg.storage_rank, model.hidden_size);
                    const std::vector<uint8_t> expert_gate_u = quantize_q4_0(gate_svd.u, cfg.expert_intermediate_size, cfg.storage_rank);
                    const std::vector<uint8_t> expert_up_v = quantize_q4_0(transpose_v(up_svd), cfg.storage_rank, model.hidden_size);
                    const std::vector<uint8_t> expert_up_u = quantize_q4_0(up_svd.u, cfg.expert_intermediate_size, cfg.storage_rank);
                    gate_v_payload.insert(gate_v_payload.end(), expert_gate_v.begin(), expert_gate_v.end());
                    gate_u_payload.insert(gate_u_payload.end(), expert_gate_u.begin(), expert_gate_u.end());
                    up_v_payload.insert(up_v_payload.end(), expert_up_v.begin(), expert_up_v.end());
                    up_u_payload.insert(up_u_payload.end(), expert_up_u.begin(), expert_up_u.end());
                    down_payload.insert(down_payload.end(), expert_down_payload.begin(), expert_down_payload.end());
                }
                add_spooled(prefix + "den2moee.gate_v.weight", GGML_TYPE_Q4_0,
                    { model.hidden_size, cfg.storage_rank, options.routed_experts, 1 }, std::move(gate_v_payload));
                add_spooled(prefix + "den2moee.gate_u.weight", GGML_TYPE_Q4_0,
                    { cfg.storage_rank, cfg.expert_intermediate_size, options.routed_experts, 1 }, std::move(gate_u_payload));
                add_spooled(prefix + "den2moee.gate_sigma.weight", GGML_TYPE_F32,
                    { cfg.storage_rank, options.routed_experts, 1, 1 }, make_f32_tensor(gate_sigma));
                add_spooled(prefix + "den2moee.up_v.weight", GGML_TYPE_Q4_0,
                    { model.hidden_size, cfg.storage_rank, options.routed_experts, 1 }, std::move(up_v_payload));
                add_spooled(prefix + "den2moee.up_u.weight", GGML_TYPE_Q4_0,
                    { cfg.storage_rank, cfg.expert_intermediate_size, options.routed_experts, 1 }, std::move(up_u_payload));
                add_spooled(prefix + "den2moee.up_sigma.weight", GGML_TYPE_F32,
                    { cfg.storage_rank, options.routed_experts, 1, 1 }, make_f32_tensor(up_sigma));
                add_spooled(prefix + "den2moee.down.weight", GGML_TYPE_Q4_0,
                    { cfg.expert_intermediate_size, model.hidden_size, options.routed_experts, 1 }, std::move(down_payload));
            }
            continue;
        }
        add_original(source_id);
    }
    if (!routed_layer_seen) throw std::runtime_error("no converted Gemma4 FFN layer was selected");

    gguf_ptr output(gguf_init_empty());
    if (!output) throw std::runtime_error("failed to create output GGUF context");
    gguf_set_kv(output.get(), source.context());
    const uint32_t version = 1;
    const uint32_t micro_experts = options.experts;
    const uint32_t shared_experts = options.shared_experts;
    const uint32_t routed_experts = options.routed_experts;
    const uint32_t null_experts = options.null_experts;
    const uint32_t top_k = options.top_k;
    std::vector<uint32_t> enabled(model.layer_count);
    std::vector<uint32_t> expert_intermediate(model.layer_count);
    std::vector<uint32_t> logical_rank(model.layer_count);
    std::vector<uint32_t> storage_rank(model.layer_count);
    for (int layer = 0; layer < model.layer_count; ++layer) {
        enabled[layer] = layer >= first_layer && layer <= last_layer ? 1 : 0;
        expert_intermediate[layer] = layer_configs[layer].expert_intermediate_size;
        logical_rank[layer] = layer_configs[layer].logical_rank;
        storage_rank[layer] = layer_configs[layer].storage_rank;
    }
    gguf_set_val_bool(output.get(), "gemma4.den2moee.enabled", true);
    gguf_set_val_u32(output.get(), "gemma4.den2moee.version", version);
    gguf_set_val_u32(output.get(), "gemma4.den2moee.micro_expert_count", micro_experts);
    gguf_set_val_u32(output.get(), "gemma4.den2moee.shared_expert_count", shared_experts);
    gguf_set_val_u32(output.get(), "gemma4.den2moee.routed_expert_count", routed_experts);
    gguf_set_val_u32(output.get(), "gemma4.den2moee.null_expert_count", null_experts);
    gguf_set_val_u32(output.get(), "gemma4.den2moee.top_k", top_k);
    gguf_set_val_f32(output.get(), "gemma4.den2moee.rank_ratio", options.rank_ratio);
    gguf_set_val_f32(output.get(), "gemma4.den2moee.coverage_threshold_ratio", options.coverage_threshold_ratio);
    gguf_set_val_f32(output.get(), "gemma4.den2moee.coverage_percentile", options.coverage_percentile);
    gguf_set_val_str(output.get(), "gemma4.den2moee.score_activation", score_activation_name(options.score_activation));
    gguf_set_arr_data(output.get(), "gemma4.den2moee.layer_enabled", GGUF_TYPE_UINT32, enabled.data(), enabled.size());
    gguf_set_arr_data(output.get(), "gemma4.den2moee.expert_intermediate_size", GGUF_TYPE_UINT32,
                      expert_intermediate.data(), expert_intermediate.size());
    gguf_set_arr_data(output.get(), "gemma4.den2moee.logical_rank", GGUF_TYPE_UINT32,
                      logical_rank.data(), logical_rank.size());
    gguf_set_arr_data(output.get(), "gemma4.den2moee.storage_rank", GGUF_TYPE_UINT32,
                      storage_rank.data(), storage_rank.size());

    const size_t tensor_metadata_memory = std::max<size_t>(64 * 1024 * 1024,
                                                            plans.size() * sizeof(ggml_tensor) * 4);
    ggml_ptr tensor_context(ggml_init({ tensor_metadata_memory, nullptr, true }));
    if (!tensor_context) throw std::runtime_error("failed to allocate output tensor metadata context");
    for (const OutputTensor & plan : plans) {
        ggml_tensor * tensor = ggml_new_tensor(tensor_context.get(), plan.type, GGML_MAX_DIMS, plan.ne.data());
        if (!tensor) throw std::runtime_error("failed to allocate output tensor metadata: " + plan.name);
        ggml_set_name(tensor, plan.name.c_str());
        gguf_add_tensor(output.get(), tensor);
        if (ggml_nbytes(tensor) != plan.size) {
            throw std::runtime_error("generated tensor size mismatch: " + plan.name);
        }
    }

    const std::string temp_output = output_path + ".tmp";
    if (!gguf_write_to_file(output.get(), temp_output.c_str(), true)) {
        throw std::runtime_error("failed to write output GGUF metadata");
    }
    std::ifstream source_file(model_path, std::ios::binary);
    std::ifstream spool_file(spool_path, std::ios::binary);
    std::ofstream output_file(temp_output, std::ios::binary | std::ios::app);
    if (!source_file || !spool_file || !output_file) throw std::runtime_error("failed to open GGUF payload streams");
    uint64_t cursor = 0;
    auto write_zeros = [&](uint64_t count) {
        std::array<char, 4096> zeros = {};
        while (count > 0) {
            const size_t chunk = static_cast<size_t>(std::min<uint64_t>(count, zeros.size()));
            output_file.write(zeros.data(), chunk);
            count -= chunk;
        }
    };
    auto copy_mapped = [&](const uint8_t * data, size_t size) {
        output_file.write(reinterpret_cast<const char *>(data), size);
    };
    auto copy_spool = [&](size_t offset, size_t size) {
        spool_file.clear();
        spool_file.seekg(static_cast<std::streamoff>(offset));
        std::vector<char> buffer(4 * 1024 * 1024);
        while (size > 0) {
            const size_t chunk = std::min(size, buffer.size());
            spool_file.read(buffer.data(), chunk);
            if (static_cast<size_t>(spool_file.gcount()) != chunk) throw std::runtime_error("failed to read GGUF spool");
            output_file.write(buffer.data(), chunk);
            size -= chunk;
        }
    };
    for (size_t i = 0; i < plans.size(); ++i) {
        const OutputTensor & plan = plans[i];
        const uint64_t target = gguf_get_tensor_offset(output.get(), static_cast<int64_t>(i));
        if (target < cursor) throw std::runtime_error("output GGUF tensor offsets overlap");
        write_zeros(target - cursor);
        if (plan.source == OutputTensor::Source::Original) {
            copy_mapped(source.tensor_data(model.tensors[static_cast<size_t>(plan.source_id)]), plan.size);
        } else {
            copy_spool(plan.spool_offset, plan.size);
        }
        cursor = target + plan.size;
    }
    const size_t alignment = gguf_get_alignment(output.get());
    write_zeros((alignment - cursor % alignment) % alignment);
    output_file.close();
    if (!output_file) throw std::runtime_error("failed to finish output GGUF");
    if (std::rename(temp_output.c_str(), output_path.c_str()) != 0) {
        std::remove(temp_output.c_str());
        throw std::runtime_error("failed to rename output GGUF: " + std::string(std::strerror(errno)));
    }
    std::fprintf(stderr, "[den2moee] wrote %s\n", output_path.c_str());
    std::fprintf(stderr, "[den2moee] peak CPU scratch: %zu MiB\n", (peak_scratch + (1u << 20) - 1) / (1u << 20));
}

} // namespace den2moee
