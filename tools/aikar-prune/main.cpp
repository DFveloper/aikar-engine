#include "dataset.h"
#include "hard-prune.h"
#include "ream.h"
#include "llama-model.h"
#include "llama-context.h"

#include "chat.h"
#include "build-info.h"
#include "common.h"
#include "log.h"
#include "moe-prune.h"

#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

constexpr int32_t routing_stats_version = COMMON_MOE_PRUNE_STATS_VERSION;

std::string profile_name(double ratio);

struct options {
    std::string command;
    std::string model;
    std::string dataset;
    std::string profile;
    std::string output;
    std::string output_dir;
    std::string importance_cache;
    std::string metric = "router-output";
    bool metric_explicit = false;
    aikar_ppl_mask mask = aikar_ppl_mask::ASSISTANT;
    std::vector<double> ratios;
    double max_layer_ratio = 0.25;
    int32_t seed = 42;
    int32_t n_ctx = 4096;
    int32_t n_batch = 512;
    int32_t n_ubatch = 512;
    int32_t n_threads = -1;
    int32_t dataset_threads = 0;
    int32_t n_gpu_layers = -1;
    bool cpu_moe = false;
    int32_t n_cpu_moe = -1;
    bool evaluate_ratios = true;
    std::string method = "reap";
    std::string ream_merging = "logits+weights";
    std::string ream_work_dir;
    std::string ream_activation_dir;
    int32_t target_experts = 64;
    int32_t ream_group_size = 16;
    int32_t ream_samples = 32768;
    int32_t ream_chunk = 64;
    int32_t ream_memory_mib = 4096;
    int32_t ream_expert_cache_mib = 8192;
    int32_t ream_input_cache_mib = 6144;
    bool dry_run = false;
    std::string save_logits;
    std::string reference_logits;
    std::string ream_compute_device = "cpu";
    bool ream_full_forward = false;
    bool ream_full_expert_forward = false;
    std::string ream_feature_precision = "f32";
};

struct route_layer_state {
    std::vector<int32_t> ids;
    std::vector<float> probabilities;
    int64_t n_used = 0;
    int64_t n_tokens = 0;
};

struct route_collector {
    int32_t n_expert = 0;
    bool collect_output_norm = false;
    common_moe_prune_stats stats;
    std::map<int32_t, route_layer_state> pending;
    uint64_t invalid_routing = 0;
    double entropy_sum = 0.0;
    uint64_t entropy_tokens = 0;
    int32_t ream_layer = -1;
    int32_t routing_layer = -1;
    std::ofstream * ream_inputs = nullptr;
    std::ofstream * ream_logits = nullptr;
    uint64_t ream_input_tokens = 0;
    uint64_t ream_logit_tokens = 0;
};

struct evaluation_result {
    std::array<double, 4> nll = {};
    std::array<int64_t, 4> evaluated = {};
    int64_t total_tokens = 0;
    int64_t processed_tokens = 0;
    double elapsed_seconds = 0.0;
    double throughput = 0.0;
    double router_load_imbalance = 0.0;
    double router_entropy = 0.0;
    uint64_t invalid_routing = 0;

    double ppl(aikar_ppl_mask mask) const {
        const size_t i = (size_t) mask;
        return evaluated[i] == 0 ? INFINITY : std::exp(nll[i] / evaluated[i]);
    }
};

struct importance_cache_data {
    common_moe_prune_model_info model;
    common_moe_prune_stats stats;
    evaluation_result baseline;
    std::string dataset_hash;
    std::string metric;
    int32_t n_ctx = 0;
    bool reap_available = false;
    int32_t routing_stats_version = 0;
    json execution = json::object();
    std::string tokenized_hash;
};

void usage() {
    std::cout <<
        "usage:\n"
        "  aikar-prune analyze --model MODEL --dataset DATA --ratios RATIO,... --output-dir DIR [options]\n"
        "  aikar-prune profiles --importance-cache CACHE --ratios RATIO,... --output-dir DIR [options]\n"
        "  aikar-prune inspect --model MODEL --profile PROFILE\n"
        "  aikar-prune hard --model MODEL --profile PROFILE --output MODEL [--dataset DATA]\n\n"
        "  aikar-prune hard --method ream --model MODEL --dataset DATA --output MODEL [options]\n\n"
        "  aikar-prune verify --model MODEL [--dataset DATA] [--output METRICS] [options]\n\n"
        "options:\n"
        "  --method reap|ream  (default: reap, unchanged legacy path)\n"
        "  --target-experts N --ream-group-size N (default: 64, 16)\n"
        "  --ream-merging logits|weights|logits+weights (default: logits+weights)\n"
        "  --ream-sequential  sequential replay is always enabled for REAM\n"
        "  --ream-feature-precision f32|f16  F16 GPU features use FP32 accumulation\n"
        "  --ream-full-expert-forward  reference all-token Down projection\n"
        "  --ream-full-forward  reference calibration without prefix graph truncation\n"
        "  --ream-calibration DATA  alias for --dataset\n"
        "  --ream-activation-samples N --ream-chunk-size N (default: 32768, 64)\n"
        "  --ream-max-memory-mib N --ream-work-dir DIR --dry-run\n"
        "  --ream-compute-device cpu|gpu  expert FFN and alignment cost backend (default: cpu)\n"
        "  --ream-activation-dir DIR  Separate temporary sampled activation storage\n"
        "  --ream-input-cache-mib N  GPU feature input cache (default: 6144, 0 disables)\n"
        "  --ream-expert-cache-mib N  GPU cache for calibration Expert weights (default: 8192, 0 disables)\n"
        "  --save-logits FILE | --reference-logits FILE  (verify: write baseline or measure KLD)\n"
        "  --metric router-output|reap|frequency\n"
        "  --ppl-mask all|assistant|reasoning|content\n"
        "  --max-layer-ratio RATIO\n"
        "  --importance-cache FNAME\n"
        "  --evaluate | --no-evaluate  evaluate each generated ratio (default: evaluate)\n"
        "  --seed N\n"
        "  --ctx-size N --batch-size N --ubatch-size N\n"
        "  --threads N --dataset-threads N --n-gpu-layers N\n"
        "  -cmoe, --cpu-moe  keep all routed MoE weights on the CPU\n"
        "  -ncmoe, --n-cpu-moe N  keep routed experts in the first N model layers on the CPU\n";
}

std::vector<double> parse_ratios(const std::string & value) {
    std::vector<double> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::runtime_error("empty pruning ratio");
        size_t used = 0;
        double ratio = std::stod(item, &used);
        if (used != item.size()) throw std::runtime_error("invalid pruning ratio: " + item);
        result.push_back(ratio);
    }
    return result;
}

options parse_options(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("missing subcommand");
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        usage();
        std::exit(0);
    }
    options result;
    result.command = argv[1];
    if (result.command != "analyze" && result.command != "profiles" && result.command != "inspect" && result.command != "hard" && result.command != "verify") {
        throw std::runtime_error("unknown subcommand: " + result.command);
    }
    auto value = [&](int & i) -> std::string {
        if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i - 1]);
        return argv[i];
    };
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(); std::exit(0); }
        else if (arg == "--model" || arg == "-m") result.model = value(i);
        else if (arg == "--dataset") result.dataset = value(i);
        else if (arg == "--ream-calibration") result.dataset = value(i);
        else if (arg == "--method") result.method = value(i);
        else if (arg == "--target-experts") result.target_experts = std::stoi(value(i));
        else if (arg == "--ream-group-size") result.ream_group_size = std::stoi(value(i));
        else if (arg == "--ream-merging") result.ream_merging = value(i);
        else if (arg == "--ream-activation-samples") result.ream_samples = std::stoi(value(i));
        else if (arg == "--ream-chunk-size") result.ream_chunk = std::stoi(value(i));
        else if (arg == "--ream-max-memory-mib") result.ream_memory_mib = std::stoi(value(i));
        else if (arg == "--ream-input-cache-mib") result.ream_input_cache_mib = std::stoi(value(i));
        else if (arg == "--ream-expert-cache-mib") result.ream_expert_cache_mib = std::stoi(value(i));
        else if (arg == "--ream-work-dir") result.ream_work_dir = value(i);
        else if (arg == "--ream-activation-dir") result.ream_activation_dir = value(i);
        else if (arg == "--ream-sequential") {}
        else if (arg == "--ream-feature-precision") result.ream_feature_precision = value(i);
        else if (arg == "--ream-full-expert-forward") result.ream_full_expert_forward = true;
        else if (arg == "--ream-full-forward") result.ream_full_forward = true;
        else if (arg == "--dry-run") result.dry_run = true;
        else if (arg == "--save-logits") result.save_logits = value(i);
        else if (arg == "--reference-logits") result.reference_logits = value(i);
        else if (arg == "--ream-compute-device") result.ream_compute_device = value(i);
        else if (arg == "--profile") result.profile = value(i);
        else if (arg == "--output") result.output = value(i);
        else if (arg == "--output-dir") result.output_dir = value(i);
        else if (arg == "--importance-cache") result.importance_cache = value(i);
        else if (arg == "--ratios") result.ratios = parse_ratios(value(i));
        else if (arg == "--metric") { result.metric = value(i); result.metric_explicit = true; }
        else if (arg == "--ppl-mask") result.mask = aikar_ppl_mask_parse(value(i));
        else if (arg == "--max-layer-ratio") result.max_layer_ratio = std::stod(value(i));
        else if (arg == "--seed") result.seed = std::stoi(value(i));
        else if (arg == "--ctx-size") result.n_ctx = std::stoi(value(i));
        else if (arg == "--batch-size") result.n_batch = std::stoi(value(i));
        else if (arg == "--ubatch-size") result.n_ubatch = std::stoi(value(i));
        else if (arg == "--threads") result.n_threads = std::stoi(value(i));
        else if (arg == "--dataset-threads") result.dataset_threads = std::stoi(value(i));
        else if (arg == "--n-gpu-layers" || arg == "-ngl") result.n_gpu_layers = std::stoi(value(i));
        else if (arg == "--cpu-moe" || arg == "-cmoe") result.cpu_moe = true;
        else if (arg == "--n-cpu-moe" || arg == "-ncmoe") {
            result.n_cpu_moe = std::stoi(value(i));
            if (result.n_cpu_moe < 0) throw std::runtime_error("n-cpu-moe must be non-negative");
        }
        else if (arg == "--evaluate") result.evaluate_ratios = true;
        else if (arg == "--no-evaluate") result.evaluate_ratios = false;
        else if (arg == "--validate") {}
        else throw std::runtime_error("unknown option: " + arg);
    }
    if (result.command != "profiles" && result.model.empty()) throw std::runtime_error("--model is required");
    if (result.command == "analyze" && (result.dataset.empty() || result.ratios.empty() || result.output_dir.empty())) {
        throw std::runtime_error("analyze requires --dataset, --ratios, and --output-dir");
    }
    if (result.command == "profiles" && (result.importance_cache.empty() || result.ratios.empty() || result.output_dir.empty())) {
        throw std::runtime_error("profiles requires --importance-cache, --ratios, and --output-dir");
    }
    if (result.method != "reap" && result.method != "ream") throw std::runtime_error("unsupported compression method");
    if (result.method == "ream" && result.command != "hard") throw std::runtime_error("REAM requires the hard subcommand");
    if (result.method == "ream" && (!result.profile.empty() || !result.importance_cache.empty())) throw std::runtime_error("REAM recalibrates each layer; profile and importance cache are unsupported");
    if (result.ream_merging != "logits" && result.ream_merging != "weights" && result.ream_merging != "logits+weights") throw std::runtime_error("unsupported REAM alignment mode");
    if (result.ream_compute_device != "cpu" && result.ream_compute_device != "gpu") throw std::runtime_error("unsupported REAM compute device");
    if (result.method == "ream" && result.metric_explicit && result.metric != "reap") throw std::runtime_error("REAM shares the existing REAP metric; use --metric reap");
    if ((result.command == "inspect" || (result.command == "hard" && result.method == "reap")) && result.profile.empty()) throw std::runtime_error("--profile is required");
    if (result.command == "hard" && result.output.empty() && !result.dry_run) throw std::runtime_error("hard requires --output");
    if (result.dry_run && result.method != "ream") throw std::runtime_error("--dry-run requires --method ream");
    if ((!result.save_logits.empty() || !result.reference_logits.empty()) && (result.command != "verify" || result.dataset.empty())) throw std::runtime_error("logit comparison requires verify with --dataset");
    if (result.ream_feature_precision != "f32" && result.ream_feature_precision != "f16") throw std::runtime_error("invalid REAM feature precision");
    if (result.ream_feature_precision == "f16" && result.ream_compute_device != "gpu") throw std::runtime_error("F16 REAM features require --ream-compute-device gpu");
    if (result.ream_group_size < 1 || result.ream_memory_mib < 1 || result.ream_expert_cache_mib < 0 || result.ream_input_cache_mib < 0) throw std::runtime_error("invalid REAM group size or memory budget");
    if (result.metric != "router-output" && result.metric != "reap" && result.metric != "frequency") {
        throw std::runtime_error("unsupported importance metric: " + result.metric);
    }
    if (result.n_ctx < 2 || result.n_batch < 1 || result.n_ubatch < 1) throw std::runtime_error("invalid context or batch size");
    if (result.dataset_threads < 0) throw std::runtime_error("dataset threads must be non-negative");
    if (result.cpu_moe && result.n_cpu_moe >= 0) throw std::runtime_error("--cpu-moe and --n-cpu-moe cannot be combined");
    if (result.command == "analyze" || result.command == "profiles") {
        if (!std::isfinite(result.max_layer_ratio) || result.max_layer_ratio <= 0.0 || result.max_layer_ratio >= 1.0) {
            throw std::runtime_error("invalid maximum layer ratio");
        }
        std::set<std::string> names;
        for (double ratio : result.ratios) {
            if (!std::isfinite(ratio) || ratio <= 0.0 || ratio > result.max_layer_ratio) throw std::runtime_error("invalid pruning ratio");
            if (!names.insert(profile_name(ratio)).second) throw std::runtime_error("ratios map to the same profile filename; use separate output directories");
        }
    }
    return result;
}

common_params make_common_params(const options & opts) {
    common_params params;
    params.model.path = opts.model;
    params.n_ctx = opts.n_ctx;
    params.n_batch = opts.n_batch;
    params.n_ubatch = opts.n_ubatch;
    params.n_gpu_layers = opts.n_gpu_layers;
    if (opts.cpu_moe) {
        params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
    } else if (opts.n_cpu_moe >= 0) {
        llm_add_n_cpu_ffn_overrides(opts.n_cpu_moe, LLM_FFN_EXPS_REGEX, params.tensor_buft_overrides);
    }
    if (!params.tensor_buft_overrides.empty()) {
        params.tensor_buft_overrides.push_back({ nullptr, nullptr });
    }
    params.cpuparams.n_threads = opts.n_threads;
    params.cpuparams_batch.n_threads = opts.n_threads;
    params.sampling.seed = opts.seed;
    params.warmup = false;
    return params;
}

int32_t tensor_layer(const char * name, const char * prefix) {
    int32_t layer = -1;
    std::string pattern = std::string(prefix) + "-%d";
    return sscanf(name, pattern.c_str(), &layer) == 1 ? layer : -1;
}

std::vector<uint8_t> tensor_bytes(ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    if (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer)) {
        memcpy(result.data(), tensor->data, result.size());
    } else {
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    }
    return result;
}

float tensor_float(const std::vector<uint8_t> & data, ggml_type type, size_t index) {
    if (type == GGML_TYPE_F32) return reinterpret_cast<const float *>(data.data())[index];
    if (type == GGML_TYPE_F16) return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(data.data())[index]);
    if (type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(reinterpret_cast<const ggml_bf16_t *>(data.data())[index]);
    throw std::runtime_error(std::string("unsupported calibration tensor type: ") + ggml_type_name(type));
}

bool named_layer_tensor(const std::string & name, const char * prefix) {
    const int32_t layer = tensor_layer(name.c_str(), prefix);
    return layer >= 0 && name == std::string(prefix) + "-" + std::to_string(layer);
}

bool route_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    route_collector & collector = *static_cast<route_collector *>(user_data);
    const std::string name = tensor->name;
    if (collector.routing_layer >= 0) {
        const auto id = "-" + std::to_string(collector.routing_layer);
        if (name != "ffn_moe_topk" + id && name != "ffn_moe_weights_norm" + id &&
            name != "ffn_moe_down" + id && name != "ffn_moe_down_scaled" + id && name != "ffn_moe_down_biased" + id) return !ask;
    }
    if (collector.ream_layer >= 0) {
        const std::string id = "-" + std::to_string(collector.ream_layer);
        const bool inputs = name == "ffn_norm_2" + id;
        const bool logits = name == "ffn_moe_logits" + id;
        if (inputs || logits) {
            if (ask) return true;
            if (tensor->ne[2] != 1 || tensor->ne[3] != 1 || tensor->nb[0] != ggml_type_size(tensor->type)) throw std::runtime_error("unsupported REAM capture layout");
            const auto data = tensor_bytes(tensor);
            std::vector<float> row(tensor->ne[0]);
            std::ofstream & stream = *(inputs ? collector.ream_inputs : collector.ream_logits);
            if (tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor)) {
                const auto * values = (const float *) data.data();
                for (size_t i = 0; i < data.size() / sizeof(float); ++i) if (!std::isfinite(values[i])) throw std::runtime_error("non-finite REAM calibration capture");
                stream.write((const char *) data.data(), data.size());
            } else for (int64_t t = 0; t < tensor->ne[1]; ++t) {
                for (int64_t d = 0; d < tensor->ne[0]; ++d) {
                    row[d] = tensor_float(data, tensor->type, (t * tensor->nb[1]) / ggml_type_size(tensor->type) + d);
                    if (!std::isfinite(row[d])) throw std::runtime_error("non-finite REAM calibration capture");
                }
                stream.write((const char *) row.data(), row.size() * sizeof(float));
            }
            if (!stream) throw std::runtime_error("REAM calibration disk write failed");
            (inputs ? collector.ream_input_tokens : collector.ream_logit_tokens) += tensor->ne[1];
            return true;
        }
        const bool layer_match = name == "ffn_moe_topk" + id || name == "ffn_moe_weights_norm" + id ||
            name == "ffn_moe_down" + id || name == "ffn_moe_down_scaled" + id || name == "ffn_moe_down_biased" + id;
        if (!layer_match) return !ask;
    }
    const bool wanted = named_layer_tensor(name, "ffn_moe_topk") || named_layer_tensor(name, "ffn_moe_weights_norm") ||
                        (collector.collect_output_norm && (named_layer_tensor(name, "ffn_moe_down") ||
                            named_layer_tensor(name, "ffn_moe_down_scaled") || named_layer_tensor(name, "ffn_moe_down_biased")));
    if (ask) return wanted;
    if (!wanted) return true;

    if (name.rfind("ffn_moe_topk-", 0) == 0) {
        const int32_t layer = tensor_layer(name.c_str(), "ffn_moe_topk");
        route_layer_state & state = collector.pending[layer];
        state.n_used = tensor->ne[0];
        state.n_tokens = tensor->ne[1];
        state.ids = common_moe_prune_selected_ids(tensor);
        state.probabilities.clear();
        return true;
    }
    if (name.rfind("ffn_moe_weights_norm-", 0) == 0) {
        const int32_t layer = tensor_layer(name.c_str(), "ffn_moe_weights_norm");
        const bool collect_reap = name == "ffn_moe_weights_norm-" + std::to_string(layer);
        if (!collect_reap) return true;
        route_layer_state & state = collector.pending[layer];
        if (state.ids.empty()) throw std::runtime_error("missing routing IDs for layer " + std::to_string(layer));
        const std::vector<uint8_t> data = tensor_bytes(tensor);
        state.probabilities.resize(state.n_used * state.n_tokens);
        for (size_t i = 0; i < state.probabilities.size(); ++i) state.probabilities[i] = tensor_float(data, tensor->type, i);
        auto & layer_stats = collector.stats[layer];
        if (layer_stats.empty()) layer_stats.resize(collector.n_expert);
        for (int64_t token = 0; token < state.n_tokens; ++token) {
            double sum = 0.0;
            for (int64_t slot = 0; slot < state.n_used; ++slot) {
                const size_t index = token * state.n_used + slot;
                const int32_t expert = state.ids[index];
                const float probability = state.probabilities[index];
                if (expert < 0 || expert >= collector.n_expert || !std::isfinite(probability) || probability < 0.0f) {
                    ++collector.invalid_routing;
                    throw std::runtime_error("invalid expert routing");
                }
                layer_stats[expert].record_selection(probability, collect_reap);
                sum += std::max(0.0f, probability);
            }
            if (sum > 0.0) {
                double entropy = 0.0;
                for (int64_t slot = 0; slot < state.n_used; ++slot) {
                    const double p = std::max(0.0f, state.probabilities[token * state.n_used + slot]) / sum;
                    if (p > 0.0) entropy -= p * std::log(p);
                }
                collector.entropy_sum += entropy;
                ++collector.entropy_tokens;
            }
        }
        return true;
    }

    const bool legacy = name.rfind("ffn_moe_down-", 0) == 0;
    const char * prefix = legacy ? "ffn_moe_down" :
        name.rfind("ffn_moe_down_scaled-", 0) == 0 ? "ffn_moe_down_scaled" : "ffn_moe_down_biased";
    const int32_t layer = tensor_layer(name.c_str(), prefix);
    route_layer_state & state = collector.pending[layer];
    if (state.ids.empty() || state.probabilities.empty() || tensor->ne[1] != state.n_used || tensor->ne[2] != state.n_tokens) {
        throw std::runtime_error("missing REAP routing data for layer " + std::to_string(layer));
    }
    auto & layer_stats = collector.stats[layer];
    if ((collector.ream_layer >= 0 || collector.routing_layer >= 0) && tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
        auto data = tensor_bytes(tensor);
        ggml_tensor host = *tensor;
        host.buffer = nullptr; host.data = (void *) data.data();
        common_moe_prune_collect_output(&host, state.ids, state.probabilities, layer_stats, true);
    } else common_moe_prune_collect_output(tensor, state.ids, state.probabilities, layer_stats, true);
    return true;
}

struct loaded_model {
    common_init_result_ptr init;
    llama_context_ptr context;
};

struct ream_expert_cache {
    struct binding { ggml_tensor * tensor; ggml_backend_buffer_t buffer; void * data; };
    struct entry {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<binding> bindings;
        ~entry() {
            for (const auto & b : bindings) { b.tensor->buffer = b.buffer; b.tensor->data = b.data; }
            if (buffer) ggml_backend_buffer_free(buffer);
            if (ctx) ggml_free(ctx);
        }
    };
    loaded_model & loaded;
    ggml_backend_t backend;
    uint64_t budget, used = 0, headroom;
    std::map<int32_t, std::unique_ptr<entry>> layers;
    uint64_t peak = 0, uploaded = 0, offloaded = 0;
    size_t peak_layers = 0;
    ~ream_expert_cache() { loaded.context.reset(); }

    void offload() {
        if (layers.empty()) return;
        loaded.context.reset();
        for (auto & item : layers) for (const auto & b : item.second->bindings) {
            ggml_tensor device = *b.tensor;
            b.tensor->buffer = b.buffer; b.tensor->data = b.data;
            ggml_backend_tensor_copy(&device, b.tensor);
        }
        offloaded += used;
        layers.clear(); used = 0;
    }

    bool add(int32_t layer) {
        if (!backend || !budget || layers.count(layer)) return false;
        auto item = std::make_unique<entry>();
        item->ctx = ggml_init({64 * 1024, nullptr, true});
        if (!item->ctx) throw std::runtime_error("cannot allocate REAM cache metadata");
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
        const auto prefix = "blk." + std::to_string(layer) + ".";
        for (const char * suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight"}) {
            auto * source = const_cast<ggml_tensor *>(loaded.init->model()->get_tensor((prefix + suffix).c_str()));
            if (!source) continue;
            auto * target = ggml_dup_tensor(item->ctx, source);
            copies.push_back({source, target});
        }
        const uint64_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(item->ctx, ggml_backend_get_default_buffer_type(backend));
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend), &free_bytes, &total_bytes);
        if (used >= budget || required > budget - used || free_bytes < required + headroom) {
            std::cerr << "REAM: calibration Expert cache keeps layer " << layer << " on CPU; cached_bytes=" << used << ", free_vram=" << free_bytes << '\n';
            return false;
        }
        loaded.context.reset();
        item->buffer = ggml_backend_alloc_ctx_tensors(item->ctx, backend);
        if (!item->buffer) throw std::runtime_error("cannot allocate REAM calibration Expert cache");
        ggml_backend_buffer_set_usage(item->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & pair : copies) ggml_backend_tensor_copy(pair.first, pair.second);
        for (const auto & pair : copies) {
            item->bindings.push_back({pair.first, pair.first->buffer, pair.first->data});
            pair.first->buffer = pair.second->buffer; pair.first->data = pair.second->data;
        }
        used += ggml_backend_buffer_get_size(item->buffer);
        layers[layer] = std::move(item);
        peak = std::max(peak, used); peak_layers = std::max(peak_layers, layers.size()); uploaded += ggml_backend_buffer_get_size(layers[layer]->buffer);
        std::cerr << "REAM: cached calibration layer " << layer << " Experts on GPU; cached_bytes=" << used << '\n';
        return true;
    }
};

loaded_model load_model(const options & opts, route_collector * collector, const common_moe_prune_profile * profile) {
    common_params params = make_common_params(opts);
    if (opts.method == "ream") {
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.no_extra_bufts = true;
        params.n_outputs_max = 1;
    }
    if (collector != nullptr) {
        params.cb_eval = route_callback;
        params.cb_eval_user_data = collector;
    }
    loaded_model result;
    result.init = common_init_from_params(params, true);
    if (!result.init || result.init->model() == nullptr) throw std::runtime_error("failed to load model");
    if (profile != nullptr) common_moe_prune_profile_apply(result.init->model(), *profile);
    llama_context_params cparams = common_context_params_to_llama(params);
    result.context.reset(llama_init_from_model(result.init->model(), cparams));
    if (!result.context) throw std::runtime_error("failed to create model context");
    return result;
}

double token_nll(const float * logits, int32_t n_vocab, llama_token target) {
    float max_logit = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) max_logit = std::max(max_logit, logits[i]);
    double sum = 0.0;
    for (int32_t i = 0; i < n_vocab; ++i) sum += std::exp((double) logits[i] - max_logit);
    return -((double) logits[target] - max_logit - std::log(sum));
}

evaluation_result evaluate(
        llama_context * context,
        const aikar_dataset & dataset,
        route_collector & collector,
        const options & opts,
        const std::string & label,
        const std::function<void(const float *, int32_t)> & observe_logits = {},
        bool calibration_only = false, uint64_t token_limit = 0,
        const std::function<void(uint64_t)> & after_batch = {}) {
    evaluation_result result;
    result.total_tokens = dataset.total_tokens;
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(context)));
    const auto started = std::chrono::steady_clock::now();
    auto last_progress = started;
    llama_batch batch = llama_batch_init(opts.n_batch, 0, 1);
    for (size_t record_index = 0; record_index < dataset.records.size() && (!token_limit || (uint64_t) result.processed_tokens < token_limit); ++record_index) {
        const aikar_dataset_record & record = dataset.records[record_index];
        for (size_t window_start = 0; window_start + 1 < record.tokens.size() && (!token_limit || (uint64_t) result.processed_tokens < token_limit); window_start += opts.n_ctx) {
            const size_t window_end = std::min(record.tokens.size(), window_start + (size_t) opts.n_ctx);
            llama_memory_clear(llama_get_memory(context), true);
            for (size_t batch_start = window_start; batch_start + 1 < window_end && (!token_limit || (uint64_t) result.processed_tokens < token_limit); batch_start += opts.n_batch) {
                size_t batch_end = std::min(window_end - 1, batch_start + (size_t) opts.n_batch);
                if (token_limit) batch_end = std::min<uint64_t>(batch_end, batch_start + token_limit - result.processed_tokens);
                common_batch_clear(batch);
                std::vector<size_t> targets;
                for (size_t i = batch_start; i < batch_end; ++i) {
                    bool need_logits = false;
                    if (calibration_only) need_logits = i + 1 == batch_end;
                    else for (size_t mask = 0; mask < 4; ++mask) need_logits |= aikar_token_is_evaluated(record, i + 1, (aikar_ppl_mask) mask);
                    common_batch_add(batch, record.tokens[i], (llama_pos) (i - window_start), { 0 }, need_logits);
                    if (need_logits && !calibration_only) targets.push_back(i + 1);
                }
                if (llama_decode(context, batch) != 0) {
                    llama_batch_free(batch);
                    throw std::runtime_error("model evaluation failed at JSONL line " + std::to_string(record.line));
                }
                const float * logits = llama_get_logits(context);
                for (size_t output = 0; output < targets.size(); ++output) {
                    if (observe_logits) observe_logits(logits + output * n_vocab, n_vocab);
                    const size_t target_index = targets[output];
                    const double nll = token_nll(logits + output * n_vocab, n_vocab, record.tokens[target_index]);
                    for (size_t mask = 0; mask < 4; ++mask) {
                        if (aikar_token_is_evaluated(record, target_index, (aikar_ppl_mask) mask)) {
                            result.nll[mask] += nll;
                            ++result.evaluated[mask];
                        }
                    }
                }
                result.processed_tokens += batch.n_tokens;
                if (after_batch) after_batch(result.processed_tokens);
                const auto now = std::chrono::steady_clock::now();
                if (now - last_progress >= std::chrono::seconds(5)) {
                    std::cerr << "aikar-prune: " << label << ": record " << record_index + 1 << '/' << dataset.records.size()
                              << ", processed " << result.processed_tokens << " tokens\n";
                    last_progress = now;
                }
            }
        }
    }
    llama_batch_free(batch);
    result.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    result.throughput = result.elapsed_seconds == 0.0 ? 0.0 : result.processed_tokens / result.elapsed_seconds;
    result.invalid_routing = collector.invalid_routing;
    result.router_entropy = collector.entropy_tokens == 0 ? 0.0 : collector.entropy_sum / collector.entropy_tokens;
    double imbalance_sum = 0.0;
    size_t imbalance_layers = 0;
    for (const auto & layer : collector.stats) {
        uint64_t total = 0;
        uint64_t maximum = 0;
        for (const auto & expert : layer.second) {
            total += expert.selection_count;
            maximum = std::max(maximum, expert.selection_count);
        }
        if (total > 0 && !layer.second.empty()) {
            imbalance_sum += maximum / ((double) total / layer.second.size());
            ++imbalance_layers;
        }
    }
    result.router_load_imbalance = imbalance_layers == 0 ? 0.0 : imbalance_sum / imbalance_layers;
    std::cerr << "aikar-prune: " << label << " complete: " << result.processed_tokens << " tokens in "
              << result.elapsed_seconds << " seconds (" << result.throughput << " tokens/s)\n";
    return result;
}

json result_json(const evaluation_result & result, aikar_ppl_mask primary) {
    return {
        { "ppl", result.ppl(primary) },
        { "ppl_all", result.ppl(aikar_ppl_mask::ALL) },
        { "ppl_assistant", result.ppl(aikar_ppl_mask::ASSISTANT) },
        { "ppl_reasoning", result.ppl(aikar_ppl_mask::REASONING) },
        { "ppl_content", result.ppl(aikar_ppl_mask::CONTENT) },
        { "evaluated_token_count", result.evaluated[(size_t) primary] },
        { "total_token_count", result.total_tokens },
        { "elapsed_seconds", result.elapsed_seconds },
        { "prompt_tokens_per_second", result.throughput },
        { "router_load_imbalance", result.router_load_imbalance },
        { "router_entropy", result.router_entropy },
        { "invalid_routing_count", result.invalid_routing },
    };
}

json stats_json(const common_moe_prune_stats & stats) {
    json result = json::object();
    for (const auto & layer : stats) {
        json experts = json::array();
        for (const auto & expert : layer.second) {
            experts.push_back({
                { "selection_count", expert.selection_count },
                { "probability_sum", expert.probability_sum },
                { "output_norm_sum", expert.output_norm_sum },
                { "weighted_output_sum", expert.weighted_output_sum },
                { "reap_count", expert.reap_count },
                { "reap_output_norm_sum", expert.reap_output_norm_sum },
                { "reap_sum", expert.reap_sum },
                { "reap_selection_count", expert.reap_selection_count },
            });
        }
        result[std::to_string(layer.first)] = experts;
    }
    return result;
}

common_moe_prune_stats parse_stats(const json & value) {
    common_moe_prune_stats result;
    for (auto it = value.begin(); it != value.end(); ++it) {
        auto & experts = result[std::stoi(it.key())];
        for (const auto & item : it.value()) {
            experts.push_back({
                item.at("selection_count").get<uint64_t>(), item.at("probability_sum").get<double>(),
                item.at("output_norm_sum").get<double>(), item.at("weighted_output_sum").get<double>(),
                item.value("reap_count", uint64_t(0)), item.value("reap_output_norm_sum", 0.0), item.value("reap_sum", 0.0),
                item.value("reap_selection_count", item.at("selection_count").get<uint64_t>()),
            });
        }
    }
    return result;
}

std::string importance_cache_path(const options & opts) {
    return opts.importance_cache.empty() ? opts.output_dir + "/importance-cache.json" : opts.importance_cache;
}

json importance_cache_json(const importance_cache_data & cache, aikar_ppl_mask primary) {
    return {
        { "format", "aikar-moe-prune-importance-cache" },
        { "version", 1 },
        { "model", {
            { "architecture", cache.model.architecture },
            { "model_hash", cache.model.model_hash },
            { "expert_tensor_hash", cache.model.expert_tensor_hash },
            { "layer_count", cache.model.layer_count },
            { "expert_count", cache.model.expert_count },
            { "experts_used", cache.model.experts_used },
            { "moe_layers", cache.model.moe_layers },
            { "expert_bytes", cache.model.expert_bytes },
        } },
        { "calibration", {
            { "dataset_hash", cache.dataset_hash },
            { "metric", cache.metric },
            { "reap_available", cache.reap_available },
            { "routing_stats_version", cache.routing_stats_version },
            { "execution", cache.execution },
            { "tokenized_hash", cache.tokenized_hash },
            { "ctx_size", cache.n_ctx },
            { "primary_ppl_mask", aikar_ppl_mask_name(primary) },
        } },
        { "baseline", {
            { "nll", cache.baseline.nll },
            { "evaluated", cache.baseline.evaluated },
            { "total_tokens", cache.baseline.total_tokens },
            { "processed_tokens", cache.baseline.processed_tokens },
            { "elapsed_seconds", cache.baseline.elapsed_seconds },
            { "throughput", cache.baseline.throughput },
            { "router_load_imbalance", cache.baseline.router_load_imbalance },
            { "router_entropy", cache.baseline.router_entropy },
            { "invalid_routing", cache.baseline.invalid_routing },
        } },
        { "stats", stats_json(cache.stats) },
    };
}

importance_cache_data load_importance_cache(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open importance cache: " + path);
    json root;
    in >> root;
    if (root.value("format", "") != "aikar-moe-prune-importance-cache" || root.value("version", 0) != 1) {
        throw std::runtime_error("unsupported importance cache format: " + path);
    }
    importance_cache_data result;
    const json & model = root.at("model");
    result.model.architecture = model.at("architecture").get<std::string>();
    result.model.model_hash = model.at("model_hash").get<std::string>();
    result.model.expert_tensor_hash = model.at("expert_tensor_hash").get<std::string>();
    result.model.layer_count = model.at("layer_count").get<int32_t>();
    result.model.expert_count = model.at("expert_count").get<int32_t>();
    result.model.experts_used = model.at("experts_used").get<int32_t>();
    result.model.moe_layers = model.at("moe_layers").get<std::vector<int32_t>>();
    result.model.expert_bytes = model.at("expert_bytes").get<uint64_t>();
    const json & calibration = root.at("calibration");
    result.dataset_hash = calibration.at("dataset_hash").get<std::string>();
    result.metric = calibration.at("metric").get<std::string>();
    result.reap_available = calibration.value("reap_available", false);
    result.routing_stats_version = calibration.value("routing_stats_version", 0);
    result.execution = calibration.value("execution", json::object());
    if (result.execution.is_object() && !result.execution.empty() &&
        !result.execution.contains("cpu_moe") && !result.execution.contains("n_cpu_moe")) {
        result.execution["cpu_moe"] = false;
        result.execution["n_cpu_moe"] = -1;
    }
    result.tokenized_hash = calibration.value("tokenized_hash", "");
    result.n_ctx = calibration.at("ctx_size").get<int32_t>();
    const json & baseline = root.at("baseline");
    result.baseline.nll = baseline.at("nll").get<std::array<double, 4>>();
    result.baseline.evaluated = baseline.at("evaluated").get<std::array<int64_t, 4>>();
    result.baseline.total_tokens = baseline.at("total_tokens").get<int64_t>();
    result.baseline.processed_tokens = baseline.at("processed_tokens").get<int64_t>();
    result.baseline.elapsed_seconds = baseline.at("elapsed_seconds").get<double>();
    result.baseline.throughput = baseline.at("throughput").get<double>();
    result.baseline.router_load_imbalance = baseline.at("router_load_imbalance").get<double>();
    result.baseline.router_entropy = baseline.at("router_entropy").get<double>();
    result.baseline.invalid_routing = baseline.at("invalid_routing").get<uint64_t>();
    result.stats = parse_stats(root.at("stats"));
    return result;
}

void write_json_atomic(const std::string & path, const json & value, const std::string & description) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    const std::string tmp = path + ".tmp";
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) throw std::runtime_error("failed to write " + description);
    out << value.dump(2) << '\n';
    out.close();
    if (!out || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("failed to replace " + description);
    }
}

json calibration_execution(const options & opts) {
    json environment = json::object();
    for (const char * key : { "CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS",
            "GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS", "GGML_BACKEND_DL_PATH" }) {
        const char * value = std::getenv(key);
        environment[key] = value ? json(value) : json(nullptr);
    }
    return {
        { "seed", opts.seed }, { "batch_size", opts.n_batch }, { "ubatch_size", opts.n_ubatch },
        { "n_gpu_layers", opts.n_gpu_layers }, { "cpu_moe", opts.cpu_moe }, { "n_cpu_moe", opts.n_cpu_moe },
        { "threads", opts.n_threads }, { "dataset_threads", opts.dataset_threads },
        { "build", llama_build_info() }, { "compiler", llama_compiler() }, { "build_target", llama_build_target() },
        { "backend_features", llama_print_system_info() }, { "environment", environment },
    };
}

void validate_importance_cache(const importance_cache_data & cache) {
    if (cache.routing_stats_version != routing_stats_version || !cache.reap_available) {
        throw std::runtime_error("cache has obsolete pruning statistics; recalibrate into a new cache");
    }
    if (!cache.execution.is_object() || cache.execution.empty() || cache.tokenized_hash.empty()) {
        throw std::runtime_error("cache lacks calibration provenance");
    }
    for (const char * key : { "seed", "batch_size", "ubatch_size", "n_gpu_layers", "cpu_moe", "n_cpu_moe", "threads", "dataset_threads",
            "build", "compiler", "build_target", "backend_features", "environment" }) {
        if (!cache.execution.contains(key)) throw std::runtime_error("cache lacks execution setting: " + std::string(key));
    }
    if (cache.baseline.processed_tokens <= 0 || cache.model.experts_used <= 0 || cache.model.expert_count < cache.model.experts_used ||
        cache.stats.size() != cache.model.moe_layers.size() || cache.baseline.invalid_routing != 0) {
        throw std::runtime_error("invalid calibration cache metadata");
    }
    if (cache.baseline.total_tokens < cache.baseline.processed_tokens) throw std::runtime_error("invalid calibration token totals");
    for (size_t mask = 0; mask < cache.baseline.nll.size(); ++mask) {
        const int64_t count = cache.baseline.evaluated[mask];
        const double nll = cache.baseline.nll[mask];
        if (count < 0 || count > cache.baseline.processed_tokens || !std::isfinite(nll) || nll < 0.0 || (count == 0 && nll != 0.0)) {
            throw std::runtime_error("invalid cached calibration loss");
        }
    }
    const uint64_t tokens = cache.baseline.processed_tokens;
    if (tokens > UINT64_MAX / cache.model.experts_used) throw std::runtime_error("calibration selection count overflow");
    for (int32_t layer : cache.model.moe_layers) {
        const auto found = cache.stats.find(layer);
        if (found == cache.stats.end() || found->second.size() != (size_t) cache.model.expert_count) {
            throw std::runtime_error("cache has missing experts or layers");
        }
        uint64_t selected = 0;
        for (const auto & stat : found->second) {
            if (stat.selection_count != stat.reap_selection_count || stat.reap_count != stat.selection_count ||
                stat.output_norm_sum != stat.reap_output_norm_sum || stat.weighted_output_sum != stat.reap_sum ||
                stat.selection_count > tokens || selected > UINT64_MAX - stat.selection_count) {
                throw std::runtime_error("cache has inconsistent expert measurements");
            }
            for (double value : { stat.probability_sum, stat.output_norm_sum, stat.weighted_output_sum }) {
                if (!std::isfinite(value) || value < 0.0) throw std::runtime_error("cache has invalid expert statistics");
            }
            if (stat.selection_count == 0 && (stat.probability_sum != 0.0 || stat.output_norm_sum != 0.0 || stat.weighted_output_sum != 0.0)) {
                throw std::runtime_error("cache has statistics for an unselected expert");
            }
            selected += stat.selection_count;
        }
        if (selected != tokens * cache.model.experts_used) throw std::runtime_error("cache has incomplete calibration routes");
    }
}

std::string importance_cache_mismatch(
        const importance_cache_data & cache,
        const common_moe_prune_model_info & model,
        const std::string & dataset_hash,
        const options & opts) {
    if (cache.model.model_hash != model.model_hash) return "model hash differs";
    if (cache.model.expert_tensor_hash != model.expert_tensor_hash) return "expert tensor hash differs";
    if (cache.model.architecture != model.architecture || cache.model.expert_count != model.expert_count ||
        cache.model.experts_used != model.experts_used || cache.model.moe_layers != model.moe_layers) return "model metadata differs";
    if (cache.dataset_hash != dataset_hash) return "dataset hash differs";
    if (cache.n_ctx != opts.n_ctx) return "context size differs";
    if (cache.routing_stats_version != routing_stats_version) return "cache has obsolete pruning statistics; recalibrate into a new cache";
    if (cache.execution != calibration_execution(opts)) return "calibration execution settings differ";
    if (opts.metric == "reap" && !cache.reap_available) return "cache lacks REAP statistics; recalibrate into a new cache";
    return {};
}

std::string profile_name(double ratio) {
    const int value = (int) std::lround(ratio * 100.0);
    std::ostringstream out;
    out << "profile-" << std::setw(3) << std::setfill('0') << value << ".json";
    return out.str();
}

void run_inspect(const options & opts) {
    const common_moe_prune_model_info model = common_moe_prune_inspect_model(opts.model);
    const common_moe_prune_profile profile = common_moe_prune_profile_load(opts.profile);
    common_moe_prune_profile_validate(profile, model);
    const size_t disabled = profile.layers.begin()->second.disabled_experts.size();
    const int32_t surviving = model.expert_count - disabled;
    std::cout << "compatible: yes\narchitecture: " << model.architecture << "\nMoE layers: " << model.moe_layers.size()
              << "\ndisabled experts per layer: " << disabled << "\nsurviving experts per layer: " << surviving
              << "\neffective ratio: " << (double) disabled / model.expert_count
              << "\nexpected expert/router savings: " << (uint64_t) (model.expert_bytes * ((double) disabled / model.expert_count)) << " bytes\n";
    for (const auto & layer : profile.layers) {
        std::cout << "layer " << layer.first << " disabled:";
        for (int32_t expert : layer.second.disabled_experts) std::cout << ' ' << expert;
        std::cout << '\n';
    }
}

std::vector<common_moe_prune_profile> make_and_write_profiles(
        const options & opts,
        const importance_cache_data & cache) {
    if (opts.metric == "reap" && !cache.reap_available) {
        throw std::runtime_error("cache lacks REAP statistics; recalibrate into a new cache");
    }
    validate_importance_cache(cache);
    std::set<std::string> names;
    for (double ratio : opts.ratios) {
        if (!names.insert(profile_name(ratio)).second) throw std::runtime_error("ratios map to the same profile filename; use separate output directories");
    }
    if (opts.metric == "router-output") {
        for (const auto & layer : cache.stats) {
            if (std::all_of(layer.second.begin(), layer.second.end(), [](const common_moe_prune_expert_stats & s) { return s.importance() == 0.0; })) {
                std::cerr << "aikar-prune: warning: router-output scores are all zero in layer " << layer.first << "; expert ID breaks ties\n";
            }
        }
    }
    const int64_t evaluated_tokens = cache.baseline.evaluated[(size_t) opts.mask];
    if (evaluated_tokens == 0) throw std::runtime_error("the selected perplexity mask evaluates zero tokens");
    std::vector<common_moe_prune_profile> profiles = common_moe_prune_make_profiles(
        cache.model, cache.stats, opts.ratios, opts.max_layer_ratio, cache.dataset_hash,
        aikar_ppl_mask_name(opts.mask), opts.metric, evaluated_tokens);
    for (common_moe_prune_profile & profile : profiles) {
        profile.calibration_tokens = cache.baseline.processed_tokens;
        profile.calibration_collector_version = cache.routing_stats_version;
        profile.calibration_tokenized_hash = cache.tokenized_hash;
        const std::string execution = cache.execution.dump();
        profile.calibration_execution_hash = common_moe_prune_sha256_data(execution.data(), execution.size());
        const std::string fingerprint = json({ { "model", cache.model.model_hash }, { "dataset", cache.dataset_hash },
            { "tokens", cache.tokenized_hash }, { "execution", cache.execution }, { "context", cache.n_ctx },
            { "processed_tokens", cache.baseline.processed_tokens }, { "collector_version", cache.routing_stats_version } }).dump();
        profile.calibration_fingerprint = common_moe_prune_sha256_data(fingerprint.data(), fingerprint.size());
        profile.calibration_context = cache.n_ctx;
        profile.calibration_seed = cache.execution.value("seed", int32_t(-1));
        profile.calibration_batch = cache.execution.value("batch_size", int32_t(0));
        profile.calibration_ubatch = cache.execution.value("ubatch_size", int32_t(0));
        common_moe_prune_profile_write(profile, opts.output_dir + "/" + profile_name(profile.requested_ratio));
        std::cerr << "aikar-prune: calibration " << profile.calibration_fingerprint << ", model " << cache.model.model_hash
                  << ", dataset " << cache.dataset_hash << ", tokens " << cache.baseline.processed_tokens
                  << ", seed " << profile.calibration_seed << ", metric " << profile.metric << ", ratio " << profile.requested_ratio << '\n';
    }
    return profiles;
}

void run_profiles(const options & opts) {
    std::filesystem::create_directories(opts.output_dir);
    const importance_cache_data cache = load_importance_cache(opts.importance_cache);
    options profile_opts = opts;
    if (!opts.metric_explicit) profile_opts.metric = cache.metric;
    const std::vector<common_moe_prune_profile> profiles = make_and_write_profiles(profile_opts, cache);
    json result = {
        { "format", "aikar-moe-prune-profile-generation" },
        { "version", 1 },
        { "importance_cache", opts.importance_cache },
        { "model_hash", cache.model.model_hash },
        { "expert_tensor_hash", cache.model.expert_tensor_hash },
        { "dataset_hash", cache.dataset_hash },
        { "ppl_mask", aikar_ppl_mask_name(opts.mask) },
        { "metric", profile_opts.metric },
        { "profiles", json::array() },
    };
    for (const common_moe_prune_profile & profile : profiles) {
        result["profiles"].push_back({
            { "requested_ratio", profile.requested_ratio },
            { "actual_ratio", profile.actual_ratio },
            { "profile", profile_name(profile.requested_ratio) },
        });
    }
    write_json_atomic(opts.output_dir + "/profiles.json", result, "profile generation report");
    std::cout << "generated " << profiles.size() << " profiles from " << opts.importance_cache << '\n';
}

void run_analyze(const options & opts) {
    std::filesystem::create_directories(opts.output_dir);
    bool model_cache_hit = false;
    const common_moe_prune_model_info model_info = common_moe_prune_inspect_model_cached(
        opts.model, opts.output_dir + "/model-info-cache.json", &model_cache_hit);
    std::cerr << "aikar-prune: " << (model_cache_hit ? "reused model hash cache" : "hashed GGUF and saved model cache") << '\n';
    aikar_dataset dataset;
    {
        llama_model_params params = llama_model_default_params();
        params.vocab_only = true;
        llama_model_ptr vocabulary(llama_model_load_from_file(opts.model.c_str(), params));
        if (!vocabulary) throw std::runtime_error("failed to load calibration vocabulary");
        auto templates = common_chat_templates_init(vocabulary.get(), "");
        dataset = aikar_dataset_load(opts.dataset, vocabulary.get(), templates.get(), opts.dataset_threads);
    }
    const std::string tokenized_hash = aikar_dataset_fingerprint(dataset);
    const std::string dataset_hash = common_moe_prune_sha256_file(opts.dataset);
    const std::string cache_path = importance_cache_path(opts);
    std::optional<importance_cache_data> cache;
    if (std::filesystem::exists(cache_path)) {
        std::optional<importance_cache_data> loaded;
        std::string mismatch;
        try {
            loaded = load_importance_cache(cache_path);
            mismatch = importance_cache_mismatch(*loaded, model_info, dataset_hash, opts);
            if (mismatch.empty() && loaded->tokenized_hash != tokenized_hash) mismatch = "tokenized calibration inputs differ";
            if (mismatch.empty()) validate_importance_cache(*loaded);
        } catch (const std::exception & e) {
            mismatch = e.what();
        }
        if (mismatch.empty()) {
            cache = std::move(*loaded);
            std::cerr << "aikar-prune: loaded importance cache " << cache_path << '\n';
        } else if (!opts.importance_cache.empty()) {
            throw std::runtime_error("importance cache is incompatible: " + mismatch);
        } else {
            std::cerr << "aikar-prune: ignoring incompatible automatic importance cache: " << mismatch << '\n';
        }
    }
    if (!cache) {
        route_collector baseline_collector;
        baseline_collector.n_expert = model_info.expert_count;
        baseline_collector.collect_output_norm = true;
        std::cerr << "aikar-prune: loading baseline model\n";
        loaded_model baseline_model = load_model(opts, &baseline_collector, nullptr);
        std::cerr << "aikar-prune: dataset contains " << dataset.records.size() << " records and " << dataset.total_tokens << " tokens\n";
        importance_cache_data created;
        created.model = model_info;
        created.baseline = evaluate(baseline_model.context.get(), dataset, baseline_collector, opts, "baseline");
        created.stats = baseline_collector.stats;
        for (int32_t layer : model_info.moe_layers) {
            const auto found = created.stats.find(layer);
            if (found == created.stats.end()) throw std::runtime_error("missing REAP layer " + std::to_string(layer));
            for (size_t expert = 0; expert < found->second.size(); ++expert) {
                const auto & stat = found->second[expert];
                if (stat.reap_count != stat.reap_selection_count) {
                    throw std::runtime_error("incomplete REAP outputs for layer " + std::to_string(layer) +
                        ", expert " + std::to_string(expert) + ": selected " + std::to_string(stat.reap_selection_count) +
                        ", measured " + std::to_string(stat.reap_count));
                }
            }
        }
        created.reap_available = true;
        created.routing_stats_version = routing_stats_version;
        created.tokenized_hash = aikar_dataset_fingerprint(dataset);
        created.execution = calibration_execution(opts);
        created.dataset_hash = dataset_hash;
        created.metric = opts.metric;
        created.n_ctx = opts.n_ctx;
        validate_importance_cache(created);
        write_json_atomic(cache_path, importance_cache_json(created, opts.mask), "importance cache");
        cache = std::move(created);
        std::cerr << "aikar-prune: saved importance cache " << cache_path << '\n';
    }
    const evaluation_result & baseline = cache->baseline;
    std::cerr << "aikar-prune: released baseline model before pruned evaluations\n";
    std::vector<common_moe_prune_profile> profiles = make_and_write_profiles(opts, *cache);

    json analysis = {
        { "format", "aikar-moe-prune-analysis" },
        { "version", 1 },
        { "model", {
            { "architecture", model_info.architecture },
            { "model_hash", model_info.model_hash },
            { "expert_tensor_hash", model_info.expert_tensor_hash },
            { "expert_count", model_info.expert_count },
            { "experts_used", model_info.experts_used },
        } },
        { "baseline", result_json(baseline, opts.mask) },
        { "evaluation_enabled", opts.evaluate_ratios },
        { "metric", opts.metric },
        { "importance_cache", cache_path },
        { "calibration", importance_cache_json(*cache, opts.mask).at("calibration") },
        { "ratios", json::array() },
        { "importance", json::object() },
    };
    for (const auto & layer : cache->stats) {
        json experts = json::array();
        uint64_t layer_total = 0;
        for (const auto & stat : layer.second) layer_total += stat.selection_count;
        for (size_t expert = 0; expert < layer.second.size(); ++expert) {
            const auto & stat = layer.second[expert];
            experts.push_back({
                { "expert", expert },
                { "selection_count", cache->reap_available ? stat.reap_selection_count : stat.selection_count },
                { "legacy_selection_count", stat.selection_count },
                { "selection_frequency", baseline.processed_tokens == 0 ? 0.0 : (double) stat.selection_count / baseline.processed_tokens },
                { "routing_slot_fraction", layer_total == 0 ? 0.0 : (double) stat.selection_count / layer_total },
                { "router_probability_sum", stat.probability_sum },
                { "mean_router_probability", stat.mean_probability() },
                { "mean_output_activation_norm", stat.mean_output_norm() },
                { "weighted_output_importance", stat.importance() },
                { "average_gate_weight", stat.mean_probability() },
                { "average_output_norm", cache->reap_available ? json(stat.mean_reap_output_norm()) : json(nullptr) },
                { "REAP_score", cache->reap_available ? json(stat.reap_score()) : json(nullptr) },
                { "existing_aikar_score", stat.importance() },
            });
        }
        analysis["importance"][std::to_string(layer.first)] = experts;
    }

    std::ofstream csv(opts.output_dir + "/analysis.csv", std::ios::trunc);
    csv << "requested_ratio,actual_ratio,baseline_ppl,pruned_ppl,absolute_delta,relative_delta_percent,evaluated_tokens,total_tokens,elapsed_seconds,tokens_per_second,pruned_experts,remaining_experts,router_load_imbalance,router_entropy,invalid_routing_count\n";
    for (common_moe_prune_profile & profile : profiles) {
        const std::string path = opts.output_dir + "/" + profile_name(profile.requested_ratio);
        const int32_t per_layer_pruned = profile.layers.begin()->second.disabled_experts.size();
        json per_layer = json::object();
        for (const auto & layer : profile.layers) per_layer[std::to_string(layer.first)] = layer.second.disabled_experts.size();
        json row = {
            { "requested_ratio", profile.requested_ratio },
            { "actual_ratio", profile.actual_ratio },
            { "number_of_pruned_experts", per_layer_pruned * profile.layers.size() },
            { "number_of_remaining_experts", (model_info.expert_count - per_layer_pruned) * profile.layers.size() },
            { "per_layer_pruned_expert_count", per_layer },
            { "profile", std::filesystem::path(path).filename().string() },
            { "metric", profile.metric },
            { "expected_expert_bytes_removed", (uint64_t) (model_info.expert_bytes * profile.actual_ratio) },
            { "evaluated", false },
        };
        if (!opts.evaluate_ratios) {
            analysis["ratios"].push_back(row);
            csv << profile.requested_ratio << ',' << profile.actual_ratio << ',' << baseline.ppl(opts.mask) << ",,,,,,,"
                << per_layer_pruned * profile.layers.size() << ',' << (model_info.expert_count - per_layer_pruned) * profile.layers.size() << ",,,\n";
            continue;
        }
        route_collector collector;
        collector.n_expert = model_info.expert_count;
        std::cerr << "aikar-prune: loading model for ratio " << profile.requested_ratio << '\n';
        loaded_model pruned_model = load_model(opts, &collector, &profile);
        if (dataset.records.empty()) {
            std::cerr << "aikar-prune: loading and tokenizing dataset\n";
            common_chat_templates_ptr templates = common_chat_templates_init(pruned_model.init->model(), "");
            dataset = aikar_dataset_load(opts.dataset, pruned_model.init->model(), templates.get(), opts.dataset_threads);
            std::cerr << "aikar-prune: dataset contains " << dataset.records.size() << " records and " << dataset.total_tokens << " tokens\n";
        }
        const evaluation_result pruned = evaluate(
            pruned_model.context.get(), dataset, collector, opts, "ratio " + std::to_string(profile.requested_ratio));
        const double baseline_ppl = baseline.ppl(opts.mask);
        const double pruned_ppl = pruned.ppl(opts.mask);
        const double delta = pruned_ppl - baseline_ppl;
        const double relative = baseline_ppl == 0.0 ? 0.0 : delta * 100.0 / baseline_ppl;
        row.update(result_json(pruned, opts.mask));
        row["requested_ratio"] = profile.requested_ratio;
        row["actual_ratio"] = profile.actual_ratio;
        row["absolute_perplexity_delta"] = delta;
        row["relative_perplexity_delta_percent"] = relative;
        row["number_of_pruned_experts"] = per_layer_pruned * profile.layers.size();
        row["number_of_remaining_experts"] = (model_info.expert_count - per_layer_pruned) * profile.layers.size();
        row["per_layer_pruned_expert_count"] = per_layer;
        row["profile"] = std::filesystem::path(path).filename().string();
        row["evaluated"] = true;
        analysis["ratios"].push_back(row);
        csv << profile.requested_ratio << ',' << profile.actual_ratio << ',' << baseline_ppl << ',' << pruned_ppl << ',' << delta << ',' << relative << ','
            << pruned.evaluated[(size_t) opts.mask] << ',' << pruned.total_tokens << ',' << pruned.elapsed_seconds << ',' << pruned.throughput << ','
            << per_layer_pruned * profile.layers.size() << ',' << (model_info.expert_count - per_layer_pruned) * profile.layers.size() << ','
            << pruned.router_load_imbalance << ',' << pruned.router_entropy << ',' << pruned.invalid_routing << '\n';
        if (model_info.expert_count - per_layer_pruned < model_info.experts_used * 2) {
            std::cerr << "warning: ratio " << profile.requested_ratio << " leaves fewer than 2x router Top-K experts\n";
        }
    }
    std::ofstream out(opts.output_dir + "/analysis.json", std::ios::trunc);
    out << analysis.dump(2) << '\n';
    std::ofstream summary(opts.output_dir + "/README.txt", std::ios::trunc);
    summary << "Gemma 4 26B A4B static MoE pruning analysis\nMetric: " << opts.metric << "\nBaseline perplexity (" << aikar_ppl_mask_name(opts.mask) << "): " << baseline.ppl(opts.mask) << "\n";
    for (const auto & row : analysis["ratios"]) {
        summary << "ratio " << row["requested_ratio"];
        if (row["evaluated"].get<bool>()) summary << ": ppl " << row["ppl"] << ", delta " << row["absolute_perplexity_delta"];
        else summary << ": evaluation skipped";
        summary << '\n';
    }
}

std::vector<float> ream_read_floats(std::ifstream & stream, size_t count) {
    std::vector<float> result(count);
    stream.read((char *) result.data(), count * sizeof(float));
    if ((size_t) stream.gcount() != count * sizeof(float)) throw std::runtime_error("short REAM activation file read");
    return result;
}

std::vector<float> ream_load_hidden(const std::string & path, size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("missing sampled REAM activations");
    return ream_read_floats(in, count);
}

std::vector<float> ream_group_distances(const std::vector<float> & means, const std::string & logit_path,
                                       uint64_t tokens, int32_t experts, int32_t embedding) {
    std::vector<double> gram((size_t) experts * experts, 0.0);
    std::ifstream in(logit_path, std::ios::binary);
    for (uint64_t t = 0; t < tokens; ++t) {
        const auto row = ream_read_floats(in, experts);
        for (int32_t i = 0; i < experts; ++i) for (int32_t j = i; j < experts; ++j) {
            gram[i * experts + j] += (double) row[i] * row[j];
        }
    }
    std::vector<float> gate_dist((size_t) experts * experts);
    double maximum = 0.0;
    for (int32_t i = 0; i < experts; ++i) for (int32_t j = i + 1; j < experts; ++j) {
        const double ni = std::sqrt(gram[i * experts + i]) + 1e-8;
        const double nj = std::sqrt(gram[j * experts + j]) + 1e-8;
        const double d = std::sqrt(std::max(0.0, gram[i * experts + i] / (ni * ni) +
            gram[j * experts + j] / (nj * nj) - 2 * gram[i * experts + j] / (ni * nj)));
        gate_dist[i * experts + j] = gate_dist[j * experts + i] = d;
        maximum = std::max(maximum, d);
    }
    std::vector<float> distance((size_t) experts * experts, 0.0f);
    for (int32_t i = 0; i < experts; ++i) for (int32_t j = i + 1; j < experts; ++j) {
        double dot = 0.0, ni = 0.0, nj = 0.0;
        for (int32_t d = 0; d < embedding; ++d) {
            const double a = means[i * embedding + d], b = means[j * embedding + d];
            dot += a * b; ni += a * a; nj += b * b;
        }
        const double cosine = std::clamp(dot / (std::max(1e-8, std::sqrt(ni)) * std::max(1e-8, std::sqrt(nj))), -1.0, 1.0);
        const double output_distance = (1.0 - cosine) / 2.0;
        const double router_distance = maximum > 0 ? gate_dist[i * experts + j] / maximum : 0.0;
        distance[i * experts + j] = distance[j * experts + i] = (output_distance + router_distance) / 2.0;
    }
    return distance;
}

void ream_set_expert(llama_model * model, int32_t layer, int32_t center, const aikar_ream_expert & merged) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    auto set = [&](const std::string & name, const std::vector<float> & values) {
        auto * tensor = const_cast<ggml_tensor *>(model->get_tensor(name.c_str()));
        if (!tensor) throw std::runtime_error("missing REAM destination tensor: " + name);
        const auto bytes = aikar_ream_encode(values, tensor->type, tensor->ne[0]);
        const size_t slice = ggml_row_size(tensor->type, tensor->ne[0]) * tensor->ne[1];
        if (bytes.size() != slice) throw std::runtime_error("REAM encoded slice size differs");
        ggml_backend_tensor_set(tensor, bytes.data(), center * slice, slice);
    };
    if (model->get_tensor((prefix + "ffn_gate_up_exps.weight").c_str())) {
        auto fused = merged.gate;
        fused.insert(fused.end(), merged.up.begin(), merged.up.end());
        set(prefix + "ffn_gate_up_exps.weight", fused);
    } else {
        set(prefix + "ffn_gate_exps.weight", merged.gate);
        set(prefix + "ffn_up_exps.weight", merged.up);
    }
    set(prefix + "ffn_down_exps.weight", merged.down);
    auto * scale = const_cast<ggml_tensor *>(model->get_tensor((prefix + "ffn_down_exps.scale").c_str()));
    if (scale) {
        const auto one = aikar_ream_encode({1.0f}, scale->type, 1);
        ggml_backend_tensor_set(scale, one.data(), center * one.size(), one.size());
    }
}

void run_ream(const options & opts) {
    const auto layout = aikar_ream_inspect(opts.model, opts.target_experts, opts.ream_samples, opts.ream_chunk);
    if ((int64_t) opts.target_experts * opts.ream_group_size < layout.experts) throw std::runtime_error("REAM groups cannot cover all experts");
    const uint64_t estimated_output = layout.source_bytes - layout.expert_bytes + layout.expert_bytes * opts.target_experts / layout.experts;
    json report = {{"method", "ream"}, {"architecture", "gemma4"}, {"source_experts", layout.experts},
        {"target_experts", opts.target_experts}, {"top_k", layout.top_k}, {"layers", layout.layers}, {"calibration", opts.dataset},
        {"source_bytes", layout.source_bytes}, {"estimated_output_bytes", estimated_output},
        {"expert_reduction_fraction", 1.0 - (double) opts.target_experts / layout.experts},
        {"workspace_ram_bytes_estimate", layout.workspace_bytes}, {"inference_weights_ram_bytes_upper_bound", layout.source_bytes},
        {"hungarian_matrix_bytes", (uint64_t) layout.hidden * layout.hidden * sizeof(float)},
        {"activation_disk_bytes_upper_bound_per_layer", (uint64_t) layout.experts * layout.hidden * opts.ream_samples * sizeof(float)},
        {"calibration_disk_bytes_per_token", (layout.embedding + layout.experts) * sizeof(float) + layout.experts * sizeof(double)},
        {"calibration_forward", opts.ream_full_forward ? "full reference, no intermediate PPL" : "prefix through target layer, no intermediate PPL"},
        {"calibration_expert_cache_budget_bytes", (uint64_t) opts.ream_expert_cache_mib * 1024 * 1024},
        {"feature_input_cache_budget_bytes", (uint64_t) opts.ream_input_cache_mib * 1024 * 1024},
        {"device_memory_note", "Expert cache uses original quantization within budget and free VRAM headroom; other Experts stay on CPU; use -ngl 0 for CPU"},
        {"formats", "F32,F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,Q2_K,Q3_K,Q4_K,Q5_K,Q6_K; other expert formats rejected"},
        {"sequential", true}, {"merging", opts.ream_merging}, {"weight_reduction", "none (exact full weight features)"},
        {"compute_device", opts.ream_compute_device}, {"compute_device_workspace_bytes_upper_bound", opts.ream_compute_device == "gpu" ? layout.workspace_bytes : 0},
        {"expert_feature_precision", opts.ream_feature_precision}, {"expert_feature_accumulation", "f32"}, {"feature_mean_accumulation", "GPU f32 chunk sums, CPU f64 totals; CPU path f64"}, {"probability_cache_layout", "expert-major f64"},
        {"expert_feature_forward", opts.ream_full_expert_forward ? "full reference" : "weighted hidden mean then one linear Down projection"},
        {"group_size", opts.ream_group_size}, {"activation_samples", opts.ream_samples}, {"seed", opts.seed},
        {"feature_chunk_size", opts.ream_chunk}, {"calibration_context_size", opts.n_ctx},
        {"calibration_batch_size", opts.n_batch}, {"calibration_ubatch_size", opts.n_ubatch},
        {"calibration_gpu_layers", opts.n_gpu_layers},
        {"reap_semantics", "existing engine normalized Top-K gate times effective expert output norm, mean over selected tokens"},
        {"reference_differences", {"no weight PCA", "common deterministic activation samples", "engine REAP saliency unchanged", "token-weighted output means"}}};
    std::cout << report.dump(2) << '\n';
    if (opts.dry_run) return;
    const auto output_directory = std::filesystem::absolute(opts.output).parent_path();
    const uint64_t output_free_bytes = std::filesystem::space(output_directory).available;
    if (output_free_bytes < estimated_output + 16 * 1024 * 1024) {
        throw std::runtime_error("REAM output directory has insufficient free space: available=" + std::to_string(output_free_bytes) +
            ", estimated output=" + std::to_string(estimated_output));
    }
    if (layout.workspace_bytes > (uint64_t) opts.ream_memory_mib * 1024 * 1024) throw std::runtime_error("REAM estimated workspace exceeds --ream-max-memory-mib; reduce activation samples/chunk size");
    if (std::filesystem::exists(opts.output) || std::filesystem::exists(opts.output + ".report.json")) throw std::runtime_error("REAM output already exists; select a new path");
    if (opts.target_experts == layout.experts) {
        std::map<int32_t, aikar_ream_groups> groups;
        for (int32_t layer : layout.layers) groups[layer] = aikar_ream_pseudo_group(std::vector<double>(layout.experts, 0),
            std::vector<float>((size_t) layout.experts * layout.experts, 0), layout.experts, opts.ream_group_size);
        report["identity"] = true;
        aikar_ream_export(opts.model, opts.output, groups, report.dump(), {});
        return;
    }
    if (opts.dataset.empty()) throw std::runtime_error("REAM compression requires --dataset or --ream-calibration");
    struct compute_guard {
        ggml_backend_t backend = nullptr;
        ~compute_guard() { aikar_ream_set_backend(nullptr); if (backend) ggml_backend_free(backend); }
    } compute;
    if (opts.ream_compute_device == "gpu") {
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) throw std::runtime_error("REAM GPU compute requested but no GPU backend is available");
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        if (layout.workspace_bytes > free_bytes) throw std::runtime_error("REAM device workspace exceeds available VRAM: free=" + std::to_string(free_bytes) + ", estimate=" + std::to_string(layout.workspace_bytes));
        compute.backend = ggml_backend_dev_init(device, nullptr);
        if (!compute.backend) throw std::runtime_error("cannot initialize REAM compute backend");
        aikar_ream_set_backend(compute.backend);
    }
    const std::string work = opts.ream_work_dir.empty() ? opts.output + ".ream-work" : opts.ream_work_dir;
    const auto work_path = std::filesystem::weakly_canonical(work);
    const auto output_path = std::filesystem::weakly_canonical(opts.output);
    auto part_work = work_path.begin(), part_output = output_path.begin();
    while (part_work != work_path.end() && part_output != output_path.end() && *part_work == *part_output) { ++part_work; ++part_output; }
    if (part_work == work_path.end()) throw std::runtime_error("REAM output must be outside the temporary work directory");
    if (!std::filesystem::create_directory(work)) throw std::runtime_error("REAM work directory already exists; select an unused path");
    struct work_guard {
        std::string path;
        ~work_guard() { if (!path.empty()) { std::error_code error; std::filesystem::remove_all(path, error); } }
    } guard {work};
    const std::string activation_work = opts.ream_activation_dir.empty() ? work : opts.ream_activation_dir;
    work_guard activation_guard {""};
    if (!opts.ream_activation_dir.empty()) {
        const auto activation_path = std::filesystem::weakly_canonical(activation_work);
        auto part_activation = activation_path.begin();
        part_output = output_path.begin();
        while (part_activation != activation_path.end() && part_output != output_path.end() && *part_activation == *part_output) { ++part_activation; ++part_output; }
        if (part_activation == activation_path.end()) throw std::runtime_error("REAM output must be outside the activation directory");
        if (!std::filesystem::create_directory(activation_work)) throw std::runtime_error("REAM activation directory already exists; select an unused path");
        activation_guard.path = activation_work;
    }
    options runtime_opts = opts;
    runtime_opts.cpu_moe = true; runtime_opts.n_cpu_moe = -1;
    route_collector collector;
    collector.n_expert = layout.experts; collector.collect_output_norm = true;
    loaded_model loaded = load_model(runtime_opts, &collector, nullptr);
    ream_expert_cache expert_cache {loaded, opts.n_gpu_layers == 0 ? nullptr : compute.backend,
        (uint64_t) opts.ream_expert_cache_mib * 1024 * 1024, 0, std::max<uint64_t>(2ULL * 1024 * 1024 * 1024, layout.workspace_bytes + 1024 * 1024 * 1024), {}};
    auto create_context = [&]() {
        if (loaded.context) return;
        auto params = make_common_params(runtime_opts);
        params.n_outputs_max = 1;
        params.cb_eval = route_callback; params.cb_eval_user_data = &collector;
        loaded.context.reset(llama_init_from_model(loaded.init->model(), common_context_params_to_llama(params)));
        if (!loaded.context) throw std::runtime_error("failed to create sequential REAM context");
    };
    expert_cache.add(layout.layers.front());
    create_context();
    auto templates = common_chat_templates_init(loaded.init->model(), "");
    const auto dataset = aikar_dataset_load(opts.dataset, loaded.init->model(), templates.get(), opts.dataset_threads);
    report["calibration_sha256"] = common_moe_prune_sha256_file(opts.dataset);
    report["tokenized_sha256"] = aikar_dataset_fingerprint(dataset);
    uint64_t baseline_tokens = 0;
    for (const auto & record : dataset.records) if (record.tokens.size() > 1) {
        baseline_tokens = std::min<uint64_t>(opts.n_batch, std::min<uint64_t>(record.tokens.size() - 1, opts.n_ctx - 1));
        break;
    }
    if (!baseline_tokens) throw std::runtime_error("REAM calibration has no usable tokens");
    collector.routing_layer = layout.layers.front();
    if (!opts.ream_full_forward) loaded.context->set_ream_calibration_layer(layout.layers.front());
    evaluate(loaded.context.get(), dataset, collector, runtime_opts, "REAM shared REAP first batch", {}, true, baseline_tokens);
    const auto baseline_stats = collector.stats.at(layout.layers.front());
    report["calibration_forward"] = opts.ream_full_forward ? "full reference, no intermediate PPL" : "prefix through target layer, no intermediate PPL";
    std::map<int32_t, aikar_ream_groups> all_groups;
    json layer_reports = json::array();
    const int32_t threads = opts.n_threads > 0 ? opts.n_threads : 4;
    for (int32_t layer : layout.layers) {
        const auto started = std::chrono::steady_clock::now();
        std::cerr << "REAM: calibrating layer " << layer << " with merged preceding layers\n";
        for (int32_t previous : layout.layers) if (previous <= layer) expert_cache.add(previous);
        create_context();
        const std::string input_path = work + "/inputs.f32", logit_path = work + "/router.f32";
        std::ofstream input_out(input_path, std::ios::binary), logit_out(logit_path, std::ios::binary);
        if (!input_out || !logit_out) throw std::runtime_error("cannot create REAM calibration files");
        collector = route_collector();
        collector.n_expert = layout.experts; collector.collect_output_norm = true; collector.ream_layer = layer;
        collector.ream_inputs = &input_out; collector.ream_logits = &logit_out;
        if (!opts.ream_full_forward) loaded.context->set_ream_calibration_layer(layer);
        const auto evaluation = evaluate(loaded.context.get(), dataset, collector, runtime_opts, "REAM layer " + std::to_string(layer), {}, true, 0, [&](uint64_t processed) {
            if (layer != layout.layers.front() || processed != baseline_tokens) return;
            const auto & stats = collector.stats.at(layer);
            double maximum_relative_error = 0;
            for (int32_t expert = 0; expert < layout.experts; ++expert) {
                const auto & baseline = baseline_stats.at(expert);
                const auto & captured = stats.at(expert);
                if (baseline.reap_selection_count != captured.reap_selection_count || baseline.reap_count != captured.reap_count) {
                    throw std::runtime_error("REAM capture differs from shared REAP selection counts at expert " + std::to_string(expert));
                }
                const double relative_error = std::abs(baseline.reap_sum - captured.reap_sum) / std::max(1.0, std::abs(baseline.reap_sum));
                maximum_relative_error = std::max(maximum_relative_error, relative_error);
                if (!std::isfinite(relative_error) || relative_error > 1e-5) {
                    throw std::runtime_error("REAM capture differs from shared REAP saliency at expert " + std::to_string(expert));
                }
            }
            report["shared_reap_saliency_check"] = {{"layer", layer}, {"checked_tokens", baseline_tokens}, {"selection_counts_equal", true}, {"maximum_relative_sum_error", maximum_relative_error}, {"relative_tolerance", 1e-5}};
            std::cerr << "REAM: shared REAP saliency check passed, maximum relative sum error=" << maximum_relative_error << '\n';
        });
        input_out.close(); logit_out.close();
        if (!input_out || !logit_out || collector.ream_input_tokens == 0 || collector.ream_input_tokens != collector.ream_logit_tokens) throw std::runtime_error("incomplete REAM input/router captures");
        const uint64_t tokens = collector.ream_input_tokens;
        const double calibration_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto stats = collector.stats.at(layer);
        std::vector<double> saliency(layout.experts);
        for (int32_t e = 0; e < layout.experts; ++e) {
            if (stats[e].reap_count != stats[e].reap_selection_count) throw std::runtime_error("incomplete shared REAP saliency");
            saliency[e] = stats[e].reap_score();
        }
        const auto merge_saliency = aikar_ream_saliency_weights(saliency);
        const size_t sample_count = std::min<uint64_t>(tokens, opts.ream_samples);
        std::vector<uint64_t> sample(sample_count);
        std::iota(sample.begin(), sample.end(), 0);
        std::mt19937_64 random((uint32_t) opts.seed);
        for (uint64_t i = sample_count; i < tokens; ++i) {
            const uint64_t j = std::uniform_int_distribution<uint64_t>(0, i)(random);
            if (j < sample.size()) sample[j] = i;
        }
        std::sort(sample.begin(), sample.end());
        const auto offload_started = std::chrono::steady_clock::now();
        expert_cache.offload();
        const double offload_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - offload_started).count();
        const auto features_started = std::chrono::steady_clock::now();
        const std::string probability_path = work + "/probabilities.f64";
        {
            std::ifstream logits(logit_path, std::ios::binary);
            std::ofstream probabilities(probability_path, std::ios::binary);
            const uint64_t probability_chunk = 8192;
            for (uint64_t t = 0; t < tokens; t += probability_chunk) {
                const size_t count = std::min<uint64_t>(tokens - t, probability_chunk);
                auto rows = ream_read_floats(logits, count * layout.experts);
                std::vector<double> values(rows.size());
                for (size_t j = 0; j < count; ++j) {
                    auto * row = rows.data() + j * layout.experts;
                    const float maximum = *std::max_element(row, row + layout.experts);
                    double denominator = 0;
                    for (int32_t e = 0; e < layout.experts; ++e) denominator += std::exp((double) row[e] - maximum);
                    for (int32_t e = 0; e < layout.experts; ++e) values[e * count + j] = std::exp((double) row[e] - maximum) / denominator;
                }
                for (int32_t e = 0; e < layout.experts; ++e) {
                    probabilities.seekp((e * tokens + t) * sizeof(double));
                    probabilities.write((const char *) (values.data() + e * count), count * sizeof(double));
                }
            }
            probabilities.close();
            if (!probabilities) throw std::runtime_error("REAM router probability disk write failed");
        }
        const uint64_t activation_disk = (uint64_t) layout.experts * layout.hidden * sample_count * sizeof(float);
        if (std::filesystem::space(activation_work).available < activation_disk + 16 * 1024 * 1024) throw std::runtime_error("REAM sampled activations exceed free temporary disk space: " + std::to_string(activation_disk));
        struct input_cache_guard {
            ggml_context * ctx = nullptr;
            ggml_backend_buffer_t buffer = nullptr;
            ~input_cache_guard() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
        };
        uint64_t input_cache_bytes = 0;
        std::vector<float> means((size_t) layout.experts * layout.embedding, 0);
        {
        input_cache_guard input_cache;
        ggml_tensor * cached_input = nullptr;
        const auto feature_type = opts.ream_feature_precision == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
        const uint64_t padded_tokens = ((tokens + opts.ream_chunk - 1) / opts.ream_chunk) * opts.ream_chunk;
        if (compute.backend && opts.ream_input_cache_mib > 0) {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(compute.backend), &free_bytes, &total_bytes);
            const uint64_t headroom = layout.workspace_bytes + 512 * 1024 * 1024;
            const uint64_t allowed = std::min<uint64_t>((uint64_t) opts.ream_input_cache_mib * 1024 * 1024, free_bytes > headroom ? free_bytes - headroom : 0);
            const uint64_t bytes_per_chunk = (uint64_t) opts.ream_chunk * layout.embedding * ggml_type_size(feature_type);
            const uint64_t cache_tokens = std::min<uint64_t>(padded_tokens, (allowed / bytes_per_chunk) * opts.ream_chunk);
            if (cache_tokens) {
                input_cache.ctx = ggml_init({64 * 1024, nullptr, true});
                if (!input_cache.ctx) throw std::runtime_error("cannot allocate feature input cache metadata");
                cached_input = ggml_new_tensor_2d(input_cache.ctx, feature_type, layout.embedding, cache_tokens);
                input_cache.buffer = ggml_backend_alloc_ctx_tensors(input_cache.ctx, compute.backend);
                if (!input_cache.buffer) throw std::runtime_error("cannot allocate feature input cache");
                std::ifstream file(input_path, std::ios::binary);
                std::vector<float> values((size_t) opts.ream_chunk * layout.embedding, 0);
                std::vector<ggml_fp16_t> half(feature_type == GGML_TYPE_F16 ? values.size() : 0);
                for (uint64_t t = 0; t < std::min<uint64_t>(tokens, cache_tokens); t += opts.ream_chunk) {
                    const size_t count = std::min<uint64_t>(tokens - t, opts.ream_chunk) * layout.embedding;
                    file.read((char *) values.data(), count * sizeof(float));
                    if ((size_t) file.gcount() != count * sizeof(float)) throw std::runtime_error("short input cache read");
                    std::fill(values.begin() + count, values.end(), 0);
                    for (float x : values) if (!std::isfinite(x) || (feature_type == GGML_TYPE_F16 && std::abs(x) > 65504)) throw std::runtime_error("input cache has invalid values; use f32 features for F16 overflow");
                    if (feature_type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
                    ggml_backend_tensor_set(cached_input, feature_type == GGML_TYPE_F16 ? (const void *) half.data() : values.data(), t * cached_input->nb[1], values.size() * ggml_type_size(feature_type));
                }
                input_cache_bytes = ggml_backend_buffer_get_size(input_cache.buffer);
            }
        }
        std::cerr << "REAM: layer " << layer << " feature_input_cache_bytes=" << input_cache_bytes << '\n';
        for (int32_t e = 0; e < layout.experts; ++e) {
            const auto expert = aikar_ream_read_expert(opts.model, layer, e);
            const bool summarize = compute.backend && !opts.ream_full_expert_forward;
            aikar_ream_forward_runner runner(expert, std::min<uint64_t>(tokens, opts.ream_chunk), threads, !opts.ream_full_expert_forward, opts.ream_feature_precision == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32, summarize);
            std::vector<float> sampled((size_t) layout.hidden * sample_count);
            const int32_t mean_dimension = opts.ream_full_expert_forward ? layout.embedding : layout.hidden;
            std::vector<double> sum(mean_dimension, 0.0);
            std::ifstream inputs(input_path, std::ios::binary), logits(probability_path, std::ios::binary);
            logits.seekg(e * tokens * sizeof(double));
            size_t next_sample = 0;
            for (uint64_t t = 0; t < tokens; t += opts.ream_chunk) {
                const size_t count = std::min<uint64_t>(tokens - t, opts.ream_chunk);
                std::vector<double> gate(count);
                logits.read((char *) gate.data(), gate.size() * sizeof(double));
                if ((size_t) logits.gcount() != gate.size() * sizeof(double)) throw std::runtime_error("short REAM probability file read");
                std::vector<float> hidden, out;
                if (summarize) {
                    const size_t sample_begin = next_sample;
                    std::vector<int32_t> ids;
                    while (next_sample < sample_count && sample[next_sample] < t + count) ids.push_back(sample[next_sample++] - t);
                    std::vector<float> probabilities(count);
                    for (size_t j = 0; j < count; ++j) probabilities[j] = gate[j];
                    const bool cached = cached_input && t + std::min<uint64_t>(tokens, opts.ream_chunk) <= (uint64_t) cached_input->ne[1];
                    if (!cached) inputs.seekg(t * layout.embedding * sizeof(float));
                    runner.run_summary(cached ? cached_input : nullptr, t, cached ? std::vector<float>() : ream_read_floats(inputs, count * layout.embedding), probabilities, ids, hidden, out);
                    for (int32_t h = 0; h < layout.hidden; ++h) sum[h] += out[h];
                    for (int32_t h = 0; h < layout.hidden; ++h) for (size_t j = 0; j < ids.size(); ++j) sampled[h * sample_count + sample_begin + j] = hidden[j * layout.hidden + h];
                    continue;
                }
                if (cached_input && t + std::min<uint64_t>(tokens, opts.ream_chunk) <= (uint64_t) cached_input->ne[1]) runner.run_device(cached_input, t, count, hidden, out);
                else {
                    inputs.seekg(t * layout.embedding * sizeof(float));
                    runner.run(ream_read_floats(inputs, count * layout.embedding), hidden, out);
                }
                for (size_t j = 0; j < count; ++j) {
                    const double probability = gate[j];
                    const auto * values = opts.ream_full_expert_forward ? out.data() : hidden.data();
                    for (int32_t d = 0; d < mean_dimension; ++d) sum[d] += probability * values[j * mean_dimension + d];
                    if (next_sample < sample_count && sample[next_sample] == t + j) {
                        for (int32_t h = 0; h < layout.hidden; ++h) sampled[h * sample_count + next_sample] = hidden[j * layout.hidden + h];
                        ++next_sample;
                    }
                }
            }
            for (auto & value : sum) value /= tokens;
            const auto projected = opts.ream_full_expert_forward ? std::vector<float>(sum.begin(), sum.end()) : aikar_ream_project_mean(expert, sum);
            std::copy(projected.begin(), projected.end(), means.begin() + e * layout.embedding);
            std::ofstream act(activation_work + "/expert-" + std::to_string(e) + ".f32", std::ios::binary);
            act.write((const char *) sampled.data(), sampled.size() * sizeof(float)); act.close();
            if (!act || next_sample != sample_count) throw std::runtime_error("incomplete sampled hidden activations");
            if (e % 8 == 0) std::cerr << "REAM: layer " << layer << " evaluated all-token expert " << e << '/' << layout.experts << ", feature_elapsed_seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now() - features_started).count() << '\n';
        }
        }
        const double features_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - features_started).count();
        const auto grouping_started = std::chrono::steady_clock::now();
        const auto distance = ream_group_distances(means, logit_path, tokens, layout.experts, layout.embedding);
        auto groups = aikar_ream_pseudo_group(merge_saliency, distance, opts.target_experts, opts.ream_group_size);
        const double grouping_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - grouping_started).count();
        const auto merging_started = std::chrono::steady_clock::now();
        double distance_seconds = 0, hungarian_seconds = 0;
        const bool use_hidden = opts.ream_merging.find("logits") != std::string::npos;
        const bool use_weights = opts.ream_merging.find("weights") != std::string::npos;
        for (size_t g = 0; g < groups.members.size(); ++g) {
            const auto & members = groups.members[g];
            if (members.size() == 1) continue;
            const int32_t center = members[0];
            const auto centroid = aikar_ream_read_expert(opts.model, layer, center);
            auto merged = centroid;
            std::fill(merged.gate.begin(), merged.gate.end(), 0);
            std::fill(merged.up.begin(), merged.up.end(), 0);
            std::fill(merged.down.begin(), merged.down.end(), 0);
            auto center_hidden = use_hidden ? ream_load_hidden(activation_work + "/expert-" + std::to_string(center) + ".f32", layout.hidden * sample_count) : std::vector<float>();
            if (use_hidden && sample_count > 1) center_hidden = aikar_ream_normalize_features(center_hidden, layout.hidden, sample_count);
            const auto center_weights = use_weights ? aikar_ream_normalize_features(aikar_ream_weight_features(centroid), layout.hidden, layout.embedding * 3) : std::vector<float>();
            double maximum_saliency = 0;
            for (int32_t id : members) maximum_saliency = std::max(maximum_saliency, merge_saliency[id]);
            double total = 0;
            for (int32_t id : members) total += merge_saliency[id] / maximum_saliency;
            aikar_ream_accumulate(merged, centroid, (merge_saliency[center] / maximum_saliency) / total);
            for (size_t j = 1; j < members.size(); ++j) {
                const int32_t id = members[j];
                const auto expert = aikar_ream_read_expert(opts.model, layer, id);
                const auto distance_started = std::chrono::steady_clock::now();
                std::vector<float> cost((size_t) layout.hidden * layout.hidden, 0);
                if (use_hidden) {
                    auto other_hidden = ream_load_hidden(activation_work + "/expert-" + std::to_string(id) + ".f32", layout.hidden * sample_count);
                    if (sample_count > 1) other_hidden = aikar_ream_normalize_features(other_hidden, layout.hidden, sample_count);
                    const auto dist = aikar_ream_distances(center_hidden, other_hidden, layout.hidden, sample_count, threads, false);
                    for (size_t i = 0; i < cost.size(); ++i) cost[i] += dist[i];
                }
                if (use_weights) {
                    const auto other_weights = aikar_ream_normalize_features(aikar_ream_weight_features(expert), layout.hidden, layout.embedding * 3);
                    const auto dist = aikar_ream_distances(center_weights, other_weights, layout.hidden, layout.embedding * 3, threads, false);
                    for (size_t i = 0; i < cost.size(); ++i) cost[i] += dist[i];
                }
                distance_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - distance_started).count();
                const auto hungarian_started = std::chrono::steady_clock::now();
                const auto perm = aikar_ream_hungarian(cost, layout.hidden);
                hungarian_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - hungarian_started).count();
                aikar_ream_accumulate(merged, aikar_ream_permute(expert, perm), (merge_saliency[id] / maximum_saliency) / total);
                std::cerr << "REAM: layer " << layer << " aligned expert " << id << " -> " << center << '\n';
            }
            ream_set_expert(loaded.init->model(), layer, center, merged);
        }
        const double merging_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - merging_started).count();
        const json timings = {{"calibration", calibration_seconds}, {"calibration_cache_offload", offload_seconds}, {"expert_features", features_seconds},
            {"grouping", grouping_seconds}, {"merging", merging_seconds}, {"distance", distance_seconds}, {"hungarian", hungarian_seconds}};
        std::cerr << "REAM: layer " << layer << " timing_seconds=" << timings.dump() << '\n';
        all_groups[layer] = groups;
        loaded.context.reset();
        auto & mask = loaded.init->model()->hparams.moe_disabled_experts[layer];
        mask.reset();
        for (int32_t e = 0; e < layout.experts; ++e) if (std::find(groups.centers.begin(), groups.centers.end(), e) == groups.centers.end()) mask.set(e);
        loaded.init->model()->hparams.moe_prune_active = true;
        layer_reports.push_back({{"layer", layer}, {"calibration", result_json(evaluation, opts.mask)}, {"saliency", saliency},
            {"merge_saliency", merge_saliency}, {"captured_tokens", tokens}, {"samples", sample_count}, {"feature_input_cache_bytes", input_cache_bytes},
            {"timing_seconds", timings},
            {"elapsed_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()}});
        for (int32_t e = 0; e < layout.experts; ++e) std::filesystem::remove(activation_work + "/expert-" + std::to_string(e) + ".f32");
        std::filesystem::remove(input_path); std::filesystem::remove(logit_path);
        std::filesystem::remove(probability_path);
    }
    loaded.context.reset();
    report["layer_measurements"] = layer_reports;
    report["calibration_expert_cache_bytes"] = expert_cache.peak;
    report["calibration_expert_cache_uploaded_bytes"] = expert_cache.uploaded;
    report["calibration_expert_cache_offloaded_bytes"] = expert_cache.offloaded;
    report["calibration_expert_cache_layers"] = expert_cache.peak_layers;
    aikar_ream_export(opts.model, opts.output, all_groups, report.dump(), [&](const std::string & name, int32_t expert, size_t slice) {
        auto * tensor = loaded.init->model()->get_tensor(name.c_str());
        if (!tensor) throw std::runtime_error("missing compacted REAM tensor");
        std::vector<uint8_t> bytes(slice);
        ggml_backend_tensor_get(tensor, bytes.data(), expert * slice, slice);
        return bytes;
    });
    std::cout << "REAM GGUF saved: " << opts.output << "\nRun aikar-prune verify in a new process to validate inference.\n";
}

void run_verify(const options & opts) {
    std::set<std::filesystem::path> destinations;
    for (const auto & path : {opts.output, opts.output.empty() ? std::string() : opts.output + ".tmp", opts.save_logits}) {
        if (path.empty()) continue;
        if (!destinations.insert(std::filesystem::weakly_canonical(path)).second) throw std::runtime_error("verification metrics, staging and baseline logits must use distinct paths");
        if (std::filesystem::exists(std::filesystem::symlink_status(path))) throw std::runtime_error("verification output or staging path already exists: " + path);
    }
    route_collector collector;
    loaded_model loaded = load_model(opts, nullptr, nullptr);
    const auto * vocab = llama_model_get_vocab(loaded.init->model());
    auto tokens = common_tokenize(vocab, "Hello", true, true);
    if (tokens.empty() || llama_decode(loaded.context.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0) throw std::runtime_error("reloaded model prefill failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 8; ++i) {
        const float * logits = llama_get_logits_ith(loaded.context.get(), -1);
        const int32_t nv = llama_vocab_n_tokens(vocab);
        for (int32_t v = 0; v < nv; ++v) if (!std::isfinite(logits[v])) throw std::runtime_error("reloaded model has non-finite logits");
        llama_token next = std::max_element(logits, logits + nv) - logits;
        std::cout << common_token_to_piece(vocab, next);
        if (llama_decode(loaded.context.get(), llama_batch_get_one(&next, 1)) != 0) throw std::runtime_error("reloaded model generation failed");
    }
    json metrics = {{"finite_logits", true}, {"generated_tokens", 8},
        {"generation_tokens_per_second", 8.0 / std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()},
        {"model_bytes", std::filesystem::file_size(opts.model)}};
    if (!opts.dataset.empty()) {
        auto templates = common_chat_templates_init(loaded.init->model(), "");
        const auto data = aikar_dataset_load(opts.dataset, loaded.init->model(), templates.get(), opts.dataset_threads);
        const json header = {{"format", "aikar-verification-logits-v1"}, {"vocab", llama_vocab_n_tokens(vocab)},
            {"tokenized_sha256", aikar_dataset_fingerprint(data)}, {"context", opts.n_ctx}};
        std::ofstream saved;
        std::ifstream reference;
        if (!opts.save_logits.empty()) {
            if (std::filesystem::exists(opts.save_logits)) throw std::runtime_error("baseline logit file already exists");
            saved.open(opts.save_logits, std::ios::binary);
            saved << header.dump() << '\n';
            if (!saved) throw std::runtime_error("cannot create baseline logits");
        }
        if (!opts.reference_logits.empty()) {
            reference.open(opts.reference_logits, std::ios::binary);
            std::string line;
            if (!std::getline(reference, line) || json::parse(line) != header) throw std::runtime_error("reference logits have incompatible vocabulary, context or calibration tokens");
        }
        double kld_sum = 0.0;
        uint64_t rows = 0;
        const auto evaluation = evaluate(loaded.context.get(), data, collector, opts, "reload verification", [&](const float * logits, int32_t nv) {
            for (int32_t i = 0; i < nv; ++i) if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite calibration logits");
            if (saved.is_open()) {
                saved.write((const char *) logits, nv * sizeof(float));
                if (!saved) throw std::runtime_error("cannot write baseline logits");
            }
            if (reference.is_open()) {
                const auto original = ream_read_floats(reference, nv);
                for (float value : original) if (!std::isfinite(value)) throw std::runtime_error("non-finite reference logits");
                const double pm = *std::max_element(original.begin(), original.end());
                const double qm = *std::max_element(logits, logits + nv);
                double ps = 0.0, qs = 0.0;
                for (int32_t i = 0; i < nv; ++i) { ps += std::exp(original[i] - pm); qs += std::exp(logits[i] - qm); }
                const double pl = pm + std::log(ps), ql = qm + std::log(qs);
                double kld = 0.0;
                for (int32_t i = 0; i < nv; ++i) kld += std::exp(original[i] - pl) * (original[i] - pl - logits[i] + ql);
                if (!std::isfinite(kld)) throw std::runtime_error("non-finite logit KLD");
                kld_sum += std::max(0.0, kld);
            }
            ++rows;
        });
        if (reference.is_open() && reference.peek() != std::char_traits<char>::eof()) throw std::runtime_error("unused reference logit rows");
        if (saved.is_open()) { saved.close(); if (!saved) throw std::runtime_error("cannot flush baseline logits"); }
        metrics["evaluation"] = result_json(evaluation, opts.mask);
        metrics["logit_rows"] = rows;
        if (!opts.reference_logits.empty()) metrics["logit_kld_original_to_model"] = rows ? json(kld_sum / rows) : json(nullptr);
    }
    std::cout << '\n' << metrics.dump(2) << '\n';
    if (!opts.output.empty()) write_json_atomic(opts.output, metrics, "verification metrics");
}

void run_hard(const options & opts) {
    const std::string staging_output = opts.output + ".validation.tmp";
    for (const std::string & target : { opts.output, opts.output + ".report.json", staging_output,
            staging_output + ".tmp", staging_output + ".report.json", staging_output + ".report.json.tmp",
            staging_output + ".previous-model", staging_output + ".previous-report",
            staging_output + ".tmp.report.json", staging_output + ".tmp.report.json.tmp",
            staging_output + ".tmp.previous-model", staging_output + ".tmp.previous-report" }) {
        if (std::filesystem::exists(target) && std::filesystem::equivalent(opts.model, target)) {
            throw std::runtime_error("hard pruning never replaces the source GGUF, including path aliases");
        }
    }
    const common_moe_prune_model_info model = common_moe_prune_inspect_model(opts.model);
    const common_moe_prune_profile profile = common_moe_prune_profile_load(opts.profile);
    common_moe_prune_profile_validate(profile, model);
    struct staging_guard {
        std::string model;
        bool committed = false;
        ~staging_guard() {
            if (!committed) {
                std::remove(model.c_str());
                std::remove((model + ".report.json").c_str());
            }
        }
    } guard { staging_output };
    const aikar_hard_prune_report report = aikar_hard_prune_gemma4_q4_0(opts.model, profile, model, staging_output);

    options validation_opts = opts;
    validation_opts.model = staging_output;
    const common_moe_prune_model_info pruned_info = common_moe_prune_inspect_model(staging_output);
    if (pruned_info.expert_count != model.expert_count - (int32_t) profile.layers.begin()->second.disabled_experts.size()) {
        throw std::runtime_error("hard-pruned model metadata validation failed");
    }
    std::optional<evaluation_result> hard_evaluation;
    {
        route_collector collector;
        collector.n_expert = pruned_info.expert_count;
        loaded_model validation = load_model(validation_opts, &collector, nullptr);
        const llama_vocab * vocab = llama_model_get_vocab(validation.init->model());
        std::vector<llama_token> tokens = common_tokenize(vocab, "Hello", true, true);
        if (tokens.empty() || llama_decode(validation.context.get(), llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
            throw std::runtime_error("hard-pruned model inference smoke test failed");
        }
        if (!opts.dataset.empty()) {
            common_chat_templates_ptr templates = common_chat_templates_init(validation.init->model(), "");
            const aikar_dataset dataset = aikar_dataset_load(opts.dataset, validation.init->model(), templates.get(), opts.dataset_threads);
            hard_evaluation = evaluate(validation.context.get(), dataset, collector, opts, "hard validation");
        }
    }
    if (hard_evaluation) {
        route_collector soft_collector;
        soft_collector.n_expert = model.expert_count;
        loaded_model soft_model = load_model(opts, &soft_collector, &profile);
        common_chat_templates_ptr templates = common_chat_templates_init(soft_model.init->model(), "");
        const aikar_dataset dataset = aikar_dataset_load(opts.dataset, soft_model.init->model(), templates.get(), opts.dataset_threads);
        const evaluation_result soft_evaluation = evaluate(soft_model.context.get(), dataset, soft_collector, opts, "soft validation");
        if (soft_evaluation.evaluated[(size_t) opts.mask] != hard_evaluation->evaluated[(size_t) opts.mask]) {
            throw std::runtime_error("soft and hard evaluations used different token counts");
        }
        const double soft_ppl = soft_evaluation.ppl(opts.mask);
        const double hard_ppl = hard_evaluation->ppl(opts.mask);
        const double difference = hard_ppl - soft_ppl;
        const std::string report_path = staging_output + ".report.json";
        std::ifstream report_in(report_path);
        json report_json_value;
        report_in >> report_json_value;
        report_json_value["validation"] = {
            { "dataset_hash", common_moe_prune_sha256_file(opts.dataset) },
            { "ppl_mask", aikar_ppl_mask_name(opts.mask) },
            { "soft_perplexity", soft_ppl },
            { "hard_perplexity", hard_ppl },
            { "absolute_difference", difference },
            { "evaluated_token_count", hard_evaluation->evaluated[(size_t) opts.mask] },
        };
        const std::string tmp_report = report_path + ".tmp";
        std::ofstream report_out(tmp_report, std::ios::trunc);
        report_out << report_json_value.dump(2) << '\n';
        report_out.close();
        if (!report_out || std::rename(tmp_report.c_str(), report_path.c_str()) != 0) {
            std::remove(tmp_report.c_str());
            throw std::runtime_error("failed to update hard-pruning validation report");
        }
        std::cout << "soft perplexity: " << soft_ppl << "\nhard perplexity: " << hard_ppl << "\nabsolute difference: " << difference << '\n';
    }
    {
        std::ifstream in(staging_output + ".report.json");
        json provenance;
        in >> provenance;
        provenance["output_path"] = opts.output;
        provenance["output_sha256"] = pruned_info.model_hash;
        provenance["profile_path"] = opts.profile;
        provenance["profile_sha256"] = common_moe_prune_sha256_file(opts.profile);
        write_json_atomic(staging_output + ".report.json", provenance, "hard-pruning provenance report");
    }
    aikar_hard_prune_publish(staging_output, opts.output);
    guard.committed = true;
    std::cout << "hard-pruned model validated\nsource bytes: " << report.source_bytes << "\noutput bytes: " << report.output_bytes
              << "\nexpert bytes removed: " << report.expert_bytes_removed << "\nreport: " << opts.output << ".report.json\n";
}

}

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        common_init();
        const bool needs_backend = opts.command == "analyze" || opts.command == "verify" || (opts.command == "hard" && !opts.dry_run);
        if (needs_backend) {
            // Profiling splits retain separate CUDA graph activation caches.
            common_set_env("GGML_CUDA_DISABLE_GRAPHS", "1");
            std::cerr << "aikar-prune: CUDA graphs disabled for memory-bounded calibration\n";
            llama_backend_init();
            llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        }
        if (opts.command == "inspect") run_inspect(opts);
        else if (opts.command == "analyze") run_analyze(opts);
        else if (opts.command == "profiles") run_profiles(opts);
        else if (opts.command == "verify") run_verify(opts);
        else if (opts.method == "ream") run_ream(opts);
        else run_hard(opts);
        if (needs_backend) llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        usage();
        return 1;
    }
}
