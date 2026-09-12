#include "direction.h"
#include "evaluator.h"
#include "surgery.h"

#include "common.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "log.h"
#include "moe-prune.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

struct options {
    std::string command;
    std::string model;
    std::string dataset;
    std::string direction_dataset;
    std::string train_dataset;
    std::string eval_dataset;
    std::string refusal_patterns = "tools/lumen-heretic/refusal-patterns.txt";
    std::string analysis;
    std::string output;
    std::string device;
    std::string layers;
    std::string direction_layer_mode = "all";
    std::string lr_schedule = "cosine";
    std::string selection = "eval";
    std::string quant = "q4_0";
    int32_t batch_size = 512;
    int32_t ctx = 512;
    int32_t gpu_layers = -1;
    int32_t steps = 20;
    int32_t residual_token = -1;
    int32_t direction_topk = 12;
    int32_t minimum_target = 32;
    int32_t minimum_control = 32;
    int32_t kl_tokens = 8;
    int32_t behavior_tokens = 32;
    int32_t patience = 8;
    int32_t eval_every = 1;
    int32_t seed = 42;
    float max_alpha = 1.5f;
    float max_alpha_attn = -1.0f;
    float max_alpha_mlp = -1.0f;
    float alpha_warm_start = 0.05f;
    float lr = 0.05f;
    float lambda_behavior = 1.0f;
    float lambda_kl = 1.0f;
    float lambda_quant = 0.0f;
    float lambda_sparse = 0.0f;
    float spsa_delta = 0.05f;
    float max_kl = 0.10f;
    float eval_split = 0.0f;
    bool train_alpha = false;
    bool attn = true;
    bool mlp = true;
    bool experts = true;
    bool cpu_offload_analysis = false;
    bool save_analysis = false;
    bool orthogonalize_control = false;
    bool resume = false;
    bool apply_chat_template = false;
};

struct candidate_record {
    int32_t step = 0;
    std::string branch;
    heretic_metrics train;
    heretic_metrics eval;
    std::vector<float> alpha_attn;
    std::vector<float> alpha_mlp;
    bool accepted = false;
};

void save_json(const std::string & path, const json & value);

double allocated_gpu_mib(const llama_context * context) {
    size_t bytes = 0;
    for (const auto & item : llama_get_memory_breakdown(context)) {
        ggml_backend_dev_t device = ggml_backend_buft_get_device(item.first);
        if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) bytes += item.second.total();
    }
    return (double) bytes / (1024.0 * 1024.0);
}

void usage(const char * argv0) {
    printf("usage:\n");
    printf("  %s analyze --model MODEL --dataset DATASET --output DIR [options]\n", argv0);
    printf("  %s evaluate --model MODEL --dataset DATASET --output REPORT.json [options]\n", argv0);
    printf("  %s optimize --model MODEL --dataset DATASET --output MODEL.gguf --train-alpha [options]\n\n", argv0);
    printf("options:\n");
    printf("  --direction-dataset FILE --train-dataset FILE --eval-dataset FILE --eval-split F\n");
    printf("  --analysis FILE --refusal-patterns FILE --apply-chat-template --raw-prompts\n");
    printf("  --device NAME --gpu-layers N --batch-size N --ctx N --kl-tokens N --behavior-tokens N\n");
    printf("  --direction-layer-mode all|topk --direction-topk N --layers LIST\n");
    printf("  --max-alpha F --max-alpha-attn F --max-alpha-mlp F --alpha-warm-start F\n");
    printf("  --lr F --lr-schedule constant|cosine --steps N --patience N --eval-every N\n");
    printf("  --lambda-behavior F --lambda-kl F --lambda-sparse F --max-kl F --selection train|eval\n");
    printf("  --opt-attn --opt-mlp --opt-both --experts --no-experts --resume --seed N\n");
}

std::string take_value(int & i, int argc, char ** argv) {
    if (++i >= argc) throw std::runtime_error(std::string(argv[i - 1]) + " requires a value");
    return argv[i];
}

options parse_options(int argc, char ** argv) {
    if (argc < 2) throw std::runtime_error("missing command");
    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        usage(argv[0]);
        std::exit(0);
    }
    options result;
    result.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" || arg == "-m") result.model = take_value(i, argc, argv);
        else if (arg == "--dataset") result.dataset = take_value(i, argc, argv);
        else if (arg == "--direction-dataset") result.direction_dataset = take_value(i, argc, argv);
        else if (arg == "--train-dataset") result.train_dataset = take_value(i, argc, argv);
        else if (arg == "--eval-dataset") result.eval_dataset = take_value(i, argc, argv);
        else if (arg == "--refusal-patterns") result.refusal_patterns = take_value(i, argc, argv);
        else if (arg == "--analysis") result.analysis = take_value(i, argc, argv);
        else if (arg == "--output" || arg == "-o") result.output = take_value(i, argc, argv);
        else if (arg == "--device") result.device = take_value(i, argc, argv);
        else if (arg == "--layers") result.layers = take_value(i, argc, argv);
        else if (arg == "--direction-layer-mode") result.direction_layer_mode = take_value(i, argc, argv);
        else if (arg == "--lr-schedule") result.lr_schedule = take_value(i, argc, argv);
        else if (arg == "--selection") result.selection = take_value(i, argc, argv);
        else if (arg == "--quant") result.quant = take_value(i, argc, argv);
        else if (arg == "--batch-size") result.batch_size = std::stoi(take_value(i, argc, argv));
        else if (arg == "--ctx") result.ctx = std::stoi(take_value(i, argc, argv));
        else if (arg == "--gpu-layers") result.gpu_layers = std::stoi(take_value(i, argc, argv));
        else if (arg == "--steps") result.steps = std::stoi(take_value(i, argc, argv));
        else if (arg == "--residual-token") result.residual_token = std::stoi(take_value(i, argc, argv));
        else if (arg == "--direction-topk") result.direction_topk = std::stoi(take_value(i, argc, argv));
        else if (arg == "--minimum-target") result.minimum_target = std::stoi(take_value(i, argc, argv));
        else if (arg == "--minimum-control") result.minimum_control = std::stoi(take_value(i, argc, argv));
        else if (arg == "--kl-tokens") result.kl_tokens = std::stoi(take_value(i, argc, argv));
        else if (arg == "--behavior-tokens") result.behavior_tokens = std::stoi(take_value(i, argc, argv));
        else if (arg == "--patience") result.patience = std::stoi(take_value(i, argc, argv));
        else if (arg == "--eval-every") result.eval_every = std::stoi(take_value(i, argc, argv));
        else if (arg == "--seed") result.seed = std::stoi(take_value(i, argc, argv));
        else if (arg == "--max-alpha") result.max_alpha = std::stof(take_value(i, argc, argv));
        else if (arg == "--max-alpha-attn") result.max_alpha_attn = std::stof(take_value(i, argc, argv));
        else if (arg == "--max-alpha-mlp") result.max_alpha_mlp = std::stof(take_value(i, argc, argv));
        else if (arg == "--alpha-warm-start") result.alpha_warm_start = std::stof(take_value(i, argc, argv));
        else if (arg == "--lr") result.lr = std::stof(take_value(i, argc, argv));
        else if (arg == "--lambda-behavior") result.lambda_behavior = std::stof(take_value(i, argc, argv));
        else if (arg == "--lambda-kl") result.lambda_kl = std::stof(take_value(i, argc, argv));
        else if (arg == "--lambda-quant") result.lambda_quant = std::stof(take_value(i, argc, argv));
        else if (arg == "--lambda-sparse") result.lambda_sparse = std::stof(take_value(i, argc, argv));
        else if (arg == "--max-kl") result.max_kl = std::stof(take_value(i, argc, argv));
        else if (arg == "--eval-split") result.eval_split = std::stof(take_value(i, argc, argv));
        else if (arg == "--spsa-delta") result.spsa_delta = std::stof(take_value(i, argc, argv));
        else if (arg == "--train-alpha") result.train_alpha = true;
        else if (arg == "--attn") result.attn = true;
        else if (arg == "--no-attn") result.attn = false;
        else if (arg == "--mlp") result.mlp = true;
        else if (arg == "--no-mlp") result.mlp = false;
        else if (arg == "--experts") result.experts = true;
        else if (arg == "--no-experts") result.experts = false;
        else if (arg == "--opt-attn") { result.attn = true; result.mlp = false; result.experts = false; }
        else if (arg == "--opt-mlp") { result.attn = false; result.mlp = true; result.experts = true; }
        else if (arg == "--opt-both") { result.attn = true; result.mlp = true; result.experts = true; }
        else if (arg == "--cpu-offload-analysis") result.cpu_offload_analysis = true;
        else if (arg == "--save-analysis") result.save_analysis = true;
        else if (arg == "--orthogonalize-control") result.orthogonalize_control = true;
        else if (arg == "--resume") result.resume = true;
        else if (arg == "--apply-chat-template") result.apply_chat_template = true;
        else if (arg == "--raw-prompts") result.apply_chat_template = false;
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else throw std::runtime_error("unknown option: " + arg);
    }
    if (result.command != "analyze" && result.command != "evaluate" && result.command != "optimize") throw std::runtime_error("command must be 'analyze', 'evaluate', or 'optimize'");
    if (result.model.empty() || result.output.empty()) throw std::runtime_error("--model and --output are required");
    if (result.dataset.empty() && result.direction_dataset.empty()) throw std::runtime_error("--dataset or --direction-dataset is required");
    if (result.command != "analyze" && result.dataset.empty() && result.train_dataset.empty()) throw std::runtime_error("evaluation and optimization require --dataset or --train-dataset");
    if (result.quant != "q4_0") throw std::runtime_error("the first prototype supports only --quant q4_0");
    if (result.max_alpha <= 0.0f || result.lr <= 0.0f || result.steps < 0 || result.ctx <= 0 || result.batch_size <= 0 || result.kl_tokens <= 0 || result.behavior_tokens <= 0 || result.spsa_delta <= 0.0f || result.max_kl <= 0.0f) throw std::runtime_error("invalid numeric option");
    if (result.max_alpha_attn < 0.0f) result.max_alpha_attn = result.max_alpha;
    if (result.max_alpha_mlp < 0.0f) result.max_alpha_mlp = result.max_alpha;
    if (result.alpha_warm_start < 0.0f || result.alpha_warm_start > std::min(result.max_alpha_attn, result.max_alpha_mlp)) throw std::runtime_error("--alpha-warm-start is outside alpha bounds");
    if (result.direction_layer_mode != "all" && result.direction_layer_mode != "topk") throw std::runtime_error("--direction-layer-mode must be all or topk");
    if (result.lr_schedule != "constant" && result.lr_schedule != "cosine") throw std::runtime_error("--lr-schedule must be constant or cosine");
    if (result.selection != "train" && result.selection != "eval") throw std::runtime_error("--selection must be train or eval");
    if (result.eval_split < 0.0f || result.eval_split >= 1.0f) throw std::runtime_error("--eval-split must be in [0, 1)");
    if (result.command == "optimize" && !result.train_alpha) throw std::runtime_error("optimize currently requires --train-alpha");
    if (result.lambda_quant != 0.0f) throw std::runtime_error("--lambda-quant requires the future FP-shadow recovery stage; use 0 for this prototype");
    return result;
}

std::string resolve_model(const std::string & path) {
    if (fs::is_regular_file(path)) return fs::canonical(path).string();
    if (!fs::is_directory(path)) throw std::runtime_error("model path does not exist: " + path);
    for (const char * preferred : { "Lumen-2-Flare-main.gguf", "Lumen-2-Pulsar-main.gguf" }) {
        const fs::path candidate = fs::path(path) / preferred;
        if (fs::exists(candidate)) return fs::canonical(candidate).string();
    }
    std::vector<fs::path> candidates;
    for (const fs::directory_entry & entry : fs::directory_iterator(path)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_regular_file() && entry.path().extension() == ".gguf" && name.find("adapter") == std::string::npos && name.find("mmproj") == std::string::npos && name.find("ckpt") == std::string::npos) candidates.push_back(entry.path());
    }
    if (candidates.size() != 1) throw std::runtime_error("model directory must contain one unambiguous full GGUF or a *-main.gguf symlink: " + path);
    return fs::canonical(candidates[0]).string();
}

llama_model * load_model(const options & opts, const std::string & path) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = opts.gpu_layers < 0 ? INT_MAX : opts.gpu_layers;
    static ggml_backend_dev_t devices[2] = { nullptr, nullptr };
    if (!opts.device.empty()) {
        devices[0] = ggml_backend_dev_by_name(opts.device.c_str());
        if (!devices[0]) throw std::runtime_error("unknown device: " + opts.device);
        params.devices = devices;
    }
    llama_model * model = llama_model_load_from_file(path.c_str(), params);
    if (!model) throw std::runtime_error("failed to load model");
    if (model->arch != LLM_ARCH_GEMMA4 || (model->type != LLM_TYPE_E2B && model->type != LLM_TYPE_26B_A4B)) {
        llama_model_free(model);
        throw std::runtime_error("prototype supports only Gemma 4 E2B (35 layers) and 26B A4B (30 layers)");
    }
    return model;
}

llama_context_params make_context_params(const options & opts) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = opts.ctx;
    params.n_batch = opts.batch_size;
    params.n_ubatch = std::min(opts.batch_size, 512);
    params.n_threads = params.n_threads_batch = std::max(1u, std::thread::hardware_concurrency());
    params.no_perf = false;
    return params;
}

std::vector<bool> selected_layers(const std::string & value, int32_t count, const options & opts, const heretic_analysis & analysis) {
    std::vector<bool> result(count, value.empty());
    if (!value.empty()) {
        std::stringstream stream(value);
        std::string part;
        while (std::getline(stream, part, ',')) {
            const size_t dash = part.find('-');
            const int32_t first = std::stoi(part.substr(0, dash));
            const int32_t last = dash == std::string::npos ? first : std::stoi(part.substr(dash + 1));
            if (first < 0 || last < first || last >= count) throw std::runtime_error("invalid --layers range");
            for (int32_t i = first; i <= last; ++i) result[i] = true;
        }
    }
    if (opts.direction_layer_mode == "topk") {
        std::vector<int32_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return analysis.layer_diagnostics[a].fisher_separation > analysis.layer_diagnostics[b].fisher_separation;
        });
        std::vector<bool> strongest(count);
        for (int32_t i = 0; i < std::min(opts.direction_topk, count); ++i) strongest[order[i]] = true;
        for (int32_t i = 0; i < count; ++i) result[i] = result[i] && strongest[i];
    }
    return result;
}

void unpack_alpha(const std::vector<float> & raw, int32_t n_layer, const options & opts, const std::vector<bool> & selected, std::vector<float> & attn, std::vector<float> & mlp) {
    attn.resize(n_layer);
    mlp.resize(n_layer);
    for (int32_t i = 0; i < n_layer; ++i) {
        attn[i] = selected[i] && opts.attn ? std::clamp(raw[i], 0.0f, opts.max_alpha_attn) : 0.0f;
        mlp[i] = selected[i] && (opts.mlp || opts.experts) ? std::clamp(raw[n_layer + i], 0.0f, opts.max_alpha_mlp) : 0.0f;
    }
}

size_t peak_ram_mib() {
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line)) {
        size_t value = 0;
        if (sscanf(line.c_str(), "VmHWM: %zu kB", &value) == 1) return value / 1024;
    }
    return 0;
}

json metrics_json(const heretic_metrics & value) {
    return {
        { "behavior_score", value.behavior_score },
        { "behavior_loss", value.behavior_loss },
        { "general_kl", value.general_kl },
        { "quantized_model_kl", value.general_kl },
        { "same_top1", value.same_top1 },
        { "mean_probability_deviation", value.mean_probability_deviation },
        { "total_loss", value.total_loss }, { "evaluation_seconds", value.evaluation_seconds },
        { "tokens_per_second", value.tokens_per_second }, { "evaluated_tokens", value.evaluated_tokens },
        { "target_count", value.target_count }, { "control_count", value.control_count },
        { "valid", value.valid }, { "rejection_reason", value.rejection_reason },
        { "response_samples", value.response_samples },
    };
}

heretic_metrics metrics_from_json(const json & value) {
    heretic_metrics result;
    result.behavior_score = value.value("behavior_score", 0.0);
    result.behavior_loss = value.value("behavior_loss", 0.0);
    result.general_kl = value.value("general_kl", 0.0);
    result.same_top1 = value.value("same_top1", 0.0);
    result.mean_probability_deviation = value.value("mean_probability_deviation", 0.0);
    result.total_loss = value.value("total_loss", 0.0);
    result.evaluation_seconds = value.value("evaluation_seconds", 0.0);
    result.tokens_per_second = value.value("tokens_per_second", 0.0);
    result.evaluated_tokens = value.value("evaluated_tokens", 0LL);
    result.target_count = value.value("target_count", 0LL);
    result.control_count = value.value("control_count", 0LL);
    result.valid = value.value("valid", true);
    result.rejection_reason = value.value("rejection_reason", "");
    result.response_samples = value.value("response_samples", std::vector<std::string>());
    return result;
}

struct alpha_statistics {
    double l1_mean = 0.0;
    double l2_norm = 0.0;
    int32_t nonzero = 0;
};

alpha_statistics alpha_stats(const std::vector<float> & attn, const std::vector<float> & mlp) {
    alpha_statistics result;
    double squares = 0.0;
    for (const std::vector<float> * values : { &attn, &mlp }) {
        for (float value : *values) {
            result.l1_mean += std::fabs(value);
            squares += value * value;
            if (value > 1e-7f) ++result.nonzero;
        }
    }
    result.l1_mean /= attn.size() + mlp.size();
    result.l2_norm = std::sqrt(squares);
    return result;
}

void set_objective(heretic_metrics & value, const std::vector<float> & attn, const std::vector<float> & mlp, const options & opts) {
    const alpha_statistics stats = alpha_stats(attn, mlp);
    value.total_loss = opts.lambda_behavior * value.behavior_loss + opts.lambda_kl * value.general_kl + opts.lambda_sparse * stats.l1_mean;
    if (value.general_kl > opts.max_kl) {
        value.valid = false;
        value.rejection_reason = "control KL exceeds --max-kl";
    }
    if (!std::isfinite(value.total_loss)) {
        value.valid = false;
        value.rejection_reason = "non-finite total loss";
    }
}

void print_analysis_table(const heretic_analysis & analysis) {
    printf("layer target_norm control_norm diff_norm cosine fisher n_target n_control\n");
    for (size_t layer = 0; layer < analysis.layer_diagnostics.size(); ++layer) {
        const heretic_layer_diagnostic & value = analysis.layer_diagnostics[layer];
        printf("%5zu %11.5f %12.5f %9.5f %7.4f %.8g %8lld %9lld\n", layer, value.target_mean_norm, value.control_mean_norm,
                value.difference_norm, value.mean_cosine, value.fisher_separation, (long long) value.target_count, (long long) value.control_count);
        const heretic_router_diagnostic & router = analysis.router_diagnostics[layer];
        if (!router.target_frequency.empty()) {
            std::vector<int32_t> target(router.target_frequency.size());
            std::vector<int32_t> control(router.control_frequency.size());
            std::iota(target.begin(), target.end(), 0);
            std::iota(control.begin(), control.end(), 0);
            const size_t top = std::min<size_t>(5, target.size());
            std::partial_sort(target.begin(), target.begin() + top, target.end(), [&](int32_t a, int32_t b) { return router.target_frequency[a] > router.target_frequency[b]; });
            std::partial_sort(control.begin(), control.begin() + top, control.end(), [&](int32_t a, int32_t b) { return router.control_frequency[a] > router.control_frequency[b]; });
            printf("      router_js=%.7f target_top=", router.js_divergence);
            for (size_t i = 0; i < top; ++i) printf("%s%d:%.3f", i ? "," : "", target[i], router.target_frequency[target[i]]);
            printf(" control_top=");
            for (size_t i = 0; i < top; ++i) printf("%s%d:%.3f", i ? "," : "", control[i], router.control_frequency[control[i]]);
            printf("\n");
        }
    }
}

json candidate_json(const candidate_record & value) {
    return {
        { "step", value.step }, { "branch", value.branch }, { "accepted", value.accepted },
        { "train", metrics_json(value.train) }, { "eval", metrics_json(value.eval) },
        { "alpha_attn", value.alpha_attn }, { "alpha_mlp", value.alpha_mlp },
        { "alpha", {
            { "l1_mean", alpha_stats(value.alpha_attn, value.alpha_mlp).l1_mean },
            { "l2_norm", alpha_stats(value.alpha_attn, value.alpha_mlp).l2_norm },
            { "nonzero", alpha_stats(value.alpha_attn, value.alpha_mlp).nonzero },
        } },
    };
}

candidate_record candidate_from_json(const json & value) {
    candidate_record result;
    result.step = value.value("step", 0);
    result.branch = value.value("branch", "");
    result.accepted = value.value("accepted", false);
    result.train = metrics_from_json(value.at("train"));
    result.eval = metrics_from_json(value.at("eval"));
    result.alpha_attn = value.at("alpha_attn").get<std::vector<float>>();
    result.alpha_mlp = value.at("alpha_mlp").get<std::vector<float>>();
    return result;
}

const heretic_metrics & selection_metrics(const candidate_record & candidate, bool use_eval) {
    return use_eval ? candidate.eval : candidate.train;
}

std::vector<candidate_record> pareto_frontier(const std::vector<candidate_record> & candidates, bool use_eval) {
    std::vector<candidate_record> result;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const heretic_metrics & current = selection_metrics(candidates[i], use_eval);
        if (!current.valid) continue;
        bool dominated = false;
        for (size_t j = 0; j < candidates.size() && !dominated; ++j) {
            if (i == j) continue;
            const heretic_metrics & other = selection_metrics(candidates[j], use_eval);
            dominated = other.valid && other.behavior_score <= current.behavior_score && other.general_kl <= current.general_kl &&
                    (other.behavior_score < current.behavior_score || other.general_kl < current.general_kl);
        }
        if (!dominated) result.push_back(candidates[i]);
    }
    std::sort(result.begin(), result.end(), [&](const candidate_record & a, const candidate_record & b) {
        return selection_metrics(a, use_eval).general_kl < selection_metrics(b, use_eval).general_kl;
    });
    return result;
}

void save_pareto(const std::string & output_path, const std::vector<candidate_record> & candidates, bool use_eval) {
    const std::vector<candidate_record> frontier = pareto_frontier(candidates, use_eval);
    json value = { { "format", "lumen-heretic-pareto" }, { "version", 1 }, { "selection", use_eval ? "eval" : "train" }, { "candidates", json::array() } };
    for (const candidate_record & candidate : frontier) value["candidates"].push_back(candidate_json(candidate));
    save_json(output_path + ".pareto.json", value);
    std::ofstream csv(output_path + ".pareto.csv", std::ios::trunc);
    if (!csv) throw std::runtime_error("failed to write Pareto CSV");
    csv << "step,branch,behavior_score,general_kl,total_loss,alpha_l1,alpha_l2,nonzero\n";
    for (const candidate_record & candidate : frontier) {
        const heretic_metrics & metric = selection_metrics(candidate, use_eval);
        const alpha_statistics stats = alpha_stats(candidate.alpha_attn, candidate.alpha_mlp);
        csv << candidate.step << ',' << candidate.branch << ',' << metric.behavior_score << ',' << metric.general_kl << ',' << metric.total_loss << ','
            << stats.l1_mean << ',' << stats.l2_norm << ',' << stats.nonzero << '\n';
    }
}

void save_json(const std::string & path, const json & value) {
    const std::string tmp = path + ".tmp";
    std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("failed to write " + path);
    output << value.dump(2) << '\n';
    output.close();
    if (!output || std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("failed to replace " + path);
}

}

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        const std::string model_path = resolve_model(opts.model);
        const std::string direction_path = opts.direction_dataset.empty() ? opts.dataset : opts.direction_dataset;
        const std::string train_path = opts.train_dataset.empty() ? opts.dataset : opts.train_dataset;
        std::vector<heretic_dataset_entry> direction_dataset = heretic_load_dataset(direction_path);
        std::vector<heretic_dataset_entry> train_dataset = opts.command != "analyze" ? heretic_load_dataset(train_path) : direction_dataset;
        std::vector<heretic_dataset_entry> eval_dataset;
        if (!opts.eval_dataset.empty()) eval_dataset = heretic_load_dataset(opts.eval_dataset);
        if (opts.eval_split > 0.0f && opts.eval_dataset.empty()) {
            auto split = heretic_stratified_split(train_dataset, opts.eval_split, opts.seed);
            train_dataset = std::move(split.first);
            eval_dataset = std::move(split.second);
            if (opts.direction_dataset.empty()) direction_dataset = train_dataset;
        }
        heretic_validate_dataset(direction_dataset, opts.minimum_target, opts.minimum_control, 16384);
        if (opts.command != "analyze") {
            heretic_validate_dataset(train_dataset, opts.minimum_target, opts.minimum_control, 16384);
            if (!eval_dataset.empty()) heretic_validate_dataset(eval_dataset, opts.minimum_target, opts.minimum_control, 16384);
        }

        common_init();
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
        llama_model * model = load_model(opts, model_path);
        const int32_t n_layer = llama_model_n_layer(model);
        direction_dataset = heretic_format_dataset(model, direction_dataset, opts.apply_chat_template);
        train_dataset = heretic_format_dataset(model, train_dataset, opts.apply_chat_template);
        eval_dataset = heretic_format_dataset(model, eval_dataset, opts.apply_chat_template);
        heretic_analysis analysis;
        const std::string model_fingerprint = common_moe_prune_sha256_file(model_path);
        std::string direction_fingerprint = common_moe_prune_sha256_file(direction_path);
        if (opts.eval_split > 0.0f && opts.direction_dataset.empty()) direction_fingerprint += ":split:" + std::to_string(opts.eval_split) + ":" + std::to_string(opts.seed);
        analysis.model_fingerprint = model_fingerprint;
        analysis.dataset_fingerprint = direction_fingerprint;

        const auto analysis_start = std::chrono::steady_clock::now();
        std::string error;
        if (!opts.analysis.empty()) {
            if (!heretic_load_analysis(opts.analysis, analysis, error)) throw std::runtime_error(error);
            if (analysis.model_fingerprint != model_fingerprint) throw std::runtime_error("analysis model fingerprint does not match input GGUF");
            if (analysis.dataset_fingerprint != direction_fingerprint) throw std::runtime_error("analysis fingerprint does not match direction dataset or split");
        } else if (!heretic_extract_directions(model, make_context_params(opts), direction_dataset, opts.residual_token, opts.orthogonalize_control, analysis, error)) {
            throw std::runtime_error(error);
        }
        const double analysis_load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - analysis_start).count();
        if (opts.analysis.empty()) analysis.analysis_seconds = analysis_load_seconds;
        if ((int32_t) analysis.directions.size() != n_layer) throw std::runtime_error("analysis layer count does not match model");
        print_analysis_table(analysis);

        if (opts.command == "analyze") {
            fs::create_directories(opts.output);
            const std::string path = (fs::path(opts.output) / "analysis.json").string();
            if (!heretic_save_analysis(path, analysis, error)) throw std::runtime_error(error);
            printf("saved %s (%d layers, %lld target, %lld control)\n", path.c_str(), n_layer, (long long) analysis.target_count, (long long) analysis.control_count);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }

        const fs::path output_parent = fs::path(opts.output).parent_path();
        if (!output_parent.empty()) fs::create_directories(output_parent);
        if (opts.save_analysis && !heretic_save_analysis(opts.output + ".analysis.json", analysis, error)) throw std::runtime_error(error);
        const heretic_prefix_scorer scorer(opts.refusal_patterns);
        heretic_evaluation_config evaluation_config;
        evaluation_config.kl_tokens = opts.kl_tokens;
        evaluation_config.behavior_tokens = opts.behavior_tokens;
        evaluation_config.ctx = opts.ctx;
        std::vector<heretic_cached_entry> train_cache = heretic_prepare_cache(model, train_dataset, opts.ctx);
        std::vector<heretic_cached_entry> eval_cache = heretic_prepare_cache(model, eval_dataset, opts.ctx);
        llama_context * context = llama_init_from_model(model, make_context_params(opts));
        if (!context) throw std::runtime_error("failed to create optimization context");
        const double peak_vram_mib = allocated_gpu_mib(context);
        std::vector<float> zero_alpha(n_layer);
        heretic_metrics before_train = heretic_evaluate(context, train_cache, true, scorer, evaluation_config);
        heretic_metrics before_eval = eval_cache.empty() ? before_train : heretic_evaluate(context, eval_cache, true, scorer, evaluation_config);
        set_objective(before_train, zero_alpha, zero_alpha, opts);
        set_objective(before_eval, zero_alpha, zero_alpha, opts);
        if (!before_train.valid || !before_eval.valid) throw std::runtime_error("source model evaluation failed");
        if (opts.command == "evaluate") {
            save_json(opts.output, {
                { "format", "lumen-heretic-evaluation" }, { "version", 1 },
                { "model_fingerprint", model_fingerprint }, { "train", metrics_json(before_train) }, { "eval", metrics_json(before_eval) },
                { "peak_ram_mib", peak_ram_mib() }, { "peak_vram_mib", peak_vram_mib },
                { "scorer", { { "type", "response-prefix-heuristic" }, { "patterns", opts.refusal_patterns } } },
            });
            llama_free(context);
            llama_model_free(model);
            llama_backend_free();
            printf("saved %s\n", opts.output.c_str());
            return 0;
        }

        heretic_runtime_editor editor(model_path, model);
        const std::vector<bool> selected = selected_layers(opts.layers, n_layer, opts, analysis);
        std::vector<float> raw(2*n_layer, opts.alpha_warm_start);
        std::vector<float> momentum(raw.size());
        std::vector<float> variance(raw.size());
        std::vector<float> best_alpha_attn(n_layer);
        std::vector<float> best_alpha_mlp(n_layer);
        std::vector<candidate_record> candidates;
        candidate_record baseline;
        baseline.branch = "baseline";
        baseline.train = before_train;
        baseline.eval = before_eval;
        baseline.alpha_attn = zero_alpha;
        baseline.alpha_mlp = zero_alpha;
        baseline.accepted = true;
        candidates.push_back(baseline);
        const bool use_eval = opts.selection == "eval" && !eval_cache.empty();
        double best_loss = use_eval ? before_eval.total_loss : before_train.total_loss;
        int32_t first_step = 0;
        int32_t stale_steps = 0;
        std::mt19937 rng(opts.seed);
        const std::string checkpoint_path = opts.output + ".checkpoint.json";
        if (opts.resume) {
            std::ifstream checkpoint(checkpoint_path);
            if (!checkpoint) throw std::runtime_error("resume checkpoint does not exist");
            json state;
            checkpoint >> state;
            if (state.value("format", "") != "lumen-heretic-checkpoint" || state.value("version", 0) != 3) throw std::runtime_error("unsupported checkpoint format");
            if (state.at("model_fingerprint") != analysis.model_fingerprint || state.at("direction_fingerprint") != analysis.dataset_fingerprint) throw std::runtime_error("checkpoint fingerprint mismatch");
            raw = state.at("alpha_state").get<std::vector<float>>();
            momentum = state.at("optimizer_m").get<std::vector<float>>();
            variance = state.at("optimizer_v").get<std::vector<float>>();
            best_alpha_attn = state.at("best_alpha_attn").get<std::vector<float>>();
            best_alpha_mlp = state.at("best_alpha_mlp").get<std::vector<float>>();
            best_loss = state.at("best_loss");
            first_step = state.at("step");
            stale_steps = state.value("stale_steps", 0);
            std::stringstream rng_state(state.at("rng_state").get<std::string>());
            rng_state >> rng;
            if (state.contains("candidate_history")) {
                candidates.clear();
                for (const json & candidate : state.at("candidate_history")) candidates.push_back(candidate_from_json(candidate));
            }
        }

        const auto optimization_start = std::chrono::steady_clock::now();
        std::uniform_int_distribution<int> sign_dist(0, 1);
        for (int32_t step = first_step; step < opts.steps && (opts.patience <= 0 || stale_steps < opts.patience); ++step) {
            std::vector<float> signs(raw.size());
            std::vector<float> plus = raw;
            std::vector<float> minus = raw;
            int32_t enabled_count = 0;
            for (size_t i = 0; i < raw.size(); ++i) {
                const int32_t layer = i % n_layer;
                const bool enabled = selected[layer] && ((i < (size_t) n_layer && opts.attn) || (i >= (size_t) n_layer && (opts.mlp || opts.experts)));
                signs[i] = enabled ? (sign_dist(rng) ? 1.0f : -1.0f) : 0.0f;
                enabled_count += enabled;
                plus[i] += opts.spsa_delta * signs[i];
                minus[i] -= opts.spsa_delta * signs[i];
            }

            auto evaluate_candidate = [&](const std::vector<float> & state, const char * branch) {
                candidate_record candidate;
                candidate.step = step + 1;
                candidate.branch = branch;
                unpack_alpha(state, n_layer, opts, selected, candidate.alpha_attn, candidate.alpha_mlp);
                if (!editor.apply(analysis.directions, candidate.alpha_attn, candidate.alpha_mlp, opts.attn, opts.mlp, opts.experts, error)) throw std::runtime_error(error);
                candidate.train = heretic_evaluate(context, train_cache, false, scorer, evaluation_config);
                set_objective(candidate.train, candidate.alpha_attn, candidate.alpha_mlp, opts);
                if (!eval_cache.empty() && ((step + 1) % opts.eval_every == 0)) {
                    candidate.eval = heretic_evaluate(context, eval_cache, false, scorer, evaluation_config);
                    set_objective(candidate.eval, candidate.alpha_attn, candidate.alpha_mlp, opts);
                } else if (eval_cache.empty()) {
                    candidate.eval = candidate.train;
                } else {
                    candidate.eval.valid = false;
                    candidate.eval.rejection_reason = "not evaluated on this step";
                }
                if (candidate.train.general_kl > opts.max_kl || (!eval_cache.empty() && candidate.eval.valid && candidate.eval.general_kl > opts.max_kl)) {
                    candidate.train.valid = false;
                    candidate.train.rejection_reason = "control KL exceeds --max-kl";
                }
                return candidate;
            };

            candidate_record candidate_plus = evaluate_candidate(plus, "plus");
            candidate_record candidate_minus = evaluate_candidate(minus, "minus");
            bool improved = false;
            for (candidate_record * candidate : { &candidate_plus, &candidate_minus }) {
                const heretic_metrics & selected_metrics = selection_metrics(*candidate, use_eval);
                if (candidate->train.valid && selected_metrics.valid && selected_metrics.total_loss < best_loss) {
                    best_loss = selected_metrics.total_loss;
                    best_alpha_attn = candidate->alpha_attn;
                    best_alpha_mlp = candidate->alpha_mlp;
                    candidate->accepted = true;
                    improved = true;
                }
                candidates.push_back(*candidate);
            }
            stale_steps = improved ? 0 : stale_steps + 1;
            const double plus_loss = candidate_plus.train.valid ? candidate_plus.train.total_loss : best_loss + 1000.0;
            const double minus_loss = candidate_minus.train.valid ? candidate_minus.train.total_loss : best_loss + 1000.0;
            const float directional_gradient = (float) ((plus_loss - minus_loss) / (2.0 * opts.spsa_delta));
            const double gradient_norm = std::fabs(directional_gradient) * std::sqrt((double) enabled_count);
            const int32_t iteration = step + 1;
            const float learning_rate = opts.lr_schedule == "cosine" ? opts.lr * 0.5f * (1.0f + std::cos(3.14159265358979323846f * step / std::max(1, opts.steps))) : opts.lr;
            for (size_t i = 0; i < raw.size(); ++i) {
                if (signs[i] == 0.0f) continue;
                const float gradient = directional_gradient * signs[i];
                momentum[i] = 0.9f * momentum[i] + 0.1f * gradient;
                variance[i] = 0.999f * variance[i] + 0.001f * gradient * gradient;
                const float m_hat = momentum[i] / (1.0f - std::pow(0.9f, iteration));
                const float v_hat = variance[i] / (1.0f - std::pow(0.999f, iteration));
                const float maximum = i < (size_t) n_layer ? opts.max_alpha_attn : opts.max_alpha_mlp;
                raw[i] = std::clamp(raw[i] - learning_rate * m_hat / (std::sqrt(v_hat) + 1e-8f), 0.0f, maximum);
            }
            std::vector<float> current_attn;
            std::vector<float> current_mlp;
            unpack_alpha(raw, n_layer, opts, selected, current_attn, current_mlp);
            const alpha_statistics current_stats = alpha_stats(current_attn, current_mlp);
            std::stringstream rng_state;
            rng_state << rng;
            json candidate_history = json::array();
            for (const candidate_record & candidate : candidates) candidate_history.push_back(candidate_json(candidate));
            save_json(checkpoint_path, {
                { "format", "lumen-heretic-checkpoint" }, { "version", 3 }, { "step", iteration },
                { "model_fingerprint", analysis.model_fingerprint }, { "direction_fingerprint", analysis.dataset_fingerprint },
                { "directions", analysis.directions }, { "direction_norms", analysis.direction_norms },
                { "alpha_state", raw }, { "optimizer_m", momentum }, { "optimizer_v", variance },
                { "best_alpha_attn", best_alpha_attn }, { "best_alpha_mlp", best_alpha_mlp }, { "best_loss", best_loss },
                { "stale_steps", stale_steps }, { "rng_state", rng_state.str() }, { "seed", opts.seed },
                { "candidate_history", std::move(candidate_history) },
                { "configuration", {
                    { "quant", opts.quant }, { "layers", opts.layers }, { "direction_layer_mode", opts.direction_layer_mode },
                    { "attn", opts.attn }, { "mlp", opts.mlp }, { "experts", opts.experts },
                    { "max_alpha_attn", opts.max_alpha_attn }, { "max_alpha_mlp", opts.max_alpha_mlp }, { "lr", opts.lr },
                    { "lr_schedule", opts.lr_schedule }, { "lambda_behavior", opts.lambda_behavior }, { "lambda_kl", opts.lambda_kl },
                    { "lambda_sparse", opts.lambda_sparse }, { "spsa_delta", opts.spsa_delta }, { "max_kl", opts.max_kl },
                    { "kl_tokens", opts.kl_tokens }, { "behavior_tokens", opts.behavior_tokens }, { "selection", use_eval ? "eval" : "train" },
                } },
            });
            printf("step %d/%d train_plus=%.7f train_minus=%.7f best=%.7f grad_norm=%.5g lr=%.5g alpha_l1=%.5g alpha_l2=%.5g nonzero=%d stale=%d\n",
                    iteration, opts.steps, candidate_plus.train.total_loss, candidate_minus.train.total_loss, best_loss, gradient_norm, learning_rate,
                    current_stats.l1_mean, current_stats.l2_norm, current_stats.nonzero, stale_steps);
        }

        if (!editor.apply(analysis.directions, best_alpha_attn, best_alpha_mlp, opts.attn, opts.mlp, opts.experts, error)) throw std::runtime_error(error);
        heretic_metrics after_train = heretic_evaluate(context, train_cache, false, scorer, evaluation_config);
        heretic_metrics after_eval = eval_cache.empty() ? after_train : heretic_evaluate(context, eval_cache, false, scorer, evaluation_config);
        set_objective(after_train, best_alpha_attn, best_alpha_mlp, opts);
        set_objective(after_eval, best_alpha_attn, best_alpha_mlp, opts);
        const double optimization_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - optimization_start).count();
        const alpha_statistics final_alpha = alpha_stats(best_alpha_attn, best_alpha_mlp);
        candidate_record final_candidate;
        final_candidate.step = opts.steps;
        final_candidate.branch = "restored_best";
        final_candidate.train = after_train;
        final_candidate.eval = after_eval;
        final_candidate.alpha_attn = best_alpha_attn;
        final_candidate.alpha_mlp = best_alpha_mlp;
        final_candidate.accepted = true;
        candidates.push_back(std::move(final_candidate));
        save_pareto(opts.output, candidates, use_eval);
        llama_free(context);
        if (!heretic_rewrite_gguf(model_path, opts.output, analysis.directions, best_alpha_attn, best_alpha_mlp, opts.attn, opts.mlp, opts.experts, error)) throw std::runtime_error(error);

        std::vector<int32_t> selected_layer_ids;
        for (int32_t layer = 0; layer < n_layer; ++layer) if (selected[layer]) selected_layer_ids.push_back(layer);
        int32_t edited_layer_count = 0;
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            if (best_alpha_attn[layer] > 1e-7f || best_alpha_mlp[layer] > 1e-7f) ++edited_layer_count;
        }
        const double train_improvement = before_train.behavior_score - after_train.behavior_score;
        const double eval_improvement = before_eval.behavior_score - after_eval.behavior_score;
        json report = {
            { "format", "lumen-heretic-report" }, { "version", 2 },
            { "architecture", "gemma4" }, { "model_type", model->type == LLM_TYPE_E2B ? "E2B" : "26B_A4B" },
            { "model_fingerprint", analysis.model_fingerprint }, { "direction_fingerprint", analysis.dataset_fingerprint },
            { "dataset", { { "direction", direction_path }, { "train", train_path }, { "eval", opts.eval_dataset }, { "eval_split", opts.eval_split } } },
            { "dataset_strength", { { "direction_target", analysis.target_count }, { "direction_control", analysis.control_count }, { "statistically_weak", analysis.target_count < opts.minimum_target || analysis.control_count < opts.minimum_control } } },
            { "quant", "q4_0" }, { "optimizer", "spsa-adam" }, { "selection", use_eval ? "eval" : "train" },
            { "configuration", {
                { "steps", opts.steps }, { "seed", opts.seed }, { "layers", opts.layers },
                { "direction_layer_mode", opts.direction_layer_mode }, { "direction_topk", opts.direction_topk },
                { "attn", opts.attn }, { "mlp", opts.mlp }, { "experts", opts.experts },
                { "alpha_warm_start", opts.alpha_warm_start }, { "max_alpha_attn", opts.max_alpha_attn }, { "max_alpha_mlp", opts.max_alpha_mlp },
                { "lr", opts.lr }, { "lr_schedule", opts.lr_schedule }, { "spsa_delta", opts.spsa_delta },
                { "lambda_behavior", opts.lambda_behavior }, { "lambda_kl", opts.lambda_kl }, { "lambda_sparse", opts.lambda_sparse },
                { "max_kl", opts.max_kl }, { "kl_tokens", opts.kl_tokens }, { "behavior_tokens", opts.behavior_tokens },
                { "eval_every", opts.eval_every }, { "patience", opts.patience },
            } },
            { "train", { { "original", metrics_json(before_train) }, { "edited", metrics_json(after_train) }, { "behavior_improvement_per_kl", train_improvement / (after_train.general_kl + 1e-12) } } },
            { "eval", { { "original", metrics_json(before_eval) }, { "edited", metrics_json(after_eval) }, { "behavior_improvement_per_kl", eval_improvement / (after_eval.general_kl + 1e-12) } } },
            { "alpha_attn", best_alpha_attn }, { "alpha_mlp", best_alpha_mlp },
            { "alpha_statistics", { { "l1_mean", final_alpha.l1_mean }, { "l2_norm", final_alpha.l2_norm }, { "nonzero", final_alpha.nonzero } } },
            { "selected_layers", selected_layer_ids }, { "edited_layers", edited_layer_count }, { "best_loss", best_loss },
            { "peak_ram_mib", peak_ram_mib() }, { "peak_vram_mib", peak_vram_mib },
            { "analysis_seconds", analysis.analysis_seconds }, { "analysis_load_seconds", analysis_load_seconds }, { "optimization_seconds", optimization_seconds },
            { "refusal_scorer", { { "type", "response-prefix-heuristic" }, { "patterns", opts.refusal_patterns }, { "pattern_count", scorer.pattern_count() } } },
            { "experts_strategy", opts.experts ? "shared-layer-direction-and-alpha" : "disabled" }, { "recovery_qat", "disabled" },
        };
        save_json(opts.output + ".report.json", report);
        printf("train behavior %.6f -> %.6f, KL %.7f; eval behavior %.6f -> %.6f, KL %.7f; nonzero alpha %d\n",
                before_train.behavior_score, after_train.behavior_score, after_train.general_kl,
                before_eval.behavior_score, after_eval.behavior_score, after_eval.general_kl, final_alpha.nonzero);
        printf("saved %s, report, checkpoint, and Pareto files\n", opts.output.c_str());
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
