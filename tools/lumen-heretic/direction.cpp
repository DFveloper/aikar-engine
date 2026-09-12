#include "direction.h"

#include "common.h"
#include "llama-ext.h"
#include "nlohmann/json.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

using json = nlohmann::ordered_json;

namespace {

json diagnostic_json(const heretic_layer_diagnostic & value) {
    return {
        { "target_mean_norm", value.target_mean_norm }, { "control_mean_norm", value.control_mean_norm },
        { "difference_norm", value.difference_norm }, { "mean_cosine", value.mean_cosine },
        { "target_variance_trace", value.target_variance_trace }, { "control_variance_trace", value.control_variance_trace },
        { "fisher_separation", value.fisher_separation }, { "target_count", value.target_count }, { "control_count", value.control_count },
    };
}

json router_json(const heretic_router_diagnostic & value) {
    return {
        { "target_frequency", value.target_frequency }, { "control_frequency", value.control_frequency }, { "js_divergence", value.js_divergence },
    };
}

struct residual_accumulator {
    int32_t n_layer;
    int32_t n_embd;
    int32_t n_tokens = 0;
    int32_t residual_token = -1;
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    bool target = false;
    std::vector<std::vector<float>> sum_target;
    std::vector<std::vector<float>> sum_control;
    std::vector<std::vector<float>> sum_sq_target;
    std::vector<std::vector<float>> sum_sq_control;
    std::vector<std::vector<double>> router_target;
    std::vector<std::vector<double>> router_control;
    std::vector<int64_t> count_target;
    std::vector<int64_t> count_control;

    residual_accumulator(int32_t layers, int32_t embedding, int32_t experts, int32_t experts_used) :
            n_layer(layers),
            n_embd(embedding),
            n_expert(experts),
            n_expert_used(experts_used),
            sum_target(layers, std::vector<float>(embedding)),
            sum_control(layers, std::vector<float>(embedding)),
            sum_sq_target(layers, std::vector<float>(embedding)),
            sum_sq_control(layers, std::vector<float>(embedding)),
            router_target(layers, std::vector<double>(experts)),
            router_control(layers, std::vector<double>(experts)),
            count_target(layers),
            count_control(layers) {
    }
};

int32_t residual_layer(const char * name) {
    int32_t layer = -1;
    return sscanf(name, "l_out-%d", &layer) == 1 ? layer : -1;
}

int32_t router_layer(const char * name) {
    int32_t layer = -1;
    return sscanf(name, "ffn_moe_logits-%d", &layer) == 1 ? layer : -1;
}

bool residual_callback(ggml_tensor * tensor, bool ask, void * userdata) {
    residual_accumulator & state = *(residual_accumulator *) userdata;
    const int32_t layer = residual_layer(ggml_get_name(tensor));
    const int32_t moe_layer = router_layer(ggml_get_name(tensor));
    if (ask) return layer >= 0 || moe_layer >= 0;
    int32_t token = state.residual_token;
    if (token < 0) token += state.n_tokens;
    if (token < 0 || token >= state.n_tokens) return true;
    if (moe_layer >= 0 && moe_layer < state.n_layer && state.n_expert > 0 && tensor->type == GGML_TYPE_F32 && tensor->ne[0] == state.n_expert && tensor->ne[1] == state.n_tokens) {
        std::vector<float> logits(state.n_expert);
        ggml_backend_tensor_get(tensor, logits.data(), token * tensor->nb[1], logits.size() * sizeof(float));
        std::vector<int32_t> order(state.n_expert);
        std::iota(order.begin(), order.end(), 0);
        const int32_t used = std::min(state.n_expert_used, state.n_expert);
        std::partial_sort(order.begin(), order.begin() + used, order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        std::vector<double> & counts = state.target ? state.router_target[moe_layer] : state.router_control[moe_layer];
        for (int32_t i = 0; i < used; ++i) counts[order[i]] += 1.0;
        return true;
    }
    if (layer < 0 || layer >= state.n_layer || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != state.n_embd || tensor->ne[1] != state.n_tokens) return true;
    std::vector<float> values(state.n_embd);
    ggml_backend_tensor_get(tensor, values.data(), token * tensor->nb[1], values.size() * sizeof(float));
    std::vector<float> & sum = state.target ? state.sum_target[layer] : state.sum_control[layer];
    std::vector<float> & sum_sq = state.target ? state.sum_sq_target[layer] : state.sum_sq_control[layer];
    for (int32_t i = 0; i < state.n_embd; ++i) {
        sum[i] += values[i];
        sum_sq[i] += values[i] * values[i];
    }
    if (state.target) ++state.count_target[layer];
    else ++state.count_control[layer];
    return true;
}

}

bool heretic_extract_directions(
        llama_model * model,
        llama_context_params context_params,
        const std::vector<heretic_dataset_entry> & dataset,
        int32_t residual_token,
        bool orthogonalize_control,
        heretic_analysis & analysis,
        std::string & error) {
    try {
        const int32_t n_layer = llama_model_n_layer(model);
        const int32_t n_embd = llama_model_n_embd(model);
        residual_accumulator accumulator(n_layer, n_embd, llama_model_n_expert(model), llama_model_n_expert_used(model));
        accumulator.residual_token = residual_token;
        context_params.cb_eval = residual_callback;
        context_params.cb_eval_user_data = &accumulator;
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) throw std::runtime_error("failed to create residual-analysis context");
        for (const heretic_dataset_entry & entry : dataset) {
            std::vector<llama_token> tokens = common_tokenize(llama_model_get_vocab(model), entry.prompt, true, true);
            if (tokens.empty()) {
                llama_free(context);
                throw std::runtime_error("JSONL line " + std::to_string(entry.line) + ": prompt tokenized to an empty sequence");
            }
            if (tokens.size() > llama_n_batch(context) || tokens.size() > llama_n_ctx(context)) {
                llama_free(context);
                throw std::runtime_error("JSONL line " + std::to_string(entry.line) + ": prompt exceeds --batch-size or --ctx");
            }
            accumulator.n_tokens = (int32_t) tokens.size();
            accumulator.target = entry.group == "target";
            llama_memory_clear(llama_get_memory(context), true);
            if (llama_decode(context, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
                llama_free(context);
                throw std::runtime_error("JSONL line " + std::to_string(entry.line) + ": residual forward failed");
            }
        }
        llama_free(context);

        analysis.directions.assign(n_layer, std::vector<float>(n_embd));
        analysis.direction_norms.assign(n_layer, 0.0f);
        analysis.layer_diagnostics.assign(n_layer, {});
        analysis.router_diagnostics.assign(n_layer, {});
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            if (accumulator.count_target[layer] == 0 || accumulator.count_control[layer] == 0) throw std::runtime_error("each residual layer requires target and control observations");
            std::vector<float> control(n_embd);
            double control_norm = 0.0;
            double target_norm = 0.0;
            double target_control_dot = 0.0;
            double target_variance = 0.0;
            double control_variance = 0.0;
            for (int32_t i = 0; i < n_embd; ++i) {
                const float target = accumulator.sum_target[layer][i] / accumulator.count_target[layer];
                control[i] = accumulator.sum_control[layer][i] / accumulator.count_control[layer];
                analysis.directions[layer][i] = target - control[i];
                target_norm += target * target;
                control_norm += control[i] * control[i];
                target_control_dot += target * control[i];
                target_variance += std::max(0.0, (double) accumulator.sum_sq_target[layer][i] / accumulator.count_target[layer] - target * target);
                control_variance += std::max(0.0, (double) accumulator.sum_sq_control[layer][i] / accumulator.count_control[layer] - control[i] * control[i]);
            }
            if (orthogonalize_control && control_norm > 0.0) {
                double dot = 0.0;
                for (int32_t i = 0; i < n_embd; ++i) dot += analysis.directions[layer][i] * control[i];
                const float scale = (float) (dot / control_norm);
                for (int32_t i = 0; i < n_embd; ++i) analysis.directions[layer][i] -= scale * control[i];
            }
            double norm = 0.0;
            for (float value : analysis.directions[layer]) norm += value * value;
            norm = std::sqrt(norm);
            if (!std::isfinite(norm) || norm <= 1e-12) throw std::runtime_error("layer " + std::to_string(layer) + " produced a non-finite or zero direction");
            analysis.direction_norms[layer] = (float) norm;
            heretic_layer_diagnostic & diagnostic = analysis.layer_diagnostics[layer];
            diagnostic.target_mean_norm = std::sqrt(target_norm);
            diagnostic.control_mean_norm = std::sqrt(control_norm);
            diagnostic.difference_norm = norm;
            diagnostic.mean_cosine = target_control_dot / (std::sqrt(target_norm * control_norm) + 1e-30);
            diagnostic.target_variance_trace = target_variance;
            diagnostic.control_variance_trace = control_variance;
            diagnostic.fisher_separation = norm * norm / (target_variance + control_variance + 1e-12);
            diagnostic.target_count = accumulator.count_target[layer];
            diagnostic.control_count = accumulator.count_control[layer];
            for (float & value : analysis.directions[layer]) value /= (float) norm;

            const double target_routes = std::accumulate(accumulator.router_target[layer].begin(), accumulator.router_target[layer].end(), 0.0);
            const double control_routes = std::accumulate(accumulator.router_control[layer].begin(), accumulator.router_control[layer].end(), 0.0);
            if (target_routes > 0.0 && control_routes > 0.0) {
                heretic_router_diagnostic & router = analysis.router_diagnostics[layer];
                router.target_frequency.resize(accumulator.n_expert);
                router.control_frequency.resize(accumulator.n_expert);
                for (int32_t expert = 0; expert < accumulator.n_expert; ++expert) {
                    const double p = (accumulator.router_target[layer][expert] + 1e-9) / (target_routes + 1e-9 * accumulator.n_expert);
                    const double q = (accumulator.router_control[layer][expert] + 1e-9) / (control_routes + 1e-9 * accumulator.n_expert);
                    const double m = 0.5 * (p + q);
                    router.target_frequency[expert] = p;
                    router.control_frequency[expert] = q;
                    router.js_divergence += 0.5 * p * std::log(p / m) + 0.5 * q * std::log(q / m);
                }
            }
        }
        analysis.target_count = accumulator.count_target[0];
        analysis.control_count = accumulator.count_control[0];
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool heretic_save_analysis(const std::string & path, const heretic_analysis & analysis, std::string & error) {
    try {
        json diagnostics = json::array();
        for (const heretic_layer_diagnostic & item : analysis.layer_diagnostics) diagnostics.push_back(diagnostic_json(item));
        json routers = json::array();
        for (const heretic_router_diagnostic & item : analysis.router_diagnostics) routers.push_back(router_json(item));
        json value = {
            { "format", "lumen-heretic-analysis" },
            { "version", 2 },
            { "model_fingerprint", analysis.model_fingerprint },
            { "dataset_fingerprint", analysis.dataset_fingerprint },
            { "target_count", analysis.target_count },
            { "control_count", analysis.control_count },
            { "analysis_seconds", analysis.analysis_seconds },
            { "direction_norms", analysis.direction_norms },
            { "layer_diagnostics", diagnostics },
            { "router_diagnostics", routers },
            { "directions", analysis.directions },
        };
        const std::string tmp = path + ".tmp";
        std::ofstream output(tmp, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("failed to open analysis output");
        output << value.dump() << '\n';
        output.close();
        if (!output || std::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error("failed to save analysis atomically");
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool heretic_load_analysis(const std::string & path, heretic_analysis & analysis, std::string & error) {
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("failed to open analysis file");
        json value;
        input >> value;
        const int32_t version = value.value("version", 0);
        if (value.value("format", "") != "lumen-heretic-analysis" || (version != 1 && version != 2)) throw std::runtime_error("unsupported analysis format");
        analysis.model_fingerprint = value.at("model_fingerprint").get<std::string>();
        analysis.dataset_fingerprint = value.at("dataset_fingerprint").get<std::string>();
        analysis.target_count = value.at("target_count").get<int64_t>();
        analysis.control_count = value.at("control_count").get<int64_t>();
        analysis.analysis_seconds = value.value("analysis_seconds", 0.0);
        analysis.direction_norms = value.at("direction_norms").get<std::vector<float>>();
        analysis.directions = value.at("directions").get<std::vector<std::vector<float>>>();
        analysis.layer_diagnostics.assign(analysis.directions.size(), {});
        analysis.router_diagnostics.assign(analysis.directions.size(), {});
        if (version == 2) {
            const json & diagnostics = value.at("layer_diagnostics");
            const json & routers = value.at("router_diagnostics");
            if (diagnostics.size() != analysis.directions.size() || routers.size() != analysis.directions.size()) throw std::runtime_error("analysis diagnostic layer count mismatch");
            for (size_t i = 0; i < diagnostics.size(); ++i) {
                heretic_layer_diagnostic & item = analysis.layer_diagnostics[i];
                item.target_mean_norm = diagnostics[i].at("target_mean_norm");
                item.control_mean_norm = diagnostics[i].at("control_mean_norm");
                item.difference_norm = diagnostics[i].at("difference_norm");
                item.mean_cosine = diagnostics[i].at("mean_cosine");
                item.target_variance_trace = diagnostics[i].at("target_variance_trace");
                item.control_variance_trace = diagnostics[i].at("control_variance_trace");
                item.fisher_separation = diagnostics[i].at("fisher_separation");
                item.target_count = diagnostics[i].at("target_count");
                item.control_count = diagnostics[i].at("control_count");
                analysis.router_diagnostics[i].target_frequency = routers[i].at("target_frequency").get<std::vector<double>>();
                analysis.router_diagnostics[i].control_frequency = routers[i].at("control_frequency").get<std::vector<double>>();
                analysis.router_diagnostics[i].js_divergence = routers[i].at("js_divergence");
            }
        } else {
            for (size_t i = 0; i < analysis.directions.size(); ++i) {
                analysis.layer_diagnostics[i].difference_norm = analysis.direction_norms[i];
                analysis.layer_diagnostics[i].fisher_separation = analysis.direction_norms[i];
                analysis.layer_diagnostics[i].target_count = analysis.target_count;
                analysis.layer_diagnostics[i].control_count = analysis.control_count;
            }
        }
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
