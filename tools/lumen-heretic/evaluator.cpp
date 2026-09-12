#include "evaluator.h"

#include "common.h"
#include "chat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>

namespace {

std::vector<float> probabilities(const float * logits, int32_t count) {
    const float maximum = *std::max_element(logits, logits + count);
    std::vector<float> result(count);
    double sum = 0.0;
    for (int32_t i = 0; i < count; ++i) {
        result[i] = std::exp(logits[i] - maximum);
        sum += result[i];
    }
    for (float & value : result) value /= (float) sum;
    return result;
}

bool repetitive(const std::vector<llama_token> & tokens) {
    if (tokens.size() < 8) return false;
    bool same = true;
    for (size_t i = tokens.size() - 7; i < tokens.size(); ++i) same = same && tokens[i] == tokens[i - 1];
    const size_t begin = tokens.size() > 16 ? tokens.size() - 16 : 0;
    std::set<llama_token> unique(tokens.begin() + begin, tokens.end());
    return same || (tokens.size() - begin >= 12 && unique.size() <= 2);
}

bool decode_token(llama_context * context, llama_token token) {
    return llama_decode(context, llama_batch_get_one(&token, 1)) == 0;
}

}

std::vector<heretic_dataset_entry> heretic_format_dataset(
        const llama_model * model,
        const std::vector<heretic_dataset_entry> & dataset,
        bool apply_chat_template) {
    if (!apply_chat_template) return dataset;
    common_chat_templates_ptr templates = common_chat_templates_init(model, "");
    if (!templates) throw std::runtime_error("model has no usable chat template; use --raw-prompts for preformatted prompts");
    std::vector<heretic_dataset_entry> result = dataset;
    for (heretic_dataset_entry & entry : result) {
        common_chat_templates_inputs inputs;
        common_chat_msg message;
        message.role = "user";
        message.content = entry.prompt;
        inputs.messages.push_back(std::move(message));
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = false;
        try {
            entry.prompt = common_chat_templates_apply(templates.get(), inputs).prompt;
        } catch (const std::exception & e) {
            throw std::runtime_error("failed to apply model chat template to JSONL line " + std::to_string(entry.line) + ": " + e.what());
        }
    }
    return result;
}

std::vector<heretic_cached_entry> heretic_prepare_cache(
        const llama_model * model,
        const std::vector<heretic_dataset_entry> & dataset,
        int32_t ctx) {
    std::vector<heretic_cached_entry> result;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    for (const heretic_dataset_entry & source : dataset) {
        heretic_cached_entry entry;
        entry.prompt = common_tokenize(vocab, source.prompt, true, true);
        if (entry.prompt.empty()) throw std::runtime_error("JSONL line " + std::to_string(source.line) + ": prompt tokenized to an empty sequence");
        if ((int32_t) entry.prompt.size() >= ctx) throw std::runtime_error("JSONL line " + std::to_string(source.line) + ": prompt leaves no generation space in --ctx");
        entry.group = source.group;
        entry.desired = source.desired;
        entry.desired_score = source.desired_score;
        for (const std::string & text : source.target_tokens) {
            const std::vector<llama_token> tokens = common_tokenize(vocab, text, false, true);
            if (tokens.size() != 1) throw std::runtime_error("JSONL line " + std::to_string(source.line) + ": each target token must encode as exactly one token");
            entry.target_tokens.push_back(tokens[0]);
        }
        if (entry.desired == "target_token" && entry.target_tokens.empty()) throw std::runtime_error("JSONL line " + std::to_string(source.line) + ": target_token scoring requires target_token or target_tokens");
        result.push_back(std::move(entry));
    }
    return result;
}

heretic_metrics heretic_evaluate(
        llama_context * context,
        std::vector<heretic_cached_entry> & dataset,
        bool cache_reference,
        const heretic_prefix_scorer & scorer,
        const heretic_evaluation_config & config) {
    const auto start = std::chrono::steady_clock::now();
    const llama_model * model = llama_get_model(context);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    heretic_metrics result;
    int64_t control_positions = 0;
    try {
        for (heretic_cached_entry & entry : dataset) {
            llama_memory_clear(llama_get_memory(context), true);
            if (llama_decode(context, llama_batch_get_one(entry.prompt.data(), entry.prompt.size())) != 0) throw std::runtime_error("prompt decode failed");
            if (entry.group == "target") {
                ++result.target_count;
                const float * first_logits = llama_get_logits_ith(context, -1);
                if (!first_logits) throw std::runtime_error("target evaluation produced no logits");
                if (entry.desired == "target_token") {
                    const std::vector<float> current = probabilities(first_logits, n_vocab);
                    double score = 0.0;
                    for (llama_token token : entry.target_tokens) score += current[token];
                    score = std::min(1.0 - 1e-9, std::max(1e-9, score));
                    result.behavior_score += score;
                    result.behavior_loss += -entry.desired_score * std::log(score) - (1.0 - entry.desired_score) * std::log(1.0 - score);
                    continue;
                }
                std::vector<llama_token> generated;
                for (int32_t position = 0; position < config.behavior_tokens && (int32_t) (entry.prompt.size() + generated.size()) < config.ctx; ++position) {
                    const float * logits = llama_get_logits_ith(context, -1);
                    if (!logits) throw std::runtime_error("target generation produced no logits");
                    const llama_token token = (llama_token) (std::max_element(logits, logits + n_vocab) - logits);
                    if (llama_vocab_is_eog(vocab, token)) break;
                    generated.push_back(token);
                    ++result.evaluated_tokens;
                    if (!decode_token(context, token)) throw std::runtime_error("target generation decode failed");
                }
                if (repetitive(generated)) throw std::runtime_error("catastrophic repetitive target output");
                if (generated.empty()) throw std::runtime_error("target generation produced an empty response");
                const std::string response = common_detokenize(vocab, generated, false);
                const double score = scorer.score(response);
                if (result.response_samples.size() < 5) result.response_samples.push_back(std::to_string(score) + "\t" + response);
                result.behavior_score += score;
                result.behavior_loss += score;
            } else {
                ++result.control_count;
                if (cache_reference) entry.reference.clear();
                for (int32_t position = 0; position < config.kl_tokens && (int32_t) (entry.prompt.size() + position) < config.ctx; ++position) {
                    const float * logits = llama_get_logits_ith(context, -1);
                    if (!logits) throw std::runtime_error("control evaluation produced no logits");
                    const std::vector<float> current = probabilities(logits, n_vocab);
                    llama_token reference_token;
                    if (cache_reference) {
                        reference_token = (llama_token) (std::max_element(current.begin(), current.end()) - current.begin());
                        entry.reference.push_back({ current, reference_token });
                    } else {
                        if ((size_t) position >= entry.reference.size()) throw std::runtime_error("control reference cache is incomplete");
                        const heretic_reference_position & reference_position = entry.reference[position];
                        reference_token = reference_position.token;
                        for (int32_t token = 0; token < n_vocab; ++token) {
                            const double reference = std::max(1e-30f, reference_position.probabilities[token]);
                            const double edited = std::max(1e-30f, current[token]);
                            result.general_kl += reference * std::log(reference / edited);
                            result.mean_probability_deviation += std::fabs(reference - edited);
                        }
                        const llama_token top1 = (llama_token) (std::max_element(current.begin(), current.end()) - current.begin());
                        result.same_top1 += top1 == reference_token ? 1.0 : 0.0;
                        ++control_positions;
                    }
                    ++result.evaluated_tokens;
                    if (llama_vocab_is_eog(vocab, reference_token)) break;
                    if (!decode_token(context, reference_token)) throw std::runtime_error("control teacher-forcing decode failed");
                }
            }
        }
    } catch (const std::exception & e) {
        result.valid = false;
        result.rejection_reason = e.what();
    }
    if (result.target_count > 0) {
        result.behavior_score /= result.target_count;
        result.behavior_loss /= result.target_count;
    }
    if (control_positions > 0) {
        result.general_kl /= control_positions;
        result.same_top1 /= control_positions;
        result.mean_probability_deviation /= control_positions * n_vocab;
    } else if (cache_reference && result.control_count > 0) {
        result.same_top1 = 1.0;
    }
    result.evaluation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result.tokens_per_second = result.evaluation_seconds > 0.0 ? result.evaluated_tokens / result.evaluation_seconds : 0.0;
    if (!std::isfinite(result.behavior_score) || !std::isfinite(result.general_kl) || !std::isfinite(result.mean_probability_deviation)) {
        result.valid = false;
        result.rejection_reason = "non-finite evaluation metric";
    }
    return result;
}
