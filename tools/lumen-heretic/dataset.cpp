#include "dataset.h"

#include "jsonl.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::ordered_json;

namespace {

bool blank(const std::string & value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
}

}

std::vector<heretic_dataset_entry> heretic_load_dataset(const std::string & path) {
    const std::vector<common_jsonl_line> lines = common_jsonl_read_lines(path, COMMON_JSONL_EMPTY_LINE_SKIP);
    std::vector<heretic_dataset_entry> result;
    result.reserve(lines.size());
    for (const common_jsonl_line & source : lines) {
        try {
            const json value = json::parse(source.text);
            if (!value.is_object() || !value.contains("prompt") || !value["prompt"].is_string()) throw std::runtime_error("'prompt' must be a string");
            heretic_dataset_entry entry;
            entry.prompt = value["prompt"].get<std::string>();
            entry.line = source.number;
            if (entry.prompt.empty() || blank(entry.prompt)) throw std::runtime_error("'prompt' must not be empty");
            if (value.contains("type")) entry.group = value["type"].get<std::string>();
            else if (value.contains("group")) entry.group = value["group"].get<std::string>();
            else throw std::runtime_error("'type' must be 'target' or 'control'");
            if (entry.group != "target" && entry.group != "control") throw std::runtime_error("'type' must be 'target' or 'control'");
            entry.desired = value.value("desired", entry.group == "target" ? "non_refusal" : "preserve");
            if (entry.group == "target" && entry.desired != "non_refusal" && entry.desired != "target_token") throw std::runtime_error("target 'desired' must be 'non_refusal' or 'target_token'");
            if (entry.group == "control" && entry.desired != "preserve") throw std::runtime_error("control 'desired' must be 'preserve'");
            entry.desired_score = value.value("desired_score", entry.group == "target" ? 0.0f : 1.0f);
            if (value.contains("target_token")) entry.target_tokens.push_back(value["target_token"].get<std::string>());
            if (value.contains("target_tokens")) {
                if (!value["target_tokens"].is_array()) throw std::runtime_error("'target_tokens' must be an array");
                for (const json & token : value["target_tokens"]) entry.target_tokens.push_back(token.get<std::string>());
            }
            if (!entry.target_tokens.empty() && !value.contains("desired")) entry.desired = "target_token";
            result.push_back(std::move(entry));
        } catch (const std::exception & e) {
            throw std::runtime_error("JSONL line " + std::to_string(source.number) + ": " + e.what());
        }
    }
    if (result.empty()) throw std::runtime_error("behavior dataset is empty");
    return result;
}

heretic_dataset_summary heretic_validate_dataset(
        const std::vector<heretic_dataset_entry> & dataset,
        int64_t minimum_target,
        int64_t minimum_control,
        size_t long_prompt_bytes) {
    heretic_dataset_summary result;
    std::unordered_map<std::string, int32_t> groups;
    std::unordered_set<std::string> seen;
    for (const heretic_dataset_entry & entry : dataset) {
        if (entry.group == "target") ++result.target_count;
        else ++result.control_count;
        const int32_t group = entry.group == "target" ? 1 : 2;
        const auto prior = groups.find(entry.prompt);
        if (prior != groups.end() && prior->second != group) ++result.overlap_count;
        groups[entry.prompt] = group;
        if (!seen.insert(entry.group + "\n" + entry.prompt).second) ++result.duplicate_count;
        if (entry.prompt.size() > long_prompt_bytes) ++result.long_prompt_count;
    }
    if (result.target_count == 0 || result.control_count == 0) throw std::runtime_error("dataset requires at least one target and one control prompt");
    if (result.overlap_count > 0) throw std::runtime_error("dataset contains prompts in both target and control groups");
    result.statistically_weak = result.target_count < minimum_target || result.control_count < minimum_control;
    if (result.duplicate_count > 0) fprintf(stderr, "warning: dataset contains %lld duplicate records\n", (long long) result.duplicate_count);
    if (result.long_prompt_count > 0) fprintf(stderr, "warning: dataset contains %lld prompts longer than %zu bytes\n", (long long) result.long_prompt_count, long_prompt_bytes);
    if (result.statistically_weak) {
        fprintf(stderr, "warning: statistically weak dataset (%lld target, %lld control); minimum recommended for analysis is %lld each, with 100-500 preferred\n",
                (long long) result.target_count, (long long) result.control_count, (long long) std::max(minimum_target, minimum_control));
    }
    return result;
}

std::pair<std::vector<heretic_dataset_entry>, std::vector<heretic_dataset_entry>> heretic_stratified_split(
        const std::vector<heretic_dataset_entry> & dataset,
        double eval_fraction,
        uint32_t seed) {
    if (eval_fraction <= 0.0 || eval_fraction >= 1.0) throw std::runtime_error("--eval-split must be greater than 0 and less than 1");
    std::vector<heretic_dataset_entry> train;
    std::vector<heretic_dataset_entry> eval;
    std::mt19937 rng(seed);
    for (const char * group : { "target", "control" }) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < dataset.size(); ++i) if (dataset[i].group == group) indices.push_back(i);
        std::shuffle(indices.begin(), indices.end(), rng);
        size_t n_eval = std::max<size_t>(1, (size_t) std::llround(indices.size() * eval_fraction));
        if (n_eval >= indices.size()) n_eval = indices.size() - 1;
        for (size_t i = 0; i < indices.size(); ++i) (i < n_eval ? eval : train).push_back(dataset[indices[i]]);
    }
    std::shuffle(train.begin(), train.end(), rng);
    std::shuffle(eval.begin(), eval.end(), rng);
    return { std::move(train), std::move(eval) };
}

