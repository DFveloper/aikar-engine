#include "moe-prune.h"
#include "dataset.h"
#include "hard-prune.h"

#include "chat.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <fstream>
#include <filesystem>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

static void make_fixture(const std::string & path) {
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "gemma4");
    gguf_set_val_u32(gguf, "gemma4.block_count", 30);
    gguf_set_val_u32(gguf, "gemma4.expert_count", 8);
    gguf_set_val_u32(gguf, "gemma4.expert_used_count", 2);

    ggml_context * tensors = ggml_init({ 1024 * 1024, nullptr, false });
    ggml_tensor * router = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 32, 8);
    ggml_set_name(router, "blk.0.ffn_gate_inp.weight");
    gguf_add_tensor(gguf, router);
    ggml_tensor * gate_up = ggml_new_tensor_3d(tensors, GGML_TYPE_Q4_0, 32, 4, 8);
    ggml_set_name(gate_up, "blk.0.ffn_gate_up_exps.weight");
    gguf_add_tensor(gguf, gate_up);
    ggml_tensor * down = ggml_new_tensor_3d(tensors, GGML_TYPE_Q4_0, 32, 32, 8);
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    gguf_add_tensor(gguf, down);

    for (int expert = 0; expert < 8; ++expert) {
        memset((char *) router->data + expert * ggml_row_size(router->type, router->ne[0]), expert, ggml_row_size(router->type, router->ne[0]));
        memset((char *) gate_up->data + expert * ggml_row_size(gate_up->type, gate_up->ne[0]) * gate_up->ne[1], expert, ggml_row_size(gate_up->type, gate_up->ne[0]) * gate_up->ne[1]);
        memset((char *) down->data + expert * ggml_row_size(down->type, down->ne[0]) * down->ne[1], expert, ggml_row_size(down->type, down->ne[0]) * down->ne[1]);
    }
    if (!gguf_write_to_file(gguf, path.c_str(), false)) throw std::runtime_error("failed to write synthetic GGUF fixture");
    ggml_free(tensors);
    gguf_free(gguf);
}

static void expect_failure(const std::function<void()> & fn) {
    bool failed = false;
    try {
        fn();
    } catch (const std::exception &) {
        failed = true;
    }
    if (!failed) throw std::runtime_error("expected failure");
}

static void require(bool value) {
    if (!value) throw std::runtime_error("test assertion failed");
}

static void test_dataset() {
    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(AIKAR_TEST_GEMMA_VOCAB, params);
    require(model != nullptr);
    common_chat_templates_ptr templates = common_chat_templates_init(model, "");

    const std::string path = "/tmp/aikar-moe-prune-test-data.jsonl";
    {
        FILE * file = fopen(path.c_str(), "wb");
        require(file != nullptr);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Question\"},{\"role\":\"assistant\",\"content\":\"Answer\"}]}\n", file);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Solve\"},{\"role\":\"assistant\",\"reasoning\":\"Work\",\"content\":\"Final\"}]}\n", file);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"Q1\"},{\"role\":\"assistant\",\"content\":\"A1\"},{\"role\":\"user\",\"content\":\"Q2\"},{\"role\":\"assistant\",\"content\":\"A2\"},{\"role\":\"user\",\"content\":\"Q3\"},{\"role\":\"assistant\",\"content\":\"A3\"},{\"role\":\"user\",\"content\":\"Q4\"},{\"role\":\"assistant\",\"content\":\"A4\"},{\"role\":\"user\",\"content\":\"Q5\"},{\"role\":\"assistant\",\"content\":\"A5\"}]}\n", file);
        fclose(file);
    }
    const aikar_dataset dataset = aikar_dataset_load(path, model, templates.get(), 1);
    const aikar_dataset parallel_dataset = aikar_dataset_load(path, model, templates.get(), 4);
    require(dataset.records.size() == 3);
    require(parallel_dataset.records.size() == dataset.records.size());
    require(parallel_dataset.total_tokens == dataset.total_tokens);
    require(aikar_dataset_fingerprint(dataset) == aikar_dataset_fingerprint(parallel_dataset));
    auto changed_tokens = dataset;
    ++changed_tokens.records[0].tokens[0];
    require(aikar_dataset_fingerprint(dataset) != aikar_dataset_fingerprint(changed_tokens));
    changed_tokens = dataset;
    changed_tokens.records[0].token_fields[0] ^= 1;
    require(aikar_dataset_fingerprint(dataset) != aikar_dataset_fingerprint(changed_tokens));
    changed_tokens = dataset;
    std::swap(changed_tokens.records[0], changed_tokens.records[1]);
    require(aikar_dataset_fingerprint(dataset) != aikar_dataset_fingerprint(changed_tokens));
    for (size_t i = 0; i < dataset.records.size(); ++i) {
        require(parallel_dataset.records[i].line == dataset.records[i].line);
        require(parallel_dataset.records[i].tokens == dataset.records[i].tokens);
        require(parallel_dataset.records[i].token_fields == dataset.records[i].token_fields);
    }
    size_t assistant = 0;
    size_t reasoning = 0;
    size_t content = 0;
    for (const aikar_dataset_record & record : dataset.records) {
        for (size_t i = 0; i < record.tokens.size(); ++i) {
            assistant += aikar_token_is_evaluated(record, i, aikar_ppl_mask::ASSISTANT);
            reasoning += aikar_token_is_evaluated(record, i, aikar_ppl_mask::REASONING);
            content += aikar_token_is_evaluated(record, i, aikar_ppl_mask::CONTENT);
        }
    }
    require(assistant > 0 && reasoning > 0 && content > 0);

    {
        FILE * file = fopen(path.c_str(), "wb");
        require(file != nullptr);
        fputs("{\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}\n", file);
        fputs("{\"messages\":42}\n", file);
        fclose(file);
    }
    bool line_error = false;
    try {
        (void) aikar_dataset_load(path, model, templates.get());
    } catch (const std::exception & e) {
        line_error = std::string(e.what()).find("line 2") != std::string::npos;
    }
    require(line_error);
    std::remove(path.c_str());
    llama_model_free(model);
}

static void test_metric_ranking() {
    common_moe_prune_model_info model;
    model.expert_count = 5;
    model.experts_used = 2;
    model.moe_layers = { 0 };
    common_moe_prune_stats stats;
    stats[0] = {
        { 100, 50.0, 100.0, 100.0 },
        {   1,  0.5,  20.0,  10.0 },
        {  10,  5.0,  40.0,  20.0 },
        {   0,  0.0,   0.0,   0.0 },
        {   1,  0.5,   6.0,   3.0 },
    };
    const auto profiles = common_moe_prune_make_profiles(
        model, stats, { 0.4 }, 0.5, "dataset", "all", "frequency", 100);
    require(profiles[0].layers.at(0).disabled_experts == std::vector<int32_t>({ 1, 3 }));
    const auto legacy = common_moe_prune_make_profiles(
        model, stats, { 0.4 }, 0.5, "dataset", "all", "router-output", 100);
    require(legacy[0].layers.at(0).disabled_experts == std::vector<int32_t>({ 0, 3 }));
    const double reap_sums[] = { 400.0, 20.0, 20.0, 0.0, 3.0 };
    for (size_t i = 0; i < stats[0].size(); ++i) {
        stats[0][i].reap_count = stats[0][i].selection_count;
        stats[0][i].reap_selection_count = stats[0][i].selection_count;
        stats[0][i].reap_sum = reap_sums[i];
    }
    const auto reap = common_moe_prune_make_profiles(
        model, stats, { 0.4 }, 0.5, "dataset", "all", "reap", 100);
    require(reap[0].metric == "reap");
    require(reap[0].layers.at(0).disabled_experts == std::vector<int32_t>({ 2, 3 }));
    const auto unchanged = common_moe_prune_make_profiles(
        model, stats, { 0.4 }, 0.5, "dataset", "all", "router-output", 100);
    require(unchanged[0].layers.at(0).disabled_experts == std::vector<int32_t>({ 0, 3 }));
    --stats[0][0].reap_count;
    expect_failure([&]() { common_moe_prune_make_profiles(model, stats, { 0.4 }, 0.5, "", "all", "reap", 100); });
    expect_failure([&]() { common_moe_prune_make_profiles(model, stats, { 0.4 }, 0.5, "", "all", "unknown", 100); });
}

static void test_reap_collection() {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    const std::vector<int32_t> ids = { 0, 1, 0, 2 };
    const std::vector<float> weights = { 0.25f, 0.75f, 0.75f, 0.25f };
    const float values[] = { 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.0f, 2.0f };
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        ggml_tensor * output = ggml_new_tensor_3d(ctx, type, 2, 2, 2);
        for (size_t i = 0; i < 8; ++i) {
            if (type == GGML_TYPE_F32) ((float *) output->data)[i] = values[i];
            else if (type == GGML_TYPE_F16) ((ggml_fp16_t *) output->data)[i] = ggml_fp32_to_fp16(values[i]);
            else ((ggml_bf16_t *) output->data)[i] = ggml_fp32_to_bf16(values[i]);
        }
        std::vector<common_moe_prune_expert_stats> stats(4);
        stats[0].selection_count = 2;
        stats[0].reap_selection_count = 2;
        stats[0].probability_sum = 1.0;
        stats[1].selection_count = stats[2].selection_count = 1;
        stats[1].reap_selection_count = stats[2].reap_selection_count = 1;
        common_moe_prune_collect_output(output, ids, weights, stats, true);
        require(stats[0].reap_count == 2);
        require(stats[0].reap_sum == 8.75);
        require(stats[0].reap_score() == 4.375);
        require(stats[0].mean_probability() == 0.5);
        require(stats[0].mean_reap_output_norm() == 7.5);
        require(stats[0].importance() == 4.375);
        require(stats[1].reap_count == 1 && stats[1].reap_score() == 0.0);
        require(stats[2].reap_score() == 0.5);
        require(stats[3].reap_count == 0 && stats[3].reap_score() == 0.0);
        require(stats[3].mean_reap_output_norm() == 0.0);
        // REAP-only collection leaves the aikar score intact.
        const double legacy = stats[0].importance();
        common_moe_prune_collect_output(output, ids, weights, stats, false);
        require(stats[0].importance() == legacy && stats[0].reap_score() == 4.375);
        expect_failure([&]() { common_moe_prune_collect_output(output, { 0 }, weights, stats, false); });
        expect_failure([&]() { common_moe_prune_collect_output(output, { 0, 1, 0, 4 }, weights, stats, false); });
        expect_failure([&]() { common_moe_prune_collect_output(output, ids, { -0.25f, 0.75f, 0.75f, 0.25f }, stats, false); });
    }
    ggml_tensor * wide = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 5000, 1, 1);
    std::fill_n((float *) wide->data, 5000, 1.0f);
    std::vector<common_moe_prune_expert_stats> wide_stats(1);
    common_moe_prune_collect_output(wide, { 0 }, { 0.5f }, wide_stats, false);
    require(std::abs(wide_stats[0].reap_score() - 35.35533905932738) < 1e-12);
    wide_stats[0] = {};
    common_moe_prune_collect_output(wide, { 0 }, { 0.0f }, wide_stats, false);
    require(wide_stats[0].reap_count == 1 && wide_stats[0].reap_score() == 0.0);
    ggml_tensor * extremes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 1, 1);
    std::vector<common_moe_prune_expert_stats> stats(1);
    ((float *) extremes->data)[0] = ((float *) extremes->data)[1] = 1e38f;
    common_moe_prune_collect_output(extremes, { 0 }, { 1.0f }, stats, false);
    require(std::isfinite(stats[0].reap_score()) && stats[0].reap_score() > 1e38);
    stats[0] = {};
    ((float *) extremes->data)[0] = ((float *) extremes->data)[1] = 1e-38f;
    common_moe_prune_collect_output(extremes, { 0 }, { 1.0f }, stats, false);
    require(stats[0].reap_score() > 0.0 && stats[0].reap_score() < 2e-38);
    ((float *) extremes->data)[0] = std::numeric_limits<float>::infinity();
    expect_failure([&]() { common_moe_prune_collect_output(extremes, { 0 }, { 0.0f }, stats, false); });
    ggml_free(ctx);
}

static void test_topk_normalized_gate() {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    ggml_tensor * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
    const float values[] = { std::log(0.6f), std::log(0.3f), std::log(0.1f) };
    std::memcpy(logits->data, values, sizeof(values));
    ggml_tensor * probabilities = ggml_soft_max(ctx, logits);
    ggml_tensor * ids = ggml_argsort_top_k(ctx, probabilities, 2);
    ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probabilities, 1, 3, 1), ids);
    weights = ggml_reshape_2d(ctx, weights, 2, 1);
    weights = ggml_div(ctx, weights, ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f, INFINITY));
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, weights);
    require(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS);
    require(common_moe_prune_selected_ids(ids) == std::vector<int32_t>({ 0, 1 }));
    const float * gate = static_cast<const float *>(weights->data);
    require(std::abs(gate[0] - 2.0f / 3.0f) < 1e-6f);
    require(std::abs(gate[1] - 1.0f / 3.0f) < 1e-6f);
    ggml_free(ctx);
}

static void test_topk_strided_ids() {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    ggml_tensor * sorted = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    const int32_t values[] = { 0, 1, 2, 3, 3, 2, 1, 0 };
    std::memcpy(sorted->data, values, sizeof(values));
    ggml_tensor * topk = ggml_view_2d(ctx, sorted, 2, 2, sorted->nb[1], 0);
    require(common_moe_prune_selected_ids(topk) == std::vector<int32_t>({ 0, 1, 3, 2 }));
    require(common_moe_prune_selected_ids(sorted) == std::vector<int32_t>(values, values + 8));
    ggml_free(ctx);
}

static void test_reap_hand_calculation() {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, false });
    ggml_tensor * output = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, 3);
    const float norms[] = { 2.0f, 4.0f, 1.0f };
    std::memcpy(output->data, norms, sizeof(norms));
    std::vector<common_moe_prune_expert_stats> stats(3);
    common_moe_prune_collect_output(output, { 0, 0, 1 }, { 0.8f, 0.4f, 0.5f }, stats, false);
    require(std::abs(stats[0].reap_score() - 1.6) < 1e-7);
    require(stats[1].reap_score() == 0.5 && stats[2].reap_score() == 0.0);
    require(stats[0].reap_score() > stats[1].reap_score());
    require(std::abs(stats[0].reap_sum / 3.0 - 1.0666666666666667) < 1e-7);
    require(std::abs(stats[0].reap_score() - 0.96) > 0.5);
    // Conditional and global contributions can rank a specialist differently.
    stats[0].reap_count = 1;
    stats[0].reap_sum = 3.2;
    stats[1].reap_count = 10;
    stats[1].reap_sum = 16.0;
    require(stats[0].reap_score() > stats[1].reap_score());
    require(stats[0].reap_sum / 11.0 < stats[1].reap_sum / 11.0);
    ggml_free(ctx);
}

static void test_pruning_publication() {
    const std::string output = "/tmp/aikar-prune-publication-test.gguf";
    const std::string stage = output + ".validation.tmp";
    auto write = [](const std::string & path, const std::string & value) { std::ofstream out(path); out << value; };
    auto read = [](const std::string & path) { std::ifstream in(path); return std::string(std::istreambuf_iterator<char>(in), {}); };
    write(output, "old model");
    write(stage, "new model");
    std::filesystem::create_directory(output + ".report.json");
    write(stage + ".report.json", "new report");
    expect_failure([&]() { aikar_hard_prune_publish(stage, output); });
    require(read(output) == "old model" && read(stage) == "new model");
    std::filesystem::remove(output + ".report.json");
    write(output + ".report.json", "old report");
    aikar_hard_prune_publish(stage, output);
    require(read(output) == "new model" && read(output + ".report.json") == "new report");
    require(!std::filesystem::exists(stage + ".previous-model"));
    require(!std::filesystem::exists(stage + ".previous-report"));
    std::remove(output.c_str());
    std::remove((output + ".report.json").c_str());
}

static void test_reap_routing_views() {
    common_moe_prune_expert_stats stat;
    stat.record_selection(0.25, true);
    stat.record_selection(0.25, false);
    stat.record_selection(0.75, true);
    stat.record_selection(0.75, false);
    require(stat.selection_count == 2 && stat.reap_selection_count == 2);
    require(stat.probability_sum == 1.0 && stat.mean_probability() == 0.5);
    stat.weighted_output_sum = 8.75;
    require(stat.importance() == 4.375);
    stat.reap_count = 2;
    stat.reap_sum = 8.75;
    require(stat.reap_score() == 4.375);
    auto overflow = stat;
    overflow.selection_count = UINT64_MAX;
    expect_failure([&]() { overflow.record_selection(0.5, true); });
    require(overflow.selection_count == UINT64_MAX);
    expect_failure([&]() { stat.record_selection(INFINITY, true); });
    expect_failure([&]() { stat.record_selection(-0.5, true); });
    common_moe_prune_model_info model;
    model.expert_count = 2;
    model.experts_used = 1;
    model.moe_layers = { 0 };
    common_moe_prune_stats stats = { { 0, { {}, stat } } };
    const auto profiles = common_moe_prune_make_profiles(model, stats, { 0.5 }, 0.5, "", "all", "reap", 2);
    require(profiles[0].layers.at(0).disabled_experts == std::vector<int32_t>({ 0 }));
}

int main() {
    test_metric_ranking();
    test_reap_collection();
    test_reap_routing_views();
    test_pruning_publication();
    test_reap_hand_calculation();
    test_topk_strided_ids();
    test_topk_normalized_gate();
    test_dataset();
    common_moe_prune_model_info model;
    model.architecture = "gemma4";
    model.model_hash = "sha256:model";
    model.expert_tensor_hash = "sha256:experts";
    model.layer_count = 30;
    model.expert_count = 8;
    model.experts_used = 2;
    model.moe_layers = { 1, 3 };

    common_moe_prune_stats stats;
    for (int32_t layer : model.moe_layers) {
        stats[layer].resize(model.expert_count);
        for (int32_t expert = 0; expert < model.expert_count; ++expert) {
            auto & value = stats[layer][expert];
            value.selection_count = 10;
            value.probability_sum = expert + 1;
            value.output_norm_sum = 2 * (expert + 1);
            value.weighted_output_sum = expert + layer;
        }
    }

    const auto profiles = common_moe_prune_make_profiles(
        model, stats, { 0.25, 0.50 }, 0.50, "sha256:dataset", "assistant", "router-output", 100);
    require(profiles.size() == 2);
    for (int32_t layer : model.moe_layers) {
        const auto & small = profiles[0].layers.at(layer).disabled_experts;
        const auto & large = profiles[1].layers.at(layer).disabled_experts;
        require(small.size() == 2);
        require(large.size() == 4);
        for (int32_t expert : small) require(std::find(large.begin(), large.end(), expert) != large.end());
    }

    const std::string path = "/tmp/aikar-moe-prune-test-profile.json";
    auto provenance_profile = profiles[0];
    provenance_profile.calibration_seed = 42;
    provenance_profile.calibration_context = 4096;
    provenance_profile.calibration_batch = 512;
    provenance_profile.calibration_ubatch = 128;
    provenance_profile.calibration_tokens = 1000;
    provenance_profile.calibration_collector_version = 3;
    provenance_profile.calibration_tokenized_hash = "sha256:tokens";
    provenance_profile.calibration_execution_hash = "sha256:execution";
    provenance_profile.calibration_fingerprint = "sha256:calibration";
    common_moe_prune_profile_write(provenance_profile, path);
    const common_moe_prune_profile loaded = common_moe_prune_profile_load(path);
    common_moe_prune_profile_validate(loaded, model);
    require(loaded.calibration_seed == 42 && loaded.calibration_context == 4096);
    require(loaded.calibration_batch == 512 && loaded.calibration_ubatch == 128);
    require(loaded.calibration_tokens == 1000);
    require(loaded.calibration_collector_version == 3);
    require(loaded.calibration_tokenized_hash == "sha256:tokens");
    require(loaded.calibration_execution_hash == "sha256:execution");
    require(loaded.calibration_fingerprint == "sha256:calibration");
    require(loaded.layers.at(1).disabled_experts == profiles[0].layers.at(1).disabled_experts);
    std::remove(path.c_str());

    common_moe_prune_profile stale = loaded;
    stale.calibration_collector_version = 2;
    expect_failure([&]() { common_moe_prune_profile_validate(stale, model); });
    common_moe_prune_profile invalid = loaded;
    invalid.model_hash = "sha256:wrong";
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 0 };
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 8 };
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });
    invalid = loaded;
    invalid.layers.at(1).disabled_experts = { 0, 1, 2, 3, 4, 5, 6 };
    invalid.layers.at(3).disabled_experts = invalid.layers.at(1).disabled_experts;
    expect_failure([&]() { common_moe_prune_profile_validate(invalid, model); });

    require(std::abs(stats[1][2].mean_probability() - 0.3) < 1e-12);
    require(std::abs(stats[1][2].mean_output_norm() - 0.6) < 1e-12);
    require(std::abs(stats[1][2].importance() - 0.3) < 1e-12);

    const std::string source_path = "/tmp/aikar-moe-prune-test-source.gguf";
    const std::string output_path = "/tmp/aikar-moe-prune-test-output.gguf";
    const std::string cache_path = "/tmp/aikar-moe-prune-test-model-cache.json";
    std::remove(cache_path.c_str());
    make_fixture(source_path);
    const common_moe_prune_model_info fixture_info = common_moe_prune_inspect_model(source_path);
    require(fixture_info.model_hash == common_moe_prune_sha256_file(source_path));
    bool cache_hit = true;
    const common_moe_prune_model_info first = common_moe_prune_inspect_model_cached(source_path, cache_path, &cache_hit);
    require(!cache_hit && first.model_hash == fixture_info.model_hash);
    const common_moe_prune_model_info second = common_moe_prune_inspect_model_cached(source_path, cache_path, &cache_hit);
    require(cache_hit && second.model_hash == first.model_hash);
    {
        FILE * file = fopen(source_path.c_str(), "ab");
        require(file != nullptr);
        require(fputc(0, file) != EOF);
        fclose(file);
    }
    const common_moe_prune_model_info changed = common_moe_prune_inspect_model_cached(source_path, cache_path, &cache_hit);
    require(!cache_hit && changed.model_hash != first.model_hash);
    std::remove(cache_path.c_str());
    make_fixture(source_path);
    require(common_moe_prune_inspect_model(source_path).model_hash == fixture_info.model_hash);
    common_moe_prune_profile fixture_profile;
    fixture_profile.architecture = fixture_info.architecture;
    fixture_profile.model_hash = fixture_info.model_hash;
    fixture_profile.expert_tensor_hash = fixture_info.expert_tensor_hash;
    fixture_profile.expert_count = fixture_info.expert_count;
    fixture_profile.experts_used = fixture_info.experts_used;
    fixture_profile.layers[0].disabled_experts = { 1, 3 };
    const std::string alias_path = source_path + ".alias";
    std::remove(alias_path.c_str());
    for (const std::string & suffix : { std::string(""), std::string(".tmp"), std::string(".report.json"), std::string(".report.json.tmp"),
            std::string(".tmp.report.json"), std::string(".tmp.report.json.tmp"), std::string(".tmp.previous-model"), std::string(".tmp.previous-report") }) {
        std::remove((alias_path + suffix).c_str());
        std::filesystem::create_hard_link(source_path, alias_path + suffix);
        expect_failure([&]() { aikar_hard_prune_gemma4_q4_0(source_path, fixture_profile, fixture_info, alias_path); });
        require(common_moe_prune_sha256_file(source_path) == fixture_info.model_hash);
        std::remove((alias_path + suffix).c_str());
    }
    {
        std::ofstream prior(output_path);
        prior << "old model";
    }
    std::filesystem::create_directory(output_path + ".report.json");
    expect_failure([&]() { aikar_hard_prune_gemma4_q4_0(source_path, fixture_profile, fixture_info, output_path); });
    {
        std::ifstream prior(output_path);
        require(std::string(std::istreambuf_iterator<char>(prior), {}) == "old model");
    }
    require(!std::filesystem::exists(output_path + ".tmp"));
    std::filesystem::remove(output_path + ".report.json");
    const aikar_hard_prune_report report = aikar_hard_prune_gemma4_q4_0(source_path, fixture_profile, fixture_info, output_path);
    require(report.original_to_new.at(0).at(0) == 0);
    require(report.original_to_new.at(0).at(2) == 1);
    require(report.expert_bytes_removed > 0);
    {
        std::ifstream in(output_path + ".report.json");
        nlohmann::ordered_json provenance;
        in >> provenance;
        require(provenance.at("model_path") == source_path);
        require(provenance.at("model_sha256") == fixture_info.model_hash);
        require(provenance.at("model_architecture") == "gemma4");
        require(provenance.at("metric") == fixture_profile.metric);
        require(provenance.at("calibration_fingerprint").at("dataset_hash") == fixture_profile.dataset_hash);
    }

    ggml_context * pruned_tensors = nullptr;
    gguf_context * pruned = gguf_init_from_file(output_path.c_str(), { false, &pruned_tensors });
    require(pruned != nullptr);
    const int64_t expert_key = gguf_find_key(pruned, "gemma4.expert_count");
    require(expert_key >= 0 && gguf_get_val_u32(pruned, expert_key) == 6);
    const int64_t router_id = gguf_find_tensor(pruned, "blk.0.ffn_gate_inp.weight");
    const int64_t expert_id = gguf_find_tensor(pruned, "blk.0.ffn_gate_up_exps.weight");
    require(router_id >= 0 && gguf_get_tensor_ne(pruned, router_id)[1] == 6);
    require(expert_id >= 0 && gguf_get_tensor_ne(pruned, expert_id)[2] == 6);
    const int expected[] = { 0, 2, 4, 5, 6, 7 };
    const ggml_tensor * pruned_router = ggml_get_tensor(pruned_tensors, "blk.0.ffn_gate_inp.weight");
    const ggml_tensor * pruned_experts = ggml_get_tensor(pruned_tensors, "blk.0.ffn_gate_up_exps.weight");
    for (int expert = 0; expert < 6; ++expert) {
        require(*((const unsigned char *) pruned_router->data + expert * ggml_row_size(pruned_router->type, pruned_router->ne[0])) == expected[expert]);
        require(*((const unsigned char *) pruned_experts->data + expert * ggml_row_size(pruned_experts->type, pruned_experts->ne[0]) * pruned_experts->ne[1]) == expected[expert]);
    }
    ggml_free(pruned_tensors);
    gguf_free(pruned);
    std::remove(source_path.c_str());
    std::remove(output_path.c_str());
    std::remove((output_path + ".report.json").c_str());
    std::remove(cache_path.c_str());
    return 0;
}
