#include "self-test.h"

#include "calibration.h"
#include "clustering.h"
#include "config.h"
#include "quantize.h"
#include "router.h"
#include "scoring.h"
#include "svd.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <vector>

int den2moee_run_self_tests() {
    using namespace den2moee;

    const CalibrationRecord chat_record = parse_calibration_record(
        common_json::parse(R"({"domain":"math","messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]})"),
        CalibrationInputFormat::Auto);
    if (chat_record.domain != "math" || chat_record.messages.size() != 2 ||
        chat_record.messages[0].role != "user" || chat_record.messages[1].content != "answer") {
        std::fprintf(stderr, "[den2moee] FAIL calibration chat parsing\n");
        return 1;
    }

    const CalibrationRecord positive_record = parse_calibration_record(
        common_json::parse(R"({"positive_messages":[{"role":"user","content":"prompt"}]})"),
        CalibrationInputFormat::Auto);
    if (positive_record.messages.size() != 1 || positive_record.messages[0].role != "user") {
        std::fprintf(stderr, "[den2moee] FAIL calibration positive_messages parsing\n");
        return 1;
    }

    const CalibrationRecord instruct_record = parse_calibration_record(
        common_json::parse(R"({"instruction":"solve","input":"2+2","output":"4"})"),
        CalibrationInputFormat::Auto);
    if (instruct_record.messages.size() != 2 || instruct_record.messages[0].content != "solve\n2+2" ||
        instruct_record.messages[1].role != "assistant" || instruct_record.messages[1].content != "4") {
        std::fprintf(stderr, "[den2moee] FAIL calibration instruct parsing\n");
        return 1;
    }

    const CalibrationRecord sharegpt_record = parse_calibration_record(
        common_json::parse(R"({"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"hello"}]})"),
        CalibrationInputFormat::Auto);
    if (sharegpt_record.messages.size() != 2 || sharegpt_record.messages[0].role != "user" ||
        sharegpt_record.messages[1].role != "assistant") {
        std::fprintf(stderr, "[den2moee] FAIL calibration ShareGPT parsing\n");
        return 1;
    }

    const CalibrationRecord raw_record = parse_calibration_record(
        common_json::parse(R"({"text":"raw sample"})"), CalibrationInputFormat::Auto);
    if (!raw_record.messages.empty() || raw_record.text != "raw sample") {
        std::fprintf(stderr, "[den2moee] FAIL calibration raw parsing\n");
        return 1;
    }

    const LayerConfig layer = make_layer_config(1536, 6144, Options{});
    if (layer.logical_rank != 307 || layer.storage_rank != 320) {
        std::fprintf(stderr, "[den2moee] FAIL rank\n");
        return 1;
    }
    if (!parse_device_spec("auto").empty() || device_spec_name(parse_device_spec("auto")) != "auto") {
        std::fprintf(stderr, "[den2moee] FAIL automatic device selection\n");
        return 1;
    }
    const auto cpu_only_devices = parse_device_spec("none");
    if (cpu_only_devices.size() != 1 || cpu_only_devices[0] != nullptr ||
        device_spec_name(cpu_only_devices) != "none") {
        std::fprintf(stderr, "[den2moee] FAIL CPU-only device selection\n");
        return 1;
    }

    const std::vector<float> logits = { 2.0f, 1.0f, 0.0f, 3.0f };
    const RouterSelection selection = select_router(logits, 3, 1, 2);
    if (selection.selected.size() != 2 || selection.selected[0] != 3 || selection.selected[1] != 0 ||
        std::fabs(selection.raw_selected_weights[0] - selection.weights[0]) > 1e-6f ||
        std::fabs(selection.weights[1] - 1.0f) > 1e-6f ||
        std::fabs(selection.weights[0] - selection.softmax[3]) > 1e-6f) {
        std::fprintf(stderr, "[den2moee] FAIL router\n");
        return 1;
    }

    bool rejected = false;
    try {
        make_layer_config(1536, 6145, Options{});
    } catch (const std::exception &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "[den2moee] FAIL divisibility\n");
        return 1;
    }

    const std::vector<std::vector<float>> features = {
        {1.0f, 0.0f}, {0.9f, 0.1f}, {0.0f, 1.0f}, {0.1f, 0.9f}
    };
    const ClusterResult clusters = balanced_cluster(features, 2, 12345);
    if (clusters.neuron_ids.size() != 2 || clusters.neuron_ids[0].size() != 2 ||
        clusters.neuron_ids[1].size() != 2) {
        std::fprintf(stderr, "[den2moee] FAIL clustering\n");
        return 1;
    }
    const ExpertRoles roles = rank_expert_roles(features, clusters, 0.5f, 50.0f, 1);
    if (roles.shared_clusters.size() != 1 || roles.routed_clusters.size() != 1 ||
        roles.coverage_vectors[roles.shared_clusters[0]].size() != 2) {
        std::fprintf(stderr, "[den2moee] FAIL coverage\n");
        return 1;
    }

    const std::vector<float> matrix = { 3.0f, 0.0f, 0.0f, 2.0f };
    const SvdFactor factor = truncated_svd(matrix, 2, 2, 1, 32, 12345);
    if (factor.logical_rank != 1 || factor.storage_rank != 32 || !(factor.relative_error < 0.8f) ||
        std::any_of(factor.sigma.begin() + factor.logical_rank, factor.sigma.end(),
                    [](float value) { return value != 0.0f; })) {
        std::fprintf(stderr, "[den2moee] FAIL SVD rank\n");
        return 1;
    }
    const SvdFactor full = truncated_svd(matrix, 2, 2, 2, 32, 12345);
    if (full.logical_rank != 2 || full.storage_rank != 32 || !(full.relative_error < 1e-4f) ||
        std::any_of(full.sigma.begin() + full.logical_rank, full.sigma.end(),
                    [](float value) { return value != 0.0f; })) {
        std::fprintf(stderr, "[den2moee] FAIL SVD full rank\n");
        return 1;
    }
    bool rejected_nan = false;
    try {
        truncated_svd({ 1.0f, NAN, 0.0f, 1.0f }, 2, 2, 1, 32, 12345);
    } catch (const std::exception &) {
        rejected_nan = true;
    }
    if (!rejected_nan) {
        std::fprintf(stderr, "[den2moee] FAIL SVD NaN\n");
        return 1;
    }

    const auto scs = calculate_scs({ 1.0f, 1.0f, 4.0f, 4.0f }, 2);
    if (scs.size() != 2 || std::fabs(scs[0]) > 1e-5f || std::fabs(scs[1] - 1.0f) > 1e-5f) {
        std::fprintf(stderr, "[den2moee] FAIL SCS\n");
        return 1;
    }
    const std::vector<float> original_phi(8, 1.0f);
    const std::vector<float> perturbed_phi(8, 0.0f);
    const auto scores = calculate_token_scores({ 1.0f, 1.0f, 4.0f, 4.0f }, original_phi, 2,
        { { 2, perturbed_phi } }, 2, 2, TokenScoreMode::ScsRss);
    if (scores.selected_spans.size() != 1 || scores.selected_spans[0] != 1 ||
        scores.rss_by_token[0] != 0.0f || scores.rss_by_token[2] <= 0.0f) {
        std::fprintf(stderr, "[den2moee] FAIL RSS\n");
        return 1;
    }
    const auto activation = weighted_activation_score({ 1.0f, -1.0f }, { 2.0f, 3.0f }, 1, 2,
                                                       { 1.0f }, ScoreActivation::Gemma4Gelu);
    const auto silu_activation = weighted_activation_score({ 1.0f, -1.0f }, { 2.0f, 3.0f }, 1, 2,
                                                            { 1.0f }, ScoreActivation::UpstreamSilu);
    if (activation.size() != 2 || silu_activation.size() != 2 || activation == silu_activation) {
        std::fprintf(stderr, "[den2moee] FAIL score activation\n");
        return 1;
    }

    const std::vector<float> q_values(32, 0.25f);
    const std::vector<uint8_t> q4 = quantize_q4_0(q_values, 1, 32);
    if (q4.size() != q4_0_row_bytes(32)) {
        std::fprintf(stderr, "[den2moee] FAIL Q4_0\n");
        return 1;
    }

    std::fprintf(stderr, "[den2moee] self-test: PASS core\n");
    return 0;
}
