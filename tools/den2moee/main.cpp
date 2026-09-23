#include "self-test.h"
#include "gguf_io.h"
#include "converter.h"
#include "config.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        return den2moee_run_self_tests();
    }

    if (argc == 3 && std::strcmp(argv[1], "--inspect") == 0) {
        try {
            den2moee::MappedGguf model(argv[2]);
            den2moee::print_model_inspection(model);
            return 0;
        } catch (const std::exception & error) {
            std::fprintf(stderr, "[den2moee] inspect failed: %s\n", error.what());
            return 1;
        }
    }

    if (argc >= 2 && std::strcmp(argv[1], "--model") == 0) {
        try {
            den2moee::Options options;
            std::string model;
            std::string calibration;
            std::string output;
            bool experts_set = false;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                auto value = [&]() -> std::string {
                    if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
                    return argv[++i];
                };
                if (arg == "--model") model = value();
                else if (arg == "--calibration") calibration = value();
                else if (arg == "--output") output = value();
                else if (arg == "--experts") { options.experts = std::stoi(value()); experts_set = true; }
                else if (arg == "--shared-experts") options.shared_experts = std::stoi(value());
                else if (arg == "--routed-experts") options.routed_experts = std::stoi(value());
                else if (arg == "--null-experts") options.null_experts = std::stoi(value());
                else if (arg == "--top-k") options.top_k = std::stoi(value());
                else if (arg == "--rank-ratio") options.rank_ratio = std::stof(value());
                else if (arg == "--coverage-percentile") options.coverage_percentile = std::stof(value());
                else if (arg == "--coverage-threshold-ratio") options.coverage_threshold_ratio = std::stof(value());
                else if (arg == "--max-seq-len") options.max_seq_len = std::stoi(value());
                else if (arg == "--max-samples") options.max_samples = std::stoull(value());
                else if (arg == "--span-size") options.span_size = std::stoi(value());
                else if (arg == "--rss-ngram") options.rss_ngram = std::stoi(value());
                else if (arg == "--rss-stride") options.rss_stride = std::stoi(value());
                else if (arg == "--perturb-token-id") options.perturb_token_id = std::stoi(value());
                else if (arg == "--token-score") options.token_score = den2moee::parse_token_score_mode(value());
                else if (arg == "--score-activation") options.score_activation = den2moee::parse_score_activation(value());
                else if (arg == "--input-format") options.input_format = den2moee::parse_calibration_input_format(value());
                else if (arg == "--chat-template") options.chat_template = value();
                else if (arg == "--layer-start") options.layer_start = std::stoi(value());
                else if (arg == "--layer-end") options.layer_end = std::stoi(value());
                else if (arg == "--seed") options.seed = std::stoull(value());
                else if (arg == "--vram-limit-mib") options.vram_limit_mib = std::stoull(value());
                else if (arg == "--gpu-layers") {
                    const std::string gpu = value();
                    options.gpu_layers = gpu == "auto" ? -1 : std::stoi(gpu);
                } else if (arg == "--device" || arg == "-dev") options.device = value();
                else if (arg == "--threads") options.n_threads = std::stoi(value());
                else if (arg == "--help" || arg == "-h") {
                    std::fprintf(stderr, "usage: %s --model MODEL --calibration JSONL --output GGUF [options]\n"
                                         "  --input-format auto|raw|chat (default: auto)\n"
                                         "  --chat-template TEMPLATE (default: model GGUF template)\n"
                                         "  --device auto|none|DEV[,DEV...] (default: auto)\n", argv[0]);
                    return 0;
                } else {
                    throw std::invalid_argument("unknown option: " + arg);
                }
            }
            if (model.empty() || calibration.empty() || output.empty()) {
                throw std::invalid_argument("--model, --calibration and --output are required");
            }
            if (experts_set && options.routed_experts == 7 && options.shared_experts == 1) {
                options.routed_experts = options.experts - options.shared_experts;
            }
            den2moee::convert_dense_gemma4(model, calibration, output, options);
            return 0;
        } catch (const std::exception & error) {
            std::fprintf(stderr, "[den2moee] conversion failed: %s\n", error.what());
            return 1;
        }
    }

    std::fprintf(stderr, "usage: %s --self-test | --inspect MODEL | --model MODEL --calibration JSONL --output GGUF [options]\n", argv[0]);
    return 2;
}
