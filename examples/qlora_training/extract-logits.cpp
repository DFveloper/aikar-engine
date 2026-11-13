#define LLAMA_LOGIT_EXTRACT_SHARED_ONLY
#include "finetune_qat.cpp"
#include <future>

int main(int argc, char ** argv) {
    try {
        common_params params;
        params.escape = false;
        params.warmup = false;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QAT)) return 1;
        logit_distillation::require(!params.distill_cache.empty() && !params.distill_student.empty() && !params.train_file.empty(),
                "--distill-cache, --distill-student and --train-file are required");
        logit_distillation::require(params.n_ctx > 0 && params.n_batch == params.n_ctx && params.n_ubatch > 0 &&
                params.n_parallel == 1 && params.critical_token_mode == "none" && params.mtp_mode == "off",
                "extraction requires -c N -b N, -np 1, no critical weighting or MTP");
        logit_distillation::require(params.distill_plan || !std::filesystem::exists(params.distill_cache), "output cache already exists");
        const bool resume = std::filesystem::exists(params.distill_cache + ".partial");
        logit_distillation::require(params.distill_plan || !resume || params.distill_resume_extraction, "partial cache exists; use --distill-resume-extraction");
        common_init();
        llama_backend_init();
        llama_numa_init(params.numa);
        auto model_params = llama_model_default_params();
        model_params.vocab_only = true;
        llama_model_ptr student(llama_model_load_from_file(params.distill_student.c_str(), model_params));
        logit_distillation::require(bool(student), "cannot load student vocabulary");
        const auto * vocab = llama_model_get_vocab(student.get());
        auto templates = common_chat_templates_init(student.get(), params.chat_template);
        auto samples = load_jsonl(params.train_file, vocab, templates.get(), params.dataset_threads,
                "none", 1.0f, params.preserve_thinking ? 1 : 0, 0);
        int64_t train_conversations, val_conversations;
        const int64_t sample_split = train_sample_split_by_conversation(samples, params.val_split,
                train_conversations, val_conversations);
        std::vector<float> rewards;
        int64_t split = 0;
        ggml_opt_dataset_t dataset = build_dataset(samples, params.n_ctx, rewards, sample_split, split,
                params.train_on_prompt, llama_vocab_bos(vocab));
        logit_distillation::require(dataset != nullptr, "cannot pack training data");
        distillation_limit_dataset(dataset, split, params.distill_max_windows, rewards, params.distill_validation_windows);
        logit_distillation::header info;
        info.context = params.n_ctx;
        info.vocab = llama_vocab_n_tokens(vocab);
        info.top_k = params.distill_top_k;
        info.temperature = params.distill_temperature;
        info.tokenizer = logit_distillation::vocab_hash(vocab);
        info.windows = ggml_opt_dataset_ndata(dataset);
        info.train_windows = split;
        info.metadata = nlohmann::json({{"teacher", params.model.path}, {"student", params.distill_student},
                {"data", params.train_file}, {"train_on_prompt", params.train_on_prompt},
                {"preserve_thinking", params.preserve_thinking}, {"val_split", params.val_split}}).dump();
        logit_distillation::require(info.top_k < info.vocab, "top-k must be smaller than vocabulary");
        if (params.distill_plan) {
            const auto * targets = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
            const uint64_t train_rows = std::count_if(targets, targets + split*info.context, [](int32_t t) { return t >= 0; });
            const uint64_t rows = std::count_if(targets, targets + info.windows*info.context, [](int32_t t) { return t >= 0; });
            std::ostringstream encoded(std::ios::binary);
            info.save(encoded);
            const uint64_t bytes = encoded.str().size() + info.windows*info.context*8 + rows*info.row_bytes();
            const auto plan = nlohmann::json({{"context", info.context}, {"train_windows", split},
                    {"validation_windows", info.windows - split}, {"train_rows", train_rows},
                    {"validation_rows", rows - train_rows}, {"cache_bytes", bytes}});
            std::printf("%s\n", plan.dump().c_str());
            ggml_opt_dataset_free(dataset);
            return 0;
        }
        auto teacher = common_init_from_params(params);
        logit_distillation::require(teacher->model() && teacher->context(), "cannot load teacher");
        logit_distillation::require(llama_n_ctx(teacher->context()) == info.context,
                "runtime rounded the context; choose an aligned context such as 256 or 512");
        logit_distillation::require(info.tokenizer == logit_distillation::vocab_hash(llama_model_get_vocab(teacher->model())),
                "teacher and student token IDs differ");
        auto * tokens = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
        auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
        uint64_t rows = 0;
        const uint64_t first_window = resume ? logit_distillation::resume_cache(
                params.distill_cache + ".partial", info, tokens, labels, rows) : 0;
        std::ofstream out(params.distill_cache + ".partial", std::ios::binary | (resume ? std::ios::app : std::ios::trunc));
        if (!resume) info.save(out);
        common_batch batch(teacher->context());
        const uint64_t resumed_rows = rows;
        LOG_INF("distill_extract: resumed_windows=%llu resumed_rows=%llu\n", (unsigned long long) first_window, (unsigned long long) rows);
        double tail_sum = 0;
        double tail_max = 0;
        const char * gpu_encode_env = std::getenv("GGML_DISTILL_GPU_ENCODE");
        const bool gpu_encode = gpu_encode_env && std::strcmp(gpu_encode_env, "0") != 0;
        LOG_INF("distill_extract: encoder=%s\n", gpu_encode ? "device-top-k" : "cpu-overlap");
        std::vector<int32_t> compact_ids;
        std::vector<double> compact_stats;
        std::vector<int32_t> staged_ids;
        std::vector<double> staged_stats;
        std::vector<float> staged_logits;
        std::future<std::vector<double>> pending;
        uint64_t pending_window = info.windows;
        const int64_t start_ms = ggml_time_ms();
        const auto log_window = [&](uint64_t w) {
            LOG_INF("distill_extract: window=%llu/%llu rows=%llu mean_tail=%.8g max_tail=%.8g elapsed_s=%.3f\n",
                    (unsigned long long) (w + 1), (unsigned long long) info.windows, (unsigned long long) rows,
                    rows > resumed_rows ? tail_sum / (rows - resumed_rows) : 0, tail_max, (ggml_time_ms() - start_ms)*0.001);
        };
        const auto record_tails = [&](const std::vector<double> & tails) {
            for (const double tail : tails) {
                tail_sum += tail;
                tail_max = std::max(tail_max, tail);
                ++rows;
            }
        };
        const auto finish_encoding = [&]() {
            if (pending.valid()) {
                record_tails(pending.get());
                if (pending_window < info.windows) log_window(pending_window);
                pending_window = info.windows;
            }
        };
        for (uint64_t w = first_window; w < info.windows; ++w) {
            const size_t base = size_t(w)*info.context;
            const auto write_window = [&]() {
                logit_distillation::write(out, tokens + base, size_t(info.context)*4);
                logit_distillation::write(out, labels + base, size_t(info.context)*4);
            };
            uint32_t prediction_end = info.context;
            while (prediction_end > 0 && labels[base + prediction_end - 1] < 0) --prediction_end;
            // Skip unused trailing microbatches, but keep batch shapes for identical quantized results.
            if (prediction_end == 0) {
                finish_encoding();
                write_window();
                log_window(w);
                continue;
            }
            llama_memory_clear(llama_get_memory(teacher->context()), true);
            for (uint32_t begin = 0; begin < prediction_end; begin += params.n_ubatch) {
                batch.clear();
                const uint32_t end = std::min<uint32_t>(info.context, begin + params.n_ubatch);
                for (uint32_t pos = begin; pos < end; ++pos) {
                    batch.add(tokens[base + pos], pos, 0, labels[base + pos] >= 0);
                }
                if (gpu_encode) {
                    size_t count = 0;
                    for (uint32_t pos = begin; pos < end; ++pos) count += labels[base + pos] >= 0;
                    logit_distillation::require(count <= SIZE_MAX/(size_t(info.top_k) + 3)/sizeof(double), "compact batch is too large");
                    compact_ids.resize(count*info.top_k);
                    compact_stats.resize(count*(info.top_k + 3));
                    logit_distillation::require(llama_decode_top_k(teacher->context(), batch.get(), info.top_k, info.temperature,
                            compact_ids.data(), compact_stats.data(), count) == 0, "compact teacher decode failed");
                    finish_encoding();
                    if (begin == 0) write_window();
                    if (end < prediction_end || w + 1 < info.windows) {
                        staged_ids = compact_ids;
                        staged_stats = compact_stats;
                        if (end >= prediction_end) pending_window = w;
                        pending = std::async(std::launch::async, [&, count]() {
                            return logit_distillation::encode_compact_rows(out, staged_ids.data(), staged_stats.data(), count, info);
                        });
                    } else {
                        record_tails(logit_distillation::encode_compact_rows(out, compact_ids.data(), compact_stats.data(), count, info));
                        log_window(w);
                    }
                    continue;
                }
                logit_distillation::require(llama_process(teacher->context(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "teacher decode failed");
                finish_encoding();
                // Decode may overlap the previous window's encoding; keep cache writes in window order.
                if (begin == 0) write_window();
                size_t output_count = 0;
                for (uint32_t pos = begin; pos < end; ++pos) output_count += labels[base + pos] >= 0;
                const bool overlap = end < prediction_end || w + 1 < info.windows;
                logit_distillation::require(output_count <= SIZE_MAX / sizeof(float) / info.vocab, "staged logits are too large");
                if (overlap) staged_logits.resize(output_count * info.vocab);
                std::vector<const float *> output_rows;
                output_rows.reserve(output_count);
                for (uint32_t pos = begin; pos < end; ++pos) {
                    if (labels[base + pos] < 0) continue;
                    const float * logits = llama_get_logits_ith(teacher->context(), pos - begin);
                    logit_distillation::require(logits != nullptr, "missing teacher logits");
                    if (overlap) {
                        float * staged = staged_logits.data() + output_rows.size()*info.vocab;
                        std::memcpy(staged, logits, size_t(info.vocab)*sizeof(float));
                        logits = staged;
                    }
                    output_rows.push_back(logits);
                }
                if (overlap) {
                    if (end >= prediction_end) pending_window = w;
                    pending = std::async(std::launch::async, [&, output_rows = std::move(output_rows)]() {
                        return logit_distillation::encode_rows(out, output_rows, info, params.cpuparams_batch.n_threads);
                    });
                } else {
                    record_tails(logit_distillation::encode_rows(out, output_rows, info, params.cpuparams_batch.n_threads));
                    log_window(w);
                }
            }
        }
        finish_encoding();
        out.close();
        logit_distillation::require(bool(out), "cache close failed");
        const double elapsed = (ggml_time_ms() - start_ms)*0.001;
        const uint64_t input_tokens = (info.windows - first_window)*info.context;
        LOG_INF("distill_throughput: input_tokens=%llu supervised_tokens=%llu elapsed_s=%.3f input_tok_s=%.3f supervised_tok_s=%.3f\n",
                (unsigned long long) input_tokens, (unsigned long long) (rows - resumed_rows), elapsed,
                elapsed > 0 ? input_tokens/elapsed : 0, elapsed > 0 ? (rows - resumed_rows)/elapsed : 0);
        std::filesystem::rename(params.distill_cache + ".partial", params.distill_cache);
        LOG_INF("distill_extract: saved=%s rows=%llu bytes_per_row=%zu\n", params.distill_cache.c_str(),
                (unsigned long long) rows, info.row_bytes());
        ggml_opt_dataset_free(dataset);
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("%s\n", e.what());
        return 1;
    }
}
