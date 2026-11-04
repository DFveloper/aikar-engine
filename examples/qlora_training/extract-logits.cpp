#define LLAMA_LOGIT_EXTRACT_SHARED_ONLY
#include "finetune_qat.cpp"

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
        logit_distillation::require(!std::filesystem::exists(params.distill_cache) &&
                !std::filesystem::exists(params.distill_cache + ".partial"), "output cache already exists");
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
        distillation_limit_dataset(dataset, split, params.distill_max_windows, rewards);
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
        auto teacher = common_init_from_params(params);
        logit_distillation::require(teacher->model() && teacher->context(), "cannot load teacher");
        logit_distillation::require(llama_n_ctx(teacher->context()) == info.context,
                "runtime rounded the context; choose an aligned context such as 256 or 512");
        logit_distillation::require(info.tokenizer == logit_distillation::vocab_hash(llama_model_get_vocab(teacher->model())),
                "teacher and student token IDs differ");
        std::ofstream out(params.distill_cache + ".partial", std::ios::binary);
        info.save(out);
        auto * tokens = static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data);
        auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
        common_batch batch(teacher->context());
        uint64_t rows = 0;
        double tail_sum = 0;
        double tail_max = 0;
        const int64_t start_ms = ggml_time_ms();
        for (uint64_t w = 0; w < info.windows; ++w) {
            const size_t base = size_t(w)*info.context;
            logit_distillation::write(out, tokens + base, size_t(info.context)*4);
            logit_distillation::write(out, labels + base, size_t(info.context)*4);
            llama_memory_clear(llama_get_memory(teacher->context()), true);
            for (uint32_t begin = 0; begin < info.context; begin += params.n_ubatch) {
                batch.clear();
                const uint32_t end = std::min<uint32_t>(info.context, begin + params.n_ubatch);
                for (uint32_t pos = begin; pos < end; ++pos) {
                    batch.add(tokens[base + pos], pos, 0, labels[base + pos] >= 0);
                }
                logit_distillation::require(llama_process(teacher->context(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "teacher decode failed");
                for (uint32_t pos = begin; pos < end; ++pos) {
                    if (labels[base + pos] < 0) continue;
                    const float * logits = llama_get_logits_ith(teacher->context(), pos - begin);
                    logit_distillation::require(logits != nullptr, "missing teacher logits");
                    const double tail = logit_distillation::encode_row(out, logits, info);
                    tail_sum += tail;
                    tail_max = std::max(tail_max, tail);
                    ++rows;
                }
            }
            LOG_INF("distill_extract: window=%llu/%llu rows=%llu mean_tail=%.8g max_tail=%.8g elapsed_s=%.3f\n",
                    (unsigned long long) (w + 1), (unsigned long long) info.windows, (unsigned long long) rows,
                    rows ? tail_sum / rows : 0, tail_max, (ggml_time_ms() - start_ms)*0.001);
        }
        out.close();
        logit_distillation::require(bool(out), "cache close failed");
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
