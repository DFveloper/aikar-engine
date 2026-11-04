#define LLAMA_FINETUNE_QLORA_SHARED_ONLY
#include "finetune_qlora.cpp"

#include "llama-ext.h"
#include "llama-impl.h"
#include "llama-model-saver.h"
#include "qlion-adapter.h"
#include "logit-distillation.h"

static int64_t train_sample_split_by_conversation(
        const std::vector<training_sample> & samples,
        float                                val_split,
        int64_t                            & train_conversations_out,
        int64_t                            & val_conversations_out) {

    train_conversations_out = 0;
    val_conversations_out   = 0;

    if (samples.empty()) {
        return 0;
    }

    std::vector<int64_t> conversation_starts;
    conversation_starts.reserve(samples.size());

    int64_t previous_id =
        std::numeric_limits<int64_t>::min();

    for (size_t i = 0; i < samples.size(); ++i) {
        const int64_t id =
            samples[i].conversation_id;

        if (i == 0 || id != previous_id) {
            conversation_starts.push_back(
                (int64_t) i);

            previous_id = id;
        }
    }

    const int64_t n_conversations =
        (int64_t) conversation_starts.size();

    const int64_t train_conversations =
        train_split_from_val_fraction(
            n_conversations,
            val_split);

    train_conversations_out =
        train_conversations;

    val_conversations_out =
        n_conversations - train_conversations;

    if (train_conversations <= 0) {
        return 0;
    }

    if (train_conversations >= n_conversations) {
        return (int64_t) samples.size();
    }

    return conversation_starts[
        (size_t) train_conversations];
}

struct qat_model_info {
    uint64_t quantized_bytes = 0;
    uint64_t nonquantized_bytes = 0;
    uint64_t momentum_bytes = 0;
    uint64_t residual_bytes = 0;
    int64_t quantized_tensors = 0;
};

static std::vector<enum ggml_type> qat_weight_types(const std::string & selected) {
    std::vector<enum ggml_type> result;
    for (const auto & name : string_split(selected, ",")) {
        enum ggml_type type = GGML_TYPE_COUNT;
        if (name == "mxfp4") type = GGML_TYPE_MXFP4;
        if (name == "q2_0") type = GGML_TYPE_Q2_0;
        if (name == "q4_0") type = GGML_TYPE_Q4_0;
        if (name == "q8_0") type = GGML_TYPE_Q8_0;
        if (type == GGML_TYPE_COUNT || std::find(result.begin(), result.end(), type) != result.end()) {
            return {};
        }
        result.push_back(type);
    }
    return result;
}

struct qat_param_filter_context {
    const std::vector<enum ggml_type> * types;
    std::vector<std::string> targets;
    int32_t freeze_layers = 0;
    bool tied_embedding = false;
};

static bool qat_name_filter(const ggml_tensor * tensor, const qat_param_filter_context & filter) {
    if (!tensor || strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return false;
    }
    if (strncmp(tensor->name, "blk.", 4) == 0) {
        char * end = nullptr;
        const long layer = std::strtol(tensor->name + 4, &end, 10);
        if (end != tensor->name + 4 && *end == '.' && layer >= 0 && layer < filter.freeze_layers) return false;
    }
    const bool tied = filter.tied_embedding && strcmp(tensor->name, "token_embd.weight") == 0;
    if (filter.targets.empty()) {
        return !tied;
    }
    return std::any_of(filter.targets.begin(), filter.targets.end(), [&](const std::string & target) {
        return strstr(tensor->name, target.c_str()) || (tied && strstr("output.weight", target.c_str()));
    });
}

static bool qat_param_filter(const ggml_tensor * tensor, void * userdata) {
    const auto & filter = *static_cast<const qat_param_filter_context *>(userdata);
    return tensor && std::find(filter.types->begin(), filter.types->end(), tensor->type) != filter.types->end() &&
        qat_name_filter(tensor, filter);
}

static bool qat_inspect_model(const std::string & path, const std::vector<enum ggml_type> & expected, struct qat_model_info & info, qat_param_filter_context & filter) {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params params = { true, &ctx_meta };
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        return false;
    }
    filter.tied_embedding = gguf_find_tensor(gctx, "token_embd.weight") >= 0 && gguf_find_tensor(gctx, "output.weight") < 0;
    const int64_t arch_key = gguf_find_key(gctx, "general.architecture");
    const std::string layers_key = arch_key >= 0 && gguf_get_kv_type(gctx, arch_key) == GGUF_TYPE_STRING ?
        std::string(gguf_get_val_str(gctx, arch_key)) + ".block_count" : "";
    const int64_t layers_id = gguf_find_key(gctx, layers_key.c_str());
    const bool valid_layers = layers_id >= 0 && gguf_get_kv_type(gctx, layers_id) == GGUF_TYPE_UINT32 &&
        filter.freeze_layers >= 0 && uint32_t(filter.freeze_layers) <= gguf_get_val_u32(gctx, layers_id);
    if (!valid_layers) {
        LOG_ERR("%s: --freeze-layers is outside the model layer range\n", __func__);
        gguf_free(gctx);
        ggml_free(ctx_meta);
        return false;
    }
    info = {};
    std::vector<int64_t> counts(expected.size(), 0);
    for (int64_t i = 0; i < gguf_get_n_tensors(gctx); ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_meta, name);
        GGML_ASSERT(tensor);
        if (ggml_is_quantized(tensor->type)) {
            const auto found = std::find(expected.begin(), expected.end(), tensor->type);
            if (found == expected.end() && qat_name_filter(tensor, filter)) {
                LOG_ERR("%s: tensor %s has unsupported type %s\n",
                    __func__, name, ggml_type_name(tensor->type));
                gguf_free(gctx);
                ggml_free(ctx_meta);
                return false;
            }
            if (found != expected.end()) counts[found - expected.begin()]++;
            info.quantized_bytes += ggml_nbytes(tensor);
            if (qat_param_filter(tensor, &filter)) {
                info.momentum_bytes += ggml_row_size(GGML_TYPE_Q8_0, tensor->ne[0]) * ggml_nrows(tensor);
                info.residual_bytes += ggml_row_size(GGML_TYPE_Q4_0, tensor->ne[0]) * ggml_nrows(tensor);
                info.quantized_tensors++;
                if (filter.tied_embedding && strcmp(name, "token_embd.weight") == 0) {
                    LOG_WRN("qat_targets: tied token_embd/output selected; gradient covers the full vocabulary; dense_f32_bytes=%llu full_adapter_bytes=%llu\n",
                        (unsigned long long) (ggml_nelements(tensor) * sizeof(float)), (unsigned long long) ggml_nbytes(tensor));
                }
            }
        } else {
            info.nonquantized_bytes += ggml_nbytes(tensor);
        }
    }
    gguf_free(gctx);
    ggml_free(ctx_meta);
    for (size_t i = 0; i < expected.size(); ++i) {
        if (counts[i] == 0) {
            LOG_ERR("%s: model contains no %s tensors\n", __func__, ggml_type_name(expected[i]));
            return false;
        }
    }
    if (info.quantized_tensors == 0) {
        LOG_ERR("%s: --qat-targets and --freeze-layers select no supported quantized tensors\n", __func__);
        return false;
    }
    return true;
}

struct qat_opt_lr_context {
    qlora_lr_schedule * schedule;
    bool fast_state_scale;
    int32_t lr_scale_unit;
    float lr_scale_tau;
    int32_t lr_scale_warmup;
};

static ggml_opt_optimizer_params qat_opt_lr_pars(
        void * userdata) {

    qat_opt_lr_context & opt =
        *(qat_opt_lr_context *) userdata;

    qlora_lr_schedule & schedule =
        *opt.schedule;

    ggml_opt_optimizer_params result =
        ggml_opt_get_default_optimizer_params(
            nullptr
        );

    schedule.current_lr =
        schedule.get_lr();

    result.qlion_qat.alpha =
        schedule.current_lr;

    result.qlion_qat.wd =
        schedule.lr->wd;

    result.qlion_qat.fast_state_scale =
        opt.fast_state_scale;

    result.qlion_qat.lr_scale_unit = opt.lr_scale_unit;
    result.qlion_qat.lr_scale_tau = opt.lr_scale_tau;
    result.qlion_qat.lr_scale_warmup = opt.lr_scale_warmup;
    return result;
}

struct qat_resume_state {
    int64_t epoch = 0;
    int64_t window = 0;
    int64_t schedule_step = 0;
};

static std::string qat_state_path(const std::string & model_path) {
    return model_path + ".qat-state.gguf";
}

static bool qat_save_state(
        struct llama_context * lctx,
        const std::string & model_path,
        int64_t epoch,
        int64_t window,
        int64_t schedule_step,
        const std::string & quant_type,
        const common_params & config) {
    const int64_t count = llama_opt_qat_state_count(lctx);
    struct ggml_init_params meta_params = {
        std::max<int64_t>(1, 3 * count) * ggml_tensor_overhead(), nullptr, true
    };
    struct ggml_context * tensor_ctx = ggml_init(meta_params);
    struct gguf_context * gctx = gguf_init_empty();
    gguf_set_val_str(gctx, "general.type", "optimizer");
    gguf_set_val_u32(gctx, "training.qat.version", 1);
    gguf_set_val_str(gctx, "training.qat.quant_type", quant_type.c_str());
    gguf_set_val_str(gctx, "training.qat.optimizer", ggml_opt_optimizer_name(config.optimizer));
    gguf_set_val_u32(gctx, "training.qat.context_length", config.n_ctx);
    gguf_set_val_u32(gctx, "training.qat.parallel", config.n_parallel);
    gguf_set_val_u32(gctx, "training.qat.grad_accum_steps", config.grad_accum_steps);
    gguf_set_val_str(gctx, "training.qat.loss", config.distill_cache.empty() ? "ce" : "top_k_tail_kl");
    if (!config.distill_cache.empty()) {
        gguf_set_val_str(gctx, "training.qat.distill_cache", std::filesystem::weakly_canonical(config.distill_cache).string().c_str());
        gguf_set_val_u32(gctx, "training.qat.distill_top_k", config.distill_top_k);
        gguf_set_val_f32(gctx, "training.qat.distill_temperature", config.distill_temperature);
    }
    if (config.qat_lr_scale_unit) {
        gguf_set_val_u32(gctx, "training.qat.lr.unit", config.qat_lr_scale_unit);
        gguf_set_val_f32(gctx, "training.qat.lr.tau", config.qat_lr_scale_tau);
        gguf_set_val_u32(gctx, "training.qat.lr.warmup", config.qat_lr_scale_warmup);
    }
    gguf_set_val_str(gctx, "training.qat.momentum_type", "q8_0");
    gguf_set_val_str(gctx, "training.qat.residual_type", "q4_0");
    gguf_set_val_i64(gctx, "training.qat.state_count", count);
    gguf_set_val_i64(gctx, "training.optimizer.step", llama_opt_step(lctx));
    gguf_set_val_i64(gctx, "training.scheduler.step", schedule_step);
    gguf_set_val_i64(gctx, "training.dataset.epoch", epoch);
    gguf_set_val_i64(gctx, "training.dataset.window", window);
    std::vector<struct ggml_tensor *> sources;
    for (int64_t i = 0; i < count; ++i) {
        struct ggml_tensor * param = llama_opt_qat_state_param(lctx, i);
        struct ggml_tensor * momentum = llama_opt_qat_state_momentum(lctx, i);
        struct ggml_tensor * residual = llama_opt_qat_state_residual(lctx, i);
        const std::string param_key = "training.qat.param." + std::to_string(i);
        gguf_set_val_str(gctx, param_key.c_str(), param->name);
        struct ggml_tensor * momentum_meta = ggml_new_tensor(tensor_ctx, GGML_TYPE_Q8_0, GGML_MAX_DIMS, momentum->ne);
        struct ggml_tensor * residual_meta = ggml_new_tensor(tensor_ctx, GGML_TYPE_Q4_0, GGML_MAX_DIMS, residual->ne);
        ggml_format_name(momentum_meta, "qat.m.%06ld", (long) i);
        ggml_format_name(residual_meta, "qat.r.%06ld", (long) i);
        gguf_add_tensor(gctx, momentum_meta);
        gguf_add_tensor(gctx, residual_meta);
        sources.push_back(momentum);
        sources.push_back(residual);
        if (struct ggml_tensor * lr_state = llama_opt_qat_state_lr_scale(lctx, i)) {
            struct ggml_tensor * meta = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 2);
            ggml_format_name(meta, "qat.lr.%06ld", (long) i);
            gguf_add_tensor(gctx, meta);
            sources.push_back(lr_state);
        }
    }
    const std::string path = qat_state_path(model_path);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        gguf_free(gctx);
        ggml_free(tensor_ctx);
        return false;
    }
    const size_t header_size = gguf_get_meta_size(gctx);
    std::vector<char> header(header_size, 0);
    out.write(header.data(), header.size());
    const size_t chunk_max = 16 * 1024 * 1024;
    std::vector<char> data(chunk_max);
    for (struct ggml_tensor * source : sources) {
        const size_t nbytes = ggml_nbytes(source);
        for (size_t offset = 0; offset < nbytes; offset += chunk_max) {
            const size_t chunk = std::min(chunk_max, nbytes - offset);
            ggml_backend_tensor_get(source, data.data(), offset, chunk);
            out.write(data.data(), chunk);
        }
        const size_t padding = GGML_PAD(nbytes, 32) - nbytes;
        if (padding) {
            std::vector<char> zeros(padding, 0);
            out.write(zeros.data(), zeros.size());
        }
    }
    std::vector<uint8_t> metadata(header_size);
    gguf_get_meta_data(gctx, metadata.data());
    out.seekp(0);
    out.write((const char *) metadata.data(), metadata.size());
    const bool ok = out.good();
    out.close();
    gguf_free(gctx);
    ggml_free(tensor_ctx);
    return ok;
}

static bool qat_load_state(
        struct llama_context * lctx,
        const std::string & model_path,
        const std::string & quant_type,
        struct qat_resume_state & resume,
        const common_params & config) {
    struct ggml_context * data_ctx = nullptr;
    struct gguf_init_params params = { true, &data_ctx };
    const std::string path = qat_state_path(model_path);
    struct gguf_context * gctx = gguf_init_from_file(path.c_str(), params);
    if (!gctx) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        return false;
    }
    const int context_key = gguf_find_key(gctx, "training.qat.context_length");
    const int loss_key = gguf_find_key(gctx, "training.qat.loss");
    const int cache_key = gguf_find_key(gctx, "training.qat.distill_cache");
    const int k_key = gguf_find_key(gctx, "training.qat.distill_top_k");
    const int t_key = gguf_find_key(gctx, "training.qat.distill_temperature");
    const bool kd = !config.distill_cache.empty();
    const bool loss_valid = loss_key < 0 ? !kd : gguf_get_kv_type(gctx, loss_key) == GGUF_TYPE_STRING &&
            std::strcmp(gguf_get_val_str(gctx, loss_key), kd ? "top_k_tail_kl" : "ce") == 0;
    const bool cache_valid = !kd || (cache_key >= 0 && gguf_get_kv_type(gctx, cache_key) == GGUF_TYPE_STRING &&
            std::filesystem::weakly_canonical(config.distill_cache).string() == gguf_get_val_str(gctx, cache_key) &&
            k_key >= 0 && gguf_get_kv_type(gctx, k_key) == GGUF_TYPE_UINT32 && gguf_get_val_u32(gctx, k_key) == uint32_t(config.distill_top_k) &&
            t_key >= 0 && gguf_get_kv_type(gctx, t_key) == GGUF_TYPE_FLOAT32 && gguf_get_val_f32(gctx, t_key) == config.distill_temperature);
    if (!loss_valid || !cache_valid) {
        LOG_ERR("%s: checkpoint loss or teacher cache configuration differs\n", __func__);
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    if (context_key >= 0 && (gguf_get_kv_type(gctx, context_key) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(gctx, context_key) != (uint32_t) config.n_ctx)) {
        LOG_ERR("%s: checkpoint sequence context differs from --ctx-size\n", __func__);
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    const int64_t lr_unit = gguf_find_key(gctx, "training.qat.lr.unit");
    const int64_t lr_tau = gguf_find_key(gctx, "training.qat.lr.tau");
    const int64_t lr_warmup = gguf_find_key(gctx, "training.qat.lr.warmup");
    const bool has_lr = lr_unit >= 0 && lr_tau >= 0 && lr_warmup >= 0 &&
        gguf_get_kv_type(gctx, lr_unit) == GGUF_TYPE_UINT32 &&
        gguf_get_kv_type(gctx, lr_tau) == GGUF_TYPE_FLOAT32 &&
        gguf_get_kv_type(gctx, lr_warmup) == GGUF_TYPE_UINT32;
    const int optimizer_key = gguf_find_key(gctx, "training.qat.optimizer");
    if (optimizer_key >= 0 && (gguf_get_kv_type(gctx, optimizer_key) != GGUF_TYPE_STRING ||
        std::strcmp(gguf_get_val_str(gctx, optimizer_key), ggml_opt_optimizer_name(config.optimizer)) != 0)) {
        LOG_ERR("%s: checkpoint optimizer does not match --optimizer\n", __func__);
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    if ((config.qat_lr_scale_unit != 0 && !has_lr) ||
        (has_lr && (gguf_get_val_u32(gctx, lr_unit) != (uint32_t) config.qat_lr_scale_unit ||
                    gguf_get_val_f32(gctx, lr_tau) != config.qat_lr_scale_tau ||
                    gguf_get_val_u32(gctx, lr_warmup) != (uint32_t) config.qat_lr_scale_warmup))) {
        LOG_ERR("%s: adaptive LR settings differ from checkpoint\n", __func__);
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    const int64_t count_key = gguf_find_key(gctx, "training.qat.state_count");
    const int64_t type_key = gguf_find_key(gctx, "training.qat.quant_type");
    const int64_t count = count_key >= 0 ? gguf_get_val_i64(gctx, count_key) : -1;
    if (type_key < 0 || quant_type != gguf_get_val_str(gctx, type_key) || count != llama_opt_qat_state_count(lctx)) {
        LOG_ERR("%s: checkpoint state does not match the current model\n", __func__);
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        LOG_ERR("%s: cannot open %s\n", __func__, path.c_str());
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    const size_t chunk_max = 16 * 1024 * 1024;
    std::vector<uint8_t> data(chunk_max);
    std::vector<uint8_t> check_data(chunk_max);
    for (int64_t i = 0; i < count; ++i) {
        const std::string param_key = "training.qat.param." + std::to_string(i);
        const int64_t key = gguf_find_key(gctx, param_key.c_str());
        struct ggml_tensor * param = llama_opt_qat_state_param(lctx, i);
        struct ggml_tensor * momentum = llama_opt_qat_state_momentum(lctx, i);
        struct ggml_tensor * residual = llama_opt_qat_state_residual(lctx, i);
        char momentum_name[32];
        char residual_name[32];
        snprintf(momentum_name, sizeof(momentum_name), "qat.m.%06ld", (long) i);
        snprintf(residual_name, sizeof(residual_name), "qat.r.%06ld", (long) i);
        struct ggml_tensor * momentum_saved = ggml_get_tensor(data_ctx, momentum_name);
        struct ggml_tensor * residual_saved = ggml_get_tensor(data_ctx, residual_name);
        if (key < 0 || strcmp(gguf_get_val_str(gctx, key), param->name) != 0 || !momentum_saved || !residual_saved ||
            ggml_nbytes(momentum_saved) != ggml_nbytes(momentum) || ggml_nbytes(residual_saved) != ggml_nbytes(residual)) {
            LOG_ERR("%s: state %ld does not match parameter %s\n", __func__, (long) i, param->name);
            gguf_free(gctx);
            ggml_free(data_ctx);
            return false;
        }
        const size_t momentum_bytes = ggml_nbytes(momentum);
        const size_t residual_bytes = ggml_nbytes(residual);
        struct ggml_tensor * lr_state = llama_opt_qat_state_lr_scale(lctx, i);
        char lr_name[32];
        snprintf(lr_name, sizeof(lr_name), "qat.lr.%06ld", (long) i);
        struct ggml_tensor * lr_saved = ggml_get_tensor(data_ctx, lr_name);
        if (lr_state && (!lr_saved || lr_saved->type != GGML_TYPE_F32 || ggml_nelements(lr_saved) != 2)) {
            LOG_ERR("%s: checkpoint lacks adaptive LR state for %s\n", __func__, param->name);
            gguf_free(gctx);
            ggml_free(data_ctx);
            return false;
        }
        struct ggml_tensor * targets[] = { momentum, residual, lr_state };
        struct ggml_tensor * saved[] = { momentum_saved, residual_saved, lr_saved };
        const size_t sizes[] = { momentum_bytes, residual_bytes, 2 * sizeof(float) };
        for (int state = 0; state < (lr_state ? 3 : 2); ++state) {
            const int64_t tensor_id = gguf_find_tensor(gctx, saved[state]->name);
            GGML_ASSERT(tensor_id >= 0);
            const size_t file_offset = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tensor_id);
            for (size_t offset = 0; offset < sizes[state]; offset += chunk_max) {
                const size_t chunk = std::min(chunk_max, sizes[state] - offset);
                in.seekg(file_offset + offset);
                in.read((char *) data.data(), chunk);
                if (!in.good()) {
                    LOG_ERR("%s: failed to read state for parameter %s\n", __func__, param->name);
                    gguf_free(gctx);
                    ggml_free(data_ctx);
                    return false;
                }
                if (state == 2) {
                    float lr[2];
                    memcpy(lr, data.data(), sizeof(lr));
                    if (!std::isfinite(lr[0]) || lr[0] < 0 || !std::isfinite(lr[1]) ||
                        lr[1] < 0 || lr[1] > 16777216 || lr[1] != std::floor(lr[1])) {
                        LOG_ERR("%s: invalid adaptive LR checkpoint state\n", __func__);
                        gguf_free(gctx);
                        ggml_free(data_ctx);
                        return false;
                    }
                }
                ggml_backend_tensor_set(targets[state], data.data(), offset, chunk);
                ggml_backend_tensor_get(targets[state], check_data.data(), offset, chunk);
                if (memcmp(check_data.data(), data.data(), chunk) != 0) {
                    LOG_ERR("%s: backend changed state data for parameter %s\n", __func__, param->name);
                    gguf_free(gctx);
                    ggml_free(data_ctx);
                    return false;
                }
            }
        }
    }
    const int64_t step_key = gguf_find_key(gctx, "training.optimizer.step");
    const int64_t schedule_key = gguf_find_key(gctx, "training.scheduler.step");
    const int64_t epoch_key = gguf_find_key(gctx, "training.dataset.epoch");
    const int64_t window_key = gguf_find_key(gctx, "training.dataset.window");
    if (step_key < 0 || schedule_key < 0 || epoch_key < 0 || window_key < 0) {
        LOG_ERR("%s: checkpoint progress metadata is incomplete\n", __func__);
        gguf_free(gctx);
        ggml_free(data_ctx);
        return false;
    }
    llama_opt_set_step(lctx, gguf_get_val_i64(gctx, step_key));
    resume.schedule_step = gguf_get_val_i64(gctx, schedule_key);
    resume.epoch = gguf_get_val_i64(gctx, epoch_key);
    resume.window = gguf_get_val_i64(gctx, window_key);
    gguf_free(gctx);
    ggml_free(data_ctx);
    return true;
}

static bool qat_validate_saved_metadata(const std::string & source_path, const std::string & output_path) {
    struct ggml_context * source_data = nullptr;
    struct ggml_context * output_data = nullptr;
    struct gguf_init_params source_params = { true, &source_data };
    struct gguf_init_params output_params = { true, &output_data };
    struct gguf_context * source = gguf_init_from_file(source_path.c_str(), source_params);
    struct gguf_context * output = gguf_init_from_file(output_path.c_str(), output_params);
    if (!source || !output) {
        LOG_ERR("%s: cannot reopen saved model metadata\n", __func__);
        if (source) {
            gguf_free(source);
        }
        if (output) {
            gguf_free(output);
        }
        ggml_free(source_data);
        ggml_free(output_data);
        return false;
    }

    bool valid = true;
    for (int64_t i = 0; i < gguf_get_n_kv(source); ++i) {
        const char * key = gguf_get_key(source, i);
        if (strcmp(key, "split.no") == 0 || strcmp(key, "split.count") == 0 || strcmp(key, "split.tensors.count") == 0) {
            continue;
        }
        const int64_t output_key = gguf_find_key(output, key);
        if (output_key < 0 || gguf_get_kv_type(source, i) != gguf_get_kv_type(output, output_key) ||
            gguf_kv_to_str(source, i) != gguf_kv_to_str(output, output_key)) {
            LOG_ERR("%s: saved metadata mismatch for %s\n", __func__, key);
            valid = false;
            break;
        }
        if (gguf_get_kv_type(source, i) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(source, i) != gguf_get_arr_type(output, output_key)) {
            LOG_ERR("%s: saved metadata array type mismatch for %s\n", __func__, key);
            valid = false;
            break;
        }
    }

    if (valid) {
        LOG_INF("%s: preserved source metadata (%ld keys)\n", __func__, (long) gguf_get_n_kv(source));
    }
    gguf_free(source);
    gguf_free(output);
    ggml_free(source_data);
    ggml_free(output_data);
    return valid;
}

static bool qat_save_model(
        const struct llama_model * model,
        const std::string        & source_path,
        const std::string        & output_path) {
    struct ggml_context * source_data = nullptr;
    struct gguf_init_params init_params = { true, &source_data };
    struct gguf_context * source = gguf_init_from_file(source_path.c_str(), init_params);
    if (!source) {
        LOG_ERR("%s: cannot open source metadata from %s\n", __func__, source_path.c_str());
        return false;
    }

    llama_model_saver saver(model);
    gguf_set_kv(saver.gguf_ctx, source);
    gguf_remove_key(saver.gguf_ctx, "split.no");
    gguf_remove_key(saver.gguf_ctx, "split.count");
    gguf_remove_key(saver.gguf_ctx, "split.tensors.count");
    saver.add_tensors_from_model();

    const bool saved = gguf_write_to_file(saver.gguf_ctx, output_path.c_str(), false);
    gguf_free(source);
    ggml_free(source_data);

    if (!saved) {
        LOG_ERR("%s: failed to save %s\n", __func__, output_path.c_str());
        return false;
    }
    return qat_validate_saved_metadata(source_path, output_path);
}

static std::string qat_normalize_chat_template(std::string source) {
    string_replace_all(source, "\r\n", "\n");
    string_replace_all(source, "\r", "\n");
    if (!source.empty() && source.back() == '\n') {
        source.pop_back();
    }
    return source;
}

static bool qat_save_checkpoint(
        struct llama_context * lctx,
        struct llama_model * model,
        const std::string & source_path,
        const std::string & path,
        int64_t epoch,
        int64_t window,
        int64_t schedule_step,
        const std::string & quant_type,
        const common_params & config) {
    llama_synchronize(lctx);
    if (!qat_save_model(model, source_path, path)) {
        return false;
    }
    if (!qat_save_state(lctx, path, epoch, window, schedule_step, quant_type, config)) {
        return false;
    }
    LOG_INF("%s: saved %s and %s\n", __func__, path.c_str(), qat_state_path(path).c_str());
    return true;
}

struct qat_callback_context {
    struct llama_context * lctx;
    struct llama_context * mtp_lctx;
    struct llama_model * model;
    mtp_training_state * mtp;
    qlora_lr_schedule * schedule;
    const common_params * params;
    int64_t epoch;
    int64_t window_start;
    int64_t ubatches_per_window;
    int64_t last_saved_step;
    int64_t last_observed_step;
    ggml_opt_result_t result_eval;
    int64_t validation_start;
    int64_t validation_end;
    bool    first_save_pending;
};

static thread_local struct qat_callback_context * g_qat_callback = nullptr;

static void qat_run_periodic_eval(ggml_opt_dataset_t dataset, int64_t step) {
    auto * callback = g_qat_callback;
    if (!callback || callback->validation_end <= callback->validation_start) return;
    ggml_opt_result_reset(callback->result_eval);
    llama_opt_eval_range(callback->lctx, dataset, callback->result_eval,
        callback->validation_start, callback->validation_end, nullptr);
    double loss = 0.0;
    double accuracy = 0.0;
    ggml_opt_result_loss(callback->result_eval, &loss, nullptr);
    ggml_opt_result_accuracy(callback->result_eval, &accuracy, nullptr);
    LOG_INF("validation: step=%ld epoch=%ld loss=%.8f acc=%.4f%%\n",
        (long) step, (long) (callback->epoch + 1), loss, 100.0 * accuracy);
    ggml_opt_result_reset(callback->result_eval);
}

static void qat_epoch_callback(
        bool train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t result,
        int64_t ibatch,
        int64_t ibatch_max,
        int64_t t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
    if (!train || !g_qat_callback) {
        return;
    }
    if (ibatch == ibatch_max && !g_qat_callback->params->distill_cache.empty()) {
        const double elapsed = (ggml_time_us() - t_start_us)*1e-6;
        const int64_t input_tokens = ibatch*g_qat_callback->params->n_ubatch;
        const auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data);
        const int64_t context = ggml_opt_dataset_labels(dataset)->ne[0];
        const int64_t supervised = std::count_if(labels + g_qat_callback->window_start*context,
                labels + g_qat_callback->validation_start*context, [](int32_t token) { return token >= 0; });
        LOG_INF("qat_throughput: input_tokens=%ld supervised_tokens=%ld elapsed_s=%.3f input_tok_s=%.3f supervised_tok_s=%.3f\n",
                (long) input_tokens, (long) supervised, elapsed, elapsed > 0 ? input_tokens/elapsed : 0,
                elapsed > 0 ? supervised/elapsed : 0);
    }
    struct llama_context * step_lctx =
        g_qat_callback->params->mtp_mode == "only" ? g_qat_callback->mtp_lctx : g_qat_callback->lctx;
    const int64_t step = llama_opt_step(step_lctx);
    if (step <= g_qat_callback->last_observed_step) {
        return;
    }
    g_qat_callback->last_observed_step = step;
    ++g_qat_callback->schedule->step;
    if (g_qat_callback->params->eval_every > 0 && step % g_qat_callback->params->eval_every == 0) {
        qat_run_periodic_eval(dataset, step);
    }
    if (g_qat_callback->params->verbose_loss) {
        double loss = 0.0;
        ggml_opt_result_loss(result, &loss, nullptr);
        LOG_INF("qat_loss: epoch=%ld window=%ld step=%ld loss=%.8f lr=%.9g\n",
            (long) g_qat_callback->epoch, (long) ibatch, (long) step,
            loss, (double) g_qat_callback->schedule->current_lr);
    }
    const bool window_complete = ibatch > 0 && ibatch % g_qat_callback->ubatches_per_window == 0;
    const bool first_save_due = g_qat_callback->first_save_pending &&
        g_qat_callback->params->save_first_at > 0 &&
        step >= g_qat_callback->params->save_first_at;
    const bool periodic_save_due = g_qat_callback->params->save_every > 0 &&
        step - g_qat_callback->last_saved_step >= g_qat_callback->params->save_every;
    if (window_complete && (first_save_due || periodic_save_due)) {
        const int64_t window = g_qat_callback->window_start + ibatch / g_qat_callback->ubatches_per_window;
        bool saved = true;
        if (g_qat_callback->params->mtp_mode != "only") {
            const std::string path = g_qat_callback->params->qat_out + ".step" + std::to_string(step) + ".gguf";
            saved = qat_save_checkpoint(g_qat_callback->lctx, g_qat_callback->model,
                g_qat_callback->params->model.path, path, g_qat_callback->epoch, window,
                g_qat_callback->schedule->step, g_qat_callback->params->qat_quant_type, *g_qat_callback->params);
        }
        if (g_qat_callback->mtp_lctx) {
            checkpoint_state state;
            state.mode            = "sft";
            state.epoch           = g_qat_callback->epoch + 1;
            state.window          = window;
            state.step            = step;
            state.dataset_windows = g_qat_callback->schedule->total_steps /
                std::max<unsigned>(1u, g_qat_callback->params->lr.epochs);
            state.context_length  = llama_n_ctx(g_qat_callback->lctx);
            state.shuffle         = g_qat_callback->params->shuffle_dataset;
            state.schedule_step   = g_qat_callback->schedule->step;
            const std::string path = g_qat_callback->params->mtp_lora_out
                + ".step" + std::to_string(step) + ".gguf";
            llama_synchronize(g_qat_callback->mtp_lctx);
            saved = save_adapter(g_qat_callback->mtp->tensors, path, g_qat_callback->mtp->arch,
                g_qat_callback->mtp->alpha, g_qat_callback->params->mtp_model, &state) && saved;
        }
        if (saved) {
            g_qat_callback->last_saved_step = step;
            g_qat_callback->first_save_pending = false;
            LOG_INF("%s: checkpoint set saved at step %ld window %ld\n",
                __func__, (long) step, (long) window);
        } else {
            LOG_ERR("%s: checkpoint set failed at step %ld window %ld\n",
                __func__, (long) step, (long) window);
        }
    }
}

static void qat_print_memory(struct llama_context * lctx, const struct qat_model_info & model_info) {
    uint64_t momentum_bytes = 0;
    uint64_t residual_bytes = 0;
    uint64_t dense_gradient_bytes = 0;
    uint64_t lr_scale_bytes = 0;
    uint64_t sparse_gradient_bytes = 0;
    for (int64_t i = 0; i < llama_opt_qat_state_count(lctx); ++i) {
        momentum_bytes += ggml_nbytes(llama_opt_qat_state_momentum(lctx, i));
        residual_bytes += ggml_nbytes(llama_opt_qat_state_residual(lctx, i));
        const auto * lr_scale = llama_opt_qat_state_lr_scale(lctx, i);
        if (lr_scale) lr_scale_bytes += ggml_nbytes(lr_scale);
        struct ggml_tensor * gradient = llama_opt_qat_state_gradient_accumulator(lctx, i);
        if (gradient) {
            const struct ggml_tensor * param = llama_opt_qat_state_param(lctx, i);
            if (ggml_are_same_shape(param, gradient)) {
                dense_gradient_bytes += ggml_nbytes(gradient);
            } else {
                sparse_gradient_bytes += ggml_nbytes(gradient);
            }
        }
    }
    uint64_t context_bytes = 0;
    uint64_t compute_bytes = 0;
    for (const auto & item : llama_get_memory_breakdown(lctx)) {
        context_bytes += item.second.context;
        compute_bytes += item.second.compute;
    }
    size_t device_used_bytes = 0;
    if (llama_opt_qat_state_count(lctx) > 0) {
        const struct ggml_tensor * param = llama_opt_qat_state_param(lctx, 0);
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(param->buffer);
        const ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
        if (device && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
            device_used_bytes = total_bytes - free_bytes;
        }
    }
    LOG_INF("qat_memory: base_quantized=%llu momentum_q8_0=%llu residual_q4_0=%llu lr_scale=%llu dense_gradient_q8_0=%llu sparse_gradient_q8_0=%llu nonquantized_model=%llu context=%llu compute_workspace=%llu persistent_qat=%llu device_used=%zu\n",
        (unsigned long long) model_info.quantized_bytes, (unsigned long long) momentum_bytes,
        (unsigned long long) residual_bytes, (unsigned long long) lr_scale_bytes, (unsigned long long) dense_gradient_bytes,
        (unsigned long long) sparse_gradient_bytes,
        (unsigned long long) model_info.nonquantized_bytes,
        (unsigned long long) context_bytes, (unsigned long long) compute_bytes,
        (unsigned long long) (model_info.quantized_bytes + momentum_bytes + residual_bytes + lr_scale_bytes + dense_gradient_bytes + sparse_gradient_bytes),
        device_used_bytes);
}

static void distillation_limit_dataset(ggml_opt_dataset_t & dataset, int64_t & split, int32_t limit,
        std::vector<float> & rewards, int32_t validation_limit = 0) {
    if ((!limit && !validation_limit) || !dataset) return;
    const int64_t total = ggml_opt_dataset_ndata(dataset);
    const int64_t train = limit ? std::min<int64_t>(limit, split) : split;
    const int64_t available_val = limit ? std::min<int64_t>(limit, total - split) : total - split;
    const int64_t val = validation_limit ? std::min<int64_t>(validation_limit, available_val) : available_val;
    const int64_t context = ggml_opt_dataset_data(dataset)->ne[0];
    ggml_opt_dataset_t subset = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, context, context, train + val, 1);
    std::vector<float> selected;
    for (int64_t i = 0; i < train + val; ++i) {
        const int64_t src = i < train ? i : split + (i - train)*available_val/val;
        std::memcpy(static_cast<int32_t *>(ggml_opt_dataset_data(subset)->data) + i*context,
                static_cast<int32_t *>(ggml_opt_dataset_data(dataset)->data) + src*context, context*4);
        std::memcpy(static_cast<int32_t *>(ggml_opt_dataset_labels(subset)->data) + i*context,
                static_cast<int32_t *>(ggml_opt_dataset_labels(dataset)->data) + src*context, context*4);
        selected.push_back(rewards.at(src));
    }
    ggml_opt_dataset_free(dataset);
    dataset = subset;
    split = train;
    rewards = std::move(selected);
}

#ifndef LLAMA_LOGIT_EXTRACT_SHARED_ONLY
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    set_default_progress_ema();
    common_params params;
    params.escape = false;
    params.optimizer = GGML_OPT_OPTIMIZER_TYPE_QLION_QAT;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--qat-resume") == 0) {
            params.model.path = argv[i + 1];
            break;
        }
    }
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QAT)) {
        return 1;
    }
    if ((!params.distill_cache.empty() || params.distill_max_windows) &&
            (params.critical_token_mode != "none" || params.mtp_mode != "off")) {
        LOG_ERR("distillation does not support critical-token weighting or MTP\n");
        return 1;
    }
    if (params.train_file.empty()) {
        LOG_ERR("%s: --train-file is required\n", __func__);
        return 1;
    }
    if (params.qat_out.empty() && (params.qat_adapter_out.empty() || params.save_every > 0 || params.save_first_at > 0)) {
        LOG_ERR("%s: empty --qat-out requires --qat-adapter-out and disabled full checkpoints\n", __func__);
        return 1;
    }
    if (params.grpo_mode || !params.lora_adapters.empty()) {
        LOG_ERR("%s: native QAT does not accept GRPO or LoRA adapters\n", __func__);
        return 1;
    }
    if (!ggml_opt_optimizer_is_qlion(params.optimizer)) {
        LOG_ERR("%s: native QAT only supports --optimizer qlion or qlion2\n", __func__);
        return 1;
    }
    if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_QLION2_QAT) {
        params.qat_lr_scale_unit = 32;
    }
    LOG_INF("%s: optimizer=%s lr_scale_unit=%d tau=%.9g lr_scale_warmup=%d\n", __func__,
            ggml_opt_optimizer_name(params.optimizer), params.qat_lr_scale_unit, params.qat_lr_scale_tau, params.qat_lr_scale_warmup);
    if (params.n_ctx <= 0 || params.n_batch != params.n_ctx || params.n_ubatch <= 0 ||
        params.n_batch % params.n_ubatch != 0) {
        LOG_ERR("%s: native QLion requires explicit -c N -b N with -ub M where N is divisible by M\n", __func__);
        return 1;
    }
    if (params.n_parallel < 1 || (size_t) params.n_parallel > llama_max_parallel_sequences() ||
        params.n_ctx > INT32_MAX / params.n_parallel ||
        (params.n_parallel > 1 && (params.n_ctx % 256 != 0 || params.kv_unified))) {
        LOG_ERR("%s: parallel QAT requires positive parallel count, separate KV streams and context divisible by 256\n", __func__);
        return 1;
    }
    if ((params.n_parallel > 1 || params.grad_accum_steps > 1) &&
        (params.mtp_mode != "off" || params.critical_token_mode != "none")) {
        LOG_ERR("%s: parallel/accumulated QAT does not yet support MTP or critical-token weighting\n", __func__);
        return 1;
    }
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    if (!params.kv_cache_training) {
        params.cache_type_k = GGML_TYPE_F32;
        params.cache_type_v = GGML_TYPE_F32;
    } else {
        LOG_INF("%s: KV cache training target: K=%s V=%s hadamard_k=%d hadamard_v=%d\n",
                __func__, ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
                params.cache_hadamard_k, params.cache_hadamard_v);
    }
    params.warmup = false;
    // Keep the requested FlashAttention policy; quantized V cache training requires it.
    params.no_extra_bufts = true;
    if (!params.qat_resume.empty()) {
        params.model.path = params.qat_resume;
    }
    std::vector<enum ggml_type> weight_types = qat_weight_types(params.qat_quant_type);
    if (weight_types.empty()) {
        LOG_ERR("%s: invalid --quant-type list\n", __func__);
        return 1;
    }
    if (params.qat_quant_type == "q2_0" && params.n_gpu_layers > 0) {
        LOG_ERR("%s: q2_0 native QAT currently requires CPU weights; set -ngl 0\n", __func__);
        return 1;
    }
    qat_param_filter_context qat_filter { &weight_types, {}, params.lora_freeze_layers, false };
    for (const auto & target : string_split(params.qat_targets, ",")) {
        if (!target.empty()) {
            qat_filter.targets.push_back(target);
        }
    }
    struct qat_model_info model_info;
    if (!qat_inspect_model(params.model.path, weight_types, model_info, qat_filter)) {
        return 1;
    }
    const uint64_t gradient_estimate = params.n_ubatch < params.n_batch ? model_info.momentum_bytes : 0;
    LOG_INF("qat_memory_estimate: base_quantized=%llu momentum_q8_0=%llu residual_q4_0=%llu gradient_q8_0=%llu nonquantized_model=%llu persistent_qat=%llu\n",
        (unsigned long long) model_info.quantized_bytes, (unsigned long long) model_info.momentum_bytes,
        (unsigned long long) model_info.residual_bytes, (unsigned long long) gradient_estimate,
        (unsigned long long) model_info.nonquantized_bytes,
        (unsigned long long) (model_info.quantized_bytes + model_info.momentum_bytes + model_info.residual_bytes + gradient_estimate));

    params.offload_input = true;
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    common_params init_params = params;
    init_params.n_ctx *= params.n_parallel;
    init_params.n_batch *= params.n_parallel;
    init_params.n_ubatch *= params.n_parallel;
    auto llama_init = common_init_from_params(init_params);
    struct llama_model * model = llama_init->model();
    struct llama_context * lctx = llama_init->context();
    if (!model || !lctx) {
        LOG_ERR("%s: failed to load model and training context\n", __func__);
        return 1;
    }
    auto templates = common_chat_templates_init(model, params.chat_template);
    if (!params.chat_template.empty() && common_chat_templates_source(templates.get()) != qat_normalize_chat_template(params.chat_template)) {
        LOG_ERR("%s: chat template override was not applied\n", __func__);
        return 1;
    }
    LOG_INF("%s: dataset chat template=%s\n", __func__, params.chat_template.empty() ? "model metadata" : "command-line override");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_ctx = llama_n_ctx(lctx) / params.n_parallel;
    if (params.n_ubatch < n_ctx) {
        LOG_WRN("%s: past KV rows are detached at microbatch boundaries; use -ub equal to sequence_context for full-sequence K/V gradients\n", __func__);
    }
    LOG_INF("%s: sequence_context=%d parallel=%d microbatch_tokens=%u grad_accum=%d effective_sequences=%ld\n",
            __func__, n_ctx, params.n_parallel, llama_n_ubatch(lctx), params.grad_accum_steps,
            (long) params.n_parallel * params.grad_accum_steps);
    const size_t render_max_tokens = 0;
    std::vector<training_sample> samples = load_jsonl(params.train_file, vocab, templates.get(),
        params.dataset_threads, params.critical_token_mode, params.critical_token_weight,
        params.preserve_thinking ? 1 : 0, render_max_tokens);
    if (samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return 1;
    }
    int64_t train_conversations = 0;
    int64_t val_conversations = 0;

    const int64_t sample_split =
        train_sample_split_by_conversation(
            samples,
            params.val_split,
            train_conversations,
            val_conversations);

    LOG_INF(
        "%s: conversation split: "
        "train_conversations=%ld "
        "val_conversations=%ld "
        "train_samples=%ld "
        "val_samples=%ld "
        "total_samples=%zu\n",
        __func__,
        (long) train_conversations,
        (long) val_conversations,
        (long) sample_split,
        (long) (samples.size() - sample_split),
        samples.size());

    if (sample_split <= 0) {

        LOG_ERR(
            "%s: no training samples after val split "
            "(samples=%zu val_split=%.3f)\n",
            __func__,
            samples.size(),
            (double) params.val_split);

        return 1;
    }

    if (params.val_split > 0.0f &&
        sample_split ==
            (int64_t) samples.size()) {

        LOG_WRN(
            "%s: validation split produced "
            "no validation samples "
            "(samples=%zu val_split=%.3f)\n",
            __func__,
            samples.size(),
            (double) params.val_split);
    }
    std::vector<float> window_rewards;
    std::vector<llama_opt_critical_token_metadata> critical_metadata;
    const bool critical_enabled = params.critical_token_mode != "none";
    int64_t idata_split = 0;

    ggml_opt_dataset_t dataset =
        build_dataset(
            samples,
            n_ctx,
            window_rewards,

            // NEW:
            // Raw JSONL sample boundary.
            sample_split,

            // NEW:
            // build_dataset() fills this with
            // the actual number of TRAIN windows.
            idata_split,

            params.train_on_prompt,
            llama_vocab_bos(vocab),
            critical_enabled,
            &critical_metadata);

    if (!dataset) {
        return 1;
    }
    distillation_limit_dataset(dataset, idata_split, params.distill_max_windows, window_rewards, params.distill_validation_windows);
    const int64_t ndata = ggml_opt_dataset_ndata(dataset);
    std::unique_ptr<logit_distillation::cache> teacher_cache;
    if (!params.distill_cache.empty()) {
        try {
            teacher_cache.reset(new logit_distillation::cache(params.distill_cache));
            teacher_cache->attach(dataset, idata_split, vocab);
            params.distill_top_k = teacher_cache->info.top_k;
            params.distill_temperature = teacher_cache->info.temperature;
            llama_opt_set_distillation(lctx, teacher_cache->info.top_k, teacher_cache->info.temperature,
                    logit_distillation::cache::fetch, teacher_cache.get());
            LOG_INF("distillation: pure top-%u + tail KL, temperature=%.6g, no hard-label CE, train_windows=%ld validation_windows=%ld\n",
                    teacher_cache->info.top_k, teacher_cache->info.temperature, (long) idata_split, (long) (ndata - idata_split));
        } catch (const std::exception & e) {
            LOG_ERR("%s\n", e.what());
            ggml_opt_dataset_free(dataset);
            return 1;
        }
    }
    if (idata_split <= 0 ||
        idata_split > ndata) {

        LOG_ERR(
            "%s: invalid packed training boundary "
            "(train_windows=%ld total_windows=%ld)\n",
            __func__,
            (long) idata_split,
            (long) ndata);

        return 1;
    }
    if (params.val_split > 0.0f &&
        idata_split == ndata) {

        LOG_WRN(
            "%s: validation samples produced "
            "no supervised validation windows; "
            "validation will be skipped\n",
            __func__);
    }
    qlora_lr_schedule schedule {
        &params.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio,
        params.lr_decay_steps, 0, 0
    };
    qat_opt_lr_context qat_lr_ctx {
        &schedule,
        params.qat_fast_state_scale, params.qat_lr_scale_unit, params.qat_lr_scale_tau, params.qat_lr_scale_warmup
    };
    struct llama_opt_params opt_params {
        0, params.mtp_mode == "only" ? lora_param_filter_none : qat_param_filter, &qat_filter, qat_opt_lr_pars, &qat_lr_ctx,
        params.optimizer, LLAMA_LORA_QAT_TYPE_NONE,
        nullptr, nullptr,
        params.grad_checkpoint_interval, params.activation_recompute,
        critical_token_mode_from_string(params.critical_token_mode),
        params.critical_token_weight, params.critical_confidence_threshold,
        critical_weight_shape_from_string(params.critical_weight_shape), params.critical_warmup_steps,
        params.critical_max_fraction, &schedule.step, params.critical_stats_every,
        params.mtp_mode != "only",
        false,
        params.qat_grad_accumulator_cpu,
        (uint32_t) params.grad_accum_steps,
    };
    llama_opt_init(lctx, model, opt_params);
    llama_opt_set_graph_reuse(lctx, params.qat_static_graph);
    if (params.qat_moe_gradient != "auto") llama_opt_set_moe_fused(lctx, params.qat_moe_gradient != "dense");
    llama_opt_set_moe_compact(lctx, params.qat_moe_gradient == "compact");
    if (params.qat_static_graph) {
        LOG_INF("qat_graph: reuse requested; compatible CPU full-sequence single-stream steps are eligible; shape changes rebuild the graph\n");
    }
    if (params.mtp_mode != "only" && llama_opt_qat_state_count(lctx) == 0) {
        LOG_ERR("%s: the training graph contains no trainable %s tensors\n", __func__, params.qat_quant_type.c_str());
        return 1;
    }
    std::vector<ggml_tensor *> adapter_tensors;
    uint64_t selected_bytes = 0;
    for (int64_t i = 0; i < llama_opt_qat_state_count(lctx); ++i) {
        auto * tensor = llama_opt_qat_state_param(lctx, i);
        adapter_tensors.push_back(tensor);
        selected_bytes += ggml_nbytes(tensor);
        LOG_INF("qat_target: %s type=%s shape=[%lld,%lld,%lld,%lld] bytes=%zu\n", tensor->name,
            ggml_type_name(tensor->type), (long long) tensor->ne[0], (long long) tensor->ne[1],
            (long long) tensor->ne[2], (long long) tensor->ne[3], ggml_nbytes(tensor));
    }
    LOG_INF("qat_targets: registered=%zu bytes=%llu\n", adapter_tensors.size(), (unsigned long long) selected_bytes);
    common_qlion_snapshot adapter_base;
    if (!params.qat_adapter_out.empty()) {
        try {
            if (params.mtp_mode == "only" || !params.qlion_adapters.empty()) {
                throw std::runtime_error("adapter export requires training the on-disk base directly");
            }
            const auto aliases = [](const std::string & a, const std::string & b) {
                return std::filesystem::weakly_canonical(a) == std::filesystem::weakly_canonical(b) ||
                    (std::filesystem::exists(a) && std::filesystem::exists(b) && std::filesystem::equivalent(a, b));
            };
            if ((!params.qat_out.empty() && (aliases(params.qat_adapter_out, params.qat_out) || aliases(params.qat_out, params.model.path))) ||
                    aliases(params.qat_adapter_out, params.model.path)) {
                throw std::runtime_error("adapter, full model output and base must be distinct files");
            }
            adapter_base = common_qlion_capture(params.model.path, adapter_tensors);
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s\n", __func__, e.what());
            return 1;
        }
    }
    qat_print_memory(lctx, model_info);

    struct qat_resume_state resume;
    if (!params.qat_resume.empty() && !qat_load_state(lctx, params.qat_resume, params.qat_quant_type, resume, params)) {
        return 1;
    }
//    const int64_t idata_split = train_split_from_val_fraction(ndata, params.val_split);
    if (idata_split <= 0) {
        LOG_ERR("%s: no training windows after validation split\n", __func__);
        return 1;
    }
    if (resume.epoch < 0 || resume.window < 0 || resume.window > idata_split) {
        LOG_ERR("%s: checkpoint dataset position is invalid\n", __func__);
        return 1;
    }
    if (resume.window == idata_split) {
        ++resume.epoch;
        resume.window = 0;
    }
    if (params.lr.epochs > 0 && idata_split > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total training step count overflows int64\n", __func__);
        return 1;
    }
    const int64_t windows_per_update = (int64_t) params.n_parallel * params.grad_accum_steps;
    schedule.total_steps = ((idata_split - 1) / windows_per_update + 1) * params.lr.epochs;
    if (schedule.total_steps <= 0 || params.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: invalid schedule: warmup_steps=%d total_steps=%ld\n",
                __func__, params.warmup_steps, (long) schedule.total_steps);
        return 1;
    }
    schedule.step = resume.schedule_step;
    if (params.lr_scheduler == "cosine" && params.lr_decay_steps > 0 &&
        params.lr_decay_steps <= params.warmup_steps) {
        LOG_ERR("%s: --lr-decay-steps %d must exceed --warmup-steps %d\n",
                __func__, params.lr_decay_steps, params.warmup_steps);
        return 1;
    }
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d decay_steps=%ld total_steps=%ld start_step=%ld lr=%.3g lr_min=%.3g\n",
            __func__, params.lr_scheduler.c_str(), params.warmup_steps,
            (long) (params.lr_decay_steps > 0 ? std::min<int64_t>(params.lr_decay_steps, schedule.total_steps) : schedule.total_steps),
            (long) schedule.total_steps, (long) schedule.step,
            (double) params.lr.lr0, (double) std::max(0.0f, params.lr.lr_min));
    if (params.save_first_at > schedule.step) {
        LOG_INF("%s: first checkpoint will be saved at or after optimizer step %d\n",
                __func__, params.save_first_at);
    }
    mtp_training_state mtp;
    if (!mtp_init_training(params, lctx, schedule, mtp)) {
        return 1;
    }
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval = ggml_opt_result_init();
    const int64_t ubatches_per_window = llama_n_ctx(lctx) / llama_n_ubatch(lctx);
    GGML_ASSERT(ubatches_per_window > 0);
    struct qat_callback_context callback_ctx {
        lctx, mtp.ctx.get(), model, &mtp, &schedule, &params, 0, 0, ubatches_per_window,
        llama_opt_step(params.mtp_mode == "only" ? mtp.ctx.get() : lctx),
        llama_opt_step(params.mtp_mode == "only" ? mtp.ctx.get() : lctx),
        result_eval, idata_split, ndata,
        params.save_first_at > schedule.step
    };
    g_qat_callback = &callback_ctx;
    if (params.shuffle_dataset) {
        for (int64_t epoch = 0; epoch < resume.epoch; ++epoch) {
            llama_opt_dataset_shuffle(lctx, dataset, idata_split);
        }
    }
    if (params.eval_initial) {
        callback_ctx.epoch = resume.epoch;
        qat_run_periodic_eval(dataset, llama_opt_step(params.mtp_mode == "only" ? mtp.ctx.get() : lctx));
    }
    for (params.lr.epoch = (int) resume.epoch; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        const int64_t start = params.lr.epoch == resume.epoch ? resume.window : 0;
        callback_ctx.epoch = params.lr.epoch;
        callback_ctx.window_start = start;
        try {
            llama_opt_epoch_range(lctx, dataset, result_train, result_eval, start, idata_split,
                qat_epoch_callback, ggml_opt_epoch_callback_progress_bar, params.shuffle_dataset);
        } catch (const std::exception & e) {
            LOG_ERR("%s: QAT epoch failed: %s\n", __func__, e.what());
            return 1;
        }
        fprintf(stderr, "\n");
        double train_loss = 0.0;
        double train_unc = 0.0;
        ggml_opt_result_loss(result_train, &train_loss, &train_unc);
        const int64_t optimizer_step = llama_opt_step(
            params.mtp_mode == "only" ? mtp.ctx.get() : lctx);
        LOG_INF("qat_epoch: epoch=%d/%d loss=%.8f uncertainty=%.8f optimizer_step=%ld\n",
            params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc, (long) optimizer_step);
        if ((int64_t) params.lr.epoch == resume.epoch) {
            qat_print_memory(lctx, model_info);
        }
        if (idata_split < ndata) {
            double val_loss = 0.0;
            double val_accuracy = 0.0;
            ggml_opt_result_loss(result_eval, &val_loss, nullptr);
            ggml_opt_result_accuracy(result_eval, &val_accuracy, nullptr);
            LOG_INF("validation: step=%ld epoch=%d loss=%.8f acc=%.4f%%\n",
                (long) optimizer_step, params.lr.epoch + 1, val_loss, 100.0 * val_accuracy);
        }
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    llama_synchronize(lctx);
    bool final_save_ok = true;
    if (params.mtp_mode != "only" && !params.qat_out.empty()) {
        if (!qat_save_model(model, params.model.path, params.qat_out)) {
            final_save_ok = false;
        } else {
            LOG_INF("%s: trained native %s %s model saved to %s\n", __func__, ggml_opt_optimizer_name(params.optimizer), params.qat_quant_type.c_str(), params.qat_out.c_str());
            if (!params.distill_cache.empty()) {
                final_save_ok = qat_save_state(lctx, params.qat_out, params.lr.epochs, 0,
                        schedule.step, params.qat_quant_type, params) && final_save_ok;
            }
        }
    }
    if (!params.qat_adapter_out.empty()) {
        try {
            common_qlion_save(adapter_base, adapter_tensors, params.qat_adapter_out, params.qat_adapter_units);
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s\n", __func__, e.what());
            final_save_ok = false;
        }
    }
    if (mtp.ctx) {
        llama_synchronize(mtp.ctx.get());
        final_save_ok = save_adapter(mtp.tensors, params.mtp_lora_out, mtp.arch, mtp.alpha,
            params.mtp_model) && final_save_ok;
        if (mtp.tensors.buf) ggml_backend_buffer_free(mtp.tensors.buf);
        if (mtp.tensors.ctx) ggml_free(mtp.tensors.ctx);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    ggml_opt_dataset_free(dataset);
    llama_backend_free();
    return final_save_ok ? 0 : 1;
}
#endif
