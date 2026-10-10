#include "native-global.h"
#include "global-loss.h"
#include "optimize.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace llama_opt {
static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct native_global::impl {
    const candidate_store & store;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    ggml_context * parameters = nullptr;
    ggml_backend_buffer_t parameter_buffer = nullptr;
    ggml_backend_t cpu = nullptr, device = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_sched_t teacher_sched = nullptr;
    std::vector<ggml_tensor *> weights;
    std::map<std::string, size_t> groups;
    size_t capacity;
    bool float32_math = false;

    impl(const candidate_store & s, size_t n) : store(s), capacity(n) {}

    void init(const std::string & backend, bool use_float32) {
        const size_t n = capacity;
        float32_math = use_float32;
        require(n >= 2 && n <= 512, "global context must be 2..512 tokens");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(store.source_model.c_str(), mp);
        require(model != nullptr, "global source model load failed");
        require(model->arch == LLM_ARCH_GEMMA4 && model->hparams.n_expert == 0, "routed global model adapter is not implemented");
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max<size_t>(128, n);
        cp.n_batch = n;
        cp.n_ubatch = n;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.op_offload = false;
        context = llama_init_from_model(model, cp);
        require(context != nullptr, "global context creation failed");
        cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, 8);
        ggml_backend_cpu_set_use_ref(cpu, float32_math);
        device = backend == "CPU" ? cpu : ggml_backend_init_by_name(backend.c_str(), nullptr);
        require(device != nullptr, "global backend unavailable");
        ggml_backend_t backends[] = { device, cpu };
        sched = ggml_backend_sched_new(backends, nullptr, device == cpu ? 1 : 2, 131072, false, true);
        require(sched != nullptr, "global scheduler creation failed");
        teacher_sched = ggml_backend_sched_new(&cpu, nullptr, 1, 65536, false, true);
        require(teacher_sched != nullptr, "CPU teacher scheduler creation failed");
        ggml_backend_sched_set_weight_streaming(sched, device != cpu);
        parameters = ggml_init({ (store.groups.size() + 1) * ggml_tensor_overhead(), nullptr, true });
        require(parameters != nullptr, "global parameter metadata allocation failed");
        for (size_t i = 0; i < store.groups.size(); ++i) {
            const auto & group = store.groups[i];
            groups.emplace(group.tensor, i);
            auto * weight = ggml_new_tensor_2d(parameters, GGML_TYPE_F32, group.shape[0], group.shape[1]);
            ggml_set_name(weight, group.tensor.c_str());
            ggml_set_param(weight);
            weights.push_back(weight);
        }
        parameter_buffer = ggml_backend_alloc_ctx_tensors(parameters, cpu);
        require(parameter_buffer != nullptr, "global parameter buffer allocation failed");
        ggml_backend_buffer_set_usage(parameter_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    ~impl() {
        if (sched) { ggml_backend_sched_free(sched); }
        if (teacher_sched) { ggml_backend_sched_free(teacher_sched); }
        if (parameter_buffer) { ggml_backend_buffer_free(parameter_buffer); }
        if (parameters) { ggml_free(parameters); }
        if (device && device != cpu) { ggml_backend_free(device); }
        if (cpu) { ggml_backend_free(cpu); }
        if (context) { llama_free(context); }
        if (model) { llama_model_free(model); }
    }

    void set_weights(const std::vector<double> & soft, const std::vector<size_t> & hard) {
        size_t offset = 0;
        require(hard.empty() || hard.size() == store.groups.size(), "hard assignment shape mismatch");
        for (size_t i = 0; i < store.groups.size(); ++i) {
            const auto & group = store.groups[i];
            require(offset + group.choices.size() <= soft.size(), "precision probability shape mismatch");
            require(hard.empty() || hard[i] < group.choices.size(), "hard assignment outside candidates");
            for (int64_t row = 0; row < group.shape[1]; row += 64) {
                const auto count = std::min<int64_t>(64, group.shape[1] - row);
                std::vector<double> accumulated(count * group.shape[0], 0);
                for (size_t k = 0; k < group.choices.size(); ++k) {
                    const double probability = hard.empty() ? soft[offset + k] : (k == hard[i] ? 1.0 : 0.0);
                    if (probability == 0) { continue; }
                    const auto tile = store.decode_tile(i, k, row, count);
                    for (size_t j = 0; j < tile.size(); ++j) { accumulated[j] += probability * tile[j]; }
                }
                std::vector<float> mixed(accumulated.begin(), accumulated.end());
                ggml_backend_tensor_set(weights[i], mixed.data(), row * group.shape[0] * sizeof(float), mixed.size() * sizeof(float));
            }
            offset += group.choices.size();
        }
        require(offset == soft.size(), "precision probability shape mismatch");
    }

    global_result run(const std::vector<int32_t> & tokens, const std::vector<float> & teacher,
        bool kl, const std::vector<double> & soft, const std::vector<size_t> & hard, double tau,
        bool derivative, bool reference) {
        require(tokens.size() >= 2 && tokens.size() <= capacity, "tokens exceed global window");
        ggml_backend_cpu_set_use_ref(cpu, float32_math && !reference);
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_reset(teacher_sched);
        llama_memory_clear(llama_get_memory(context), false);
        if (!reference) { set_weights(soft, hard); }
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        batch.n_tokens = tokens.size();
        for (size_t j = 0; j < tokens.size(); ++j) {
            batch.token[j] = tokens[j]; batch.pos[j] = j;
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = true;
        }
        llama_batch_compat compat(context, batch);
        llama_batch_free(batch);
        llama_batch_allocr allocator(model->hparams.n_pos_per_embd());
        require(allocator.init(*compat.batch_ext, model->vocab, true), "invalid global token batch");
        const auto ubatch = allocator.split_simple(tokens.size());
        llm_graph_result graph_result(65536);
        std::vector<bool> used(store.groups.size(), false);
        auto replace = [&](ggml_context * ctx, ggml_tensor * weight) {
            if (reference) { return weight; }
            const auto found = groups.find(weight->name);
            if (found == groups.end()) {
                return float32_math && (weight->type == GGML_TYPE_BF16 || weight->type == GGML_TYPE_F16)
                    ? ggml_cast(ctx, weight, GGML_TYPE_F32) : weight;
            }
            used[found->second] = true;
            require(ggml_are_same_shape(weight, weights[found->second]), "global candidate shape mismatch");
            return weights[found->second];
        };
        auto active_sched = reference ? teacher_sched : sched;
        auto * gf = context->build_training_graph(graph_result, ubatch, active_sched, replace, float32_math && !reference);
        if (float32_math && !reference) {
            for (int j = 0; j < ggml_graph_n_nodes(gf); ++j) {
                auto * node = ggml_graph_node(gf, j);
                if (node->op == GGML_OP_MUL_MAT) { ggml_prec_set_acc(node, GGML_PREC_F32); }
            }
        }
        if (float32_math && !reference && derivative) {
            for (int j = 0; j < ggml_graph_n_nodes(gf); ++j) {
                const auto * node = ggml_graph_node(gf, j);
                if (node->op == GGML_OP_MUL_MAT && (node->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_F32)) {
                    std::fprintf(stderr, "validation matmul %s: %s %s\n", node->name, ggml_type_name(node->src[0]->type), ggml_type_name(node->src[1]->type));
                }
                if (node->op == GGML_OP_GLU || (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_GELU)) {
                    std::fprintf(stderr, "validation activation %s: %s\n", node->name, ggml_op_name(node->op));
                }
            }
        }
        if (!reference) {
            require(std::all_of(used.begin(), used.end(), [](bool x) { return x; }), "candidate tensor is not used by model graph");
        }
        require(ggml_backend_sched_alloc_graph(active_sched, gf), "global forward allocation failed");
        graph_result.set_inputs(&ubatch);
        require(ggml_backend_sched_graph_compute(active_sched, gf) == GGML_STATUS_SUCCESS, "global forward failed");
        auto * logits_tensor = graph_result.get_logits();
        global_result result;
        result.logits.resize(ggml_nelements(logits_tensor));
        ggml_backend_tensor_get(logits_tensor, result.logits.data(), 0, result.logits.size() * sizeof(float));
        const size_t vocab = logits_tensor->ne[0];
        const std::vector<int32_t> targets(tokens.begin() + 1, tokens.end());
        std::vector<float> shifted(result.logits.begin(), result.logits.begin() + targets.size() * vocab);
        const auto loss = token_loss(shifted, targets, teacher, vocab, kl);
        result.loss = loss.value;
        if (!derivative) { ggml_backend_sched_reset(active_sched); return result; }

        ggml_backend_sched_reset(sched);
        graph_result.reset();
        gf = context->build_training_graph(graph_result, ubatch, sched, replace, float32_math && !reference);
        if (float32_math) {
            for (int j = 0; j < ggml_graph_n_nodes(gf); ++j) {
                auto * node = ggml_graph_node(gf, j);
                if (node->op == GGML_OP_MUL_MAT) { ggml_prec_set_acc(node, GGML_PREC_F32); }
            }
        }
        auto * seed = ggml_new_tensor_2d(graph_result.get_ctx(), GGML_TYPE_F32, vocab, tokens.size());
        ggml_set_input(seed);
        auto * seeded_logits = ggml_mul(graph_result.get_ctx(), graph_result.get_logits(), seed);
        ggml_build_forward_expand(gf, seeded_logits);
        auto params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
        params.build_type = GGML_OPT_BUILD_TYPE_GRAD;
        params.activation_recompute = false;
        params.fused_backward = false;
        auto optimizer = ggml_opt_init(params);
        ggml_opt_prepare_alloc(optimizer, graph_result.get_ctx(), gf, graph_result.get_inp_tokens(), seeded_logits);
        ggml_opt_alloc(optimizer, true);
        std::vector<ggml_tensor *> gradients;
        for (auto * weight : weights) {
            auto * grad = ggml_opt_grad(optimizer, weight);
            require(grad != nullptr, "global weight gradient is disconnected");
            require(ggml_is_contiguous(grad), "global weight gradient has unsupported strides");
            gradients.push_back(grad);
        }
        graph_result.set_inputs(&ubatch);
        std::vector<float> seed_data(tokens.size() * vocab, 0);
        std::copy(loss.logits_gradient.begin(), loss.logits_gradient.end(), seed_data.begin());
        ggml_backend_tensor_set(seed, seed_data.data(), 0, seed_data.size() * sizeof(float));
        ggml_opt_eval(optimizer, nullptr);
        std::vector<float> replay_logits(result.logits.size());
        ggml_backend_tensor_get(graph_result.get_logits(), replay_logits.data(), 0, replay_logits.size() * sizeof(float));
        for (size_t j = 0; j < replay_logits.size(); ++j) {
            result.forward_recompute_max_error = std::max(result.forward_recompute_max_error,
                std::abs(double(replay_logits[j]) - result.logits[j]));
        }
        std::fprintf(stderr, "global forward replay max error %.9g\n", result.forward_recompute_max_error);
        size_t offset = 0;
        for (size_t i = 0; i < store.groups.size(); ++i) {
            const auto & group = store.groups[i];
            std::vector<double> option_grad(group.choices.size(), 0);
            for (int64_t row = 0; row < group.shape[1]; row += 64) {
                const int64_t count = std::min<int64_t>(64, group.shape[1] - row);
                std::vector<float> dw(count * group.shape[0]);
                ggml_backend_tensor_get(gradients[i], dw.data(), row * group.shape[0] * sizeof(float), dw.size() * sizeof(float));
                require(std::all_of(dw.begin(), dw.end(), [](float d) { return std::isfinite(d); }), "non-finite global weight gradient");
                for (size_t k = 0; k < group.choices.size(); ++k) {
                    const auto tile = store.decode_tile(i, k, row, count);
                    for (size_t j = 0; j < tile.size(); ++j) { option_grad[k] += double(dw[j]) * tile[j]; }
                }
            }
            const std::vector<double> p(soft.begin() + offset, soft.begin() + offset + group.choices.size());
            const auto grad = precision_vjp(p, option_grad, tau);
            result.gradient.insert(result.gradient.end(), grad.begin(), grad.end());
            offset += group.choices.size();
        }
        result.recompute_regions = ggml_opt_recompute_regions(optimizer);
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            result.compute_bytes += ggml_backend_sched_get_buffer_size(sched, ggml_backend_sched_get_backend(sched, i));
        }
        ggml_backend_sched_reset(sched);
        ggml_opt_free(optimizer);
        return result;
    }
};

native_global::native_global(const candidate_store & store, size_t context, const std::string & backend, bool float32_math) :
    state(new impl(store, context)) { state->init(backend, float32_math); }
native_global::~native_global() = default;

std::vector<float> native_global::reference_logits(const std::vector<int32_t> & tokens, bool cached) {
    if (!cached) { return state->run(tokens, {}, false, {}, {}, 1, false, true).logits; }
    llama_memory_clear(llama_get_memory(state->context), true);
    auto batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = tokens.size();
    for (size_t j = 0; j < tokens.size(); ++j) {
        batch.token[j] = tokens[j]; batch.pos[j] = j;
        batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = true;
    }
    const int status = llama_decode(state->context, batch);
    llama_batch_free(batch);
    require(status == 0, "teacher decode failed");
    llama_synchronize(state->context);
    const size_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(state->model));
    const float * logits = llama_get_logits(state->context);
    require(logits != nullptr, "teacher logits missing");
    return std::vector<float>(logits, logits + tokens.size() * vocab);
}

global_result native_global::evaluate(const std::vector<int32_t> & tokens, const std::vector<float> & teacher,
    bool kl, const std::vector<double> & soft, const std::vector<size_t> & hard, double tau, bool gradient) {
    return state->run(tokens, teacher, kl, soft, hard, tau, gradient, false);
}
} // namespace llama_opt
