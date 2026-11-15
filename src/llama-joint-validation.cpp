#include "llama-joint-validation.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-paired-forward.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

llama_opt_joint_validation::llama_opt_joint_validation(llama_context * owner, llama_context * validation) :
    owner(owner), validation(validation) {
    batch = llama_batch_init(validation->n_ctx(), 0, 1);
}

llama_opt_joint_validation::~llama_opt_joint_validation() {
    llama_batch_free(batch);
}

void llama_opt_joint_validation::start(ggml_opt_dataset_t data, int64_t begin, int64_t stop, int64_t step) {
    if (!data || status.active || begin < 0 || begin >= stop || stop > ggml_opt_dataset_ndata(data) ||
        !ggml_opt_dataset_data(data) || !ggml_opt_dataset_labels(data) ||
        ggml_opt_dataset_data(data)->type != GGML_TYPE_I32 || ggml_opt_dataset_labels(data)->type != GGML_TYPE_I32 ||
        ggml_opt_dataset_data(data)->ne[0] != validation->n_ctx() ||
        ggml_opt_dataset_labels(data)->ne[0] != validation->n_ctx() ||
        ggml_opt_dataset_aux_size(data) != validation->n_ctx()*sizeof(uint64_t)) {
        throw std::runtime_error("invalid joint validation range or active snapshot");
    }
    owner->synchronize();
    memory.reset();
    snapshot_buffers.clear();
    snapshot_ctx.reset();
    weights.clear();
    const int64_t count = ggml_opt_qat_state_count(owner->opt_ctx);
    snapshot_ctx.reset(ggml_init({ std::max<int64_t>(1, count)*ggml_tensor_overhead(), nullptr, true }));
    if (!snapshot_ctx) throw std::runtime_error("cannot allocate validation snapshot metadata");
    ggml_backend_buffer_type_t buffer_type = nullptr;
    for (int64_t i = 0; i < count; ++i) {
        auto * source = ggml_opt_qat_state_param(owner->opt_ctx, i);
        const auto type = ggml_backend_buffer_get_type(source->buffer);
        if (buffer_type && buffer_type != type) throw std::runtime_error("joint snapshot requires one parameter buffer type");
        buffer_type = type;
        auto * copy = ggml_dup_tensor(snapshot_ctx.get(), source);
        ggml_set_name(copy, source->name);
        weights.emplace(source, copy);
    }
    if (!buffer_type) throw std::runtime_error("joint validation needs QLion parameters");
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(snapshot_ctx.get(), buffer_type);
    if (!buffer) throw std::runtime_error("cannot allocate validation weight snapshot");
    snapshot_buffers.emplace_back(buffer);
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (const auto & entry : weights) {
        std::vector<uint8_t> bytes(ggml_nbytes(entry.first));
        ggml_backend_tensor_get(entry.first, bytes.data(), 0, bytes.size());
        ggml_backend_tensor_set(entry.second, bytes.data(), 0, bytes.size());
    }
    dataset = data; first = next = begin; end = stop;
    microbatches = 0; loss_sum = 0;
    status = {};
    status.step = step;
    status.total_windows = end-first;
    status.active = true;
}

ggml_cgraph * llama_opt_joint_validation::build() {
    if (!status.active) return nullptr;
    const uint32_t context = validation->n_ctx();
    if (!memory) {
        const auto * tokens = static_cast<const int32_t *>(ggml_opt_dataset_data(dataset)->data) + next*context;
        const auto * targets = static_cast<const int32_t *>(ggml_opt_dataset_labels(dataset)->data) + next*context;
        labels.assign(targets, targets+context);
        teacher_rows.resize(context);
        ggml_opt_dataset_get_batch_host_aux(dataset, teacher_rows.data(), context*sizeof(uint64_t), next);
        supervised = std::count_if(labels.begin(), labels.end(), [](int32_t id) { return id >= 0; });
        batch.n_tokens = context;
        for (uint32_t i = 0; i < context; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = labels[i] >= 0;
        }
        validation->memory->clear(true);
        {
            llama_batch_compat compat(validation, batch);
            if (!validation->balloc->init(*compat.batch_ext, validation->model.vocab, false)) {
                throw std::runtime_error("cannot initialize validation batch");
            }
        }
        memory = validation->memory->init_batch(*validation->balloc, validation->n_ubatch(), true);
        if (!memory || memory->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            throw std::runtime_error("cannot initialize validation KV memory");
        }
    }
    ubatch = memory->get_ubatch();
    output_rows.clear();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) if (ubatch.output[i]) output_rows.push_back(i);
    if (output_rows.empty()) {
        ubatch.output[ubatch.n_tokens-1] = 1;
        output_rows.push_back(ubatch.n_tokens-1);
    }
    validation->n_outputs = output_rows.size();
    if (!memory->apply()) throw std::runtime_error("cannot advance validation KV memory");
    result = validation->get_gf_res_prev();
    result->reset();
    auto params = validation->graph_params(result, ubatch, memory.get(), LLM_GRAPH_TYPE_DEFAULT, false);
    params.sched = owner->sched.get();
    params.backend_cpu = owner->backend_cpu;
    params.cb = owner->graph_get_cb();
    auto * graph = validation->model.build_graph(params);
    set_head_precision(graph);
    auto * ctx = result->get_ctx();
    const int64_t rows = output_rows.size();
    const int32_t k = owner->opt_distillation_top_k;
    ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, rows);
    logp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k+1, rows);
    row_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rows);
    ggml_set_input(ids); ggml_set_input(logp); ggml_set_input(row_weights);
    loss = ggml_opt_distillation_loss(ctx, result->get_logits(), ids, logp, row_weights, owner->opt_distillation_temperature);
    ggml_set_output(loss);
    ggml_build_forward_expand(graph, loss);
    targets = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rows);
    ggml_set_input(targets);
    ncorrect = ggml_count_equal(ctx, ggml_argmax(ctx, result->get_logits()), targets);
    ggml_set_output(ncorrect);
    ggml_build_forward_expand(graph, ncorrect);
    return graph;
}

void llama_opt_joint_validation::set_head_precision(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_MUL_MAT && node->src[0] == validation->model.output && node->src[0]->type == GGML_TYPE_Q4_0) {
            ggml_prec_set_src(node, GGML_PREC_Q4, 0);
        }
    }
}

ggml_context * llama_opt_joint_validation::pairing_context(ggml_cgraph * train, ggml_cgraph * val) {
    size_t candidates = 0;
    for (int i = 0; i < ggml_graph_n_nodes(val); ++i) {
        const auto op = ggml_graph_node(val, i)->op;
        candidates += op == GGML_OP_MUL_MAT || op == GGML_OP_MUL_MAT_ID;
    }
    const size_t capacity = ggml_graph_size(train) + ggml_graph_size(val) + 8*candidates + 64;
    const size_t bytes = 8*candidates*ggml_tensor_overhead() + ggml_graph_overhead_custom(capacity, true);
    paired_ctx.reset();
    paired_meta.resize(bytes);
    paired_ctx.reset(ggml_init({bytes, paired_meta.data(), true}));
    if (!paired_ctx) throw std::runtime_error("cannot allocate paired graph metadata");
    return paired_ctx.get();
}

void llama_opt_joint_validation::set_inputs() {
    if (!status.active || !result) return;
    result->set_inputs(&ubatch);
    const int32_t k = owner->opt_distillation_top_k;
    std::vector<int32_t> indices(output_rows.size()*k);
    std::vector<float> probabilities(output_rows.size()*(k+1));
    std::vector<float> scales(output_rows.size(), 0);
    std::vector<int32_t> classes(output_rows.size(), -1);
    const float count = 1 + (validation->n_ctx()-1)/validation->n_ubatch();
    const float partial = float(ubatch.n_tokens)*count/validation->n_ctx();
    for (size_t row = 0; row < output_rows.size(); ++row) {
        const auto position = ubatch.pos[output_rows[row]];
        if (labels[position] < 0) {
            for (int32_t j = 0; j < k; ++j) indices[row*k+j] = j;
            std::fill_n(probabilities.data()+row*(k+1), k+1, -std::log(float(k+1)));
            continue;
        }
        owner->opt_distillation_callback(owner->opt_distillation_userdata, teacher_rows[position],
                indices.data()+row*k, probabilities.data()+row*(k+1));
        classes[row] = labels[position];
        scales[row] = float(validation->n_ctx())/supervised*output_rows.size()/ubatch.n_tokens*partial;
    }
    ggml_backend_tensor_set(ids, indices.data(), 0, indices.size()*sizeof(int32_t));
    ggml_backend_tensor_set(logp, probabilities.data(), 0, probabilities.size()*sizeof(float));
    ggml_backend_tensor_set(row_weights, scales.data(), 0, scales.size()*sizeof(float));
    ggml_backend_tensor_set(targets, classes.data(), 0, classes.size()*sizeof(int32_t));
}

void llama_opt_joint_validation::collect() {
    if (!status.active || !result) return;
    float value;
    ggml_backend_tensor_get(loss, &value, 0, sizeof(value));
    int64_t correct;
    ggml_backend_tensor_get(ncorrect, &correct, 0, sizeof(correct));
    status.correct += correct;
    for (const auto row : output_rows) status.supervised += labels[ubatch.pos[row]] >= 0;
    loss_sum += value;
    ++microbatches;
    if (!memory->next()) {
        memory.reset();
        ++next;
        ++status.windows;
        if (next == end) {
            status.loss = loss_sum/microbatches;
            status.active = false;
            status.complete = true;
        }
    }
    result = nullptr;
}

bool llama_context::opt_joint_attach(llama_context * other, bool fuse_all) {
    if (!other || other->opt_ctx || &model != &other->model || !opt_ctx || !opt_distillation_callback ||
        n_seq_max() != 1 || other->n_seq_max() != 1 || other->n_ctx() != n_ctx() ||
        other->n_ubatch() > n_ubatch() || opt_joint || other->opt_joint || opt_graph_reuse || opt_mtp_context ||
        (model.arch != LLM_ARCH_GEMMA4 && model.arch != LLM_ARCH_LUMEN4EXP)) {
        throw std::runtime_error("joint validation requires compatible single-sequence Gemma contexts");
    }
    other->synchronize();
    other->sched_new(other->gf_res_reserve->get_max_nodes(), false);
    opt_joint = std::make_shared<llama_opt_joint_validation>(this, other);
    opt_joint->fuse_all = fuse_all;
    other->opt_joint = opt_joint;
    std::weak_ptr<llama_opt_joint_validation> weak = opt_joint;
    other->opt_weight_replace = [weak](ggml_context *, ggml_tensor * weight) {
        auto state = weak.lock();
        if (!state) return weight;
        const auto found = state->weights.find(weight);
        return found == state->weights.end() ? weight : found->second;
    };
    if (std::getenv("QLION_DIAG_JOINT_INDICES")) {
        ggml_backend_sched_set_eval_callback(sched.get(), [](ggml_tensor * node, bool ask, void * opaque) {
            if (node->op != GGML_OP_SET_ROWS || node->src[1]->type != GGML_TYPE_I64) return false;
            auto * state = static_cast<llama_opt_joint_validation *>(opaque);
            bool val = false;
            if (state->result) {
                auto * graph = state->result->get_gf();
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) val |= ggml_graph_node(graph, i) == node;
            }
            if (ask) {
                LLAMA_LOG_ERROR("joint_node: val=%d name=%s type=%s\n", val, node->name, ggml_type_name(node->type));
                return true;
            }
            std::vector<int64_t> indices(ggml_nelements(node->src[1]));
            ggml_backend_tensor_get(node->src[1], indices.data(), 0, indices.size()*sizeof(int64_t));
            const auto bounds = std::minmax_element(indices.begin(), indices.end());
            LLAMA_LOG_ERROR("joint_indices: val=%d name=%s shape=%lld,%lld,%lld index=%lld..%lld input=%d\n", val, node->name,
                (long long)node->ne[0], (long long)node->ne[1], (long long)node->ne[2],
                (long long)*bounds.first, (long long)*bounds.second, node->src[1]->flags);
            return true;
        }, opt_joint.get());
    }
    return true;
}

bool llama_context::opt_joint_start(ggml_opt_dataset_t dataset, int64_t first, int64_t end, int64_t step) {
    if (!opt_joint || opt_joint->owner != this) throw std::runtime_error("joint validation is not attached");
    opt_joint->start(dataset, first, end, step);
    return true;
}

llama_joint_validation_status llama_context::opt_joint_status() const {
    return opt_joint ? opt_joint->status : llama_joint_validation_status{};
}

void llama_context::opt_joint_finish() {
    if (!opt_joint || opt_joint->owner != this) return;
    while (opt_joint->status.active) {
        auto * graph = opt_joint->build();
        ggml_backend_sched_reset(sched.get());
        if (!ggml_backend_sched_alloc_graph(sched.get(), graph)) throw std::runtime_error("cannot allocate final validation graph");
        opt_joint->set_inputs();
        if (ggml_backend_sched_graph_compute(sched.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("final joint validation compute failed");
        }
        opt_joint->collect();
    }
}

void llama_context::opt_joint_detach() {
    if (!opt_joint || opt_joint->owner != this) return;
    synchronize();
    if (std::getenv("QLION_DIAG_JOINT_INDICES")) ggml_backend_sched_set_eval_callback(sched.get(), nullptr, nullptr);
    auto * other = opt_joint->validation;
    other->opt_weight_replace = {};
    other->opt_joint.reset();
    opt_joint.reset();
}

bool llama_opt_joint_validation_attach(llama_context * train, llama_context * validation, bool fuse_all) {
    try { return train && train->opt_joint_attach(validation, fuse_all); }
    catch (const std::exception & e) { LLAMA_LOG_ERROR("joint validation: %s\n", e.what()); return false; }
}

bool llama_opt_joint_validation_start(llama_context * train, ggml_opt_dataset_t data, int64_t first, int64_t end, int64_t step) {
    try { return train && train->opt_joint_start(data, first, end, step); }
    catch (const std::exception & e) { LLAMA_LOG_ERROR("joint validation: %s\n", e.what()); return false; }
}

llama_joint_validation_status llama_opt_joint_validation_status(const llama_context * train) {
    return train ? train->opt_joint_status() : llama_joint_validation_status{};
}

bool llama_opt_joint_validation_finish(llama_context * train) {
    try { if (train) train->opt_joint_finish(); return train != nullptr; }
    catch (const std::exception & e) { LLAMA_LOG_ERROR("joint validation: %s\n", e.what()); return false; }
}

void llama_opt_joint_validation_detach(llama_context * train) {
    if (train) train->opt_joint_detach();
}
