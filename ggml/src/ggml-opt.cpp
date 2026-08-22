#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <vector>
#include <unordered_map>

struct ggml_opt_dataset {
    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    struct ggml_tensor    * data   = nullptr;
    struct ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;
    size_t  nbs_aux     = 0;

    std::vector<int64_t> permutation;
    std::vector<uint8_t> aux;
};

struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    enum ggml_opt_loss_type    loss_type;
    enum ggml_opt_build_type   build_type;
    enum ggml_opt_build_type   build_type_alloc;

    struct ggml_tensor * inputs  = nullptr;
    struct ggml_tensor * outputs = nullptr;
    struct ggml_tensor * labels  = nullptr;
    struct ggml_tensor * sparse_targets = nullptr;
    struct ggml_tensor * sparse_weights = nullptr;
    struct ggml_tensor * critical_span_weights   = nullptr;
    struct ggml_tensor * critical_reward_weights = nullptr;
    struct ggml_tensor * critical_warmup_scale   = nullptr;
    struct ggml_tensor * critical_selected       = nullptr;
    struct ggml_tensor * critical_effective_weights = nullptr;
    struct ggml_tensor * critical_unweighted_loss = nullptr;

    struct ggml_tensor * loss     = nullptr;
    struct ggml_tensor * pred     = nullptr;
    struct ggml_tensor * ncorrect = nullptr;

    struct ggml_cgraph * gf      = nullptr;
    struct ggml_cgraph * gb_grad = nullptr;
    struct ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<struct ggml_tensor *> grad_accs;
    std::vector<struct ggml_tensor *> grad_m;
    std::vector<struct ggml_tensor *> grad_v;
    std::vector<ggml_backend_buffer_t> bufs_momenta;  // per-param moment buffers (one per param node)
    std::vector<struct ggml_context *> ctxs_momenta;  // corresponding ggml contexts (keep alive for tensor metadata)

    struct quantized_state {
        std::vector<uint8_t> m;
        std::vector<uint8_t> v;
        int64_t ne = 0;
        int64_t ne_padded = 0;
    };
    std::vector<quantized_state> quantized_states;
    std::vector<struct ggml_tensor *> quantized_params;
    std::vector<struct ggml_tensor *> quantized_grads;
    std::vector<struct ggml_tensor *> qat_params;
    std::vector<struct ggml_tensor *> qat_momentum;
    std::vector<struct ggml_tensor *> qat_residual;
    std::vector<struct ggml_tensor *> qat_grad_accumulator;
    std::vector<std::vector<struct ggml_tensor *>> qat_aliases;
    std::vector<std::vector<struct ggml_tensor *>> qat_pending_grads;
    std::vector<size_t> qat_expected_grads;
    std::set<struct ggml_tensor *> qat_forward_nodes;
    std::vector<ggml_backend_buffer_t> qat_buffers;
    std::vector<struct ggml_context *> qat_contexts;

    //
    // QLion dynamic-graph dependency-plan cache.
    //
    // Graph tensor pointers are rebuilt every step,
    // but fixed-shape training keeps node ordering/topology stable.
    // Cache node indices rather than tensor pointers.
    //
    std::unordered_map<struct ggml_tensor *, size_t> qat_alias_to_state;

    std::vector<std::vector<int32_t> > qat_dependency_plan;

    uint64_t qat_dependency_signature = 0;

    bool qat_dependency_plan_ready = false;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    int32_t grad_checkpoint_interval = 0;
    bool    loss_per_datapoint = false;
    bool    critical_token_weighting = false;
    bool    critical_confidence_weighting = false;
    float   critical_token_weight = 1.0f;
    float   critical_confidence_threshold = 0.25f;
    bool    critical_weight_linear = false;
    bool    sparse_labels = false;
    bool    fused_backward = false;
    int32_t critical_max_tokens = -1;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    void *                        get_opt_pars_ud = nullptr;
    struct ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

static bool ggml_opt_optimizer_is_adamw(enum ggml_opt_optimizer_type optimizer) {
    return optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW ||
        optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW_F16 ||
        optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0 ||
        optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q6_K ||
        optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW_IQ4_NL;
}

static bool ggml_opt_qat_same_logical_param(
        const ggml_tensor * a,
        const ggml_tensor * b) {

    if (!a || !b) {
        return false;
    }

    return
        a->type == b->type &&
        strcmp(a->name, b->name) == 0 &&
        memcmp(
            a->ne,
            b->ne,
            sizeof(a->ne)
        ) == 0;
}


static int ggml_opt_qat_backend_priority(
        const ggml_tensor * tensor) {

    if (!tensor ||
        !tensor->buffer) {
        return -1;
    }

    //
    // IMPORTANT:
    // CUDA_Host belongs to CUDA0 and therefore its device type is GPU,
    // but the actual storage is still host memory.
    //
    // Check physical buffer residency FIRST.
    //
    if (ggml_backend_buffer_is_host(
            tensor->buffer)) {
        return 0;
    }

    ggml_backend_buffer_type_t buft =
        ggml_backend_buffer_get_type(
            tensor->buffer
        );

    if (!buft) {
        return -1;
    }

    ggml_backend_dev_t dev =
        ggml_backend_buft_get_device(
            buft
        );

    if (!dev) {
        return -1;
    }

    switch (ggml_backend_dev_type(dev)) {

        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return 100;

        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return 90;

        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return 50;

        case GGML_BACKEND_DEVICE_TYPE_META:
            return 40;

        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return 0;

        default:
            return -1;
    }
}

void ggml_opt_qat_register_param(ggml_opt_context_t opt_ctx, struct ggml_tensor * param) {
    GGML_ASSERT(opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_QLION_QAT);
    GGML_ASSERT(param && (param->type == GGML_TYPE_MXFP4 || param->type == GGML_TYPE_Q4_0));
    GGML_ASSERT(ggml_is_contiguous(param));
    for (size_t i = 0; i < opt_ctx->qat_params.size(); ++i) {
        struct ggml_tensor * canonical = opt_ctx->qat_params[i];
        if (canonical == param) {
            return;
        }
        if (canonical->type == param->type &&
            strcmp(canonical->name, param->name) == 0 &&
            memcmp(
                canonical->ne,
                param->ne,
                sizeof(param->ne)
            ) == 0) {

            //
            // Register this tensor as another physical alias.
            //
            if (std::find(
                    opt_ctx->qat_aliases[i].begin(),
                    opt_ctx->qat_aliases[i].end(),
                    param
                ) == opt_ctx->qat_aliases[i].end()) {

                opt_ctx->qat_aliases[i].push_back(
                    param
                );
            }

            const int old_priority =
                ggml_opt_qat_backend_priority(
                    canonical
                );

            const int new_priority =
                ggml_opt_qat_backend_priority(
                    param
                );

            //
            // The first alias is not necessarily the best optimizer location.
            //
            // Example for tied Gemma embedding:
            //
            //   token_embd alias 0 -> CUDA_Host
            //   token_embd alias 1 -> CUDA0
            //
            // If a better resident alias appears later, promote it to canonical
            // and move the QLion states to the same buffer type.
            //
            if (new_priority > old_priority) {

                ggml_backend_buffer_type_t param_buft =
                    param->buffer
                        ? ggml_backend_buffer_get_type(
                            param->buffer
                        )
                        : ggml_backend_cpu_buffer_type();

                GGML_ASSERT(param_buft);

                //
                // Create replacement state metadata.
                //
                const size_t n_state_tensors = opt_ctx->opt_period > 1 ? 3 : 2;
                struct ggml_init_params state_params = {
                    /* .mem_size   = */
                        n_state_tensors * ggml_tensor_overhead(),
                    /* .mem_buffer = */
                        nullptr,
                    /* .no_alloc   = */
                        true,
                };

                struct ggml_context * new_state_ctx =
                    ggml_init(
                        state_params
                    );

                GGML_ASSERT(
                    new_state_ctx
                );

                struct ggml_tensor * new_momentum =
                    ggml_new_tensor(
                        new_state_ctx,
                        GGML_TYPE_Q8_0,
                        GGML_MAX_DIMS,
                        param->ne
                    );

                struct ggml_tensor * new_residual =
                    ggml_new_tensor(
                        new_state_ctx,
                        GGML_TYPE_Q4_0,
                        GGML_MAX_DIMS,
                        param->ne
                    );

                struct ggml_tensor * new_grad_accumulator = opt_ctx->opt_period > 1
                    ? ggml_new_tensor(new_state_ctx, GGML_TYPE_Q8_0, GGML_MAX_DIMS, param->ne)
                    : nullptr;

                ggml_format_name(
                    new_momentum,
                    "QLion Q8_0 momentum for %s",
                    param->name
                );

                ggml_format_name(
                    new_residual,
                    "QLion Q4_0 residual for %s",
                    param->name
                );

                if (new_grad_accumulator) {
                    ggml_format_name(new_grad_accumulator, "QLion Q8_0 gradient accumulator for %s", param->name);
                }

                ggml_backend_buffer_t new_state_buffer =
                    ggml_backend_alloc_ctx_tensors_from_buft(
                        new_state_ctx,
                        param_buft
                    );

                GGML_ASSERT(
                    new_state_buffer
                );

                //
                // Preserve existing state.
                //
                // During normal opt_init this is currently all-zero,
                // but copying makes promotion safe even if registration
                // happens after state initialization in another caller.
                //
                ggml_backend_tensor_copy(
                    opt_ctx->qat_momentum[i],
                    new_momentum
                );

                ggml_backend_tensor_copy(
                    opt_ctx->qat_residual[i],
                    new_residual
                );

                if (new_grad_accumulator) {
                    ggml_backend_tensor_copy(opt_ctx->qat_grad_accumulator[i], new_grad_accumulator);
                }

                GGML_LOG_INFO(
                    "QLion QAT: promoting canonical %s: %s -> %s\n",
                    param->name,

                    canonical->buffer
                        ? ggml_backend_buffer_name(
                            canonical->buffer
                        )
                        : "(null)",

                    param->buffer
                        ? ggml_backend_buffer_name(
                            param->buffer
                        )
                        : "(null)"
                );

                //
                // Old state can now be released.
                //
                ggml_backend_buffer_free(
                    opt_ctx->qat_buffers[i]
                );

                ggml_free(
                    opt_ctx->qat_contexts[i]
                );

                //
                // Install new canonical + states.
                //
                opt_ctx->qat_params[i] =
                    param;

                opt_ctx->qat_momentum[i] =
                    new_momentum;

                opt_ctx->qat_residual[i] =
                    new_residual;

                opt_ctx->qat_grad_accumulator[i] =
                    new_grad_accumulator;

                opt_ctx->qat_buffers[i] =
                    new_state_buffer;

                opt_ctx->qat_contexts[i] =
                    new_state_ctx;

                //
                // Tied token embedding:
                //
                // Keep distinct tensor metadata objects so autodiff still sees
                // two parameter aliases, but make both aliases reference the
                // same GPU-resident weight storage.
                //
                // This removes the full CUDA -> CUDA_Host synchronization after
                // every QLion step.
                //
                if (
                    strcmp(param->name, "token_embd.weight") == 0 &&
                    canonical->buffer &&
                    param->buffer &&
                    ggml_backend_buffer_is_host(canonical->buffer) &&
                    !ggml_backend_buffer_is_host(param->buffer) &&
                    canonical->type == param->type &&
                    ggml_are_same_shape(canonical, param) &&
                    ggml_are_same_layout(canonical, param)
                ) {

                    GGML_LOG_INFO(
                        "QLion QAT: binding host tied alias %s to canonical device storage (%s -> %s)\n",
                        param->name,
                        ggml_backend_buffer_name(canonical->buffer),
                        ggml_backend_buffer_name(param->buffer)
                    );

                    canonical->buffer =
                        param->buffer;

                    canonical->data =
                        param->data;
                }
            }
            return;
        }
    }
    ggml_backend_buffer_type_t param_buft = param->buffer
        ? ggml_backend_buffer_get_type(param->buffer)
        : ggml_backend_cpu_buffer_type();
    const size_t n_state_tensors = opt_ctx->opt_period > 1 ? 3 : 2;
    struct ggml_init_params state_params = {
        /*.mem_size   =*/ n_state_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * state_ctx = ggml_init(state_params);
    struct ggml_tensor * momentum = ggml_new_tensor(state_ctx, GGML_TYPE_Q8_0, GGML_MAX_DIMS, param->ne);
    struct ggml_tensor * residual = ggml_new_tensor(state_ctx, GGML_TYPE_Q4_0, GGML_MAX_DIMS, param->ne);
    struct ggml_tensor * grad_accumulator = opt_ctx->opt_period > 1
        ? ggml_new_tensor(state_ctx, GGML_TYPE_Q8_0, GGML_MAX_DIMS, param->ne)
        : nullptr;
    ggml_format_name(momentum, "QLion Q8_0 momentum for %s", param->name);
    ggml_format_name(residual, "QLion Q4_0 residual for %s", param->name);
    if (grad_accumulator) {
        ggml_format_name(grad_accumulator, "QLion Q8_0 gradient accumulator for %s", param->name);
    }
    ggml_backend_buffer_t state_buffer = ggml_backend_alloc_ctx_tensors_from_buft(state_ctx, param_buft);
    GGML_ASSERT(state_buffer);
    ggml_backend_buffer_clear(state_buffer, 0);
    opt_ctx->qat_params.push_back(param);
    opt_ctx->qat_momentum.push_back(momentum);
    opt_ctx->qat_residual.push_back(residual);
    opt_ctx->qat_grad_accumulator.push_back(grad_accumulator);
    opt_ctx->qat_aliases.push_back({ param });
    opt_ctx->qat_buffers.push_back(state_buffer);
    opt_ctx->qat_contexts.push_back(state_ctx);
}

static enum ggml_type ggml_opt_optimizer_state_type(enum ggml_opt_optimizer_type optimizer) {
    switch (optimizer) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_F16:   return GGML_TYPE_F16;
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0:  return GGML_TYPE_Q8_0;
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q6_K:  return GGML_TYPE_Q6_K;
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_IQ4_NL:return GGML_TYPE_IQ4_NL;
        default:                                  return GGML_TYPE_COUNT;
    }
}

static bool ggml_opt_has_device_q8_adamw(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (!dev) {
        return false;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) {
        return false;
    }

    const char * name = ggml_backend_reg_name(reg);
    return strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0;
}

static void ggml_opt_step_adamw_quantized(ggml_opt_context_t opt_ctx, const ggml_opt_optimizer_params & opt_pars) {
    const enum ggml_type state_type = ggml_opt_optimizer_state_type(opt_ctx->optimizer);
    const enum ggml_type state_type_v = state_type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    GGML_ASSERT(state_type != GGML_TYPE_COUNT);

    const ggml_type_traits * traits = ggml_get_type_traits(state_type);
    const ggml_type_traits * traits_v = ggml_get_type_traits(state_type_v);
    GGML_ASSERT(traits->to_float && traits->from_float_ref && traits_v->to_float && traits_v->from_float_ref);

    const float beta1h = 1.0f/(1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
    const float beta2h = 1.0f/(1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));
    const float keep   = 1.0f - opt_pars.adamw.alpha*opt_pars.adamw.wd;
    const float v_floor = state_type_v == GGML_TYPE_Q8_0 ? 8.0e-6f : 1.0e-6f;

    for (size_t i = 0; i < opt_ctx->quantized_states.size(); ++i) {
        ggml_opt_context::quantized_state & state = opt_ctx->quantized_states[i];
        ggml_tensor * param = opt_ctx->quantized_params[i];
        ggml_tensor * grad  = opt_ctx->quantized_grads[i];
        GGML_ASSERT(param && grad);
        GGML_ASSERT(param->type == GGML_TYPE_F32 && grad->type == GGML_TYPE_F32);

        std::vector<float> weight(state.ne_padded, 0.0f);
        std::vector<float> gradient(state.ne_padded, 0.0f);
        std::vector<float> m(state.ne_padded);
        std::vector<float> v(state.ne_padded);
        ggml_backend_tensor_get(param, weight.data(), 0, state.ne*sizeof(float));
        ggml_backend_tensor_get(grad, gradient.data(), 0, state.ne*sizeof(float));
        traits->to_float(state.m.data(), m.data(), state.ne_padded);
        traits_v->to_float(state.v.data(), v.data(), state.ne_padded);

        for (int64_t j = 0; j < state.ne; ++j) {
            float g = gradient[j];
            if (!std::isfinite(g)) {
                g = 0.0f;
            }
            if (opt_pars.adamw.gclip > 0.0f) {
                g = fmaxf(-opt_pars.adamw.gclip, fminf(opt_pars.adamw.gclip, g));
            }
            m[j] = m[j]*opt_pars.adamw.beta1 + g*(1.0f - opt_pars.adamw.beta1);
            v[j] = fmaxf(v_floor, v[j]*opt_pars.adamw.beta2 + g*g*(1.0f - opt_pars.adamw.beta2));
            const float update = opt_pars.adamw.alpha*(m[j]*beta1h)/(sqrtf(v[j]*beta2h) + opt_pars.adamw.eps);
            if (std::isfinite(update) && std::isfinite(weight[j])) {
                weight[j] = weight[j]*keep - update;
            }
        }

        traits->from_float_ref(m.data(), state.m.data(), state.ne_padded);
        traits_v->from_float_ref(v.data(), state.v.data(), state.ne_padded);
        ggml_backend_tensor_set(param, weight.data(), 0, state.ne*sizeof(float));
    }
}

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

void ggml_opt_dataset_set_aux(ggml_opt_dataset_t dataset, const void * data, size_t nbytes_per_datapoint) {
    if (!data || nbytes_per_datapoint == 0) {
        dataset->aux.clear();
        dataset->nbs_aux = 0;
        return;
    }
    dataset->nbs_aux = nbytes_per_datapoint * dataset->ndata_shard;
    dataset->aux.resize(nbytes_per_datapoint * dataset->ndata);
    memcpy(dataset->aux.data(), data, dataset->aux.size());
}

size_t ggml_opt_dataset_aux_size(ggml_opt_dataset_t dataset) {
    return dataset->nbs_aux / dataset->ndata_shard;
}

void ggml_opt_dataset_get_batch_host_aux(
        ggml_opt_dataset_t dataset,
        void             * data_batch,
        size_t             nb_data_batch,
        int64_t            ibatch) {
    GGML_ASSERT(dataset->nbs_aux > 0);
    GGML_ASSERT(nb_data_batch % dataset->nbs_aux == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_aux;
    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];
        memcpy((char *) data_batch + ishard_batch*dataset->nbs_aux,
               dataset->aux.data() + ishard*dataset->nbs_aux,
               dataset->nbs_aux);
    }
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data       = (const char *) dataset->data->data + ishard      *dataset->nbs_data;
        char       * ptr_data_batch = (char       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels       = (const char *) dataset->labels->data + ishard      *dataset->nbs_labels;
        char       * ptr_labels_batch = (char       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;
    result.adamw.gclip = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    result.qlion_qat.alpha = 1e-4f;
    result.qlion_qat.beta  = 0.9f;
    result.qlion_qat.wd    = 0.0f;
    result.qlion_qat.gclip = 0.0f;
    result.qlion_qat.fast_state_scale = false;

    return result;
}


struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct ggml_opt_optimizer_params *) userdata);
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        enum ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars              =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud          =*/ nullptr,
        /*grad_checkpoint_interval =*/ 0,
        /*critical_token_weighting =*/ false,
        /*critical_confidence_weighting =*/ false,
        /*critical_token_weight    =*/ 1.0f,
        /*critical_confidence_threshold =*/ 0.25f,
        /*critical_weight_linear   =*/ false,
        /*sparse_labels            =*/ false,
        /*fused_backward           =*/ false,
        /*optimizer                =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static bool ggml_opt_depends_on_qat_alias(
        struct ggml_tensor * tensor,
        const std::vector<struct ggml_tensor *> & aliases,
        const std::set<struct ggml_tensor *> & forward_nodes,
        std::map<struct ggml_tensor *, bool> & memo) {
    if (std::find(aliases.begin(), aliases.end(), tensor) != aliases.end()) {
        return true;
    }
    if (forward_nodes.find(tensor) != forward_nodes.end()) {
        return false;
    }
    const auto cached = memo.find(tensor);
    if (cached != memo.end()) {
        return cached->second;
    }
    bool result = false;
    for (struct ggml_tensor * src : tensor->src) {
        if (src && ggml_opt_depends_on_qat_alias(src, aliases, forward_nodes, memo)) {
            result = true;
            break;
        }
    }
    memo[tensor] = result;
    return result;
}

static inline void ggml_opt_qat_hash_mix(
        uint64_t & h,
        uint64_t value) {

    h ^= value;
    h *= 1099511628211ULL;
}


static uint64_t ggml_opt_qat_graph_signature(
        const ggml_opt_context_t opt_ctx) {

    const ggml_cgraph * gf =
        opt_ctx->gf;

    uint64_t h =
        1469598103934665603ULL;

    ggml_opt_qat_hash_mix(
        h,
        (uint64_t) gf->n_nodes
    );

    ggml_opt_qat_hash_mix(
        h,
        (uint64_t)
            opt_ctx->critical_max_tokens
    );

    ggml_opt_qat_hash_mix(
        h,
        (uint64_t)
            opt_ctx->critical_token_weighting
    );

    ggml_opt_qat_hash_mix(
        h,
        (uint64_t)
            opt_ctx->critical_confidence_weighting
    );

    ggml_opt_qat_hash_mix(
        h,
        (uint64_t)
            opt_ctx->sparse_labels
    );

    ggml_opt_qat_hash_mix(
        h,
        (uint64_t)
            opt_ctx->fused_backward
    );

    for (int i = 0;
         i < gf->n_nodes;
         ++i) {

        const ggml_tensor * node =
            gf->nodes[i];

        ggml_opt_qat_hash_mix(
            h,
            (uint64_t) node->op
        );

        ggml_opt_qat_hash_mix(
            h,
            (uint64_t) node->type
        );

        ggml_opt_qat_hash_mix(
            h,
            (uint64_t) (
                node->flags &
                (
                    GGML_TENSOR_FLAG_PARAM |
                    GGML_TENSOR_FLAG_INPUT |
                    GGML_TENSOR_FLAG_OUTPUT
                )
            )
        );

        for (int d = 0;
             d < GGML_MAX_DIMS;
             ++d) {

            ggml_opt_qat_hash_mix(
                h,
                (uint64_t)
                    node->ne[d]
            );
        }

        //
        // Parameter names are stable across
        // rebuilt dynamic graphs.
        //
        if (node->flags &
            GGML_TENSOR_FLAG_PARAM) {

            for (const char * p =
                     node->name;
                 *p;
                 ++p) {

                ggml_opt_qat_hash_mix(
                    h,
                    (uint8_t) *p
                );
            }
        }
    }

    return h;
}
static struct ggml_tensor *
ggml_opt_qlion_qat_backward_callback(
        struct ggml_context * ctx,
        struct ggml_tensor  * param,
        struct ggml_tensor  * grad,
        void                * userdata) {

    ggml_opt_context_t opt_ctx =
        (ggml_opt_context_t)
            userdata;

    const auto found =
        opt_ctx->
            qat_alias_to_state.find(
                param
            );

    if (found ==
        opt_ctx->
            qat_alias_to_state.end()) {

        return nullptr;
    }

    const size_t i =
        found->second;
    
    ggml_tensor * canonical_param =
        opt_ctx->qat_params[i];

    GGML_ASSERT(
        canonical_param
    );

    static bool dumped_qat_placement = false;

if (!dumped_qat_placement &&
    strcmp(
        canonical_param->name,
        "token_embd.weight"
    ) == 0) {

    dumped_qat_placement = true;

    auto dump_tensor_backend =
        [](const char * label,
           const ggml_tensor * t) {

        if (!t) {
            fprintf(
                stderr,
                "%s: NULL\n",
                label
            );
            return;
        }

        ggml_backend_buffer_type_t buft =
            t->buffer
                ? ggml_backend_buffer_get_type(
                    t->buffer
                )
                : nullptr;

        ggml_backend_dev_t dev =
            buft
                ? ggml_backend_buft_get_device(
                    buft
                )
                : nullptr;

        //
        // Dump tensor backend information
        //
        fprintf(
            stderr,
            "%s:"
            " tensor=%p"
            " data=%p"
            " buffer=%s"
            " buft=%s"
            " device=%s"
            " dev_type=%d\n",

            label,

            (const void *) t,
            t->data,

            t->buffer
                ? ggml_backend_buffer_name(
                    t->buffer
                )
                : "(null)",

            buft
                ? ggml_backend_buft_name(
                    buft
                )
                : "(null)",

            dev
                ? ggml_backend_dev_name(
                    dev
                )
                : "(none)",

            dev
                ? (int)
                    ggml_backend_dev_type(
                        dev
                    )
                : -1
        );
    };

    fprintf(
        stderr,
        "\n=== REAL QAT PLACEMENT ===\n"
    );

    dump_tensor_backend(
        "canonical",
        canonical_param
    );

    dump_tensor_backend(
        "momentum",
        opt_ctx->qat_momentum[i]
    );

    dump_tensor_backend(
        "residual",
        opt_ctx->qat_residual[i]
    );

    for (size_t j = 0;
         j < opt_ctx->qat_aliases[i].size();
         ++j) {

        char label[64];

        snprintf(
            label,
            sizeof(label),
            "alias[%zu]",
            j
        );

        dump_tensor_backend(
            label,
            opt_ctx->qat_aliases[i][j]
        );
    }

    fprintf(
        stderr,
        "==========================\n"
    );
}

    auto & pending =
        opt_ctx->
            qat_pending_grads[i];

    pending.push_back(
        grad
    );

    if (pending.size() <
        opt_ctx->
            qat_expected_grads[i]) {

        return nullptr;
    }

    //
    // These dependency gradients must execute
    // before the in-place QLion update changes
    // the quantized weight.
    //
    if (opt_ctx->
        qat_dependency_plan_ready) {

        const auto & plan =
            opt_ctx->
                qat_dependency_plan[i];

        for (int32_t j :
             plan) {

            GGML_ASSERT(
                j >= 0 &&
                j <
                opt_ctx->gf->n_nodes
            );

            struct ggml_tensor * node =
                opt_ctx->
                    gb_grad->nodes[j];

            struct ggml_tensor *
                pending_grad =
                    ggml_graph_get_grad(
                        opt_ctx->gb_grad,
                        node
                    );

            if (pending_grad &&
                !(node->flags &
                  GGML_TENSOR_FLAG_PARAM)) {

                ggml_build_forward_expand(
                    opt_ctx->gb_grad,
                    pending_grad
                );
            }
        }

    } else {

        //
        // First build for this graph shape:
        // calculate dependency set and remember node indices.
        //
        auto & plan =
            opt_ctx->
                qat_dependency_plan[i];

        plan.clear();

        std::map<
            struct ggml_tensor *,
            bool
        > dependency_memo;

        const auto & aliases =
            opt_ctx->
                qat_aliases[i];

        for (int j = 0;
             j <
             opt_ctx->gf->n_nodes;
             ++j) {

            struct ggml_tensor * node =
                opt_ctx->
                    gb_grad->nodes[j];

            struct ggml_tensor *
                pending_grad =
                    ggml_graph_get_grad(
                        opt_ctx->gb_grad,
                        node
                    );

            if (!pending_grad ||
                (node->flags &
                 GGML_TENSOR_FLAG_PARAM)) {

                continue;
            }

            if (
                ggml_opt_depends_on_qat_alias(
                    pending_grad,
                    aliases,
                    opt_ctx->
                        qat_forward_nodes,
                    dependency_memo
                )
            ) {

                plan.push_back(
                    j
                );

                ggml_build_forward_expand(
                    opt_ctx->gb_grad,
                    pending_grad
                );
            }
        }
    }

    struct ggml_tensor * step =
        nullptr;


    // ============================================================
    // Tied embedding specialization.
    //
    // Exact pattern:
    //
    //   OUT_PROD
    //     src0 = dense activations
    //     src1 = dense output gradient
    //
    // + GET_ROWS_BACK
    //     src0 = sparse embedding gradient
    //     src1 = token ids
    //
    // DO NOT create ggml_add() for this case.
    // ============================================================

    if (pending.size() == 2) {

        ggml_tensor * out_prod =
            nullptr;

        ggml_tensor * rows_back =
            nullptr;

        for (ggml_tensor * g :
            pending) {

            if (g->op ==
                GGML_OP_OUT_PROD) {

                out_prod =
                    g;

            } else if (
                g->op ==
                GGML_OP_GET_ROWS_BACK) {

                rows_back =
                    g;
            }
        }

        if (out_prod &&
            rows_back &&

            ggml_are_same_shape(
                out_prod,
                canonical_param
            ) &&

            ggml_are_same_shape(
                rows_back,
                canonical_param
            ) &&

            out_prod->src[0] &&
            out_prod->src[1] &&

            rows_back->src[0] &&
            rows_back->src[1] &&

            out_prod->src[0]->type ==
                GGML_TYPE_F32 &&

            out_prod->src[1]->type ==
                GGML_TYPE_F32 &&

            rows_back->src[0]->type ==
                GGML_TYPE_F32 &&

            rows_back->src[1]->type ==
                GGML_TYPE_I32 &&

            ggml_is_contiguous(
                out_prod->src[0]
            ) &&

            ggml_is_contiguous(
                out_prod->src[1]
            ) &&

            ggml_is_contiguous(
                rows_back->src[0]
            ) &&

            ggml_is_contiguous(
                rows_back->src[1]
            )) {

            if (opt_ctx->opt_period > 1) {
                ggml_tensor * accumulated = ggml_acc_qlion_qat_tied(
                    ctx,
                    opt_ctx->qat_grad_accumulator[i],
                    out_prod->src[0],
                    out_prod->src[1],
                    rows_back->src[0],
                    rows_back->src[1],
                    opt_ctx->opt_i == 0);

                ggml_format_name(accumulated, "QLion Q8_0 tied gradient accumulation for %s", canonical_param->name);
                if (opt_ctx->build_type != GGML_OPT_BUILD_TYPE_OPT) {
                    return accumulated;
                }

                step = ggml_opt_step_qlion_qat(
                    ctx,
                    canonical_param,
                    accumulated,
                    opt_ctx->qat_momentum[i],
                    opt_ctx->qat_residual[i],
                    opt_ctx->opt_step_params);
            } else {
                step = ggml_opt_step_qlion_qat_tied(
                    ctx,
                    canonical_param,
                    out_prod->src[0],
                    out_prod->src[1],
                    rows_back->src[0],
                    rows_back->src[1],
                    opt_ctx->qat_momentum[i],
                    opt_ctx->qat_residual[i],
                    opt_ctx->opt_step_params);
            }
        }
    }


    // ============================================================
    // Existing paths.
    // ============================================================

    if (!step) {

        ggml_tensor * combined =
            pending[0];

        
        for (size_t j = 1;
            j < pending.size();
            ++j) {

            combined =
                ggml_add(
                    ctx,
                    combined,
                    pending[j]
                );
        }

        if (opt_ctx->opt_period > 1) {
            ggml_tensor * accumulated = ggml_acc_qlion_qat(
                ctx,
                opt_ctx->qat_grad_accumulator[i],
                combined,
                opt_ctx->opt_i == 0);

            ggml_format_name(accumulated, "QLion Q8_0 gradient accumulation for %s", canonical_param->name);
            if (opt_ctx->build_type != GGML_OPT_BUILD_TYPE_OPT) {
                return accumulated;
            }

            step = ggml_opt_step_qlion_qat(
                ctx,
                canonical_param,
                accumulated,
                opt_ctx->qat_momentum[i],
                opt_ctx->qat_residual[i],
                opt_ctx->opt_step_params);
        } else if (combined->op ==
            GGML_OP_OUT_PROD_ID) {

            step =
                ggml_opt_step_qlion_qat_id(
                    ctx,

                    canonical_param,

                    combined->src[0],
                    combined->src[1],
                    combined->src[2],

                    opt_ctx->
                        qat_momentum[i],

                    opt_ctx->
                        qat_residual[i],

                    opt_ctx->
                        opt_step_params
                );

        } else if (
            combined->op ==
            GGML_OP_GET_ROWS_BACK) {

            step =
                ggml_opt_step_qlion_qat_rows(
                    ctx,

                    canonical_param,

                    combined->src[0],
                    combined->src[1],

                    opt_ctx->
                        qat_momentum[i],

                    opt_ctx->
                        qat_residual[i],

                    opt_ctx->
                        opt_step_params
                );

        } else {

            step =
                ggml_opt_step_qlion_qat(
                    ctx,

                    canonical_param,
                    combined,

                    opt_ctx->
                        qat_momentum[i],

                    opt_ctx->
                        qat_residual[i],

                    opt_ctx->
                        opt_step_params
                );
        }
    }

    static bool dumped_tied_grad = false;

    if (!dumped_tied_grad &&
        strcmp(param->name, "token_embd.weight") == 0) {

        dumped_tied_grad = true;

        fprintf(
            stderr,
            "\n=== TIED GRAD DUMP ===\n"
            "param=%s pending=%zu aliases=%zu\n",
            param->name,
            pending.size(),
            opt_ctx->qat_aliases[i].size()
        );

        for (size_t k = 0;
            k < pending.size();
            ++k) {

            ggml_tensor * t =
                pending[k];

            fprintf(
                stderr,
                "pending[%zu]:"
                " op=%s"
                " type=%s"
                " ne=[%lld,%lld,%lld,%lld]\n",
                k,
                ggml_op_desc(t),
                ggml_type_name(t->type),
                (long long) t->ne[0],
                (long long) t->ne[1],
                (long long) t->ne[2],
                (long long) t->ne[3]
            );

            for (int s = 0;
                s < GGML_MAX_SRC;
                ++s) {

                if (!t->src[s]) {
                    continue;
                }

                ggml_tensor * src =
                    t->src[s];

                fprintf(
                    stderr,
                    "  src[%d]:"
                    " op=%s"
                    " type=%s"
                    " ne=[%lld,%lld,%lld,%lld]"
                    " name=%s\n",
                    s,
                    ggml_op_desc(src),
                    ggml_type_name(src->type),
                    (long long) src->ne[0],
                    (long long) src->ne[1],
                    (long long) src->ne[2],
                    (long long) src->ne[3],
                    src->name
                );
            }
        }

        fprintf(
            stderr,
            "======================\n"
        );
    }

    ggml_format_name(
        step,
        "QLion QAT step for %s",
        canonical_param->name
    );

    struct ggml_tensor * result =
        step;

    for (struct ggml_tensor * alias :
         opt_ctx->qat_aliases[i]) {

        if (alias == canonical_param) {
            continue;
        }

        if (alias->buffer ==
                canonical_param->buffer &&
            alias->data ==
                canonical_param->data) {

            continue;
        }

        result =
            ggml_cpy(
                ctx,
                result,
                alias
            );
    }

    return result;
}
static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const enum ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    const bool qlion_qat = optimizer == GGML_OPT_OPTIMIZER_TYPE_QLION_QAT;
    const bool accumulate = !qlion_qat && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    const bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    const bool need_q8_device_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0;
    const bool need_quantized_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        ggml_opt_optimizer_state_type(opt_ctx->optimizer) != GGML_TYPE_COUNT;
    const bool need_qat_state = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && qlion_qat;

    ggml_set_input(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    int n_param = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor * node = opt_ctx->gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) + ((need_momenta || need_q8_device_momenta) ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? (opt_ctx->critical_token_weighting ? 64 : 9) : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;
    ggml_set_fused_backward(ctx_results, opt_ctx->fused_backward);

    switch (opt_ctx->loss_type) {
        case GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = false;
            break;
        }
        case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            if (opt_ctx->sparse_labels && !opt_ctx->critical_token_weighting) {
                const int64_t nrows = ggml_nrows(opt_ctx->outputs);
                opt_ctx->sparse_targets = ggml_new_tensor_1d(ctx_results, GGML_TYPE_I32, nrows);
                opt_ctx->sparse_weights = ggml_new_tensor_1d(ctx_results, GGML_TYPE_F32, nrows);
                ggml_set_input(opt_ctx->sparse_targets);
                ggml_set_input(opt_ctx->sparse_weights);
                ggml_set_name(opt_ctx->sparse_targets, "sparse_targets");
                ggml_set_name(opt_ctx->sparse_weights, "sparse_weights");
                opt_ctx->loss = ggml_cross_entropy_loss_sparse(
                    ctx_results, opt_ctx->outputs, opt_ctx->sparse_targets, opt_ctx->sparse_weights);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_sparse");
                if (opt_ctx->opt_period > 1) {
                    opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                    ggml_set_name(opt_ctx->loss, "loss_cross_entropy_sparse_scaled");
                }
                opt_ctx->loss_per_datapoint = true;
                break;
            }
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            struct ggml_tensor * labels_loss = opt_ctx->labels;
            if (opt_ctx->critical_token_weighting) {
                const int64_t nrows = ggml_nrows(opt_ctx->outputs);
                opt_ctx->critical_span_weights   = ggml_new_tensor_1d(ctx_results, GGML_TYPE_F32, nrows);
                opt_ctx->critical_reward_weights = ggml_new_tensor_1d(ctx_results, GGML_TYPE_F32, nrows);
                opt_ctx->critical_warmup_scale   = ggml_new_tensor_1d(ctx_results, GGML_TYPE_F32, 1);
                ggml_set_input(opt_ctx->critical_span_weights);
                ggml_set_input(opt_ctx->critical_reward_weights);
                ggml_set_input(opt_ctx->critical_warmup_scale);
                ggml_set_name(opt_ctx->critical_span_weights, "critical_span_weights");
                ggml_set_name(opt_ctx->critical_reward_weights, "critical_reward_weights");
                ggml_set_name(opt_ctx->critical_warmup_scale, "critical_warmup_scale");

                struct ggml_tensor * active = ggml_reshape_1d(ctx_results, ggml_sum_rows(ctx_results, opt_ctx->labels), nrows);
                struct ggml_tensor * critical_weights = opt_ctx->critical_span_weights;
                if (opt_ctx->critical_confidence_weighting) {
                    struct ggml_tensor * probabilities = ggml_soft_max(ctx_results, opt_ctx->outputs);
                    struct ggml_tensor * target_probabilities = ggml_sum_rows(ctx_results, ggml_mul(ctx_results, probabilities, opt_ctx->labels));
                    target_probabilities = ggml_reshape_1d(ctx_results, target_probabilities, nrows);
                    target_probabilities = ggml_cpy_no_grad(ctx_results, target_probabilities, ggml_dup_tensor(ctx_results, target_probabilities));
                    struct ggml_tensor * selected = ggml_step(ctx_results,
                            ggml_scale_bias(ctx_results, target_probabilities, -1.0f, opt_ctx->critical_confidence_threshold));
                    selected = ggml_mul(ctx_results, selected, active);
                    if (opt_ctx->critical_max_tokens >= 0 && opt_ctx->critical_max_tokens < nrows) {
                        if (opt_ctx->critical_max_tokens == 0) {
                            selected = ggml_scale(ctx_results, selected, 0.0f);
                        } else {
                            struct ggml_tensor * sort_values = ggml_add(ctx_results, target_probabilities,
                                    ggml_scale_bias(ctx_results, active, -2.0f, 2.0f));
                            struct ggml_tensor * indices = ggml_argsort(ctx_results, sort_values, GGML_SORT_ORDER_ASC);
                            indices = ggml_view_1d(ctx_results, indices, opt_ctx->critical_max_tokens, 0);
                            struct ggml_tensor * mask = ggml_reshape_2d(ctx_results, ggml_scale(ctx_results, active, 0.0f), 1, nrows);
                            struct ggml_tensor * ones = ggml_view_1d(ctx_results, active, opt_ctx->critical_max_tokens, 0);
                            ones = ggml_reshape_2d(ctx_results, ggml_scale_bias(ctx_results, ones, 0.0f, 1.0f), 1, opt_ctx->critical_max_tokens);
                            mask = ggml_set_rows(ctx_results, mask, ones, indices);
                            selected = ggml_mul(ctx_results, selected, ggml_reshape_1d(ctx_results, mask, nrows));
                        }
                    }
                    opt_ctx->critical_selected = selected;

                    struct ggml_tensor * confidence_weights;
                    if (opt_ctx->critical_weight_linear) {
                        struct ggml_tensor * interpolation = ggml_scale_bias(ctx_results, target_probabilities,
                                -1.0f / opt_ctx->critical_confidence_threshold, 1.0f);
                        confidence_weights = ggml_scale_bias(ctx_results,
                                ggml_mul(ctx_results, interpolation, selected), opt_ctx->critical_token_weight - 1.0f, 1.0f);
                        confidence_weights = ggml_clamp(ctx_results, confidence_weights, 1.0f, opt_ctx->critical_token_weight);
                    } else {
                        confidence_weights = ggml_scale_bias(ctx_results, selected, opt_ctx->critical_token_weight - 1.0f, 1.0f);
                    }
                    critical_weights = ggml_add(ctx_results, opt_ctx->critical_span_weights,
                            ggml_relu(ctx_results, ggml_sub(ctx_results, confidence_weights, opt_ctx->critical_span_weights)));
                } else {
                    opt_ctx->critical_selected = ggml_scale(ctx_results, active, 0.0f);
                }
                critical_weights = ggml_scale_bias(ctx_results,
                        ggml_mul(ctx_results, ggml_scale_bias(ctx_results, critical_weights, 1.0f, -1.0f),
                            ggml_repeat(ctx_results, opt_ctx->critical_warmup_scale, critical_weights)), 1.0f, 1.0f);
                struct ggml_tensor * effective_weights = ggml_mul(ctx_results, active,
                        ggml_mul(ctx_results, opt_ctx->critical_reward_weights, critical_weights));
                opt_ctx->critical_effective_weights = effective_weights;
                struct ggml_tensor * weight_sum = ggml_clamp(ctx_results, ggml_sum(ctx_results, effective_weights), 1e-12f, FLT_MAX);
                struct ggml_tensor * normalized_weights = ggml_scale(ctx_results,
                        ggml_div(ctx_results, effective_weights, ggml_repeat(ctx_results, weight_sum, effective_weights)), (float) nrows);
                labels_loss = ggml_mul(ctx_results, opt_ctx->labels,
                        ggml_repeat(ctx_results, ggml_reshape_2d(ctx_results, normalized_weights, 1, nrows), opt_ctx->labels));
                ggml_set_name(labels_loss, "critical_weighted_labels");

                struct ggml_tensor * active_sum = ggml_clamp(ctx_results, ggml_sum(ctx_results, active), 1e-12f, FLT_MAX);
                struct ggml_tensor * unweighted_scale = ggml_scale(ctx_results,
                        ggml_div(ctx_results, active, ggml_repeat(ctx_results, active_sum, active)), (float) nrows);
                struct ggml_tensor * unweighted_labels = ggml_mul(ctx_results, opt_ctx->labels,
                        ggml_repeat(ctx_results, ggml_reshape_2d(ctx_results, unweighted_scale, 1, nrows), opt_ctx->labels));
                opt_ctx->critical_unweighted_loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, unweighted_labels);
                if (opt_ctx->opt_period > 1) {
                    opt_ctx->critical_unweighted_loss = ggml_scale(ctx_results, opt_ctx->critical_unweighted_loss, 1.0f / opt_ctx->opt_period);
                }
                ggml_set_name(opt_ctx->critical_unweighted_loss, "critical_unweighted_loss");
            }
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, labels_loss);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            if (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            if (opt_ctx->critical_token_weighting) {
                ggml_set_output(opt_ctx->critical_selected);
                ggml_set_output(opt_ctx->critical_effective_weights);
                ggml_set_output(opt_ctx->critical_unweighted_loss);
                ggml_build_forward_expand(opt_ctx->gf, opt_ctx->critical_selected);
                ggml_build_forward_expand(opt_ctx->gf, opt_ctx->critical_effective_weights);
                ggml_build_forward_expand(opt_ctx->gf, opt_ctx->critical_unweighted_loss);
            }
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        struct ggml_tensor * target_classes = opt_ctx->sparse_targets
            ? opt_ctx->sparse_targets
            : ggml_argmax(ctx_results, opt_ctx->labels);
        opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, target_classes);
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if ((need_momenta || need_q8_device_momenta) && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    // Allocate moments on the same buffer type as the param tensor so
                    // the ADAMW op runs on the correct backend (avoids cross-device mismatch
                    // when some LoRA tensors are on CPU and others on GPU with partial offload).
                    ggml_backend_buffer_type_t param_buft = node->buffer
                        ? ggml_backend_buffer_get_type(node->buffer)
                        : ggml_backend_cpu_buffer_type();

                    if (need_q8_device_momenta && !ggml_opt_has_device_q8_adamw(param_buft)) {
                        opt_ctx->grad_m[i] = nullptr;
                        opt_ctx->grad_v[i] = nullptr;
                        continue;
                    }

                    // Allocate a tiny context + buffer for this pair of moment tensors.
                    const size_t sz = 2 * ggml_tensor_overhead();
                    struct ggml_init_params mip = { sz, nullptr, true };
                    struct ggml_context * mctx = ggml_init(mip);
                    if (need_q8_device_momenta) {
                        const int64_t ne = ggml_nelements(node);
                        const int64_t block_size = ggml_blck_size(GGML_TYPE_Q8_0);
                        const int64_t ne_padded = (ne + block_size - 1)/block_size*block_size;
                        opt_ctx->grad_m[i] = ggml_new_tensor_1d(mctx, GGML_TYPE_Q8_0, ne_padded);
                        opt_ctx->grad_v[i] = ggml_new_tensor_1d(mctx, GGML_TYPE_Q8_0, ne_padded);
                    } else {
                        opt_ctx->grad_m[i] = ggml_new_tensor(mctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                        opt_ctx->grad_v[i] = ggml_new_tensor(mctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                    }
                    ggml_backend_buffer_t mbuf = ggml_backend_alloc_ctx_tensors_from_buft(mctx, param_buft);
                    ggml_backend_buffer_clear(mbuf, 0);
                    opt_ctx->bufs_momenta.push_back(mbuf);
                    opt_ctx->ctxs_momenta.push_back(mctx); // keep alive for tensor metadata
                } else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    if (need_qat_state &&
        opt_ctx->qat_params.empty()) {

        struct qat_param_group {
            std::vector<ggml_tensor *> aliases;
        };

        std::vector<qat_param_group> groups;

        //
        // First pass:
        // group logical aliases without allocating any QAT state.
        //
        for (int i = 0;
            i < opt_ctx->gf->n_nodes;
            ++i) {

            ggml_tensor * node =
                opt_ctx->gf->nodes[i];

            if (!(node->flags &
                GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            if (node->type != GGML_TYPE_MXFP4 &&
                node->type != GGML_TYPE_Q4_0) {
                continue;
            }

            bool grouped =
                false;

            for (auto & group :
                groups) {

                if (
                    strcmp(
                        group.aliases[0]->name,
                        "token_embd.weight"
                    ) == 0
                ) {

                    fprintf(
                        stderr,
                        "\n=== QAT ALIAS PLACEMENT ===\n"
                    );

                    for (size_t j = 0;
                        j < group.aliases.size();
                        ++j) {

                        ggml_tensor * t =
                            group.aliases[j];

                        const char * buffer_name =
                            t->buffer
                                ? ggml_backend_buffer_name(
                                    t->buffer
                                )
                                : "(null)";

                        ggml_backend_dev_t dev =
                            nullptr;

                        if (t->buffer) {

                            ggml_backend_buffer_type_t buft =
                                ggml_backend_buffer_get_type(
                                    t->buffer
                                );

                            if (buft) {
                                dev =
                                    ggml_backend_buft_get_device(
                                        buft
                                    );
                            }
                        }

                        fprintf(
                            stderr,
                            "alias[%zu]:"
                            " ptr=%p"
                            " data=%p"
                            " buffer=%s"
                            " device=%s"
                            " dev_type=%d"
                            " priority=%d\n",

                            j,

                            (void *) t,
                            t->data,

                            buffer_name,

                            dev
                                ? ggml_backend_dev_name(dev)
                                : "(none)",

                            dev
                                ? (int)
                                    ggml_backend_dev_type(dev)
                                : -1,

                            ggml_opt_qat_backend_priority(t)
                        );
                    }

                    fprintf(
                        stderr,
                        "===========================\n"
                    );
                }



                        GGML_ASSERT(
                            !group.aliases.empty()
                        );

                        if (
                            ggml_opt_qat_same_logical_param(
                                group.aliases[0],
                                node
                            )
                        ) {

                            //
                            // Avoid duplicate pointer insertion.
                            //
                            if (std::find(
                                    group.aliases.begin(),
                                    group.aliases.end(),
                                    node
                                ) ==
                                group.aliases.end()) {

                                group.aliases.push_back(
                                    node
                                );
                            }

                            grouped =
                                true;

                            break;
                        }
                    }

                    if (!grouped) {

                        qat_param_group group;

                        group.aliases.push_back(
                            node
                        );

                        groups.push_back(
                            std::move(group)
                        );
                    }
                }

        //
        // Second pass:
        // preserve logical parameter ordering, but choose
        // the best device alias as canonical inside each group.
        //
        for (auto & group :
            groups) {

            GGML_ASSERT(
                !group.aliases.empty()
            );

            size_t best =
                0;

            int best_priority =
                ggml_opt_qat_backend_priority(
                    group.aliases[0]
                );

            for (size_t j = 1;
                j < group.aliases.size();
                ++j) {

                const int priority =
                    ggml_opt_qat_backend_priority(
                        group.aliases[j]
                    );

                if (priority >
                    best_priority) {

                    best =
                        j;

                    best_priority =
                        priority;
                }
            }

            if (best != 0) {
                std::swap(
                    group.aliases[0],
                    group.aliases[best]
                );
            }

            //
            // First registration determines QAT state placement.
            //
            ggml_opt_qat_register_param(
                opt_ctx,
                group.aliases[0]
            );

            for (size_t j = 1;
                j < group.aliases.size();
                ++j) {

                ggml_opt_qat_register_param(
                    opt_ctx,
                    group.aliases[j]
                );
            }
        }
    }

    if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT) {
        const int64_t n_opt_params =
            ggml_opt_optimizer_is_adamw(optimizer)
                ? 8
                : (qlion_qat ? 5 : 2);
        opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, n_opt_params);
        ggml_set_input(opt_ctx->opt_step_params);
        ggml_format_name(opt_ctx->opt_step_params, "%s_params", ggml_opt_optimizer_name(optimizer));
    }

    // Gradient checkpointing: mark every Nth forward node as OUTPUT so the allocator
    // keeps its memory alive through the backward pass.  The backward graph already
    // contains the forward ops (gb_grad is a superset of gf), so the checkpointed
    // activations are naturally available for backward matmuls without recomputation.
    // This prevents the allocator from aliasing those buffers to later ops, cutting
    // peak activation VRAM at the cost of slightly larger static allocation.
    if (opt_ctx->grad_checkpoint_interval > 0) {
        const int interval = opt_ctx->grad_checkpoint_interval;
        const int n_fwd    = opt_ctx->gf->n_nodes;
        int ckpt_count = 0;
        for (int i = interval - 1; i < n_fwd; i += interval) {
            struct ggml_tensor * node = opt_ctx->gf->nodes[i];
            // Only checkpoint F32 compute nodes — skip I32 index tensors and already-output nodes.
            if (node->type != GGML_TYPE_F32) continue;
            if (node->flags & GGML_TENSOR_FLAG_OUTPUT)  continue;
            if (node->flags & GGML_TENSOR_FLAG_INPUT)   continue;
            node->flags |= GGML_TENSOR_FLAG_OUTPUT;
            ckpt_count++;
        }
        if (ckpt_count > 0) {
            GGML_LOG_DEBUG("%s: gradient checkpointing: marked %d/%d nodes as persistent (interval=%d)\n",
                __func__, ckpt_count, n_fwd, interval);
        }
    }

    if (qlion_qat) {
        const uint64_t signature = ggml_opt_qat_graph_signature(opt_ctx);
        const bool changed = signature != opt_ctx->qat_dependency_signature ||
            opt_ctx->qat_dependency_plan.size() != opt_ctx->qat_params.size();

        if (changed) {
            opt_ctx->qat_dependency_signature = signature;
            opt_ctx->qat_dependency_plan_ready = false;
            opt_ctx->qat_dependency_plan.assign(opt_ctx->qat_params.size(), {});
        }

        opt_ctx->qat_alias_to_state.clear();
        for (size_t state = 0; state < opt_ctx->qat_aliases.size(); ++state) {
            for (ggml_tensor * alias : opt_ctx->qat_aliases[state]) {
                opt_ctx->qat_alias_to_state[alias] = state;
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    if (qlion_qat) {
        opt_ctx->qat_forward_nodes.clear();
        for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
            opt_ctx->qat_forward_nodes.insert(opt_ctx->gf->nodes[i]);
        }
        opt_ctx->qat_pending_grads.assign(opt_ctx->qat_params.size(), {});
        opt_ctx->qat_expected_grads.assign(opt_ctx->qat_params.size(), 0);
        for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
            struct ggml_tensor * node = opt_ctx->gf->nodes[i];
            if (!(node->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }
            for (size_t state = 0; state < opt_ctx->qat_aliases.size(); ++state) {
                const auto & aliases = opt_ctx->qat_aliases[state];
                if (std::find(aliases.begin(), aliases.end(), node) != aliases.end()) {
                    opt_ctx->qat_expected_grads[state]++;
                    break;
                }
            }
        }
        ggml_build_backward_expand_with_callback(
            opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data(),
            ggml_opt_qlion_qat_backward_callback, opt_ctx);
        if (qlion_qat &&
            !opt_ctx->
                qat_dependency_plan_ready) {

            opt_ctx->
                qat_dependency_plan_ready =
                    true;

            size_t dependency_count =
                0;

            for (const auto & plan :
                opt_ctx->
                    qat_dependency_plan) {

                dependency_count +=
                    plan.size();
            }

            GGML_LOG_INFO(
                "%s: cached QLion dependency plan: %zu dependencies across %zu states\n",
                __func__,
                dependency_count,
                opt_ctx->
                    qat_dependency_plan.size()
            );
        }
    } else {
        ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());
    }

    if (need_quantized_momenta) {
        const enum ggml_type state_type = ggml_opt_optimizer_state_type(opt_ctx->optimizer);
        const ggml_type_traits * traits = ggml_get_type_traits(state_type);
        GGML_ASSERT(traits->to_float && traits->from_float_ref);
        ggml_quantize_init(state_type);

        opt_ctx->quantized_params.clear();
        opt_ctx->quantized_grads.clear();
        size_t state_i = 0;
        for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_grad, node);
            if (!grad || !(node->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }
            GGML_ASSERT(node->type == GGML_TYPE_F32 && grad->type == GGML_TYPE_F32);
            if (need_q8_device_momenta && opt_ctx->grad_m[i]) {
                continue;
            }
            const int64_t ne = ggml_nelements(node);
            const enum ggml_type state_type_v = state_type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
            const int64_t block_size = std::max(ggml_blck_size(state_type), ggml_blck_size(state_type_v));
            const int64_t ne_padded = ((ne + block_size - 1)/block_size)*block_size;
            if (state_i == opt_ctx->quantized_states.size()) {
                ggml_opt_context::quantized_state state;
                state.ne = ne;
                state.ne_padded = ne_padded;
                const size_t nbytes = ggml_row_size(state_type, ne_padded);
                state.m.resize(nbytes);
                state.v.resize(ggml_row_size(state_type_v, ne_padded));
                std::vector<float> zeros(ne_padded, 0.0f);
                traits->from_float_ref(zeros.data(), state.m.data(), ne_padded);
                ggml_get_type_traits(state_type_v)->from_float_ref(zeros.data(), state.v.data(), ne_padded);
                opt_ctx->quantized_states.push_back(std::move(state));
            } else {
                GGML_ASSERT(opt_ctx->quantized_states[state_i].ne == ne);
            }
            opt_ctx->quantized_params.push_back(node);
            opt_ctx->quantized_grads.push_back(grad);
            state_i++;
        }
        GGML_ASSERT(state_i == opt_ctx->quantized_states.size());
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    const char * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            struct ggml_tensor * m = nullptr;
            struct ggml_tensor * v = nullptr;
            if (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", node->name);
                ggml_format_name(v, "AdamW v for %s", node->name);
            } else if (need_q8_device_momenta && opt_ctx->grad_m[i]) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW Q8_0 m for %s", node->name);
                ggml_format_name(v, "AdamW Q8_0 v for %s", node->name);
            }
            struct ggml_tensor * opt_step;
            switch (optimizer) {
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
                    break;
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0:
                    if (!m || !v) {
                        continue;
                    }
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
                    break;
                case GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, node, grad, adamw_params);
                    break;
                default:
                    continue;
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, node->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->loss_type        = params.loss_type;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs          = params.outputs;
    result->opt_period                = params.opt_period;
    result->grad_checkpoint_interval  = params.grad_checkpoint_interval;
    result->get_opt_pars              = params.get_opt_pars;
    result->get_opt_pars_ud           = params.get_opt_pars_ud;
    result->optimizer                 = params.optimizer;
    result->critical_token_weighting  = params.critical_token_weighting;
    result->critical_confidence_weighting = params.critical_confidence_weighting;
    result->critical_token_weight     = params.critical_token_weight;
    result->critical_confidence_threshold = params.critical_confidence_threshold;
    result->critical_weight_linear    = params.critical_weight_linear;
    result->sparse_labels             = params.sparse_labels;
    result->fused_backward            = params.fused_backward;

    GGML_ASSERT(result->opt_period >= 1);
    result->static_graphs = result->ctx_compute;

    GGML_ASSERT(!result->static_graphs || result->optimizer != GGML_OPT_OPTIMIZER_TYPE_QLION_QAT || result->opt_period == 1);

    if (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs);
        return result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    ggml_opt_build(result);

    return result;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    for (ggml_backend_buffer_t buf : opt_ctx->bufs_momenta) {
        ggml_backend_buffer_free(buf);
    }
    for (struct ggml_context * ctx : opt_ctx->ctxs_momenta) {
        ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buf : opt_ctx->qat_buffers) {
        ggml_backend_buffer_free(buf);
    }
    for (struct ggml_context * ctx : opt_ctx->qat_contexts) {
        ggml_free(ctx);
    }
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        for (ggml_backend_buffer_t buf : opt_ctx->bufs_momenta) {
            ggml_backend_buffer_clear(buf, 0);
        }
        for (ggml_opt_context::quantized_state & state : opt_ctx->quantized_states) {
            std::fill(state.m.begin(), state.m.end(), 0);
            std::fill(state.v.begin(), state.v.end(), 0);
        }
        for (ggml_backend_buffer_t buf : opt_ctx->qat_buffers) {
            ggml_backend_buffer_clear(buf, 0);
        }
        opt_ctx->iter = 1;
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct ggml_tensor * ggml_opt_sparse_targets(ggml_opt_context_t opt_ctx) {
    return opt_ctx->sparse_targets;
}

struct ggml_tensor * ggml_opt_sparse_weights(ggml_opt_context_t opt_ctx) {
    return opt_ctx->sparse_weights;
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct ggml_tensor * ggml_opt_critical_span_weights(ggml_opt_context_t opt_ctx) {
    return opt_ctx->critical_span_weights;
}

struct ggml_tensor * ggml_opt_critical_reward_weights(ggml_opt_context_t opt_ctx) {
    return opt_ctx->critical_reward_weights;
}

struct ggml_tensor * ggml_opt_critical_warmup_scale(ggml_opt_context_t opt_ctx) {
    return opt_ctx->critical_warmup_scale;
}

struct ggml_tensor * ggml_opt_critical_selected(ggml_opt_context_t opt_ctx) {
    return opt_ctx->critical_selected;
}

struct ggml_tensor * ggml_opt_critical_effective_weights(ggml_opt_context_t opt_ctx) {
    return opt_ctx->critical_effective_weights;
}

struct ggml_tensor * ggml_opt_critical_unweighted_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->critical_unweighted_loss;
}

void ggml_opt_set_critical_max_tokens(ggml_opt_context_t opt_ctx, int32_t max_tokens) {
    opt_ctx->critical_max_tokens = max_tokens;
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    return ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    // For non-static graphs the compute graph is rebuilt every call, so ggml_graph_reset
    // is not called and grad_accs may carry over values from the previous accumulation window.
    // Explicitly zero them at the start of each gradient-accumulation cycle.
    if (!opt_ctx->static_graphs && backward && opt_ctx->opt_i == 0) {
        for (struct ggml_tensor * ga : opt_ctx->grad_accs) {
            if (ga) {
                ggml_set_zero(ga);
            }
        }
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    struct ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case GGML_OPT_BUILD_TYPE_OPT: {
            const bool has_device_q8_step =
                opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0 &&
                !opt_ctx->bufs_momenta.empty();
            graph = ggml_opt_optimizer_state_type(opt_ctx->optimizer) == GGML_TYPE_COUNT || has_device_q8_step
                ? opt_ctx->gb_opt
                : opt_ctx->gb_grad;
        } break;
    }
    GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    const bool do_optimizer_step = opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT;
    if (do_optimizer_step) {
        const ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        switch (opt_ctx->optimizer) {
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW_F16:
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0:
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q6_K:
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW_IQ4_NL: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.gclip >= 0.0f);

                // beta1, beta2 after applying warmup
                const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                float * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
                adamw_par_data[7] = opt_pars.adamw.gclip;
            } break;
            case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                float * sgd = ggml_get_data_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } break;
            case GGML_OPT_OPTIMIZER_TYPE_QLION_QAT: {
                GGML_ASSERT(opt_pars.qlion_qat.alpha > 0.0f);
                GGML_ASSERT(opt_pars.qlion_qat.beta >= 0.0f);
                GGML_ASSERT(opt_pars.qlion_qat.beta < 1.0f);
                GGML_ASSERT(opt_pars.qlion_qat.wd >= 0.0f);
                GGML_ASSERT(opt_pars.qlion_qat.wd <= 1.0f);
                GGML_ASSERT(opt_pars.qlion_qat.gclip >= 0.0f);
                float * qlion_qat = ggml_get_data_f32(opt_ctx->opt_step_params);
                qlion_qat[0] = opt_pars.qlion_qat.alpha;
                qlion_qat[1] = opt_pars.qlion_qat.beta;
                qlion_qat[2] = opt_pars.qlion_qat.wd;
                qlion_qat[3] = opt_pars.qlion_qat.gclip;
                qlion_qat[4] = opt_pars.qlion_qat.fast_state_scale ? 1.0f : 0.0f;
            } break;
            default:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    if (do_optimizer_step && ggml_opt_optimizer_state_type(opt_ctx->optimizer) != GGML_TYPE_COUNT) {
        const ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);
        ggml_opt_step_adamw_quantized(opt_ctx, opt_pars);
    }
    opt_ctx->iter += do_optimizer_step;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    if (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    if (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    struct ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_loss_type         loss_type,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    return c->optimizer;
}

int64_t ggml_opt_qat_state_count(ggml_opt_context_t opt_ctx) {
    return (int64_t) opt_ctx->qat_params.size();
}

struct ggml_tensor * ggml_opt_qat_state_param(ggml_opt_context_t opt_ctx, int64_t index) {
    GGML_ASSERT(index >= 0 && index < (int64_t) opt_ctx->qat_params.size());
    return opt_ctx->qat_params[index];
}

struct ggml_tensor * ggml_opt_qat_state_momentum(ggml_opt_context_t opt_ctx, int64_t index) {
    GGML_ASSERT(index >= 0 && index < (int64_t) opt_ctx->qat_momentum.size());
    return opt_ctx->qat_momentum[index];
}

struct ggml_tensor * ggml_opt_qat_state_residual(ggml_opt_context_t opt_ctx, int64_t index) {
    GGML_ASSERT(index >= 0 && index < (int64_t) opt_ctx->qat_residual.size());
    return opt_ctx->qat_residual[index];
}

struct ggml_tensor * ggml_opt_qat_state_gradient_accumulator(ggml_opt_context_t opt_ctx, int64_t index) {
    GGML_ASSERT(index >= 0 && index < (int64_t) opt_ctx->qat_grad_accumulator.size());
    return opt_ctx->qat_grad_accumulator[index];
}

int64_t ggml_opt_step(ggml_opt_context_t opt_ctx) {
    return opt_ctx->iter - 1;
}

void ggml_opt_set_step(ggml_opt_context_t opt_ctx, int64_t step) {
    GGML_ASSERT(step >= 0);
    opt_ctx->iter = step + 1;
}

GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o) {
    switch (o) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            return "adamw";
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_F16:
            return "adamw_f16";
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q8_0:
            return "adamw_q8_0";
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_Q6_K:
            return "adamw_q6_k";
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW_IQ4_NL:
            return "adamw_iq4_nl";
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            return "sgd";
        case GGML_OPT_OPTIMIZER_TYPE_QLION_QAT:
            return "qlion_qat";
        default:
            return "undefined";
    };
}
