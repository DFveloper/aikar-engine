#include "models.h"
#include <cstdlib>
#include <sstream>
#include "llama-impl.h"

void llama_model_gemma4::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    hparams.kv_global_only = arch == LLM_ARCH_LUMEN4EXP;
    uint32_t n_kv_shared_layers = 0;
    if (!hparams.kv_global_only) {
        ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);
    }
    if (hparams.kv_global_only) {
        if (hparams.n_layer_all != 30) {
            throw std::runtime_error("Lumen4Exp Global sharing requires 30 layers");
        }
        hparams.kvsharer_map[23] = 18;
    }

    if (hparams.kv_global_only && n_kv_shared_layers >= hparams.n_layer_all) {
        throw std::runtime_error("shared KV layers must be smaller than the layer count");
    }
    hparams.n_layer_kv_from_start = hparams.n_layer_all - (int32_t)n_kv_shared_layers;
    if (hparams.kv_global_only) {
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (!hparams.is_swa(il)) {
                const int32_t source = hparams.kv_source(il);
                if (source < 0) {
                    throw std::runtime_error("Lumen KV sharing requires an earlier independent global layer");
                }
                LLAMA_LOG_INFO("%s: global layer %u KV source %d\n", __func__, il, source);
            }
        }
    }
    if (const char * spec = std::getenv("KVSHARER_MAP")) {
        std::istringstream input(spec);
        std::string pair;
        while (std::getline(input, pair, ',')) {
            int dst, src;
            char extra;
            if (std::sscanf(pair.c_str(), "%d:%d%c", &dst, &src, &extra) != 2 ||
                src < 0 || dst <= src || dst >= hparams.n_layer_kv_from_start ||
                hparams.is_swa(dst) != hparams.is_swa(src)) {
                throw std::runtime_error("invalid KVSHARER_MAP; require earlier source of same attention type");
            }
            hparams.kvsharer_map[dst] = src + 1;
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            int src = il;
            while (hparams.kvsharer_map[src]) { src = hparams.kvsharer_map[src] - 1; }
            if (src != int(il)) {
                hparams.kvsharer_map[il] = src + 1;
                LLAMA_LOG_INFO("KVSharer: %u -> %d\n", il, src);
            }
        }
    }
    if (hparams.kv_global_only) {
        if (hparams.n_embd_head_k_full != 512 || hparams.n_embd_head_v_full != 512) {
            throw std::runtime_error("Lumen4Exp requires 512-dimensional Global K/V heads");
        }
        hparams.n_rot_full = 128;
        LLAMA_LOG_INFO("Lumen4Exp: Global RoPE=128/512, V cache=128, V tail=K tail/K norm scale\n");
    }
    hparams.f_attention_scale     = 1.0f; // Gemma4 uses self.scaling = 1.0 (no pre-attn scaling)

    // shared layers read the KV of the last layer with own KV and the same SWA type
    GGML_ASSERT(hparams.n_layer_kv_from_start >= 2);
    for (uint32_t il = hparams.n_layer_kv_from_start; il < hparams.n_layer_all; il++) {
        hparams.kv_source_impl[il] = hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
    }

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // when non_causal is set, the model will use bidirectional attention on SWA layers only, while dense layers will remain causal
    // ref: use_bidirectional_attention == "vision" in HF config
    // note: E2B/E4B are always causal, bypassing this logic
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_ONLY;

    switch (hparams.n_layer()) {
        case 30: type = LLM_TYPE_26B_A4B; break;
        case 35: type = LLM_TYPE_E2B; break;
        case 42: type = LLM_TYPE_E4B; break;
        case 60: type = LLM_TYPE_31B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gemma4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const uint32_t n_embd_per_layer = hparams.n_embd_per_layer;
    const int64_t  n_ff_exp         = hparams.n_ff_exp();

    if (n_embd_head_k != n_embd_head_v) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("Gemma 4 requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    if (n_embd_per_layer > 0) {
        per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),    {n_embd_per_layer * n_layer, n_vocab}, TENSOR_READ_LAZY);
        per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_embd_per_layer * n_layer}, 0);
        per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_embd_per_layer}, 0);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_embd_k    = hparams.n_embd_k_gqa(i);
        const int64_t n_embd_v    = hparams.n_embd_v_gqa(i);
        const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // note: use_alternative_attention (v_proj is optional, if it's not present, use k_proj)
        layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i),
            {n_embd, n_embd_head * n_head + n_embd_k + n_embd_v}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
        if (!layer.wqkv) {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, kv_flags);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, n_embd_v}, TENSOR_NOT_REQUIRED);
        }
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_k_norm    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), {n_embd_head}, kv_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, TENSOR_NOT_REQUIRED);

        if (!hparams.is_swa(i)) {
            // full_attention layers use rope_freqs for proportional rope
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        // handle use_double_wide_mlp
        int64_t n_ff_cur = hparams.n_ff(i);

        // for expert layers, we use normal FFN as shared expert (same as python code)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        // MoE router
        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, TENSOR_NOT_REQUIRED);
        bool has_expert = layer.ffn_gate_inp != nullptr;

        // norm
        if (has_expert) {
            layer.ffn_gate_inp_s = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "scale", i), {n_embd}, 0);

            layer.ffn_pre_norm_2  = create_tensor(tn(LLM_TENSOR_FFN_PRE_NORM_2,  "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_1 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_1, "weight", i), {n_embd}, 0);
            layer.ffn_post_norm_2 = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM_2, "weight", i), {n_embd}, 0);

            // MoE FFN
            layer.ffn_gate_up_exps  = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS,  "weight", i), {n_embd, n_ff_exp * 2, n_expert}, TENSOR_NOT_REQUIRED);

            if (layer.ffn_gate_up_exps == nullptr) {
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
                layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            }

            layer.ffn_down_exps     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,     "weight", i), {n_ff_exp, n_embd, n_expert}, 0);

            // per-expert scale will be loaded as down_exps_s at the end of the current switch case
        }

        // per-layer embeddings
        if (n_embd_per_layer > 0) {
            layer.per_layer_inp_gate   = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);
            layer.per_layer_proj       = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);
            layer.per_layer_post_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// get 2D slice view from a 3D tensor, the idx corresponds to the 3rd dim
static ggml_tensor * gemma4_view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1], x->nb[1], idx * x->nb[2]);
}

class llm_graph_input_gemma4_prune : public llm_graph_input_i {
    int64_t n_slices;
    int64_t n_slice_tokens;
    int64_t n_slice_outputs;
public:
    ggml_tensor * full_mask      = nullptr;
    ggml_tensor * full_mask_swa  = nullptr;
    ggml_tensor * out_mask       = nullptr;
    ggml_tensor * out_mask_swa   = nullptr;
    ggml_tensor * inp_prune_ids  = nullptr;
    ggml_tensor * inp_prune_keep = nullptr;
    llm_graph_input_gemma4_prune(int64_t n_slices, int64_t n_slice_tokens, int64_t n_slice_outputs)
            : n_slices(n_slices), n_slice_tokens(n_slice_tokens), n_slice_outputs(n_slice_outputs) {}

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    static int64_t max_outputs_per_slice(
        const llama_ubatch & ubatch, int64_t n_slices, int64_t n_slice_tokens
    );
};

llama_model_gemma4::graph::graph(const llm_graph_params & params) :
        llm_graph_context(params),
        n_embd_per_layer(model.hparams.n_embd_per_layer) {
    if (hparams.kv_global_only) {
        for (const auto & layer : model.layers) { kv_reconstruction_scales.push_back(layer.attn_k_norm); }
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // do the PLE first to guarantee it is done in the host buffer
    // ref: https://github.com/ggml-org/llama.cpp/pull/30160
    ggml_tensor * inp_per_layer = nullptr;
    if (model.per_layer_tok_embd) {
        inp_per_layer = build_inp_per_layer();
        ggml_build_forward_expand(gf, inp_per_layer);
    }

    // important: do not normalize weights for raw embeddings input (i.e. encoded image emdeddings)
    inpL = build_inp_embd(model.tok_embd, sqrtf(n_embd));
    cb(inpL, "inp_scaled", -1);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    // TODO: is causal == true correct? might need some changes
    llm_graph_input_i * inp_attn;
    if (full_sequence) { inp_attn = build_attn_inp_no_cache(); }
    else if (cparams.kv_paged) { inp_attn = build_attn_inp_kv_paged(); }
    else { inp_attn = build_attn_inp_kv_iswa(); }

    std::vector<ggml_tensor *> sequence_k(n_layer, nullptr), sequence_v(n_layer, nullptr);

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn_prune;

    if (model.per_layer_tok_embd) {
        // fold per_layer_input_scale (1/sqrt(2)) from project_per_layer_inputs() into tok_embd_scale
        const float tok_embd_scale = sqrtf((float) n_embd_per_layer) / sqrtf(2.0f);

        inp_per_layer = ggml_scale     (ctx0, inp_per_layer, tok_embd_scale);
        inp_per_layer = ggml_reshape_3d(ctx0, inp_per_layer, n_embd_per_layer, n_layer, inp_per_layer->ne[1]);

        // inp_per_layer shape: [n_embd_per_layer, n_tokens, n_layer]
        inp_per_layer = project_per_layer_inputs(inpL, inp_per_layer);
    }

    const bool needs_unmasked_nextn =
        cparams.embeddings_nextn && !cparams.embeddings_nextn_masked;

    // KV-shared layers do not write to the KV cache, so prune non-output token rows before the first KV-shared layer
    bool prune_shared = !training && !full_sequence && !cparams.kv_paged && !hparams.kv_global_only &&
                        cparams.ream_calibration_layer < 0 && !needs_unmasked_nextn &&
                        hparams.n_layer_kv_from_start > 0 && hparams.n_layer_kv_from_start < n_layer &&
                        !cparams.embeddings;
    for (int il = (int) hparams.n_layer_kv_from_start; prune_shared && il <= n_layer; ++il) {
        prune_shared = !cparams.embeddings_layer_inp[il];
    }

    int64_t n_rows = n_tokens;

    ggml_tensor * inp_prune_ids = nullptr;
    ggml_tensor * inp_prune_keep = nullptr;
    ggml_tensor * inp_out_ids = prune_shared || cparams.ream_calibration_layer >= 0 ? nullptr : build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        if (prune_shared && il == (int) hparams.n_layer_kv_from_start) {
            inp_attn_prune = std::make_unique<llm_graph_input_attn_kv_iswa>(*static_cast<llm_graph_input_attn_kv_iswa *>(inp_attn));
            ggml_tensor * full     = inp_attn_prune->self_attn_mask.raw;
            ggml_tensor * full_swa = inp_attn_prune->self_attn_mask_swa.raw;

            const int64_t n_slices = full->ne[3], n_slice_tokens = full->ne[1];
            const int64_t n_slice_outputs = llm_graph_input_gemma4_prune::max_outputs_per_slice(ubatch, n_slices, n_slice_tokens);

            n_rows = n_slices * n_slice_outputs;

            auto prune_input = std::make_unique<llm_graph_input_gemma4_prune>(n_slices, n_slice_tokens, n_slice_outputs);
            prune_input->full_mask = full;
            prune_input->full_mask_swa = full_swa;
            prune_input->out_mask = ggml_new_tensor_4d(ctx0, full->type, full->ne[0], n_slice_outputs, 1, n_slices);
            prune_input->out_mask_swa = ggml_new_tensor_4d(ctx0, full_swa->type, full_swa->ne[0], n_slice_outputs, 1, n_slices);
            prune_input->inp_prune_ids  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
            prune_input->inp_prune_keep = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);

            ggml_set_input(prune_input->out_mask);
            ggml_set_input(prune_input->out_mask_swa);
            ggml_set_input(prune_input->inp_prune_ids);
            ggml_set_input(prune_input->inp_prune_keep);

            inp_prune_ids  = prune_input->inp_prune_ids;
            inp_prune_keep = prune_input->inp_prune_keep;

            inpL = ggml_get_rows(ctx0, inpL, inp_prune_ids);
            inp_pos = ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inp_pos, 1, n_tokens), inp_prune_ids), n_rows);

            inp_attn_prune->self_attn_mask.raw = prune_input->out_mask;
            inp_attn_prune->self_attn_mask_swa.raw = prune_input->out_mask_swa;
            inp_attn = inp_attn_prune.get();

            res->add_input(std::move(prune_input));
        }

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        if (full_sequence && activation_recompute) {
            inpL = ggml_view_2d(ctx0, inpL, inpL->ne[0], inpL->ne[1], inpL->nb[1], 0);
            inpL->flags |= GGML_TENSOR_FLAG_RECOMPUTE_INPUT;
        }
        res->t_layer_inp[il] = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Q projection (shared for both non-KV and KV layers)
        // this is to mirror Gemma4Attention in pytorch code
        ggml_tensor * qkv_fused = nullptr;
        ggml_tensor * Qcur;
        if (model.layers[il].wqkv) {
            qkv_fused = build_lora_mm(model.layers[il].wqkv, cur, model.layers[il].wqkv_s);
            cb(qkv_fused, "wqkv", il);
            const int64_t q_dim = n_embd_head * n_head;
            Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, qkv_fused, q_dim, n_rows, qkv_fused->nb[1], 0));
        } else {
            Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        }
        {
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_rows);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = build_rope(Qcur, inp_pos, il);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il) && !hparams.kvsharer_map[il]) {
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;
            if (qkv_fused) {
                const int64_t q_dim = n_embd_head * n_head;
                const int64_t k_dim = n_embd_head * n_head_kv;
                const size_t  esize = ggml_element_size(qkv_fused);
                Kcur = ggml_view_3d(ctx0, qkv_fused, n_embd_head, n_head_kv, n_tokens,
                        n_embd_head * esize, qkv_fused->nb[1], q_dim * esize);
                Vcur = ggml_view_3d(ctx0, qkv_fused, n_embd_head, n_head_kv, n_tokens,
                        n_embd_head * esize, qkv_fused->nb[1], (q_dim + k_dim) * esize);
                if (training) {
                    Kcur = ggml_cont(ctx0, Kcur);
                    Vcur = ggml_cont(ctx0, Vcur);
                }
            } else {
                Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
                Vcur = model.layers[il].wv
                       ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                       : Kcur; // if v_proj is not present, use Kcur as Vcur
                Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
            }
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = build_rope(Kcur, inp_pos, il);

            cb(Kcur, "Kcur_pos", il);

            if (hparams.kv_global_only && !hparams.is_swa(il)) {
                if (model.layers[il].wv || model.layers[il].wqkv) {
                    throw std::runtime_error("Lumen4Exp K/V reconstruction requires tied K/V projection");
                }
                Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, Vcur, n_rot_l, n_head_kv, n_tokens,
                        Vcur->nb[1], Vcur->nb[2], 0));
            }
            if (full_sequence && !mctx && hparams.kv_global_only && !hparams.is_swa(il)) {
                Kcur = build_kv_fake_quant(Kcur, GGML_TYPE_FP4_KV);
                Vcur = build_kv_fake_quant(Vcur, GGML_TYPE_FP4_KV);
            }
            if (full_sequence) {
                sequence_k[il] = Kcur;
                sequence_v[il] = Vcur;
            }

            cur = build_attn(inp_attn, model.layers[il].wo,
                    nullptr, model.layers[il].wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
        } else {
            // reuse KV cache of earlier layers
            int source_layer = hparams.kvsharer_map[il] ? hparams.kvsharer_map[il] - 1 : hparams.kv_global_only ? hparams.kv_source(il) :
                hparams.n_layer_kv_from_start - (hparams.is_swa(il) ? 2 : 1);
            if (hparams.kvsharer_map[source_layer]) { source_layer = hparams.kvsharer_map[source_layer] - 1; }
            ggml_tensor * shared_k = full_sequence ? sequence_k.at(source_layer) : nullptr;
            ggml_tensor * shared_v = full_sequence ? sequence_v.at(source_layer) : nullptr;
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, shared_k, shared_v, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        if (hparams.kv_global_only && il >= 11 && !hparams.is_swa(il)) {
            cb(cur, "cvec_attn", il);
            cur = build_cvec(cur, il);
        }

        // keep all rows when extracting unmasked nextn embeddings (MTP target needs the hidden state for every token)
        if (!prune_shared && il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network
        if (training && activation_recompute && !full_sequence) {
            attn_out->flags |= GGML_TENSOR_FLAG_RECOMPUTE_INPUT;
        }
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            ggml_tensor * router_scale = model.layers[il].ffn_gate_inp_s;
            if (router_scale->type != tmp->type) {
                router_scale = ggml_cast(ctx0, router_scale, tmp->type);
            }
            tmp = ggml_mul(ctx0, tmp, router_scale);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s,
                    /* selected_experts_in */ nullptr, /* post_exp_norm */ nullptr, 1e-5f, attn_out);
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            if (training && activation_recompute && !full_sequence) {
                cur->flags |= GGML_TENSOR_FLAG_RECOMPUTE;
            }
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor * ffn_down_input = nullptr;
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il, &ffn_down_input);
            GGML_ASSERT(ffn_down_input);
            if (training && activation_recompute && !full_sequence) {
                ffn_down_input->flags |= GGML_TENSOR_FLAG_RECOMPUTE;
            }
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);
        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer embedding
        if (inp_per_layer) {
            ggml_tensor * pe_in = cur;
            cb(cur, "pe_in", il);

            cur = build_lora_mm(model.layers[il].per_layer_inp_gate, cur); // [n_embd_per_layer, n_tokens]
            cur = build_gelu(cur);

            ggml_tensor * inp_this_layer = gemma4_view_2d_slice(ctx0, inp_per_layer, il); // [n_embd_per_layer, n_tokens]

            if (prune_shared && !hparams.has_kv(il)) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_prune_ids);
            } else if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
                inp_this_layer = ggml_get_rows(ctx0, inp_this_layer, inp_out_ids);
            }

            cur = ggml_mul(ctx0, cur, inp_this_layer);
            cur = build_lora_mm(model.layers[il].per_layer_proj, cur); // [n_embd, n_rows]
            cur = build_norm(cur, model.layers[il].per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "per_layer_embd_out", il);

            // residual connection
            cur = ggml_add(ctx0, pe_in, cur);
        }

        // layer_scalar
        if (model.layers[il].out_scale) {
            ggml_tensor * out_scale = model.layers[il].out_scale;
            if (out_scale->type != cur->type) {
                out_scale = ggml_cast(ctx0, out_scale, cur->type);
            }
            cur = ggml_mul(ctx0, cur, out_scale);
            cb(cur, "out_scaled", il);
        }

        if (!hparams.kv_global_only) {
            cur = build_cvec(cur, il);
        }
        if (full_sequence && activation_recompute) {
            cur->flags |= GGML_TENSOR_FLAG_RECOMPUTE;
            if (model.dev_layer(il) != ggml_backend_get_device(backend_cpu)) {
                // Save exact boundaries in RAM and keep the backward edge.
                cur = ggml_cont(ctx0, cur);
                cur->flags |= GGML_TENSOR_FLAG_HOST;
            }
        }
        cb(cur, "l_out", il);

        if (il == cparams.ream_calibration_layer) {
            ggml_build_forward_expand(gf, cur);
            return;
        }

        // input for next layer
        inpL = cur;
    }
    cur = inpL;
    if (prune_shared) {
        cur = ggml_get_rows(ctx0, cur, inp_prune_keep);
    }

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    // Expose the post-output-norm hidden state (the LM-head input feature) so that
    // MTP draft contexts can read it via llama_get_embeddings_nextn_ith() as the
    // recurrent h input. This matches the reference (transformers/vLLM/SGLang),
    // which feeds the drafter the target's post-final-norm hidden state.
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    const char * fused_head = std::getenv("GGML_DISTILL_FUSED_HEAD");
    if (params.top_k > 0 && params.top_k <= 512 && params.top_k < model.output->ne[1] &&
            !training && n_outputs > 0 && n_outputs <= 65535 && !replace_weight && loras->empty() &&
            !model.output_s && model.split_mode() == LLAMA_SPLIT_MODE_NONE &&
            (!fused_head || std::strcmp(fused_head, "0") != 0) &&
            (model.output->type == GGML_TYPE_F32 || model.output->type == GGML_TYPE_F16 ||
             model.output->type == GGML_TYPE_Q4_0 || model.output->type == GGML_TYPE_Q8_0) &&
            ggml_is_contiguous(model.output) && model.output->ne[0] <= 16384 && model.output->ne[1] <= 262144) {
        auto * hidden = build_hadamard_input(model.output, cur);
        hidden = ggml_is_contiguous(hidden) ? hidden : ggml_cont(ctx0, hidden);
        auto * packed = ggml_mul_mat_top_k(ctx0, model.output, hidden, params.top_k,
                params.top_k_temperature, hparams.f_final_logit_softcapping);
        if (model.dev_output() && ggml_backend_dev_supports_op(model.dev_output(), packed)) {
            LLAMA_LOG_DEBUG("distill_fused_head: vocab=%lld hidden=%lld rows=%lld k=%d\n",
                    (long long) model.output->ne[1], (long long) model.output->ne[0],
                    (long long) hidden->ne[1], params.top_k);
            cb(packed, "distill_fused_head", -1);
            res->t_top_k_packed = packed;
            return;
        }
    }
    const auto head = [&](ggml_tensor * input) {
        auto * logits = build_lora_mm(model.output, input, model.output_s);
        if (hparams.f_final_logit_softcapping) {
            logits = ggml_scale(ctx0, logits, 1.0f / hparams.f_final_logit_softcapping);
            logits = ggml_tanh(ctx0, logits);
            logits = ggml_scale(ctx0, logits, hparams.f_final_logit_softcapping);
        }
        if (full_sequence && activation_recompute) {
            input->flags |= GGML_TENSOR_FLAG_RECOMPUTE_INPUT;
            logits->flags |= GGML_TENSOR_FLAG_RECOMPUTE;
        }
        return logits;
    };
    if (full_sequence && cur->ne[1] > LLM_TRAINING_TILE_SIZE) {
        std::vector<ggml_tensor *> tiles;
        for (int64_t start = 0; start < cur->ne[1]; start += LLM_TRAINING_TILE_SIZE) {
            const int64_t count = std::min<int64_t>(LLM_TRAINING_TILE_SIZE, cur->ne[1] - start);
            auto * input = ggml_view_2d(ctx0, cur, cur->ne[0], count, cur->nb[1], start*cur->nb[1]);
            tiles.push_back(head(input));
        }
        while (tiles.size() > 1) {
            size_t count = 0;
            for (size_t i = 0; i < tiles.size(); i += 2) {
                tiles[count++] = i + 1 < tiles.size() ? ggml_concat(ctx0, tiles[i], tiles[i + 1], 1) : tiles[i];
            }
            tiles.resize(count);
        }
        cur = tiles[0];
    } else {
        cur = head(cur);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

class llm_graph_input_gemma4_ple : public llm_graph_input_i {
public:
    llm_graph_input_gemma4_ple(const llama_model & model) : model(model) {}

    void set_input(const llama_ubatch * ubatch) override {
        ggml_tensor * ple = model.per_layer_tok_embd;

        const bool prefetch = model.can_prefetch.count(ple);

        if (ubatch->token) {
            if (prefetch) {
                llama_prefetch_rows(ple, ubatch->token, ubatch->n_tokens);
            }
            ggml_backend_tensor_set(tokens, ubatch->token, 0, ubatch->n_tokens * ggml_element_size(tokens));
        } else {
            const int32_t padding = 0;
            if (prefetch) {
                // [TAG_GEMMA4_IMG_PADDING]
                llama_prefetch_rows(ple, &padding, 1);
            }
            ggml_backend_tensor_set(token0, &padding, 0, ggml_element_size(token0));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.token ? tokens && tokens->ne[0] == params.ubatch.n_tokens : tokens == nullptr;
    }

    ggml_tensor * tokens = nullptr;
    ggml_tensor * token0 = nullptr;

    const llama_model & model;
};

// equivalent to get_per_layer_inputs() in python code
// output shape: [n_embd_per_layer, n_layer, n_tokens]
ggml_tensor * llama_model_gemma4::graph::build_inp_per_layer() {
    auto inp = std::make_unique<llm_graph_input_gemma4_ple>(model);

    ggml_tensor * inp_per_layer;
    // mixed ubatch: embd rows have token id 0, same padding row as below
    // TODO: use ggml_build_forward_select
    if (ubatch.token) {
        inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp->tokens);
        res->t_inp_tokens = inp->tokens;

        inp_per_layer = ggml_get_rows(ctx0, model.per_layer_tok_embd, inp->tokens);
        cb(inp_per_layer, "inp_per_layer_selected", -1);
    } else {
        // Multimodal embedding path: use padding token (ID=0) embedding
        // TODO: verify if this is the correct behavior in transformers implementation

        // [TAG_GEMMA4_IMG_PADDING]
        inp->token0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_input(inp->token0);
        res->t_inp_tokens = inp->token0;

        inp_per_layer = ggml_get_rows(ctx0, model.per_layer_tok_embd, inp->token0);
        cb(inp_per_layer, "inp_per_layer_multimodal", -1);
    }

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(model.per_layer_tok_embd);
        if (!lw) { continue; }
        ggml_tensor * a = ggml_get_rows(ctx0, lw->a, inp->tokens ? inp->tokens : inp->token0);
        const float scale = lw->get_scale(lora.first->alpha, lora.second);
        inp_per_layer = ggml_add(ctx0, inp_per_layer, ggml_scale(ctx0, ggml_mul_mat(ctx0, lw->b, a), scale));
    }
    res->add_input(std::move(inp));
    return inp_per_layer;
}

// equivalent to project_per_layer_inputs() in python code
// this calculates the per-layer inputs, so the final tensor shape will have n_layer as the last dim
// inp_batch     shape: [n_embd, n_tokens]
// inp_per_layer shape: [n_embd_per_layer, n_layer, n_tokens] (from build_inp_per_layer)
// output shape: [n_embd_per_layer, n_tokens, n_layer]
ggml_tensor * llama_model_gemma4::graph::project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer) {
    const float per_layer_projection_scale = 1.0f / sqrtf((float) n_embd);
    const float per_layer_input_scale      = 1.0f / sqrtf(2.0f);

    // note: this matrix multiplication will be performed in the input layer (i.e. on the CPU)
    // fold per_layer_input_scale into proj_norm_w and per_layer_projection_scale into proj_eps
    ggml_tensor * proj_norm_w = ggml_scale(ctx0, model.per_layer_proj_norm, per_layer_input_scale);
    ggml_build_forward_expand(gf, proj_norm_w);

    ggml_tensor * per_layer_proj;
    per_layer_proj = build_lora_mm (model.per_layer_model_proj, inp_batch);
    per_layer_proj = ggml_reshape_3d(ctx0, per_layer_proj, n_embd_per_layer, n_layer, n_tokens);

    const float proj_eps = hparams.f_norm_rms_eps / (per_layer_projection_scale * per_layer_projection_scale);
    per_layer_proj = ggml_rms_norm(ctx0, per_layer_proj, proj_eps);
    per_layer_proj = ggml_mul(ctx0, per_layer_proj, proj_norm_w);
    cb(per_layer_proj, "per_layer_proj", -1);

    inp_per_layer = ggml_add(ctx0, per_layer_proj, inp_per_layer);
    cb(inp_per_layer, "inp_per_layer", -1);

    // permute to shape: [n_embd_per_layer, n_tokens, n_layer]
    return ggml_permute(ctx0, inp_per_layer, 0, 2, 1, 3);
}

void llm_graph_input_gemma4_prune::set_input(const llama_ubatch * ubatch) {
    std::vector<std::vector<int32_t>> selected(n_slices);

    for (int32_t s = 0; s < n_slices; ++s) {
        for (int32_t t = 0; t < n_slice_tokens; ++t) {
            const int32_t i = s * n_slice_tokens + t;
            if (ubatch->output[i]) {
                selected[s].push_back(i);
            }
        }
    }

    std::vector<int32_t> take, keep;
    for (int32_t s = 0; s < n_slices; ++s) {
        const auto & ids = selected[s];

        for (int32_t j = 0; j < int32_t(ids.size()); ++j) {
            keep.push_back(s * n_slice_outputs + j);
        }

        take.insert(take.end(), ids.begin(), ids.end());
        const int32_t dummy = ids.empty() ? s * n_slice_tokens : ids.back();
        take.insert(take.end(), n_slice_outputs - ids.size(), dummy);
    }

    GGML_ASSERT(take.size() == size_t(inp_prune_ids->ne[0]));
    if (!take.empty()) {
        ggml_backend_tensor_set(inp_prune_ids, take.data(), 0,
                                take.size() * sizeof(int32_t));
    }

    auto copy_mask = [&](const ggml_tensor * full, ggml_tensor * out) {
        if (take.empty() || !out->buffer) {
            return;
        }

        GGML_ASSERT(full->buffer && ggml_backend_buffer_is_host(full->buffer));
        const size_t row_bytes = full->ne[0] * ggml_element_size(full);
        const auto * src = static_cast<const uint8_t *>(full->data);
        std::vector<uint8_t> packed(take.size() * row_bytes);

        for (size_t j = 0; j < take.size(); ++j) {
            memcpy(packed.data() + j * row_bytes,
                        src + take[j] * row_bytes,
                        row_bytes);
        }

        ggml_backend_tensor_set(out, packed.data(), 0, packed.size());
    };

    copy_mask(full_mask, out_mask);
    copy_mask(full_mask_swa, out_mask_swa);

    if (!keep.empty()) {
        ggml_backend_tensor_set(
            inp_prune_keep, keep.data(), 0,
            keep.size() * sizeof(int32_t));
    }
}

bool llm_graph_input_gemma4_prune::can_reuse(const llm_graph_params & params) {
    const auto & ubatch = params.ubatch;

    if (ubatch.n_tokens != n_slices * n_slice_tokens || !ubatch.output) {
        return false;
    }

    return max_outputs_per_slice(ubatch, n_slices, n_slice_tokens) == n_slice_outputs;
}

int64_t llm_graph_input_gemma4_prune::max_outputs_per_slice(const llama_ubatch & ubatch, int64_t n_slices, int64_t n_slice_tokens) {
    int64_t res = 0;

    for (int32_t s = 0; s < n_slices; ++s) {
        int64_t m = 0;
        for (int32_t t = 0; t < n_slice_tokens; ++t) {
            const int32_t i = s * n_slice_tokens + t;
            if (ubatch.output[i]) {
                m++;
            }
        }
        res = std::max(res, m);
    }
    return res;
}
