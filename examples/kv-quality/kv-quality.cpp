#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf -p \"Hello\" -n 128 -c 2048 -b 2048 -ngl 99 -fa on -ctlk q8_0 -ctlv q8_0 -ctgk f16 -ctgv f16\n", argv[0]);
    printf("\n");
}

struct divergence_stats {
    double kl_pq = 0.0;
    double kl_qp = 0.0;
    double js    = 0.0;
    llama_token token_ref       = LLAMA_TOKEN_NULL;
    llama_token token_candidate = LLAMA_TOKEN_NULL;
};

static divergence_stats compare_logits(const float * logits_ref, const float * logits_candidate, int n_vocab) {
    divergence_stats result;

    const float max_ref       = *std::max_element(logits_ref,       logits_ref       + n_vocab);
    const float max_candidate = *std::max_element(logits_candidate, logits_candidate + n_vocab);

    double sum_ref       = 0.0;
    double sum_candidate = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_ref       += std::exp(double(logits_ref[i]       - max_ref));
        sum_candidate += std::exp(double(logits_candidate[i] - max_candidate));
    }

    const double log_sum_ref       = std::log(sum_ref);
    const double log_sum_candidate = std::log(sum_candidate);
    for (int i = 0; i < n_vocab; ++i) {
        const double log_p = double(logits_ref[i]       - max_ref)       - log_sum_ref;
        const double log_q = double(logits_candidate[i] - max_candidate) - log_sum_candidate;
        const double p = std::exp(log_p);
        const double q = std::exp(log_q);
        const double m = 0.5*(p + q);

        result.kl_pq += p*(log_p - log_q);
        result.kl_qp += q*(log_q - log_p);
        if (p > 0.0) {
            result.js += 0.5*p*(log_p - std::log(m));
        }
        if (q > 0.0) {
            result.js += 0.5*q*(log_q - std::log(m));
        }
    }

    result.token_ref       = std::max_element(logits_ref,       logits_ref       + n_vocab) - logits_ref;
    result.token_candidate = std::max_element(logits_candidate, logits_candidate + n_vocab) - logits_candidate;
    return result;
}

static bool decode(llama_context * ctx, llama_token * tokens, int n_tokens) {
    if (llama_decode(ctx, llama_batch_get_one(tokens, n_tokens)) != 0) {
        fprintf(stderr, "failed to decode %d tokens\n", n_tokens);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {
        return 1;
    }
    if (params.n_predict <= 0) {
        params.n_predict = 128;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model * model         = llama_init->model();
    llama_context * ctx_candidate = llama_init->context();
    if (model == nullptr || ctx_candidate == nullptr) {
        fprintf(stderr, "failed to initialize candidate context\n");
        return 1;
    }

    llama_context_params ref_params = common_context_params_to_llama(params);
    ref_params.type_k     = GGML_TYPE_F16;
    ref_params.type_v     = GGML_TYPE_F16;
    ref_params.type_k_swa = GGML_TYPE_F16;
    ref_params.type_v_swa = GGML_TYPE_F16;

    llama_context * ctx_ref = llama_init_from_model(model, ref_params);
    if (ctx_ref == nullptr) {
        fprintf(stderr, "failed to initialize reference context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt = common_tokenize(ctx_ref, params.prompt, llama_vocab_get_add_bos(vocab));
    if (prompt.empty()) {
        fprintf(stderr, "prompt has no tokens\n");
        llama_free(ctx_ref);
        return 1;
    }
    if (prompt.size() > size_t(params.n_batch)) {
        fprintf(stderr, "prompt has %zu tokens but batch size is %d\n", prompt.size(), params.n_batch);
        llama_free(ctx_ref);
        return 1;
    }

    if (!decode(ctx_ref, prompt.data(), prompt.size()) || !decode(ctx_candidate, prompt.data(), prompt.size())) {
        llama_free(ctx_ref);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    double sum_kl_pq = 0.0;
    double sum_kl_qp = 0.0;
    double sum_js    = 0.0;
    double max_kl_pq = 0.0;
    double max_kl_qp = 0.0;
    double max_js    = 0.0;
    int first_divergence = 0;
    int n_compared = 0;

    for (int step = 1; step <= params.n_predict; ++step) {
        const float * logits_ref       = llama_get_logits(ctx_ref);
        const float * logits_candidate = llama_get_logits(ctx_candidate);
        if (logits_ref == nullptr || logits_candidate == nullptr) {
            fprintf(stderr, "failed to get logits at step %d\n", step);
            llama_free(ctx_ref);
            return 1;
        }

        const divergence_stats stats = compare_logits(logits_ref, logits_candidate, n_vocab);
        if (!std::isfinite(stats.kl_pq) || !std::isfinite(stats.kl_qp) || !std::isfinite(stats.js)) {
            fprintf(stderr, "non-finite divergence at step %d\n", step);
            llama_free(ctx_ref);
            return 1;
        }

        sum_kl_pq += stats.kl_pq;
        sum_kl_qp += stats.kl_qp;
        sum_js    += stats.js;
        max_kl_pq = std::max(max_kl_pq, stats.kl_pq);
        max_kl_qp = std::max(max_kl_qp, stats.kl_qp);
        max_js    = std::max(max_js, stats.js);
        ++n_compared;

        const bool same = stats.token_ref == stats.token_candidate;
        printf("{\"step\":%d,\"token_ref\":%d,\"token_candidate\":%d,\"same\":%s,\"kl_pq\":%.12g,\"kl_qp\":%.12g,\"js\":%.12g}\n",
                step, stats.token_ref, stats.token_candidate, same ? "true" : "false", stats.kl_pq, stats.kl_qp, stats.js);

        if (!same && first_divergence == 0) {
            first_divergence = step;
        }
        if (llama_vocab_is_eog(vocab, stats.token_ref)) {
            break;
        }

        llama_token token = stats.token_ref;
        if (!decode(ctx_ref, &token, 1) || !decode(ctx_candidate, &token, 1)) {
            llama_free(ctx_ref);
            return 1;
        }
    }

    printf("{\"summary\":true,\"post_divergence_history\":\"reference\",\"prompt_tokens\":%zu,\"steps_compared\":%d,\"first_divergence\":%d,\"mean_kl_pq\":%.12g,\"max_kl_pq\":%.12g,\"mean_kl_qp\":%.12g,\"max_kl_qp\":%.12g,\"mean_js\":%.12g,\"max_js\":%.12g}\n",
            prompt.size(), n_compared, first_divergence,
            sum_kl_pq/n_compared, max_kl_pq, sum_kl_qp/n_compared, max_kl_qp, sum_js/n_compared, max_js);

    llama_free(ctx_ref);
    llama_backend_free();
    return 0;
}
