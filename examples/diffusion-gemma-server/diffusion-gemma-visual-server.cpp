// Persistent "visual" generation server for DiffusionGemma: load the GGUF once, then run the OPTIMIZED
// entropy-bound decoder (the same diffusion_generate_entropy_bound the CLI's --diffusion-visual uses)
// and stream the per-step argmax canvas back so a UI can watch the denoise resolve in place. Unlike the
// raw-logits server, NO [C, n_vocab] logits are shipped to the client and no host-side sampling happens:
// argmax/entropy/multinomial and self-conditioning stay on the GPU (Stage 1 + Stage 2).
//
// Tokenization, chat templating and detokenization all happen here, from the GGUF's own embedded tokenizer
// + chat template (same path as llama-diffusion-cli), so the client needs no tokenizer files of its own.
//
// Protocol (synchronous, one request per line on stdin):
//   stdin  : a line containing a request-file path R
//   file R : UTF-8 JSON  {"seed": <int>, "n_blocks": <int>, "messages": [ {"role","content"}, ... ]}
//            (messages are OpenAI chat-completion format; the GGUF chat template is applied here)
//   stdout : a stream of newline records, then "DONE":
//              F <block> <step> <total> <json-string>   one per denoising step (current canvas, decoded)
//              C <block> <json-string>                  cumulative committed answer text after this block
//              DONE                                      end of this request
//              ERR <msg>                                request failed
//   "QUIT"/EOF -> exit.
//
// Usage: llama-diffusion-gemma-visual-server <model.gguf>   (env NGL for gpu layers, MAXTOK, FA for flash-attn)

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "../diffusion/diffusion.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string read_text_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s;
    if (sz > 0) {
        s.resize((size_t) sz);
        if (fread(&s[0], 1, (size_t) sz, f) != (size_t) sz) { fclose(f); return {}; }
    }
    fclose(f);
    return s;
}

// per-request callback state: which block we are on, where to stream frames, and how to decode them
struct vis_cb_data {
    int                  block   = 0;
    int                  n_input = 0;
    FILE *               out     = nullptr;
    const llama_vocab *  vocab   = nullptr;
};

// Stream the current argmax canvas (tokens[n_input .. n_tokens)) as one "F" record per denoising step,
// decoded to text and JSON-escaped so it survives the line protocol intact (spaces/newlines/unicode).
static bool vis_step_callback(int32_t step, int32_t total_steps, const llama_token * tokens,
                              int32_t n_tokens, void * user_data) {
    auto * d = (vis_cb_data *) user_data;
    const int n_canvas = n_tokens - d->n_input;
    if (n_canvas <= 0) return true;
    std::vector<llama_token> canvas(tokens + d->n_input, tokens + n_tokens);
    const std::string text = common_detokenize(d->vocab, canvas, /*special*/ false);
    fprintf(d->out, "F %d %d %d %s\n", d->block, step, total_steps, nlohmann::json(text).dump().c_str());
    fflush(d->out);
    return true;
}

// Trim a denoised canvas like the CLI: cut at the first end-of-generation token, else at the onset of a
// repetition loop (a token recurring at stride 1-2 for >= 6 steps).
static size_t trim_canvas(const llama_vocab * vocab, const llama_token * canvas, size_t n) {
    size_t cut = n;
    for (size_t i = 0; i < n; i++) {
        if (llama_vocab_is_eog(vocab, canvas[i])) { cut = i; break; }
    }
    for (size_t i = 0; i + 1 < cut; i++) {
        bool loop = false;
        for (size_t stride = 1; stride <= 2 && !loop; stride++) {
            size_t reps = 0;
            for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) { reps++; }
            loop = reps >= 6;
        }
        if (loop) { cut = i; break; }
    }
    return cut;
}

static float meta_f(llama_model * m, const char * key, float def) {
    char buf[32];
    return llama_model_meta_val_str(m, key, buf, sizeof(buf)) >= 0 ? strtof(buf, nullptr) : def;
}
static int32_t meta_i(llama_model * m, const char * key, int32_t def) {
    char buf[32];
    return llama_model_meta_val_str(m, key, buf, sizeof(buf)) >= 0 ? (int32_t) strtol(buf, nullptr, 10) : def;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 1; }
    const int MAXTOK = atoi(getenv("MAXTOK") ? getenv("MAXTOK") : "8192");
    const int MAXTOK_ENV = atoi(getenv("MAXTOK") ? getenv("MAXTOK") : "0");
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = atoi(getenv("NGL") ? getenv("NGL") : "0");
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    if (!llama_model_is_diffusion(model)) { fprintf(stderr, "not a diffusion model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // chat template + tokenizer come from the GGUF itself (same as the CLI): no client-side tokenizer needed
    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, "");

    int64_t canvas_length = 0;
    {
        char canvas_str[32];
        if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_str, sizeof(canvas_str)) >= 0) {
            canvas_length = strtol(canvas_str, nullptr, 10);
        }
    }
    if (canvas_length <= 0) { fprintf(stderr, "model has no diffusion.canvas_length\n"); return 1; }

    // Enable the self-conditioning graph before context creation so the reserve sizes the compute buffer
    // (matches the CLI). The entropy-bound decoder supplies the real SC state per step.
    llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);

    // Device enumeration: count GPUs (Stage 1+2 are single-device features) and grab a GPU device handle
    // so the auto-sizer can read free VRAM. Done before context creation: the weights are already resident,
    // so free VRAM here is exactly the budget left for the per-turn compute buffer.
    int gpu_devs = 0;
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto dt = ggml_backend_dev_type(d);
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_devs++;
            if (!gpu_dev) gpu_dev = d;
        }
    }
    const bool one_gpu = (gpu_devs <= 1);

    const bool fa_on  = getenv("FA") && atoi(getenv("FA"));
    const int  n_head = std::max(1, (int) llama_model_n_head(model));
    auto make_cparams = [&](int n) {
        llama_context_params c = llama_context_default_params();
        c.n_ctx   = (uint32_t) n;
        c.n_batch = (uint32_t) n;
        // chunked causal prefill: keep n_head*chunk*n_kv under 2^31 (CUDA softcap is 32-bit indexed)
        const int chunk = (int) std::clamp<int64_t>((int64_t(1) << 30) / (int64_t(n_head) * n), 256, 2048);
        c.n_ubatch = (uint32_t) std::min(n, chunk);
        c.n_outputs_max = (uint32_t) canvas_length;   // only the canvas rows need logits
        c.no_perf = true;
        c.flash_attn_type = fa_on ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        return c;
    };

    // Resolve MAXTOK + create the context. A descending probe keeps the largest context that actually
    // allocates. The model can spill to system RAM (NGL exceeds what fits VRAM), so when the VRAM-gated
    // pass collapses we re-probe against free RAM -- the per-turn compute buffer lives wherever the
    // layers landed. llama_init_from_model returns null on OOM (graph_reserve throws), so probing is safe.
    const int n_ctx_train = (int) llama_model_n_ctx_train(model);
    const int floor_ctx   = std::max((int) canvas_length * 4, 2048);
    const int auto_ceil   = n_ctx_train > 0 ? std::min(n_ctx_train, 65536) : 65536;
    const int cands[] = {65536, 49152, 40960, 32768, 24576, 20480, 16384, 12288, 8192, 6144, 4096, 2048};

    // VRAM budget (where NGL asks the model to live) and RAM budget (where it spills if it overflows VRAM).
    size_t v_free = 0, v_total = 0;
    if (gpu_dev) ggml_backend_dev_memory(gpu_dev, &v_free, &v_total);
    if (const char * e = getenv("DG_FREE_VRAM_MB")) {   // diagnostics: simulate a small card (probe only)
        v_free = v_total = (size_t) atoll(e) * 1024 * 1024;
        fprintf(stderr, "DG_FREE_VRAM_MB=%s active\n", e);
    }
    size_t r_free = 0, r_total = 0;
    if (ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))
        ggml_backend_dev_memory(cpu_dev, &r_free, &r_total);
    const size_t weights = llama_model_size(model);
    size_t ram_budget = r_free > weights ? (size_t) ((r_free - weights) * 0.7) : 0;  // leave room for KV/OS
    if (const char * e = getenv("DG_FREE_RAM_MB")) {    // diagnostics: simulate tight RAM (probe only)
        ram_budget = (size_t) atoll(e) * 1024 * 1024;
        fprintf(stderr, "DG_FREE_RAM_MB=%s active\n", e);
    }

    // Probe descending candidates within `budget` bytes; return the largest that allocates. gpu_headroom>0
    // also requires that much VRAM free after creation (the per-step pool); 0 skips that GPU-only check.
    auto probe = [&](int ceil_ctx, size_t budget, size_t gpu_headroom, int * out_n) -> llama_context * {
        for (int raw : cands) {
            if (raw > ceil_ctx) continue;
            int N = (int) ((raw / canvas_length) * canvas_length);   // whole canvases only
            if (N < floor_ctx) break;
            if (budget) {   // an fp32 [n_head, N, N] scores buffer is unavoidable (FA off): skip if it can't fit
                const double min_scores = (double) n_head * (double) N * (double) N * 4.0;
                if (min_scores > (double) budget * 0.9) continue;
            }
            llama_context * c = llama_init_from_model(model, make_cparams(N));
            if (!c) continue;
            if (gpu_headroom && gpu_dev) {
                size_t f = 0, t = 0; ggml_backend_dev_memory(gpu_dev, &f, &t);
                if (f < gpu_headroom) { llama_free(c); continue; }
            }
            *out_n = N;
            return c;
        }
        return nullptr;
    };

    MAXTOK = 0;
    llama_context * ctx = nullptr;
    const char * reason = "auto";

    if (MAXTOK_ENV > 0) {   // explicit budget: honour exactly if it fits, else degrade through the probe
        const double sc = (double) n_head * (double) MAXTOK_ENV * (double) MAXTOK_ENV * 4.0;
        const size_t budget = std::max(v_free, ram_budget);
        if (!budget || sc <= (double) budget * 0.9) {
            ctx = llama_init_from_model(model, make_cparams(MAXTOK_ENV));
            if (ctx) { MAXTOK = MAXTOK_ENV; reason = "requested"; }
        }
    }
    if (!ctx) {
        const int ceil_ctx = MAXTOK_ENV > 0 ? std::min(auto_ceil, MAXTOK_ENV) : auto_ceil;
        const size_t vram_headroom = v_total ? (size_t) (v_total * 0.08) : (size_t) 1536 * 1024 * 1024;
        int n1 = 0;
        ctx = probe(ceil_ctx, v_free, vram_headroom, &n1);       // pass 1: VRAM-gated (unchanged on ample VRAM)
        if (ctx) { MAXTOK = n1; reason = "vram"; }
        if ((!ctx || n1 < ceil_ctx) && ram_budget > v_free) {    // pass 2: model is RAM-resident -- probe RAM
            int n2 = 0;
            llama_context * c = probe(ceil_ctx, ram_budget, 0, &n2);
            if (c && n2 > MAXTOK) { if (ctx) llama_free(ctx); ctx = c; MAXTOK = n2; reason = "ram"; }
            else if (c) { llama_free(c); }
        }
    }
    if (!ctx) {   // last resort: the floor so a very tight machine still loads
        int N = std::max((int) canvas_length, (int) ((floor_ctx / canvas_length) * canvas_length));
        ctx = llama_init_from_model(model, make_cparams(N));
        if (ctx) { MAXTOK = N; reason = "floor"; }
    }
    if (!ctx) {
        fprintf(stderr, "failed to size a context that fits (VRAM free=%zu MiB, RAM budget=%zu MiB): model too "
                "large for this machine. Try a smaller quant, lower NGL, or more VRAM/RAM.\n",
                v_free / (1024 * 1024), ram_budget / (1024 * 1024));
        return 1;
    }
    fprintf(stderr, "context: MAXTOK=%d requested=%d budget=%s\n", MAXTOK, MAXTOK_ENV, reason);
    llama_set_causal_attn(ctx, false);

    // entropy-bound params from GGUF metadata + reference defaults (kept in sync with the CLI)
    diffusion_eb_params base;
    base.max_denoising_steps  = meta_i(model, "diffusion.eb_max_steps", 48);
    base.t_min                = meta_f(model, "diffusion.eb_t_min", 0.4f);
    base.t_max                = meta_f(model, "diffusion.eb_t_max", 0.8f);
    base.entropy_bound        = meta_f(model, "diffusion.eb_entropy_bound", 0.1f);
    base.stability_threshold  = meta_i(model, "diffusion.eb_stability_threshold", 1);
    base.confidence_threshold = meta_f(model, "diffusion.eb_confidence_threshold", 0.005f);

    // Stage 1 + Stage 2 are single-device features (sc_dev / prompt-KV store are single-GPU). Auto-enable
    // them for one CUDA device, exactly like the CLI's --diffusion-* auto resolution.
    gpu_devs = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const auto dt = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) { gpu_devs++; }
    }
    //const bool one_gpu = (gpu_devs <= 1);
    base.kv_cache          = one_gpu;
    base.gpu_sampling      = one_gpu;
    base.gpu_sample_reduce = one_gpu;

    std::vector<llama_token> output_tokens(MAXTOK);

    fprintf(stderr, "diffusion-gemma-visual-server ready (n_vocab=%d, canvas=%d, MAXTOK=%d, NGL=%d, "
            "gpu_sampling=%s sample_reduce=%s kv_cache=%s)\n",
            n_vocab, (int) canvas_length, MAXTOK, mparams.n_gpu_layers,
            base.gpu_sampling ? "on" : "off", base.gpu_sample_reduce ? "on" : "off",
            base.kv_cache ? "on" : "off");
    printf("READY %d\n", n_vocab); fflush(stdout);

    char line[4096];
    while (fgets(line, sizeof(line), stdin)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        if (strcmp(line, "QUIT") == 0) break;

        // parse the request file: {"seed", "n_blocks", "messages":[...]} -> chat template -> token prefix
        int seed = 0, n_blocks = 1;
        std::vector<llama_token> prefix;
        try {
            const std::string raw = read_text_file(line);
            if (raw.empty()) { printf("ERR badreq\n"); fflush(stdout); continue; }
            const nlohmann::ordered_json req = nlohmann::ordered_json::parse(raw);
            seed     = req.value("seed", 0);
            n_blocks = req.value("n_blocks", 1);
            std::vector<common_chat_msg> messages = common_chat_msgs_parse_oaicompat(req.at("messages"));
            common_chat_templates_inputs inputs;
            inputs.messages              = messages;
            inputs.add_generation_prompt = true;
            const std::string prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            prefix = common_tokenize(vocab, prompt, /*add special*/ true, /*parse special*/ true);
        } catch (const std::exception & e) {
            printf("ERR parse %s\n", e.what()); fflush(stdout); continue;
        }
        if (prefix.empty()) { printf("ERR emptyprompt\n"); fflush(stdout); continue; }

        const int P = (int) prefix.size();          // original prompt length; the answer is what grows past it
        std::vector<llama_token> answer;             // cumulative committed canvas tokens (across blocks)

        for (int b = 0; b < std::max(1, n_blocks); b++) {
            const int32_t prefix_len = (int32_t) prefix.size();
            const int32_t max_length = prefix_len + (int32_t) canvas_length;
            if (max_length > MAXTOK) { printf("ERR toolong %d\n", (int) max_length); break; }

            diffusion_eb_params eb = base;
            eb.max_length              = max_length;
            eb.seed                    = seed + b;   // distinct per block, deterministic from the request seed
            eb.visual_mode             = true;
            vis_cb_data cb{ b, prefix_len, stdout, vocab };
            eb.step_callback           = vis_step_callback;
            eb.step_callback_user_data = &cb;

            int32_t n_generated = 0;
            diffusion_generate_entropy_bound(ctx, prefix.data(), output_tokens.data(), prefix_len, eb, n_generated);
            if (n_generated <= prefix_len) { if (b == 0) printf("ERR gen\n"); break; }

            const llama_token * canvas = output_tokens.data() + prefix_len;
            const size_t cut = trim_canvas(vocab, canvas, (size_t) canvas_length);

            answer.insert(answer.end(), canvas, canvas + cut);
            const std::string answer_text = common_detokenize(vocab, answer, /*special*/ false);
            printf("C %d %s\n", b, nlohmann::json(answer_text).dump().c_str()); fflush(stdout);

            if (cut < (size_t) canvas_length) break;                 // eog / repetition loop: answer complete
            prefix.insert(prefix.end(), canvas, canvas + cut);       // commit the block, denoise the next
        }
        (void) P;
        printf("DONE\n"); fflush(stdout);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
