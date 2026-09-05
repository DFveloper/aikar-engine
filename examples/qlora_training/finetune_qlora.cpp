// QLoRA fine-tuning for quantized GGUF models.
//
// The base model weights stay frozen (quantized tensors are skipped by
// llama_set_param because they are not GGML_TYPE_F32).  Only the freshly
// allocated F32 LoRA A/B tensors are trained.  After training the adapter
// is saved as a GGUF file that is directly compatible with the existing
// llama_adapter_lora_init() loader and llama-export-lora merge tool.
//
// Usage example:
/*   llama-finetune-qlora \
         --model model-q4_k_m.gguf \
         --train-file train.jsonl \
         --lora-rank 16 --lora-alpha 16 \
         --lr-scheduler cosine --warmup-steps 10 --warmup-init-ratio 0.1 \
         --lora-out adapter.gguf \
         --epochs 3 -c 4096 -b 4096 -ub 512
*/
// Default targets: attn_q, attn_output, ffn_gate, ffn_up, ffn_down
// Override with --lora-targets "comma,separated,substrings"
//
// NOTE: attn_k and attn_v are excluded from defaults.  The KV write path uses
// ggml_set_rows (scatter op) — backward cannot propagate gradients through it.
// LoRA K/V would receive zero gradient.
//
// NOTE: ssm_in and ssm_out (Mamba/NemotronH) are excluded from defaults.
// SSM_SCAN/SSM_CONV have no backward implementation — LoRA on these layers
// would receive zero gradient.  Adding them wastes memory with no benefit.
//
// NOTE: MoE expert tensors (*_exps) are excluded regardless of --lora-targets.
// The quantized expert weights are frozen (stop-gradient), but LoRA on dense
// FFN layers (ffn_gate, ffn_up, ffn_down) works via MUL_MAT_ID backward.
//
// Target substrings use llama.cpp internal GGUF names (NOT HuggingFace names):
//   attn_q      = q_proj       attn_k     = k_proj
//   attn_v      = v_proj       attn_output= o_proj
//   ffn_gate    = gate_proj    ffn_up     = up_proj    ffn_down = down_proj
//   ssm_in      = in_proj (Mamba/NemotronH)  — zero gradient, not in defaults
//   ssm_out     = out_proj (Mamba/NemotronH)  — zero gradient, not in defaults

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "jsonl.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "critical-token-sft.h"

// Internal adapter struct — included directly to avoid the temp-GGUF roundtrip
// for wiring trainable LoRA tensors into the compute graph.
#include "../../src/llama-adapter.h"

#include <cerrno>
#include <csignal>
#include <iostream>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <clocale>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Expand a leading ~/ to the HOME directory (the shell doesn't do this for us
// when a path is passed as a string argument to std::ofstream).
static std::string expand_tilde(const std::string & path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        const char * home = getenv("HOME");
        if (!home) home = getenv("USERPROFILE"); // Windows fallback
        if (home) return std::string(home) + path.substr(1);
    }
    return path;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

static std::string format_duration_seconds(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return "unknown";
    }
    const int64_t total = (int64_t) std::llround(seconds);
    const int64_t hours = total / 3600;
    const int64_t mins = (total % 3600) / 60;
    const int64_t secs = total % 60;
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%02ld:%02ld:%02ld", (long) hours, (long) mins, (long) secs);
    return buffer;
}

static std::string format_progress_bar(size_t completed, size_t total, size_t width = 30) {
    const size_t filled = total > 0 ? std::min(width, completed * width / total) : width;
    return std::string(filled, '=') + (filled < width ? ">" : "") +
           std::string(width - filled - (filled < width ? 1 : 0), '.');
}

class loop_eta_progress {
public:
    loop_eta_progress(const char * phase, size_t total) : phase(phase), total(total),
            started(std::chrono::steady_clock::now()), next_report(started + std::chrono::seconds(5)) {
        LOG_INF("data_progress: phase=%s started total=%zu\n", phase, total);
    }

    void update(size_t completed, bool force = false) {
        const auto now = std::chrono::steady_clock::now();
        if (!force && now < next_report) {
            return;
        }
        next_report = now + std::chrono::seconds(5);
        const double elapsed = std::chrono::duration<double>(now - started).count();
        const double rate = elapsed > 0.0 ? completed / elapsed : 0.0;
        const double eta = rate > 0.0 ? (total - std::min(completed, total)) / rate : INFINITY;
        LOG_INF("data_progress: phase=%s [%s] %zu/%zu (%.1f%%) rate=%.1f/s elapsed=%s eta=%s\n",
                phase, format_progress_bar(completed, total).c_str(), completed, total,
                total > 0 ? 100.0 * completed / total : 100.0, rate,
                format_duration_seconds(elapsed).c_str(), format_duration_seconds(eta).c_str());
    }

    void finish() {
        update(total, true);
    }

private:
    const char * phase;
    size_t total;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point next_report;
};

static void parallel_for_with_eta(
        common_jsonl_worker_pool                         & pool,
        size_t                                             n_tasks,
        size_t                                             n_workers,
        const char                                       * phase,
        const std::function<void(size_t, size_t)>         & fn) {
    std::atomic<size_t> completed { 0 };
    std::mutex progress_mutex;
    std::condition_variable progress_cv;
    bool finished = false;
    const auto started = std::chrono::steady_clock::now();

    LOG_INF("load_jsonl: phase=%s started total=%zu workers=%zu\n", phase, n_tasks, n_workers);

    std::thread reporter([&] {
        std::unique_lock<std::mutex> lock(progress_mutex);
        while (!progress_cv.wait_for(lock, std::chrono::seconds(5), [&] { return finished; })) {
            const size_t done = completed.load(std::memory_order_relaxed);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const double rate = elapsed > 0.0 ? done / elapsed : 0.0;
            const double eta = rate > 0.0 ? (n_tasks - done) / rate : INFINITY;
            LOG_INF("load_jsonl: phase=%s [%s] %zu/%zu (%.1f%%) rate=%.1f/s elapsed=%s eta=%s workers=%zu\n",
                    phase, format_progress_bar(done, n_tasks).c_str(), done, n_tasks,
                    n_tasks > 0 ? 100.0 * done / n_tasks : 100.0, rate,
                    format_duration_seconds(elapsed).c_str(), format_duration_seconds(eta).c_str(), n_workers);
        }
    });

    try {
        pool.parallel_for(n_tasks, [&](size_t i, size_t worker_index) {
            fn(i, worker_index);
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            finished = true;
        }
        progress_cv.notify_one();
        reporter.join();
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(progress_mutex);
        finished = true;
    }
    progress_cv.notify_one();
    reporter.join();

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    LOG_INF("load_jsonl: phase=%s [%s] complete=%zu/%zu elapsed=%s average_rate=%.1f/s workers=%zu\n",
            phase, format_progress_bar(n_tasks, n_tasks).c_str(), completed.load(std::memory_order_relaxed), n_tasks,
            format_duration_seconds(elapsed).c_str(), elapsed > 0.0 ? n_tasks / elapsed : 0.0, n_workers);
}

static int64_t train_split_from_val_fraction(int64_t ndata, float val_split) {
    if (ndata <= 0) {
        return 0;
    }
    if (val_split <= 0.0f) {
        return ndata;
    }
    if (val_split >= 1.0f) {
        return 0;
    }

    const int64_t split = (int64_t) (ndata * (1.0f - val_split));
    return std::max<int64_t>(1, split);
}

static std::string preview_text(const std::string & s, size_t max_len = 240) {
    std::string out;
    out.reserve(std::min(s.size(), max_len));
    for (char c : s) {
        if (out.size() >= max_len) {
            out += "...";
            break;
        }
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out += c;
        }
    }
    return out;
}

struct qlora_lr_schedule {
    const lr_opt * lr;
    std::string    type;
    int64_t        warmup_steps;
    float          warmup_init_ratio;
    int64_t        decay_steps;
    int64_t        total_steps;
    int64_t        step;
    float          current_lr = 0.0f;

    float get_lr() const {
        const float lr_base = lr->lr0;
        if (step < warmup_steps) {
            const float lr_start = lr_base * warmup_init_ratio;
            const float progress = (float) step / (float) warmup_steps;
            return lr_start + (lr_base - lr_start) * progress;
        }
        if (type != "cosine" || total_steps <= warmup_steps) {
            return lr->get_lr();
        }

        const float lr_min = std::max(0.0f, lr->lr_min);
        const int64_t cosine_end = decay_steps > 0 ? std::min(decay_steps, total_steps) : total_steps;
        if (cosine_end <= warmup_steps) {
            return lr_min;
        }
        const double progress = (double) (step - warmup_steps) / (double) (cosine_end - warmup_steps);
        const double cosine = 0.5 * (1.0 + std::cos(std::acos(-1.0) * std::min(1.0, progress)));
        return lr_min + (lr_base - lr_min) * (float) cosine;
    }
};

static ggml_opt_optimizer_params qlora_opt_lr_pars(void * userdata) {
    qlora_lr_schedule & schedule = *(qlora_lr_schedule *) userdata;
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    schedule.current_lr = schedule.get_lr();
    result.adamw.alpha = result.sgd.alpha = schedule.current_lr;
    result.sgd.wd = result.adamw.wd = schedule.lr->wd;
    return result;
}

// Tensors whose names contain these substrings use MUL_MAT_ID (sparse MoE expert dispatch)
// which has no backward implementation — exclude them from LoRA targets unconditionally.
static const std::vector<std::string> EXCLUDED_SUBSTRINGS = {
    "_exps",      // MoE expert weight stacks (ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_up_exps)
};

static bool tensor_is_excluded(const char * name) {
    const std::string n(name);
    for (const auto & ex : EXCLUDED_SUBSTRINGS) {
        if (n.find(ex) != std::string::npos) return true;
    }
    return false;
}

// Extract the transformer block index from a tensor name of the form "blk.NN.<rest>".
// Returns -1 if the name does not follow this pattern.
static int tensor_layer_index(const char * name) {
    // All per-layer tensors in llama.cpp GGUF are named "blk.<N>.<suffix>"
    const char * p = strstr(name, "blk.");
    if (!p) return -1;
    p += 4; // skip "blk."
    char * end = nullptr;
    long idx = strtol(p, &end, 10);
    if (end == p || (*end != '.' && *end != '\0')) return -1;
    return (int) idx;
}

static bool tensor_matches_targets(const char * name, const std::vector<std::string> & targets,
                                   int freeze_layers = 0) {
    if (tensor_is_excluded(name)) return false;
    if (freeze_layers > 0) {
        const int layer = tensor_layer_index(name);
        if (layer >= 0 && layer < freeze_layers) return false;
    }
    for (const auto & t : targets) {
        if (std::string(name).find(t) != std::string::npos) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// JSONL dataset loading
// ---------------------------------------------------------------------------

struct training_sample {
    std::vector<llama_token> tokens;
    std::vector<bool>        is_label;
    std::vector<float>       critical_weights;

    float reward = 1.0f;

    // Original JSONL conversation identity.
    // All expanded assistant targets from the same JSONL row
    // MUST have the same conversation_id.
    int64_t conversation_id = -1;

    // 0, 1, 2, ... among assistant messages in this conversation.
    int32_t target_assistant_ordinal = -1;
};

struct tokenization_request {
    std::string prompt;
    std::string response;

    // Final-answer text only.
    // Critical-SFT spans continue to align against this string,
    // not against reasoning + final.
    std::string annotated_response;

    std::vector<critical_span> critical_spans;

    float reward = 1.0f;

    int64_t conversation_id = -1;
    int32_t target_assistant_ordinal = -1;

    bool   has_reasoning = false;
    size_t reasoning_bytes = 0;
};

static bool critical_mode_uses_spans(const std::string & mode) {
    return mode == "spans" || mode == "hybrid";
}

static bool response_token_ranges(
        const llama_vocab                   * vocab,
        const std::vector<llama_token>      & tokens,
        const std::string                   & formatted_response,
        const std::string                   & annotated_response,
        std::vector<critical_token_range>   & ranges,
        std::string                         & error) {
    std::vector<critical_token_range> decoded_ranges;
    decoded_ranges.reserve(tokens.size());

    std::string decoded;
    for (llama_token token : tokens) {
        const size_t start = decoded.size();
        decoded += common_token_to_piece(vocab, token, true);
        decoded_ranges.push_back({ start, decoded.size() });
    }

    const std::string detokenized = common_detokenize(vocab, tokens, true);
    if (decoded != detokenized) {
        decoded.clear();
        decoded_ranges.clear();
        std::vector<llama_token> prefix;
        prefix.reserve(tokens.size());
        for (llama_token token : tokens) {
            const size_t start = decoded.size();
            prefix.push_back(token);
            const std::string current = common_detokenize(vocab, prefix, true);
            if (current.rfind(decoded, 0) != 0) {
                error = "token pieces cannot reconstruct response byte ranges";
                return false;
            }
            decoded = current;
            decoded_ranges.push_back({ start, decoded.size() });
        }
    }

    const size_t decoded_offset = decoded.rfind(annotated_response);
    if (formatted_response.rfind(annotated_response) == std::string::npos || decoded_offset == std::string::npos) {
        error = "annotated response text was not preserved by formatting/tokenization";
        return false;
    }

    const size_t annotated_end = decoded_offset + annotated_response.size();
    ranges.resize(decoded_ranges.size());
    for (size_t i = 0; i < decoded_ranges.size(); ++i) {
        const size_t start = std::max(decoded_ranges[i].start, decoded_offset);
        const size_t end = std::min(decoded_ranges[i].end, annotated_end);
        if (start < end) {
            ranges[i] = { start - decoded_offset, end - decoded_offset };
        } else {
            ranges[i] = { 0, 0 };
        }
    }
    return true;
}



static std::string apply_qlora_chat_template(
        common_chat_templates              * tmpls,
        const std::vector<common_chat_msg> & messages,
        bool                                  add_generation_prompt,
        bool                                  enable_thinking,
        int                                   preserve_thinking) {

    common_chat_templates_inputs inputs;

    inputs.messages = messages;
    inputs.add_generation_prompt = add_generation_prompt;
    // A recorded assistant tool call is part of the supervised completion.
    // Do not put ordinary SFT conversations into a template's tool mode:
    // some templates use a different assistant channel when tool_choice is
    // auto.  Enable it only for conversations that actually contain calls.
    inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
    for (const common_chat_msg & message : messages) {
        if (!message.tool_calls.empty()) {
            inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
            break;
        }
    }

    inputs.reasoning_format =
        COMMON_REASONING_FORMAT_DEEPSEEK;

    inputs.enable_thinking =
        enable_thinking;

    if (preserve_thinking >= 0) {
        inputs.chat_template_kwargs["preserve_reasoning"] =
            preserve_thinking ? "true" : "false";
    }

    return common_chat_templates_apply(
        tmpls,
        inputs).prompt;
}

static void replace_chat_message_content_with_marker(
        common_chat_msg & message,
        const std::string & marker) {

    if (!message.content_parts.empty()) {
        message.content.clear();
        message.content_parts.clear();

        common_chat_msg_content_part part;
        part.type = "text";
        part.text = marker;

        message.content_parts.push_back(
            std::move(part));
    } else {
        message.content = marker;
        message.content_parts.clear();
    }
}

static size_t chat_message_size_estimate(const common_chat_msg & message) {
    size_t size = message.role.size() + message.content.size() + message.reasoning_content.size() + 64;
    for (const common_chat_msg_content_part & part : message.content_parts) {
        size += part.text.size() + 32;
    }
    for (const common_chat_tool_call & call : message.tool_calls) {
        size += call.id.size() + call.name.size() + call.arguments.size() + 64;
    }
    return size;
}

static size_t supervised_context_begin(
        const std::vector<common_chat_msg> & conversation,
        size_t                                target_index,
        size_t                                max_context_tokens) {
    if (max_context_tokens == 0) {
        return 0;
    }

    // Bound both bytes and message count. Some templates scan preceding messages
    // inside their message loop, so hundreds of tiny turns otherwise become
    // quadratic even though only a short suffix can survive token truncation.
    const size_t max_context_bytes = std::max<size_t>(16 * 1024, max_context_tokens * 32);
    const size_t max_context_messages = std::clamp<size_t>(max_context_tokens / 64, 4, 16);
    size_t begin = target_index;
    size_t context_bytes = 0;
    size_t context_messages = 0;
    while (begin > 0 && context_bytes < max_context_bytes && context_messages < max_context_messages) {
        --begin;
        context_bytes += chat_message_size_estimate(conversation[begin]);
        ++context_messages;
    }

    while (begin > 0 && conversation[begin].role != "user" && conversation[begin].role != "system") {
        --begin;
    }
    return begin;
}

static bool format_supervised_chat_turn(
        common_chat_templates              * tmpls,
        const std::vector<common_chat_msg> & conversation,
        size_t                                target_index,
        int                                   preserve_thinking,
        size_t                                max_context_tokens,
        tokenization_request                & result,
        std::string                         & error,
        size_t                              * messages_skipped = nullptr) {

    if (target_index >= conversation.size()) {
        error = "target assistant index is out of range";
        return false;
    }

    if (conversation[target_index].role != "assistant") {
        error = "target message is not assistant";
        return false;
    }

    // ---------------------------------------------------------
    // Build context PREFIX only.
    //
    // Everything after the current assistant target is excluded.
    // ---------------------------------------------------------

    const size_t context_begin = supervised_context_begin(conversation, target_index, max_context_tokens);
    if (messages_skipped) {
        *messages_skipped = context_begin;
    }

    std::vector<common_chat_msg> prompt_messages(
        conversation.begin() + context_begin,
        conversation.begin() + target_index);

    const common_chat_msg & target =
        conversation[target_index];

    result.annotated_response =
        target.render_content();

    result.has_reasoning =
        !target.reasoning_content.empty();

    result.reasoning_bytes =
        target.reasoning_content.size();

    // ---------------------------------------------------------
    // No-template fallback
    // ---------------------------------------------------------

    /*if (!tmpls) {
        result.prompt.clear();

        for (const common_chat_msg & message :
                prompt_messages) {

            result.prompt +=
                apply_chatml_message(message);
        }

        // Generation starts at assistant role.
        result.prompt +=
            "<|im_start|>assistant\n";

        result.response =
            apply_chatml_payload(target);

        result.response +=
            "<|im_end|>\n";

        return true;
    }*/

    // ---------------------------------------------------------
    // Real model template
    // ---------------------------------------------------------

    const bool enable_thinking =
        result.has_reasoning;

    // ============================================================
    // 1. Render prompt exclusively through chat_template.jinja.
    // ============================================================

    const bool has_prompt_messages = !prompt_messages.empty();
    if (has_prompt_messages) {
        result.prompt =
            apply_qlora_chat_template(
                tmpls,
                prompt_messages,
                /*add_generation_prompt=*/true,
                enable_thinking,
                preserve_thinking);
    }

    // ============================================================
    // 2. Render the real completed target.
    // ============================================================

    std::vector<common_chat_msg> full_messages =
        prompt_messages;

    full_messages.push_back(target);

    const std::string full_text =
        apply_qlora_chat_template(
            tmpls,
            full_messages,
            /*add_generation_prompt=*/false,
            enable_thinking,
            preserve_thinking);

    // ============================================================
    // 3. Easy case: generation prompt is an exact prefix.
    // ============================================================

    if (has_prompt_messages && full_text.rfind(result.prompt, 0) == 0) {
        result.response =
            full_text.substr(
                result.prompt.size());

        return true;
    }

    // ============================================================
    // 4. Non-prefix templates:
    //
    // Derive the target boundary entirely through Jinja.
    // Never invent ChatML / channel / think tokens ourselves.
    // ============================================================

    std::string reasoning_marker =
        "__AIKAR_REASONING_BOUNDARY_7C912E__";

    std::string final_marker =
        "__AIKAR_FINAL_BOUNDARY_B45D31__";

    // Avoid pathological collision with user-provided data.
    while (
        target.reasoning_content.find(
            reasoning_marker) != std::string::npos ||
        target.render_content().find(
            reasoning_marker) != std::string::npos) {

        reasoning_marker += "_";
    }

    while (
        target.reasoning_content.find(
            final_marker) != std::string::npos ||
        target.render_content().find(
            final_marker) != std::string::npos) {

        final_marker += "_";
    }

    common_chat_msg marked_target =
        target;

    // Current target reasoning, if present.
    if (result.has_reasoning) {
        marked_target.reasoning_content =
            reasoning_marker;
    }

    // Current target final content.
    replace_chat_message_content_with_marker(
        marked_target,
        final_marker);

    std::vector<common_chat_msg> marked_messages =
        prompt_messages;

    marked_messages.push_back(
        std::move(marked_target));

    const std::string marked_text =
        apply_qlora_chat_template(
            tmpls,
            marked_messages,
            /*add_generation_prompt=*/false,
            enable_thinking,
            preserve_thinking);

    // ============================================================
    // 5. Locate semantic slots in what Jinja actually emitted.
    // ============================================================

    const size_t reasoning_marker_pos =
        result.has_reasoning
            ? marked_text.find(reasoning_marker)
            : std::string::npos;

    const size_t final_marker_pos =
        marked_text.find(final_marker);

    // A normal assistant target with final content should cause the
    // final marker to appear.
    //
    // Exception: templates may represent special assistant turns such
    // as pure tool-call messages without normal textual final content.
    if (!result.annotated_response.empty() &&
        final_marker_pos == std::string::npos) {

        error =
            "chat_template.jinja did not render "
            "the assistant content slot";

        return false;
    }

    if (result.has_reasoning &&
        reasoning_marker_pos == std::string::npos) {
        // The template may intentionally omit reasoning for this message type.
        // Keep the final answer trainable instead of dropping the whole target.
        result.has_reasoning = false;
        result.reasoning_bytes = 0;
    }

    // ============================================================
    // 6. Determine start of supervised target.
    //
    // Reasoning comes first when present; otherwise final content.
    // ============================================================

    size_t target_marker_pos =
        std::string::npos;

    if (result.has_reasoning) {
        target_marker_pos =
            reasoning_marker_pos;
    } else {
        target_marker_pos =
            final_marker_pos;
    }

    if (target_marker_pos == std::string::npos) {
        error =
            "could not locate assistant target boundary "
            "in chat-template rendering";

        return false;
    }

    const std::string target_prefix =
        marked_text.substr(
            0,
            target_marker_pos);

    // The actual completed Jinja rendering should share everything
    // before the semantic target slot.
    if (full_text.rfind(target_prefix, 0) != 0) {
        error =
            "full chat-template rendering does not match "
            "marker-derived assistant boundary";

        return false;
    }

    result.response =
        full_text.substr(
            target_prefix.size());

    return true;
}

struct jsonl_render_task {
    size_t line_index;
    size_t target_index;
    size_t target_ordinal;
    bool final_target;
};

struct jsonl_chat_line {
    std::vector<common_chat_msg> messages;
    std::vector<size_t> assistant_indices;
    nlohmann::json critical_record;
    float reward = 1.0f;
    int lineno = 0;
};

static std::vector<training_sample> load_jsonl(
        const std::string       & path,
        const llama_vocab       * vocab,
        common_chat_templates   * tmpls,
        int32_t                   n_threads,
        const std::string       & critical_mode,
        float                     critical_default_weight,
        int                       preserve_thinking,
        size_t                    max_sample_tokens = 0) {

    struct pending_sample {
        tokenization_request request;
        int                  lineno = 0;
        bool                 valid = false;
    };

    struct tokenized_sample {
        std::vector<llama_token> prompt_tokens;
        std::vector<llama_token> response_tokens;
        bool                     valid = false;
    };

    std::vector<training_sample> samples;

    std::vector<common_jsonl_line> lines;

    try {
        lines = common_jsonl_read_lines(
            path,
            COMMON_JSONL_EMPTY_LINE_SKIP);
    } catch (const std::exception & e) {
        LOG_ERR(
            "%s: %s\n",
            __func__,
            e.what());

        return {};
    }
    const size_t n_input_lines = lines.size();

    std::vector<std::vector<pending_sample>> pending_by_line(lines.size());
    std::vector<std::unique_ptr<jsonl_chat_line>> chat_lines(lines.size());
    std::atomic<size_t> n_chat_conversations { 0 };
    std::atomic<size_t> n_nonchat_samples { 0 };
    std::atomic<size_t> n_expanded_assistant_targets { 0 };
    const size_t n_parse_workers = common_jsonl_worker_count(n_threads, lines.size());
    std::vector<size_t> line_order(lines.size());
    std::iota(line_order.begin(), line_order.end(), 0);
    std::stable_sort(line_order.begin(), line_order.end(), [&](size_t a, size_t b) {
        return lines[a].text.size() > lines[b].text.size();
    });

    {
        common_jsonl_worker_pool pool(n_parse_workers);
        parallel_for_with_eta(pool, lines.size(), n_parse_workers, "parse", [&](size_t task_index, size_t) {
            const size_t line_index = line_order[task_index];
            const common_jsonl_line & input = lines[line_index];
            nlohmann::json data;
            try {
                data = nlohmann::json::parse(input.text);
            } catch (const std::exception & e) {
                LOG_WRN("%s: skipping invalid JSON on line %ld: %s\n", __func__, (long) input.number, e.what());
                return;
            }

            float reward = 1.0f;
            try {
                if (data.contains("reward")) reward = data.at("reward").get<float>();
                else if (data.contains("score")) reward = data.at("score").get<float>();
            } catch (const std::exception & e) {
                LOG_WRN("%s: invalid reward on line %ld: %s\n", __func__, (long) input.number, e.what());
                return;
            }

            if (data.contains("messages")) {
                ++n_chat_conversations;
                auto parsed = std::make_unique<jsonl_chat_line>();
                parsed->reward = reward;
                parsed->lineno = (int) input.number;
                try {
                    const nlohmann::json & raw_messages = data.at("messages");
                    if (!raw_messages.is_array()) {
                        LOG_WRN("%s: messages is not an array on line %ld\n", __func__, (long) input.number);
                        return;
                    }
                    parsed->messages.reserve(raw_messages.size());
                    for (const nlohmann::json & raw : raw_messages) {
                        parsed->messages.push_back(common_jsonl_parse_chat_message(raw, COMMON_JSONL_CHAT_PARSE_STRICT_REASONING));
                    }
                } catch (const std::exception & e) {
                    LOG_WRN("%s: invalid chat messages on line %ld: %s\n", __func__, (long) input.number, e.what());
                    return;
                }
                for (size_t i = 0; i < parsed->messages.size(); ++i) {
                    if (parsed->messages[i].role == "assistant") parsed->assistant_indices.push_back(i);
                }
                if (parsed->assistant_indices.empty()) return;
                if (critical_mode_uses_spans(critical_mode) && data.contains("critical_spans")) {
                    parsed->critical_record["critical_spans"] = data["critical_spans"];
                }
                pending_by_line[line_index].resize(parsed->assistant_indices.size());
                chat_lines[line_index] = std::move(parsed);
                return;
            }

            tokenization_request request;
            try {
                if (data.contains("prompt") && data.contains("response")) {
                    request.prompt = data.at("prompt").get<std::string>();
                    request.response = data.at("response").get<std::string>();
                } else if (data.contains("text")) {
                    request.response = data.at("text").get<std::string>();
                } else {
                    LOG_WRN("%s: unknown format on line %ld, skipping\n", __func__, (long) input.number);
                    return;
                }
                request.annotated_response = request.response;
                request.reward = reward;
                request.conversation_id = (int64_t) input.number;
                request.target_assistant_ordinal = 0;
                if (critical_mode_uses_spans(critical_mode)) {
                    std::string span_error;
                    if (!critical_token_parse_spans(data, request.annotated_response, critical_default_weight, request.critical_spans, span_error)) {
                        LOG_WRN("%s: invalid critical_spans on line %ld: %s\n", __func__, (long) input.number, span_error.c_str());
                        return;
                    }
                }
            } catch (const std::exception & e) {
                LOG_WRN("%s: invalid non-chat sample on line %ld: %s\n", __func__, (long) input.number, e.what());
                return;
            }
            pending_by_line[line_index].push_back({ std::move(request), (int) input.number, true });
            ++n_nonchat_samples;
        });
    }

    std::vector<jsonl_render_task> render_tasks;
    for (size_t line_index = 0; line_index < chat_lines.size(); ++line_index) {
        if (!chat_lines[line_index]) continue;
        const jsonl_chat_line & parsed = *chat_lines[line_index];
        for (size_t ordinal = 0; ordinal < parsed.assistant_indices.size(); ++ordinal) {
            render_tasks.push_back({ line_index, parsed.assistant_indices[ordinal], ordinal,
                ordinal + 1 == parsed.assistant_indices.size() });
        }
    }
    std::stable_sort(render_tasks.begin(), render_tasks.end(), [](const jsonl_render_task & a, const jsonl_render_task & b) {
        return a.target_index > b.target_index;
    });

    lines.clear();
    lines.shrink_to_fit();

    std::atomic<size_t> n_cropped_targets { 0 };
    std::atomic<size_t> n_skipped_messages { 0 };
    const size_t n_render_workers = common_jsonl_worker_count(n_threads, render_tasks.size());
    {
        common_jsonl_worker_pool pool(n_render_workers);
        parallel_for_with_eta(pool, render_tasks.size(), n_render_workers, "render-targets", [&](size_t task_index, size_t) {
            const jsonl_render_task & task = render_tasks[task_index];
            const jsonl_chat_line & parsed = *chat_lines[task.line_index];
            tokenization_request request;
            request.reward = parsed.reward;
            request.conversation_id = parsed.lineno;
            request.target_assistant_ordinal = (int32_t) task.target_ordinal;
            std::string format_error;
            size_t messages_skipped = 0;
            try {
                if (!format_supervised_chat_turn(tmpls, parsed.messages, task.target_index, preserve_thinking,
                        max_sample_tokens, request, format_error, &messages_skipped)) {
                    LOG_WRN("%s: formatting failed on line %d assistant=%zu: %s\n",
                            __func__, parsed.lineno, task.target_ordinal, format_error.c_str());
                    return;
                }
            } catch (const std::exception & e) {
                LOG_WRN("%s: template exception on line %d assistant=%zu: %s\n",
                        __func__, parsed.lineno, task.target_ordinal, e.what());
                return;
            }
            if (task.final_target && critical_mode_uses_spans(critical_mode)) {
                std::string span_error;
                if (!critical_token_parse_spans(parsed.critical_record, request.annotated_response,
                        critical_default_weight, request.critical_spans, span_error)) {
                    LOG_WRN("%s: invalid critical_spans on line %d: %s\n", __func__, parsed.lineno, span_error.c_str());
                    return;
                }
            }
            if (messages_skipped > 0) {
                ++n_cropped_targets;
                n_skipped_messages.fetch_add(messages_skipped);
            }
            pending_by_line[task.line_index][task.target_ordinal] = { std::move(request), parsed.lineno, true };
            ++n_expanded_assistant_targets;
        });
    }

    std::vector<pending_sample> pending;
    pending.reserve(n_expanded_assistant_targets.load() + n_nonchat_samples.load());
    for (std::vector<pending_sample> & line_pending : pending_by_line) {
        for (pending_sample & sample : line_pending) {
            if (sample.valid) pending.push_back(std::move(sample));
        }
    }
    LOG_INF("%s: context crop: targets=%zu skipped_messages=%zu max_sample_tokens=%zu\n",
            __func__, n_cropped_targets.load(), n_skipped_messages.load(), max_sample_tokens);

    if (pending.empty()) {
        LOG_ERR(
            "%s: no usable training targets in %s\n",
            __func__,
            path.c_str());

        return {};
    }

    // =========================================================
    // Phase 2:
    // Parallel tokenization.
    // =========================================================

    std::vector<tokenized_sample>
        tokenized(pending.size());

    const size_t n_workers =
        common_jsonl_worker_count(
            n_threads,
            pending.size());

    {
        common_jsonl_worker_pool pool(
            n_workers);

        parallel_for_with_eta(
            pool,
            pending.size(),
            n_workers,
            "tokenize",
            [&](size_t i, size_t) {

                const tokenization_request & request =
                    pending[i].request;

                tokenized_sample & output =
                    tokenized[i];

                try {
                    output.prompt_tokens =
                        common_tokenize(
                            vocab,
                            request.prompt,
                            /*add_special=*/true,
                            /*parse_special=*/true);

                    output.response_tokens =
                        common_tokenize(
                            vocab,
                            request.response,
                            /*add_special=*/false,
                            /*parse_special=*/true);

                    output.valid = true;

                } catch (const std::exception & e) {
                    LOG_WRN(
                        "%s: tokenization failed "
                        "on line %d assistant=%d: %s\n",
                        __func__,
                        pending[i].lineno,
                        request.target_assistant_ordinal,
                        e.what());
                }
            });
    }

    // =========================================================
    // Phase 3:
    // Construct training_sample objects and response-only labels.
    // =========================================================

    samples.reserve(pending.size());

    bool logged_preview = false;

    size_t n_valid_reasoning_targets = 0;

    loop_eta_progress construct_progress("construct-samples", pending.size());

    for (size_t i = 0;
         i < pending.size();
         ++i) {

        construct_progress.update(i);

        if (!tokenized[i].valid) {
            continue;
        }

        const tokenization_request & request =
            pending[i].request;

        const std::vector<llama_token> & prompt_tokens =
            tokenized[i].prompt_tokens;

        const std::vector<llama_token> & response_tokens =
            tokenized[i].response_tokens;

        if (prompt_tokens.empty() &&
            response_tokens.empty()) {

            continue;
        }

        training_sample sample;

        sample.reward =
            request.reward;

        sample.conversation_id =
            request.conversation_id;

        sample.target_assistant_ordinal =
            request.target_assistant_ordinal;

        sample.tokens.reserve(
            prompt_tokens.size() +
            response_tokens.size());

        sample.tokens.insert(
            sample.tokens.end(),
            prompt_tokens.begin(),
            prompt_tokens.end());

        sample.tokens.insert(
            sample.tokens.end(),
            response_tokens.begin(),
            response_tokens.end());

        // Prompt = masked.
        // Current assistant response = supervised.
        sample.is_label.resize(
            sample.tokens.size(),
            false);

        for (size_t j = prompt_tokens.size();
             j < sample.tokens.size();
             ++j) {

            sample.is_label[j] = true;
        }

        // -----------------------------------------------------
        // Critical-SFT
        // -----------------------------------------------------

        if (critical_mode != "none") {
            sample.critical_weights.resize(
                sample.tokens.size(),
                1.0f);
        }

        if (critical_mode_uses_spans(
                critical_mode) &&
            !request.critical_spans.empty()) {

            std::vector<critical_token_range> ranges;
            std::vector<float> response_weights;

            std::string alignment_error;

            if (!response_token_ranges(
                    vocab,
                    response_tokens,
                    request.response,
                    request.annotated_response,
                    ranges,
                    alignment_error) ||

                !critical_token_apply_spans(
                    request.annotated_response.size(),
                    request.critical_spans,
                    ranges,
                    response_weights,
                    alignment_error)) {

                LOG_WRN(
                    "%s: invalid critical_spans "
                    "alignment on line %d assistant=%d: %s\n",
                    __func__,
                    pending[i].lineno,
                    request.target_assistant_ordinal,
                    alignment_error.c_str());

                continue;
            }

            std::copy(
                response_weights.begin(),
                response_weights.end(),
                sample.critical_weights.begin() +
                    prompt_tokens.size());
        }

        if (request.has_reasoning) {
            ++n_valid_reasoning_targets;
        }

        if (!logged_preview) {
            logged_preview = true;

            LOG_INF(
                "%s: first target: "
                "conversation=%ld assistant=%d "
                "reasoning=%s reasoning_bytes=%zu\n",
                __func__,
                (long) request.conversation_id,
                request.target_assistant_ordinal,
                request.has_reasoning ? "yes" : "no",
                request.reasoning_bytes);

            LOG_INF(
                "%s: first prompt preview: \"%s\"\n",
                __func__,
                preview_text(
                    request.prompt).c_str());

            LOG_INF(
                "%s: first response preview: \"%s\"\n",
                __func__,
                preview_text(
                    request.response).c_str());

            LOG_INF(
                "%s: first sample tokens: "
                "prompt=%zu response=%zu\n",
                __func__,
                prompt_tokens.size(),
                response_tokens.size());
        }

        samples.push_back(
            std::move(sample));
    }

    construct_progress.finish();

    LOG_INF(
        "%s: loaded %zu training samples from %zu JSONL lines "
        "(chat_conversations=%zu "
        "expanded_assistant_targets=%zu "
        "nonchat=%zu "
        "reasoning_targets=%zu "
        "preserve_thinking=%s "
        "tokenizer_workers=%zu)\n",
        __func__,
        samples.size(),
        n_input_lines,
        n_chat_conversations.load(),
        n_expanded_assistant_targets.load(),
        n_nonchat_samples.load(),
        n_valid_reasoning_targets,
        preserve_thinking < 0 ? "template-default" : preserve_thinking ? "yes" : "no",
        n_workers);

    return samples;
}

struct packed_dataset_partition {
    std::vector<llama_token> tokens;
    std::vector<int32_t>     labels;
    std::vector<float>       rewards;
    std::vector<float>       span_weights;
    std::vector<int64_t>     window_offsets;

    int64_t ndata_candidate = 0;
};

static packed_dataset_partition pack_dataset_partition(
        const std::vector<training_sample> & samples,
        size_t                                sample_begin,
        size_t                                sample_end,
        int32_t                               n_ctx,
        bool                                  train_on_prompt,
        llama_token                           bos_token,
        bool                                  critical_enabled) {

    packed_dataset_partition out;

    const char * progress_phase = sample_begin == 0 ? "pack-train" : "pack-validation";
    loop_eta_progress pack_progress(progress_phase, sample_end - sample_begin);

    for (size_t si = sample_begin;
         si < sample_end;
         ++si) {

        pack_progress.update(si - sample_begin);

        const training_sample & s = samples[si];

        // Separator ONLY between samples in this partition.
        // Do not inject a separator before the first validation sample.
        if (si > sample_begin &&
            bos_token >= 0 &&
            !s.tokens.empty()) {

            out.tokens.push_back(bos_token);
            out.labels.push_back(-1);
            out.rewards.push_back(s.reward);

            if (critical_enabled) {
                out.span_weights.push_back(0.0f);
            }
        }

        for (size_t i = 0;
             i + 1 < s.tokens.size();
             ++i) {

            out.tokens.push_back(s.tokens[i]);

            bool active = false;

            if (train_on_prompt) {
                out.labels.push_back(
                    (int32_t) s.tokens[i + 1]);
                active = true;
            } else {
                out.labels.push_back(
                    s.is_label[i + 1]
                        ? (int32_t) s.tokens[i + 1]
                        : -1);

                active = s.is_label[i + 1];
            }

            out.rewards.push_back(s.reward);

            if (critical_enabled) {
                GGML_ASSERT(
                    s.critical_weights.size()
                    == s.tokens.size());

                const float span_weight =
                    s.is_label[i + 1]
                        ? s.critical_weights[i + 1]
                        : 1.0f;

                out.span_weights.push_back(
                    active ? span_weight : 0.0f);
            }
        }
    }

    pack_progress.finish();

    if (out.tokens.empty()) {
        return out;
    }

    const int64_t stride =
        std::max<int64_t>(1, n_ctx / 2);

    const int64_t n_tokens =
        (int64_t) out.tokens.size();

    out.ndata_candidate = 1;

    if (n_tokens > n_ctx) {
        out.ndata_candidate +=
            (n_tokens - n_ctx + stride - 1)
            / stride;
    }

    out.window_offsets.reserve(
        out.ndata_candidate);

    for (int64_t i = 0;
         i < out.ndata_candidate;
         ++i) {

        const int64_t off = i * stride;

        bool has_label = false;

        for (int32_t j = 0;
             j < n_ctx && off + j < n_tokens;
             ++j) {

            if (out.labels[off + j] >= 0) {
                has_label = true;
                break;
            }
        }

        if (has_label) {
            out.window_offsets.push_back(off);
        }
    }

    return out;
}


// Pack variable-length samples into fixed-context-length windows and create
// an ggml_opt_dataset. Labels for prompt tokens are set to -1 (ignored by
// the loss in the epoch loop).
// window_rewards is filled with one reward weight per window (averaged over
// the sample tokens that fall in that window). If all samples have reward=1.0
// the vector is all-ones and has no effect.
static ggml_opt_dataset_t build_dataset(
        const std::vector<training_sample> & samples,
        int32_t                              n_ctx,
        std::vector<float>                 & window_rewards,
        int64_t                              sample_split,
        int64_t                            & idata_split_out,
        bool                                 train_on_prompt = false,
        llama_token                          bos_token = -1,
        bool                                 critical_enabled = false,
        std::vector<llama_opt_critical_token_metadata> * critical_metadata_out = nullptr) {

    idata_split_out = 0;
    window_rewards.clear();

    if (critical_metadata_out) {
        critical_metadata_out->clear();
    }

    // ------------------------------------------------------------
    // 1. Basic validation
    // ------------------------------------------------------------

    if (samples.empty()) {
        LOG_ERR("%s: dataset has no samples\n", __func__);
        return nullptr;
    }

    if (n_ctx <= 0) {
        LOG_ERR(
            "%s: invalid context length %d\n",
            __func__,
            n_ctx);
        return nullptr;
    }

    if (sample_split <= 0 ||
        sample_split > (int64_t) samples.size()) {

        LOG_ERR(
            "%s: invalid sample split %ld for %zu samples\n",
            __func__,
            (long) sample_split,
            samples.size());

        return nullptr;
    }

    // ------------------------------------------------------------
    // 2. Pack TRAIN and VAL separately
    //
    // IMPORTANT:
    // We split samples BEFORE sliding-window packing.
    // Therefore train and validation windows can never overlap.
    // ------------------------------------------------------------

    packed_dataset_partition train =
        pack_dataset_partition(
            samples,
            0,
            (size_t) sample_split,
            n_ctx,
            train_on_prompt,
            bos_token,
            critical_enabled);

    packed_dataset_partition val =
        pack_dataset_partition(
            samples,
            (size_t) sample_split,
            samples.size(),
            n_ctx,
            train_on_prompt,
            bos_token,
            critical_enabled);

    if (train.window_offsets.empty()) {
        LOG_ERR(
            "%s: training partition has no supervised labels\n",
            __func__);
        return nullptr;
    }

    // Number of packed TRAIN windows.
    // This is what the training loop wants as idata_split.
    idata_split_out =
        (int64_t) train.window_offsets.size();

    const int64_t n_val_windows =
        (int64_t) val.window_offsets.size();

    const int64_t ndata =
        idata_split_out + n_val_windows;

    // One reward per packed window.
    window_rewards.assign(
        (size_t) ndata,
        1.0f);

    // ------------------------------------------------------------
    // 3. Allocate ggml dataset
    // ------------------------------------------------------------

    ggml_opt_dataset_t dataset =
        ggml_opt_dataset_init(
            GGML_TYPE_I32,
            GGML_TYPE_I32,
            n_ctx,
            n_ctx,
            ndata,
            1);

    if (!dataset) {
        LOG_ERR(
            "%s: failed to allocate dataset\n",
            __func__);
        return nullptr;
    }

    int32_t * data =
        (int32_t *)
        ggml_opt_dataset_data(dataset)->data;

    int32_t * labels =
        (int32_t *)
        ggml_opt_dataset_labels(dataset)->data;

    int64_t n_labels = 0;
    int64_t n_padded = 0;

    // ------------------------------------------------------------
    // 4. Allocate Critical-SFT metadata
    // ------------------------------------------------------------

    std::vector<llama_opt_critical_token_metadata>
        critical_metadata;

    if (critical_enabled) {
        critical_metadata.resize(
            (size_t) ndata *
            (size_t) n_ctx);
    }

    // ------------------------------------------------------------
    // 5. Helper:
    //    copy one packed partition into ggml dataset
    //
    // TRAIN is written starting at window 0.
    // VAL is written starting at idata_split_out.
    // ------------------------------------------------------------

    auto write_partition =
        [&](const packed_dataset_partition & part,
            int64_t                         window_base,
            const char                    * phase) {

        loop_eta_progress write_progress(phase, part.window_offsets.size());

        for (size_t wi = 0;
             wi < part.window_offsets.size();
             ++wi) {

            write_progress.update(wi);

            const int64_t dst_window =
                window_base + (int64_t) wi;

            const int64_t off =
                part.window_offsets[wi];

            float reward_sum = 0.0f;
            int64_t reward_count = 0;

            for (int32_t j = 0;
                 j < n_ctx;
                 ++j) {

                const int64_t idx =
                    off + j;

                const int64_t dst =
                    dst_window *
                    (int64_t) n_ctx +
                    j;

                // ------------------------------------------------
                // Padding
                // ------------------------------------------------

                if (idx >=
                    (int64_t) part.tokens.size()) {

                    data[dst] =
                        bos_token >= 0
                            ? bos_token
                            : part.tokens.back();

                    labels[dst] = -1;

                    if (critical_enabled) {
                        critical_metadata[
                            (size_t) dst] =
                            { 0.0f, 0.0f };
                    }

                    ++n_padded;
                    continue;
                }

                // ------------------------------------------------
                // Real token
                // ------------------------------------------------

                data[dst] =
                    part.tokens[
                        (size_t) idx];

                labels[dst] =
                    part.labels[
                        (size_t) idx];

                // Supervised token
                if (part.labels[
                        (size_t) idx] >= 0) {

                    ++n_labels;

                    reward_sum +=
                        part.rewards[
                            (size_t) idx];

                    ++reward_count;

                    if (critical_enabled) {
                        critical_metadata[
                            (size_t) dst] = {

                            part.span_weights[
                                (size_t) idx],

                            part.rewards[
                                (size_t) idx]
                        };
                    }
                } else if (critical_enabled) {

                    // Prompt / masked token
                    critical_metadata[
                        (size_t) dst] =
                        { 0.0f, 0.0f };
                }
            }

            // pack_dataset_partition() already removes
            // windows without labels, so this must be > 0.
            GGML_ASSERT(reward_count > 0);

            window_rewards[
                (size_t) dst_window] =
                reward_sum /
                (float) reward_count;
        }

        write_progress.finish();
    };

    // ------------------------------------------------------------
    // 6. Write windows in this EXACT order:
    //
    // [ TRAIN ][ VALIDATION ]
    //
    // This ordering is important because the QAT training loop uses
    // idata_split as the boundary.
    // ------------------------------------------------------------

    write_partition(
        train,
        0,
        "write-train-windows");

    write_partition(
        val,
        idata_split_out,
        "write-validation-windows");

    // ------------------------------------------------------------
    // 7. Diagnostics
    // ------------------------------------------------------------

    LOG_INF(
        "%s: sample split: "
        "train=%ld val=%ld total=%zu\n",
        __func__,
        (long) sample_split,
        (long) (
            (int64_t) samples.size()
            - sample_split),
        samples.size());

    LOG_INF(
        "%s: packed windows: "
        "train=%ld val=%ld total=%ld "
        "(ctx=%d stride=%d supervised=%ld padded=%ld, "
        "dropped_train=%ld dropped_val=%ld)\n",
        __func__,
        (long) idata_split_out,
        (long) n_val_windows,
        (long) ndata,
        n_ctx,
        std::max<int32_t>(
            1,
            n_ctx / 2),
        (long) n_labels,
        (long) n_padded,
        (long) (
            train.ndata_candidate -
            (int64_t)
            train.window_offsets.size()),
        (long) (
            val.ndata_candidate -
            (int64_t)
            val.window_offsets.size()));

    // ------------------------------------------------------------
    // 8. Window reward normalization
    //
    // IMPORTANT:
    // Fit normalization using TRAIN ONLY.
    //
    // Do NOT independently normalize validation.
    // Do NOT let validation statistics influence training.
    // ------------------------------------------------------------

    for (float & r : window_rewards) {
        r = std::max(
            -1.0f,
            std::min(
                1.0f,
                r));
    }

    auto train_reward_end =
        window_rewards.begin()
        + idata_split_out;

    const float rmin =
        *std::min_element(
            window_rewards.begin(),
            train_reward_end);

    const float rmax =
        *std::max_element(
            window_rewards.begin(),
            train_reward_end);

    const float rrange =
        rmax - rmin;

    if (rrange > 1e-6f) {

        // Apply TRAIN-fitted transform to both train and val.
        for (float & r : window_rewards) {

            r =
                (r - rmin) /
                rrange;

            // A validation reward may lie outside the
            // training reward range.
            r = std::max(
                0.0f,
                std::min(
                    1.0f,
                    r));
        }

        LOG_INF(
            "%s: reward normalization fitted "
            "on training windows: "
            "[%.4f, %.4f] -> [0,1]\n",
            __func__,
            rmin,
            rmax);

    } else {

        // Pure SFT / all rewards identical.
        std::fill(
            window_rewards.begin(),
            window_rewards.end(),
            1.0f);

        LOG_INF(
            "%s: training rewards are constant; "
            "using weight 1.0\n",
            __func__);
    }

    // ------------------------------------------------------------
    // 9. Critical-SFT reward normalization
    // ------------------------------------------------------------

    if (critical_enabled) {

        bool have_train_token_reward =
            false;

        float token_reward_min =
            1.0f;

        float token_reward_max =
            -1.0f;

        // --------------------------------------------------------
        // 9-A. Fit min/max using TRAIN critical tokens only.
        // --------------------------------------------------------

        for (int64_t wi = 0;
             wi < idata_split_out;
             ++wi) {

            for (int32_t j = 0;
                 j < n_ctx;
                 ++j) {

                llama_opt_critical_token_metadata &
                    metadata =
                        critical_metadata[
                            (size_t) wi *
                            (size_t) n_ctx +
                            (size_t) j];

                if (metadata.span_weight == 0.0f) {
                    continue;
                }

                const float clipped =
                    std::max(
                        -1.0f,
                        std::min(
                            1.0f,
                            metadata.reward_weight));

                metadata.reward_weight =
                    clipped;

                token_reward_min =
                    std::min(
                        token_reward_min,
                        clipped);

                token_reward_max =
                    std::max(
                        token_reward_max,
                        clipped);

                have_train_token_reward =
                    true;
            }
        }

        // --------------------------------------------------------
        // 9-B. Apply the TRAIN-fitted transform to ALL metadata.
        // --------------------------------------------------------

        if (have_train_token_reward) {

            const float token_reward_range =
                token_reward_max -
                token_reward_min;

            for (auto & metadata :
                 critical_metadata) {

                if (metadata.span_weight == 0.0f) {
                    continue;
                }

                const float clipped =
                    std::max(
                        -1.0f,
                        std::min(
                            1.0f,
                            metadata.reward_weight));

                if (token_reward_range >
                    1e-6f) {

                    metadata.reward_weight =
                        (clipped -
                         token_reward_min) /
                        token_reward_range;

                    metadata.reward_weight =
                        std::max(
                            0.0f,
                            std::min(
                                1.0f,
                                metadata.reward_weight));

                } else {

                    metadata.reward_weight =
                        1.0f;
                }
            }

        } else {

            // No active Critical-SFT token in train.
            // Keep active token weights neutral.
            for (auto & metadata :
                 critical_metadata) {

                if (metadata.span_weight != 0.0f) {
                    metadata.reward_weight =
                        1.0f;
                }
            }
        }

        // --------------------------------------------------------
        // 9-C. Attach metadata to ggml dataset
        // --------------------------------------------------------

        ggml_opt_dataset_set_aux(
            dataset,
            critical_metadata.data(),
            n_ctx *
            sizeof(
                critical_metadata[0]));

        GGML_ASSERT(
            ggml_opt_dataset_aux_size(
                dataset) ==
            n_ctx *
            sizeof(
                critical_metadata[0]));

        if (critical_metadata_out) {

            *critical_metadata_out =
                std::move(
                    critical_metadata);
        }
    }

    return dataset;
}

// ---------------------------------------------------------------------------
// LoRA tensor allocation
// ---------------------------------------------------------------------------

struct lora_tensors {
    struct ggml_context      * ctx  = nullptr;
    struct ggml_backend_buffer * buf = nullptr;
    // map: base tensor name → {lora_a, lora_b}
    std::unordered_map<std::string, std::pair<ggml_tensor*, ggml_tensor*>> ab;
};

static lora_tensors alloc_lora_tensors(
        const std::string        & model_path,
        const std::vector<std::string> & targets,
        int32_t                   rank,
        std::mt19937            & rng,
        int32_t                   freeze_layers = 0) {

    lora_tensors lt;

    // Open the model GGUF to discover tensor names and shapes
    // without needing access to private llama_model internals.
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params gguf_params = { /*.no_alloc=*/true, /*.ctx=*/&ctx_meta };
    struct gguf_context * ctx_gguf = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open model GGUF for tensor discovery: %s\n",
                __func__, model_path.c_str());
        return lt;
    }

    // Collect matching 2-D tensors
    struct tensor_info { std::string name; int64_t ne0, ne1; };
    std::vector<tensor_info> matched;

    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta);
         t; t = ggml_get_next_tensor(ctx_meta, t)) {
        if (ggml_n_dims(t) < 2) continue;
        if (!tensor_matches_targets(t->name, targets, freeze_layers)) continue;
        matched.push_back({t->name, t->ne[0], t->ne[1]});
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);

    if (matched.empty()) {
        LOG_ERR("%s: no model tensors matched --lora-targets; check spelling\n", __func__);
        return lt;
    }

    if (freeze_layers > 0) {
        LOG_INF("%s: freezing layers blk.0 .. blk.%d (no LoRA allocated; backward already pruned by grads_needed)\n",
                __func__, freeze_layers - 1);
    }
    LOG_INF("%s: allocating LoRA A/B tensors for %zu weight matrices, rank=%d\n",
            __func__, matched.size(), rank);

    // Allocate ggml context for A+B tensors (2 tensors per matched weight)
    const size_t mem = (2 * matched.size() + 16) * ggml_tensor_overhead();
    struct ggml_init_params ip = { mem, nullptr, /*no_alloc=*/true };
    lt.ctx = ggml_init(ip);

    for (const auto & ti : matched) {
        const int64_t in_dim  = ti.ne0; // columns (input features)
        const int64_t out_dim = ti.ne1; // rows    (output features)

        // lora_a: [in_dim, rank]   applied first: a @ x
        // lora_b: [rank,   out_dim] applied second: b @ (a @ x)
        // Convention matches llama-adapter.cpp:48-60:
        //   a->ne[0] == in_dim,  a->ne[1] == rank
        //   b->ne[0] == rank,    b->ne[1] == out_dim
        ggml_tensor * la = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, in_dim, rank);
        ggml_tensor * lb = ggml_new_tensor_2d(lt.ctx, GGML_TYPE_F32, rank,   out_dim);

        ggml_set_name(la, (ti.name + ".lora_a").c_str());
        ggml_set_name(lb, (ti.name + ".lora_b").c_str());

        lt.ab[ti.name] = {la, lb};
    }

    // Allocate backend buffer for all LoRA tensors at once
    lt.buf = ggml_backend_alloc_ctx_tensors_from_buft(lt.ctx, ggml_backend_cpu_buffer_type());

    // Initialize: A ~ N(0, 1/sqrt(rank)), B = 0
    const float std_a = 1.0f / std::sqrt((float)rank);
    std::normal_distribution<float> dist(0.0f, std_a);

    for (auto & kv : lt.ab) {
        ggml_tensor * la = kv.second.first;
        ggml_tensor * lb = kv.second.second;

        // Fill A
        float * data_a = (float *) la->data;
        for (int64_t i = 0; i < ggml_nelements(la); ++i) data_a[i] = dist(rng);
        // Zero B
        memset(lb->data, 0, ggml_nbytes(lb));
    }

    return lt;
}

// ---------------------------------------------------------------------------
// Param filter: only train lora_a / lora_b tensors
// ---------------------------------------------------------------------------

static bool lora_param_filter(const struct ggml_tensor * t, void * /*ud*/) {
    const char * n = t->name;
    const size_t len = strlen(n);
    if (len > 7 && strcmp(n + len - 7, ".lora_a") == 0) return true;
    if (len > 7 && strcmp(n + len - 7, ".lora_b") == 0) return true;
    return false;
}

static bool lora_param_filter_none(const struct ggml_tensor * t, void * userdata) {
    GGML_UNUSED(t);
    GGML_UNUSED(userdata);
    return false;
}

static enum llama_lora_qat_type lora_qat_type_from_string(const std::string & type) {
    if (type == "q3_k") return LLAMA_LORA_QAT_TYPE_Q3_K;
    if (type == "q4_k") return LLAMA_LORA_QAT_TYPE_Q4_K;
    if (type == "q4_0") return LLAMA_LORA_QAT_TYPE_Q4_0;
    if (type == "mxfp4") return LLAMA_LORA_QAT_TYPE_MXFP4;
    if (type == "q6_k") return LLAMA_LORA_QAT_TYPE_Q6_K;
    if (type == "q8_0") return LLAMA_LORA_QAT_TYPE_Q8_0;
    return LLAMA_LORA_QAT_TYPE_NONE;
}

static enum llama_opt_critical_token_mode critical_token_mode_from_string(const std::string & mode) {
    if (mode == "spans") return LLAMA_OPT_CRITICAL_TOKEN_MODE_SPANS;
    if (mode == "confidence") return LLAMA_OPT_CRITICAL_TOKEN_MODE_CONFIDENCE;
    if (mode == "hybrid") return LLAMA_OPT_CRITICAL_TOKEN_MODE_HYBRID;
    return LLAMA_OPT_CRITICAL_TOKEN_MODE_NONE;
}

static enum llama_opt_critical_weight_shape critical_weight_shape_from_string(const std::string & shape) {
    return shape == "linear" ? LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_LINEAR : LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_CONSTANT;
}

// ---------------------------------------------------------------------------
// Save adapter GGUF
// ---------------------------------------------------------------------------

static std::string basename_from_path(const std::string & p) {
    const size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return p;
    return p.substr(pos + 1);
}

struct checkpoint_state {
    std::string mode;
    int64_t     epoch           = 0;
    int64_t     window          = 0;
    int64_t     step            = 0;
    int64_t     dataset_windows = -1;
    int64_t     context_length  = -1;
    bool        shuffle         = false;
    bool        has_metadata    = false;
    float       alpha           = 0.0f;
    bool        has_alpha       = false;
    int64_t     schedule_step   = -1;
};

static bool parse_checkpoint_number(
        const std::string & path,
        size_t              begin,
        size_t              end,
        int64_t           & value) {
    if (begin >= end) return false;

    const std::string number = path.substr(begin, end - begin);
    char * number_end = nullptr;
    errno = 0;
    const long long parsed = strtoll(number.c_str(), &number_end, 10);
    if (errno != 0 || number_end != number.c_str() + number.size() || parsed < 0) return false;

    value = (int64_t) parsed;
    return true;
}

static bool checkpoint_state_from_filename(
        const std::string & path,
        const std::string & mode,
        checkpoint_state  & state) {
    const size_t suffix = path.rfind(".gguf");
    if (suffix == std::string::npos || suffix + 5 != path.size()) return false;

    if (mode == "sft") {
        const size_t epoch_tag = path.rfind(".epoch", suffix);
        const size_t ckpt_tag  = path.rfind(".ckpt",  suffix);
        if (epoch_tag == std::string::npos || ckpt_tag == std::string::npos || epoch_tag >= ckpt_tag) return false;
        if (!parse_checkpoint_number(path, epoch_tag + 6, ckpt_tag, state.epoch)) return false;
        if (!parse_checkpoint_number(path, ckpt_tag + 5, suffix, state.window)) return false;
    } else {
        const size_t ckpt_tag = path.rfind(".ckpt", suffix);
        if (ckpt_tag == std::string::npos) return false;
        if (!parse_checkpoint_number(path, ckpt_tag + 5, suffix, state.step)) return false;
    }

    state.mode = mode;
    return true;
}

static bool gguf_get_i64(const gguf_context * gctx, const char * key, int64_t & value) {
    const int64_t key_id = gguf_find_key(gctx, key);
    if (key_id < 0) return false;

    switch (gguf_get_kv_type(gctx, key_id)) {
        case GGUF_TYPE_UINT32: value = gguf_get_val_u32(gctx, key_id); return true;
        case GGUF_TYPE_INT32:  value = gguf_get_val_i32(gctx, key_id); return true;
        case GGUF_TYPE_UINT64: {
            const uint64_t val = gguf_get_val_u64(gctx, key_id);
            if (val > INT64_MAX) return false;
            value = (int64_t) val;
            return true;
        }
        case GGUF_TYPE_INT64: value = gguf_get_val_i64(gctx, key_id); return true;
        default: return false;
    }
}

static bool load_checkpoint_state(
        const std::string & path,
        const std::string & expected_mode,
        checkpoint_state  & state) {
    ggml_context * ctx_meta = nullptr;
    const gguf_init_params params = { true, &ctx_meta };
    gguf_context * gctx = gguf_init_from_file(expand_tilde(path).c_str(), params);
    if (!gctx) {
        LOG_ERR("%s: cannot open checkpoint %s\n", __func__, path.c_str());
        return false;
    }

    const int64_t mode_id = gguf_find_key(gctx, "training.checkpoint.mode");
    if (mode_id >= 0 && gguf_get_kv_type(gctx, mode_id) == GGUF_TYPE_STRING) {
        state.mode = gguf_get_val_str(gctx, mode_id);
        state.has_metadata = true;
        gguf_get_i64(gctx, "training.checkpoint.epoch", state.epoch);
        gguf_get_i64(gctx, "training.checkpoint.window", state.window);
        gguf_get_i64(gctx, "training.checkpoint.step", state.step);
        gguf_get_i64(gctx, "training.dataset.windows", state.dataset_windows);
        gguf_get_i64(gctx, "training.context_length", state.context_length);
        gguf_get_i64(gctx, "training.scheduler.step", state.schedule_step);

        const int64_t shuffle_id = gguf_find_key(gctx, "training.dataset.shuffle");
        if (shuffle_id >= 0 && gguf_get_kv_type(gctx, shuffle_id) == GGUF_TYPE_BOOL) {
            state.shuffle = gguf_get_val_bool(gctx, shuffle_id);
        }
    }

    const int64_t alpha_id = gguf_find_key(gctx, "adapter.lora.alpha");
    if (alpha_id >= 0 && gguf_get_kv_type(gctx, alpha_id) == GGUF_TYPE_FLOAT32) {
        state.alpha = gguf_get_val_f32(gctx, alpha_id);
        state.has_alpha = true;
    }

    gguf_free(gctx);
    ggml_free(ctx_meta);

    if (!state.has_metadata && !checkpoint_state_from_filename(path, expected_mode, state)) {
        LOG_ERR("%s: %s has no checkpoint metadata and its filename does not contain a checkpoint position\n",
                __func__, path.c_str());
        return false;
    }
    if (state.mode != expected_mode) {
        LOG_ERR("%s: checkpoint mode is %s, expected %s\n",
                __func__, state.mode.c_str(), expected_mode.c_str());
        return false;
    }
    if (state.mode == "sft" && state.epoch <= 0) {
        LOG_ERR("%s: invalid checkpoint epoch %ld\n", __func__, (long) state.epoch);
        return false;
    }
    if (state.window < 0 || state.step < 0 || state.schedule_step < -1) {
        LOG_ERR("%s: checkpoint contains a negative training position\n", __func__);
        return false;
    }
    return true;
}

static bool save_adapter(
        const lora_tensors & lt,
        const std::string  & out_path,
        const std::string  & arch,
        float                alpha,
        const std::string  & base_model_path,
        const checkpoint_state * checkpoint = nullptr) {

    // Build output GGUF context
    struct gguf_context * gctx = gguf_init_empty();

    // Metadata required by llama_adapter_lora_init
    gguf_set_val_str(gctx, "general.type",         "adapter");
    gguf_set_val_str(gctx, "general.architecture",  arch.c_str());
    gguf_set_val_str(gctx, "adapter.type",          "lora");
    gguf_set_val_f32(gctx, "adapter.lora.alpha",    alpha);
    gguf_set_val_str(gctx, "adapter.base_model",    basename_from_path(base_model_path).c_str());

    if (checkpoint) {
        gguf_set_val_u32(gctx, "training.checkpoint.version", 1);
        gguf_set_val_str(gctx, "training.checkpoint.mode", checkpoint->mode.c_str());
        gguf_set_val_i64(gctx, "training.checkpoint.epoch", checkpoint->epoch);
        gguf_set_val_i64(gctx, "training.checkpoint.window", checkpoint->window);
        gguf_set_val_i64(gctx, "training.checkpoint.step", checkpoint->step);
        gguf_set_val_i64(gctx, "training.dataset.windows", checkpoint->dataset_windows);
        gguf_set_val_i64(gctx, "training.context_length", checkpoint->context_length);
        gguf_set_val_bool(gctx, "training.dataset.shuffle", checkpoint->shuffle);
        gguf_set_val_i64(gctx, "training.scheduler.step", checkpoint->schedule_step);
    }

    // Register tensors
    for (const auto & kv : lt.ab) {
        gguf_add_tensor(gctx, kv.second.first);   // lora_a
        gguf_add_tensor(gctx, kv.second.second);  // lora_b
    }

    // Write: meta placeholder → tensor data → rewrite meta
    const std::string real_path = expand_tilde(out_path);
    std::ofstream fout(real_path, std::ios::binary);
    if (!fout.is_open()) {
        LOG_ERR("%s: cannot open %s for writing\n", __func__, real_path.c_str());
        gguf_free(gctx);
        return false;
    }

    // Write meta placeholder
    const size_t meta_size = gguf_get_meta_size(gctx);
    std::vector<char> zeros_buf(meta_size, 0);
    fout.write(zeros_buf.data(), meta_size);

    // Write tensor data — copy to CPU first in case tensors live on GPU
    for (const auto & kv : lt.ab) {
        for (ggml_tensor * t : {kv.second.first, kv.second.second}) {
            const size_t nb = ggml_nbytes(t);
            std::vector<char> cpu_buf(nb);
            ggml_backend_tensor_get(t, cpu_buf.data(), 0, nb);
            fout.write(cpu_buf.data(), nb);
            // GGUF tensors are 32-byte aligned
            const size_t pad = GGML_PAD(nb, 32) - nb;
            if (pad > 0) {
                std::vector<char> pad_buf(pad, 0);
                fout.write(pad_buf.data(), pad);
            }
        }
    }

    // Re-write metadata at offset 0
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(gctx, meta.data());
    fout.seekp(0);
    fout.write((const char *) meta.data(), meta_size);

    fout.close();
    gguf_free(gctx);

    if (!fout) {
        LOG_ERR("%s: failed while writing %s\n", __func__, real_path.c_str());
        return false;
    }

    LOG_INF("%s: adapter saved to %s\n", __func__, real_path.c_str());
    return true;
}

struct adapter_stats {
    double a_l2      = 0.0;
    double b_l2      = 0.0;
    double total_l2  = 0.0;
    double a_max_abs = 0.0;
    double b_max_abs = 0.0;
};

static void adapter_stats_accum(ggml_tensor * t, double & sum_sq, double & max_abs) {
    const size_t nb = ggml_nbytes(t);
    std::vector<float> buf(nb / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    for (float v : buf) {
        const double vd = (double) v;
        const double av = std::abs(vd);
        sum_sq += vd * vd;
        max_abs = std::max(max_abs, av);
    }
}

static adapter_stats adapter_get_stats(const lora_tensors & lt) {
    adapter_stats stats;
    double a_sum_sq = 0.0;
    double b_sum_sq = 0.0;

    for (const auto & kv : lt.ab) {
        adapter_stats_accum(kv.second.first,  a_sum_sq, stats.a_max_abs);
        adapter_stats_accum(kv.second.second, b_sum_sq, stats.b_max_abs);
    }

    stats.a_l2     = std::sqrt(a_sum_sq);
    stats.b_l2     = std::sqrt(b_sum_sq);
    stats.total_l2 = std::sqrt(a_sum_sq + b_sum_sq);
    return stats;
}

// ---------------------------------------------------------------------------
// Periodic checkpoint callback
// ---------------------------------------------------------------------------

struct save_ctx {
    llama_context      * target_ctx;
    const lora_tensors * lt;
    const std::string  * lora_out;
    const std::string  * arch;
    const std::string  * base_model_path;
    float                lora_alpha;
    bool                 save_target;
    llama_context      * mtp_ctx;
    const lora_tensors * mtp_lt;
    const std::string  * mtp_lora_out;
    const std::string  * mtp_arch;
    const std::string  * mtp_model_path;
    float                mtp_alpha;
    int32_t              save_every;     // 0 = disabled
    int32_t              ubatch_per_ctx;
    int64_t              window_offset;  // completed windows before this epoch call
    int64_t              last_saved;     // last window index at which we saved
    int64_t              epoch;
    int64_t              dataset_windows;
    int64_t              context_length;
    bool                 shuffle;
    bool                 verbose_loss;
    qlora_lr_schedule * schedule;
    bool                 loss_ema_initialized = false;
    double               loss_ema16 = 0.0;
    double               loss_ema64 = 0.0;
    double               loss_cumulative_sum = 0.0;
    int64_t              loss_cumulative_count = 0;
    double               loss_epoch_sum = 0.0;
    int64_t              loss_epoch_count = 0;
};

// TLS pointer set before each epoch so the static callback can access it.
static thread_local save_ctx * g_save_ctx = nullptr;

static void save_every_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);

    // Log loss at every window boundary so we can see if/when it diverges.
    if (train && g_save_ctx) {
        const int64_t window = g_save_ctx->window_offset + ibatch / g_save_ctx->ubatch_per_ctx;
        const int64_t ubatch_in_window = ibatch % g_save_ctx->ubatch_per_ctx;
        if (ibatch > 0 && ubatch_in_window == 0) {
            if (g_save_ctx->verbose_loss) {
                double cumulative_loss_mean = 0.0;
                ggml_opt_result_loss(result, &cumulative_loss_mean, nullptr);

                const int64_t cumulative_loss_count = ibatch;
                const double cumulative_loss_sum = cumulative_loss_mean * cumulative_loss_count;
                const double window_loss_sum = cumulative_loss_sum - g_save_ctx->loss_cumulative_sum;
                const int64_t window_loss_count = cumulative_loss_count - g_save_ctx->loss_cumulative_count;
                const double window_loss = window_loss_sum / std::max<int64_t>(1, window_loss_count);

                g_save_ctx->loss_ema16 = g_save_ctx->loss_ema_initialized
                    ? 0.8825 * g_save_ctx->loss_ema16 + 0.1175 * window_loss
                    : window_loss;
                g_save_ctx->loss_ema64 = g_save_ctx->loss_ema_initialized
                    ? 0.9692 * g_save_ctx->loss_ema64 + 0.0308 * window_loss
                    : window_loss;
                g_save_ctx->loss_ema_initialized = true;
                g_save_ctx->loss_cumulative_sum = cumulative_loss_sum;
                g_save_ctx->loss_cumulative_count = cumulative_loss_count;
                g_save_ctx->loss_epoch_sum += window_loss_sum;
                g_save_ctx->loss_epoch_count += window_loss_count;

                const double epoch_mean = g_save_ctx->loss_epoch_sum /
                    std::max<int64_t>(1, g_save_ctx->loss_epoch_count);
                fprintf(stderr,
                        "\nepoch=%ld window=%ld window_loss=%.6f ema16=%.6f ema64=%.6f epoch_mean=%.6f lr=%.6g\n",
                        (long) g_save_ctx->epoch, (long) window, window_loss,
                        g_save_ctx->loss_ema16, g_save_ctx->loss_ema64, epoch_mean,
                        (double) g_save_ctx->schedule->current_lr);
            }
            ++g_save_ctx->schedule->step;
        }
    }

    if (!train || !g_save_ctx || g_save_ctx->save_every <= 0) return;
    const int64_t window = g_save_ctx->window_offset + ibatch / g_save_ctx->ubatch_per_ctx;
    if (window > 0 && window != g_save_ctx->last_saved && window % g_save_ctx->save_every == 0) {
        checkpoint_state state;
        state.mode            = "sft";
        state.epoch           = g_save_ctx->epoch;
        state.window          = window;
        state.dataset_windows = g_save_ctx->dataset_windows;
        state.context_length  = g_save_ctx->context_length;
        state.shuffle         = g_save_ctx->shuffle;
        state.schedule_step   = g_save_ctx->schedule->step;

        bool saved = true;
        if (g_save_ctx->save_target) {
            const std::string ckpt = *g_save_ctx->lora_out
                + ".epoch" + std::to_string(g_save_ctx->epoch)
                + ".ckpt" + std::to_string(window) + ".gguf";
            llama_synchronize(g_save_ctx->target_ctx);
            saved = save_adapter(*g_save_ctx->lt, ckpt, *g_save_ctx->arch, g_save_ctx->lora_alpha,
                                 *g_save_ctx->base_model_path, &state) && saved;
            if (saved) {
                LOG_INF("save_every_callback: target checkpoint saved -> %s (window %ld)\n",
                        ckpt.c_str(), (long) window);
            }
        }
        if (g_save_ctx->mtp_ctx) {
            const std::string ckpt = *g_save_ctx->mtp_lora_out
                + ".epoch" + std::to_string(g_save_ctx->epoch)
                + ".ckpt" + std::to_string(window) + ".gguf";
            llama_synchronize(g_save_ctx->mtp_ctx);
            const bool mtp_saved = save_adapter(*g_save_ctx->mtp_lt, ckpt, *g_save_ctx->mtp_arch,
                g_save_ctx->mtp_alpha, *g_save_ctx->mtp_model_path, &state);
            saved = mtp_saved && saved;
            if (mtp_saved) {
                LOG_INF("save_every_callback: MTP checkpoint saved -> %s (window %ld)\n",
                        ckpt.c_str(), (long) window);
            }
        }
        fprintf(stderr, "\n");
        if (saved) {
            g_save_ctx->last_saved = window;
        } else {
            LOG_ERR("save_every_callback: checkpoint set failed at window %ld\n", (long) window);
        }
    }
}

// ---------------------------------------------------------------------------
// IPC helpers  (stdout protocol, stdin commands)
// ---------------------------------------------------------------------------

// Escape newlines and backslashes for single-line IPC transmission.
// Mirrors _escape() in gguf_trainer.py.
static std::string ipc_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if      (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static void ipc_emit(const char * msg) {
    fputs(msg, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

// Read one line from stdin, trimming the trailing newline.
// Returns false on EOF or error.
static bool ipc_read_line(std::string & out) {
    out.clear();
    if (!std::getline(std::cin, out)) return false;
    // Strip trailing \r if present (Windows line endings)
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
}

// Parse "REWARD r1 r2 ... rN" into a float vector.
static std::vector<float> ipc_parse_rewards(const std::string & line) {
    std::vector<float> rewards;
    if (line.size() < 8 || line.substr(0, 7) != "REWARD ") return rewards;
    std::istringstream ss(line.substr(7));
    float r;
    while (ss >> r) rewards.push_back(r);
    return rewards;
}

// ---------------------------------------------------------------------------
// Greedy / temperature sampling for GRPO rollout generation
// ---------------------------------------------------------------------------

static std::string generate_response(
        llama_context      * ctx,
        llama_model        * model,
        const std::string  & prompt,
        int32_t              max_tokens,
        float                temperature,
        std::mt19937       & rng) {

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(ctx, prompt, /*add_special=*/true, /*parse_special=*/true);
    if (tokens.empty()) return "";

    // Clear KV cache before each generation (don't carry over previous prompt state)
    llama_memory_clear(llama_get_memory(ctx), true);
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed on prompt\n", __func__);
            return "";
        }
    }

    std::string output;
    const llama_token eos = llama_vocab_eos(vocab);
    //const llama_token nl  = llama_vocab_nl(vocab);

    // For ChatML models <|im_end|> is the turn-end marker but may not be the
    // vocab EOS token.  Look it up by tokenizing the string and taking the
    // first token if it tokenizes to exactly one piece.
    llama_token im_end = -1;
    {
        std::vector<llama_token> im_end_tokens(8);
        static const char im_end_str[] = "<|im_end|>";
        int n = llama_tokenize(vocab, im_end_str, (int32_t)strlen(im_end_str), im_end_tokens.data(), (int32_t)im_end_tokens.size(), /*add_special=*/false, /*parse_special=*/true);
        if (n == 1) im_end = im_end_tokens[0];
    }
    const llama_token eot = llama_vocab_eot(vocab);  // may equal eos on some models

    for (int32_t i = 0; i < max_tokens; ++i) {
        // Sample next token — use ith=-1 to always get the LAST output position's
        // logits.  llama_get_logits(ctx) returns position 0 which is wrong when the
        // prompt batch has multiple output tokens (training context).
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            LOG_ERR("%s: llama_get_logits_ith(-1) returned NULL\n", __func__);
            break;
        }
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);

        llama_token next_token;
        if (temperature <= 0.0f) {
            // Greedy
            next_token = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
        } else {
            // Temperature sampling via softmax + categorical draw
            std::vector<float> probs(n_vocab);
            float max_logit = *std::max_element(logits, logits + n_vocab);
            float sum = 0.0f;
            for (int32_t k = 0; k < n_vocab; ++k) {
                probs[k] = std::exp((logits[k] - max_logit) / temperature);
                sum += probs[k];
            }
            for (float & p : probs) p /= sum;
            std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
            next_token = dist(rng);
        }

        if (next_token == eos)     break;
        if (next_token == eot)     break;
        if (im_end >= 0 && next_token == im_end && !output.empty()) break;

        // Decode token to text
        char buf[256] = {};
        llama_token_to_piece(vocab, next_token, buf, sizeof(buf) - 1, 0, true);
        output += buf;

        // Feed token back for next step
        llama_batch batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }

    return output;
}

// ---------------------------------------------------------------------------
// Compatibility overload for legacy QLoRA / GRPO call sites.
//
// Native QAT uses the sample_split/idata_split_out overload directly.
// Existing callers that do not provide a sample-level split keep treating
// the full input as one training partition.
// ---------------------------------------------------------------------------

static ggml_opt_dataset_t build_dataset(
        const std::vector<training_sample> & samples,
        int32_t                              n_ctx,
        std::vector<float>                 & window_rewards,
        bool                                 train_on_prompt = false,
        llama_token                          bos_token = -1,
        bool                                 critical_enabled = false,
        std::vector<llama_opt_critical_token_metadata> * critical_metadata_out = nullptr) {

    int64_t idata_split_unused = 0;

    return build_dataset(
        samples,
        n_ctx,
        window_rewards,

        // Legacy paths provide their own split behavior later.
        // Pack all raw samples as one partition here.
        (int64_t) samples.size(),

        idata_split_unused,
        train_on_prompt,
        bos_token,
        critical_enabled,
        critical_metadata_out);
}

// ---------------------------------------------------------------------------
// GRPO IPC training loop
// ---------------------------------------------------------------------------

// Volatile flag set by SIGINT so the loop can exit cleanly.
static volatile sig_atomic_t g_grpo_stop = 0;
static void grpo_sigint_handler(int) { g_grpo_stop = 1; }

static int run_grpo_mode(
        common_params    & params,
        llama_model      * model,
        llama_context    * ctx,
        lora_tensors     & lt,
        const std::string & arch,
        float               lora_alpha,
        const std::string & base_model_path,
        const checkpoint_state * resume) {

    const int32_t n_ctx    = llama_n_ctx(ctx);
    const int32_t n_gen    = params.grpo_n_gen;
    const int32_t n_steps  = params.grpo_n_steps;
    const float   temp     = params.grpo_temperature;
    const int32_t max_tok  = params.grpo_max_tokens;
    int           step     = resume ? (int) resume->step : 0;

    if (resume && resume->context_length >= 0 && resume->context_length != n_ctx) {
        LOG_ERR("%s: checkpoint context length is %ld, current context length is %d\n",
                __func__, (long) resume->context_length, n_ctx);
        return 1;
    }
    if (n_steps <= 0 || step > n_steps) {
        LOG_ERR("%s: invalid GRPO step range %d/%d\n", __func__, step, n_steps);
        return 1;
    }
    if (params.warmup_steps > n_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total GRPO steps %d\n",
                __func__, params.warmup_steps, n_steps);
        return 1;
    }
    if (params.lr_scheduler == "cosine" && params.lr_decay_steps > 0 &&
        params.lr_decay_steps <= params.warmup_steps) {
        LOG_ERR("%s: --lr-decay-steps %d must exceed --warmup-steps %d\n",
                __func__, params.lr_decay_steps, params.warmup_steps);
        return 1;
    }
    if (params.lr.lr0 <= 0.0f || (params.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n", __func__);
        return 1;
    }

    const int64_t schedule_step = resume && resume->schedule_step >= 0
        ? resume->schedule_step : step;
    if (schedule_step != step) {
        LOG_ERR("%s: checkpoint scheduler step %ld does not match GRPO step %d\n",
                __func__, (long) schedule_step, step);
        return 1;
    }
    qlora_lr_schedule schedule {
        &params.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio,
        params.lr_decay_steps, n_steps, schedule_step
    };

    std::mt19937 rng(params.sampling.seed != LLAMA_DEFAULT_SEED
                     ? params.sampling.seed : 42);

    // Initialize optimizer
    struct llama_opt_params lopt_params {
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/qlora_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&schedule,
        /*.optimizer_type           =*/params.optimizer,
        /*.lora_qat_type            =*/lora_qat_type_from_string(params.lora_qat),
        /*.grad_checkpoint_interval =*/params.grad_checkpoint_interval,
        /*.critical_token_mode      =*/LLAMA_OPT_CRITICAL_TOKEN_MODE_NONE,
        /*.critical_token_weight    =*/1.0f,
        /*.critical_confidence_threshold =*/0.25f,
        /*.critical_weight_shape    =*/LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_CONSTANT,
        /*.critical_warmup_steps    =*/0,
        /*.critical_max_fraction    =*/1.0f,
        /*.critical_step            =*/nullptr,
        /*.critical_stats_every     =*/0,
        /*.train_target             =*/true,
    };
    llama_opt_init(ctx, model, lopt_params);

    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));

    signal(SIGINT, grpo_sigint_handler);

    // Signal Python that we are ready
    ipc_emit("[QLORA:READY]");

    float last_loss = 0.0f;

    if (resume) {
        LOG_INF("%s: resuming GRPO after step %d/%d; optimizer state starts fresh\n",
                __func__, step, n_steps);
    }
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g total_steps=%d start_step=%ld lr=%.3g lr_min=%.3g\n",
            __func__, params.lr_scheduler.c_str(), params.warmup_steps, (double) params.warmup_init_ratio, n_steps,
            (long) schedule.step, (double) params.lr.lr0, (double) std::max(0.0f, params.lr.lr_min));

    while (step < n_steps && !g_grpo_stop) {
        params.lr.epoch = step;

        // ── Request prompt ────────────────────────────────────────────────
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[QLORA:PROMPT_REQ:%d]", step + 1);
            ipc_emit(buf);
        }

        std::string prompt_line;
        if (!ipc_read_line(prompt_line)) break;
        if (prompt_line == "STOP") {
            LOG_INF("grpo: received STOP from Python\n");
            break;
        }
        if (prompt_line.size() < 8 || prompt_line.substr(0, 7) != "PROMPT ") {
            char buf[128];
            snprintf(buf, sizeof(buf), "[QLORA:ERROR] expected PROMPT, got: %.80s", prompt_line.c_str());
            ipc_emit(buf);
            return 1;
        }
        // Unescape the prompt (\\n → \n etc.)
        std::string prompt;
        {
            const std::string esc = prompt_line.substr(7);
            prompt.reserve(esc.size());
            for (size_t i = 0; i < esc.size(); ++i) {
                if (esc[i] == '\\' && i + 1 < esc.size()) {
                    char next = esc[i+1];
                    if      (next == 'n')  { prompt += '\n'; ++i; }
                    else if (next == 'r')  { prompt += '\r'; ++i; }
                    else if (next == '\\') { prompt += '\\'; ++i; }
                    else                   { prompt += esc[i]; }
                } else {
                    prompt += esc[i];
                }
            }
        }

        // ── Generate N responses ──────────────────────────────────────────
        std::vector<std::string> generations(n_gen);
        for (int k = 0; k < n_gen; ++k) {
            generations[k] = generate_response(ctx, model, prompt, max_tok, temp, rng);

            char hdr[64];
            snprintf(hdr, sizeof(hdr), "[QLORA:GEN:%d/%d] ", k + 1, n_gen);
            std::string msg = std::string(hdr) + ipc_escape(generations[k]);
            ipc_emit(msg.c_str());
        }

        // ── Request rewards ───────────────────────────────────────────────
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "[QLORA:REWARD_REQ:%d]", n_gen);
            ipc_emit(buf);
        }

        std::string reward_line;
        if (!ipc_read_line(reward_line)) break;
        if (reward_line == "STOP") {
            LOG_INF("grpo: received STOP from Python\n");
            break;
        }
        std::vector<float> rewards = ipc_parse_rewards(reward_line);
        if ((int32_t)rewards.size() != n_gen) {
            char buf[128];
            snprintf(buf, sizeof(buf), "[QLORA:ERROR] expected %d rewards, got %zu", n_gen, rewards.size());
            ipc_emit(buf);
            return 1;
        }

        // ── Build single-step mini-dataset: prompt+generations with rewards ─
        // Each generation is a separate sample; prompt = no-loss, generation = loss.
        std::vector<training_sample> step_samples;
        step_samples.reserve(n_gen);
        for (int k = 0; k < n_gen; ++k) {
            training_sample s;
            s.reward = rewards[k];

            auto tok_prompt = common_tokenize(ctx, prompt,         /*add_special=*/true,  /*parse_special=*/true);
            auto tok_gen    = common_tokenize(ctx, generations[k], /*add_special=*/false, /*parse_special=*/true);

            s.tokens.insert(s.tokens.end(), tok_prompt.begin(), tok_prompt.end());
            s.tokens.insert(s.tokens.end(), tok_gen.begin(),    tok_gen.end());
            s.is_label.resize(s.tokens.size(), false);
            for (size_t i = tok_prompt.size(); i < s.tokens.size(); ++i) {
                s.is_label[i] = true;
            }
            step_samples.push_back(std::move(s));
        }

        // Ensure minimum token count for one context window.
        // build_dataset drops the last token per sample during flattening,
        // so we need total raw tokens > n_ctx to guarantee ndata >= 1.
        while (true) {
            size_t total = 0;
            for (const auto & s : step_samples) total += s.tokens.size();
            if ((int64_t)total > n_ctx + (int64_t)step_samples.size()) break;
            step_samples.push_back(step_samples.back());
        }

        std::vector<float> window_rewards;
        ggml_opt_dataset_t step_dataset = build_dataset(
                step_samples, n_ctx, window_rewards, /*train_on_prompt=*/false, bos);
        if (!step_dataset) {
            ipc_emit("[QLORA:ERROR] build_dataset failed for step");
            return 1;
        }

        // Apply reward weights for this step
        const bool has_rewards = std::any_of(window_rewards.begin(), window_rewards.end(),
                                             [](float r){ return std::abs(r - 1.0f) > 1e-4f; });
        if (has_rewards) {
            llama_opt_set_reward_weights(window_rewards.data(), (int64_t)window_rewards.size());
        }

        // ── One optimizer step (full dataset = one mini-epoch) ────────────
        const int64_t idata_all = ggml_opt_dataset_ndata(step_dataset);
        ggml_opt_result_t step_result = ggml_opt_result_init();

        llama_opt_epoch(ctx, step_dataset, step_result, nullptr, idata_all,
                        nullptr,   // no progress bar callback — clean stdout
                        nullptr,
                        false);    // no shuffle for single-step

        double loss = 0.0, loss_unc = 0.0;
        ggml_opt_result_loss(step_result, &loss, &loss_unc);
        last_loss = (float)loss;

        ggml_opt_result_free(step_result);
        ggml_opt_dataset_free(step_dataset);
        llama_opt_set_reward_weights(nullptr, 0);

        ++step;
        ++schedule.step;

        // ── Emit progress ─────────────────────────────────────────────────
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "[QLORA:PROGRESS] step=%d/%d loss=%.4f epoch=1/1",
                     step, n_steps, last_loss);
            ipc_emit(buf);
        }

        // ── Optional checkpoint ───────────────────────────────────────────
        if (params.save_every > 0 && step % params.save_every == 0) {
            std::string ckpt = params.lora_out + ".ckpt" + std::to_string(step) + ".gguf";
            checkpoint_state state;
            state.mode           = "grpo";
            state.step           = step;
            state.context_length = n_ctx;
            state.schedule_step  = schedule.step;
            save_adapter(lt, ckpt, arch, lora_alpha, base_model_path, &state);
            char buf[512];
            snprintf(buf, sizeof(buf), "[QLORA:CHECKPOINT] %s", ckpt.c_str());
            ipc_emit(buf);
        }
    }

    // Save final adapter
    save_adapter(lt, params.lora_out, arch, lora_alpha, base_model_path);

    {
        char buf[64];
        snprintf(buf, sizeof(buf), "[QLORA:DONE] final_loss=%.4f", last_loss);
        ipc_emit(buf);
    }

    return 0;
}

// ---------------------------------------------------------------------------
struct mtp_training_state {
    llama_model_ptr model;
    llama_context_ptr ctx;
    llama_adapter_lora_ptr adapter;
    lora_tensors tensors;
    lr_opt lr;
    qlora_lr_schedule schedule {};
    int64_t * schedule_step = nullptr;
    std::string arch;
    float alpha = 0.0f;
};

static ggml_opt_optimizer_params mtp_opt_lr_pars(void * userdata) {
    mtp_training_state & state = *(mtp_training_state *) userdata;
    state.schedule.step = *state.schedule_step;
    return qlora_opt_lr_pars(&state.schedule);
}

static bool mtp_read_arch(const std::string & path, std::string & arch) {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_init_params params = { true, &ctx_meta };
    struct gguf_context * ctx_gguf = gguf_init_from_file(path.c_str(), params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open MTP model GGUF %s\n", __func__, path.c_str());
        return false;
    }
    const int kid = gguf_find_key(ctx_gguf, "general.architecture");
    if (kid >= 0) {
        arch = gguf_get_val_str(ctx_gguf, kid);
    }
    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    return !arch.empty();
}

static bool mtp_init_training(
        common_params       & params,
        llama_context       * ctx_target,
        qlora_lr_schedule   & target_schedule,
        mtp_training_state  & state) {
    if (params.mtp_mode == "off") {
        return true;
    }
    if (params.mtp_model.empty()) {
        LOG_ERR("%s: --mtp-model is required when --mtp-mode is not off\n", __func__);
        return false;
    }
    const uint32_t target_ubatch = llama_n_ubatch(ctx_target);
    if (params.mtp_ubatch > (int32_t) target_ubatch || target_ubatch % params.mtp_ubatch != 0) {
        LOG_ERR("%s: --mtp-ubatch must divide the target microbatch (%u)\n", __func__, target_ubatch);
        return false;
    }
    if (!mtp_read_arch(params.mtp_model, state.arch)) {
        return false;
    }

    state.alpha = params.mtp_lora_alpha > 0.0f ? params.mtp_lora_alpha : (float) params.mtp_lora_rank;
    std::string adapter_path = params.mtp_lora_resume;
    const bool resume_adapter = !adapter_path.empty();
    if (!resume_adapter) {
        std::mt19937 rng(43);
        state.tensors = alloc_lora_tensors(params.mtp_model, split_csv(params.mtp_lora_targets), params.mtp_lora_rank, rng, 0);
        if (state.tensors.ab.empty()) {
            return false;
        }
        adapter_path = params.mtp_lora_out + ".init.gguf";
        if (!save_adapter(state.tensors, adapter_path, state.arch, state.alpha, params.mtp_model)) {
            return false;
        }
    }

    auto mparams = common_model_params_to_llama(params);
    state.model.reset(llama_model_load_from_file(params.mtp_model.c_str(), mparams));
    if (!state.model) {
        LOG_ERR("%s: failed to load MTP model %s\n", __func__, params.mtp_model.c_str());
        return false;
    }
    state.adapter.reset(llama_adapter_lora_init(state.model.get(), adapter_path.c_str()));
    if (!resume_adapter) {
        std::remove(expand_tilde(adapter_path).c_str());
    }
    if (!state.adapter) {
        LOG_ERR("%s: failed to load MTP LoRA adapter %s\n", __func__, adapter_path.c_str());
        return false;
    }
    if (resume_adapter) {
        LOG_INF("%s: resumed MTP LoRA weights from %s; optimizer state starts fresh\n",
                __func__, adapter_path.c_str());
    }

    auto cparams = common_context_params_to_llama(params);
    cparams.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cparams.ctx_other = ctx_target;
    cparams.n_ctx     = llama_n_ctx(ctx_target);
    cparams.n_batch   = llama_n_batch(ctx_target);
    cparams.n_ubatch  = params.mtp_ubatch;
    state.ctx.reset(llama_init_from_model(state.model.get(), cparams));
    if (!state.ctx) {
        LOG_ERR("%s: failed to create MTP training context\n", __func__);
        return false;
    }

    llama_adapter_lora * adapter = state.adapter.get();
    float scale = 1.0f;
    llama_set_adapters_lora(state.ctx.get(), &adapter, 1, &scale);
    state.tensors.ab.clear();
    for (auto & item : state.adapter->ab_map) {
        ggml_set_param(item.second.a);
        ggml_set_param(item.second.b);
        state.tensors.ab[item.first] = { item.second.a, item.second.b };
    }

    state.lr = params.lr;
    // MTP is initialized before the target training loop assigns
    // params.lr.epoch.  Do not inherit that indeterminate field: apart from
    // misleading `epoch <garbage> lr=...` logs it can affect epoch-based
    // schedules in the MTP optimizer.
    state.lr.epoch = 0;
    if (params.mtp_learning_rate > 0.0f) {
        state.lr.lr0 = params.mtp_learning_rate;
        if (params.lr_scheduler == "cosine" && state.lr.lr_min > state.lr.lr0) {
            state.lr.lr_min = 0.0f;
        }
    }
    state.schedule = { &state.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio,
        target_schedule.decay_steps, target_schedule.total_steps, target_schedule.step };
    state.schedule_step = &target_schedule.step;

    struct llama_opt_params opt_params {
        /*.n_ctx_train              =*/ llama_n_ctx(ctx_target),
        /*.param_filter             =*/ lora_param_filter,
        /*.param_filter_ud          =*/ nullptr,
        /*.get_opt_pars             =*/ mtp_opt_lr_pars,
        /*.get_opt_pars_ud          =*/ &state,
        /*.optimizer_type           =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
        /*.lora_qat_type            =*/ LLAMA_LORA_QAT_TYPE_NONE,
        /*.grad_checkpoint_interval =*/ 0,
        /*.critical_token_mode      =*/ LLAMA_OPT_CRITICAL_TOKEN_MODE_NONE,
        /*.critical_token_weight    =*/ 1.0f,
        /*.critical_confidence_threshold =*/ 0.25f,
        /*.critical_weight_shape    =*/ LLAMA_OPT_CRITICAL_WEIGHT_SHAPE_CONSTANT,
        /*.critical_warmup_steps    =*/ 0,
        /*.critical_max_fraction    =*/ 1.0f,
        /*.critical_step            =*/ &target_schedule.step,
        /*.critical_stats_every     =*/ 0,
        /*.train_target             =*/ true,
    };
    llama_opt_init(state.ctx.get(), state.model.get(), opt_params);
    llama_opt_set_mtp_context(ctx_target, state.ctx.get());
    LOG_INF("%s: enabled MTP LoRA mode=%s rank=%d alpha=%.1f ubatch=%d lr=%.6g\n",
            __func__, params.mtp_mode.c_str(), params.mtp_lora_rank, state.alpha,
            params.mtp_ubatch, (double) state.lr.lr0);
    return true;
}

#ifndef LLAMA_FINETUNE_QLORA_SHARED_ONLY
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE_QLORA)) {
        return 1;
    }
    if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_QLION_QAT) {
        LOG_ERR("%s: qlion is only supported by llama-finetune-qlion\n", __func__);
        return 1;
    }

    if (!params.grpo_mode && params.train_file.empty()) {
        LOG_ERR("%s: --train-file is required (or use --grpo-mode for IPC training)\n", __func__);
        return 1;
    }
    if (params.mtp_mode != "off" && params.grpo_mode) {
        LOG_ERR("%s: MTP training is not available with --grpo-mode\n", __func__);
        return 1;
    }

    checkpoint_state resume_state;
    const bool resume_requested = !params.lora_resume.empty();
    if (resume_requested) {
        if (!params.lora_adapters.empty()) {
            LOG_ERR("%s: --resume cannot be combined with --lora\n", __func__);
            return 1;
        }
        const std::string mode = params.grpo_mode ? "grpo" : "sft";
        if (!load_checkpoint_state(params.lora_resume, mode, resume_state)) return 1;
        if (mode == "grpo" && resume_state.step > INT32_MAX) {
            LOG_ERR("%s: checkpoint step is too large\n", __func__);
            return 1;
        }
        common_adapter_lora_info adapter_info;
        adapter_info.path  = params.lora_resume;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    }

    // Force settings required for training
    params.load_mode    = LLAMA_LOAD_MODE_NONE;
    if (!params.kv_cache_training) {
        params.cache_type_k = GGML_TYPE_F32;
        params.cache_type_v = GGML_TYPE_F32;
    } else {
        LOG_INF("%s: KV cache training target: K=%s V=%s hadamard_k=%d hadamard_v=%d\n",
                __func__, ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
                params.cache_hadamard_k, params.cache_hadamard_v);
    }
    // Warmup runs inference with PARAM-flagged tensors which causes a segfault;
    // training never benefits from warmup, so disable it unconditionally.
    params.warmup       = false;
    // Keep the requested FlashAttention policy. CUDA backward supports the quantized
    // KV formats used by --kv-cache-training, and quantized V requires FA.

    float lora_alpha = (params.lora_alpha > 0.0f)
        ? params.lora_alpha : (float) params.lora_rank;
    if (resume_requested && resume_state.has_alpha) {
        if (params.lora_alpha > 0.0f && std::abs(params.lora_alpha - resume_state.alpha) > 1e-6f) {
            LOG_WRN("%s: ignoring --lora-alpha %.3f; checkpoint alpha is %.3f\n",
                    __func__, (double) params.lora_alpha, (double) resume_state.alpha);
        }
        lora_alpha = resume_state.alpha;
    }
    const auto targets = split_csv(params.lora_targets);

    // --- Step 1: Discover tensor shapes from model GGUF (no model load yet) ---
    std::string arch;
    {
        struct ggml_context * ctx_meta = nullptr;
        struct gguf_init_params gp = { true, &ctx_meta };
        struct gguf_context * ctx_gguf = gguf_init_from_file(params.model.path.c_str(), gp);
        if (!ctx_gguf) { LOG_ERR("failed to open model GGUF\n"); return 1; }
        int kid = gguf_find_key(ctx_gguf, "general.architecture");
        if (kid >= 0) arch = gguf_get_val_str(ctx_gguf, kid);
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
    }

    // --- Step 2: Allocate LoRA tensors and save initial adapter GGUF ---
    // If the user already supplied a --lora adapter we reuse it (resume training).
    // Otherwise we allocate fresh tensors (B=0, A=random), write them to a temp
    // .init.gguf so common_init_from_params can load them before context creation
    // (this makes sched_reserve size the graph to include LoRA nodes).
    const bool resume_from_lora = !params.lora_adapters.empty();

    std::mt19937 rng(42);
    lora_tensors lt; // will be populated after context load (Step 4)
    std::string init_adapter_path;

    if (!resume_from_lora) {
        lt = alloc_lora_tensors(params.model.path, targets, params.lora_rank, rng, params.lora_freeze_layers);
        if (lt.ab.empty()) return 1;

        init_adapter_path = params.lora_out + ".init.gguf";
        save_adapter(lt, init_adapter_path, arch, lora_alpha, params.model.path);

        // Register adapter so common_init_from_params loads it before context creation
        common_adapter_lora_info adapter_info;
        adapter_info.path  = init_adapter_path;
        adapter_info.scale = 1.0f;
        params.lora_adapters.push_back(adapter_info);
    } else {
        LOG_INF("%s: loading existing LoRA adapter: %s\n",
                __func__, params.lora_adapters.back().path.c_str());
    }

    // --- Step 3: Load model + context (graph sized with LoRA nodes) ---
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model) { LOG_ERR("failed to load model\n"); return 1; }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // Arch fallback if not in GGUF metadata
    if (arch.empty()) {
        char buf[256] = {};
        llama_model_desc(model, buf, sizeof(buf));
        arch = std::string(buf);
        arch = arch.substr(0, arch.find_first_of(" /"));
    }

    // --- Step 4: Mark the loaded adapter tensors as trainable ---
    // common_init_from_params loaded the adapter; params.lora_adapters[back].ptr
    // points to the live llama_adapter_lora with its own tensor copies in device
    // memory. Mark those tensors trainable so the optimizer graph includes them.
    {
        llama_adapter_lora * loaded = params.lora_adapters.back().ptr;
        if (!loaded) {
            LOG_ERR("%s: adapter was not loaded by common_init_from_params\n", __func__);
            return 1;
        }
        for (auto & kv : loaded->ab_map) {
            if (params.mtp_mode == "only") {
                kv.second.a->flags &= ~GGML_TENSOR_FLAG_PARAM;
                kv.second.b->flags &= ~GGML_TENSOR_FLAG_PARAM;
            } else {
                ggml_set_param(kv.second.a);
                ggml_set_param(kv.second.b);
            }
        }
        // Point lt.ab at the live device tensors so save_adapter writes
        // the trained weights (not the original init tensors).
        lt.ab.clear();
        for (auto & kv : loaded->ab_map) {
            lt.ab[kv.first] = {kv.second.a, kv.second.b};
        }
    }

    // Remove temp init file when we created it (resume path has no init file)
    if (!resume_from_lora && !init_adapter_path.empty()) {
        std::remove(expand_tilde(init_adapter_path).c_str());
    }

    // --- Step 5: Load dataset ---
    // In GRPO mode the dataset comes from Python via stdin/stdout — skip file loading.
    auto tmpls = common_chat_templates_init(model, params.chat_template);
    if (params.grpo_mode) {
        int rc = run_grpo_mode(params, model, ctx, lt, arch, lora_alpha, params.model.path,
                               resume_requested ? &resume_state : nullptr);
        if (lt.buf) ggml_backend_buffer_free(lt.buf);
        if (lt.ctx) ggml_free(lt.ctx);
        llama_backend_free();
        return rc;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto samples = load_jsonl(params.train_file, vocab, tmpls.get(), params.dataset_threads,
                              params.critical_token_mode, params.critical_token_weight,
                              params.preserve_thinking ? 1 : 0);
    if (samples.empty()) {
        LOG_ERR("%s: no training samples loaded\n", __func__);
        return 1;
    }

    const int32_t n_ctx = llama_n_ctx(ctx);
    std::vector<float> window_rewards;
    const llama_token bos = llama_vocab_bos(llama_model_get_vocab(model));
    const bool critical_enabled = params.critical_token_mode != "none";
    std::vector<llama_opt_critical_token_metadata> critical_metadata;
    auto dataset = build_dataset(samples, n_ctx, window_rewards, params.train_on_prompt, bos, critical_enabled, &critical_metadata);
    if (!dataset) return 1;

    if (critical_enabled) {
        LOG_INF("critical_sft: mode=%s token_weight=%.6f confidence_threshold=%.6f weight_shape=%s warmup_steps=%d max_fraction=%.6f stats_every=%d\n",
                params.critical_token_mode.c_str(), (double) params.critical_token_weight,
                (double) params.critical_confidence_threshold, params.critical_weight_shape.c_str(),
                params.critical_warmup_steps, (double) params.critical_max_fraction, params.critical_stats_every);
    }

    qlora_lr_schedule schedule {
        &params.lr, params.lr_scheduler, params.warmup_steps, params.warmup_init_ratio,
        params.lr_decay_steps, 0, 0
    };

    // Initialize optimizer - our custom param filter restricts training to lora_a/b.
    struct llama_opt_params lopt_params {
        /*.n_ctx_train              =*/0,
        /*.param_filter             =*/params.mtp_mode == "only" ? lora_param_filter_none : lora_param_filter,
        /*.param_filter_ud          =*/nullptr,
        /*.get_opt_pars             =*/qlora_opt_lr_pars,
        /*.get_opt_pars_ud          =*/&schedule,
        /*.optimizer_type           =*/params.optimizer,
        /*.lora_qat_type            =*/lora_qat_type_from_string(params.lora_qat),
        /*.grad_checkpoint_interval =*/params.grad_checkpoint_interval,
        /*.critical_token_mode      =*/critical_token_mode_from_string(params.critical_token_mode),
        /*.critical_token_weight    =*/params.critical_token_weight,
        /*.critical_confidence_threshold =*/params.critical_confidence_threshold,
        /*.critical_weight_shape    =*/critical_weight_shape_from_string(params.critical_weight_shape),
        /*.critical_warmup_steps    =*/params.critical_warmup_steps,
        /*.critical_max_fraction    =*/params.critical_max_fraction,
        /*.critical_step            =*/&schedule.step,
        /*.critical_stats_every     =*/params.critical_stats_every,
        /*.train_target             =*/params.mtp_mode != "only",
    };
    llama_opt_init(ctx, model, lopt_params);

    const int64_t ndata = ggml_opt_dataset_ndata(dataset);
    const int64_t idata_split = train_split_from_val_fraction(ndata, params.val_split);
    if (idata_split <= 0) {
        LOG_ERR("%s: no training windows after val split (ndata=%ld val_split=%.3f)\n",
                __func__, (long) ndata, (double) params.val_split);
        return 1;
    }
    if (params.val_split > 0.0f && idata_split == ndata) {
        LOG_WRN("%s: validation split skipped because dataset has only %ld training window(s)\n",
                __func__, (long) ndata);
    }
    if (params.lr.epochs > 0 && idata_split > INT64_MAX / params.lr.epochs) {
        LOG_ERR("%s: total training step count overflows int64\n", __func__);
        return 1;
    }
    schedule.total_steps = idata_split * params.lr.epochs;
    if (schedule.total_steps <= 0) {
        LOG_ERR("%s: total training step count must be positive\n", __func__);
        return 1;
    }
    if (params.warmup_steps > schedule.total_steps) {
        LOG_ERR("%s: --warmup-steps %d exceeds total training steps %ld\n",
                __func__, params.warmup_steps, (long) schedule.total_steps);
        return 1;
    }
    if (params.lr_scheduler == "cosine" && params.lr_decay_steps > 0 &&
        params.lr_decay_steps <= params.warmup_steps) {
        LOG_ERR("%s: --lr-decay-steps %d must exceed --warmup-steps %d\n",
                __func__, params.lr_decay_steps, params.warmup_steps);
        return 1;
    }
    if (params.lr.lr0 <= 0.0f || (params.lr_scheduler == "cosine" && params.lr.lr_min > params.lr.lr0)) {
        LOG_ERR("%s: learning-rate must be positive and cosine requires learning-rate-min <= learning-rate\n", __func__);
        return 1;
    }
    const auto train_reward_end = window_rewards.begin() + idata_split;
    const bool all_train_rewards_zero = std::all_of(
            window_rewards.begin(), train_reward_end, [](float r) { return std::abs(r) <= 1e-6f; });
    if (all_train_rewards_zero) {
        std::fill(window_rewards.begin(), train_reward_end, 1.0f);
        if (critical_enabled) {
            for (int64_t i = 0; i < idata_split; ++i) {
                for (int32_t j = 0; j < n_ctx; ++j) {
                    llama_opt_critical_token_metadata & metadata = critical_metadata[i * n_ctx + j];
                    if (metadata.span_weight > 0.0f) {
                        metadata.reward_weight = 1.0f;
                    }
                }
            }
            ggml_opt_dataset_set_aux(dataset, critical_metadata.data(), n_ctx * sizeof(critical_metadata[0]));
        }
        LOG_WRN("%s: all training window rewards normalized to zero; using weight 1.0 for the small training split\n",
                __func__);
    }
    critical_metadata.clear();
    critical_metadata.shrink_to_fit();

    const bool has_rewards = std::any_of(window_rewards.begin(), window_rewards.end(),
                                         [](float r){ return std::abs(r - 1.0f) > 1e-4f; });
    if (has_rewards && !critical_enabled) {
        LOG_INF("%s: reward-weighted SFT enabled (found non-uniform rewards in dataset)\n", __func__);
        llama_opt_set_reward_weights(window_rewards.data(), (int64_t)window_rewards.size());
    }

    unsigned epoch_start = 0;
    int64_t window_start = 0;
    if (resume_requested) {
        if ((uint64_t) resume_state.epoch > UINT32_MAX) {
            LOG_ERR("%s: checkpoint epoch is too large\n", __func__);
            return 1;
        }
        epoch_start  = (unsigned) resume_state.epoch - 1;
        window_start = resume_state.window;

        if (resume_state.context_length >= 0 && resume_state.context_length != n_ctx) {
            LOG_ERR("%s: checkpoint context length is %ld, current context length is %d\n",
                    __func__, (long) resume_state.context_length, n_ctx);
            return 1;
        }
        if (resume_state.dataset_windows >= 0 && resume_state.dataset_windows != idata_split) {
            LOG_ERR("%s: checkpoint has %ld training windows, current dataset has %ld\n",
                    __func__, (long) resume_state.dataset_windows, (long) idata_split);
            return 1;
        }
        if (window_start > idata_split) {
            LOG_ERR("%s: checkpoint window %ld exceeds the %ld training windows in this dataset\n",
                    __func__, (long) window_start, (long) idata_split);
            return 1;
        }
        if (resume_state.has_metadata && resume_state.shuffle != params.shuffle_dataset) {
            LOG_ERR("%s: checkpoint shuffle setting does not match --shuffle-dataset\n", __func__);
            return 1;
        }
        if (window_start == idata_split) {
            ++epoch_start;
            window_start = 0;
        }
        LOG_INF("%s: resuming SFT at epoch %u/%u after window %ld; next window is %ld/%ld; optimizer state starts fresh\n",
                __func__, epoch_start + 1, params.lr.epochs, (long) window_start,
                (long) window_start + 1, (long) idata_split);
    }
    if (epoch_start > params.lr.epochs) {
        LOG_ERR("%s: checkpoint epoch %u exceeds target epochs %u\n",
                __func__, epoch_start, params.lr.epochs);
        return 1;
    }
    const int64_t schedule_step = (int64_t) epoch_start * idata_split + window_start;
    if (schedule_step > schedule.total_steps) {
        LOG_ERR("%s: checkpoint step %ld exceeds total training steps %ld\n",
                __func__, (long) schedule_step, (long) schedule.total_steps);
        return 1;
    }
    if (resume_requested && resume_state.schedule_step >= 0 && resume_state.schedule_step != schedule_step) {
        LOG_ERR("%s: checkpoint scheduler step %ld does not match derived step %ld\n",
                __func__, (long) resume_state.schedule_step, (long) schedule_step);
        return 1;
    }
    schedule.step = schedule_step;

    mtp_training_state mtp;
    if (!mtp_init_training(params, ctx, schedule, mtp)) {
        return 1;
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    const int32_t n_ubatch       = llama_n_ubatch(ctx);
    const int32_t ubatch_per_ctx = (n_ubatch > 0) ? (n_ctx / n_ubatch) : 1;

    save_ctx sctx {
        ctx, &lt, &params.lora_out, &arch, &params.model.path, lora_alpha,
        params.mtp_mode != "only",
        mtp.ctx.get(), mtp.ctx ? &mtp.tensors : nullptr,
        &params.mtp_lora_out, mtp.ctx ? &mtp.arch : nullptr, &params.mtp_model,
        mtp.alpha,
        params.save_every, ubatch_per_ctx, 0, 0, 0, idata_split, n_ctx,
        params.shuffle_dataset, params.verbose_loss, &schedule
    };
    g_save_ctx = &sctx;

    if (resume_requested && params.shuffle_dataset && epoch_start > 0) {
        for (unsigned epoch = 0; epoch < epoch_start; ++epoch) {
            llama_opt_dataset_shuffle(ctx, dataset, idata_split);
            if (params.optimizer_restart_every > 0 &&
                epoch + 1 < params.lr.epochs &&
                (epoch + 1) % params.optimizer_restart_every == 0) {
                llama_opt_reset(ctx, true);
            }
        }
        LOG_INF("%s: replayed %u completed epoch shuffle(s)\n", __func__, epoch_start);
    }

    const int64_t total_windows = ggml_opt_dataset_ndata(dataset);
    LOG_INF("%s: starting QLoRA training — rank=%d alpha=%.1f epochs=%d loss=%s\n",
            __func__, params.lora_rank, lora_alpha, params.lr.epochs,
            params.train_on_prompt ? "prompt+response" : "response-only");
    llama_synchronize(ctx);
    const adapter_stats adapter_start = adapter_get_stats(lt);
    LOG_INF("%s: adapter_start: total_l2=%.9f a_l2=%.9f b_l2=%.9f a_max=%.9f b_max=%.9f\n",
            __func__, adapter_start.total_l2, adapter_start.a_l2, adapter_start.b_l2,
            adapter_start.a_max_abs, adapter_start.b_max_abs);
    LOG_INF("%s: dataset: %ld windows × %d ubatches = %ld steps per epoch  (n_ctx=%d n_ubatch=%d stride=%d)\n",
            __func__, (long)total_windows, ubatch_per_ctx, (long)(idata_split * ubatch_per_ctx),
            n_ctx, n_ubatch, n_ctx / 2);
    LOG_INF("%s: lr scheduler=%s warmup_steps=%d warmup_init_ratio=%.3g decay_steps=%ld total_steps=%ld start_step=%ld lr=%.3g lr_min=%.3g\n",
            __func__, params.lr_scheduler.c_str(), params.warmup_steps, (double) params.warmup_init_ratio,
            (long) (params.lr_decay_steps > 0 ? std::min<int64_t>(params.lr_decay_steps, schedule.total_steps) : schedule.total_steps),
            (long) schedule.total_steps,
            (long) schedule.step, (double) params.lr.lr0, (double) std::max(0.0f, params.lr.lr_min));
    if (params.save_every > 0) {
        LOG_INF("%s: will save checkpoint every %d windows: target=%s MTP=%s\n",
                __func__, params.save_every,
                params.mtp_mode != "only" ? params.lora_out.c_str() : "disabled",
                mtp.ctx ? params.mtp_lora_out.c_str() : "disabled");
    }

    ggml_opt_epoch_callback cb_train = save_every_callback;

    for (params.lr.epoch = epoch_start; params.lr.epoch < params.lr.epochs; ++params.lr.epoch) {
        const int64_t idata_start = params.lr.epoch == epoch_start ? window_start : 0;
        sctx.window_offset = idata_start;
        sctx.last_saved = idata_start;
        sctx.epoch = params.lr.epoch + 1;
        sctx.loss_cumulative_sum = 0.0;
        sctx.loss_cumulative_count = 0;
        sctx.loss_epoch_sum = 0.0;
        sctx.loss_epoch_count = 0;
        llama_opt_epoch_range(ctx, dataset, result_train, result_eval, idata_start, idata_split,
                              cb_train,
                              ggml_opt_epoch_callback_progress_bar,
                              params.shuffle_dataset);
        fprintf(stderr, "\n");

        // Per-epoch loss summary
        {
            double train_loss = 0.0, train_unc = 0.0;
            ggml_opt_result_loss(result_train, &train_loss, &train_unc);
            if (idata_split < ggml_opt_dataset_ndata(dataset)) {
                double val_loss = 0.0, val_unc = 0.0;
                ggml_opt_result_loss(result_eval, &val_loss, &val_unc);
                LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f  val_loss=%.4f ± %.4f\n",
                        params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc, val_loss, val_unc);
            } else {
                LOG_INF("epoch %d/%d: train_loss=%.4f ± %.4f\n",
                        params.lr.epoch + 1, params.lr.epochs, train_loss, train_unc);
            }
            llama_synchronize(ctx);
            const adapter_stats adapter_cur = adapter_get_stats(lt);
            LOG_INF("epoch %d/%d: adapter: total_l2=%.9f delta=%.9f a_l2=%.9f b_l2=%.9f a_max=%.9f b_max=%.9f\n",
                    params.lr.epoch + 1, params.lr.epochs,
                    adapter_cur.total_l2, adapter_cur.total_l2 - adapter_start.total_l2,
                    adapter_cur.a_l2, adapter_cur.b_l2,
                    adapter_cur.a_max_abs, adapter_cur.b_max_abs);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);

        if (params.optimizer_restart_every > 0 &&
            params.lr.epoch + 1 < params.lr.epochs &&
            (params.lr.epoch + 1) % params.optimizer_restart_every == 0) {
            LOG_INF("%s: resetting optimizer state after epoch %d\n", __func__, params.lr.epoch + 1);
            llama_opt_reset(ctx, true);
        }
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);
    llama_opt_set_reward_weights(nullptr, 0);

    // Save final trained adapter
    bool final_save_ok = true;
    if (params.mtp_mode != "only") {
        final_save_ok = save_adapter(lt, params.lora_out, arch, lora_alpha, params.model.path);
    }
    if (mtp.ctx) {
        llama_synchronize(mtp.ctx.get());
        final_save_ok = save_adapter(mtp.tensors, params.mtp_lora_out, mtp.arch, mtp.alpha,
            params.mtp_model) && final_save_ok;
    }

    // Free scratch buffers only when we allocated them (not in resume path)
    if (lt.buf) ggml_backend_buffer_free(lt.buf);
    if (lt.ctx) ggml_free(lt.ctx);
    if (mtp.tensors.buf) ggml_backend_buffer_free(mtp.tensors.buf);
    if (mtp.tensors.ctx) ggml_free(mtp.tensors.ctx);
    ggml_opt_dataset_free(dataset);
    llama_backend_free();

    return final_save_ok ? 0 : 1;
}
#endif
