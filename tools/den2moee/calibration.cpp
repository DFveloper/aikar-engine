#include "calibration.h"

#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "scoring.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace den2moee {

namespace {

struct ggml_tensor_bytes {
    std::vector<uint8_t> data;
    size_t bytes = 0;
};

struct callback_state {
    enum class Phase { TokenScore, NeuronScore };

    Phase phase = Phase::TokenScore;
    const CalibrationSample * sample = nullptr;
    const std::vector<float> * token_score = nullptr;
    int hidden_size = 0;
    int final_layer = 0;
    std::vector<int> intermediate_sizes;
    std::vector<DomainScoreAccumulator> * accumulators = nullptr;
    std::vector<std::vector<float>> * mean_input_sums = nullptr;
    std::vector<uint64_t> * mean_input_counts = nullptr;
    ScoreActivation score_activation = ScoreActivation::Gemma4Gelu;
    std::vector<std::vector<float>> pending_up;
    std::vector<std::vector<float>> pending_gate;
    std::vector<bool> pending_up_seen;
    std::vector<bool> pending_gate_seen;
    std::vector<size_t> pending_offsets;
    size_t scratch_live = 0;
    size_t * peak_scratch = nullptr;
    size_t expected_tokens = 0;
    size_t neuron_tokens_seen = 0;
    int neuron_chunk_tokens = 0;
    size_t neuron_chunk_offset = 0;
    std::vector<float> final_hidden;
    std::vector<float> attention;
};

bool has_layer_name(const char * name, const char * prefix, int & layer) {
    const size_t prefix_len = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_len) != 0) return false;
    return std::sscanf(name + prefix_len, "%d", &layer) == 1;
}

float tensor_value(const std::vector<uint8_t> & data, ggml_type type, size_t index) {
    switch (type) {
        case GGML_TYPE_F32: return reinterpret_cast<const float *>(data.data())[index];
        case GGML_TYPE_F16: return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(data.data())[index]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(reinterpret_cast<const ggml_bf16_t *>(data.data())[index]);
        default: throw std::runtime_error("calibration tensor is not floating point");
    }
}

ggml_tensor_bytes get_tensor_bytes(ggml_tensor * tensor, size_t * peak_scratch) {
    ggml_tensor_bytes result;
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    result.bytes = row_bytes * static_cast<size_t>(tensor->ne[1]) *
                   static_cast<size_t>(tensor->ne[2]) * static_cast<size_t>(tensor->ne[3]);
    result.data.resize(result.bytes);
    const size_t rows = static_cast<size_t>(tensor->ne[1]);
    const size_t planes = static_cast<size_t>(tensor->ne[2]) * static_cast<size_t>(tensor->ne[3]);
    for (size_t plane = 0; plane < planes; ++plane) {
        const size_t d2 = plane % static_cast<size_t>(tensor->ne[2]);
        const size_t d3 = plane / static_cast<size_t>(tensor->ne[2]);
        const size_t offset = d2 * tensor->nb[2] + d3 * tensor->nb[3];
        ggml_backend_tensor_get_2d(tensor, result.data.data() + plane * rows * row_bytes,
                                   offset, row_bytes, rows, tensor->nb[1], row_bytes);
    }
    if (peak_scratch != nullptr) *peak_scratch = std::max(*peak_scratch, result.bytes);
    return result;
}

std::vector<float> tensor_values(ggml_tensor * tensor, size_t * peak_scratch) {
    const ggml_tensor_bytes bytes = get_tensor_bytes(tensor, peak_scratch);
    const size_t count = ggml_nelements(tensor);
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = tensor_value(bytes.data, tensor->type, i);
    return result;
}

std::vector<float> token_major(const std::vector<float> & column_major, int tokens, int neurons) {
    if (column_major.size() != static_cast<size_t>(tokens) * neurons) {
        throw std::runtime_error("calibration activation shape mismatch");
    }
    std::vector<float> result(column_major.size());
    for (int token = 0; token < tokens; ++token) {
        for (int neuron = 0; neuron < neurons; ++neuron) {
            result[token * neurons + neuron] = column_major[neuron + token * neurons];
        }
    }
    return result;
}

void add_attention(callback_state & state, ggml_tensor * tensor) {
    const std::vector<float> values = tensor_values(tensor, state.peak_scratch);
    if (tensor->ne[1] <= 0 || tensor->ne[0] <= 0 || tensor->ne[2] <= 0) {
        throw std::runtime_error("invalid attention tensor shape");
    }
    const int keys = static_cast<int>(tensor->ne[0]);
    const int queries = static_cast<int>(tensor->ne[1]);
    const int heads = static_cast<int>(tensor->ne[2]);
    const int query = queries - 1;
    state.attention.assign(std::min(keys, static_cast<int>(state.sample->tokens.size())), 0.0f);
    for (int head = 0; head < heads; ++head) {
        for (int key = 0; key < static_cast<int>(state.attention.size()); ++key) {
            const size_t index = key + static_cast<size_t>(query) * keys + static_cast<size_t>(head) * keys * queries;
            if (index < values.size()) state.attention[key] += values[index];
        }
    }
}

void add_final_hidden(callback_state & state, ggml_tensor * tensor) {
    const std::vector<float> values = tensor_values(tensor, state.peak_scratch);
    const int tokens = static_cast<int>(tensor->ne[1]);
    if (tensor->ne[0] != state.hidden_size || tokens <= 0 ||
        state.final_hidden.size() + static_cast<size_t>(tokens) * state.hidden_size >
            state.expected_tokens * static_cast<size_t>(state.hidden_size)) {
        throw std::runtime_error("final hidden shape does not match calibration tokens: tensor=" +
                                 std::to_string(tensor->ne[0]) + "x" + std::to_string(tensor->ne[1]) + "x" +
                                 std::to_string(tensor->ne[2]) + "x" + std::to_string(tensor->ne[3]) +
                                 ", expected_hidden=" + std::to_string(state.hidden_size) +
                                 ", expected_tokens=" + std::to_string(state.expected_tokens) +
                                 ", accumulated_tokens=" + std::to_string(state.final_hidden.size() /
                                                                               static_cast<size_t>(state.hidden_size)));
    }
    const std::vector<float> chunk = token_major(values, tokens, state.hidden_size);
    state.final_hidden.insert(state.final_hidden.end(), chunk.begin(), chunk.end());
}

void add_mean_input(callback_state & state, ggml_tensor * tensor, int layer) {
    const std::vector<float> values = tensor_values(tensor, state.peak_scratch);
    const int tokens = static_cast<int>(tensor->ne[1]);
    if (tensor->ne[0] != state.hidden_size || tokens <= 0) {
        throw std::runtime_error("FFN input shape does not match calibration tokens: tensor=" +
                                 std::to_string(tensor->ne[0]) + "x" + std::to_string(tensor->ne[1]) + "x" +
                                 std::to_string(tensor->ne[2]) + "x" + std::to_string(tensor->ne[3]) +
                                 ", expected_hidden=" + std::to_string(state.hidden_size) +
                                 ", expected_tokens=" + std::to_string(state.expected_tokens));
    }
    auto & sum = (*state.mean_input_sums)[layer];
    for (int token = 0; token < tokens; ++token) {
        for (int hidden = 0; hidden < state.hidden_size; ++hidden) {
            sum[hidden] += values[hidden + token * state.hidden_size];
        }
    }
    (*state.mean_input_counts)[layer] += tokens;
}

void begin_neuron_chunk(callback_state & state, int tokens) {
    if (tokens <= 0 || state.neuron_tokens_seen + static_cast<size_t>(tokens) > state.expected_tokens) {
        throw std::runtime_error("invalid calibration activation chunk length");
    }
    if (state.neuron_chunk_tokens == 0) {
        state.neuron_chunk_tokens = tokens;
        state.neuron_chunk_offset = state.neuron_tokens_seen;
    } else if (state.neuron_chunk_tokens != tokens) {
        throw std::runtime_error("calibration activation chunk lengths do not match");
    }
}

void add_neuron_score(callback_state & state, int layer) {
    const std::vector<float> & gate = state.pending_gate[layer];
    const std::vector<float> & up = state.pending_up[layer];
    const int tokens = static_cast<int>(gate.size() / state.intermediate_sizes[layer]);
    const size_t offset = state.pending_offsets[layer];
    if (offset + static_cast<size_t>(tokens) > state.token_score->size()) {
        throw std::runtime_error("calibration activation chunk exceeds token score length");
    }
    const std::vector<float> token_scores(state.token_score->begin() + offset,
                                           state.token_score->begin() + offset + tokens);
    const std::vector<float> score = weighted_activation_score(
        gate, up, tokens, state.intermediate_sizes[layer], token_scores, state.score_activation);
    (*state.accumulators)[layer].add(state.sample->domain, score);
    state.pending_up[layer].clear();
    state.pending_gate[layer].clear();
    state.pending_up_seen[layer] = false;
    state.pending_gate_seen[layer] = false;
    if (layer == static_cast<int>(state.intermediate_sizes.size()) - 1) {
        state.neuron_tokens_seen += static_cast<size_t>(tokens);
        state.neuron_chunk_tokens = 0;
    }
}

bool calibration_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<callback_state *>(user_data);
    int layer = -1;
    const bool is_up = has_layer_name(tensor->name, "ffn_up-", layer);
    const bool is_gate = has_layer_name(tensor->name, "ffn_gate-", layer);
    const bool is_norm = has_layer_name(tensor->name, "ffn_norm-", layer);
    const bool is_attention = has_layer_name(tensor->name, "kq_soft_max-", layer);
    const bool is_final_hidden = std::strcmp(tensor->name, "result_norm") == 0;

    if (ask) {
        if (state.phase == callback_state::Phase::TokenScore) {
            return is_final_hidden || (is_attention && layer == state.final_layer);
        }
        return is_up || is_gate || is_norm;
    }

    if (state.phase == callback_state::Phase::TokenScore) {
        if (is_attention && layer == state.final_layer) add_attention(state, tensor);
        if (is_final_hidden) add_final_hidden(state, tensor);
        return true;
    }

    if (is_up || is_gate || is_norm) begin_neuron_chunk(state, static_cast<int>(tensor->ne[1]));
    if (is_norm) add_mean_input(state, tensor, layer);
    if (is_up) {
        state.pending_up[layer] = token_major(tensor_values(tensor, state.peak_scratch),
                                              static_cast<int>(tensor->ne[1]), state.intermediate_sizes[layer]);
        state.pending_up_seen[layer] = true;
        state.pending_offsets[layer] = state.neuron_chunk_offset;
        if (state.pending_gate_seen[layer]) add_neuron_score(state, layer);
    }
    if (is_gate) {
        state.pending_gate[layer] = token_major(tensor_values(tensor, state.peak_scratch),
                                                static_cast<int>(tensor->ne[1]), state.intermediate_sizes[layer]);
        state.pending_gate_seen[layer] = true;
        state.pending_offsets[layer] = state.neuron_chunk_offset;
        if (state.pending_up_seen[layer]) add_neuron_score(state, layer);
    }
    return true;
}

void run_forward(llama_context * context, callback_state & state, const std::vector<int32_t> & tokens) {
    if (tokens.empty()) throw std::runtime_error("calibration sample tokenization is empty");
    state.expected_tokens = tokens.size();
    state.neuron_tokens_seen = 0;
    state.neuron_chunk_tokens = 0;
    state.neuron_chunk_offset = 0;
    llama_memory_clear(llama_get_memory(context), true);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    const int result = llama_decode(context, batch);
    llama_batch_free(batch);
    if (result != 0) {
        throw std::runtime_error("calibration forward failed");
    }
    if (state.phase == callback_state::Phase::TokenScore) {
        if (state.final_hidden.size() != tokens.size() * static_cast<size_t>(state.hidden_size)) {
            throw std::runtime_error("calibration final hidden chunks do not cover all tokens");
        }
        if (state.attention.size() != tokens.size()) {
            throw std::runtime_error("calibration attention shape does not cover all tokens: got=" +
                                     std::to_string(state.attention.size()) +
                                     ", expected=" + std::to_string(tokens.size()));
        }
    } else if (state.neuron_tokens_seen != tokens.size() || state.neuron_chunk_tokens != 0) {
        throw std::runtime_error("calibration FFN chunks do not cover all tokens: got=" +
                                 std::to_string(state.neuron_tokens_seen) +
                                 ", expected=" + std::to_string(tokens.size()));
    }
}

std::string string_field(const common_json & item, const char * key) {
    if (!item.contains(key) || !item.at(key).is_string()) return {};
    return item.at(key).get<std::string>();
}

bool parse_messages_field(const common_json & item, const char * key, std::vector<common_chat_msg> & messages) {
    if (!item.contains(key)) return false;
    if (!item.at(key).is_array()) {
        throw std::runtime_error(std::string("calibration field '") + key + "' must be an array");
    }
    messages = common_chat_msgs_parse_oaicompat(item.at(key));
    return true;
}

std::string normalize_conversation_role(std::string role) {
    if (role == "human" || role == "user") return "user";
    if (role == "gpt" || role == "assistant" || role == "bot" || role == "model") return "assistant";
    if (role == "system" || role == "developer" || role == "tool") return role;
    return {};
}

bool parse_conversations(const common_json & item, std::vector<common_chat_msg> & messages) {
    if (!item.contains("conversations")) return false;
    const common_json & conversations = item.at("conversations");
    if (!conversations.is_array()) throw std::runtime_error("calibration field 'conversations' must be an array");

    common_json converted = common_json::array();
    for (const common_json & conversation : conversations) {
        if (!conversation.is_object()) continue;
        std::string role = normalize_conversation_role(string_field(conversation, "role"));
        if (role.empty()) role = normalize_conversation_role(string_field(conversation, "from"));
        std::string content = string_field(conversation, "content");
        if (content.empty()) content = string_field(conversation, "value");
        if (role.empty() || content.empty()) continue;
        common_json message = common_json::object();
        message["role"] = role;
        message["content"] = content;
        converted.push_back(message);
    }
    if (converted.empty()) throw std::runtime_error("calibration conversations contain no usable messages");
    messages = common_chat_msgs_parse_oaicompat(converted);
    return true;
}

bool parse_instruction_record(const common_json & item, std::vector<common_chat_msg> & messages) {
    std::string user = string_field(item, "instruction");
    if (user.empty()) user = string_field(item, "prompt");
    if (user.empty()) user = string_field(item, "question");
    if (user.empty()) return false;

    const std::string input = string_field(item, "input");
    if (!input.empty()) user += '\n' + input;

    common_json converted = common_json::array();
    const std::string system = string_field(item, "system").empty()
        ? string_field(item, "system_prompt") : string_field(item, "system");
    if (!system.empty()) {
        common_json message = common_json::object();
        message["role"] = "system";
        message["content"] = system;
        converted.push_back(message);
    }
    common_json user_message = common_json::object();
    user_message["role"] = "user";
    user_message["content"] = user;
    converted.push_back(user_message);

    std::string assistant;
    for (const char * key : { "output", "response", "answer", "completion", "assistant" }) {
        assistant = string_field(item, key);
        if (!assistant.empty()) break;
    }
    if (!assistant.empty()) {
        common_json assistant_message = common_json::object();
        assistant_message["role"] = "assistant";
        assistant_message["content"] = assistant;
        converted.push_back(assistant_message);
    }
    messages = common_chat_msgs_parse_oaicompat(converted);
    return true;
}

size_t device_used_bytes(const std::vector<ggml_backend_dev_t> & selected_devices) {
    size_t used = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (!selected_devices.empty() &&
            std::find(selected_devices.begin(), selected_devices.end(), device) == selected_devices.end()) {
            continue;
        }
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(device, &free, &total);
        if (total > 0 && free <= total) used = std::max(used, total - free);
    }
    return used;
}

} // namespace

CalibrationRecord parse_calibration_record(const common_json & item, CalibrationInputFormat format) {
    if (!item.is_object()) throw std::runtime_error("calibration line is not an object");

    CalibrationRecord record;
    record.domain = item.value("domain", "default");

    if (format == CalibrationInputFormat::Raw) {
        record.text = string_field(item, "text");
        if (record.text.empty()) throw std::runtime_error("raw calibration line has no non-empty text field");
        return record;
    }

    bool found = false;
    if (item.contains("messages")) {
        found = parse_messages_field(item, "messages", record.messages);
    } else if (item.contains("positive_messages")) {
        found = parse_messages_field(item, "positive_messages", record.messages);
    }
    if (!found) found = parse_conversations(item, record.messages);
    if (!found) found = parse_instruction_record(item, record.messages);

    if (!found && format == CalibrationInputFormat::Auto) {
        record.text = string_field(item, "text");
        if (record.text.empty()) record.text = string_field(item, "content");
        if (!record.text.empty()) return record;
    }
    if (!found || record.messages.empty()) {
        throw std::runtime_error("calibration line has no supported chat/instruct fields");
    }
    return record;
}

std::vector<CalibrationSample> load_calibration_jsonl(const std::string & path,
                                                       const llama_model * model,
                                                       int max_seq_len,
                                                       size_t max_samples,
                                                       CalibrationInputFormat format,
                                                       const std::string & chat_template) {
    if (path.empty()) throw std::invalid_argument("calibration path is empty");
    std::ifstream input(path);
    if (!input) throw std::runtime_error("failed to open calibration JSONL: " + path);
    std::vector<CalibrationSample> result;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_chat_templates_ptr chat_templates;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const CalibrationRecord record = parse_calibration_record(common_json::parse(line), format);
        CalibrationSample sample;
        sample.domain = record.domain;
        if (!record.messages.empty()) {
            if (!chat_templates) {
                chat_templates = common_chat_templates_init(model, chat_template);
                if (!chat_templates) throw std::runtime_error("failed to initialize model chat template");
            }
            common_chat_templates_inputs inputs;
            inputs.messages = record.messages;
            inputs.add_generation_prompt = record.messages.back().role != "assistant";
            inputs.enable_thinking = true;
            sample.text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            if (sample.text.empty()) throw std::runtime_error("chat template rendered an empty calibration prompt");
        } else {
            sample.text = record.text;
        }
        std::vector<llama_token> tokens = common_tokenize(vocab, sample.text, true, true);
        if (tokens.empty()) throw std::runtime_error("calibration text tokenized to zero tokens");
        if (static_cast<int>(tokens.size()) > max_seq_len) tokens.resize(max_seq_len);
        sample.tokens.assign(tokens.begin(), tokens.end());
        result.push_back(std::move(sample));
        if (max_samples > 0 && result.size() >= max_samples) break;
    }
    if (result.empty()) throw std::runtime_error("calibration JSONL contains no samples");
    return result;
}

CalibrationResult run_streaming_calibration(const std::string & model_path,
                                             const std::vector<CalibrationSample> & samples,
                                             const std::vector<int> & intermediate_sizes,
                                             const Options & options) {
    if (samples.empty() || intermediate_sizes.empty()) throw std::invalid_argument("calibration inputs are empty");
    common_params params;
    params.model.path = model_path;
    params.n_ctx = options.max_seq_len;
    params.n_batch = options.max_seq_len;
    params.n_ubatch = std::min(512, options.max_seq_len);
    params.n_gpu_layers = options.gpu_layers;
    const std::vector<ggml_backend_dev_t> devices = parse_device_spec(options.device);
    params.devices = devices;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;
    params.cpuparams.n_threads = options.n_threads;
    params.cpuparams_batch.n_threads = options.n_threads;
    std::fprintf(stderr, "[den2moee] devices: %s\n", device_spec_name(devices).c_str());

    std::vector<std::string> domains;
    for (const CalibrationSample & sample : samples) domains.push_back(sample.domain);
    std::sort(domains.begin(), domains.end());
    domains.erase(std::unique(domains.begin(), domains.end()), domains.end());

    callback_state state;
    state.intermediate_sizes = intermediate_sizes;
    state.final_layer = static_cast<int>(intermediate_sizes.size()) - 1;
    params.cb_eval = calibration_callback;
    params.cb_eval_user_data = &state;
    common_init_result_ptr init = common_init_from_params(params);
    if (!init || init->model() == nullptr || init->context() == nullptr) {
        throw std::runtime_error("failed to initialize calibration model");
    }
    state.hidden_size = llama_model_n_embd(init->model());
    size_t peak_scratch = 0;
    state.peak_scratch = &peak_scratch;
    state.score_activation = options.score_activation;

    CalibrationResult result;
    result.hidden_size = state.hidden_size;
    result.intermediate_sizes = intermediate_sizes;
    result.domains = domains;
    result.peak_vram = device_used_bytes(devices);

    state.sample = nullptr;
    for (const CalibrationSample & sample : samples) {
        state.phase = callback_state::Phase::TokenScore;
        state.sample = &sample;
        state.attention.clear();
        state.final_hidden.clear();
        run_forward(init->context(), state, sample.tokens);
        if (state.attention.empty() || state.final_hidden.empty()) {
            throw std::runtime_error("calibration callback did not capture attention or final hidden state");
        }
        const std::vector<float> original_attention = state.attention;
        const std::vector<float> original_final_hidden = state.final_hidden;
        std::vector<RssPerturbation> perturbations;
        if (options.token_score == TokenScoreMode::ScsRss) {
            const std::vector<float> scs = calculate_scs(state.attention, options.span_size);
            const std::vector<int> selected = select_top_spans(scs);
            for (int start = 0; start + options.rss_ngram <= static_cast<int>(sample.tokens.size()); start += options.rss_stride) {
                bool selected_start = false;
                for (int span : selected) {
                    if (start >= span * options.span_size && start < (span + 1) * options.span_size) {
                        selected_start = true;
                        break;
                    }
                }
                if (!selected_start) continue;
                std::vector<int32_t> perturbed = sample.tokens;
                std::fill(perturbed.begin() + start, perturbed.begin() + start + options.rss_ngram,
                          options.perturb_token_id);
                state.phase = callback_state::Phase::TokenScore;
                state.attention.clear();
                state.final_hidden.clear();
                run_forward(init->context(), state, perturbed);
                perturbations.push_back({ start, state.final_hidden });
                ++result.rss_forward_count;
                result.peak_vram = std::max(result.peak_vram, device_used_bytes(devices));
            }
        }
        const TokenScoreResult token_scores = calculate_token_scores(
            original_attention, original_final_hidden, state.hidden_size, perturbations,
            options.span_size, options.rss_ngram, options.token_score);

        if (result.layer_features.empty()) {
            result.layer_features.resize(intermediate_sizes.size());
            result.mean_inputs.resize(intermediate_sizes.size());
            state.accumulators = new std::vector<DomainScoreAccumulator>();
            state.accumulators->reserve(intermediate_sizes.size());
            state.mean_input_sums = new std::vector<std::vector<float>>(intermediate_sizes.size(),
                                                                         std::vector<float>(state.hidden_size, 0.0f));
            state.mean_input_counts = new std::vector<uint64_t>(intermediate_sizes.size(), 0);
            for (int neurons : intermediate_sizes) state.accumulators->emplace_back(domains, neurons);
            state.pending_up.resize(intermediate_sizes.size());
            state.pending_gate.resize(intermediate_sizes.size());
            state.pending_up_seen.assign(intermediate_sizes.size(), false);
            state.pending_gate_seen.assign(intermediate_sizes.size(), false);
            state.pending_offsets.assign(intermediate_sizes.size(), 0);
        }

        state.phase = callback_state::Phase::NeuronScore;
        state.token_score = &token_scores.token_score;
        for (size_t layer = 0; layer < intermediate_sizes.size(); ++layer) {
            state.pending_up[layer].clear();
            state.pending_gate[layer].clear();
            state.pending_up_seen[layer] = false;
            state.pending_gate_seen[layer] = false;
        }
        run_forward(init->context(), state, sample.tokens);
        ++result.forward_count;
        result.peak_vram = std::max(result.peak_vram, device_used_bytes(devices));
        std::fprintf(stderr, "[den2moee] calibration sample %zu/%zu domain=%s tokens=%zu rss_forwards=%zu\n",
                     result.forward_count, samples.size(), sample.domain.c_str(), sample.tokens.size(),
                     result.rss_forward_count);
    }

    for (size_t layer = 0; layer < intermediate_sizes.size(); ++layer) {
        result.layer_features[layer] = (*state.accumulators)[layer].normalized_features();
        result.mean_inputs[layer] = (*state.mean_input_sums)[layer];
        if ((*state.mean_input_counts)[layer] > 0) {
            for (float & value : result.mean_inputs[layer]) {
                value /= static_cast<float>((*state.mean_input_counts)[layer]);
            }
        }
    }
    delete state.accumulators;
    delete state.mean_input_sums;
    delete state.mean_input_counts;
    result.peak_cpu_scratch = peak_scratch;
    return result;
}

} // namespace den2moee
