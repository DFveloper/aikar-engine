#define LLAMA_MERGE_NO_MAIN
#include "../tools/model-merge/model-merge.cpp"

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void write_test_model(
        const std::string & path,
        const std::string & token,
        int32_t table_offset = 0,
        const std::string & chat_template = "{{ messages }}") {
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_u32(gguf, "general.file_type", LLAMA_FTYPE_ALL_F32);
    gguf_set_val_u32(gguf, "llama.block_count", 1);
    gguf_set_val_u32(gguf, "llama.embedding_length", 256);
    gguf_set_val_u32(gguf, "llama.feed_forward_length", 256);
    gguf_set_val_u32(gguf, "llama.attention.head_count", 2);
    gguf_set_val_u32(gguf, "llama.attention.head_count_kv", 2);
    gguf_set_val_u32(gguf, "llama.attention.key_length", 128);
    gguf_set_val_u32(gguf, "llama.attention.value_length", 128);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "test");
    const char * tokens[] = { token.c_str() };
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, 1);
    gguf_set_val_str(gguf, "tokenizer.chat_template", chat_template.c_str());

    ggml_init_params params = {
        /*.mem_size   = */ 2 * 1024 * 1024,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "failed to create test tensor context");
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 256);
    ggml_set_name(weight, "blk.0.attn_q.weight");
    for (int64_t i = 0; i < ggml_nelements(weight); ++i) ((float *) weight->data)[i] = (float) (i % 31) / 31.0f;
    gguf_add_tensor(gguf, weight);

    ggml_tensor * norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    ggml_set_name(norm, "output_norm.weight");
    for (int64_t i = 0; i < ggml_nelements(norm); ++i) ((float *) norm->data)[i] = 1.0f;
    gguf_add_tensor(gguf, norm);

    ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 4);
    ggml_set_name(table, "blk.0.ffn_gate_tid2eid.weight");
    for (int64_t i = 0; i < ggml_nelements(table); ++i) ((int32_t *) table->data)[i] = i + table_offset;
    gguf_add_tensor(gguf, table);

    check(gguf_write_to_file(gguf, path.c_str(), false), "failed to write test GGUF");
    ggml_free(ctx);
    gguf_free(gguf);
}

int main() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
        ("llama-merge-test-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directories(dir);
    const std::string model_a = (dir / "a.gguf").string();
    const std::string model_b = (dir / "b.gguf").string();
    const std::string bad_model = (dir / "bad.gguf").string();
    const std::string bad_data_model = (dir / "bad-data.gguf").string();
    const std::string other_template_model = (dir / "other-template.gguf").string();
    const std::string bad_output = (dir / "bad-output.gguf").string();
    const std::string output = (dir / "output.gguf").string();
    write_test_model(model_a, "same");
    write_test_model(model_b, "same");
    write_test_model(bad_model, "different");
    write_test_model(bad_data_model, "same", 1);
    write_test_model(other_template_model, "same", 0, "{{ bos_token }}{{ messages }}");

    gguf_input a(model_a);
    gguf_input b(model_b);
    gguf_input bad(bad_model);
    validate_metadata_compatibility(a, b);
    bool rejected = false;
    try {
        validate_metadata_compatibility(a, bad);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "tokenizer mismatch was not rejected");

    gguf_input other_template(other_template_model);
    rejected = false;
    try {
        validate_metadata_compatibility(a, other_template);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "chat template mismatch was not rejected by default");
    validate_metadata_compatibility(a, other_template, true);

    std::vector<uint8_t> disk_tensor;
    std::vector<uint8_t> cached_tensor;
    b.read_tensor("blk.0.attn_q.weight", disk_tensor);
    b.load_into_memory(1, 1);
    b.read_tensor("blk.0.attn_q.weight", cached_tensor);
    check(disk_tensor == cached_tensor, "RAM-cached tensor data changed");

    nlohmann::json response_message = {
        { "role", "assistant" },
        { "content", "answer" },
        { "reasoning", "think" },
    };
    check(calibration_message_payload(response_message) == "think\nanswer", "template-free assistant payload is wrong");

    llama_quant_model_desc desc = {
        /*.architecture  = */ "llama",
        /*.n_embd        = */ 256,
        /*.n_ff          = */ 256,
        /*.n_layer       = */ 1,
        /*.n_head        = */ 2,
        /*.n_head_kv     = */ 2,
        /*.n_expert      = */ 0,
        /*.n_embd_head_k = */ 128,
        /*.n_embd_head_v = */ 128,
    };
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_quant_model_from_metadata(&desc), llama_model_free);
    check(model != nullptr, "failed to create quantization test model");

    std::vector<std::string> names;
    for (const auto & entry : a.tensors) names.push_back(entry.first);
    merge_params params;
    params.n_threads = 2;
    params.memory_budget = 64 * 1024 * 1024;
    const std::vector<ggml_type> types = plan_evo_types(
            params, a, names, model.get(), LLAMA_FTYPE_MOSTLY_Q4_K_M);
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == "blk.0.attn_q.weight") check(ggml_is_quantized(types[i]), "F32 weight was not quantized");
        if (names[i] == "output_norm.weight") check(types[i] == GGML_TYPE_F32, "norm tensor was quantized");
        if (names[i] == "blk.0.ffn_gate_tid2eid.weight") check(types[i] == GGML_TYPE_I32, "I32 tensor type changed");
    }

    std::vector<std::unique_ptr<gguf_input>> inputs;
    inputs.emplace_back(new gguf_input(model_a));
    inputs.emplace_back(new gguf_input(model_b));
    std::vector<size_t> gene_offsets(names.size(), SIZE_MAX);
    size_t n_gene_tensors = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        if (tensor_can_decode(inputs[0]->tensors.at(names[i]).tensor)) {
            gene_offsets[i] = n_gene_tensors++ * inputs.size();
        }
    }
    evo_candidate candidate;
    candidate.genes.assign(n_gene_tensors * inputs.size(), 0.5f);

    std::vector<std::unique_ptr<gguf_input>> bad_inputs;
    bad_inputs.emplace_back(new gguf_input(model_a));
    bad_inputs.emplace_back(new gguf_input(bad_data_model));
    bool data_rejected = false;
    try {
        write_evo_candidates(
                { bad_output }, params, bad_inputs, names, { candidate }, gene_offsets, types,
                LLAMA_FTYPE_MOSTLY_Q4_K_M, nullptr);
    } catch (const std::runtime_error &) {
        data_rejected = true;
    }
    check(data_rejected, "non-floating tensor mismatch was not rejected");

    write_evo_candidates(
            { output }, params, inputs, names, { candidate }, gene_offsets, types, LLAMA_FTYPE_MOSTLY_Q4_K_M, nullptr);

    gguf_input result(output);
    check(get_file_type(result.gguf) == LLAMA_FTYPE_MOSTLY_Q4_K_M, "output file type is wrong");
    check(ggml_is_quantized(result.tensors.at("blk.0.attn_q.weight").tensor->type), "output weight is not quantized");
    check(result.tensors.at("blk.0.ffn_gate_tid2eid.weight").tensor->type == GGML_TYPE_I32, "output I32 type changed");
    std::vector<uint8_t> expected;
    std::vector<uint8_t> actual;
    a.read_tensor("blk.0.ffn_gate_tid2eid.weight", expected);
    result.read_tensor("blk.0.ffn_gate_tid2eid.weight", actual);
    check(expected == actual, "output I32 data changed");

    std::filesystem::remove(output);
    std::filesystem::remove(bad_output);
    std::filesystem::remove(bad_data_model);
    std::filesystem::remove(other_template_model);
    std::filesystem::remove(bad_model);
    std::filesystem::remove(model_b);
    std::filesystem::remove(model_a);
    std::filesystem::remove(dir);
    return 0;
}
