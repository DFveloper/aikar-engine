#include "dataset.h"
#include "surgery.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

static std::vector<float> normalized_direction(int64_t count) {
    std::vector<float> result(count);
    double norm = 0.0;
    for (int64_t i = 0; i < count; ++i) {
        result[i] = (float) (i + 1);
        norm += result[i] * result[i];
    }
    norm = std::sqrt(norm);
    for (float & value : result) value /= (float) norm;
    return result;
}

static float projected_norm(const std::vector<float> & weights, int64_t n_input, int64_t n_output, const std::vector<float> & direction) {
    double sum = 0.0;
    for (int64_t col = 0; col < n_input; ++col) {
        double value = 0.0;
        for (int64_t row = 0; row < n_output; ++row) value += direction[row] * weights[row*n_input + col];
        sum += value * value;
    }
    return (float) std::sqrt(sum);
}

int main() {
    const int64_t n_input = 64;
    const int64_t n_output = 17;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> weights(n_input * n_output);
    for (float & value : weights) value = dist(rng);
    const std::vector<float> direction = normalized_direction(n_output);
    const float before = projected_norm(weights, n_input, n_output, direction);
    heretic_surgery_f32(weights.data(), n_input, n_output, 1, direction.data(), 1.0f);
    const float after = projected_norm(weights, n_input, n_output, direction);
    require(after < before * 1e-5f, "F32 row-major projection was not removed");

    std::vector<float> source(n_input * n_output);
    for (float & value : source) value = dist(rng);
    heretic_tensor_shape shape;
    shape.type = GGML_TYPE_Q4_0;
    shape.ne[0] = n_input;
    shape.ne[1] = n_output;
    const size_t q_size = ggml_row_size(shape.type, n_input) * n_output;
    std::vector<uint8_t> quantized(q_size);
    require(ggml_quantize_chunk(shape.type, source.data(), quantized.data(), 0, n_output, n_input, nullptr) == q_size, "Q4_0 setup quantization failed");
    std::vector<uint8_t> edited;
    std::string error;
    require(heretic_surgery_quantized(quantized.data(), quantized.size(), shape, direction, 1.0f, edited, error), error.c_str());
    std::vector<float> decoded_before(n_input * n_output);
    std::vector<float> decoded(n_input * n_output);
    ggml_get_type_traits(shape.type)->to_float(quantized.data(), decoded_before.data(), decoded_before.size());
    ggml_get_type_traits(shape.type)->to_float(edited.data(), decoded.data(), decoded.size());
    for (float value : decoded) require(std::isfinite(value), "Q4_0 round trip produced a non-finite value");
    const float q_after = projected_norm(decoded, n_input, n_output, direction);
    const float q_before = projected_norm(decoded_before, n_input, n_output, direction);
    require(q_after < q_before * 0.25f, "Q4_0 requantization left an unexpectedly large projected component");

    std::vector<float> experts(n_input * n_output * 3);
    for (float & value : experts) value = dist(rng);
    const std::vector<float> expert_source = experts;
    heretic_surgery_f32(experts.data(), n_input, n_output, 3, direction.data(), 1.0f);
    for (int64_t i = 0; i < 3; ++i) {
        std::vector<float> slice(experts.begin() + i*n_input*n_output, experts.begin() + (i + 1)*n_input*n_output);
        require(projected_norm(slice, n_input, n_output, direction) < 1e-4f, "3D expert orientation is incorrect");
    }

    shape.ne[2] = 3;
    std::vector<uint8_t> quantized_experts(q_size * 3);
    require(ggml_quantize_chunk(shape.type, expert_source.data(), quantized_experts.data(), 0, n_output * 3, n_input, nullptr) == q_size * 3,
            "Q4_0 expert setup quantization failed");
    std::vector<uint8_t> edited_experts;
    std::vector<uint8_t> edited_experts_again;
    require(heretic_surgery_quantized(quantized_experts.data(), quantized_experts.size(), shape, direction, 1.0f, edited_experts, error), error.c_str());
    require(heretic_surgery_quantized(quantized_experts.data(), quantized_experts.size(), shape, direction, 1.0f, edited_experts_again, error), error.c_str());
    require(edited_experts == edited_experts_again, "parallel Q4_0 expert surgery is not deterministic");
    std::vector<float> decoded_experts(expert_source.size());
    for (int64_t i = 0; i < 3; ++i) {
        ggml_get_type_traits(shape.type)->to_float(edited_experts.data() + i * q_size, decoded_experts.data() + i * n_input * n_output, n_input * n_output);
        std::vector<float> slice(decoded_experts.begin() + i*n_input*n_output, decoded_experts.begin() + (i + 1)*n_input*n_output);
        require(projected_norm(slice, n_input, n_output, direction) < before * 0.25f, "Q4_0 expert projection was not removed");
    }

    std::vector<heretic_dataset_entry> dataset;
    for (int32_t group = 0; group < 2; ++group) {
        for (int32_t i = 0; i < 64; ++i) {
            heretic_dataset_entry entry;
            entry.prompt = std::to_string(group) + ":" + std::to_string(i);
            entry.group = group == 0 ? "target" : "control";
            dataset.push_back(std::move(entry));
        }
    }
    const heretic_dataset_summary summary = heretic_validate_dataset(dataset, 32, 32, 1024);
    require(summary.target_count == 64 && summary.control_count == 64 && !summary.statistically_weak, "dataset validation counts are incorrect");
    const auto split = heretic_stratified_split(dataset, 0.25, 42);
    require(split.first.size() == 96 && split.second.size() == 32, "stratified dataset split size is incorrect");
    const auto split_again = heretic_stratified_split(dataset, 0.25, 42);
    require(split.first[0].prompt == split_again.first[0].prompt, "stratified dataset split is not deterministic");
    printf("lumen-heretic surgery tests passed\n");
    return 0;
}
