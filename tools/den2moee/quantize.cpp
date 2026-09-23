#include "quantize.h"

#include "ggml.h"

#include <cstring>
#include <stdexcept>

namespace den2moee {

static void validate_q4_shape(int rows, int cols) {
    if (rows <= 0 || cols <= 0 || cols % ggml_blck_size(GGML_TYPE_Q4_0) != 0) {
        throw std::invalid_argument("Q4_0 shape is not valid");
    }
}

size_t q4_0_row_bytes(int ncols) {
    if (ncols <= 0 || ncols % ggml_blck_size(GGML_TYPE_Q4_0) != 0) {
        throw std::invalid_argument("Q4_0 row width is not block aligned");
    }
    return ggml_row_size(GGML_TYPE_Q4_0, ncols);
}

std::vector<uint8_t> copy_q4_0_rows(const void * source, int source_cols,
                                    const std::vector<int> & row_ids) {
    if (source == nullptr || row_ids.empty()) throw std::invalid_argument("empty Q4_0 row copy");
    const size_t row_bytes = q4_0_row_bytes(source_cols);
    std::vector<uint8_t> result(row_bytes * row_ids.size());
    const auto * src = static_cast<const uint8_t *>(source);
    for (size_t i = 0; i < row_ids.size(); ++i) {
        if (row_ids[i] < 0) throw std::invalid_argument("negative Q4_0 row index");
        std::memcpy(result.data() + i * row_bytes, src + row_ids[i] * row_bytes, row_bytes);
    }
    return result;
}

std::vector<float> dequantize_q4_0_rows(const void * source, int source_cols,
                                        const std::vector<int> & row_ids) {
    if (source == nullptr || row_ids.empty()) throw std::invalid_argument("empty Q4_0 dequantization");
    const size_t row_bytes = q4_0_row_bytes(source_cols);
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    if (!traits || !traits->to_float) throw std::runtime_error("Q4_0 dequantizer is unavailable");

    std::vector<float> result(row_ids.size() * source_cols);
    const auto * src = static_cast<const uint8_t *>(source);
    for (size_t i = 0; i < row_ids.size(); ++i) {
        if (row_ids[i] < 0) throw std::invalid_argument("negative Q4_0 row index");
        traits->to_float(src + row_ids[i] * row_bytes, result.data() + i * source_cols, source_cols);
    }
    return result;
}

std::vector<uint8_t> gather_q4_0_columns(const void * source, int source_rows, int source_cols,
                                         const std::vector<int> & column_ids) {
    if (source == nullptr || column_ids.empty()) throw std::invalid_argument("empty Q4_0 column gather");
    validate_q4_shape(source_rows, source_cols);
    if (static_cast<int>(column_ids.size()) % ggml_blck_size(GGML_TYPE_Q4_0) != 0) {
        throw std::invalid_argument("gathered Q4_0 width is not block aligned");
    }
    const size_t source_row_bytes = q4_0_row_bytes(source_cols);
    const size_t output_row_bytes = q4_0_row_bytes(column_ids.size());
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    if (!traits || !traits->to_float) throw std::runtime_error("Q4_0 dequantizer is unavailable");

    std::vector<uint8_t> result(output_row_bytes * source_rows);
    std::vector<float> source_row(source_cols);
    std::vector<float> output_row(column_ids.size());
    const auto * src = static_cast<const uint8_t *>(source);
    for (int row = 0; row < source_rows; ++row) {
        traits->to_float(src + row * source_row_bytes, source_row.data(), source_cols);
        for (size_t col = 0; col < column_ids.size(); ++col) {
            if (column_ids[col] < 0 || column_ids[col] >= source_cols) {
                throw std::invalid_argument("Q4_0 column index is out of range");
            }
            output_row[col] = source_row[column_ids[col]];
        }
        const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_0, output_row.data(),
                                                    result.data() + row * output_row_bytes,
                                                    0, 1, column_ids.size(), nullptr);
        if (written != output_row_bytes) throw std::runtime_error("Q4_0 column requantization failed");
    }
    return result;
}

std::vector<uint8_t> quantize_q4_0(const std::vector<float> & values, int rows, int cols) {
    validate_q4_shape(rows, cols);
    if (values.size() != static_cast<size_t>(rows) * cols) {
        throw std::invalid_argument("Q4_0 input size does not match shape");
    }
    std::vector<uint8_t> result(ggml_row_size(GGML_TYPE_Q4_0, cols) * rows);
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), result.data(),
                                               0, rows, cols, nullptr);
    if (written != result.size()) throw std::runtime_error("Q4_0 quantization failed");
    return result;
}

} // namespace den2moee
