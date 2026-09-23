#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace den2moee {

size_t q4_0_row_bytes(int ncols);

std::vector<uint8_t> copy_q4_0_rows(const void * source, int source_cols,
                                    const std::vector<int> & row_ids);

std::vector<float> dequantize_q4_0_rows(const void * source, int source_cols,
                                        const std::vector<int> & row_ids);

std::vector<uint8_t> gather_q4_0_columns(const void * source, int source_rows, int source_cols,
                                         const std::vector<int> & column_ids);

std::vector<uint8_t> quantize_q4_0(const std::vector<float> & values, int rows, int cols);

} // namespace den2moee
