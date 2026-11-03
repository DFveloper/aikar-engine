#pragma once

#include "common.cuh"

void ggml_cuda_cpy_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
void ggml_cuda_set_rows_nvfp4(ggml_backend_cuda_context & ctx, const ggml_tensor * src, const ggml_tensor * rows, ggml_tensor * dst);
