#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_set_rows(ggml_backend_cuda_context & ctx,
                                    ggml_tensor *               rms_norm,
                                    ggml_tensor *               set_rows);
