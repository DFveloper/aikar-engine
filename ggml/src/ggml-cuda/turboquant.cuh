#pragma once

#include "common.cuh"

void ggml_cuda_op_turboquant_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_turboquant_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
