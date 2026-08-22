#pragma once

#include "common.cuh"

void ggml_cuda_acc_qlion_qat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_opt_step_qlion_qat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_opt_step_qlion_qat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_opt_step_qlion_qat_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_opt_step_qlion_qat_tied(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
