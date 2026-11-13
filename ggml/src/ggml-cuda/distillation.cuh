#include "common.cuh"

void ggml_cuda_distillation(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_top_k_stats(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_mul_mat_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
