#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_turboquant_matrix {
    GGML_TURBOQUANT_ROTATION_K,
    GGML_TURBOQUANT_ROTATION_V,
    GGML_TURBOQUANT_PROJECTION,
};

struct ggml_turboquant_op_params {
    int32_t head_dim;
    int32_t bits_k;
    int32_t bits_v;
    int32_t n_kv_max;
    float scale;
    float max_bias;
    float logit_softcap;
    int32_t key;
};

GGML_API struct ggml_tensor * ggml_turboquant_pack(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * indices,
    struct ggml_tensor * cache, struct ggml_tensor * parameters, int32_t head_dim, int32_t total_bits, bool key);
GGML_API struct ggml_tensor * ggml_turboquant_attn(struct ggml_context * ctx, struct ggml_tensor * query, struct ggml_tensor * cache_k,
    struct ggml_tensor * cache_v, struct ggml_tensor * mask, struct ggml_tensor * sinks, struct ggml_tensor * parameters,
    int32_t head_dim, int32_t bits_k, int32_t bits_v, float scale, float max_bias, float logit_softcap, int32_t n_kv_max);

GGML_API size_t ggml_turboquant_params_size(int32_t head_dim);
GGML_API bool ggml_turboquant_params_init(float * parameters, int32_t head_dim, uint64_t seed);
GGML_API const float * ggml_turboquant_matrix(const float * parameters, int32_t head_dim, enum ggml_turboquant_matrix matrix);
GGML_API const float * ggml_turboquant_codebook(const float * parameters, int32_t head_dim, int32_t total_bits, bool key);
GGML_API size_t ggml_turboquant_row_size(int32_t head_dim, int32_t total_bits, bool key);
GGML_API bool ggml_turboquant_pack_row(const float * input, uint8_t * packed, const float * parameters, int32_t head_dim, int32_t total_bits, bool key);
GGML_API void ggml_turboquant_unpack_row(const uint8_t * packed, float * output, const float * parameters, int32_t head_dim, int32_t total_bits, bool key);
GGML_API float ggml_turboquant_dot_row(const uint8_t * packed, const float * rotated_query, const float * projected_query, const float * parameters, int32_t head_dim, int32_t total_bits);

#ifdef __cplusplus
}
#endif
