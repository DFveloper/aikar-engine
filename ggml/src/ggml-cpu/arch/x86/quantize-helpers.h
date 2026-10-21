#pragma once

#include <immintrin.h>
#include <stdint.h>

#if defined(__AVX__)
static inline float ggml_q8_0_scale_avx(__m256 v0, __m256 v1, __m256 v2, __m256 v3) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 a0 = _mm256_max_ps(_mm256_andnot_ps(sign, v0), zero);
    const __m256 a1 = _mm256_max_ps(_mm256_andnot_ps(sign, v1), zero);
    const __m256 a2 = _mm256_max_ps(_mm256_andnot_ps(sign, v2), zero);
    const __m256 a3 = _mm256_max_ps(_mm256_andnot_ps(sign, v3), zero);
    const __m256 amax = _mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3));
    __m128 max4 = _mm_max_ps(_mm256_castps256_ps128(amax), _mm256_extractf128_ps(amax, 1));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_shuffle_ps(max4, max4, 1));
    const float magnitude = _mm_cvtss_f32(max4);
    if (magnitude == 0.0f) {
        return -0.0f;
    }

    const __m256 max8 = _mm256_set1_ps(magnitude);
    const uint32_t matches = (uint32_t) _mm256_movemask_ps(_mm256_cmp_ps(a0, max8, _CMP_EQ_OQ))
                          | (uint32_t) _mm256_movemask_ps(_mm256_cmp_ps(a1, max8, _CMP_EQ_OQ)) << 8
                          | (uint32_t) _mm256_movemask_ps(_mm256_cmp_ps(a2, max8, _CMP_EQ_OQ)) << 16
                          | (uint32_t) _mm256_movemask_ps(_mm256_cmp_ps(a3, max8, _CMP_EQ_OQ)) << 24;
    const uint32_t signs = (uint32_t) _mm256_movemask_ps(v0)
                        | (uint32_t) _mm256_movemask_ps(v1) << 8
                        | (uint32_t) _mm256_movemask_ps(v2) << 16
                        | (uint32_t) _mm256_movemask_ps(v3) << 24;
    // The first equal magnitude determines the scale sign.
    const uint32_t first = matches & (0u - matches);
    return signs & first ? magnitude / 128.0f : magnitude / -128.0f;
}
#endif
