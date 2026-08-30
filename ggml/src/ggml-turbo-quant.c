/*
 * TurboQuant KV codec.
 *
 * Algorithm and packed layout follow TheTom/llama-cpp-turboquant commit
 * df7f5472949ce37cdc6a2155ef6b8836a8c10bac (MIT), derived from TurboQuant (arXiv:2504.19874).
 */

#include "ggml-quants.h"

#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#define TURBO_GROUP_SIZE 128
#define TURBO_INV_SQRT_128 0.08838834764831845f

static const float turbo_centroids_3[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f,
};

static const float turbo_centroids_4[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f,
    -0.083292f, -0.058050f, -0.034299f, -0.011349f,
     0.011349f,  0.034299f,  0.058050f,  0.083292f,
     0.111036f,  0.143016f,  0.182877f,  0.241529f,
};

static const float turbo_signs_1[TURBO_GROUP_SIZE] = {
    -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
    -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
};

static const float turbo_signs_2[TURBO_GROUP_SIZE] = {
     1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
     1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
     1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
};

static void turbo_wht_block(float * x, int inverse) {
    const float * signs_first  = inverse ? turbo_signs_2 : turbo_signs_1;
    const float * signs_second = inverse ? turbo_signs_1 : turbo_signs_2;

    for (int i = 0; i < TURBO_GROUP_SIZE; ++i) {
        x[i] *= signs_first[i];
    }

    for (int step = 1; step < TURBO_GROUP_SIZE; step *= 2) {
        for (int i = 0; i < TURBO_GROUP_SIZE; i += 2*step) {
            for (int j = i; j < i + step; ++j) {
                const float a = x[j];
                const float b = x[j + step];
                x[j]        = a + b;
                x[j + step] = a - b;
            }
        }
    }

    for (int i = 0; i < TURBO_GROUP_SIZE; ++i) {
        x[i] *= TURBO_INV_SQRT_128*signs_second[i];
    }
}

void ggml_turbo_wht_forward_f32(float * x, int64_t k) {
    assert(k % TURBO_GROUP_SIZE == 0);
    for (int64_t i = 0; i < k; i += TURBO_GROUP_SIZE) {
        turbo_wht_block(x + i, 0);
    }
}

void ggml_turbo_wht_inverse_f32(float * x, int64_t k) {
    assert(k % TURBO_GROUP_SIZE == 0);
    for (int64_t i = 0; i < k; i += TURBO_GROUP_SIZE) {
        turbo_wht_block(x + i, 1);
    }
}

static int turbo_nearest_3(float x) {
    if (x < -0.154496f) return 0;
    if (x < -0.092804f) return 1;
    if (x < -0.044243f) return 2;
    if (x <  0.000000f) return 3;
    if (x <  0.044243f) return 4;
    if (x <  0.092804f) return 5;
    if (x <  0.154496f) return 6;
    return 7;
}

static int turbo_nearest_4(float x) {
    if (x < -0.212203f) return 0;
    if (x < -0.162947f) return 1;
    if (x < -0.127026f) return 2;
    if (x < -0.097164f) return 3;
    if (x < -0.070671f) return 4;
    if (x < -0.046174f) return 5;
    if (x < -0.022824f) return 6;
    if (x <  0.000000f) return 7;
    if (x <  0.022824f) return 8;
    if (x <  0.046174f) return 9;
    if (x <  0.070671f) return 10;
    if (x <  0.097164f) return 11;
    if (x <  0.127026f) return 12;
    if (x <  0.162947f) return 13;
    if (x <  0.212203f) return 14;
    return 15;
}

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    for (int64_t ib = 0; ib < k/QK_TURBO3; ++ib) {
        float values[QK_TURBO3];
        float norm_sq = 0.0f;

        for (int i = 0; i < QK_TURBO3; ++i) {
            values[i] = x[ib*QK_TURBO3 + i];
            norm_sq += values[i]*values[i];
        }

        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int i = 0; i < QK_TURBO3; ++i) {
            values[i] *= inv_norm;
        }
        turbo_wht_block(values, 0);

        memset(y[ib].qs,    0, sizeof(y[ib].qs));
        memset(y[ib].signs, 0, sizeof(y[ib].signs));

        float recon_sq = 0.0f;
        for (int i = 0; i < QK_TURBO3; ++i) {
            const int q = turbo_nearest_3(values[i]);
            y[ib].qs[i/4] |= (uint8_t) ((q & 3) << (2*(i % 4)));
            y[ib].signs[i/8] |= (uint8_t) ((q >> 2) << (i % 8));
            recon_sq += turbo_centroids_3[q]*turbo_centroids_3[q];
        }

        const float recon_norm = sqrtf(recon_sq);
        const float corrected_norm = recon_norm > 1e-10f ? norm/recon_norm : norm;
        y[ib].norm = GGML_FP32_TO_FP16(corrected_norm);
    }
}

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);

    for (int64_t ib = 0; ib < k/QK_TURBO4; ++ib) {
        float values[QK_TURBO4];
        float norm_sq = 0.0f;

        for (int i = 0; i < QK_TURBO4; ++i) {
            values[i] = x[ib*QK_TURBO4 + i];
            norm_sq += values[i]*values[i];
        }

        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int i = 0; i < QK_TURBO4; ++i) {
            values[i] *= inv_norm;
        }
        turbo_wht_block(values, 0);

        memset(y[ib].qs, 0, sizeof(y[ib].qs));

        float recon_sq = 0.0f;
        for (int i = 0; i < QK_TURBO4; ++i) {
            const int q = turbo_nearest_4(values[i]);
            y[ib].qs[i/2] |= (uint8_t) (q << (4*(i % 2)));
            recon_sq += turbo_centroids_4[q]*turbo_centroids_4[q];
        }

        const float recon_norm = sqrtf(recon_sq);
        const float corrected_norm = recon_norm > 1e-10f ? norm/recon_norm : norm;
        y[ib].norm = GGML_FP32_TO_FP16(corrected_norm);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    for (int64_t ib = 0; ib < k/QK_TURBO3; ++ib) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int i = 0; i < QK_TURBO3; ++i) {
            const int ql = (x[ib].qs[i/4] >> (2*(i % 4))) & 3;
            const int qh = (x[ib].signs[i/8] >> (i % 8)) & 1;
            y[ib*QK_TURBO3 + i] = norm*turbo_centroids_3[ql | (qh << 2)];
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO4 == 0);

    for (int64_t ib = 0; ib < k/QK_TURBO4; ++ib) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int i = 0; i < QK_TURBO4; ++i) {
            const int q = (x[ib].qs[i/2] >> (4*(i % 2))) & 15;
            y[ib*QK_TURBO4 + i] = norm*turbo_centroids_4[q];
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    const size_t row_size = n_per_row/QK_TURBO3*sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo3_0_ref(src + row*n_per_row, (block_turbo3_0 *) ((char *) dst + row*row_size), n_per_row);
    }
    return nrows*row_size;
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    const size_t row_size = n_per_row/QK_TURBO4*sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo4_0_ref(src + row*n_per_row, (block_turbo4_0 *) ((char *) dst + row*row_size), n_per_row);
    }
    return nrows*row_size;
}
