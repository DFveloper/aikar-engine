#pragma once

#include "common.h"

void quantize_q1_0(device const float * src, device block_q1_0 & dst) {
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; j++) {
        sum_abs += fabs(src[j]);
    }
    dst.d = sum_abs / QK1_0;

    for (int j = 0; j < QK1_0 / 8; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK1_0; j++) {
        if (src[j] >= 0.0f) {
            dst.qs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_q2_0(device const float * src, device block_q2_0 & dst) {
    float amax = 0.0f;
    for (int j = 0; j < QK2_0; j++) {
        float a = fabs(src[j]);
        if (a > amax) amax = a;
    }
    const float d = amax;
    dst.d = d;

    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    for (int j = 0; j < QK2_0 / 4; j++) {
        dst.qs[j] = 0;
    }
    for (int j = 0; j < QK2_0; j++) {
        int q = (int)round(src[j] * id) + 1;
        q = max(0, min(3, q));
        dst.qs[j / 4] |= (q << (2 * (j % 4)));
    }
}

void quantize_q4_0(device const float * src, device block_q4_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -8;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK4_0/2 + j]*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q4_1(device const float * src, device block_q4_1 & dst) {
#pragma METAL fp math_mode(safe)
    float min = FLT_MAX;
    float max = -FLT_MAX;

    for (int j = 0; j < QK4_1; j++) {
        const float v = src[j];
        if (min > v) min = v;
        if (max < v) max = v;
    }

    const float d = (max - min) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK4_1/2 + j] - min)*id;

        const uint8_t xi0 = MIN(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = MIN(15, (int8_t)(x1 + 0.5f));

        dst.qs[j]  = xi0;
        dst.qs[j] |= xi1 << 4;
    }
}

void quantize_q5_0(device const float * src, device block_q5_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK5_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / -16;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = src[0       + j]*id;
        const float x1 = src[QK5_0/2 + j]*id;

        const uint8_t xi0 = MIN(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = MIN(31, (int8_t)(x1 + 16.5f));

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q5_1(device const float * src, device block_q5_1 & dst) {
#pragma METAL fp math_mode(safe)
    float max = src[0];
    float min = src[0];

    for (int j = 1; j < QK5_1; j++) {
        const float v = src[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;
    dst.m = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (src[0       + j] - min)*id;
        const float x1 = (src[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        dst.qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }

    thread const uint8_t * qh8 = (thread const uint8_t *)&qh;

    for (int j = 0; j < 4; ++j) {
        dst.qh[j] = qh8[j];
    }
}

void quantize_q8_0(device const float * src, device block_q8_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float vmax = 0.0f;

    for (int j = 0; j < QK8_0; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            vmax = v;
        }
    }

    const float d = vmax / -128.0f;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = src[j]*id;

        dst.qs[j] = clamp((int) round(x0), -128, 127);
    }
}

void quantize_iq4_nl(device const float * src, device block_iq4_nl & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max
    float max  = 0.0f;

    for (int j = 0; j < QK4_NL; j++) {
        const float v = src[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            max  = v;
        }
    }

    const float d = max / kvalues_iq4nl_f[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = src[0        + j]*id;
        const float x1 = src[QK4_NL/2 + j]*id;

        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl_f, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl_f, x1);

        dst.qs[j] = xi0 | (xi1 << 4);

        const float v0 = kvalues_iq4nl_f[xi0];
        const float v1 = kvalues_iq4nl_f[xi1];
        const float w0 = src[0        + j]*src[0        + j];
        const float w1 = src[QK4_NL/2 + j]*src[QK4_NL/2 + j];
        sumqx += w0*v0*src[j] + w1*v1*src[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;

    }

    dst.d = sumq2 > 0 ? sumqx/sumq2 : d;
}

void quantize_q3_K(device const float * src, device block_q3_K & dst) {
#pragma METAL fp math_mode(safe)
    float scales[16];
    int quants[QK_K];
    float max_scale = 0.0f;
    float max_scale_value = 0.0f;

    for (int j = 0; j < 16; ++j) {
        float amax = 0.0f;
        float vmax = 0.0f;
        for (int l = 0; l < 16; ++l) {
            const float v = src[16*j + l];
            if (amax < fabs(v)) {
                amax = fabs(v);
                vmax = v;
            }
        }
        scales[j] = vmax / -4.0f;
        if (max_scale < fabs(scales[j])) {
            max_scale = fabs(scales[j]);
            max_scale_value = scales[j];
        }
    }

    uint8_t scale_bytes[12] = {};
    float d_all = 0.0f;
    if (max_scale != 0.0f) {
        const float iscale = -32.0f/max_scale_value;
        d_all = 1.0f/iscale;
        for (int j = 0; j < 16; ++j) {
            int l = int(round(iscale*scales[j]));
            l = min(31, max(-32, l)) + 32;
            if (j < 8) {
                scale_bytes[j] = uint8_t(l & 0xF);
            } else {
                scale_bytes[j - 8] |= uint8_t((l & 0xF) << 4);
            }
            scale_bytes[8 + j % 4] |= uint8_t((uint(l) >> 4) << (2*(j/4)));
        }
    }

    for (int j = 0; j < 16; ++j) {
        const int sc = (j < 8 ? int(scale_bytes[j] & 0xF) : int(scale_bytes[j - 8] >> 4)) |
            (int((scale_bytes[8 + j % 4] >> (2*(j/4))) & 3) << 4);
        const float d = d_all * float(sc - 32);
        for (int l = 0; l < 16; ++l) {
            const int q = d == 0.0f ? 0 : int(round(src[16*j + l]/d));
            quants[16*j + l] = min(3, max(-4, q)) + 4;
        }
    }

    for (int j = 0; j < 32; ++j) {
        dst.hmask[j] = 0;
    }
    for (int j = 0; j < QK_K; ++j) {
        if (quants[j] > 3) {
            dst.hmask[j % 32] |= uint8_t(1 << (j / 32));
            quants[j] -= 4;
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            dst.qs[j/4 + l] = uint8_t(quants[j + l] | (quants[j + l + 32] << 2) |
                (quants[j + l + 64] << 4) | (quants[j + l + 96] << 6));
        }
    }
    dst.d = d_all;
    for (int j = 0; j < 12; ++j) {
        dst.scales[j] = scale_bytes[j];
    }
}

void quantize_q4_K(device const float * src, device block_q4_K & dst) {
#pragma METAL fp math_mode(safe)
    float scales[8];
    float mins[8];
    int quants[QK_K];
    float max_scale = 0.0f;
    float max_min = 0.0f;

    for (int j = 0; j < 8; ++j) {
        float vmin = src[32*j];
        float vmax = vmin;
        for (int l = 1; l < 32; ++l) {
            const float v = src[32*j + l];
            vmin = min(vmin, v);
            vmax = max(vmax, v);
        }
        mins[j] = max(0.0f, -vmin);
        scales[j] = (vmax + mins[j])/15.0f;
        max_scale = max(max_scale, scales[j]);
        max_min = max(max_min, mins[j]);
    }

    uint8_t scale_bytes[12] = {};
    const float d_all = float(half(max_scale/63.0f));
    const float dmin_all = float(half(max_min/63.0f));
    dst.d = d_all;
    dst.dmin = dmin_all;

    for (int j = 0; j < 8; ++j) {
        const int ls = min(63, int(round(max_scale == 0.0f ? 0.0f : 63.0f*scales[j]/max_scale)));
        const int lm = min(63, int(round(max_min == 0.0f ? 0.0f : 63.0f*mins[j]/max_min)));
        if (j < 4) {
            scale_bytes[j] = uint8_t(ls);
            scale_bytes[j + 4] = uint8_t(lm);
        } else {
            scale_bytes[j + 4] = uint8_t((ls & 0xF) | ((lm & 0xF) << 4));
            scale_bytes[j - 4] |= uint8_t((ls >> 4) << 6);
            scale_bytes[j] |= uint8_t((lm >> 4) << 6);
        }
    }

    for (int j = 0; j < 8; ++j) {
        const int ls = j < 4 ? int(scale_bytes[j] & 0x3F) : int((scale_bytes[j + 4] & 0xF) | ((scale_bytes[j - 4] >> 6) << 4));
        const int lm = j < 4 ? int(scale_bytes[j + 4] & 0x3F) : int((scale_bytes[j + 4] >> 4) | ((scale_bytes[j] >> 6) << 4));
        const float d = d_all*float(ls);
        const float dm = dmin_all*float(lm);
        for (int l = 0; l < 32; ++l) {
            const int q = d == 0.0f ? 0 : int(round((src[32*j + l] + dm)/d));
            quants[32*j + l] = min(15, max(0, q));
        }
    }

    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; ++l) {
            dst.qs[j/2 + l] = uint8_t(quants[j + l] | (quants[j + l + 32] << 4));
        }
    }
    for (int j = 0; j < 12; ++j) {
        dst.scales[j] = scale_bytes[j];
    }
}

void quantize_mxfp4(device const float * src, device block_mxfp4 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = max(amax, fabs(src[j]));
    }

    const int exponent = amax > 0.0f ? int(floor(log2(amax))) - 2 + 127 : 0;
    dst.e = uint8_t(clamp(exponent, 0, 255));
    const float d = e8m0_to_fp32(dst.e);

    for (int j = 0; j < QK_MXFP4/2; ++j) {
        uint8_t q0 = 0;
        uint8_t q1 = 0;
        float err0 = fabs(kvalues_mxfp4_f[0]*d - src[j]);
        float err1 = fabs(kvalues_mxfp4_f[0]*d - src[QK_MXFP4/2 + j]);
        for (uint8_t q = 1; q < 16; ++q) {
            const float cur0 = fabs(kvalues_mxfp4_f[q]*d - src[j]);
            const float cur1 = fabs(kvalues_mxfp4_f[q]*d - src[QK_MXFP4/2 + j]);
            if (cur0 < err0) {
                q0 = q;
                err0 = cur0;
            }
            if (cur1 < err1) {
                q1 = q;
                err1 = cur1;
            }
        }
        dst.qs[j] = q0 | (q1 << 4);
    }
}

void quantize_tq2_0(device const float * src, device block_tq2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK_K; j++) {
        const float v = src[j];
        amax = MAX(amax, fabs(v));
    }

    const float d = amax;
    const float id = d ? 1.0f/d : 0.0f;

    dst.d = (half) d;

    for (int j = 0; j < QK_K/4; j += 32) {
        for (int m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (int n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = (int)round(src[m + n*32] * id) + 1;
                q += (uint8_t)((xi & 3) << (2*n));
            }
            dst.qs[j + m] = q;
        }
        src += 4*32;
    }
}
