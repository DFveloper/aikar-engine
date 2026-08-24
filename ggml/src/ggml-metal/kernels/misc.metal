#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

template <typename T>
kernel void kernel_pad_impl(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t i03 = i3;
    const int32_t i02 = i2;
    const int32_t i01 = i1;

    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T * dst_ptr  = (device       T *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            dst_ptr[i0] = src0_ptr[i0];
        } else {
            dst_ptr[i0] = 0.0f;
        }
    }
}

typedef decltype(kernel_pad_impl<float>) kernel_pad_t;

template [[host_name("kernel_pad_f32")]]   kernel kernel_pad_t kernel_pad_impl<float>;
template [[host_name("kernel_pad_f32_4")]] kernel kernel_pad_t kernel_pad_impl<float4>;

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];
    const float gclip  = pars[7];

    const float gi = (gclip > 0.0f) ? clamp(g[gid], -gclip, gclip) : g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

kernel void kernel_out_prod_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device float * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    uint index = gid;
    const uint i0 = index % args.ne0; index /= args.ne0;
    const uint i1 = index % args.ne1; index /= args.ne1;
    const uint i2 = index % args.ne2; index /= args.ne2;
    const uint i3 = index;

    const uint a0 = i0 % args.ne00;
    const uint a2 = i2/(args.ne2/args.ne02);
    const uint a3 = i3/(args.ne3/args.ne03);
    const uint b0 = i1 % args.ne10;

    float sum = 0.0f;
    for (uint k = 0; k < args.ne01; ++k) {
        const uint ai = ((a3*args.ne02 + a2)*args.ne01 + k)*args.ne00 + a0;
        const uint bi = ((i3*args.ne12 + i2)*args.ne11 + k)*args.ne10 + b0;
        sum += src0[ai]*src1[bi];
    }
    dst[gid] = sum;
}

kernel void kernel_out_prod_id_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * src0 [[buffer(1)]],
        device const float * src1 [[buffer(2)]],
        device const int32_t * ids [[buffer(3)]],
        device float * dst [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    uint index = gid;
    const uint col = index % args.ne0; index /= args.ne0;
    const uint row = index % args.ne1; index /= args.ne1;
    const uint expert = index;

    float sum = 0.0f;
    for (uint token = 0; token < args.ne02; ++token) {
        for (uint slot = 0; slot < args.ne01; ++slot) {
            if (ids[token*args.ne01 + slot] == int(expert)) {
                sum += src0[(token*args.ne01 + slot)*args.ne00 + col] *
                       src1[(token*args.ne11 + slot)*args.ne10 + row];
            }
        }
    }
    dst[gid] = sum;
}

kernel void kernel_get_rows_back_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * src0 [[buffer(1)]],
        device const int32_t * ids [[buffer(2)]],
        device float * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    uint index = gid;
    const uint col = index % args.ne0; index /= args.ne0;
    const uint row = index % args.ne1; index /= args.ne1;
    const uint batch = index;
    float sum = 0.0f;
    for (uint i = 0; i < args.ne10; ++i) {
        if (ids[batch*args.ne10 + i] == int(row)) {
            sum += src0[(batch*args.ne01 + i)*args.ne00 + col];
        }
    }
    dst[gid] = sum;
}

kernel void kernel_repeat_back_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * src0 [[buffer(1)]],
        device float * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    uint index = gid;
    const uint i0 = index % args.ne0; index /= args.ne0;
    const uint i1 = index % args.ne1; index /= args.ne1;
    const uint i2 = index % args.ne2; index /= args.ne2;
    const uint i3 = index;

    float sum = 0.0f;
    for (uint s3 = i3; s3 < args.ne03; s3 += args.ne3) {
        for (uint s2 = i2; s2 < args.ne02; s2 += args.ne2) {
            for (uint s1 = i1; s1 < args.ne01; s1 += args.ne1) {
                for (uint s0 = i0; s0 < args.ne00; s0 += args.ne0) {
                    sum += src0[((s3*args.ne02 + s2)*args.ne01 + s1)*args.ne00 + s0];
                }
            }
        }
    }
    dst[gid] = sum;
}

kernel void kernel_rms_norm_back_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * grad [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const uint ncols = args.ne00;
    const uint row = gid/ncols;
    const uint base = row*ncols;
    float sum_xx = 0.0f;
    float sum_xg = 0.0f;
    for (uint col = 0; col < ncols; ++col) {
        sum_xx += x[base + col]*x[base + col];
        sum_xg += x[base + col]*grad[base + col];
    }
    const float scale_g = rsqrt(sum_xx/float(ncols) + args.param);
    const float scale_x = -scale_g*sum_xg/(sum_xx + float(ncols)*args.param);
    dst[gid] = scale_g*grad[gid] + scale_x*x[gid];
}

kernel void kernel_soft_max_back_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * grad [[buffer(1)]],
        device const float * y [[buffer(2)]],
        device float * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const uint ncols = args.ne00;
    const uint base = (gid/ncols)*ncols;
    float dot = 0.0f;
    for (uint col = 0; col < ncols; ++col) {
        dot += y[base + col]*grad[base + col];
    }
    dst[gid] = args.param*(grad[gid] - dot)*y[gid];
}

kernel void kernel_cross_entropy_loss_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device const float * labels [[buffer(2)]],
        device float * dst [[buffer(3)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid != 0) {
        return;
    }

    const uint nclasses = args.ne00;
    const uint nrows = args.ne00 > 0 ? args.ne/args.ne00 : 0;
    float total = 0.0f;
    for (uint row = 0; row < nrows; ++row) {
        const uint base = row*nclasses;
        float labels_sum = 0.0f;
        float max_logit = -INFINITY;
        for (uint col = 0; col < nclasses; ++col) {
            labels_sum += labels[base + col];
            max_logit = max(max_logit, logits[base + col]);
        }
        if (labels_sum == 0.0f) {
            continue;
        }

        float sum_exp = 0.0f;
        for (uint col = 0; col < nclasses; ++col) {
            sum_exp += exp(logits[base + col] - max_logit);
        }
        const float log_sum = log(sum_exp);
        for (uint col = 0; col < nclasses; ++col) {
            if (labels[base + col] != 0.0f) {
                total -= (logits[base + col] - max_logit - log_sum)*labels[base + col];
            }
        }
    }
    dst[0] = total/float(nrows);
}

kernel void kernel_cross_entropy_loss_back_f32(
        constant ggml_metal_kargs_training & args [[buffer(0)]],
        device const float * grad [[buffer(1)]],
        device const float * logits [[buffer(2)]],
        device const float * labels [[buffer(3)]],
        device float * dst [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.ne) {
        return;
    }

    const uint nclasses = args.ne00;
    const uint nrows = args.ne/nclasses;
    const uint base = (gid/nclasses)*nclasses;
    float labels_sum = 0.0f;
    float max_logit = -INFINITY;
    for (uint col = 0; col < nclasses; ++col) {
        labels_sum += labels[base + col];
        max_logit = max(max_logit, logits[base + col]);
    }
    if (labels_sum == 0.0f) {
        dst[gid] = 0.0f;
        return;
    }

    float sum_exp = 0.0f;
    for (uint col = 0; col < nclasses; ++col) {
        sum_exp += exp(logits[base + col] - max_logit);
    }
    const float prob = exp(logits[gid] - max_logit)/sum_exp;
    dst[gid] = (prob*labels_sum - labels[gid])*grad[0]/float(nrows);
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N>
kernel void kernel_fwht_f32(
        constant ggml_metal_kargs_fwht & args,
        device const float * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = src[i*NW + lane]*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

typedef decltype(kernel_fwht_f32<64>) kernel_fwht_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_t kernel_fwht_f32<64>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_t kernel_fwht_f32<128>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_t kernel_fwht_f32<256>;
template [[host_name("kernel_fwht_f32_512")]] kernel kernel_fwht_t kernel_fwht_f32<512>;

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + it*args.nb_w1);
    }

    float w[hc];
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        w[ih] = simd_shuffle(weight_lane, ih);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        result = fma(*(device const float *) (xb + ih*args.nb_x1), w[ih], result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = result;
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}
