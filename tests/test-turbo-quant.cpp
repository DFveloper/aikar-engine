#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

struct metrics {
    double mse;
    double max_error;
    double cosine;
    double norm_error;
};

static metrics measure(const std::vector<float> & a, const std::vector<float> & b) {
    double error_sq = 0.0;
    double max_error = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        const double error = (double) a[i] - b[i];
        error_sq += error*error;
        max_error = std::max(max_error, std::abs(error));
        dot += (double) a[i]*b[i];
        norm_a += (double) a[i]*a[i];
        norm_b += (double) b[i]*b[i];
    }

    const double cosine = norm_a == 0.0 && norm_b == 0.0 ? 1.0 : dot/std::sqrt(norm_a*norm_b);
    const double norm_error = norm_a == 0.0 ? std::sqrt(norm_b) : std::abs(std::sqrt(norm_b/norm_a) - 1.0);
    return { error_sq/a.size(), max_error, cosine, norm_error };
}

static bool test_wht_round_trip() {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    std::vector<std::vector<float>> cases(6, std::vector<float>(256));
    for (int i = 0; i < 256; ++i) {
        cases[0][i] = normal(rng);
        cases[1][i] = uniform(rng);
        cases[2][i] = 0.0f;
        cases[3][i] = i % 31 == 0 ? 100.0f*normal(rng) : 0.01f*normal(rng);
        cases[4][i] = 1e-20f*normal(rng);
        cases[5][i] = 1e20f*uniform(rng);
    }

    bool ok = true;
    for (size_t i = 0; i < cases.size(); ++i) {
        std::vector<float> actual = cases[i];
        ggml_turbo_wht_forward_f32(actual.data(), actual.size());
        ggml_turbo_wht_inverse_f32(actual.data(), actual.size());
        const metrics result = measure(cases[i], actual);
        const double scale = *std::max_element(cases[i].begin(), cases[i].end(), [](float a, float b) {
            return std::abs(a) < std::abs(b);
        });
        const double relative_max = result.max_error/std::max(1e-30, std::abs(scale));
        const bool case_ok = relative_max < 2e-6 && result.cosine > 0.999999;
        std::printf("WHT case %zu: max_rel=%g cosine=%.9f %s\n", i, relative_max, result.cosine, case_ok ? "ok" : "FAILED");
        ok = ok && case_ok;
    }
    return ok;
}

static std::vector<float> make_distribution(int id) {
    std::mt19937 rng(100 + id);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> values(256);

    for (int i = 0; i < 256; ++i) {
        if (id == 0) values[i] = normal(rng);
        if (id == 1) values[i] = uniform(rng);
        if (id == 2) values[i] = 0.0f;
        if (id == 3) values[i] = i % 29 == 0 ? 30.0f*normal(rng) : 0.05f*normal(rng);
        if (id == 4) values[i] = std::exp(-0.02f*i)*normal(rng);
    }
    return values;
}

static bool test_codec(ggml_type type, double min_cosine, double max_norm_error, bool transformed = true) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    bool ok = true;

    for (int id = 0; id < 5; ++id) {
        const std::vector<float> input = make_distribution(id);
        std::vector<uint8_t> packed(ggml_row_size(type, input.size()));
        std::vector<float> output(input.size());

        traits->from_float_ref(input.data(), packed.data(), input.size());
        traits->to_float(packed.data(), output.data(), output.size());
        if (transformed) {
            ggml_turbo_wht_inverse_f32(output.data(), output.size());
        }

        const metrics result = measure(input, output);
        const bool finite = std::isfinite(result.mse) && std::isfinite(result.max_error) && std::isfinite(result.cosine);
        const bool case_ok = finite && result.cosine >= min_cosine && result.norm_error <= max_norm_error;
        std::printf("%s case %d: mse=%g max=%g cosine=%.7f norm_err=%g %s\n",
                ggml_type_name(type), id, result.mse, result.max_error, result.cosine, result.norm_error,
                case_ok ? "ok" : "FAILED");
        ok = ok && case_ok;
    }

    return ok;
}

static double dot(const std::vector<float> & a, const std::vector<float> & b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += (double) a[i]*b[i];
    }
    return result;
}

static double dot(const float * a, const float * b, size_t n) {
    double result = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result += (double) a[i]*b[i];
    }
    return result;
}

static void normalized_hadamard_ref(float * values, int n, int block) {
    std::vector<float> input(block);
    for (int offset = 0; offset < n; offset += block) {
        std::memcpy(input.data(), values + offset, block*sizeof(float));
        const float scale = 1.0f/std::sqrt((float) block);
        for (int row = 0; row < block; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < block; ++col) {
                int bits = row & col;
                int parity = 0;
                while (bits != 0) {
                    parity ^= bits & 1;
                    bits >>= 1;
                }
                sum += (parity == 0 ? 1.0f : -1.0f)*input[col];
            }
            values[offset + row] = scale*sum;
        }
    }
}

static bool test_normalized_hadamard_attention() {
    std::mt19937 rng(7128);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    bool ok = true;

    for (const auto & shape : { std::pair{ 64, 64 }, std::pair{ 192, 64 }, std::pair{ 256, 256 }, std::pair{ 512, 512 } }) {
        std::vector<float> q(shape.first);
        std::vector<float> k(shape.first);
        std::vector<float> v(shape.first);
        for (float & value : q) value = normal(rng);
        for (float & value : k) value = normal(rng);
        for (float & value : v) value = normal(rng);

        const double dot_ref = dot(q, k);
        std::vector<float> q_rot = q;
        std::vector<float> k_rot = k;
        std::vector<float> v_rot = v;
        normalized_hadamard_ref(q_rot.data(), shape.first, shape.second);
        normalized_hadamard_ref(k_rot.data(), shape.first, shape.second);
        normalized_hadamard_ref(v_rot.data(), shape.first, shape.second);
        const double dot_rot = dot(q_rot, k_rot);
        normalized_hadamard_ref(v_rot.data(), shape.first, shape.second);

        const metrics round_trip = measure(v, v_rot);
        const double dot_rel = std::abs(dot_ref - dot_rot)/std::max(1.0, std::abs(dot_ref));
        const bool shape_ok = dot_rel < 4e-6 && round_trip.max_error < 2e-5 && round_trip.cosine > 0.999999;
        std::printf("Hadamard D%d/B%d: dot_rel=%g round_trip_max=%g %s\n",
                shape.first, shape.second, dot_rel, round_trip.max_error, shape_ok ? "ok" : "FAILED");
        ok = shape_ok && ok;
    }

    return ok;
}

static bool test_transformed_attention(ggml_type type) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    std::vector<float> query = make_distribution(0);
    std::vector<float> value = make_distribution(3);
    std::vector<uint8_t> packed(ggml_row_size(type, value.size()));
    std::vector<float> transformed(value.size());
    std::vector<float> reconstructed(value.size());

    traits->from_float_ref(value.data(), packed.data(), value.size());
    traits->to_float(packed.data(), transformed.data(), transformed.size());
    reconstructed = transformed;
    ggml_turbo_wht_inverse_f32(reconstructed.data(), reconstructed.size());

    const double reference = dot(query, reconstructed);
    ggml_turbo_wht_forward_f32(query.data(), query.size());
    const double transformed_dot = dot(query, transformed);
    const double dot_error = std::abs(reference - transformed_dot)/std::max(1.0, std::abs(reference));

    std::vector<float> sum_transformed(value.size(), 0.0f);
    std::vector<float> sum_reconstructed(value.size(), 0.0f);
    const float weights[3] = { 0.15f, 0.25f, 0.60f };
    for (int row = 0; row < 3; ++row) {
        std::vector<float> current = make_distribution(row);
        traits->from_float_ref(current.data(), packed.data(), current.size());
        traits->to_float(packed.data(), transformed.data(), transformed.size());
        reconstructed = transformed;
        ggml_turbo_wht_inverse_f32(reconstructed.data(), reconstructed.size());
        for (size_t i = 0; i < current.size(); ++i) {
            sum_transformed[i] += weights[row]*transformed[i];
            sum_reconstructed[i] += weights[row]*reconstructed[i];
        }
    }
    ggml_turbo_wht_inverse_f32(sum_transformed.data(), sum_transformed.size());
    const metrics accumulation = measure(sum_reconstructed, sum_transformed);

    const bool ok = dot_error < 2e-6 && accumulation.max_error < 2e-5 && accumulation.cosine > 0.999999;
    std::printf("%s transformed attention: dot_rel=%g v_max=%g %s\n",
            ggml_type_name(type), dot_error, accumulation.max_error, ok ? "ok" : "FAILED");
    return ok;
}

static bool test_packed_size() {
    ggml_init_params params = { 1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    bool ok = true;

    for (const auto & item : {
            std::pair{ GGML_TYPE_TURBO3_0, size_t(100) },
            std::pair{ GGML_TYPE_TURBO4_0, size_t(132) },
            std::pair{ GGML_TYPE_MXFP4, size_t(136) } }) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, item.first, 256, 7);
        const size_t expected = item.second*7;
        const bool type_ok = ggml_row_size(item.first, 256) == item.second && ggml_nbytes(tensor) == expected;
        std::printf("%s packed size: expected=%zu tensor=%zu %s\n",
                ggml_type_name(item.first), expected, ggml_nbytes(tensor), type_ok ? "ok" : "FAILED");
        ok = type_ok && ok;
    }

    ggml_free(ctx);
    return ok;
}

static bool test_backend_wht(ggml_backend_t backend, const char * label) {
    ggml_init_params params = { 2*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 5);
    ggml_tensor * forward = ggml_turbo_wht(ctx, src, false);
    ggml_tensor * inverse = ggml_turbo_wht(ctx, forward, true);

    std::vector<float> input(ggml_nelements(src));
    std::mt19937 rng(8128);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float & value : input) {
        value = normal(rng);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, inverse);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buffer != nullptr;
    if (ok) {
        ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
    }

    std::vector<float> output(input.size());
    if (ok) {
        ggml_backend_tensor_get(inverse, output.data(), 0, output.size()*sizeof(float));
        const metrics result = measure(input, output);
        ok = result.max_error < 3e-6 && result.cosine > 0.999999;
        std::printf("%s WHT round trip: max=%g cosine=%.9f %s\n", label, result.max_error, result.cosine, ok ? "ok" : "FAILED");
    }

    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return ok;
}

static bool test_backend_set_rows(ggml_backend_t backend, const char * label, ggml_type type) {
    ggml_init_params params = { 2*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx, type, 256, 7);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 7);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 7);
    ggml_tensor * out = ggml_set_rows(ctx, dst, src, rows);

    std::vector<float> input(ggml_nelements(src));
    std::mt19937 rng(9000 + type);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (float & value : input) {
        value = normal(rng);
    }
    const int32_t row_ids[7] = { 6, 2, 4, 0, 5, 1, 3 };

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buffer != nullptr;
    if (ok) {
        std::vector<uint8_t> zeros(ggml_nbytes(dst), 0);
        ggml_backend_tensor_set(dst, zeros.data(), 0, zeros.size());
        ggml_backend_tensor_set(src, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(rows, row_ids, 0, sizeof(row_ids));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
    }

    if (ok) {
        std::vector<uint8_t> packed(ggml_nbytes(out));
        ggml_backend_tensor_get(out, packed.data(), 0, packed.size());
        bool packed_match = true;
        if (type == GGML_TYPE_MXFP4) {
            std::vector<uint8_t> reference(packed.size(), 0);
            const ggml_type_traits * traits = ggml_get_type_traits(type);
            for (int src_row = 0; src_row < 7; ++src_row) {
                traits->from_float_ref(input.data() + src_row*256, reference.data() + row_ids[src_row]*ggml_row_size(type, 256), 256);
            }
            packed_match = packed == reference;
        }
        std::vector<float> reconstructed(input.size());
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        for (int src_row = 0; src_row < 7; ++src_row) {
            const int dst_row = row_ids[src_row];
            traits->to_float(packed.data() + dst_row*ggml_row_size(type, 256), reconstructed.data() + dst_row*256, 256);
            if (type == GGML_TYPE_TURBO3_0 || type == GGML_TYPE_TURBO4_0) {
                ggml_turbo_wht_inverse_f32(reconstructed.data() + dst_row*256, 256);
            }
        }
        std::vector<float> reordered(input.size());
        for (int src_row = 0; src_row < 7; ++src_row) {
            std::memcpy(reordered.data() + row_ids[src_row]*256, input.data() + src_row*256, 256*sizeof(float));
        }
        const metrics result = measure(reordered, reconstructed);
        const double min_cosine = type == GGML_TYPE_TURBO3_0 ? 0.94 : 0.98;
        const double max_norm_error = type == GGML_TYPE_MXFP4 ? 0.10 : 0.01;
        ok = packed_match && result.cosine > min_cosine && result.norm_error < max_norm_error;
        std::printf("%s %s SET_ROWS: cosine=%.7f norm_err=%g packed=%s %s\n",
            label, ggml_type_name(type), result.cosine, result.norm_error, packed_match ? "match" : "different", ok ? "ok" : "FAILED");
    }

    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return ok;
}

static bool test_backend_attention(
        ggml_backend_t backend, const char * label, ggml_type type_k, ggml_type type_v, int d = 256, int kv = 256) {
    constexpr int q_heads = 16;
    constexpr int kv_heads = 2;

    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, q_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type_k, d, kv, kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type_v, d, kv, kv_heads, 1);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    const bool turbo_k = type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0;
    const bool turbo_v = type_v == GGML_TYPE_TURBO3_0 || type_v == GGML_TYPE_TURBO4_0;
    ggml_tensor * out = turbo_v ? ggml_turbo_wht(ctx, attn, true) : attn;

    std::mt19937 rng(12000 + 10*type_k + type_v);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<float> q_host(d*q_heads);
    std::vector<float> k_host(d*kv*kv_heads);
    std::vector<float> v_host(d*kv*kv_heads);
    for (float & value : q_host) value = normal(rng);
    for (float & value : k_host) value = normal(rng);
    for (float & value : v_host) value = normal(rng);

    if (turbo_k) {
        for (int h = 0; h < q_heads; ++h) {
            ggml_turbo_wht_forward_f32(q_host.data() + h*d, d);
        }
    }

    const ggml_type_traits * traits_k = ggml_get_type_traits(type_k);
    const ggml_type_traits * traits_v = ggml_get_type_traits(type_v);
    const size_t row_k = ggml_row_size(type_k, d);
    const size_t row_v = ggml_row_size(type_v, d);
    std::vector<uint8_t> k_packed(row_k*kv*kv_heads);
    std::vector<uint8_t> v_packed(row_v*kv*kv_heads);
    std::vector<float> k_dequant(k_host.size());
    std::vector<float> v_dequant(v_host.size());
    for (int row = 0; row < kv*kv_heads; ++row) {
        traits_k->from_float_ref(k_host.data() + row*d, k_packed.data() + row*row_k, d);
        traits_v->from_float_ref(v_host.data() + row*d, v_packed.data() + row*row_v, d);
        traits_k->to_float(k_packed.data() + row*row_k, k_dequant.data() + row*d, d);
        traits_v->to_float(v_packed.data() + row*row_v, v_dequant.data() + row*d, d);
    }

    std::vector<float> expected(d*q_heads);
    std::vector<float> scores(kv);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh/(q_heads/kv_heads);
        float max_score = -INFINITY;
        for (int token = 0; token < kv; ++token) {
            scores[token] = (float) dot(q_host.data() + qh*d, k_dequant.data() + (kvh*kv + token)*d, d)/std::sqrt((float) d);
            max_score = std::max(max_score, scores[token]);
        }
        float denominator = 0.0f;
        for (float & score : scores) {
            score = std::exp(score - max_score);
            denominator += score;
        }
        for (int token = 0; token < kv; ++token) {
            const float weight = scores[token]/denominator;
            const float * value = v_dequant.data() + (kvh*kv + token)*d;
            for (int i = 0; i < d; ++i) {
                expected[qh*d + i] += weight*value[i];
            }
        }
        if (turbo_v) {
            ggml_turbo_wht_inverse_f32(expected.data() + qh*d, d);
        }
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buffer != nullptr;
    if (ok) {
        ggml_backend_tensor_set(q, q_host.data(), 0, q_host.size()*sizeof(float));
        ggml_backend_tensor_set(k, k_packed.data(), 0, k_packed.size());
        ggml_backend_tensor_set(v, v_packed.data(), 0, v_packed.size());
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        ggml_backend_synchronize(backend);
    }

    if (ok) {
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
        const metrics result = measure(expected, actual);
        ok = result.cosine > 0.999 && result.norm_error < 0.02 && result.max_error < 0.02;
        std::printf("%s attention D%d/KV%d %s/%s: mse=%g max=%g cosine=%.7f %s\n",
            label, d, kv, ggml_type_name(type_k), ggml_type_name(type_v), result.mse, result.max_error, result.cosine,
            ok ? "ok" : "FAILED");
    }

    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return ok;
}

static bool test_backend_kind(const char * prefix) {
    bool found = false;
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(dev);
        if (std::strncmp(name, prefix, std::strlen(prefix)) != 0) {
            continue;
        }
        found = true;
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        ok = backend != nullptr && ok;
        if (backend == nullptr) {
            continue;
        }
        ok = test_backend_wht(backend, name) && ok;
        ok = test_backend_set_rows(backend, name, GGML_TYPE_TURBO3_0) && ok;
        ok = test_backend_set_rows(backend, name, GGML_TYPE_TURBO4_0) && ok;
        ok = test_backend_set_rows(backend, name, GGML_TYPE_MXFP4) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_F16) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_F16, GGML_TYPE_TURBO3_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_F16) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_F16, GGML_TYPE_TURBO4_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_MXFP4, GGML_TYPE_F16) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_F16, GGML_TYPE_MXFP4) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_MXFP4, GGML_TYPE_TURBO3_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_MXFP4) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_MXFP4, GGML_TYPE_TURBO4_0) && ok;
        ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_MXFP4) && ok;
        if (std::strcmp(prefix, "CUDA") == 0) {
            ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 512) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 512) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 512, 16384) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 512, 16384) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, 512) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, 512, 16384) && ok;
        }
        ggml_backend_free(backend);
        if (std::strcmp(prefix, "CUDA") == 0) {
            break;
        }
    }
    if (!found) {
        std::printf("%s backend not available: skipped\n", prefix);
        return true;
    }
    return ok;
}

int main() {
    ggml_backend_load_all();
    bool ok = true;
    ok = test_wht_round_trip() && ok;
    ok = test_codec(GGML_TYPE_TURBO3_0, 0.94, 0.01) && ok;
    ok = test_codec(GGML_TYPE_TURBO4_0, 0.98, 0.01) && ok;
    ok = test_codec(GGML_TYPE_MXFP4, 0.98, 0.10, false) && ok;
    ok = test_normalized_hadamard_attention() && ok;
    ok = test_transformed_attention(GGML_TYPE_TURBO3_0) && ok;
    ok = test_transformed_attention(GGML_TYPE_TURBO4_0) && ok;
    ok = test_packed_size() && ok;
    ok = test_backend_kind("CPU") && ok;
    ok = test_backend_kind("CUDA") && ok;
    ok = test_backend_kind("Vulkan") && ok;

    if (ggml_row_size(GGML_TYPE_TURBO3_0, 128) != 50 || ggml_row_size(GGML_TYPE_TURBO4_0, 128) != 66 || ggml_row_size(GGML_TYPE_MXFP4, 32) != 17) {
        std::fprintf(stderr, "KV packed size mismatch\n");
        ok = false;
    }

    return ok ? 0 : 1;
}
