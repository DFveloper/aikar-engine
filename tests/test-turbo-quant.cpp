#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "ggml-turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

static bool test_q8_kv_vs_q4_hadamard() {
    constexpr int d = 256;
    const ggml_type_traits * q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_KV);
    const ggml_type_traits * q4_traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    std::vector<float> source;
    std::vector<float> q8_all;
    std::vector<float> q4h_all;

    for (int id = 0; id < 5; ++id) {
        std::vector<float> input = make_distribution(id);
        std::vector<float> q4_input = input;
        std::vector<float> q8_out(d);
        std::vector<float> q4_out(d);
        std::vector<uint8_t> q8_packed(ggml_row_size(GGML_TYPE_Q8_KV, d));
        std::vector<uint8_t> q4_packed(ggml_row_size(GGML_TYPE_Q4_0, d));

        normalized_hadamard_ref(q4_input.data(), d, d);
        q8_traits->from_float_ref(input.data(), q8_packed.data(), d);
        q8_traits->to_float(q8_packed.data(), q8_out.data(), d);
        q4_traits->from_float_ref(q4_input.data(), q4_packed.data(), d);
        q4_traits->to_float(q4_packed.data(), q4_out.data(), d);
        normalized_hadamard_ref(q4_out.data(), d, d);

        source.insert(source.end(), input.begin(), input.end());
        q8_all.insert(q8_all.end(), q8_out.begin(), q8_out.end());
        q4h_all.insert(q4h_all.end(), q4_out.begin(), q4_out.end());
    }

    const metrics q8 = measure(source, q8_all);
    const metrics q4h = measure(source, q4h_all);
    const double dot_ref = dot(source.data(), source.data() + d, d);
    const double q8_dot_error = std::abs(dot(q8_all.data(), q8_all.data() + d, d) - dot_ref)/std::max(1.0, std::abs(dot_ref));
    const double q4h_dot_error = std::abs(dot(q4h_all.data(), q4h_all.data() + d, d) - dot_ref)/std::max(1.0, std::abs(dot_ref));
    const bool ok = q8.mse < q4h.mse && q8.cosine > q4h.cosine && q8_dot_error < q4h_dot_error;
    std::printf("Q8_KV G64 vs Q4_0 Hadamard: mse=%g/%g cosine=%.7f/%.7f dot_rel=%g/%g bytes=%zu/%zu %s\n",
        q8.mse, q4h.mse, q8.cosine, q4h.cosine, q8_dot_error, q4h_dot_error,
        ggml_row_size(GGML_TYPE_Q8_KV, d), ggml_row_size(GGML_TYPE_Q4_0, d), ok ? "ok" : "FAILED");
    return ok;
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
        ggml_backend_t backend, const char * label, ggml_type type_k, ggml_type type_v, int d = 256, int kv = 256, int nq = 1) {
    constexpr int q_heads = 16;
    constexpr int kv_heads = 2;

    ggml_init_params params = { 8*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nq, q_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, type_k, d, kv, kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, type_v, d, kv, kv_heads, 1);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f/std::sqrt((float) d), 0.0f, 0.0f);
    const bool turbo_k = type_k == GGML_TYPE_TURBO3_0 || type_k == GGML_TYPE_TURBO4_0;
    const bool turbo_v = type_v == GGML_TYPE_TURBO3_0 || type_v == GGML_TYPE_TURBO4_0;
    ggml_tensor * out = turbo_v ? ggml_turbo_wht(ctx, attn, true) : attn;

    std::mt19937 rng(12000 + 10*type_k + type_v);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    std::vector<float> q_host(d*nq*q_heads);
    std::vector<float> k_host(d*kv*kv_heads);
    std::vector<float> v_host(d*kv*kv_heads);
    for (float & value : q_host) value = normal(rng);
    for (float & value : k_host) value = normal(rng);
    for (float & value : v_host) value = normal(rng);

    if (turbo_k) {
        for (int row = 0; row < nq*q_heads; ++row) {
            ggml_turbo_wht_forward_f32(q_host.data() + row*d, d);
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

    std::vector<float> expected(d*nq*q_heads);
    std::vector<float> scores(kv);
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh/(q_heads/kv_heads);
        for (int qt = 0; qt < nq; ++qt) {
            const int qrow = qh*nq + qt;
            const int orow = qt*q_heads + qh;
            float max_score = -INFINITY;
            for (int token = 0; token < kv; ++token) {
                scores[token] = (float) dot(q_host.data() + qrow*d, k_dequant.data() + (kvh*kv + token)*d, d)/std::sqrt((float) d);
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
                    expected[orow*d + i] += weight*value[i];
                }
            }
            if (turbo_v) {
                ggml_turbo_wht_inverse_f32(expected.data() + orow*d, d);
            }
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
        if (turbo_k && std::strncmp(label, "CUDA", 4) == 0) {
            ok = ok && result.max_error < 2e-6;
        }
        std::printf("%s attention D%d/KV%d/Q%d %s/%s: mse=%g max=%g cosine=%.7f %s\n",
            label, d, kv, nq, ggml_type_name(type_k), ggml_type_name(type_v), result.mse, result.max_error, result.cosine,
            ok ? "ok" : "FAILED");
    }

    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(ctx);
    return ok;
}

static void reference_google_turboquant(const float * input, uint8_t * packed, float * decoded,
        const float * parameters, int head_dim, int total_bits, bool key) {
    const int base_bits = total_bits - int(key);
    const int header_bytes = key ? 8 : 4;
    const int book_offset = key ? (total_bits == 3 ? 0 : 4) : (total_bits == 3 ? 12 : 20);
    const float * book = parameters + 3*head_dim*head_dim + book_offset;
    const float * rotation = parameters + (key ? 0 : head_dim*head_dim);
    const float * projection = parameters + 2*head_dim*head_dim;
    const int row_bytes = header_bytes + head_dim*total_bits/8;
    std::fill(packed, packed + row_bytes, uint8_t(0));
    std::vector<long double> reconstruction(head_dim, 0.0L), residual(head_dim);
    long double norm_squared = 0.0L;
    for (int component = 0; component < head_dim; ++component) norm_squared += (long double) input[component]*input[component];
    const long double norm = std::sqrt(norm_squared);
    const float stored_norm = float(norm);
    std::memcpy(packed, &stored_norm, sizeof(stored_norm));
    if (norm == 0.0L) {
        std::fill(decoded, decoded + head_dim, 0.0f);
        return;
    }
    for (int row = 0; row < head_dim; ++row) {
        long double rotated = 0.0L;
        for (int component = 0; component < head_dim; ++component) rotated += (long double) rotation[row*head_dim + component]*input[component]/norm;
        int nearest = 0;
        for (int candidate = 1; candidate < (1 << base_bits); ++candidate) {
            if (std::abs(rotated - book[candidate]) < std::abs(rotated - book[nearest])) nearest = candidate;
        }
        for (int bit = 0; bit < base_bits; ++bit) {
            const int position = row*base_bits + bit;
            packed[header_bytes + position/8] |= uint8_t(((nearest >> bit) & 1) << (position % 8));
        }
        for (int component = 0; component < head_dim; ++component) reconstruction[component] += norm*book[nearest]*rotation[row*head_dim + component];
    }
    long double residual_squared = 0.0L;
    for (int component = 0; component < head_dim; ++component) {
        residual[component] = input[component] - reconstruction[component];
        residual_squared += residual[component]*residual[component];
        decoded[component] = float(reconstruction[component]*stored_norm/norm);
    }
    if (key) {
        const float residual_norm = float(std::sqrt(residual_squared));
        std::memcpy(packed + 4, &residual_norm, sizeof(residual_norm));
        std::vector<long double> correction(head_dim, 0.0L);
        for (int row = 0; row < head_dim; ++row) {
            long double product = 0.0L;
            for (int component = 0; component < head_dim; ++component) product += projection[row*head_dim + component]*residual[component];
            const int sign = product >= 0.0L ? 1 : -1;
            if (sign > 0) packed[header_bytes + head_dim*base_bits/8 + row/8] |= uint8_t(1 << (row % 8));
            for (int component = 0; component < head_dim; ++component) correction[component] += sign*(long double) projection[row*head_dim + component];
        }
        for (int component = 0; component < head_dim; ++component) {
            decoded[component] = float(reconstruction[component]*stored_norm/norm + residual_norm*std::sqrt(std::acos(-1.0L)/2.0L)/head_dim*correction[component]);
        }
    }
}

static bool test_google_turboquant_backend(ggml_backend_t backend, const char * label, int head_dim, int bits_k, int bits_v,
        int cached_tokens = 17, int query_tokens = 3, bool noncontiguous_query = false, bool use_sinks = true) {
    constexpr int kv_heads = 2;
    constexpr int query_heads = 4;
    constexpr int streams = 2;
    const size_t row_k = ggml_turboquant_row_size(head_dim, bits_k, true);
    const size_t row_v = ggml_turboquant_row_size(head_dim, bits_v, false);
    ggml_init_params init { 8*1024*1024, nullptr, true };
    ggml_context * context = ggml_init(init);
    ggml_tensor * query = noncontiguous_query
            ? ggml_permute(context, ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, query_heads, query_tokens, streams), 0, 2, 1, 3)
            : ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, query_tokens, query_heads, streams);
    ggml_tensor * input_k = ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, kv_heads, cached_tokens, streams);
    ggml_tensor * input_v = ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, kv_heads, cached_tokens, streams);
    ggml_tensor * cache_k = ggml_new_tensor_4d(context, GGML_TYPE_I8, row_k, kv_heads, cached_tokens, streams);
    ggml_tensor * cache_v = ggml_new_tensor_4d(context, GGML_TYPE_I8, row_v, kv_heads, cached_tokens, streams);
    ggml_tensor * indices = ggml_new_tensor_2d(context, GGML_TYPE_I64, cached_tokens, streams);
    ggml_tensor * parameters = ggml_new_tensor_1d(context, GGML_TYPE_F32, ggml_turboquant_params_size(head_dim));
    ggml_tensor * mask = ggml_new_tensor_4d(context, GGML_TYPE_F32, cached_tokens, query_tokens, 1, streams);
    ggml_tensor * sinks = ggml_new_tensor_1d(context, GGML_TYPE_F32, query_heads);
    ggml_tensor * write_k = ggml_turboquant_pack(context, input_k, indices, cache_k, parameters, head_dim, bits_k, true);
    ggml_tensor * write_v = ggml_turboquant_pack(context, input_v, indices, cache_v, parameters, head_dim, bits_v, false);
    ggml_tensor * output = ggml_turboquant_attn(context, query, write_k, write_v, mask, use_sinks ? sinks : nullptr, parameters,
        head_dim, bits_k, bits_v, 0.5f/std::sqrt(float(head_dim)), 0.0f, 2.0f, cached_tokens);
    if (write_k == nullptr || write_v == nullptr || output == nullptr) {
        std::printf("%s Google TurboQuant graph construction FAILED\n", label);
        ggml_free(context);
        return false;
    }
    std::vector<float> parameters_host(ggml_turboquant_params_size(head_dim));
    ggml_turboquant_params_init(parameters_host.data(), head_dim, 42);
    std::vector<float> query_host(ggml_nelements(query)), key_host(ggml_nelements(input_k)), value_host(ggml_nelements(input_v));
    std::vector<float> mask_host(ggml_nelements(mask)), sinks_host(query_heads, -0.75f);
    std::vector<int64_t> indices_host(ggml_nelements(indices));
    std::mt19937 generator(1267);
    std::normal_distribution<float> normal(0.0f, 0.5f);
    for (float & value : query_host) value = normal(generator);
    for (float & value : key_host) value = normal(generator);
    for (float & value : value_host) value = normal(generator);
    for (int stream = 0; stream < streams; ++stream) {
        for (int token = 0; token < cached_tokens; ++token) {
            indices_host[stream*cached_tokens + token] = cached_tokens - 1 - token;
        }
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            for (int token = 0; token < cached_tokens; ++token) {
                mask_host[(stream*query_tokens + query_token)*cached_tokens + token] = token > cached_tokens - 5 + query_token || query_token == query_tokens - 1 ? -INFINITY : -0.01f*token;
            }
        }
    }
    std::vector<uint8_t> packed_keys(ggml_nbytes(cache_k)), packed_values(ggml_nbytes(cache_v));
    std::vector<float> decoded_keys(key_host.size()), decoded_values(value_host.size());
    for (int stream = 0; stream < streams; ++stream) {
        for (int token = 0; token < cached_tokens; ++token) {
            for (int head = 0; head < kv_heads; ++head) {
                const int source_row = (stream*cached_tokens + token)*kv_heads + head;
                const int cache_row = (stream*cached_tokens + cached_tokens - 1 - token)*kv_heads + head;
                reference_google_turboquant(key_host.data() + source_row*head_dim, packed_keys.data() + cache_row*row_k,
                        decoded_keys.data() + cache_row*head_dim, parameters_host.data(), head_dim, bits_k, true);
                reference_google_turboquant(value_host.data() + source_row*head_dim, packed_values.data() + cache_row*row_v,
                        decoded_values.data() + cache_row*head_dim, parameters_host.data(), head_dim, bits_v, false);
            }
        }
    }
    std::vector<float> expected(ggml_nelements(output), 0.0f);
    for (int stream = 0; stream < streams; ++stream) {
        for (int query_token = 0; query_token < query_tokens; ++query_token) {
            for (int head = 0; head < query_heads; ++head) {
                const int query_row_index = noncontiguous_query ? (stream*query_tokens + query_token)*query_heads + head
                        : (stream*query_heads + head)*query_tokens + query_token;
                const float * query_row = query_host.data() + query_row_index*head_dim;
                std::vector<double> scores(cached_tokens);
                double maximum = use_sinks ? sinks_host[head] : -INFINITY;
                for (int token = 0; token < cached_tokens; ++token) {
                    const float * key_row = decoded_keys.data() + ((stream*cached_tokens + token)*kv_heads + head/(query_heads/kv_heads))*head_dim;
                    double product = 0.0;
                    for (int component = 0; component < head_dim; ++component) product += double(query_row[component])*key_row[component];
                    scores[token] = 2.0*std::tanh(product*0.25/std::sqrt(double(head_dim))) + mask_host[(stream*query_tokens + query_token)*cached_tokens + token];
                    maximum = std::max(maximum, scores[token]);
                }
                double denominator = use_sinks ? std::exp(double(sinks_host[head]) - maximum) : 0.0;
                for (double & score : scores) {
                    score = score == -INFINITY ? 0.0 : std::exp(score - maximum);
                    denominator += score;
                }
                float * result = expected.data() + ((stream*query_tokens + query_token)*query_heads + head)*head_dim;
                for (int token = 0; token < cached_tokens; ++token) {
                    const float * value_row = decoded_values.data() + ((stream*cached_tokens + token)*kv_heads + head/(query_heads/kv_heads))*head_dim;
                    for (int component = 0; component < head_dim; ++component) result[component] += denominator > 0.0 ? float(scores[token]/denominator*value_row[component]) : 0.0f;
                }
            }
        }
    }
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    bool ok = buffer != nullptr && ggml_backend_supports_op(backend, output) && ggml_backend_supports_op(backend, write_k);
    if (ok) {
        ggml_backend_tensor_set(parameters, parameters_host.data(), 0, ggml_nbytes(parameters));
        ggml_backend_tensor_set(query, query_host.data(), 0, ggml_nbytes(query));
        ggml_backend_tensor_set(input_k, key_host.data(), 0, ggml_nbytes(input_k));
        ggml_backend_tensor_set(input_v, value_host.data(), 0, ggml_nbytes(input_v));
        ggml_backend_tensor_set(indices, indices_host.data(), 0, ggml_nbytes(indices));
        ggml_backend_tensor_set(mask, mask_host.data(), 0, ggml_nbytes(mask));
        ggml_backend_tensor_set(sinks, sinks_host.data(), 0, ggml_nbytes(sinks));
        ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        std::vector<uint8_t> actual_keys(packed_keys.size()), actual_values(packed_values.size());
        ggml_backend_tensor_get(cache_k, actual_keys.data(), 0, actual_keys.size());
        ggml_backend_tensor_get(cache_v, actual_values.data(), 0, actual_values.size());
        ok = ok && actual_keys == packed_keys && actual_values == packed_values;
        const metrics result = measure(expected, actual);
        ok = result.max_error < 3e-6 && ok;
        std::printf("%s Google TurboQuant D%d K%d/V%d: max=%g cosine=%.9f %s\n", label, head_dim, bits_k, bits_v, result.max_error, result.cosine, ok ? "ok" : "FAILED");
    } else {
        std::printf("%s Google TurboQuant backend support FAILED\n", label);
    }
    if (buffer) ggml_backend_buffer_free(buffer);
    ggml_free(context);
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
        if (std::strcmp(prefix, "CPU") == 0 || std::strcmp(prefix, "CUDA") == 0) {
            ok = test_google_turboquant_backend(backend, name, 256, 3, 3) && ok;
            ok = test_google_turboquant_backend(backend, name, 512, 4, 4) && ok;
            ok = test_google_turboquant_backend(backend, name, 256, 4, 3, 33, 5, true, false) && ok;
            ok = test_google_turboquant_backend(backend, name, 512, 3, 4, 257, 2, true, false) && ok;
            ok = test_google_turboquant_backend(backend, name, 512, 4, 4, 1025, 2, true, false) && ok;
            ok = test_google_turboquant_backend(backend, name, 256, 3, 3, 513, 3, false, true) && ok;
        }
        ok = test_backend_set_rows(backend, name, GGML_TYPE_TURBO3_0) && ok;
        ok = test_backend_set_rows(backend, name, GGML_TYPE_TURBO4_0) && ok;
        ok = test_backend_set_rows(backend, name, GGML_TYPE_MXFP4) && ok;
        const bool supports_q8_kv = std::strcmp(prefix, "CPU") == 0 || std::strcmp(prefix, "CUDA") == 0;
        if (supports_q8_kv) {
            ok = test_backend_set_rows(backend, name, GGML_TYPE_Q8_KV) && ok;
        }
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
        if (supports_q8_kv) {
            ok = test_backend_attention(backend, name, GGML_TYPE_Q8_KV, GGML_TYPE_Q8_KV, 256) && ok;
        }
        if (std::strcmp(prefix, "CUDA") == 0) {
            ok = test_backend_attention(backend, name, GGML_TYPE_Q8_KV, GGML_TYPE_Q8_KV, 512) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, 256, 256, 32) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_Q8_KV, GGML_TYPE_Q8_KV, 256, 256, 32) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0, 256, 256, 32) && ok;
            ok = test_backend_attention(backend, name, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0, 512, 256, 32) && ok;
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

static bool test_google_turboquant() {
    bool ok = true;
    ok = ggml_turboquant_params_size(128) == 0 && ok;
    ok = ggml_turboquant_row_size(256, 2, true) == 0 && ok;
    for (const int head_dim : {256, 512}) {
        std::vector<float> parameters(ggml_turboquant_params_size(head_dim));
        if (!ggml_turboquant_params_init(parameters.data(), head_dim, 0x12345678)) {
            std::printf("Google TurboQuant D%d parameter generation FAILED\n", head_dim);
            return false;
        }
        double orthogonality_error = 0.0;
        for (const auto matrix_kind : {GGML_TURBOQUANT_ROTATION_K, GGML_TURBOQUANT_ROTATION_V}) {
            const float * rotation = ggml_turboquant_matrix(parameters.data(), head_dim, matrix_kind);
            for (int row = 0; row < head_dim; ++row) {
                for (int other = 0; other < head_dim; ++other) {
                    double product = 0.0;
                    for (int column = 0; column < head_dim; ++column) {
                        product += double(rotation[row*head_dim + column])*rotation[other*head_dim + column];
                    }
                    orthogonality_error = std::max(orthogonality_error, std::abs(product - double(row == other)));
                }
            }
        }
        ok = orthogonality_error < 2e-7 && ok;
        std::mt19937 generator(83);
        std::normal_distribution<float> normal;
        std::vector<float> input(head_dim), query(head_dim), output(head_dim), rotated_query(head_dim), projected_query(head_dim);
        for (int component = 0; component < head_dim; ++component) {
            input[component] = normal(generator);
            query[component] = normal(generator);
        }
        for (int row = 0; row < head_dim; ++row) {
            double rotated = 0.0, projected = 0.0;
            for (int column = 0; column < head_dim; ++column) {
                rotated += double(ggml_turboquant_matrix(parameters.data(), head_dim, GGML_TURBOQUANT_ROTATION_K)[row*head_dim + column])*query[column];
                projected += double(ggml_turboquant_matrix(parameters.data(), head_dim, GGML_TURBOQUANT_PROJECTION)[row*head_dim + column])*query[column];
            }
            rotated_query[row] = float(rotated);
            projected_query[row] = float(projected);
        }
        double input_norm_sq = 0.0;
        for (const float value : input) {
            input_norm_sq += double(value)*value;
        }
        for (const int total_bits : {3, 4}) {
            for (const bool key : {false, true}) {
                const size_t row_size = ggml_turboquant_row_size(head_dim, total_bits, key);
                ok = row_size == size_t((key ? 8 : 4) + head_dim*total_bits/8) && ok;
                std::vector<uint8_t> packed(row_size + 16, 0xa5);
                ok = ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, total_bits, key) && ok;
                float stored_norm;
                std::memcpy(&stored_norm, packed.data(), sizeof(stored_norm));
                ok = std::abs(stored_norm/std::sqrt(input_norm_sq) - 1.0) < 1e-7 && ok;
                ok = std::all_of(packed.begin() + row_size, packed.end(), [](uint8_t value) { return value == 0xa5; }) && ok;
                ggml_turboquant_unpack_row(packed.data(), output.data(), parameters.data(), head_dim, total_bits, key);
                const metrics result = measure(input, output);
                if (!key) {
                    const double relative_mse = result.mse*head_dim/input_norm_sq;
                    ok = relative_mse < (total_bits == 3 ? 0.045 : 0.014) && ok;
                } else {
                    double oracle_dot = 0.0;
                    for (int component = 0; component < head_dim; ++component) {
                        oracle_dot += double(query[component])*output[component];
                    }
                    const double packed_dot = ggml_turboquant_dot_row(packed.data(), rotated_query.data(), projected_query.data(), parameters.data(), head_dim, total_bits);
                    ok = std::abs(packed_dot - oracle_dot) < 2e-5*std::max(1.0, std::abs(oracle_dot)) && ok;
                }
                std::printf("Google TurboQuant D%d %s%d: relative_mse=%g norm=%g %s\n", head_dim, key ? "K" : "V", total_bits,
                    result.mse*head_dim/input_norm_sq, stored_norm, ok ? "ok" : "FAILED");
                std::fill(input.begin(), input.end(), 0.0f);
                ok = ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, total_bits, key) && ok;
                ok = std::all_of(packed.begin(), packed.begin() + row_size, [](uint8_t value) { return value == 0; }) && ok;
                ggml_turboquant_unpack_row(packed.data(), output.data(), parameters.data(), head_dim, total_bits, key);
                ok = std::all_of(output.begin(), output.end(), [](float value) { return value == 0.0f; }) && ok;
                input[0] = std::numeric_limits<float>::infinity();
                ok = !ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, total_bits, key) && ok;
                input[0] = std::numeric_limits<float>::quiet_NaN();
                ok = !ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, total_bits, key) && ok;
                for (float & value : input) {
                    value = normal(generator);
                }
                input_norm_sq = 0.0;
                for (const float value : input) {
                    input_norm_sq += double(value)*value;
                }
            }
        }
        std::printf("Google TurboQuant D%d orthogonality=%g %s\n", head_dim, orthogonality_error, ok ? "ok" : "FAILED");
        for (const int total_bits : {3, 4}) {
            for (const bool key : {false, true}) {
                const float * codebook = ggml_turboquant_codebook(parameters.data(), head_dim, total_bits, key);
                const int count = 1 << (total_bits - int(key));
                for (int center = 0; center < count; ++center) {
                    const double lower = center == 0 ? -1.0 : 0.5*(double(codebook[center - 1]) + codebook[center]);
                    const double upper = center + 1 == count ? 1.0 : 0.5*(double(codebook[center]) + codebook[center + 1]);
                    constexpr int intervals = 8192;
                    double mass = 0.0, moment = 0.0;
                    for (int point = 0; point <= intervals; ++point) {
                        const double coordinate = lower + (upper - lower)*point/intervals;
                        const double density = std::abs(coordinate) >= 1.0 ? 0.0 : std::pow(1.0 - coordinate*coordinate, 0.5*(head_dim - 3));
                        const int weight = point == 0 || point == intervals ? 1 : (point % 2 == 0 ? 2 : 4);
                        mass += weight*density;
                        moment += weight*density*coordinate;
                    }
                    ok = std::abs(moment/mass - codebook[center]) < 1e-6 && ok;
                }
                std::vector<uint8_t> packed(ggml_turboquant_row_size(head_dim, total_bits, key));
                for (const float magnitude : {1e-20f, 1e20f}) {
                    std::fill(input.begin(), input.end(), 0.0f);
                    input[0] = magnitude;
                    ok = ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, total_bits, key) && ok;
                    ggml_turboquant_unpack_row(packed.data(), output.data(), parameters.data(), head_dim, total_bits, key);
                    ok = std::all_of(output.begin(), output.end(), [](float value) { return std::isfinite(value); }) && ok;
                }
                std::fill(input.begin(), input.end(), std::numeric_limits<float>::max());
                ok = !ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, total_bits, key) && ok;
            }
        }
        if (head_dim == 256) {
            std::fill(input.begin(), input.end(), 0.0f);
            std::fill(query.begin(), query.end(), 0.0f);
            input[0] = query[0] = 1.0f;
            const float * rotation = ggml_turboquant_matrix(parameters.data(), head_dim, GGML_TURBOQUANT_ROTATION_K);
            for (int row = 0; row < head_dim; ++row) rotated_query[row] = rotation[row*head_dim];
            float * projection = parameters.data() + 2*head_dim*head_dim;
            std::vector<uint8_t> packed(ggml_turboquant_row_size(head_dim, 3, true));
            constexpr int samples = 256;
            double sum = 0.0, sum_sq = 0.0;
            for (int sample = 0; sample < samples; ++sample) {
                std::mt19937 sample_generator(3000 + sample);
                for (int index = 0; index < head_dim*head_dim; ++index) projection[index] = normal(sample_generator);
                for (int row = 0; row < head_dim; ++row) projected_query[row] = projection[row*head_dim];
                ok = ggml_turboquant_pack_row(input.data(), packed.data(), parameters.data(), head_dim, 3, true) && ok;
                const double error = ggml_turboquant_dot_row(packed.data(), rotated_query.data(), projected_query.data(), parameters.data(), head_dim, 3) - 1.0;
                sum += error;
                sum_sq += error*error;
            }
            const double mean = sum/samples;
            const double variance = (sum_sq - sum*sum/samples)/(samples - 1);
            const double standard_error = std::sqrt(variance/samples);
            ok = std::abs(mean) < 5.0*standard_error + 1e-6 && variance < 0.003 && ok;
            std::printf("Google QJL D256: bias=%g standard_error=%g variance=%g samples=%d %s\n", mean, standard_error, variance, samples, ok ? "ok" : "FAILED");
        }
    }
    return ok;
}

int main() {
    ggml_backend_load_all();
    bool ok = true;
    ok = test_google_turboquant() && ok;
    ok = test_wht_round_trip() && ok;
    ok = test_codec(GGML_TYPE_TURBO3_0, 0.94, 0.01) && ok;
    ok = test_codec(GGML_TYPE_TURBO4_0, 0.98, 0.01) && ok;
    ok = test_codec(GGML_TYPE_MXFP4, 0.98, 0.10, false) && ok;
    ok = test_normalized_hadamard_attention() && ok;
    ok = test_q8_kv_vs_q4_hadamard() && ok;
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
