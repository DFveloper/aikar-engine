#include "ggml-turboquant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

size_t ggml_turboquant_params_size(int32_t head_dim) {
    return head_dim == 256 || head_dim == 512 ? 3*size_t(head_dim)*head_dim + 64 : 0;
}

const float * ggml_turboquant_matrix(const float * parameters, int32_t head_dim, enum ggml_turboquant_matrix matrix) {
    return parameters + size_t(matrix)*head_dim*head_dim;
}

const float * ggml_turboquant_codebook(const float * parameters, int32_t head_dim, int32_t total_bits, bool key) {
    const int offset = key ? (total_bits == 3 ? 0 : 4) : (total_bits == 3 ? 12 : 20);
    return parameters + 3*size_t(head_dim)*head_dim + offset;
}

size_t ggml_turboquant_row_size(int32_t head_dim, int32_t total_bits, bool key) {
    if (ggml_turboquant_params_size(head_dim) == 0 || (total_bits != 3 && total_bits != 4)) {
        return 0;
    }
    return (key ? 8 : 4) + size_t(head_dim)*total_bits/8;
}

struct turboquant_random {
    uint64_t state;

    double uniform() {
        uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
        value = (value ^ (value >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27))*UINT64_C(0x94d049bb133111eb);
        value ^= value >> 31;
        return (double(value >> 11) + 0.5)*0x1.0p-53;
    }

    double gaussian() {
        const double radius = std::sqrt(-2.0*std::log(uniform()));
        return radius*std::cos(6.2831853071795864769*uniform());
    }
};

static void turboquant_haar(float * output, int head_dim, uint64_t seed) {
    turboquant_random generator { seed };
    std::vector<double> factor(size_t(head_dim)*head_dim);
    std::vector<double> rotation(factor.size(), 0.0);
    std::vector<double> reflector(head_dim);
    for (double & value : factor) {
        value = generator.gaussian();
    }
    for (int diagonal = 0; diagonal < head_dim; ++diagonal) {
        rotation[diagonal*head_dim + diagonal] = 1.0;
    }
    for (int column = 0; column < head_dim; ++column) {
        double norm_sq = 0.0;
        for (int row = column; row < head_dim; ++row) {
            reflector[row] = factor[row*head_dim + column];
            norm_sq += reflector[row]*reflector[row];
        }
        reflector[column] += std::copysign(std::sqrt(norm_sq), reflector[column]);
        double reflector_sq = 0.0;
        for (int row = column; row < head_dim; ++row) {
            reflector_sq += reflector[row]*reflector[row];
        }
        if (reflector_sq == 0.0) {
            continue;
        }
        const double coefficient = 2.0/reflector_sq;
        for (int trailing = column; trailing < head_dim; ++trailing) {
            double product = 0.0;
            for (int row = column; row < head_dim; ++row) {
                product += reflector[row]*factor[row*head_dim + trailing];
            }
            for (int row = column; row < head_dim; ++row) {
                factor[row*head_dim + trailing] -= coefficient*reflector[row]*product;
            }
        }
        for (int row = 0; row < head_dim; ++row) {
            double product = 0.0;
            for (int trailing = column; trailing < head_dim; ++trailing) {
                product += rotation[row*head_dim + trailing]*reflector[trailing];
            }
            for (int trailing = column; trailing < head_dim; ++trailing) {
                rotation[row*head_dim + trailing] -= coefficient*product*reflector[trailing];
            }
        }
    }
    for (int row = 0; row < head_dim; ++row) {
        for (int column = 0; column < head_dim; ++column) {
            output[row*head_dim + column] = float(rotation[row*head_dim + column]*std::copysign(1.0, factor[column*head_dim + column]));
        }
    }
}

static void turboquant_lloyd(float * output, int head_dim, int bits) {
    constexpr int grid_size = 65536;
    constexpr double step = 2.0/grid_size;
    std::vector<double> mass(grid_size + 1, 0.0), moment(grid_size + 1, 0.0);
    double previous_density = 0.0;
    for (int index = 1; index <= grid_size; ++index) {
        const double coordinate = -1.0 + step*index;
        const double density = index == grid_size ? 0.0 : std::exp(0.5*(head_dim - 3)*std::log1p(-coordinate*coordinate));
        mass[index] = mass[index - 1] + step*0.5*(density + previous_density);
        moment[index] = moment[index - 1] + step*0.5*(coordinate*density + (coordinate - step)*previous_density);
        previous_density = density;
    }
    const auto integral = [&](const std::vector<double> & values, double coordinate) {
        const double position = std::clamp((coordinate + 1.0)/step, 0.0, double(grid_size));
        const int index = std::min(int(position), grid_size - 1);
        return values[index] + (position - index)*(values[index + 1] - values[index]);
    };
    const int count = 1 << bits;
    std::vector<double> centroids(count);
    for (int index = 0; index < count; ++index) {
        centroids[index] = (2.0*index - count + 1)*2.5/(count*std::sqrt(double(head_dim)));
    }
    for (int iteration = 0; iteration < 4096; ++iteration) {
        std::vector<double> updated(count);
        for (int index = 0; index < count/2; ++index) {
            const double lower = index == 0 ? -1.0 : 0.5*(centroids[index - 1] + centroids[index]);
            const double upper = 0.5*(centroids[index] + centroids[index + 1]);
            const double probability = integral(mass, upper) - integral(mass, lower);
            updated[index] = (integral(moment, upper) - integral(moment, lower))/probability;
            updated[count - 1 - index] = -updated[index];
        }
        double difference = 0.0;
        for (int index = 0; index < count; ++index) {
            difference = std::max(difference, std::abs(updated[index] - centroids[index]));
        }
        centroids.swap(updated);
        if (difference < 1e-13) {
            break;
        }
    }
    for (int index = 0; index < count; ++index) {
        output[index] = float(centroids[index]);
    }
}

bool ggml_turboquant_params_init(float * parameters, int32_t head_dim, uint64_t seed) {
    if (parameters == nullptr || ggml_turboquant_params_size(head_dim) == 0) {
        return false;
    }
    std::fill(parameters, parameters + ggml_turboquant_params_size(head_dim), 0.0f);
    turboquant_haar(parameters, head_dim, seed ^ UINT64_C(0x4b524f544154494f));
    turboquant_haar(parameters + size_t(head_dim)*head_dim, head_dim, seed ^ UINT64_C(0x56524f544154494f));
    turboquant_random generator { seed ^ UINT64_C(0x514a4c50524f4a45) };
    float * projection = parameters + 2*size_t(head_dim)*head_dim;
    for (size_t index = 0; index < size_t(head_dim)*head_dim; ++index) {
        projection[index] = float(generator.gaussian());
    }
    float * books = parameters + 3*size_t(head_dim)*head_dim;
    turboquant_lloyd(books, head_dim, 2);
    turboquant_lloyd(books + 4, head_dim, 3);
    std::copy(books + 4, books + 12, books + 12);
    turboquant_lloyd(books + 20, head_dim, 4);
    return true;
}

static void turboquant_store_float(uint8_t * output, float value) {
    uint32_t representation;
    std::memcpy(&representation, &value, sizeof(value));
    for (int byte = 0; byte < 4; ++byte) {
        output[byte] = uint8_t(representation >> (8*byte));
    }
}

static float turboquant_load_float(const uint8_t * input) {
    const uint32_t representation = uint32_t(input[0]) | (uint32_t(input[1]) << 8) | (uint32_t(input[2]) << 16) | (uint32_t(input[3]) << 24);
    float value;
    std::memcpy(&value, &representation, sizeof(value));
    return value;
}

static int turboquant_index(const uint8_t * input, int component, int bits) {
    const int bit_offset = component*bits;
    const int shift = bit_offset % 8;
    unsigned value = input[bit_offset/8];
    if (shift + bits > 8) {
        value |= unsigned(input[bit_offset/8 + 1]) << 8;
    }
    return (value >> shift) & ((1 << bits) - 1);
}

bool ggml_turboquant_pack_row(const float * input, uint8_t * packed, const float * parameters, int32_t head_dim, int32_t total_bits, bool key) {
    const size_t row_size = ggml_turboquant_row_size(head_dim, total_bits, key);
    if (row_size == 0 || input == nullptr || packed == nullptr || parameters == nullptr) {
        return false;
    }
    double norm_sq = 0.0;
    for (int component = 0; component < head_dim; ++component) {
        if (!std::isfinite(input[component])) {
            return false;
        }
        norm_sq += double(input[component])*input[component];
    }
    const double original_norm = std::sqrt(norm_sq);
    if (original_norm > std::numeric_limits<float>::max()) {
        return false;
    }
    std::memset(packed, 0, row_size);
    if (original_norm == 0.0) {
        return true;
    }
    turboquant_store_float(packed, float(original_norm));
    const float * rotation = ggml_turboquant_matrix(parameters, head_dim, key ? GGML_TURBOQUANT_ROTATION_K : GGML_TURBOQUANT_ROTATION_V);
    const float * codebook = ggml_turboquant_codebook(parameters, head_dim, total_bits, key);
    const int bits = total_bits - int(key);
    std::vector<double> reconstructed(head_dim), residual(head_dim);
    uint8_t * indices = packed + (key ? 8 : 4);
    for (int row = 0; row < head_dim; ++row) {
        double transformed = 0.0;
        for (int column = 0; column < head_dim; ++column) {
            transformed += double(rotation[row*head_dim + column])*input[column]/original_norm;
        }
        int index = 0;
        while (index + 1 < (1 << bits) && transformed > 0.5*(double(codebook[index]) + codebook[index + 1])) {
            ++index;
        }
        const int bit_offset = row*bits;
        indices[bit_offset/8] |= uint8_t(index << (bit_offset % 8));
        if (bit_offset % 8 + bits > 8) {
            indices[bit_offset/8 + 1] |= uint8_t(index >> (8 - bit_offset % 8));
        }
        reconstructed[row] = original_norm*codebook[index];
    }
    if (!key) {
        return true;
    }
    double residual_sq = 0.0;
    for (int column = 0; column < head_dim; ++column) {
        double base = 0.0;
        for (int row = 0; row < head_dim; ++row) {
            base += double(rotation[row*head_dim + column])*reconstructed[row];
        }
        residual[column] = double(input[column]) - base;
        residual_sq += residual[column]*residual[column];
    }
    const double residual_norm = std::sqrt(residual_sq);
    if (residual_norm > std::numeric_limits<float>::max()) {
        return false;
    }
    turboquant_store_float(packed + 4, float(residual_norm));
    const float * projection = ggml_turboquant_matrix(parameters, head_dim, GGML_TURBOQUANT_PROJECTION);
    uint8_t * signs = indices + size_t(head_dim)*bits/8;
    if (residual_norm != 0.0) {
        for (int row = 0; row < head_dim; ++row) {
            double projected = 0.0;
            for (int column = 0; column < head_dim; ++column) {
                projected += double(projection[row*head_dim + column])*residual[column];
            }
            if (projected >= 0.0) {
                signs[row/8] |= uint8_t(1 << (row % 8));
            }
        }
    }
    return true;
}

void ggml_turboquant_unpack_row(const uint8_t * packed, float * output, const float * parameters, int32_t head_dim, int32_t total_bits, bool key) {
    const float * rotation = ggml_turboquant_matrix(parameters, head_dim, key ? GGML_TURBOQUANT_ROTATION_K : GGML_TURBOQUANT_ROTATION_V);
    const float * projection = ggml_turboquant_matrix(parameters, head_dim, GGML_TURBOQUANT_PROJECTION);
    const float * codebook = ggml_turboquant_codebook(parameters, head_dim, total_bits, key);
    const double norm = turboquant_load_float(packed);
    const double residual_scale = key ? double(turboquant_load_float(packed + 4))*1.2533141373155002512/head_dim : 0.0;
    const int bits = total_bits - int(key);
    const uint8_t * indices = packed + (key ? 8 : 4);
    const uint8_t * signs = indices + size_t(head_dim)*bits/8;
    for (int column = 0; column < head_dim; ++column) {
        double value = 0.0;
        for (int row = 0; row < head_dim; ++row) {
            value += norm*rotation[row*head_dim + column]*codebook[turboquant_index(indices, row, bits)];
            if (key) {
                const int sign = ((signs[row/8] >> (row % 8)) & 1) ? 1 : -1;
                value += residual_scale*projection[row*head_dim + column]*sign;
            }
        }
        output[column] = float(value);
    }
}

float ggml_turboquant_dot_row(const uint8_t * packed, const float * rotated_query, const float * projected_query, const float * parameters, int32_t head_dim, int32_t total_bits) {
    const float * codebook = ggml_turboquant_codebook(parameters, head_dim, total_bits, true);
    const int bits = total_bits - 1;
    const uint8_t * indices = packed + 8;
    const uint8_t * signs = indices + size_t(head_dim)*bits/8;
    double base = 0.0, correction = 0.0;
    for (int component = 0; component < head_dim; ++component) {
        base += double(rotated_query[component])*codebook[turboquant_index(indices, component, bits)];
        const int sign = ((signs[component/8] >> (component % 8)) & 1) ? 1 : -1;
        correction += double(projected_query[component])*sign;
    }
    return float(turboquant_load_float(packed)*base + double(turboquant_load_float(packed + 4))*1.2533141373155002512/head_dim*correction);
}

struct ggml_tensor * ggml_turboquant_pack(struct ggml_context * ctx, struct ggml_tensor * input, struct ggml_tensor * indices,
    struct ggml_tensor * cache, struct ggml_tensor * parameters, int32_t head_dim, int32_t total_bits, bool key) {
    GGML_ASSERT(ggml_turboquant_row_size(head_dim, total_bits, key) > 0);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->ne[0] == head_dim && input->nb[0] == sizeof(float));
    GGML_ASSERT(cache->type == GGML_TYPE_I8 && cache->ne[0] == int64_t(ggml_turboquant_row_size(head_dim, total_bits, key)));
    GGML_ASSERT(cache->nb[0] == 1 && cache->ne[1] == input->ne[1] && cache->ne[3] == input->ne[3]);
    GGML_ASSERT(indices->type == GGML_TYPE_I64 && indices->ne[0] == input->ne[2] && indices->ne[1] == input->ne[3]);
    GGML_ASSERT(indices->ne[2] == 1 && indices->ne[3] == 1);
    GGML_ASSERT(parameters->type == GGML_TYPE_F32 && ggml_is_contiguous(parameters));
    GGML_ASSERT(ggml_nelements(parameters) == int64_t(ggml_turboquant_params_size(head_dim)));
    struct ggml_tensor * result = ggml_view_tensor(ctx, cache);
    result->op = GGML_OP_TURBOQUANT_PACK;
    result->src[0] = input;
    result->src[1] = indices;
    result->src[2] = cache;
    result->src[3] = parameters;
    const ggml_turboquant_op_params options { head_dim, total_bits, total_bits, 0, 0.0f, 0.0f, 0.0f, int32_t(key) };
    std::memcpy(result->op_params, &options, sizeof(options));
    return result;
}

struct ggml_tensor * ggml_turboquant_attn(struct ggml_context * ctx, struct ggml_tensor * query, struct ggml_tensor * cache_k,
    struct ggml_tensor * cache_v, struct ggml_tensor * mask, struct ggml_tensor * sinks, struct ggml_tensor * parameters,
    int32_t head_dim, int32_t bits_k, int32_t bits_v, float scale, float max_bias, float logit_softcap, int32_t n_kv_max) {
    GGML_ASSERT(ggml_turboquant_row_size(head_dim, bits_k, true) > 0 && ggml_turboquant_row_size(head_dim, bits_v, false) > 0);
    GGML_ASSERT(query->type == GGML_TYPE_F32 && query->ne[0] == head_dim && query->nb[0] == sizeof(float));
    GGML_ASSERT(cache_k->type == GGML_TYPE_I8 && cache_v->type == GGML_TYPE_I8);
    GGML_ASSERT(cache_k->ne[0] == int64_t(ggml_turboquant_row_size(head_dim, bits_k, true)));
    GGML_ASSERT(cache_v->ne[0] == int64_t(ggml_turboquant_row_size(head_dim, bits_v, false)));
    GGML_ASSERT(cache_k->nb[0] == 1 && cache_v->nb[0] == 1);
    GGML_ASSERT(cache_k->ne[1] == cache_v->ne[1] && cache_k->ne[2] == cache_v->ne[2] && cache_k->ne[3] == cache_v->ne[3]);
    GGML_ASSERT(cache_k->ne[1] > 0 && query->ne[2] % cache_k->ne[1] == 0);
    GGML_ASSERT(cache_k->ne[3] == query->ne[3]);
    GGML_ASSERT(n_kv_max >= 0 && n_kv_max <= cache_k->ne[2]);
    GGML_ASSERT(parameters->type == GGML_TYPE_F32 && ggml_is_contiguous(parameters));
    GGML_ASSERT(ggml_nelements(parameters) == int64_t(ggml_turboquant_params_size(head_dim)));
    GGML_ASSERT(std::isfinite(scale) && std::isfinite(max_bias) && max_bias >= 0.0f && std::isfinite(logit_softcap) && logit_softcap >= 0.0f);
    if (mask != nullptr) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(mask->ne[0] >= n_kv_max && mask->ne[1] >= query->ne[1]);
        GGML_ASSERT(query->ne[2] % mask->ne[2] == 0 && query->ne[3] % mask->ne[3] == 0);
    }
    if (sinks != nullptr) {
        GGML_ASSERT(sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(sinks) && ggml_nelements(sinks) == query->ne[2]);
    }
    struct ggml_tensor * result = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, query->ne[2], query->ne[1], query->ne[3]);
    result->op = GGML_OP_TURBOQUANT_ATTN;
    result->src[0] = query;
    result->src[1] = cache_k;
    result->src[2] = cache_v;
    result->src[3] = mask;
    result->src[4] = sinks;
    result->src[5] = parameters;
    ggml_turboquant_op_params options { head_dim, bits_k, bits_v, n_kv_max, scale, max_bias, logit_softcap, 0 };
    std::memcpy(result->op_params, &options, sizeof(options));
    return result;
}
