#include "svd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#if defined(DEN2MOEE_USE_OPENMP)
#include <omp.h>
#endif

#if defined(DEN2MOEE_USE_LAPACK)
extern "C" {
void sgesdd_(const char * jobz, const int * m, const int * n, float * a, const int * lda,
             float * s, float * u, const int * ldu, float * vt, const int * ldvt,
             float * work, const int * lwork, int * iwork, int * info);
}
#endif

namespace den2moee {

static void require_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("SVD input contains NaN or Inf");
    }
}

static float norm2(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) sum += static_cast<double>(value) * value;
    return static_cast<float>(std::sqrt(sum));
}

static void normalize(std::vector<float> & values) {
    const float norm = norm2(values);
    if (norm > 0.0f) {
        for (float & value : values) value /= norm;
    }
}

static SvdFactor fallback_factor(const std::vector<float> & matrix, int rows, int cols,
                                 int logical_rank, int storage_rank, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int q = logical_rank;
    std::vector<float> omega(static_cast<size_t>(cols) * q);
    for (float & value : omega) value = normal(rng);

    std::vector<float> y(static_cast<size_t>(rows) * q, 0.0f);
#if defined(DEN2MOEE_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const float value = matrix[row * cols + col];
            for (int r = 0; r < q; ++r) y[row * q + r] += value * omega[col * q + r];
        }
    }

    auto orthogonalize_columns = [](std::vector<float> & values, int n_rows, int n_cols) {
        for (int col = 0; col < n_cols; ++col) {
            for (int previous = 0; previous < col; ++previous) {
                float dot = 0.0f;
                for (int row = 0; row < n_rows; ++row) dot += values[row * n_cols + col] * values[row * n_cols + previous];
                for (int row = 0; row < n_rows; ++row) values[row * n_cols + col] -= dot * values[row * n_cols + previous];
            }
            float norm = 0.0f;
            for (int row = 0; row < n_rows; ++row) norm += values[row * n_cols + col] * values[row * n_cols + col];
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                for (int row = 0; row < n_rows; ++row) values[row * n_cols + col] /= norm;
            }
        }
    };

    orthogonalize_columns(y, rows, q);
    std::vector<float> z(static_cast<size_t>(cols) * q, 0.0f);
    std::vector<float> next_y(static_cast<size_t>(rows) * q, 0.0f);
    for (int power = 0; power < 1; ++power) {
        std::fill(z.begin(), z.end(), 0.0f);
#if defined(DEN2MOEE_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                const float value = matrix[row * cols + col];
                for (int r = 0; r < q; ++r) z[col * q + r] += value * y[row * q + r];
            }
        }
        std::fill(next_y.begin(), next_y.end(), 0.0f);
#if defined(DEN2MOEE_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const float value = matrix[row * cols + col];
                for (int r = 0; r < q; ++r) next_y[row * q + r] += value * z[col * q + r];
            }
        }
        y.swap(next_y);
        orthogonalize_columns(y, rows, q);
    }

    std::vector<float> b(static_cast<size_t>(q) * cols, 0.0f);
#if defined(DEN2MOEE_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < q; ++r) {
        for (int row = 0; row < rows; ++row) {
            const float value = y[row * q + r];
            for (int col = 0; col < cols; ++col) b[r * cols + col] += value * matrix[row * cols + col];
        }
    }

    std::vector<float> gram(static_cast<size_t>(q) * q, 0.0f);
#if defined(DEN2MOEE_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j <= i; ++j) {
            float value = 0.0f;
            for (int col = 0; col < cols; ++col) value += b[i * cols + col] * b[j * cols + col];
            gram[i * q + j] = value;
        }
    }
    for (int i = 0; i < q; ++i) {
        for (int j = 0; j < i; ++j) gram[j * q + i] = gram[i * q + j];
    }

    std::vector<float> eigenvectors(static_cast<size_t>(q) * q, 0.0f);
    std::vector<float> eigenvalues(q, 0.0f);
    for (int component = 0; component < q; ++component) {
        std::vector<float> vector(q);
        for (float & value : vector) value = normal(rng);
        for (int iteration = 0; iteration < 4; ++iteration) {
            std::vector<float> next(q, 0.0f);
#if defined(DEN2MOEE_USE_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < q; ++i) {
                for (int j = 0; j < q; ++j) next[i] += gram[i * q + j] * vector[j];
            }
            for (int previous = 0; previous < component; ++previous) {
                float dot = 0.0f;
                for (int i = 0; i < q; ++i) dot += next[i] * eigenvectors[previous * q + i];
                for (int i = 0; i < q; ++i) next[i] -= dot * eigenvectors[previous * q + i];
            }
            normalize(next);
            vector.swap(next);
        }
        float eigenvalue = 0.0f;
        for (int i = 0; i < q; ++i) {
            float row_value = 0.0f;
            for (int j = 0; j < q; ++j) row_value += gram[i * q + j] * vector[j];
            eigenvalue += vector[i] * row_value;
            eigenvectors[component * q + i] = vector[i];
        }
        eigenvalues[component] = std::max(0.0f, eigenvalue);
    }

    SvdFactor result;
    result.rows = rows;
    result.cols = cols;
    result.logical_rank = logical_rank;
    result.storage_rank = storage_rank;
    result.u.assign(static_cast<size_t>(rows) * storage_rank, 0.0f);
    result.sigma.assign(storage_rank, 0.0f);
    result.v.assign(static_cast<size_t>(cols) * storage_rank, 0.0f);

    for (int r = 0; r < logical_rank; ++r) {
        const float sigma = std::sqrt(eigenvalues[r]);
        result.sigma[r] = sigma;
        for (int row = 0; row < rows; ++row) {
            float value = 0.0f;
            for (int i = 0; i < q; ++i) value += y[row * q + i] * eigenvectors[r * q + i];
            result.u[row * storage_rank + r] = value;
        }
        if (sigma > 0.0f) {
            for (int col = 0; col < cols; ++col) {
                float value = 0.0f;
                for (int i = 0; i < q; ++i) value += b[i * cols + col] * eigenvectors[r * q + i];
                result.v[col * storage_rank + r] = value / sigma;
            }
        }
    }
    return result;
}

SvdFactor truncated_svd(const std::vector<float> & matrix, int rows, int cols,
                        int logical_rank, int storage_rank, uint64_t seed) {
    if (rows <= 0 || cols <= 0 || logical_rank <= 0 || storage_rank < logical_rank ||
        matrix.size() != static_cast<size_t>(rows) * cols) {
        throw std::invalid_argument("invalid SVD dimensions");
    }
    if (logical_rank > std::min(rows, cols)) throw std::invalid_argument("logical SVD rank is too large");
    require_finite(matrix);

    SvdFactor result;
#if defined(DEN2MOEE_USE_LAPACK) && 0
    const int thin_rank = std::min(rows, cols);
    const int lda = rows;
    const int ldu = rows;
    const int ldvt = thin_rank;
    std::vector<float> a(static_cast<size_t>(rows) * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) a[col * rows + row] = matrix[row * cols + col];
    }
    std::vector<float> singular_values(thin_rank);
    std::vector<float> u(static_cast<size_t>(rows) * thin_rank);
    std::vector<float> vt(static_cast<size_t>(thin_rank) * cols);
    std::vector<int> iwork(8 * thin_rank);
    int lwork = -1;
    float query = 0.0f;
    int info = 0;
    const char jobz = 'S';
    sgesdd_(&jobz, &rows, &cols, a.data(), &lda, singular_values.data(), u.data(), &ldu,
            vt.data(), &ldvt, &query, &lwork, iwork.data(), &info);
    if (info != 0 || !std::isfinite(query)) throw std::runtime_error("LAPACK SVD workspace query failed");
    lwork = static_cast<int>(query);
    std::vector<float> work(std::max(1, lwork));
    sgesdd_(&jobz, &rows, &cols, a.data(), &lda, singular_values.data(), u.data(), &ldu,
            vt.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
    if (info != 0) throw std::runtime_error("LAPACK SVD failed");

    result.rows = rows;
    result.cols = cols;
    result.logical_rank = logical_rank;
    result.storage_rank = storage_rank;
    result.u.assign(static_cast<size_t>(rows) * storage_rank, 0.0f);
    result.sigma.assign(storage_rank, 0.0f);
    result.v.assign(static_cast<size_t>(cols) * storage_rank, 0.0f);
    for (int r = 0; r < logical_rank; ++r) {
        result.sigma[r] = singular_values[r];
        for (int row = 0; row < rows; ++row) result.u[row * storage_rank + r] = u[r * rows + row];
        for (int col = 0; col < cols; ++col) result.v[col * storage_rank + r] = vt[r * cols + col];
    }
#else
    result = fallback_factor(matrix, rows, cols, logical_rank, storage_rank, seed);
#endif

    double original_norm = 0.0;
    double error_norm = 0.0;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float reconstructed = 0.0f;
            for (int r = 0; r < logical_rank; ++r) {
                reconstructed += result.u[row * storage_rank + r] * result.sigma[r] * result.v[col * storage_rank + r];
            }
            const float original = matrix[row * cols + col];
            original_norm += static_cast<double>(original) * original;
            const float error = original - reconstructed;
            error_norm += static_cast<double>(error) * error;
        }
    }
    result.relative_error = original_norm > 0.0 ? static_cast<float>(std::sqrt(error_norm / original_norm)) : 0.0f;
    if (!std::isfinite(result.relative_error)) throw std::runtime_error("SVD reconstruction is not finite");
    return result;
}

} // namespace den2moee
