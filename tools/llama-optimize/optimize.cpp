#include "optimize.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>

namespace llama_opt {
static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<double> precision_vjp(const std::vector<double> & p, const std::vector<double> & dp, double tau) {
    require(!p.empty() && p.size() == dp.size() && std::isfinite(tau) && tau > 0, "invalid precision VJP");
    double mean = 0, sum = 0;
    for (size_t k = 0; k < p.size(); ++k) {
        require(std::isfinite(p[k]) && p[k] >= 0 && std::isfinite(dp[k]), "nonfinite precision VJP");
        mean += p[k] * dp[k];
        sum += p[k];
    }
    require(std::abs(sum - 1) < 1e-6, "precision probabilities must sum to one");
    std::vector<double> gradient(p.size());
    for (size_t k = 0; k < p.size(); ++k) {
        gradient[k] = p[k] * (dp[k] - mean) / tau;
    }
    return gradient;
}

gsq_state initialize(const std::vector<float> & w, int64_t columns, int64_t rows) {
    require(columns > 0 && columns % 64 == 0 && rows > 0 && w.size() == size_t(columns * rows),
            "GSQ requires complete 64-element row blocks");
    std::vector<float> scales(w.size() / 64), quantized(w.size());
    for (size_t b = 0; b < scales.size(); ++b) {
        float positive = 0, negative = 0;
        for (size_t j = 0; j < 64; ++j) {
            require(std::isfinite(w[b * 64 + j]), "nonfinite teacher weight");
            positive = std::max(positive, w[b * 64 + j]);
            negative = std::max(negative, -w[b * 64 + j]);
        }
        scales[b] = std::max(1e-8f, std::max(positive, negative / 2));
        for (size_t j = 0; j < 64; ++j) {
            quantized[b * 64 + j] = scales[b] * std::max(-2.f, std::min(1.f, std::round(w[b * 64 + j] / scales[b])));
        }
    }
    return initialize_prior(quantized, scales, columns, rows);
}

gsq_state initialize_prior(const std::vector<float> & w,
                           const std::vector<float> & scales,
                           int64_t                    columns,
                           int64_t                    rows) {
    require(columns > 0 && columns % 64 == 0 && rows > 0 && w.size() == size_t(columns * rows) &&
                scales.size() * 64 == w.size(),
            "invalid GSQ prior shape");
    gsq_state s;
    s.columns = columns;
    s.rows    = rows;
    s.scales  = scales;
    s.logits.resize(w.size() * 4);
    std::mt19937                    rng(42);
    std::normal_distribution<float> gaussian;
    for (size_t b = 0; b < s.scales.size(); ++b) {
        require(std::isfinite(s.scales[b]) && s.scales[b] != 0, "invalid GSQ prior scale");
        for (size_t j = 0; j < 64; ++j) {
            const size_t i = b * 64 + j;
            const float  q = std::round(w[i] / s.scales[b]);
            require(std::isfinite(q) && q >= -2 && q <= 1 && std::abs(w[i] / s.scales[b] - q) < 1e-4f,
                    "prior is not on the 2-bit grid");
            float prior[4], mean = 0;
            for (int k = 0; k < 4; ++k) {
                prior[k] = -0.5f * (q - (k - 2)) * (q - (k - 2));
                mean += prior[k] / 4;
            }
            for (int k = 0; k < 4; ++k) {
                s.logits[i * 4 + k] = 0.01f * (gaussian(rng) + 6 * (prior[k] - mean));
            }
        }
    }
    s.momentum.resize(s.logits.size(), 0);
    s.scale_momentum.resize(s.scales.size(), 0);
    return s;
}

gsq_sample sample(const gsq_state & s, uint64_t seed, float tau, float kappa) {
    require(tau > 0 && std::isfinite(tau) && kappa > 0 && std::isfinite(kappa), "invalid GSQ schedule");
    gsq_sample v;
    v.noise.resize(s.logits.size());
    v.probabilities.resize(s.logits.size());
    v.weights.resize(s.logits.size() / 4);
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < v.weights.size(); ++i) {
        double z[4], maximum = -std::numeric_limits<double>::infinity(), total = 0;
        for (int k = 0; k < 4; ++k) {
            const double u     = (double(rng() >> 11) + 0.5) / 9007199254740992.0;
            v.noise[i * 4 + k] = float(-std::log(-std::log(u)));
            z[k]               = (double(s.logits[i * 4 + k]) * kappa + v.noise[i * 4 + k]) / tau;
            maximum            = std::max(maximum, z[k]);
        }
        for (int k = 0; k < 4; ++k) {
            z[k] = std::exp(z[k] - maximum);
            total += z[k];
        }
        double q = 0;
        for (int k = 0; k < 4; ++k) {
            v.probabilities[i * 4 + k] = float(z[k] / total);
            q += v.probabilities[i * 4 + k] * (k - 2);
        }
        v.weights[i] = float(q * s.scales[i / 64]);
    }
    return v;
}

void backward(const gsq_state &          s,
              const gsq_sample &         v,
              const std::vector<float> & dw,
              float                      tau,
              float                      kappa,
              std::vector<float> &       dl,
              std::vector<float> &       ds) {
    require(dw.size() == v.weights.size() && v.probabilities.size() == s.logits.size(), "invalid GSQ gradient shape");
    dl.resize(s.logits.size());
    ds.assign(s.scales.size(), 0);
    for (size_t i = 0; i < dw.size(); ++i) {
        float q = 0;
        for (int k = 0; k < 4; ++k) {
            q += v.probabilities[i * 4 + k] * (k - 2);
        }
        ds[i / 64] += dw[i] * q;
        for (int k = 0; k < 4; ++k) {
            dl[i * 4 + k] = dw[i] * s.scales[i / 64] * v.probabilities[i * 4 + k] * ((k - 2) - q) * kappa / tau;
        }
    }
}

void lion(std::vector<float> &       p,
          std::vector<float> &       m,
          const std::vector<float> & g,
          float                      lr,
          float                      b1,
          float                      b2,
          float                      wd) {
    require(p.size() == m.size() && p.size() == g.size() && lr > 0 && b1 >= 0 && b1 < 1 && b2 >= 0 && b2 < 1 && wd >= 0,
            "invalid Lion arguments");
    for (size_t i = 0; i < p.size(); ++i) {
        require(std::isfinite(g[i]) && std::isfinite(p[i]) && std::isfinite(m[i]), "nonfinite Lion state");
        const float u = b1 * m[i] + (1 - b1) * g[i];
        p[i]          = p[i] * (1 - lr * wd) - lr * ((u > 0) - (u < 0));
        m[i]          = b2 * m[i] + (1 - b2) * g[i];
    }
}

std::vector<uint8_t> pack_q2(const gsq_state & s) {
    require(s.columns > 0 && s.columns % 64 == 0 && s.logits.size() == size_t(s.columns * s.rows) * 4 &&
                s.scales.size() * 64 * 4 == s.logits.size(),
            "invalid Q2_0 shape");
    std::vector<uint8_t> bytes(s.scales.size() * 18, 0);
    for (size_t b = 0; b < s.scales.size(); ++b) {
        require(std::isfinite(s.scales[b]) && std::abs(s.scales[b]) <= 65504, "Q2_0 scale out of range");
        const ggml_fp16_t d = ggml_fp32_to_fp16(-s.scales[b]);
        std::memcpy(bytes.data() + b * 18, &d, 2);
        for (size_t j = 0; j < 64; ++j) {
            const float * l = s.logits.data() + (b * 64 + j) * 4;
            for (int k = 0; k < 4; ++k) {
                require(std::isfinite(l[k]), "nonfinite GSQ logit");
            }
            const int k    = int(std::max_element(l, l + 4) - l);
            const int code = 3 - k;
            bytes[b * 18 + 2 + j / 4] |= uint8_t(code << (2 * (j % 4)));
        }
    }
    require(ggml_validate_row_data(GGML_TYPE_Q2_0, bytes.data(), bytes.size()), "invalid packed Q2_0 data");
    return bytes;
}

std::vector<double> probabilities(const std::vector<double> & a, size_t k) {
    require(k > 1 && a.size() % k == 0, "invalid RCO logit shape");
    std::vector<double> p(a.size());
    for (size_t i = 0; i < a.size(); i += k) {
        const double maximum = *std::max_element(a.begin() + i, a.begin() + i + k);
        double       sum     = 0;
        for (size_t j = 0; j < k; ++j) {
            p[i + j] = std::exp(a[i + j] - maximum);
            sum += p[i + j];
        }
        require(std::isfinite(sum) && sum > 0, "invalid RCO logits");
        for (size_t j = 0; j < k; ++j) {
            p[i + j] /= sum;
        }
    }
    return p;
}

static size_t check_costs(const std::vector<std::vector<uint64_t>> & c, size_t count) {
    require(!c.empty() && c[0].size() > 1, "empty RCO costs");
    const size_t k = c[0].size();
    require(count == c.size() * k, "RCO cost shape mismatch");
    for (const auto & row : c) {
        require(row.size() == k, "ragged RCO costs");
    }
    return k;
}

std::vector<double> normal(const std::vector<double> & a, const std::vector<std::vector<uint64_t>> & c) {
    const size_t k = check_costs(c, a.size());
    auto         p = probabilities(a, k), n = p;
    for (size_t i = 0; i < c.size(); ++i) {
        double mean = 0;
        for (size_t j = 0; j < k; ++j) {
            mean += p[i * k + j] * c[i][j];
        }
        for (size_t j = 0; j < k; ++j) {
            n[i * k + j] *= double(c[i][j]) - mean;
        }
    }
    return n;
}

void project(std::vector<double> & v, const std::vector<double> & n) {
    require(v.size() == n.size(), "RCO vector shape mismatch");
    double dot = 0, norm = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        dot += v[i] * n[i];
        norm += n[i] * n[i];
    }
    if (norm < 1e-24) {
        return;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] -= dot / norm * n[i];
    }
}

double budget(const std::vector<double> & a, const std::vector<std::vector<uint64_t>> & c) {
    const size_t k = check_costs(c, a.size());
    auto         p = probabilities(a, k);
    double       b = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        for (size_t j = 0; j < k; ++j) {
            b += p[i * k + j] * c[i][j];
        }
    }
    return b;
}

double retract(std::vector<double> & a, const std::vector<std::vector<uint64_t>> & c, double target) {
    const size_t k       = check_costs(c, a.size());
    double       minimum = 0, maximum = 0, unit = 1;
    for (const auto & row : c) {
        minimum += *std::min_element(row.begin(), row.end());
        maximum += *std::max_element(row.begin(), row.end());
        unit = std::max(unit, double(*std::max_element(row.begin(), row.end())));
    }
    require(target > minimum && target < maximum, "RCO target must be in the interior of feasible costs");
    auto shifted = [&](double t) {
        auto x = a;
        for (size_t i = 0; i < c.size(); ++i) {
            for (size_t j = 0; j < k; ++j) {
                x[i * k + j] += t * c[i][j] / unit;
            }
        }
        return x;
    };
    double lo = -1, hi = 1;
    for (int i = 0; i < 60 && budget(shifted(lo), c) > target; ++i) {
        lo *= 2;
    }
    for (int i = 0; i < 60 && budget(shifted(hi), c) < target; ++i) {
        hi *= 2;
    }
    require(budget(shifted(lo), c) <= target && budget(shifted(hi), c) >= target, "RCO retraction bracket failed");
    for (int i = 0; i < 100; ++i) {
        const double mid = (lo + hi) / 2;
        if (budget(shifted(mid), c) > target) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    a = shifted((lo + hi) / 2);
    return budget(a, c);
}

std::vector<size_t> assign(const std::vector<std::vector<uint64_t>> & c,
                           const std::vector<std::vector<double>> &   s,
                           uint64_t                                   limit) {
    require(c.size() == s.size() && !c.empty(), "invalid assignment groups");

    struct entry {
        double              score;
        std::vector<size_t> choice;
    };

    std::map<uint64_t, entry> states;
    states.emplace(0, entry{ 0, {} });
    for (size_t i = 0; i < c.size(); ++i) {
        require(c[i].size() == s[i].size() && !c[i].empty(), "invalid assignment options");
        std::map<uint64_t, entry> next;
        for (const auto & state : states) {
            for (size_t k = 0; k < c[i].size(); ++k) {
                require(std::isfinite(s[i][k]), "nonfinite assignment score");
                if (c[i][k] > limit || state.first > limit - c[i][k]) {
                    continue;
                }
                const auto   cost  = state.first + c[i][k];
                const double score = state.second.score + s[i][k];
                auto         it    = next.find(cost);
                if (it == next.end() || score > it->second.score) {
                    auto choice = state.second.choice;
                    choice.push_back(k);
                    next[cost] = { score, std::move(choice) };
                }
            }
        }
        require(!next.empty(), "infeasible byte budget");
        double best = -std::numeric_limits<double>::infinity();
        for (auto it = next.begin(); it != next.end();) {
            if (it->second.score <= best) {
                it = next.erase(it);
            } else {
                best = it->second.score;
                ++it;
            }
        }
        require(next.size() <= 1000000, "assignment state limit exceeded");
        states = std::move(next);
    }
    auto best = std::max_element(states.begin(), states.end(),
                                 [](const auto & x, const auto & y) { return x.second.score < y.second.score; });
    return best->second.choice;
}
}  // namespace llama_opt
