#include "scoring.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace den2moee {

std::vector<float> calculate_scs(const std::vector<float> & attention, int span_size) {
    if (attention.empty() || span_size <= 0) throw std::invalid_argument("invalid SCS input");
    const int spans = (static_cast<int>(attention.size()) + span_size - 1) / span_size;
    std::vector<float> scores(spans, 0.0f);
    for (int span = 0; span < spans; ++span) {
        const int begin = span * span_size;
        const int end = std::min(begin + span_size, static_cast<int>(attention.size()));
        scores[span] = std::accumulate(attention.begin() + begin, attention.begin() + end, 0.0f);
    }
    const auto [min_it, max_it] = std::minmax_element(scores.begin(), scores.end());
    const float minimum = *min_it;
    const float denominator = *max_it - minimum + 1e-9f;
    for (float & score : scores) score = (score - minimum) / denominator;
    return scores;
}

std::vector<int> select_top_spans(const std::vector<float> & scs_by_span) {
    if (scs_by_span.empty()) throw std::invalid_argument("SCS has no spans");
    const int count = std::max(1, static_cast<int>(scs_by_span.size()) / 2);
    std::vector<int> indices(scs_by_span.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        if (scs_by_span[a] != scs_by_span[b]) return scs_by_span[a] > scs_by_span[b];
        return a < b;
    });
    indices.resize(count);
    std::sort(indices.begin(), indices.end());
    return indices;
}

static float gelu(float value) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    return 0.5f * value * (1.0f + std::tanh(sqrt_2_over_pi * value * (1.0f + 0.044715f * value * value)));
}

static float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

TokenScoreResult calculate_token_scores(const std::vector<float> & attention,
                                        const std::vector<float> & original_final_hidden,
                                        int hidden_size,
                                        const std::vector<RssPerturbation> & perturbations,
                                        int span_size, int ngram, TokenScoreMode mode) {
    if (attention.empty() || hidden_size <= 0 || original_final_hidden.size() != attention.size() * hidden_size ||
        span_size <= 0 || ngram <= 0) {
        throw std::invalid_argument("invalid token score input");
    }
    TokenScoreResult result;
    result.scs_by_span = calculate_scs(attention, span_size);
    result.selected_spans = select_top_spans(result.scs_by_span);
    result.rss_by_token.assign(attention.size(), 0.0f);
    std::vector<float> rss_counts(attention.size(), 0.0f);
    if (mode == TokenScoreMode::ScsRss) {
        double original_norm_sq = 0.0;
        for (float value : original_final_hidden) original_norm_sq += static_cast<double>(value) * value;
        const float original_norm = static_cast<float>(std::sqrt(original_norm_sq));
        for (const RssPerturbation & perturbation : perturbations) {
            if (perturbation.start < 0 || perturbation.start + ngram > static_cast<int>(attention.size()) ||
                perturbation.final_hidden.size() != original_final_hidden.size()) {
                throw std::invalid_argument("invalid RSS perturbation");
            }
            bool selected = false;
            for (int span : result.selected_spans) {
                if (perturbation.start >= span * span_size && perturbation.start < (span + 1) * span_size) {
                    selected = true;
                    break;
                }
            }
            if (!selected) continue;
            double perturbed_norm_sq = 0.0;
            double diff_norm_sq = 0.0;
            for (size_t i = 0; i < original_final_hidden.size(); ++i) {
                const double original = original_final_hidden[i];
                const double perturbed = perturbation.final_hidden[i];
                perturbed_norm_sq += perturbed * perturbed;
                const double diff = original - perturbed;
                diff_norm_sq += diff * diff;
            }
            const float rss = static_cast<float>(std::sqrt(diff_norm_sq) /
                (original_norm + std::sqrt(perturbed_norm_sq) + 1e-9f));
            for (int token = perturbation.start; token < perturbation.start + ngram; ++token) {
                result.rss_by_token[token] += rss;
                rss_counts[token] += 1.0f;
            }
        }
        for (size_t token = 0; token < result.rss_by_token.size(); ++token) {
            result.rss_by_token[token] /= rss_counts[token] + 1e-9f;
        }
    }

    result.token_score.resize(attention.size());
    for (size_t token = 0; token < attention.size(); ++token) {
        if (mode == TokenScoreMode::Uniform) {
            result.token_score[token] = 1.0f;
        } else {
            const size_t span = std::min(token / static_cast<size_t>(span_size), result.scs_by_span.size() - 1);
            result.token_score[token] = mode == TokenScoreMode::ScsOnly
                ? result.scs_by_span[span]
                : 0.5f * (result.scs_by_span[span] + result.rss_by_token[token]);
        }
    }
    return result;
}

std::vector<float> weighted_activation_score(const std::vector<float> & gate_projection,
                                             const std::vector<float> & up_projection,
                                             int seq_len, int neurons,
                                             const std::vector<float> & token_score,
                                             ScoreActivation activation) {
    if (seq_len <= 0 || neurons <= 0 || gate_projection.size() != static_cast<size_t>(seq_len) * neurons ||
        up_projection.size() != gate_projection.size() || token_score.size() != static_cast<size_t>(seq_len)) {
        throw std::invalid_argument("invalid activation score input");
    }
    std::vector<float> result(neurons, 0.0f);
    for (int token = 0; token < seq_len; ++token) {
        for (int neuron = 0; neuron < neurons; ++neuron) {
            const float gate = activation == ScoreActivation::Gemma4Gelu
                ? gelu(gate_projection[token * neurons + neuron])
                : silu(gate_projection[token * neurons + neuron]);
            result[neuron] += token_score[token] * std::fabs(gate * up_projection[token * neurons + neuron]);
        }
    }
    for (float & value : result) value /= static_cast<float>(seq_len);
    return result;
}

DomainScoreAccumulator::DomainScoreAccumulator(std::vector<std::string> domains, int neurons) :
        domains_(std::move(domains)), score_sum_(domains_.size() * static_cast<size_t>(neurons), 0.0f),
        sample_count_(domains_.size(), 0), neurons_(neurons) {
    if (domains_.empty() || neurons <= 0) throw std::invalid_argument("invalid domain accumulator");
}

void DomainScoreAccumulator::add(const std::string & domain, const std::vector<float> & domain_score) {
    const auto it = std::find(domains_.begin(), domains_.end(), domain);
    if (it == domains_.end() || domain_score.size() != static_cast<size_t>(neurons_)) {
        throw std::invalid_argument("unknown domain or wrong neuron count");
    }
    const size_t index = static_cast<size_t>(std::distance(domains_.begin(), it));
    for (int neuron = 0; neuron < neurons_; ++neuron) score_sum_[index * neurons_ + neuron] += domain_score[neuron];
    ++sample_count_[index];
}

std::vector<std::vector<float>> DomainScoreAccumulator::normalized_features() const {
    std::vector<std::vector<float>> result(neurons_, std::vector<float>(domains_.size(), 0.0f));
    for (size_t domain = 0; domain < domains_.size(); ++domain) {
        std::vector<float> scores(neurons_);
        for (int neuron = 0; neuron < neurons_; ++neuron) {
            scores[neuron] = sample_count_[domain] > 0
                ? score_sum_[domain * neurons_ + neuron] / static_cast<float>(sample_count_[domain]) : 0.0f;
        }
        const float mean = std::accumulate(scores.begin(), scores.end(), 0.0f) / static_cast<float>(neurons_);
        float variance = 0.0f;
        for (float score : scores) variance += (score - mean) * (score - mean);
        const float standard_deviation = std::sqrt(variance / static_cast<float>(neurons_));
        for (int neuron = 0; neuron < neurons_; ++neuron) {
            result[neuron][domain] = (scores[neuron] - mean) / (standard_deviation + 1e-9f);
        }
    }
    return result;
}

} // namespace den2moee
