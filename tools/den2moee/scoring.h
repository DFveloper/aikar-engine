#pragma once

#include "config.h"

#include <cstddef>
#include <string>
#include <vector>

namespace den2moee {

struct RssPerturbation {
    int start = 0;
    std::vector<float> final_hidden;
};

struct TokenScoreResult {
    std::vector<float> scs_by_span;
    std::vector<int> selected_spans;
    std::vector<float> rss_by_token;
    std::vector<float> token_score;
};

std::vector<float> calculate_scs(const std::vector<float> & attention, int span_size);
std::vector<int> select_top_spans(const std::vector<float> & scs_by_span);
TokenScoreResult calculate_token_scores(const std::vector<float> & attention,
                                        const std::vector<float> & original_final_hidden,
                                        int hidden_size,
                                        const std::vector<RssPerturbation> & perturbations,
                                        int span_size, int ngram, TokenScoreMode mode);

std::vector<float> weighted_activation_score(const std::vector<float> & gate_projection,
                                             const std::vector<float> & up_projection,
                                             int seq_len, int neurons,
                                             const std::vector<float> & token_score,
                                             ScoreActivation activation);

class DomainScoreAccumulator {
public:
    DomainScoreAccumulator(std::vector<std::string> domains, int neurons);

    void add(const std::string & domain, const std::vector<float> & domain_score);
    std::vector<std::vector<float>> normalized_features() const;
    const std::vector<std::string> & domains() const { return domains_; }

private:
    std::vector<std::string> domains_;
    std::vector<float> score_sum_;
    std::vector<int> sample_count_;
    int neurons_ = 0;
};

} // namespace den2moee
