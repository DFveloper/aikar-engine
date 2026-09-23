#include "router.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace den2moee {

RouterSelection select_router(const std::vector<float> & logits, int routed_experts,
                              int null_experts, int top_k) {
    const int n_candidates = routed_experts + null_experts;
    if (routed_experts < 0 || null_experts < 0 || static_cast<int>(logits.size()) != n_candidates ||
        top_k <= 0 || top_k > n_candidates) {
        throw std::invalid_argument("invalid router candidate configuration");
    }

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(n_candidates);
    float sum = 0.0f;
    for (int i = 0; i < n_candidates; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (float & value : probs) {
        value /= sum;
    }

    std::vector<int> order(n_candidates);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });

    RouterSelection result;
    result.softmax = probs;
    result.selected.assign(order.begin(), order.begin() + top_k);
    result.raw_selected_weights.reserve(top_k);
    result.weights.reserve(top_k);

    float valid_sum = 0.0f;
    for (int id : result.selected) {
        const float raw = probs[id];
        result.raw_selected_weights.push_back(raw);
        if (id < routed_experts) valid_sum += raw;
    }

    for (size_t i = 0; i < result.selected.size(); ++i) {
        const int id = result.selected[i];
        const float raw = result.raw_selected_weights[i];
        result.weights.push_back(id < routed_experts && valid_sum > 0.0f ? raw / valid_sum : raw);
    }
    return result;
}

} // namespace den2moee
