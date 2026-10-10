#include "global-loss.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llama_opt {
loss_result token_loss(const std::vector<float> & logits,
                       const std::vector<int32_t> & targets,
                       const std::vector<float> & teacher,
                       size_t vocabulary, bool kl) {
    if (targets.empty() || vocabulary == 0 || logits.size() / vocabulary != targets.size() ||
        logits.size() % vocabulary != 0 || (kl && teacher.size() != logits.size())) {
        throw std::runtime_error("invalid global loss dimensions");
    }
    loss_result result;
    result.logits_gradient.resize(logits.size());
    for (size_t t = 0; t < targets.size(); ++t) {
        if (targets[t] < 0 || size_t(targets[t]) >= vocabulary) {
            throw std::runtime_error("global loss target outside vocabulary");
        }
        const size_t offset = t * vocabulary;
        const float high = *std::max_element(logits.begin() + offset, logits.begin() + offset + vocabulary);
        double sum = 0, teacher_sum = 0;
        for (size_t k = 0; k < vocabulary; ++k) {
            const size_t j = offset + k;
            if (!std::isfinite(logits[j]) || (kl && !std::isfinite(teacher[j]))) {
                throw std::runtime_error("nonfinite global loss input");
            }
            sum += std::exp(double(logits[j]) - high);
            if (kl) {
                teacher_sum += std::exp(double(teacher[j]));
            }
        }
        const double norm = high + std::log(sum);
        if (!kl) {
            result.value += norm - logits[offset + targets[t]];
        }
        for (size_t k = 0; k < vocabulary; ++k) {
            const size_t j = offset + k;
            const double lp = logits[j] - norm;
            const double q = kl ? std::exp(double(teacher[j])) : (k == size_t(targets[t]) ? 1.0 : 0.0);
            if (kl) {
                result.value += q * (teacher[j] - lp);
            }
            result.logits_gradient[j] = float((std::exp(lp) * (kl ? teacher_sum : 1.0) - q) / targets.size());
        }
    }
    result.value /= targets.size();
    return result;
}
} // namespace llama_opt
