#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace llama_opt {
struct loss_result {
    double value = 0;
    std::vector<float> logits_gradient;
};

loss_result token_loss(const std::vector<float> & logits,
                       const std::vector<int32_t> & targets,
                       const std::vector<float> & teacher_log_probs,
                       size_t vocabulary, bool kl);
} // namespace llama_opt
