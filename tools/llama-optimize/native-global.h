#pragma once

#include "candidate-store.h"

#include <memory>
#include <vector>

namespace llama_opt {
struct global_result {
    double loss = 0;
    std::vector<double> gradient;
    std::vector<float> logits;
    int recompute_regions = 0;
    uint64_t compute_bytes = 0;
    double forward_recompute_max_error = 0;
};

class native_global {
public:
    native_global(const candidate_store & store, size_t context, const std::string & backend, bool float32_math = false);
    ~native_global();
    std::vector<float> reference_logits(const std::vector<int32_t> & tokens, bool cached);
    global_result evaluate(const std::vector<int32_t> & tokens, const std::vector<float> & teacher_log_probs,
        bool kl, const std::vector<double> & soft, const std::vector<size_t> & hard,
        double temperature, bool gradient);
private:
    struct impl;
    std::unique_ptr<impl> state;
};
} // namespace llama_opt
