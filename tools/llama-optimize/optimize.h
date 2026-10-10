#pragma once
#include "ggml.h"

#include <cstdint>
#include <vector>

namespace llama_opt {
struct gsq_state {
    int64_t            columns = 0, rows = 0;
    std::vector<float> logits, scales, momentum, scale_momentum;
};

struct gsq_sample {
    std::vector<float> noise, probabilities, weights;
};

gsq_state            initialize(const std::vector<float> & weights, int64_t columns, int64_t rows);
gsq_state            initialize_prior(const std::vector<float> & quantized,
                                      const std::vector<float> & scales,
                                      int64_t                    columns,
                                      int64_t                    rows);
gsq_sample           sample(const gsq_state & state, uint64_t seed, float temperature, float logit_scale);
void                 backward(const gsq_state &          state,
                              const gsq_sample &         sample,
                              const std::vector<float> & gradient,
                              float                      temperature,
                              float                      logit_scale,
                              std::vector<float> &       logits_grad,
                              std::vector<float> &       scales_grad);
void                 lion(std::vector<float> &       parameters,
                          std::vector<float> &       momentum,
                          const std::vector<float> & gradient,
                          float                      lr,
                          float                      beta1,
                          float                      beta2,
                          float                      decay);
std::vector<uint8_t> pack_q2(const gsq_state & state);
std::vector<double>  probabilities(const std::vector<double> & alpha, size_t options);
std::vector<double> precision_vjp(const std::vector<double> & probabilities,
                                  const std::vector<double> & option_gradients,
                                  double temperature);
std::vector<double>  normal(const std::vector<double> & alpha, const std::vector<std::vector<uint64_t>> & costs);
void                 project(std::vector<double> & vector, const std::vector<double> & normal);
double               budget(const std::vector<double> & alpha, const std::vector<std::vector<uint64_t>> & costs);
double retract(std::vector<double> & alpha, const std::vector<std::vector<uint64_t>> & costs, double target);
std::vector<size_t> assign(const std::vector<std::vector<uint64_t>> & costs,
                           const std::vector<std::vector<double>> &   scores,
                           uint64_t                                   budget);
}  // namespace llama_opt
