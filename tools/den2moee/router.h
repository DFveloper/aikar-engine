#pragma once

#include <vector>

namespace den2moee {

struct RouterSelection {
    std::vector<int> selected;
    std::vector<float> softmax;
    std::vector<float> raw_selected_weights;
    std::vector<float> weights;
};

RouterSelection select_router(const std::vector<float> & logits, int routed_experts,
                              int null_experts, int top_k);

} // namespace den2moee
