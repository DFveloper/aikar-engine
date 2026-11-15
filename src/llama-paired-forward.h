#pragma once

#include "ggml.h"

#include <cstring>
#include <vector>
#include <map>

// Frontend transformation: call before either graph has been allocated.
struct llama_paired_forward_result {
    ggml_cgraph * graph;
    int pairs;
};

inline llama_paired_forward_result llama_pair_forward_graphs(ggml_context * ctx, ggml_cgraph * train, ggml_cgraph * validation,
        const ggml_tensor * only_weight = nullptr) {
    std::vector<ggml_tensor *> original, val_nodes, candidates;
    std::map<ggml_tensor *, size_t> val_index, pairs;
    for (int i = 0; i < ggml_graph_n_nodes(train); ++i) original.push_back(ggml_graph_node(train, i));
    for (int i = 0; i < ggml_graph_n_nodes(validation); ++i) {
        auto * node = ggml_graph_node(validation, i);
        val_index[node] = val_nodes.size();
        val_nodes.push_back(node);
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) candidates.push_back(node);
    }
    size_t cursor = 0;
    int paired = 0;
    const auto compatible_precision = [](const ggml_tensor * a, const ggml_tensor * b) {
        auto x = *a, y = *b;
        if (a->src[0]->type == GGML_TYPE_Q4_0) {
            auto * xp = reinterpret_cast<int32_t *>(x.op_params);
            auto * yp = reinterpret_cast<int32_t *>(y.op_params);
            if (xp[2] == GGML_PREC_UNDEFINED) xp[2] = GGML_PREC_Q4;
            if (yp[2] == GGML_PREC_UNDEFINED) yp[2] = GGML_PREC_Q4;
        }
        return std::memcmp(x.op_params, y.op_params, sizeof(x.op_params)) == 0;
    };
    for (int i = 0; i < ggml_graph_n_nodes(train); ++i) {
        auto * a = ggml_graph_node(train, i);
        if (!(a->flags & GGML_TENSOR_FLAG_COMPUTE) || (a->op != GGML_OP_MUL_MAT && a->op != GGML_OP_MUL_MAT_ID) ||
            (only_weight && a->src[0] != only_weight) ||
            (a->src[0]->flags & GGML_TENSOR_FLAG_PARAM) || a->src[1]->type != GGML_TYPE_F32) continue;
        const bool routed = a->op == GGML_OP_MUL_MAT_ID;
        if (a->ne[3] != 1 || a->src[1]->ne[3] != 1 ||
            (!routed && (!ggml_is_matrix(a) || !ggml_is_matrix(a->src[1])))) continue;
        size_t match = cursor;
        for (; match < candidates.size(); ++match) {
            const auto * b = candidates[match];
            if (b->op == a->op && b->src[0] == a->src[0] && b->src[1]->type == GGML_TYPE_F32 &&
                b->ne[0] == a->ne[0] && b->ne[3] == 1 && b->src[1]->ne[0] == a->src[1]->ne[0] &&
                b->src[1]->ne[3] == 1 && (routed ? b->ne[1] == a->ne[1] && b->src[1]->ne[1] == a->src[1]->ne[1] :
                    ggml_is_matrix(b) && ggml_is_matrix(b->src[1])) &&
                compatible_precision(a, b)) break;
        }
        if (match == candidates.size()) continue;
        auto * b = candidates[match];
        cursor = match + 1;
        const int axis = routed ? 2 : 1;
        auto * input = ggml_concat(ctx, a->src[1], b->src[1], axis);
        auto * joint = routed ? ggml_mul_mat_id(ctx, a->src[0], input, ggml_concat(ctx, a->src[2], b->src[2], 1)) :
                               ggml_mul_mat(ctx, a->src[0], input);
        std::memcpy(joint->op_params, a->op_params, sizeof(joint->op_params));
        auto * train_view = routed ? ggml_view_3d(ctx, joint, a->ne[0], a->ne[1], a->ne[2], joint->nb[1], joint->nb[2], 0) :
                                    ggml_view_2d(ctx, joint, a->ne[0], a->ne[1], joint->nb[1], 0);
        if (only_weight && !routed) GGML_ASSERT(ggml_set_matmul_prefix_grad(train_view));
        const size_t offset = a->ne[axis]*joint->nb[axis];
        auto * val_view = routed ? ggml_view_3d(ctx, joint, b->ne[0], b->ne[1], b->ne[2], joint->nb[1], joint->nb[2], offset) :
                                  ggml_view_2d(ctx, joint, b->ne[0], b->ne[1], joint->nb[1], offset);
        if ((a->flags | b->flags) & GGML_TENSOR_FLAG_OUTPUT) ggml_set_output(joint);
        ggml_tensor * detached;
        if (only_weight) {
            detached = ggml_detach(ctx, val_view);
        } else {
            auto * storage = ggml_dup_tensor(ctx, val_view);
            if (b->flags & GGML_TENSOR_FLAG_OUTPUT) ggml_set_output(storage);
            detached = ggml_cpy_no_grad(ctx, val_view, storage);
        }
        const auto replace = [](ggml_tensor * dst, const ggml_tensor * src) {
            char name[GGML_MAX_NAME];
            std::memcpy(name, dst->name, sizeof(name));
            const int flags = dst->flags;
            *dst = *src;
            std::memcpy(dst->name, name, sizeof(name));
            dst->flags = flags | (src->flags & GGML_TENSOR_FLAG_STOP_GRAD);
        };
        pairs[a] = val_index.at(b);
        replace(a, train_view);
        replace(b, detached);
        ++paired;
    }
    if (!paired) return { train, 0 };
    auto * combined = ggml_new_graph_custom(ctx, ggml_graph_size(train) + ggml_graph_size(validation) + paired*8 + 64, true);
    const auto expand = [&](ggml_tensor * node) {
        if (node->flags & GGML_TENSOR_FLAG_COMPUTE) ggml_build_forward_expand(combined, node);
        else ggml_build_forward_order(combined, node);
    };
    size_t next_val = 0;
    for (auto * node : original) {
        const auto pair = pairs.find(node);
        if (pair != pairs.end()) {
            // Keep cache writes in each original graph ahead of their readers.
            while (next_val < pair->second) expand(val_nodes[next_val++]);
        }
        expand(node);
    }
    while (next_val < val_nodes.size()) expand(val_nodes[next_val++]);
    return { combined, paired };
}
