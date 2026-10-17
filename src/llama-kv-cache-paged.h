#pragma once

#include "llama-memory.h"
#include "llama-block-manager.h"
#include "llama-batch.h"
#include "ggml-cpp.h"

#include <array>
#include <bitset>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_cparams;
class llama_kv_cache_paged_context;

class llama_kv_cache_paged : public llama_memory_i {
public:
    struct page {
        uint32_t id;
        std::bitset<128> mask;
    };
    struct domain {
        llama_block_manager pool;
        std::vector<std::map<uint32_t, page>> seqs;
        uint32_t capacity = 0;
    };
    using state = std::array<domain, 2>;
    struct page_copy { uint32_t domain_id, src, dst; };
    struct table_range { size_t offset, size; };
    struct table_page { uint32_t seq, logical; };
    struct batch_plan {
        llama_ubatch ubatch;
        state after;
        std::array<std::vector<int32_t>, 2> slots;
        std::vector<int32_t> queries;
        std::vector<page_copy> copies;
        std::array<std::vector<table_page>, 2> table_pages;
        std::array<std::vector<table_range>, 2> table_ranges;
        std::array<bool, 2> table_full = {};
    };
    struct layer {
        bool swa;
        ggml_tensor * cache;
        ggml_backend_dev_t device;
        mutable ggml_backend_t backend = nullptr;
    };

    llama_kv_cache_paged(const llama_model & model, const llama_memory_params & params, const llama_cparams & cparams);
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;
    bool get_can_shift() const override { return false; }
    llama_memory_prefix_id prefix_retain(llama_seq_id src, llama_pos end) override;
    bool prefix_attach(llama_memory_prefix_id id, llama_seq_id dst) override;
    void prefix_release(llama_memory_prefix_id id) override;
    bool prefix_get_usage(llama_memory_prefix_usage & out) const override;
    uint32_t prepared_contexts = 0;
    void clear(bool data) override;
    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) override;
    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;
    void set_kv_hadamard_policy(bool k, bool v, bool ks, bool vs, bool explicit_policy) override;
    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) override;

    uint32_t block_size, max_blocks, n_seq_max, window;
    uint32_t table_blocks() const { return max_blocks*((block_size + 31)/32); }
    bool swa_full;
    ggml_tensor * table_input(ggml_context * ctx, bool swa) const;
    std::vector<layer> layers;
    state current;
    bool prepare(batch_plan & plan, state & candidate, bool force_full = false) const;
    void apply(const batch_plan & plan);
    void set_table_input(ggml_tensor * tensor, bool swa, const batch_plan & plan) const;
    int64_t rotation_size(bool swa, bool value) const { return value ? v_rot_size[swa] : k_rot_size[swa]; }
    ggml_tensor * build_input_k_rot(ggml_context * ctx, bool swa) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx, bool swa) const;
    void set_input_rot(ggml_tensor * dst) const;

private:
    using prefix_pages = std::array<std::map<uint32_t, page>, 2>;
    std::map<llama_memory_prefix_id, prefix_pages> prefixes;
    std::array<std::vector<uint32_t>, 2> retained_owners;
    std::array<uint64_t, 2> retained_pages = {};
    uint64_t prefix_metadata_bytes = 0;
    mutable std::array<uint64_t, 2> retry_pages = {};
    std::array<int64_t, 2> k_rot_size = {}, v_rot_size = {};
    std::unordered_map<int64_t, std::vector<float>> rotations;
    std::vector<ggml_context_ptr> contexts;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::array<ggml_tensor *, 2> device_tables = {};
    std::array<std::vector<int32_t>, 2> table_cache;
    std::array<ggml_backend_buffer_ptr, 2> staging;
    bool table_cache_valid = false;
    const bool full_upload = std::getenv("LLAMA_PAGED_KV_FULL_UPLOAD") != nullptr;
    void synchronize() const;
    void copy_page(uint32_t domain_id, uint32_t src, uint32_t dst);
    void remove(state & value, llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void rebuild_table_cache(bool swa);
    void update_table_page(bool swa, uint32_t seq, uint32_t logical);
    void mark_table_page_dirty(batch_plan & plan, uint32_t dom, uint32_t seq, uint32_t logical) const;
};

class llama_kv_cache_paged_context : public llama_memory_context_i {
public:
    llama_kv_cache_paged_context(llama_kv_cache_paged * kv, std::vector<llama_kv_cache_paged::batch_plan> plans,
            llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS) : kv(kv), plans(std::move(plans)), status(status) { ++kv->prepared_contexts; }
    ~llama_kv_cache_paged_context() override { --kv->prepared_contexts; }
    bool next() override { return ++index < plans.size(); }
    bool apply() override;
    const llama_ubatch & get_ubatch() const override { return plans.at(index).ubatch; }
    llama_memory_status get_status() const override { return status; }
    llama_kv_cache_paged * kv;
    std::vector<llama_kv_cache_paged::batch_plan> plans;
    size_t index = 0;
    llama_memory_status status;
};
