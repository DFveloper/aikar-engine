// Page pool and layout adapted from llama.cpp PR #22569 (matiaslin).
#include "llama-kv-cache-paged.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "llama-io.h"
#include "llama-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <set>
#include <stdexcept>

llama_kv_cache_paged::llama_kv_cache_paged(
        const llama_model & model, const llama_memory_params & params, const llama_cparams & cp) :
        block_size(cp.kv_paged_block_size), max_blocks((uint64_t(cp.n_ctx_seq) + block_size - 1)/block_size),
        n_seq_max(cp.n_seq_max), window(model.hparams.n_swa), swa_full(params.swa_full) {
    const uint32_t base_pages = cp.kv_paged_n_blocks ? cp.kv_paged_n_blocks : (uint64_t(cp.n_ctx) + block_size - 1)/block_size;
    if (uint64_t(max_blocks)*block_size > INT_MAX) {
        throw std::runtime_error("paged KV logical token budget exceeds signed indexing range");
    }
    if (!n_seq_max || uint64_t(max_blocks)*n_seq_max > INT_MAX/2) {
        throw std::runtime_error("paged KV page table exceeds signed indexing range");
    }
    if (!base_pages || uint64_t(base_pages)*block_size > INT_MAX) {
        throw std::runtime_error("paged KV physical token budget exceeds signed indexing range");
    }
    const uint64_t swa_tokens = std::min<uint64_t>(cp.n_ctx, uint64_t(window)*n_seq_max + cp.n_ubatch + 2*uint64_t(block_size)*n_seq_max);
    const uint32_t swa_pages = swa_full || !window ? base_pages : std::min<uint64_t>(base_pages, (swa_tokens + block_size - 1)/block_size);
    for (int dom = 0; dom < 2; ++dom) {
        current[dom].capacity = dom ? swa_pages : base_pages;
        current[dom].pool.init(current[dom].capacity, 0, 0.0f);
        current[dom].seqs.resize(n_seq_max);
    }
    std::set<ggml_backend_dev_t> accelerators;
    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        if (!model.hparams.has_kv(il)) { throw std::runtime_error("paged KV does not support shared KV layers"); }
        const bool swa = model.hparams.is_swa(il);
        const ggml_type type_k = swa ? params.type_k_swa : params.type_k;
        const ggml_type type_v = swa ? params.type_v_swa : params.type_v;
        if (type_k != type_v || (type_k != GGML_TYPE_F16 && type_k != GGML_TYPE_Q8_KV)) {
            throw std::runtime_error("paged KV requires matching F16 or Q8_KV K/V in each cache domain");
        }
        const int64_t dim = model.hparams.n_embd_head_k(il), hk = model.hparams.n_head_kv(il);
        if (dim != model.hparams.n_embd_head_v(il) || dim%64 || dim > 1024 || !hk) {
            throw std::runtime_error("unsupported paged KV head geometry");
        }
        auto * weight_dev = model.dev_layer(il);
        if (weight_dev && ggml_backend_dev_type(weight_dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            accelerators.insert(weight_dev);
        }
        auto * dev = cp.offload_kqv ? weight_dev : nullptr;
        auto * buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            accelerators.insert(dev);
            const char * name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
            if (strcmp(name, "CUDA") && strcmp(name, "Vulkan")) { throw std::runtime_error("paged KV supports CPU, CUDA and Vulkan only"); }
        }
        if (accelerators.size() > 1) { throw std::runtime_error("paged KV supports CPU plus one GPU device"); }
        ggml_init_params ip = {ggml_tensor_overhead(), nullptr, true};
        ggml_context_ptr ctx(ggml_init(ip));
        if (!ctx) { throw std::runtime_error("failed to create paged KV context"); }
        ggml_tensor * cache = ggml_new_tensor_4d(ctx.get(), type_k, dim, block_size, 2*hk, current[swa].capacity);
        ggml_format_name(cache, "paged_kv_l%u", il);
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buf) { throw std::runtime_error("failed to allocate paged KV pool"); }
        ggml_backend_buffer_clear(buf.get(), 0);
        layers.push_back({swa, cache, dev});
        contexts.push_back(std::move(ctx)); buffers.push_back(std::move(buf));
    }
    LLAMA_LOG_INFO("%s: paged KV block_size=%u, full pages=%u, SWA pages=%u\n", __func__, block_size, base_pages, swa_pages);
    for (const auto & entry : memory_breakdown()) {
        LLAMA_LOG_INFO("%s: %s pool: %.2f MiB\n", __func__, ggml_backend_buft_name(entry.first), entry.second/(1024.0*1024.0));
    }
}

bool llama_kv_cache_paged::prepare(batch_plan & plan, state & candidate) const {
    const auto & ub = plan.ubatch;
    std::set<llama_seq_id> active;
    for (uint32_t t = 0; t < ub.n_tokens; ++t) {
        for (int s = 0; s < ub.n_seq_id[t]; ++s) { active.insert(ub.seq_id[t][s]); }
    }
    for (int dom = 0; dom < 2; ++dom) {
        auto & value = candidate[dom];
        plan.slots[dom].assign(size_t(ub.n_tokens)*n_seq_max, -1);
        if (dom == 1 && window && !swa_full) {
            for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
                llama_pos earliest = INT_MAX;
                for (uint32_t t = 0; t < ub.n_tokens; ++t) {
                    for (int s = 0; s < ub.n_seq_id[t]; ++s) {
                        if (ub.seq_id[t][s] == (llama_seq_id) seq) { earliest = std::min(earliest, ub.pos[t]); }
                    }
                }
                if (earliest != INT_MAX) {
                    const int64_t limit = int64_t(earliest) - window + 1;
                    auto & pages = value.seqs[seq];
                    for (auto it = pages.begin(); it != pages.end();) {
                        if (int64_t(it->first + 1)*block_size <= limit) {
                            value.pool.release_gpu_blocks({it->second.id}); it = pages.erase(it);
                        } else { ++it; }
                    }
                }
            }
        }
        auto checkout = [&]() {
            auto ids = value.pool.checkout_gpu_blocks(1);
            if (ids.empty() && dom == 1 && window && !swa_full) {
                // Keep the last query window of inactive sequences.
                for (uint32_t other = 0; other < n_seq_max; ++other) {
                    auto & retained = value.seqs[other];
                    if (active.count(other) || retained.empty()) { continue; }
                    const auto & last = *retained.rbegin();
                    uint32_t tail = block_size - 1;
                    while (!(last.second.mask & (1u << tail))) { --tail; }
                    const int64_t limit = int64_t(last.first)*block_size + tail - window + 1;
                    for (auto old = retained.begin(); old != retained.end();) {
                        if (int64_t(old->first + 1)*block_size > limit) { break; }
                        value.pool.release_gpu_blocks({old->second.id}); old = retained.erase(old);
                    }
                }
                ids = value.pool.checkout_gpu_blocks(1);
            }
            return ids;
        };
        for (uint32_t t = 0; t < ub.n_tokens; ++t) {
            const llama_pos pos = ub.pos[t];
            if (pos < 0 || uint32_t(pos)/block_size >= max_blocks) { return false; }
            std::set<uint32_t> written;
            for (int s = 0; s < ub.n_seq_id[t]; ++s) {
                const llama_seq_id seq = ub.seq_id[t][s];
                if (seq < 0 || uint32_t(seq) >= n_seq_max) { return false; }
                auto & pages = value.seqs[seq];
                const uint32_t logical = pos/block_size;
                auto it = pages.find(logical);
                if (it == pages.end()) {
                    auto ids = checkout();
                    if (ids.empty()) { return false; }
                    it = pages.emplace(logical, page{ids[0], 0}).first;
                } else if (value.pool.gpu_ref_count(it->second.id) > 1) {
                    auto ids = checkout();
                    if (ids.empty()) { return false; }
                    plan.copies.push_back({uint32_t(dom), it->second.id, ids[0]});
                    value.pool.release_gpu_blocks({it->second.id}); it->second.id = ids[0];
                }
                it->second.mask |= 1u << (pos%block_size);
                if (written.insert(it->second.id).second) {
                    plan.slots[dom][size_t(t)*n_seq_max + seq] = it->second.id*block_size + pos%block_size;
                }
            }
        }
    }
    plan.queries.resize(2*ub.n_tokens);
    for (uint32_t t = 0; t < ub.n_tokens; ++t) {
        plan.queries[2*t] = ub.pos[t]; plan.queries[2*t + 1] = ub.seq_id[t][0];
    }
    plan.after = candidate;
    return true;
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    GGML_UNUSED(embd_all);
    balloc.split_reset();
    state candidate = current;
    std::vector<batch_plan> plans;
    while (true) {
        batch_plan plan;
        plan.ubatch = balloc.split_simple(n_ubatch);
        if (!plan.ubatch.n_tokens) { break; }
        if (!prepare(plan, candidate)) {
            return std::make_unique<llama_kv_cache_paged_context>(this, std::vector<batch_plan>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        plans.push_back(std::move(plan));
    }
    if (plans.empty() || balloc.get_n_used() != balloc.get_n_tokens()) {
        return std::make_unique<llama_kv_cache_paged_context>(this, std::vector<batch_plan>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    return std::make_unique<llama_kv_cache_paged_context>(this, std::move(plans));
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    return std::make_unique<llama_kv_cache_paged_context>(this, std::vector<batch_plan>{});
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx); GGML_UNUSED(optimize);
    return std::make_unique<llama_kv_cache_paged_context>(this, std::vector<batch_plan>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
}

void llama_kv_cache_paged::synchronize() const {
    std::set<ggml_backend_t> owners;
    for (const auto & l : layers) {
        if (l.backend && owners.insert(l.backend).second) { ggml_backend_synchronize(l.backend); }
    }
}

void llama_kv_cache_paged::copy_page(uint32_t dom, uint32_t src, uint32_t dst) {
    for (const auto & l : layers) {
        if (l.swa != bool(dom)) { continue; }
        std::vector<uint8_t> bytes(l.cache->nb[3]);
        ggml_backend_tensor_get(l.cache, bytes.data(), src*l.cache->nb[3], bytes.size());
        ggml_backend_tensor_set(l.cache, bytes.data(), dst*l.cache->nb[3], bytes.size());
    }
}

void llama_kv_cache_paged::apply(const batch_plan & plan) {
    if (!plan.copies.empty()) { synchronize(); }
    for (const auto & copy : plan.copies) { copy_page(copy.domain_id, copy.src, copy.dst); }
    current = plan.after;
    LLAMA_LOG_DEBUG("%s: full pages live=%zu free=%zu, SWA pages live=%zu free=%zu\n", __func__,
            current[0].capacity - current[0].pool.n_free_gpu_blocks(), current[0].pool.n_free_gpu_blocks(),
            current[1].capacity - current[1].pool.n_free_gpu_blocks(), current[1].pool.n_free_gpu_blocks());
}

bool llama_kv_cache_paged_context::apply() {
    if (llama_memory_status_is_fail(status)) { return false; }
    if (!plans.empty()) { kv->apply(plans.at(index)); }
    return true;
}

std::vector<int32_t> llama_kv_cache_paged::table(bool swa) const {
    std::vector<int32_t> out(size_t(2)*max_blocks*n_seq_max, 0);
    for (size_t i = 0; i < out.size(); i += 2) { out[i] = -1; }
    for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
        for (const auto & p : current[swa].seqs[seq]) {
            out[2*(size_t(seq)*max_blocks + p.first)] = p.second.id;
            out[2*(size_t(seq)*max_blocks + p.first) + 1] = p.second.mask;
        }
    }
    return out;
}

void llama_kv_cache_paged::remove(state & value, llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    p0 = std::max(0, p0); if (p1 < 0) { p1 = INT_MAX; }
    for (auto & dom : value) {
        for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
            if (seq_id >= 0 && uint32_t(seq_id) != seq) { continue; }
            auto & pages = dom.seqs[seq];
            for (auto it = pages.begin(); it != pages.end();) {
                for (uint32_t j = 0; j < block_size; ++j) {
                    const int64_t pos = int64_t(it->first)*block_size + j;
                    if (pos >= p0 && pos < p1) { it->second.mask &= ~(1u << j); }
                }
                if (!it->second.mask) {
                    dom.pool.release_gpu_blocks({it->second.id}); it = pages.erase(it);
                } else { ++it; }
            }
        }
    }
}

void llama_kv_cache_paged::clear(bool data) {
    if (data) { synchronize(); }
    for (auto & dom : current) {
        dom.pool.init(dom.capacity, 0, 0.0f); dom.seqs.assign(n_seq_max, {});
    }
    if (data) { for (const auto & buf : buffers) { ggml_backend_buffer_clear(buf.get(), 0); } }
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id >= (llama_seq_id) n_seq_max) { return false; }
    remove(current, seq_id, p0, p1); return true;
}

void llama_kv_cache_paged::seq_cp(llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    if (src == dst) { return; }
    if (src < 0 || dst < 0 || uint32_t(src) >= n_seq_max || uint32_t(dst) >= n_seq_max) {
        throw std::runtime_error("invalid paged KV sequence copy");
    }
    p0 = std::max(0, p0); if (p1 < 0) { p1 = INT_MAX; }
    synchronize();
    state candidate = current;
    remove(candidate, dst, p0, p1);
    struct row_copy { ggml_tensor * cache; size_t offset; std::vector<uint8_t> bytes; };
    std::vector<row_copy> writes;
    for (uint32_t d = 0; d < 2; ++d) {
        auto & dom = candidate[d];
        for (const auto & entry : dom.seqs[src]) {
            uint32_t mask = 0;
            for (uint32_t j = 0; j < block_size; ++j) {
                const int64_t pos = int64_t(entry.first)*block_size + j;
                if (pos >= p0 && pos < p1) { mask |= entry.second.mask & (1u << j); }
            }
            if (!mask) { continue; }
            auto it = dom.seqs[dst].find(entry.first);
            if (it == dom.seqs[dst].end()) {
                dom.pool.retain_gpu_block(entry.second.id);
                dom.seqs[dst][entry.first] = {entry.second.id, mask};
            } else {
                if (it->second.id != entry.second.id) {
                    const uint32_t old_id = it->second.id;
                    if (dom.pool.gpu_ref_count(old_id) > 1) {
                        auto ids = dom.pool.checkout_gpu_blocks(1);
                        if (ids.empty()) { throw std::runtime_error("paged KV copy exceeds free page budget"); }
                        dom.pool.release_gpu_blocks({old_id});
                        it->second.id = ids[0];
                    }
                    for (const auto & l : layers) {
                        if (l.swa != bool(d)) { continue; }
                        row_copy copy{l.cache, it->second.id*l.cache->nb[3], std::vector<uint8_t>(l.cache->nb[3])};
                        ggml_backend_tensor_get(l.cache, copy.bytes.data(), old_id*l.cache->nb[3], copy.bytes.size());
                        for (int64_t h = 0; h < l.cache->ne[2]; ++h) {
                            for (uint32_t j = 0; j < block_size; ++j) {
                                if (!(mask & (1u << j))) { continue; }
                                const size_t offset = h*l.cache->nb[2] + j*l.cache->nb[1];
                                ggml_backend_tensor_get(l.cache, copy.bytes.data() + offset, entry.second.id*l.cache->nb[3] + offset, l.cache->nb[1]);
                            }
                        }
                        writes.push_back(std::move(copy));
                    }
                }
                it->second.mask |= mask;
            }
        }
    }
    for (const auto & copy : writes) { ggml_backend_tensor_set(copy.cache, copy.bytes.data(), copy.offset, copy.bytes.size()); }
    current = std::move(candidate);
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    if (seq_id < 0 || uint32_t(seq_id) >= n_seq_max) { throw std::runtime_error("invalid paged KV sequence"); }
    for (uint32_t s = 0; s < n_seq_max; ++s) { if (s != uint32_t(seq_id)) { seq_rm(s, -1, -1); } }
}

void llama_kv_cache_paged::seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos delta) {
    if (delta) { throw std::runtime_error("paged KV context shifting is not supported"); }
}

void llama_kv_cache_paged::seq_div(llama_seq_id, llama_pos, llama_pos, int d) {
    if (d != 1) { throw std::runtime_error("paged KV position division is not supported"); }
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || uint32_t(seq_id) >= n_seq_max) { return -1; }
    const auto & pages = current[window ? 1 : 0].seqs[seq_id];
    for (const auto & p : pages) {
        for (uint32_t j = 0; j < block_size; ++j) { if (p.second.mask & (1u << j)) { return p.first*block_size + j; } }
    }
    return -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || uint32_t(seq_id) >= n_seq_max) { return -1; }
    const auto & pages = current[window ? 1 : 0].seqs[seq_id];
    for (auto it = pages.rbegin(); it != pages.rend(); ++it) {
        for (int j = block_size - 1; j >= 0; --j) { if (it->second.mask & (1u << j)) { return it->first*block_size + j; } }
    }
    return -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & buf : buffers) { result[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get()); }
    return result;
}

void llama_kv_cache_paged::set_kv_hadamard_policy(bool k, bool v, bool ks, bool vs, bool explicit_policy) {
    GGML_UNUSED(explicit_policy);
    if (k || v || ks || vs) { throw std::runtime_error("paged KV Hadamard rotation is not supported"); }
}

void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (flags || seq_id >= (llama_seq_id) n_seq_max) { throw std::runtime_error("unsupported paged KV state flags or sequence"); }
    const uint32_t header[3] = {0x504b5632, block_size, uint32_t(layers.size())};
    io.write(header, sizeof(header));
    for (const auto & l : layers) {
        const uint32_t geometry[4] = {uint32_t(l.cache->type), uint32_t(l.cache->ne[0]), uint32_t(l.cache->ne[2]), uint32_t(l.swa)};
        io.write(geometry, sizeof(geometry));
    }
    const uint32_t count = seq_id < 0 ? n_seq_max : 1;
    io.write(&count, sizeof(count));
    std::array<std::set<uint32_t>, 2> saved_pages;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t seq = seq_id < 0 ? i : seq_id;
        io.write(&seq, sizeof(seq));
        for (int dom = 0; dom < 2; ++dom) {
            const auto & pages = current[dom].seqs[seq];
            const uint32_t n_pages = pages.size();
            io.write(&n_pages, sizeof(n_pages));
            for (const auto & p : pages) {
                const bool first = saved_pages[dom].insert(p.second.id).second;
                const uint32_t metadata[4] = {p.first, p.second.mask, p.second.id, uint32_t(first)};
                io.write(metadata, sizeof(metadata));
                if (first) {
                    for (const auto & l : layers) {
                        if (l.swa == bool(dom)) { io.write_tensor(l.cache, size_t(p.second.id)*l.cache->nb[3], l.cache->nb[3]); }
                    }
                }
            }
        }
    }
}

void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (flags || seq_id >= (llama_seq_id) n_seq_max) { throw std::runtime_error("unsupported paged KV state flags or sequence"); }
    uint32_t header[3]; io.read(header, sizeof(header));
    if (header[0] != 0x504b5632 || header[1] != block_size || header[2] != layers.size()) {
        throw std::runtime_error("incompatible paged KV state header");
    }
    for (const auto & l : layers) {
        uint32_t geometry[4]; io.read(geometry, sizeof(geometry));
        if (geometry[0] != uint32_t(l.cache->type) || geometry[1] != l.cache->ne[0] ||
                geometry[2] != l.cache->ne[2] || geometry[3] != uint32_t(l.swa)) {
            throw std::runtime_error("incompatible paged KV state geometry");
        }
    }
    uint32_t count; io.read(&count, sizeof(count));
    if (count > n_seq_max || (seq_id >= 0 && count != 1)) { throw std::runtime_error("invalid paged KV state sequence count"); }
    state candidate = current;
    remove(candidate, seq_id, -1, -1);
    struct pending_page { uint32_t physical; std::vector<std::pair<uint32_t, std::vector<uint8_t>>> data; };
    std::vector<pending_page> pending;
    std::set<uint32_t> seen;
    std::array<std::map<uint32_t, std::pair<uint32_t, uint32_t>>, 2> restored_pages;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t saved_seq; io.read(&saved_seq, sizeof(saved_seq));
        const uint32_t seq = seq_id < 0 ? saved_seq : seq_id;
        if (seq >= n_seq_max || !seen.insert(seq).second) { throw std::runtime_error("invalid paged KV state sequence ID"); }
        for (int dom = 0; dom < 2; ++dom) {
            auto & domain = candidate[dom];
            uint32_t n_pages; io.read(&n_pages, sizeof(n_pages));
            if (n_pages > max_blocks) { throw std::runtime_error("invalid paged KV state page count"); }
            for (uint32_t j = 0; j < n_pages; ++j) {
                uint32_t metadata[4]; io.read(metadata, sizeof(metadata));
                const uint32_t valid_mask = UINT32_MAX >> (32 - block_size);
                if (metadata[0] >= max_blocks || !metadata[1] || (metadata[1] & ~valid_mask) || domain.seqs[seq].count(metadata[0])) {
                    throw std::runtime_error("invalid paged KV state page metadata");
                }
                auto & restored = restored_pages[dom];
                const auto old = restored.find(metadata[2]);
                if (!metadata[3]) {
                    if (old == restored.end() || old->second.second != metadata[0]) { throw std::runtime_error("invalid paged KV state page alias"); }
                    domain.pool.retain_gpu_block(old->second.first);
                    domain.seqs[seq][metadata[0]] = {old->second.first, metadata[1]};
                    continue;
                }
                if (metadata[3] != 1 || old != restored.end()) { throw std::runtime_error("invalid paged KV state page record"); }
                const auto ids = domain.pool.checkout_gpu_blocks(1);
                if (ids.empty()) { throw std::runtime_error("paged KV state exceeds free page budget"); }
                restored[metadata[2]] = {ids[0], metadata[0]};
                domain.seqs[seq][metadata[0]] = {ids[0], metadata[1]};
                pending_page p; p.physical = ids[0];
                for (uint32_t il = 0; il < layers.size(); ++il) {
                    const auto & l = layers[il];
                    if (l.swa != bool(dom)) { continue; }
                    std::vector<uint8_t> bytes(l.cache->nb[3]);
                    io.read(bytes.data(), bytes.size());
                    p.data.emplace_back(il, std::move(bytes));
                }
                pending.push_back(std::move(p));
            }
        }
    }
    for (const auto & p : pending) {
        for (const auto & data : p.data) {
            auto * cache = layers[data.first].cache;
            ggml_backend_tensor_set(cache, data.second.data(), size_t(p.physical)*cache->nb[3], data.second.size());
        }
    }
    current = std::move(candidate);
}
