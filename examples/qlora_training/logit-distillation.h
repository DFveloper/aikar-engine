#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-opt.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace logit_distillation {

inline void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("distillation: " + message);
}

inline void check_endian() {
    const uint32_t one = 1;
    require(*reinterpret_cast<const uint8_t *>(&one) == 1, "cache requires a little-endian host");
}

inline void write(std::ostream & out, const void * data, size_t size) {
    out.write(static_cast<const char *>(data), size);
    require(bool(out), "cache write failed");
}

inline void read(std::istream & in, void * data, size_t size) {
    in.read(static_cast<char *>(data), size);
    require(bool(in), "truncated cache");
}

inline uint64_t vocab_hash(const llama_vocab * vocab) {
    uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&](const void * ptr, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(ptr);
        for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
    };
    const int32_t n = llama_vocab_n_tokens(vocab);
    mix(&n, sizeof(n));
    for (int32_t i = 0; i < n; ++i) {
        const char * text = llama_vocab_get_text(vocab, i);
        const float score = llama_vocab_get_score(vocab, i);
        const int32_t attr = llama_vocab_get_attr(vocab, i);
        mix(text, std::strlen(text) + 1);
        mix(&score, sizeof(score));
        mix(&attr, sizeof(attr));
    }
    const int32_t special[] = {llama_vocab_bos(vocab), llama_vocab_eos(vocab),
        int32_t(llama_vocab_get_add_bos(vocab)), int32_t(llama_vocab_get_add_eos(vocab))};
    mix(special, sizeof(special));
    return hash;
}

struct header {
    uint32_t context = 0;
    uint32_t vocab = 0;
    uint32_t top_k = 0;
    float temperature = 1.0f;
    uint64_t tokenizer = 0;
    uint64_t windows = 0;
    uint64_t train_windows = 0;
    std::string metadata;

    size_t row_bytes() const { return size_t(top_k)*6 + 4; }

    void save(std::ostream & out) const {
        check_endian();
        write(out, "QLKD0001", 8);
        write(out, &context, 4);
        write(out, &vocab, 4);
        write(out, &top_k, 4);
        write(out, &temperature, 4);
        write(out, &tokenizer, 8);
        write(out, &windows, 8);
        write(out, &train_windows, 8);
        require(metadata.size() <= 1024*1024, "metadata is too large");
        const uint32_t length = metadata.size();
        write(out, &length, 4);
        write(out, metadata.data(), length);
    }

    void load(std::istream & in) {
        check_endian();
        char magic[8];
        read(in, magic, 8);
        require(std::memcmp(magic, "QLKD0001", 8) == 0, "invalid cache version");
        read(in, &context, 4);
        read(in, &vocab, 4);
        read(in, &top_k, 4);
        read(in, &temperature, 4);
        read(in, &tokenizer, 8);
        read(in, &windows, 8);
        read(in, &train_windows, 8);
        require(context > 0 && context <= 1048576 && vocab > 1 && vocab <= 16777216 &&
                top_k > 0 && top_k < vocab && top_k <= 65536, "invalid cache dimensions");
        require(std::isfinite(temperature) && temperature >= 0.01f && temperature <= 100, "invalid temperature");
        require(train_windows > 0 && train_windows <= windows, "invalid training partition");
        uint32_t length;
        read(in, &length, 4);
        require(length <= 1024*1024, "invalid metadata length");
        metadata.resize(length);
        read(in, metadata.data(), length);
    }
};

inline double encode_row(std::ostream & out, const float * logits, const header & h) {
    std::vector<int32_t> ids(h.vocab);
    std::iota(ids.begin(), ids.end(), 0);
    for (uint32_t i = 0; i < h.vocab; ++i) require(std::isfinite(logits[i]), "non-finite teacher logits");
    const auto greater = [&](int32_t a, int32_t b) {
        return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
    };
    std::partial_sort(ids.begin(), ids.begin() + h.top_k, ids.end(), greater);
    const double maximum = logits[ids[0]];
    std::vector<uint8_t> selected(h.vocab, 0);
    for (uint32_t i = 0; i < h.top_k; ++i) selected[ids[i]] = 1;
    double total = 0;
    double tail = 0;
    for (uint32_t i = 0; i < h.vocab; ++i) {
        const double p = std::exp((double(logits[i]) - maximum) / h.temperature);
        total += p;
        if (!selected[i]) tail += p;
    }
    const double log_total = std::log(total);
    std::vector<ggml_fp16_t> logp(h.top_k);
    for (uint32_t i = 0; i < h.top_k; ++i) {
        const float value = (double(logits[ids[i]]) - maximum) / h.temperature - log_total;
        logp[i] = ggml_fp32_to_fp16(value);
        require(std::isfinite(ggml_fp16_to_fp32(logp[i])), "top-k logp exceeds FP16 range");
    }
    const float tail_logp = std::log(std::max(tail, 1e-300)) - log_total;
    write(out, ids.data(), size_t(h.top_k)*sizeof(ids[0]));
    write(out, logp.data(), logp.size()*sizeof(logp[0]));
    write(out, &tail_logp, sizeof(tail_logp));
    return tail / total;
}

class cache {
public:
    header info;
    std::ifstream in;
    std::vector<uint64_t> offsets;

    explicit cache(const std::string & path) : in(path, std::ios::binary) {
        require(bool(in), "cannot open " + path);
        info.load(in);
    }

    void attach(ggml_opt_dataset_t dataset, int64_t split, const llama_vocab * vocab) {
        require(info.vocab == uint32_t(llama_vocab_n_tokens(vocab)) && info.tokenizer == vocab_hash(vocab), "tokenizer mismatch");
        auto * data = ggml_opt_dataset_data(dataset);
        auto * labels = ggml_opt_dataset_labels(dataset);
        require(data->type == GGML_TYPE_I32 && labels && labels->type == GGML_TYPE_I32, "dataset must contain token IDs");
        require(info.context == data->ne[0] && info.windows == uint64_t(data->ne[1]) &&
                info.train_windows == uint64_t(split), "context, window count or train/validation partition mismatch");
        const auto start = in.tellg();
        in.seekg(0, std::ios::end);
        const uint64_t length = in.tellg();
        in.seekg(start);
        require(ggml_nelements(data) <= int64_t(SIZE_MAX / sizeof(uint64_t)), "dataset is too large");
        offsets.assign(size_t(ggml_nelements(data)), 0);
        std::vector<int32_t> tokens(info.context), targets(info.context);
        for (uint64_t w = 0; w < info.windows; ++w) {
            read(in, tokens.data(), tokens.size()*4);
            read(in, targets.data(), targets.size()*4);
            const size_t base = size_t(w)*info.context;
            require(std::memcmp(tokens.data(), static_cast<int32_t *>(data->data) + base, tokens.size()*4) == 0 &&
                    std::memcmp(targets.data(), static_cast<int32_t *>(labels->data) + base, targets.size()*4) == 0,
                    "packed input or prediction mask mismatch at window " + std::to_string(w));
            for (uint32_t j = 0; j < info.context; ++j) {
                require(tokens[j] >= 0 && uint32_t(tokens[j]) < info.vocab && targets[j] >= -1 &&
                        targets[j] < int64_t(info.vocab), "invalid token or target ID");
                if (targets[j] < 0) continue;
                const uint64_t offset = in.tellg();
                require(offset <= length && info.row_bytes() <= length - offset, "truncated teacher row");
                offsets[base + j] = offset;
                in.seekg(info.row_bytes(), std::ios::cur);
                require(bool(in), "cannot seek teacher cache");
            }
        }
        require(uint64_t(in.tellg()) == length, "unexpected trailing cache data");
        ggml_opt_dataset_set_aux(dataset, offsets.data(), size_t(info.context)*sizeof(uint64_t));
    }

    void row(uint64_t offset, int32_t * ids, float * logp) {
        require(offset > 0 && offset <= uint64_t(std::numeric_limits<std::streamoff>::max()), "missing teacher row");
        in.clear();
        in.seekg(offset);
        read(in, ids, size_t(info.top_k)*4);
        std::vector<ggml_fp16_t> values(info.top_k);
        read(in, values.data(), values.size()*2);
        read(in, logp + info.top_k, 4);
        std::vector<int32_t> sorted(ids, ids + info.top_k);
        std::sort(sorted.begin(), sorted.end());
        require(sorted.front() >= 0 && uint32_t(sorted.back()) < info.vocab &&
                std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "invalid or duplicate teacher IDs");
        double total = 0;
        for (uint32_t i = 0; i <= info.top_k; ++i) {
            if (i < info.top_k) logp[i] = ggml_fp16_to_fp32(values[i]);
            require(std::isfinite(logp[i]) && logp[i] <= 0.01f, "invalid teacher log-probability");
            total += std::exp(double(logp[i]));
        }
        require(total > 0 && std::abs(total - 1) < 0.02, "teacher distribution is not normalized");
        const float correction = std::log(total);
        for (uint32_t i = 0; i <= info.top_k; ++i) logp[i] -= correction;
    }

    static void fetch(void * userdata, uint64_t offset, int32_t * ids, float * logp) {
        static_cast<cache *>(userdata)->row(offset, ids, logp);
    }
};

} // namespace logit_distillation
