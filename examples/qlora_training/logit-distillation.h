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
#include <thread>
#include <atomic>
#include <mutex>
#include <exception>
#include <sstream>
#include <filesystem>

namespace logit_distillation {

inline void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("distillation: " + message);
}

inline void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(std::string("distillation: ") + message);
}

inline void check_endian() {
    const uint32_t one = 1;
    require(*reinterpret_cast<const uint8_t *>(&one) == 1, "cache requires a little-endian host");
}

inline void write(std::ostream & out, const void * data, size_t size) {
    if (!size) return;
    out.write(static_cast<const char *>(data), size);
    require(bool(out), "cache write failed");
}

inline void read(std::istream & in, void * data, size_t size) {
    if (!size) return;
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

inline uint64_t resume_cache(const std::string & path, const header & expected,
        const int32_t * inputs, const int32_t * labels, uint64_t & rows) {
    std::ifstream in(path, std::ios::binary);
    header actual;
    actual.load(in);
    std::ostringstream a(std::ios::binary), b(std::ios::binary);
    expected.save(a);
    actual.save(b);
    require(a.str() == b.str(), "partial cache configuration differs");
    const uint64_t length = std::filesystem::file_size(path);
    std::vector<int32_t> tokens(expected.context), targets(expected.context);
    rows = 0;
    for (uint64_t w = 0; w < expected.windows; ++w) {
        const uint64_t start = in.tellg();
        require(start <= length, "invalid partial cache offset");
        if (uint64_t(expected.context)*8 > length - start) {
            in.close();
            std::filesystem::resize_file(path, start);
            return w;
        }
        read(in, tokens.data(), tokens.size()*4);
        read(in, targets.data(), targets.size()*4);
        const size_t base = size_t(w)*expected.context;
        require(std::memcmp(tokens.data(), inputs + base, tokens.size()*4) == 0 &&
                std::memcmp(targets.data(), labels + base, targets.size()*4) == 0, "partial cache input/mask differs");
        const uint64_t count = std::count_if(targets.begin(), targets.end(), [](int32_t t) { return t >= 0; });
        const uint64_t position = in.tellg();
        if (count > (length - position)/expected.row_bytes()) {
            in.close();
            std::filesystem::resize_file(path, start);
            return w;
        }
        in.seekg(count*expected.row_bytes(), std::ios::cur);
        require(bool(in), "partial cache seek failed");
        rows += count;
    }
    require(uint64_t(in.tellg()) == length, "partial cache has extra data");
    return expected.windows;
}

class row_encoder {
    const header & h;
    std::vector<int32_t> ids;
    std::vector<uint8_t> selected;
    std::vector<ggml_fp16_t> logp;
public:
    explicit row_encoder(const header & info) : h(info), ids(h.vocab), selected(h.vocab, 0), logp(h.top_k) {
        std::iota(ids.begin(), ids.end(), 0);
    }

    double encode(uint8_t * output, const float * logits) {
        for (uint32_t i = 0; i < h.vocab; ++i) require(std::isfinite(logits[i]), "non-finite teacher logits");
        const auto greater = [&](int32_t a, int32_t b) {
            return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
        };
        std::partial_sort(ids.begin(), ids.begin() + h.top_k, ids.end(), greater);
        const double maximum = logits[ids[0]];
        for (uint32_t i = 0; i < h.top_k; ++i) selected[ids[i]] = 1;
        double total = 0;
        double tail = 0;
        for (uint32_t i = 0; i < h.vocab; ++i) {
            const double p = std::exp((double(logits[i]) - maximum) / h.temperature);
            total += p;
            if (!selected[i]) tail += p;
        }
        const double log_total = std::log(total);
        for (uint32_t i = 0; i < h.top_k; ++i) {
            const float value = (double(logits[ids[i]]) - maximum) / h.temperature - log_total;
            logp[i] = ggml_fp32_to_fp16(value);
            require(std::isfinite(ggml_fp16_to_fp32(logp[i])), "top-k logp exceeds FP16 range");
            selected[ids[i]] = 0;
        }
        const float tail_logp = std::log(std::max(tail, 1e-300)) - log_total;
        std::memcpy(output, ids.data(), size_t(h.top_k)*4);
        std::memcpy(output + size_t(h.top_k)*4, logp.data(), size_t(h.top_k)*2);
        std::memcpy(output + size_t(h.top_k)*6, &tail_logp, 4);
        return tail / total;
    }
};

inline double encode_row(std::ostream & out, const float * logits, const header & h) {
    row_encoder encoder(h);
    std::vector<uint8_t> data(h.row_bytes());
    const double tail = encoder.encode(data.data(), logits);
    write(out, data.data(), data.size());
    return tail;
}

inline std::vector<double> encode_rows(std::ostream & out, const std::vector<const float *> & logits,
        const header & h, int threads) {
    require(logits.size() <= SIZE_MAX/h.row_bytes(), "encoded batch is too large");
    std::vector<uint8_t> data(logits.size()*h.row_bytes());
    std::vector<double> tails(logits.size());
    std::atomic<size_t> next {0};
    std::mutex mutex;
    std::exception_ptr failure;
    const auto worker = [&]() {
        try {
            row_encoder encoder(h);
            for (size_t i = next.fetch_add(1); i < logits.size(); i = next.fetch_add(1)) {
                tails[i] = encoder.encode(data.data() + i*h.row_bytes(), logits[i]);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = std::current_exception();
            next.store(logits.size());
        }
    };
    std::vector<std::thread> workers;
    const int count = std::min<size_t>(std::max(1, threads), logits.size());
    try {
        for (int i = 1; i < count; ++i) workers.emplace_back(worker);
        if (count) worker();
    } catch (...) {
        next.store(logits.size());
        for (auto & thread : workers) thread.join();
        throw;
    }
    for (auto & thread : workers) thread.join();
    if (failure) std::rethrow_exception(failure);
    write(out, data.data(), data.size());
    return tails;
}

class cache {
public:
    header info;
    std::ifstream in;
    std::vector<uint64_t> offsets;
    std::vector<uint8_t> row_buffer;
    std::vector<int32_t> sorted_ids;
    uint64_t next_offset = 0;

    explicit cache(const std::string & path) : in(path, std::ios::binary) {
        require(bool(in), "cannot open " + path);
        info.load(in);
        row_buffer.resize(info.row_bytes());
        sorted_ids.resize(info.top_k);
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
            uint64_t offset = in.tellg();
            for (uint32_t j = 0; j < info.context; ++j) {
                require(tokens[j] >= 0 && uint32_t(tokens[j]) < info.vocab && targets[j] >= -1 &&
                        targets[j] < int64_t(info.vocab), "invalid token or target ID");
                if (targets[j] < 0) continue;
                require(offset <= length && info.row_bytes() <= length - offset, "truncated teacher row");
                offsets[base + j] = offset;
                offset += info.row_bytes();
            }
            in.seekg(offset);
            require(bool(in), "cannot seek teacher cache");
        }
        require(uint64_t(in.tellg()) == length, "unexpected trailing cache data");
        ggml_opt_dataset_set_aux(dataset, offsets.data(), size_t(info.context)*sizeof(uint64_t));
        next_offset = 0;
    }

    void row(uint64_t offset, int32_t * ids, float * logp) {
        require(offset > 0 && offset <= uint64_t(std::numeric_limits<std::streamoff>::max()), "missing teacher row");
        if (offset != next_offset) {
            in.clear();
            in.seekg(offset);
        }
        read(in, row_buffer.data(), row_buffer.size());
        next_offset = offset + row_buffer.size();
        std::memcpy(ids, row_buffer.data(), size_t(info.top_k)*4);
        std::memcpy(logp + info.top_k, row_buffer.data() + size_t(info.top_k)*6, 4);
        std::copy_n(ids, info.top_k, sorted_ids.begin());
        std::sort(sorted_ids.begin(), sorted_ids.end());
        require(sorted_ids.front() >= 0 && uint32_t(sorted_ids.back()) < info.vocab &&
                std::adjacent_find(sorted_ids.begin(), sorted_ids.end()) == sorted_ids.end(), "invalid or duplicate teacher IDs");
        double total = 0;
        for (uint32_t i = 0; i <= info.top_k; ++i) {
            if (i < info.top_k) {
                ggml_fp16_t value;
                std::memcpy(&value, row_buffer.data() + size_t(info.top_k)*4 + i*2, 2);
                logp[i] = ggml_fp16_to_fp32(value);
            }
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
