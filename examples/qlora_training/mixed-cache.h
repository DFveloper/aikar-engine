#pragma once
#include "logit-distillation.h"
#include "nlohmann/json.hpp"
#include <memory>

// A virtual cache: retain each source's packed tokens, masks and logits exactly.
// Only the window order changes; no logits are copied to a new disk cache.
class mixed_cache {
    struct window { uint64_t offset; uint64_t bytes; };
    std::vector<std::unique_ptr<logit_distillation::cache>> sources;
    std::vector<std::vector<window>> windows;
    std::vector<uint64_t> offsets;

public:
    logit_distillation::header info;

    explicit mixed_cache(const std::string & path) {
        using logit_distillation::require;
        std::ifstream manifest(path);
        nlohmann::json config;
        manifest >> config;
        require(config.at("format") == "lumen-mixed-cache-v1", "unknown mixed manifest");
        const auto & paths = config.at("caches");
        require(paths.is_array() && paths.size() >= 2 && paths.size() <= 255, "use 2 to 255 caches");
        for (const auto & entry : paths) {
            auto source = std::make_unique<logit_distillation::cache>(entry.at("path").get<std::string>());
            const auto & h = source->info;
            if (sources.empty()) {
                info = h;
                info.windows = info.train_windows = 0;
            }
            require(h.context == info.context && h.vocab == info.vocab && h.top_k == info.top_k &&
                    h.temperature == info.temperature && h.tokenizer == info.tokenizer,
                    "source cache dimensions/tokenizer differ");
            require(h.windows == entry.at("windows").get<uint64_t>() &&
                    h.train_windows == entry.at("train_windows").get<uint64_t>(), "source partition changed");
            const auto length = std::filesystem::file_size(entry.at("path").get<std::string>());
            std::vector<window> index;
            std::vector<int32_t> labels(h.context);
            for (uint64_t i = 0; i < h.windows; ++i) {
                const uint64_t start = source->in.tellg();
                source->in.seekg(uint64_t(h.context)*4, std::ios::cur);
                logit_distillation::read(source->in, labels.data(), labels.size()*4);
                const auto rows = std::count_if(labels.begin(), labels.end(), [](int32_t v) { return v >= 0; });
                const uint64_t bytes = uint64_t(h.context)*8 + rows*h.row_bytes();
                require(start <= length && bytes <= length - start, "truncated source cache");
                index.push_back({start, bytes});
                source->in.seekg(start + bytes);
            }
            require(uint64_t(source->in.tellg()) == length, "extra source cache data");
            require(h.windows <= uint64_t(INT64_MAX)/h.context - info.windows, "mixed dataset is too large");
            info.windows += h.windows;
            info.train_windows += h.train_windows;
            windows.push_back(std::move(index));
            sources.push_back(std::move(source));
        }
    }

    ggml_opt_dataset_t dataset(int32_t context, int64_t & split, int32_t limit = 0, int32_t validation_limit = 0) {
        using logit_distillation::require;
        require(context == int32_t(info.context), "mixed context mismatch");
        const uint64_t train_count = limit > 0 ? std::min<uint64_t>(limit, info.train_windows) : info.train_windows;
        uint64_t val_count = info.windows - info.train_windows;
        if (limit > 0) val_count = std::min<uint64_t>(limit, val_count);
        if (validation_limit > 0) val_count = std::min<uint64_t>(validation_limit, val_count);
        const uint64_t total = train_count + val_count;
        require(total <= SIZE_MAX / (size_t(context)*sizeof(uint64_t)), "mixed offsets are too large");
        auto result = ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, context, context, total, 1);
        require(result != nullptr, "mixed dataset allocation failed");
        auto * tokens = static_cast<int32_t *>(ggml_opt_dataset_data(result)->data);
        auto * labels = static_cast<int32_t *>(ggml_opt_dataset_labels(result)->data);
        offsets.assign(size_t(total)*context, 0);
        uint64_t destination = 0;
        // Interleave TRAIN and VAL independently. Midpoint ordering spreads the
        // smaller source evenly without repeating or dropping any window.
        for (int partition = 0; partition < 2; ++partition) {
            const size_t n = sources.size();
            std::vector<uint64_t> cursor(n, 0), count(n), begin(n);
            for (size_t s = 0; s < n; ++s) {
                const auto & h = sources[s]->info;
                begin[s] = partition ? h.train_windows : 0;
                count[s] = partition ? h.windows - h.train_windows : h.train_windows;
            }
            const uint64_t end = partition ? total : train_count;
            while (destination < end) {
                size_t s = n;
                for (size_t candidate = 0; candidate < n; ++candidate) {
                    if (cursor[candidate] == count[candidate]) continue;
                    if (s == n || static_cast<long double>(2*cursor[candidate]+1)*count[s] <
                                  static_cast<long double>(2*cursor[s]+1)*count[candidate]) s = candidate;
                }
                if (s == n) break;
                auto & source = *sources[s];
                const auto & w = windows[s][begin[s] + cursor[s]++];
                source.in.seekg(w.offset);
                const size_t base = destination++ * context;
                logit_distillation::read(source.in, tokens + base, size_t(context)*4);
                logit_distillation::read(source.in, labels + base, size_t(context)*4);
                uint64_t offset = source.in.tellg();
                for (int32_t j = 0; j < context; ++j) {
                    require(tokens[base+j] >= 0 && uint32_t(tokens[base+j]) < info.vocab &&
                            labels[base+j] >= -1 && labels[base+j] < int64_t(info.vocab), "invalid source token");
                    if (labels[base+j] < 0) continue;
                    // Encode source and original byte offset as offset*N+source.
                    // The native row decoder still reads the original file.
                    require(offset < UINT64_MAX/n, "source offset overflow");
                    offsets[base+j] = offset*n + s;
                    offset += info.row_bytes();
                }
            }
        }
        info.windows = total;
        info.train_windows = train_count;
        split = train_count;
        ggml_opt_dataset_set_aux(result, offsets.data(), size_t(info.context)*sizeof(uint64_t));
        return result;
    }

    void attach(ggml_opt_dataset_t dataset, int64_t split, const llama_vocab * vocab) {
        logit_distillation::require(info.vocab == uint32_t(llama_vocab_n_tokens(vocab)) &&
            info.tokenizer == logit_distillation::vocab_hash(vocab), "mixed tokenizer mismatch");
        logit_distillation::require(uint64_t(split) == info.train_windows &&
            uint64_t(ggml_opt_dataset_ndata(dataset)) == info.windows, "mixed partition mismatch");
        ggml_opt_dataset_set_aux(dataset, offsets.data(), size_t(info.context)*sizeof(uint64_t));
    }

    static void fetch(void * userdata, uint64_t encoded, int32_t * ids, float * logp) {
        auto & self = *static_cast<mixed_cache *>(userdata);
        const size_t n = self.sources.size();
        self.sources[encoded % n]->row(encoded / n, ids, logp);
    }
};
