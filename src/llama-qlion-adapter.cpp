#include "llama-qlion-adapter.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <sys/stat.h>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr size_t chunk_size = 4 * 1024 * 1024;
using reader = std::function<void(size_t, void *, size_t)>;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error("QLion adapter: " + message);
    }
}

std::string digest(sha256_t & hash) {
    unsigned char bytes[SHA256_DIGEST_SIZE];
    sha256_final(&hash, bytes);
    const char * hex = "0123456789abcdef";
    std::string result = "sha256:";
    for (auto byte : bytes) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}

std::string hash_data(const reader & read, size_t size, std::ostream * out = nullptr) {
    sha256_t hash;
    sha256_init(&hash);
    std::vector<unsigned char> buffer(std::min(size, chunk_size));
    for (size_t offset = 0; offset < size;) {
        const size_t n = std::min(buffer.size(), size - offset);
        read(offset, buffer.data(), n);
        sha256_update(&hash, buffer.data(), n);
        if (out) {
            out->write(reinterpret_cast<const char *>(buffer.data()), n);
            require(bool(*out), "write failed");
        }
        offset += n;
    }
    return digest(hash);
}

bool equal_data(const reader & a, const reader & b, size_t size) {
    std::vector<unsigned char> left(std::min(size, chunk_size)), right(left.size());
    for (size_t offset = 0; offset < size;) {
        const size_t n = std::min(left.size(), size - offset);
        a(offset, left.data(), n);
        b(offset, right.data(), n);
        if (memcmp(left.data(), right.data(), n) != 0) return false;
        offset += n;
    }
    return true;
}

reader tensor_reader(const ggml_tensor * tensor, size_t start = 0) {
    return [tensor, start](size_t offset, void * data, size_t n) {
        ggml_backend_tensor_get(tensor, data, start + offset, n);
    };
}

struct file {
    std::string path;
    std::ifstream stream;
    gguf_context_ptr meta;
    ggml_context_ptr tensors;
    size_t size;
    std::filesystem::file_time_type modified;
    struct stat identity {};

    explicit file(const std::string & path) : path(path), stream(path, std::ios::binary) {
        require(bool(stream), "cannot open " + path);
        require(stat(path.c_str(), &identity) == 0, "cannot stat " + path);
        const auto file_size = std::filesystem::file_size(path);
        require(file_size <= SIZE_MAX && file_size <= INT64_MAX, "file is too large");
        size = size_t(file_size);
        modified = std::filesystem::last_write_time(path);
        ggml_context * ctx = nullptr;
        meta.reset(gguf_init_from_file(path.c_str(), { true, &ctx }));
        tensors.reset(ctx);
        require(bool(meta), "invalid GGUF: " + path);
        require(gguf_find_key(meta.get(), "split.count") < 0 && gguf_find_key(meta.get(), "split.no") < 0,
                "split GGUF is unsupported");
        std::map<std::string, bool> names;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            const std::string name = gguf_get_tensor_name(meta.get(), i);
            require(names.emplace(name, true).second, "duplicate logical tensor name");
            const auto * t = tensor(i);
            for (auto ne : t->ne) {
                require(ne > 0, "empty tensor: " + name);
            }
            require(uint64_t(t->ne[0]) <= SIZE_MAX / ggml_type_size(t->type), "row size overflow");
            size_t bytes = ggml_row_size(t->type, t->ne[0]);
            require(bytes > 0, "invalid row size");
            for (int d = 1; d < GGML_MAX_DIMS; ++d) {
                require(uint64_t(t->ne[d]) <= SIZE_MAX / bytes, "tensor size overflow");
                bytes *= t->ne[d];
            }
            require(ggml_nbytes(t) == bytes, "noncontiguous tensor layout");
            const size_t offset = gguf_get_tensor_offset(meta.get(), i);
            const size_t data_offset = gguf_get_data_offset(meta.get());
            require(data_offset <= size && offset <= size - data_offset &&
                    ggml_nbytes(t) <= size - data_offset - offset, "truncated tensor: " + std::string(t->name));
        }
        check_identity();
    }

    ggml_tensor * tensor(int64_t i) const {
        return ggml_get_tensor(tensors.get(), gguf_get_tensor_name(meta.get(), i));
    }

    size_t offset(int64_t i) const {
        return gguf_get_data_offset(meta.get()) + gguf_get_tensor_offset(meta.get(), i);
    }

    reader read(size_t start = 0) {
        return [this, start](size_t offset, void * data, size_t n) {
            require(start <= size && offset <= size - start && n <= size - start - offset, "read out of range");
            stream.clear();
            stream.seekg(start + offset);
            stream.read(static_cast<char *>(data), n);
            require(bool(stream), "read failed: " + path);
        };
    }

    std::string hash() {
        return hash_data(read(), size);
    }

    void check_identity() const {
        struct stat current {};
        require(stat(path.c_str(), &current) == 0 && current.st_dev == identity.st_dev &&
                current.st_ino == identity.st_ino && current.st_ctime == identity.st_ctime, "file identity changed: " + path);
#if defined(__APPLE__)
        require(current.st_ctimespec.tv_nsec == identity.st_ctimespec.tv_nsec, "file change time differs: " + path);
#elif !defined(_WIN32)
        require(current.st_ctim.tv_nsec == identity.st_ctim.tv_nsec, "file change time differs: " + path);
#endif
        require(std::filesystem::file_size(path) == size && std::filesystem::last_write_time(path) == modified,
                "file changed during use: " + path);
    }
};

std::string get_string(const gguf_context * meta, const std::string & key) {
    const int64_t id = gguf_find_key(meta, key.c_str());
    require(id >= 0 && gguf_get_kv_type(meta, id) == GGUF_TYPE_STRING, "missing or invalid " + key);
    return gguf_get_val_str(meta, id);
}

int64_t get_array(const gguf_context * meta, const std::string & key, gguf_type type, size_t count) {
    const int64_t id = gguf_find_key(meta, key.c_str());
    require(id >= 0 && gguf_get_kv_type(meta, id) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(meta, id) == type && gguf_get_arr_n(meta, id) == count, "invalid " + key);
    return id;
}

bool valid_hash(const std::string & hash) {
    return hash.size() == 71 && hash.compare(0, 7, "sha256:") == 0 &&
        hash.find_first_not_of("0123456789abcdef", 7) == std::string::npos;
}

std::string get_hash(const gguf_context * meta, int64_t key, size_t index) {
    const std::string hash = gguf_get_arr_str(meta, key, index);
    require(valid_hash(hash), "invalid SHA-256");
    return hash;
}

bool supported(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_MXFP4;
}

void distinct_output(const std::string & output, const std::string & input) {
    require(std::filesystem::weakly_canonical(output) != std::filesystem::canonical(input) &&
            (!std::filesystem::exists(output) || !std::filesystem::equivalent(output, input)), "output aliases input");
}

struct output_file {
    std::string path;
    std::string temporary;
    std::string directory;
    std::ofstream stream;

    explicit output_file(const std::string & path) : path(path) {
        static std::atomic<uint64_t> id { 0 };
#if defined(_WIN32)
        const int pid = _getpid();
#else
        const int pid = getpid();
#endif
        directory = path + "." + std::to_string(pid) + "." + std::to_string(id++) + ".tmp";
        require(std::filesystem::create_directory(directory), "cannot reserve temporary output directory");
        try {
            std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
            temporary = directory + "/output.gguf";
            stream.open(temporary, std::ios::binary | std::ios::trunc);
            require(bool(stream), "cannot create " + temporary);
        } catch (...) {
            std::error_code error;
            std::filesystem::remove(directory, error);
            throw;
        }
    }

    ~output_file() {
        stream.close();
        std::remove(temporary.c_str());
        std::error_code error;
        std::filesystem::remove(directory, error);
    }

    void commit() {
        stream.flush();
        require(bool(stream), "flush failed");
        stream.close();
        require(!stream.fail(), "close failed");
        require(std::rename(temporary.c_str(), path.c_str()) == 0, "cannot replace " + path);
    }
};

struct replacement {
    std::string name;
    size_t base_offset;
    size_t tensor_offset;
    size_t size;
    size_t source_offset;
    file * source;
    std::string hash;
};

struct validated {
    file base;
    std::vector<std::unique_ptr<file>> adapters;
    std::vector<replacement> replacements;
    std::string base_hash;

    validated(const std::string & path, const std::vector<std::string> & paths, bool compatible) : base(path) {
        require(!paths.empty(), "no adapters specified");
        base_hash = base.hash();
        const auto architecture = get_string(base.meta.get(), "general.architecture");
        for (const auto & path_adapter : paths) {
            adapters.emplace_back(new file(path_adapter));
            file & adapter = *adapters.back();
            const auto * meta = adapter.meta.get();
            require(get_string(meta, "general.type") == "adapter" &&
                    get_string(meta, "adapter.type") == "qlion", "not a QLion adapter");
            const int64_t version_key = gguf_find_key(meta, "qlion.adapter.version");
            require(version_key >= 0 && gguf_get_kv_type(meta, version_key) == GGUF_TYPE_UINT32, "invalid adapter version");
            const uint32_t version = gguf_get_val_u32(meta, version_key);
            require(version == 1 || version == 2, "unsupported adapter version");
            require(get_string(meta, "qlion.adapter.layout") == (version == 1 ? "full" : "units"), "invalid layout");
            require(get_string(meta, "general.architecture") == architecture, "architecture mismatch");
            const auto expected_base_hash = get_string(meta, "qlion.adapter.base_sha256");
            require(valid_hash(expected_base_hash), "invalid base SHA-256");
            const bool exact = expected_base_hash == base_hash;
            require(exact || compatible, "base file hash mismatch (use --allow-compatible-base for tensor validation)");
            if (!exact) {
                LLAMA_LOG_WARN("QLion adapter: base file differs; validating replacement units only; other tensors may differ\n");
            }
            const size_t count = gguf_get_n_tensors(meta);
            const int64_t hashes = get_array(meta, "qlion.adapter.tensor_payload_sha256", GGUF_TYPE_STRING, count);
            const int64_t base_hashes = version == 1 ? get_array(meta, "qlion.adapter.tensor_base_sha256", GGUF_TYPE_STRING, count) : -1;
            for (size_t i = 0; i < count; ++i) {
                const auto * payload = adapter.tensor(i);
                require(supported(payload->type), "unsupported replacement type");
                const int64_t base_id = gguf_find_tensor(base.meta.get(), payload->name);
                require(base_id >= 0, "missing base tensor: " + std::string(payload->name));
                const auto * original = base.tensor(base_id);
                require(original->type == payload->type, "tensor type mismatch");
                const auto prefix = "qlion.adapter.tensor." + std::to_string(i) + ".";
                const int64_t rows_key = gguf_find_key(meta, (prefix + "rows").c_str());
                const int64_t experts_key = gguf_find_key(meta, (prefix + "experts").c_str());
                require(hash_data(adapter.read(adapter.offset(i)), ggml_nbytes(payload)) == get_hash(meta, hashes, i), "payload hash mismatch");
                if (version == 1) {
                    require(rows_key < 0 && experts_key < 0 && gguf_find_key(meta, (prefix + "shape").c_str()) < 0,
                            "v1 partial replacements are unsupported");
                    require(ggml_are_same_shape(original, payload), "tensor shape mismatch");
                    const auto expected = get_hash(meta, base_hashes, i);
                    require(hash_data(base.read(base.offset(base_id)), ggml_nbytes(original)) == expected, "base tensor hash mismatch");
                    replacements.push_back({ payload->name, base.offset(base_id), 0, ggml_nbytes(payload), adapter.offset(i), &adapter, get_hash(meta, hashes, i) });
                } else {
                    require((rows_key >= 0) != (experts_key >= 0), "exactly one unit index is required");
                    const int64_t shape_key = get_array(meta, prefix + "shape", GGUF_TYPE_UINT64, GGML_MAX_DIMS);
                    const auto * shape = static_cast<const uint64_t *>(gguf_get_arr_data(meta, shape_key));
                    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                        require(shape[d] == uint64_t(original->ne[d]), "original shape mismatch");
                    }
                    const bool experts = experts_key >= 0;
                    const int64_t indices_key = experts ? experts_key : rows_key;
                    require(gguf_get_kv_type(meta, indices_key) == GGUF_TYPE_ARRAY && gguf_get_arr_type(meta, indices_key) == GGUF_TYPE_UINT64, "invalid unit indices");
                    const size_t n = gguf_get_arr_n(meta, indices_key);
                    const size_t row_size = ggml_row_size(original->type, original->ne[0]);
                    const size_t limit = experts ? original->ne[2] : ggml_nrows(original);
                    require(n > 0 && n <= limit && (!experts || original->ne[3] == 1), "invalid unit count or shape");
                    const size_t unit_size = experts ? row_size * original->ne[1] : row_size;
                    require(payload->ne[0] == original->ne[0] && payload->ne[3] == 1 &&
                            (experts ? payload->ne[1] == original->ne[1] && payload->ne[2] == int64_t(n) :
                            payload->ne[1] == int64_t(n) && payload->ne[2] == 1), "payload shape mismatch");
                    const auto * indices = static_cast<const uint64_t *>(gguf_get_arr_data(meta, indices_key));
                    const int64_t unit_hashes = get_array(meta, prefix + "base_unit_sha256", GGUF_TYPE_STRING, n);
                    for (size_t j = 0; j < n; ++j) {
                        require(indices[j] < limit && (j == 0 || indices[j-1] < indices[j]), "unit indices must be sorted, unique and in range");
                        const size_t offset = indices[j] * unit_size;
                        require(hash_data(base.read(base.offset(base_id) + offset), unit_size) == get_hash(meta, unit_hashes, j), "base unit hash mismatch");
                        const size_t source_offset = adapter.offset(i) + j * unit_size;
                        replacements.push_back({ payload->name, base.offset(base_id) + offset, offset, unit_size, source_offset, &adapter,
                                hash_data(adapter.read(source_offset), unit_size) });
                    }
                }
            }
            adapter.check_identity();
        }
        std::sort(replacements.begin(), replacements.end(), [](const replacement & a, const replacement & b) {
            return a.base_offset < b.base_offset;
        });
        for (size_t i = 1; i < replacements.size(); ++i) {
            require(replacements[i].base_offset >= replacements[i-1].base_offset + replacements[i-1].size, "overlapping adapters (including aliases)");
        }
        base.check_identity();
    }

    void check_identity() {
        base.check_identity();
        for (auto & adapter : adapters) {
            adapter->check_identity();
        }
    }
};

void set_hashes(gguf_context * meta, const std::string & key, const std::vector<std::string> & hashes) {
    std::vector<const char *> values;
    for (const auto & hash : hashes) {
        values.push_back(hash.c_str());
    }
    gguf_set_arr_str(meta, key.c_str(), values.data(), values.size());
}

} // namespace

llama_qlion_snapshot llama_qlion_capture(const std::string & path, const std::vector<ggml_tensor *> & tensors) {
    file base(path);
    llama_qlion_snapshot result;
    result.path = path;
    result.file_hash = base.hash();
    for (const auto * tensor : tensors) {
        if (std::find(result.names.begin(), result.names.end(), tensor->name) != result.names.end()) {
            continue;
        }
        const int64_t id = gguf_find_tensor(base.meta.get(), tensor->name);
        require(id >= 0 && supported(tensor->type) && tensor->type == base.tensor(id)->type &&
                ggml_are_same_shape(tensor, base.tensor(id)), "invalid training target");
        const auto hash = hash_data(base.read(base.offset(id)), ggml_nbytes(tensor));
        require(hash_data(tensor_reader(tensor), ggml_nbytes(tensor)) == hash, "loaded tensor differs from base before training");
        result.names.emplace_back(tensor->name);
        result.hashes.push_back(hash);
    }
    base.check_identity();
    return result;
}

void llama_qlion_save(const llama_qlion_snapshot & snapshot, const std::vector<ggml_tensor *> & tensors,
        const std::string & output, const std::string & units) {
    require(snapshot.names.size() == snapshot.hashes.size(), "invalid training snapshot");
    require(units == "full" || units == "rows" || units == "experts", "invalid save units");
    distinct_output(output, snapshot.path);
    file base(snapshot.path);
    require(base.hash() == snapshot.file_hash, "base changed since training began");
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.type", "adapter");
    gguf_set_val_str(meta.get(), "adapter.type", "qlion");
    gguf_set_val_str(meta.get(), "general.architecture", get_string(base.meta.get(), "general.architecture").c_str());
    gguf_set_val_u32(meta.get(), "qlion.adapter.version", units == "full" ? 1 : 2);
    gguf_set_val_str(meta.get(), "qlion.adapter.layout", units == "full" ? "full" : "units");
    gguf_set_val_str(meta.get(), "qlion.adapter.base_sha256", snapshot.file_hash.c_str());
    gguf_set_val_str(meta.get(), "qlion.adapter.generator", "llama-finetune-qlion");
    set_hashes(meta.get(), "qlion.adapter.targets", snapshot.names);
    if (gguf_find_tensor(base.meta.get(), "token_embd.weight") >= 0 && gguf_find_tensor(base.meta.get(), "output.weight") < 0) {
        set_hashes(meta.get(), "qlion.adapter.tied_names", { "token_embd.weight", "output.weight" });
    }
    std::vector<std::string> hashes_base, hashes_payload;
    struct saved_tensor {
        const ggml_tensor * tensor;
        std::vector<size_t> offsets;
        size_t unit_size;
    };
    std::vector<saved_tensor> saved;
    uint64_t payload_bytes = 0;
    for (size_t target = 0; target < snapshot.names.size(); ++target) {
        const ggml_tensor * tensor = nullptr;
        for (const auto * candidate : tensors) {
            if (snapshot.names[target] == candidate->name) {
                if (tensor) {
                    require(hash_data(tensor_reader(candidate), ggml_nbytes(candidate)) == hash_data(tensor_reader(tensor), ggml_nbytes(tensor)), "tied copies diverged");
                }
                tensor = candidate;
            }
        }
        require(tensor != nullptr, "training target disappeared");
        const int64_t base_id = gguf_find_tensor(base.meta.get(), tensor->name);
        const size_t bytes = ggml_nbytes(tensor);
        require(hash_data(base.read(base.offset(base_id)), bytes) == snapshot.hashes[target], "base tensor changed");
        if (equal_data(base.read(base.offset(base_id)), tensor_reader(tensor), bytes)) {
            continue;
        }
        // Expert units use rows for tensors without an expert axis.
        const bool experts = units == "experts" && tensor->ne[2] > 1 && tensor->ne[3] == 1;
        const size_t unit_size = units == "full" ? bytes : ggml_row_size(tensor->type, tensor->ne[0]) * (experts ? tensor->ne[1] : 1);
        std::vector<uint64_t> indices;
        std::vector<std::string> hashes_units;
        saved_tensor entry { tensor, {}, unit_size };
        for (size_t offset = 0; offset < bytes; offset += unit_size) {
            if (!equal_data(base.read(base.offset(base_id) + offset), tensor_reader(tensor, offset), unit_size)) {
                indices.push_back(offset / unit_size);
                hashes_units.push_back(hash_data(base.read(base.offset(base_id) + offset), unit_size));
                entry.offsets.push_back(offset);
            }
        }
        ggml_tensor payload = *tensor;
        if (units != "full") {
            const auto prefix = "qlion.adapter.tensor." + std::to_string(saved.size()) + ".";
            uint64_t shape[GGML_MAX_DIMS];
            for (int d = 0; d < GGML_MAX_DIMS; ++d) shape[d] = tensor->ne[d];
            gguf_set_arr_data(meta.get(), (prefix + "shape").c_str(), GGUF_TYPE_UINT64, shape, GGML_MAX_DIMS);
            gguf_set_arr_data(meta.get(), (prefix + (experts ? "experts" : "rows")).c_str(), GGUF_TYPE_UINT64, indices.data(), indices.size());
            set_hashes(meta.get(), prefix + "base_unit_sha256", hashes_units);
            payload.ne[1] = experts ? tensor->ne[1] : indices.size();
            payload.ne[2] = experts ? indices.size() : 1;
            payload.ne[3] = 1;
            payload.nb[0] = ggml_type_size(payload.type);
            payload.nb[1] = ggml_row_size(payload.type, payload.ne[0]);
            for (int d = 2; d < GGML_MAX_DIMS; ++d) payload.nb[d] = payload.nb[d-1] * payload.ne[d-1];
        }
        const reader read_payload = [entry](size_t offset, void * data, size_t n) {
            size_t copied = 0;
            while (copied < n) {
                const size_t pos = offset + copied;
                const size_t part = std::min(n - copied, entry.unit_size - pos % entry.unit_size);
                ggml_backend_tensor_get(entry.tensor, static_cast<char *>(data) + copied,
                        entry.offsets[pos / entry.unit_size] + pos % entry.unit_size, part);
                copied += part;
            }
        };
        hashes_payload.push_back(hash_data(read_payload, unit_size * indices.size()));
        hashes_base.push_back(snapshot.hashes[target]);
        gguf_add_tensor(meta.get(), &payload);
        payload_bytes += unit_size * indices.size();
        saved.push_back(std::move(entry));
    }
    set_hashes(meta.get(), "qlion.adapter.tensor_base_sha256", hashes_base);
    set_hashes(meta.get(), "qlion.adapter.tensor_payload_sha256", hashes_payload);
    LLAMA_LOG_INFO("QLion adapter: tensors=%zu payload_bytes=%llu layout=%s\n", saved.size(), (unsigned long long) payload_bytes, units.c_str());
    output_file out(output);
    std::vector<char> header(gguf_get_meta_size(meta.get()));
    gguf_get_meta_data(meta.get(), header.data());
    out.stream.write(header.data(), header.size());
    for (size_t i = 0; i < saved.size(); ++i) {
        const auto & entry = saved[i];
        sha256_t hash;
        sha256_init(&hash);
        std::vector<unsigned char> buffer(std::min(chunk_size, entry.unit_size));
        for (auto start : entry.offsets) {
            for (size_t offset = 0; offset < entry.unit_size;) {
                const size_t n = std::min(buffer.size(), entry.unit_size - offset);
                ggml_backend_tensor_get(entry.tensor, buffer.data(), start + offset, n);
                sha256_update(&hash, buffer.data(), n);
                out.stream.write(reinterpret_cast<const char *>(buffer.data()), n);
                require(bool(out.stream), "payload write failed");
                offset += n;
            }
        }
        require(digest(hash) == hashes_payload[i], "tensor changed during save");
        const size_t alignment = gguf_get_alignment(meta.get());
        const size_t bytes = entry.unit_size * entry.offsets.size();
        const size_t padding = (alignment - bytes % alignment) % alignment;
        for (size_t j = 0; j < padding; ++j) out.stream.put(0);
        require(bool(out.stream), "padding write failed");
    }
    base.check_identity();
    out.commit();
    LLAMA_LOG_INFO("QLion adapter: saved %s (%llu bytes)\n", output.c_str(), (unsigned long long) std::filesystem::file_size(output));
}

void llama_qlion_merge(const std::string & base, const std::vector<std::string> & adapters,
        const std::string & output, bool allow_compatible) {
    distinct_output(output, base);
    for (const auto & path : adapters) distinct_output(output, path);
    validated inputs(base, adapters, allow_compatible);
    output_file out(output);
    require(hash_data(inputs.base.read(), inputs.base.size, &out.stream) == inputs.base_hash, "base changed during copy");
    for (const auto & replacement : inputs.replacements) {
        out.stream.seekp(replacement.base_offset);
        require(hash_data(replacement.source->read(replacement.source_offset), replacement.size, &out.stream) == replacement.hash, "payload changed during copy");
    }
    inputs.check_identity();
    out.commit();
}

void llama_qlion_apply(const std::string & base, const std::vector<std::string> & adapters,
        const std::vector<ggml_tensor *> & tensors, bool allow_compatible) {
    validated inputs(base, adapters, allow_compatible);
    // Verify every destination before changing any model buffer.
    for (const auto & replacement : inputs.replacements) {
        bool found = false;
        for (auto * tensor : tensors) {
            if (replacement.name == tensor->name) {
                const int64_t id = gguf_find_tensor(inputs.base.meta.get(), tensor->name);
                require(tensor->buffer && tensor->type == inputs.base.tensor(id)->type &&
                        ggml_are_same_shape(tensor, inputs.base.tensor(id)), "invalid runtime destination");
                require(hash_data(tensor_reader(tensor, replacement.tensor_offset), replacement.size) ==
                        hash_data(inputs.base.read(replacement.base_offset), replacement.size), "runtime base bytes differ");
                found = true;
            }
        }
        require(found, "runtime tensor missing: " + replacement.name);
    }
    std::vector<unsigned char> buffer(chunk_size);
    for (const auto & replacement : inputs.replacements) {
        sha256_t hash;
        sha256_init(&hash);
        const auto read = replacement.source->read(replacement.source_offset);
        for (size_t offset = 0; offset < replacement.size;) {
            const size_t n = std::min(buffer.size(), replacement.size - offset);
            read(offset, buffer.data(), n);
            sha256_update(&hash, buffer.data(), n);
            for (auto * tensor : tensors) {
                if (replacement.name == tensor->name) {
                    ggml_backend_tensor_set(tensor, buffer.data(), replacement.tensor_offset + offset, n);
                }
            }
            offset += n;
        }
        require(digest(hash) == replacement.hash, "payload changed during runtime load");
    }
    inputs.check_identity();
}

struct llama_qlion_overlay::impl {
    validated inputs;
    impl(const std::string & base, const std::vector<std::string> & paths, bool compatible) : inputs(base, paths, compatible) {}
};

llama_qlion_overlay::llama_qlion_overlay(const std::string & base, const std::vector<std::string> & paths, bool compatible)
    : pimpl(new impl(base, paths, compatible)) {}
llama_qlion_overlay::~llama_qlion_overlay() = default;

bool llama_qlion_overlay::has(const char * name) const {
    for (const auto & r : pimpl->inputs.replacements) if (r.name == name) return true;
    return false;
}

void llama_qlion_overlay::patch_mapping(void * data, size_t size) const {
    std::vector<unsigned char> buffer(chunk_size);
    for (const auto & r : pimpl->inputs.replacements) {
        require(r.base_offset <= size && r.size <= size - r.base_offset, "mapped replacement out of bounds");
        const auto read = r.source->read(r.source_offset);
        sha256_t hash;
        sha256_init(&hash);
        for (size_t offset = 0; offset < r.size;) {
            const size_t n = std::min(chunk_size, r.size - offset);
            read(offset, buffer.data(), n);
            sha256_update(&hash, buffer.data(), n);
            memcpy(static_cast<unsigned char *>(data) + r.base_offset + offset, buffer.data(), n);
            offset += n;
        }
        require(digest(hash) == r.hash, "payload changed during mapped load");
    }
    pimpl->inputs.check_identity();
}

void llama_qlion_overlay::read_tensor(const char * name, void * data, size_t size) const {
    auto & base = pimpl->inputs.base;
    const int64_t id = gguf_find_tensor(base.meta.get(), name);
    require(id >= 0 && ggml_nbytes(base.tensor(id)) == size, "invalid overlay destination");
    base.read(base.offset(id))(0, data, size);
    for (const auto & r : pimpl->inputs.replacements) {
        if (r.name != name) continue;
        require(r.tensor_offset <= size && r.size <= size - r.tensor_offset, "replacement out of bounds");
        auto * target = static_cast<unsigned char *>(data) + r.tensor_offset;
        const auto read = r.source->read(r.source_offset);
        for (size_t offset = 0; offset < r.size;) {
            const size_t n = std::min(chunk_size, r.size - offset);
            read(offset, target + offset, n);
            offset += n;
        }
        require(hash_data([target](size_t offset, void * out, size_t n) { memcpy(out, target + offset, n); }, r.size) == r.hash,
            "payload changed during tensor assembly");
    }
    pimpl->inputs.check_identity();
}
