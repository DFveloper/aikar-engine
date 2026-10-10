#include "candidate-store.h"
#include "gguf.h"
#include "hash/hash.h"
extern "C" {
#include "hash/sha256/sha256.h"
}
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace llama_opt {
using json = nlohmann::ordered_json;

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static std::vector<uint8_t> read_range(const std::string & file, uint64_t offset, uint64_t count) {
    std::ifstream stream(file, std::ios::binary);
    require(bool(stream), "candidate file unavailable");
    require(offset <= std::filesystem::file_size(file) && count <= std::filesystem::file_size(file) - offset,
            "candidate file range outside payload");
    stream.seekg(offset);
    std::vector<uint8_t> bytes(count);
    stream.read(reinterpret_cast<char *>(bytes.data()), count);
    require(bool(stream), "short candidate read");
    return bytes;
}

std::string file_sha256(const std::string & file) {
    std::ifstream stream(file, std::ios::binary);
    require(bool(stream), "source identity read failed");
    sha256_t state;
    sha256_init(&state);
    std::vector<uint8_t> bytes(1024 * 1024);
    while (stream) {
        stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
        sha256_update(&state, bytes.data(), stream.gcount());
    }
    require(stream.eof(), "source identity read failed");
    uint8_t digest[SHA256_DIGEST_SIZE];
    sha256_final(&state, digest);
    const char * hex = "0123456789abcdef";
    std::string result;
    for (uint8_t byte : digest) {
        result += hex[byte >> 4];
        result += hex[byte & 15];
    }
    return result;
}

static uint64_t aligned(uint64_t size, uint64_t alignment) {
    require(alignment > 0 && size <= UINT64_MAX - alignment + 1, "candidate byte size overflow");
    return ((size + alignment - 1) / alignment) * alignment;
}

static ggml_type parse_type(const std::string & name) {
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        if (ggml_type_name(ggml_type(type)) && name == ggml_type_name(ggml_type(type)) && ggml_blck_size(ggml_type(type)) > 0) {
            return ggml_type(type);
        }
    }
    throw std::runtime_error("unknown candidate codec: " + name);
}

static void decode(ggml_type type, const std::vector<uint8_t> & bytes, std::vector<float> & weights) {
    if (type == GGML_TYPE_F32) {
        require(bytes.size() == weights.size() * sizeof(float), "F32 candidate byte mismatch");
        std::memcpy(weights.data(), bytes.data(), bytes.size());
    } else {
        require(ggml_get_type_traits(type)->to_float != nullptr, "source tensor cannot be decoded");
        ggml_get_type_traits(type)->to_float(bytes.data(), weights.data(), weights.size());
    }
}

static void save(const std::string & path, const json & value) {
    require(!std::filesystem::exists(path), "refusing to overwrite candidate manifest/report");
    std::ofstream file(path);
    file << value.dump(2) << '\n';
    require(bool(file), "candidate manifest write failed");
}

candidate_store::candidate_store(const std::string & source, const std::string & manifest) {
    std::ifstream file(manifest);
    json index;
    file >> index;
    require(index.at("schema") == 1, "unsupported candidate manifest schema");
    auto * metadata = gguf_init_from_file(source.c_str(), { true, nullptr });
    require(metadata != nullptr, "source GGUF unavailable");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> owner(metadata, gguf_free);
    source_model = source;
    source_identity = file_sha256(source);
    const bool matches = source_identity == index.at("source_identity").get<std::string>();
    require(matches, "candidate source identity mismatch");
    fixed = index.at("fixed_gguf_bytes");
    alignment = index.at("alignment");
    require(alignment == gguf_get_alignment(metadata), "candidate alignment mismatch");
    uint64_t calculated_fixed = gguf_get_meta_size(metadata);
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
        calculated_fixed += aligned(gguf_get_tensor_size(metadata, i), alignment);
    }
    std::set<std::string> names;
    for (const auto & entry : index.at("groups")) {
        candidate_group_info group;
        group.tensor = entry.at("tensor");
        group.operation = entry.at("operation");
        group.shape = entry.at("shape").get<std::vector<int64_t>>();
        require(names.insert(group.tensor).second && group.shape.size() == 2 && group.shape[0] > 0 && group.shape[1] > 0,
                "invalid or duplicate candidate group");
        require(group.operation == "dense_matmul", "routed candidate operation is not implemented");
        const auto source_id = gguf_find_tensor(metadata, group.tensor.c_str());
        require(source_id >= 0, "candidate tensor missing in source model");
        const auto * source_shape = gguf_get_tensor_ne(metadata, source_id);
        require(group.shape[0] == source_shape[0] && group.shape[1] == source_shape[1] && source_shape[2] == 1 && source_shape[3] == 1,
                "candidate dimensions differ from source model");
        calculated_fixed -= aligned(gguf_get_tensor_size(metadata, source_id), alignment);
        std::set<std::string> choices;
        for (const auto & option : entry.at("choices")) {
            packed_choice choice;
            choice.type = parse_type(option.at("type"));
            choice.producer = option.at("producer");
            choice.file = option.at("file");
            choice.sha256 = option.at("sha256");
            choice.bytes = option.at("bytes");
            choice.aligned_bytes = option.at("aligned_bytes");
            require(choices.insert(choice.producer + ":" + ggml_type_name(choice.type)).second,
                    "duplicate candidate producer/codec");
            require(group.shape[0] % ggml_blck_size(choice.type) == 0 &&
                    choice.bytes == ggml_row_size(choice.type, group.shape[0]) * group.shape[1] &&
                    choice.aligned_bytes == aligned(choice.bytes, alignment), "candidate shape or byte cost mismatch");
            group.choices.push_back(std::move(choice));
        }
        require(!group.choices.empty(), "empty candidate options");
        groups.push_back(std::move(group));
    }
    require(!groups.empty(), "empty candidate store");
    require(calculated_fixed == fixed, "candidate fixed GGUF cost mismatch");
    for (size_t i = 0; i < groups.size(); ++i) {
        for (size_t k = 0; k < groups[i].choices.size(); ++k) {
            read_packed(i, k);
        }
    }
}

std::vector<uint8_t> candidate_store::read_packed(size_t group, size_t option) const {
    const auto & choice = groups.at(group).choices.at(option);
    require(std::filesystem::file_size(choice.file) == choice.bytes, "candidate payload size mismatch");
    auto bytes = read_range(choice.file, 0, choice.bytes);
    require(hash_sha256_hex(bytes.data(), bytes.size()) == choice.sha256, "candidate payload hash mismatch");
    require(ggml_validate_row_data(choice.type, bytes.data(), bytes.size()), "invalid packed candidate");
    return bytes;
}

std::vector<float> candidate_store::decode_tile(size_t group, size_t option, int64_t first, int64_t count) const {
    const auto & info = groups.at(group);
    const auto & choice = info.choices.at(option);
    require(first >= 0 && count > 0 && first <= info.shape[1] && count <= info.shape[1] - first,
            "candidate tile outside tensor");
    const auto row_bytes = ggml_row_size(choice.type, info.shape[0]);
    const auto bytes = read_range(choice.file, first * row_bytes, count * row_bytes);
    std::vector<float> decoded(count * info.shape[0]);
    decode(choice.type, bytes, decoded);
    return decoded;
}

std::vector<std::vector<uint64_t>> candidate_store::costs() const {
    std::vector<std::vector<uint64_t>> result;
    for (const auto & group : groups) {
        std::vector<uint64_t> row;
        for (const auto & choice : group.choices) {
            row.push_back(choice.aligned_bytes);
        }
        result.push_back(std::move(row));
    }
    return result;
}

void create_candidate_store(const std::string & source, const std::string & config,
                            const std::string & directory, const std::string & report) {
    require(!std::filesystem::exists(directory) && !std::filesystem::exists(report), "candidate output already exists");
    std::ifstream config_file(config);
    json configuration;
    config_file >> configuration;
    auto * metadata = gguf_init_from_file(source.c_str(), { true, nullptr });
    require(metadata != nullptr, "source GGUF unavailable");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> owner(metadata, gguf_free);
    const uint64_t alignment = gguf_get_alignment(metadata);
    uint64_t fixed = gguf_get_meta_size(metadata);
    for (int64_t i = 0; i < gguf_get_n_tensors(metadata); ++i) {
        fixed += aligned(gguf_get_tensor_size(metadata, i), alignment);
    }
    json index = { { "schema", 1 }, { "source_model", source }, { "source_identity", file_sha256(source) },
                   { "alignment", alignment }, { "groups", json::array() } };
    std::set<std::string> names;
    std::filesystem::create_directories(directory);
    for (const auto & entry : configuration.at("tensors")) {
        const std::string name = entry.at("tensor");
        const int64_t id = gguf_find_tensor(metadata, name.c_str());
        require(id >= 0 && names.insert(name).second, "candidate tensor missing or duplicate");
        const auto * shape = gguf_get_tensor_ne(metadata, id);
        require(shape[0] > 0 && shape[1] > 0 && shape[2] == 1 && shape[3] == 1, "candidate store requires dense 2D tensors");
        const auto type = gguf_get_tensor_type(metadata, id);
        const auto raw = read_range(source, gguf_get_data_offset(metadata) + gguf_get_tensor_offset(metadata, id),
                                    gguf_get_tensor_size(metadata, id));
        std::vector<float> weights(shape[0] * shape[1]);
        decode(type, raw, weights);
        fixed -= aligned(raw.size(), alignment);
        json group = { { "tensor", name }, { "shape", { shape[0], shape[1] } },
                       { "operation", "dense_matmul" }, { "choices", json::array() } };
        json options = entry.value("candidates", json::array({ { { "type", "q2_0" }, { "producer", "ptq" } },
                                                               { { "type", "q4_0" }, { "producer", "ptq" } },
                                                               { { "type", "q8_0" }, { "producer", "ptq" } } }));
        std::set<std::string> keys;
        for (const auto & option : options) {
            const std::string producer = option.at("producer");
            const std::string codec = option.at("type");
            const auto candidate_type = parse_type(codec);
            require(keys.insert(producer + ":" + codec).second, "duplicate candidate producer/codec");
            require(shape[0] % ggml_blck_size(candidate_type) == 0, "candidate block alignment mismatch");
            std::vector<uint8_t> bytes(ggml_row_size(candidate_type, shape[0]) * shape[1]);
            if (option.contains("packed_file")) {
                const std::string path = option.at("packed_file");
                require(std::filesystem::file_size(path) == bytes.size(), "import packed candidate size mismatch");
                bytes = read_range(path, 0, bytes.size());
                require(hash_sha256_hex(bytes.data(), bytes.size()) == option.at("sha256").get<std::string>(),
                        "import packed candidate hash mismatch");
            } else if (option.contains("model")) {
                const std::string path = option.at("model");
                auto * other = gguf_init_from_file(path.c_str(), { true, nullptr });
                require(other != nullptr, "import candidate GGUF unavailable");
                std::unique_ptr<gguf_context, decltype(&gguf_free)> other_owner(other, gguf_free);
                const auto other_id = gguf_find_tensor(other, name.c_str());
                require(other_id >= 0 && gguf_get_tensor_type(other, other_id) == candidate_type &&
                        gguf_get_tensor_size(other, other_id) == bytes.size(), "import candidate type/size mismatch");
                const auto * other_shape = gguf_get_tensor_ne(other, other_id);
                require(std::equal(shape, shape + 4, other_shape), "import candidate shape mismatch");
                bytes = read_range(path, gguf_get_data_offset(other) + gguf_get_tensor_offset(other, other_id), bytes.size());
            } else if (option.contains("imatrix_file")) {
                require(producer == "imatrix-ptq" && candidate_type == GGML_TYPE_Q4_0,
                        "weighted candidate generation currently supports Q4_0 only");
                const std::string path = option.at("imatrix_file");
                require(std::filesystem::file_size(path) == size_t(shape[0]) * sizeof(float), "IMatrix column count mismatch");
                const auto raw_importance = read_range(path, 0, std::filesystem::file_size(path));
                require(hash_sha256_hex(raw_importance.data(), raw_importance.size()) == option.at("imatrix_sha256").get<std::string>(),
                        "IMatrix hash mismatch");
                std::vector<float> importance(shape[0]);
                std::memcpy(importance.data(), raw_importance.data(), raw_importance.size());
                require(std::all_of(importance.begin(), importance.end(), [](float x) { return std::isfinite(x) && x >= 0; }) &&
                        std::any_of(importance.begin(), importance.end(), [](float x) { return x > 0; }), "invalid IMatrix weights");
                require(ggml_quantize_chunk(candidate_type, weights.data(), bytes.data(), 0, shape[1], shape[0], importance.data()) == bytes.size(),
                        "weighted quantizer size mismatch");
            } else {
                require(producer == "ptq", "optimized candidates require exact packed import");
                require(ggml_get_type_traits(candidate_type)->from_float_ref != nullptr, "candidate codec cannot be encoded");
                ggml_get_type_traits(candidate_type)->from_float_ref(weights.data(), bytes.data(), weights.size());
            }
            require(ggml_validate_row_data(candidate_type, bytes.data(), bytes.size()), "invalid candidate quantization");
            const auto payload = std::filesystem::absolute(std::filesystem::path(directory) /
                (std::to_string(index["groups"].size()) + "." + std::to_string(group["choices"].size()) + ".bin"));
            std::ofstream file(payload, std::ios::binary);
            file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
            require(bool(file), "candidate payload write failed");
            group["choices"].push_back({ { "type", codec }, { "producer", producer }, { "file", payload.string() },
                { "sha256", hash_sha256_hex(bytes.data(), bytes.size()) }, { "bytes", bytes.size() },
                { "aligned_bytes", aligned(bytes.size(), alignment) } });
        }
        index["groups"].push_back(std::move(group));
    }
    require(!index["groups"].empty(), "empty candidate configuration");
    index["fixed_gguf_bytes"] = fixed;
    save((std::filesystem::path(directory) / "manifest.json").string(), index);
    save(report, index);
}
} // namespace llama_opt
