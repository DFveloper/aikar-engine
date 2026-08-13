#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "llama.h"
#include "llama-ext.h"
#include "chat.h"
#include "jsonl.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct merge_params {
    std::string base;
    std::vector<std::string> models;
    std::string output;
    std::string method = "ties";
    float density = 0.5f;
    int n_threads = std::max(1u, std::thread::hardware_concurrency());
    size_t memory_budget = 2ull * 1024 * 1024 * 1024;
    std::string calibration;
    std::string target_type;
    int population = 8;
    int generations = 10;
    int elite_count = 2;
    int gpu_layers = -1;
    std::string device;
    std::vector<std::string> devices;
    bool merge_gpu = false;
    int context_size = 512;
    float mutation = 0.10f;
    uint32_t seed = 0;
    bool seed_set = false;
    std::string temp_dir;
    bool ignore_chat_template = false;
    std::string evo_mode = "low-mem";
};

static void usage(const char * executable) {
    printf("usage: %s --base BASE.gguf --model MODEL.gguf [--model MODEL.gguf ...] --output OUT.gguf [options]\n", executable);
    printf("       %s --config merge.ini\n\n", executable);
    printf("options:\n");
    printf("  --method {ties,evo}           merge method (default: ties)\n");
    printf("  --density N                   TIES task-vector density in (0, 1] (default: 0.5)\n");
    printf("  -m, --model MODEL.gguf        input model; may be specified multiple times\n");
    printf("  -o, --output OUT.gguf         output GGUF\n");
    printf("  -t, --threads N               merge worker count (default: CPU core count)\n");
    printf("  --memory-budget SIZE          worker memory budget, e.g. 8G or 16384M (default: 2G)\n");
    printf("  --calibration FILE            Evo calibration .txt or .jsonl\n");
    printf("  --target-type TYPE            Evo target: q4_0, q3_k, q4_k, or mxfp4\n");
    printf("  --population N                Evo population size (default: 8)\n");
    printf("  --generations N               Evo generation count (default: 10)\n");
    printf("  --elite-count N               Evo elites retained per generation (default: 2)\n");
    printf("  --sigma0 N                    Evo CMA-ES initial sigma (default: 0.10)\n");
    printf("  --seed N                      Evo random seed (default: random)\n");
    printf("  --gpu-layers N                layers offloaded for fitness; -1 means all (default: -1)\n");
    printf("  --device NAME                 fitness backend device, e.g. CUDA0 or Vulkan0\n");
    printf("  --devices LIST                evaluate candidates concurrently, one per comma-separated device\n");
    printf("  --merge-gpu                  run Evo weighted merge math on the selected GPU\n");
    printf("  --ctx-size N                  fitness context size (default: 512)\n");
    printf("  --temp-dir DIR                local directory for Evo candidate files (default: system temp)\n");
    printf("  --ignore-chat-template        allow input models with different tokenizer.chat_template metadata\n");
    printf("  --evo-mode MODE               candidate flow: low-mem or ram (default: low-mem)\n");
    printf("  --config FILE                 INI file: base=, models=comma,separated, output=, method=, density=, threads=, memory_budget=\n");
}

static size_t parse_memory_size(const std::string & value) {
    if (value.empty()) {
        throw std::runtime_error("empty memory budget");
    }
    size_t suffix_at = value.find_first_not_of("0123456789");
    const std::string number = value.substr(0, suffix_at);
    std::string suffix = suffix_at == std::string::npos ? "" : value.substr(suffix_at);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), ::toupper);
    size_t multiplier = 1;
    if (suffix == "K" || suffix == "KB") {
        multiplier = 1024;
    } else if (suffix == "M" || suffix == "MB") {
        multiplier = 1024 * 1024;
    } else if (suffix == "G" || suffix == "GB") {
        multiplier = 1024 * 1024 * 1024;
    } else if (!suffix.empty() && suffix != "B") {
        throw std::runtime_error("invalid memory budget suffix '" + suffix + "'");
    }
    const size_t amount = std::stoull(number);
    if (amount == 0 || amount > SIZE_MAX / multiplier) {
        throw std::runtime_error("invalid memory budget '" + value + "'");
    }
    return amount * multiplier;
}

static bool parse_bool(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "true" || value == "1" || value == "yes" || value == "on") return true;
    if (value == "false" || value == "0" || value == "no" || value == "off") return false;
    throw std::runtime_error("invalid boolean value '" + value + "'");
}

static std::string trim(const std::string & value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static std::vector<std::string> split(const std::string & value, char separator) {
    std::vector<std::string> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(separator, begin);
        const std::string item = trim(value.substr(begin, end - begin));
        if (!item.empty()) {
            result.push_back(item);
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return result;
}

static void load_config(const std::string & path, merge_params & params) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config " + path);
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') {
            continue;
        }
        const size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("invalid config line " + std::to_string(line_number));
        }
        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (key == "base") {
            params.base = value;
        } else if (key == "models") {
            params.models = split(value, ',');
        } else if (key == "output") {
            params.output = value;
        } else if (key == "method") {
            params.method = value;
        } else if (key == "density") {
            params.density = std::stof(value);
        } else if (key == "threads") {
            params.n_threads = std::stoi(value);
        } else if (key == "memory_budget") {
            params.memory_budget = parse_memory_size(value);
        } else if (key == "calibration") {
            params.calibration = value;
        } else if (key == "target_type") {
            params.target_type = value;
        } else if (key == "population") {
            params.population = std::stoi(value);
        } else if (key == "generations") {
            params.generations = std::stoi(value);
        } else if (key == "elite_count") {
            params.elite_count = std::stoi(value);
        } else if (key == "mutation") {
            params.mutation = std::stof(value);
        } else if (key == "sigma0") {
            params.mutation = std::stof(value);
        } else if (key == "seed") {
            params.seed = std::stoul(value);
            params.seed_set = true;
        } else if (key == "gpu_layers") {
            params.gpu_layers = std::stoi(value);
        } else if (key == "device") {
            params.device = value;
        } else if (key == "devices") {
            params.devices = split(value, ',');
        } else if (key == "merge_gpu") {
            params.merge_gpu = parse_bool(value);
        } else if (key == "ctx_size") {
            params.context_size = std::stoi(value);
        } else if (key == "temp_dir") {
            params.temp_dir = value;
        } else if (key == "ignore_chat_template") {
            params.ignore_chat_template = parse_bool(value);
        } else if (key == "evo_mode") {
            params.evo_mode = value;
        } else {
            throw std::runtime_error("unknown config key '" + key + "'");
        }
    }
}

static merge_params parse_args(int argc, char ** argv) {
    merge_params params;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            exit(0);
        }
        if (arg == "--config" && i + 1 < argc) {
            load_config(argv[++i], params);
        } else if (arg == "--base" && i + 1 < argc) {
            params.base = argv[++i];
        } else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            params.models.push_back(argv[++i]);
        } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
            params.output = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            params.method = argv[++i];
        } else if (arg == "--density" && i + 1 < argc) {
            params.density = std::stof(argv[++i]);
        } else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--memory-budget" && i + 1 < argc) {
            params.memory_budget = parse_memory_size(argv[++i]);
        } else if (arg == "--calibration" && i + 1 < argc) {
            params.calibration = argv[++i];
        } else if (arg == "--target-type" && i + 1 < argc) {
            params.target_type = argv[++i];
        } else if (arg == "--population" && i + 1 < argc) {
            params.population = std::stoi(argv[++i]);
        } else if (arg == "--generations" && i + 1 < argc) {
            params.generations = std::stoi(argv[++i]);
        } else if (arg == "--elite-count" && i + 1 < argc) {
            params.elite_count = std::stoi(argv[++i]);
        } else if (arg == "--mutation" && i + 1 < argc) {
            params.mutation = std::stof(argv[++i]);
        } else if (arg == "--sigma0" && i + 1 < argc) {
            params.mutation = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            params.seed = std::stoul(argv[++i]);
            params.seed_set = true;
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            params.gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            params.device = argv[++i];
        } else if (arg == "--devices" && i + 1 < argc) {
            params.devices = split(argv[++i], ',');
        } else if (arg == "--merge-gpu") {
            params.merge_gpu = true;
        } else if (arg == "--ctx-size" && i + 1 < argc) {
            params.context_size = std::stoi(argv[++i]);
        } else if (arg == "--temp-dir" && i + 1 < argc) {
            params.temp_dir = argv[++i];
        } else if (arg == "--ignore-chat-template") {
            params.ignore_chat_template = true;
        } else if (arg == "--evo-mode" && i + 1 < argc) {
            params.evo_mode = argv[++i];
        } else {
            throw std::runtime_error("unknown or incomplete option '" + arg + "'");
        }
    }
    if (params.base.empty() || params.models.empty() || params.output.empty()) {
        throw std::runtime_error("--base, at least one --model, and --output are required");
    }
    if (params.method != "ties" && params.method != "evo") {
        throw std::runtime_error("--method must be ties or evo");
    }
    if (params.density <= 0.0f || params.density > 1.0f) {
        throw std::runtime_error("--density must be in (0, 1]");
    }
    if (params.n_threads < 1) {
        throw std::runtime_error("--threads must be positive");
    }
    if (params.method == "evo") {
        if (params.calibration.empty() || params.target_type.empty()) {
            throw std::runtime_error("evo requires --calibration and --target-type");
        }
        if (params.population < 2 || params.generations < 1 || params.elite_count < 1 ||
                params.elite_count >= params.population || params.mutation <= 0.0f || params.context_size < 2) {
            throw std::runtime_error("invalid Evo population, generation, elite, sigma, or context setting");
        }
        if (params.evo_mode != "low-mem" && params.evo_mode != "ram") {
            throw std::runtime_error("--evo-mode must be low-mem or ram");
        }
        if (params.evo_mode == "ram" && params.merge_gpu) {
            throw std::runtime_error("--evo-mode ram uses CPU merge and cannot be combined with --merge-gpu");
        }
        if (params.evo_mode == "ram" && params.devices.size() > 1) {
            throw std::runtime_error("--evo-mode ram accepts at most one fitness device");
        }
        if (params.evo_mode == "ram" && params.temp_dir.empty()) {
            throw std::runtime_error("--evo-mode ram requires --temp-dir on a RAM-backed filesystem");
        }
    }
    if (params.merge_gpu && params.method != "evo") {
        throw std::runtime_error("--merge-gpu currently requires --method evo");
    }
    if (!params.device.empty() && !params.devices.empty()) {
        throw std::runtime_error("use either --device or --devices, not both");
    }
    return params;
}

static void write_zeros(std::ofstream & output, size_t count) {
    static const char zero = 0;
    for (size_t i = 0; i < count; ++i) {
        output.write(&zero, 1);
    }
}

static std::string get_kv_string(const gguf_context * gguf, const char * key) {
    const int64_t idx = gguf_find_key(gguf, key);
    return idx < 0 ? "" : gguf_get_val_str(gguf, idx);
}

static size_t gguf_scalar_size(gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0;
    }
}

static bool same_kv_value(const gguf_context * a, int64_t ia, const gguf_context * b, int64_t ib) {
    const gguf_type type = gguf_get_kv_type(a, ia);
    if (type != gguf_get_kv_type(b, ib)) {
        return false;
    }
    if (type == GGUF_TYPE_STRING) {
        return std::string(gguf_get_val_str(a, ia)) == gguf_get_val_str(b, ib);
    }
    if (type == GGUF_TYPE_ARRAY) {
        const gguf_type element_type = gguf_get_arr_type(a, ia);
        const size_t count = gguf_get_arr_n(a, ia);
        if (element_type != gguf_get_arr_type(b, ib) || count != gguf_get_arr_n(b, ib)) {
            return false;
        }
        if (element_type == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < count; ++i) {
                if (std::string(gguf_get_arr_str(a, ia, i)) != gguf_get_arr_str(b, ib, i)) {
                    return false;
                }
            }
            return true;
        }
        const size_t element_size = gguf_scalar_size(element_type);
        if (element_size == 0 || count > SIZE_MAX / element_size) {
            throw std::runtime_error("unsupported or oversized GGUF metadata array");
        }
        return memcmp(gguf_get_arr_data(a, ia), gguf_get_arr_data(b, ib), count * element_size) == 0;
    }
    const size_t size = gguf_scalar_size(type);
    if (size == 0) {
        throw std::runtime_error("unsupported GGUF metadata type " + std::string(gguf_type_name(type)));
    }
    return memcmp(gguf_get_val_data(a, ia), gguf_get_val_data(b, ib), size) == 0;
}

static bool is_compatibility_key(
        const std::string & key,
        const std::string & architecture,
        bool ignore_chat_template) {
    if (ignore_chat_template && key == "tokenizer.chat_template") {
        return false;
    }
    return key.rfind("tokenizer.", 0) == 0 || key.rfind(architecture + ".", 0) == 0 || key == "general.architecture";
}

struct gguf_input;
static void validate_metadata_compatibility(
        const gguf_input & base,
        const gguf_input & input,
        bool ignore_chat_template = false);

static uint32_t get_file_type(const gguf_context * gguf) {
    const int64_t idx = gguf_find_key(gguf, "general.file_type");
    if (idx < 0 || gguf_get_kv_type(gguf, idx) != GGUF_TYPE_UINT32) {
        return LLAMA_FTYPE_GUESSED;
    }
    return gguf_get_val_u32(gguf, idx);
}

struct gguf_input {
    struct tensor_ref {
        ggml_tensor * tensor;
    };

    std::string path;
    mutable std::ifstream file;
    mutable std::mutex mutex;
    ggml_context * ctx = nullptr;
    gguf_context * gguf = nullptr;
    std::map<std::string, tensor_ref> tensors;
    uint32_t file_type = LLAMA_FTYPE_GUESSED;

    explicit gguf_input(const std::string & path) : path(path), file(path, std::ios::binary) {
        if (!file) {
            throw std::runtime_error("failed to open input GGUF " + path);
        }
        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };
        gguf = gguf_init_from_file(path.c_str(), params);
        if (!gguf) {
            throw std::runtime_error("failed to parse input GGUF " + path);
        }
        for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            if (!tensors.emplace(tensor->name, tensor_ref { tensor }).second) {
                throw std::runtime_error("duplicate tensor '" + std::string(tensor->name) + "' in " + path);
            }
        }
        file_type = get_file_type(gguf);
    }

    ~gguf_input() {
        gguf_free(gguf);
        ggml_free(ctx);
    }

    void read_tensor(const std::string & name, std::vector<uint8_t> & data) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("missing tensor '" + name + "' in " + path);
        }
        const size_t size = ggml_nbytes(it->second.tensor);
        data.resize(size);
        read_tensor_part(name, 0, data.data(), size);
    }

    void read_tensor_part(const std::string & name, size_t tensor_offset, void * data, size_t size) const {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = tensors.find(name);
        if (it == tensors.end()) {
            throw std::runtime_error("missing tensor '" + name + "' in " + path);
        }
        const size_t tensor_size = ggml_nbytes(it->second.tensor);
        if (tensor_offset > tensor_size || size > tensor_size - tensor_offset) {
            throw std::runtime_error("tensor read is out of bounds for '" + name + "'");
        }
        const int tensor_index = gguf_find_tensor(gguf, name.c_str());
        const size_t offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_index);
        file.clear();
        file.seekg(offset + tensor_offset);
        file.read((char *) data, size);
        if (!file) {
            throw std::runtime_error("failed to read tensor '" + name + "' from " + path);
        }
    }
};

static void validate_metadata_compatibility(
        const gguf_input & base,
        const gguf_input & input,
        bool ignore_chat_template) {
    const std::string architecture = get_kv_string(base.gguf, "general.architecture");
    for (int pass = 0; pass < 2; ++pass) {
        const gguf_context * source = pass == 0 ? base.gguf : input.gguf;
        const gguf_context * other = pass == 0 ? input.gguf : base.gguf;
        for (int64_t i = 0; i < gguf_get_n_kv(source); ++i) {
            const std::string key = gguf_get_key(source, i);
            if (!is_compatibility_key(key, architecture, ignore_chat_template)) {
                continue;
            }
            const int64_t other_index = gguf_find_key(other, key.c_str());
            if (other_index < 0 || !same_kv_value(source, i, other, other_index)) {
                throw std::runtime_error("incompatible model metadata '" + key + "' in " + input.path);
            }
        }
    }
}

static bool same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return true;
}

static ggml_type select_precision_type(const std::vector<std::unique_ptr<gguf_input>> & inputs) {
    bool has_f32 = false;
    bool has_bf16 = false;
    bool all_same = true;
    const uint32_t first = inputs[0]->file_type;
    for (const auto & input : inputs) {
        has_f32 |= input->file_type == LLAMA_FTYPE_ALL_F32;
        has_bf16 |= input->file_type == LLAMA_FTYPE_MOSTLY_BF16;
        all_same &= input->file_type == first;
    }
    if (all_same && first != LLAMA_FTYPE_ALL_F32 && first != LLAMA_FTYPE_MOSTLY_BF16 && first != LLAMA_FTYPE_MOSTLY_F16) {
        return GGML_TYPE_COUNT;
    }
    if (has_f32) {
        return GGML_TYPE_F32;
    }
    if (has_bf16) {
        return GGML_TYPE_BF16;
    }
    return GGML_TYPE_F16;
}

static uint32_t type_to_ftype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return LLAMA_FTYPE_ALL_F32;
        case GGML_TYPE_BF16: return LLAMA_FTYPE_MOSTLY_BF16;
        case GGML_TYPE_F16:  return LLAMA_FTYPE_MOSTLY_F16;
        case GGML_TYPE_Q4_0: return LLAMA_FTYPE_MOSTLY_Q4_0;
        case GGML_TYPE_Q3_K: return LLAMA_FTYPE_MOSTLY_Q3_K_M;
        case GGML_TYPE_Q4_K: return LLAMA_FTYPE_MOSTLY_Q4_K_M;
        case GGML_TYPE_MXFP4:return LLAMA_FTYPE_MOSTLY_MXFP4_MOE;
        default:             return LLAMA_FTYPE_GUESSED;
    }
}

static llama_ftype parse_target_ftype(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if (value == "q4_0")  return LLAMA_FTYPE_MOSTLY_Q4_0;
    if (value == "q3_k")  return LLAMA_FTYPE_MOSTLY_Q3_K_M;
    if (value == "q4_k")  return LLAMA_FTYPE_MOSTLY_Q4_K_M;
    if (value == "mxfp4") return LLAMA_FTYPE_MOSTLY_MXFP4_MOE;
    throw std::runtime_error("Evo target type must be q4_0, q3_k, q4_k, or mxfp4");
}

static int select_worker_count(const merge_params & params, const gguf_input & base, size_t n_inputs) {
    // TIES retains one F32 task vector per input model plus the base, result,
    // and a temporary magnitude vector. Bound concurrent tensors so a large
    // embedding or output tensor cannot multiply that allocation by core count.
    size_t largest_tensor = 0;
    for (const auto & entry : base.tensors) {
        largest_tensor = std::max(largest_tensor, (size_t) ggml_nelements(entry.second.tensor));
    }
    if (largest_tensor == 0) {
        throw std::runtime_error("input model has no tensors");
    }

    const size_t vectors_per_worker = n_inputs + 2;
    const size_t bytes_per_worker = largest_tensor > SIZE_MAX / (vectors_per_worker * sizeof(float))
        ? SIZE_MAX
        : largest_tensor * vectors_per_worker * sizeof(float);
    const size_t memory_workers = bytes_per_worker == 0 ? 1 : std::max<size_t>(1, params.memory_budget / bytes_per_worker);
    return std::min<int>(params.n_threads, std::min<size_t>(base.tensors.size(), memory_workers));
}

static void decode_tensor(const gguf_input & input, const std::string & name, std::vector<uint8_t> & bytes, std::vector<float> & result) {
    const ggml_tensor * tensor = input.tensors.at(name).tensor;
    input.read_tensor(name, bytes);
    const int64_t count = ggml_nelements(tensor);
    result.resize(count);
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(result.data(), bytes.data(), count * sizeof(float));
        return;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error("tensor '" + name + "' has unsupported type " + ggml_type_name(tensor->type));
    }
    traits->to_float(bytes.data(), result.data(), count);
}

static void trim_task_vector(std::vector<float> & values, float density) {
    if (density >= 1.0f || values.empty()) {
        return;
    }
    const size_t keep = std::max<size_t>(1, (size_t) std::ceil(values.size() * density));
    std::vector<float> magnitudes(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        magnitudes[i] = std::fabs(values[i]);
    }
    std::nth_element(magnitudes.begin(), magnitudes.end() - keep, magnitudes.end());
    const float threshold = magnitudes[magnitudes.size() - keep];
    for (float & value : values) {
        if (std::fabs(value) < threshold) {
            value = 0.0f;
        }
    }
}

static void ties_merge(const std::vector<float> & base, std::vector<std::vector<float>> & tasks, float density, std::vector<float> & result) {
    for (std::vector<float> & task : tasks) {
        trim_task_vector(task, density);
    }
    result.resize(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        float sign_sum = 0.0f;
        for (const std::vector<float> & task : tasks) {
            sign_sum += task[i];
        }
        if (sign_sum == 0.0f) {
            result[i] = base[i];
            continue;
        }
        const bool positive = sign_sum > 0.0f;
        float sum = 0.0f;
        int count = 0;
        for (const std::vector<float> & task : tasks) {
            if ((positive && task[i] > 0.0f) || (!positive && task[i] < 0.0f)) {
                sum += task[i];
                ++count;
            }
        }
        result[i] = base[i] + sum / count;
    }
}

static std::vector<uint8_t> encode_tensor(
        ggml_type type,
        const ggml_tensor * shape,
        const std::vector<float> & values,
        common_jsonl_worker_pool * worker_pool = nullptr,
        int n_threads = 1) {
    if (type == GGML_TYPE_F32) {
        const size_t size = values.size() * sizeof(float);
        std::vector<uint8_t> encoded(size);
        memcpy(encoded.data(), values.data(), size);
        return encoded;
    }
    if (ggml_is_quantized(type)) {
        const int64_t n_per_row = shape->ne[0];
        if (n_per_row <= 0 || values.size() % n_per_row != 0 || n_per_row % ggml_blck_size(type) != 0) {
            throw std::runtime_error("tensor '" + std::string(shape->name) + "' is incompatible with " + ggml_type_name(type));
        }
        const int64_t n_rows = values.size() / n_per_row;
        const size_t size = ggml_row_size(type, n_per_row) * n_rows;
        std::vector<uint8_t> quantized(size);
        const int workers = std::max<int>(1, std::min<int64_t>(n_threads, n_rows));
        if (workers == 1 || !worker_pool) {
            ggml_quantize_chunk(type, values.data(), quantized.data(), 0, n_rows, n_per_row, nullptr);
        } else {
            ggml_quantize_init(type);
            worker_pool->parallel_for(workers, [&](size_t worker, size_t) {
                const int64_t first_row = n_rows * worker / workers;
                const int64_t last_row = n_rows * (worker + 1) / workers;
                ggml_quantize_chunk(
                        type, values.data(), quantized.data(), first_row * n_per_row,
                        last_row - first_row, n_per_row, nullptr);
            });
        }
        return quantized;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    if (!traits || !traits->from_float_ref) {
        throw std::runtime_error("cannot encode output type " + std::string(ggml_type_name(type)));
    }
    const size_t size = ggml_row_size(type, shape->ne[0]) * (values.size() / shape->ne[0]);
    std::vector<uint8_t> encoded(size);
    traits->from_float_ref(values.data(), encoded.data(), values.size());
    return encoded;
}

static void write_encoded_tensor(std::ofstream & output, const std::vector<uint8_t> & encoded, size_t alignment) {
    output.write((const char *) encoded.data(), encoded.size());
    write_zeros(output, GGML_PAD(encoded.size(), alignment) - encoded.size());
}

static void write_tensor(
        std::ofstream & output,
        ggml_type type,
        const ggml_tensor * shape,
        const std::vector<float> & values,
        size_t alignment) {
    write_encoded_tensor(output, encode_tensor(type, shape, values), alignment);
}

static bool tensor_can_decode(const ggml_tensor * tensor) {
    if (tensor->type == GGML_TYPE_F32) {
        return true;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    return traits && traits->to_float;
}

static std::vector<ggml_type> plan_evo_types(
        const merge_params & params,
        const gguf_input & base,
        const std::vector<std::string> & tensor_names,
        llama_model * metadata_model,
        llama_ftype target_ftype) {
    llama_model_quantize_params quant_params = llama_model_quantize_default_params();
    quant_params.nthread = params.n_threads;
    quant_params.ftype = target_ftype;
    quant_params.allow_requantize = true;
    std::unique_ptr<quantize_state_impl, decltype(&llama_quant_free)> quant_state(
            llama_quant_init(metadata_model, &quant_params), llama_quant_free);
    if (!quant_state) {
        throw std::runtime_error("failed to initialize Evo quantization planner");
    }

    std::vector<ggml_type> result(tensor_names.size());
    std::vector<ggml_tensor *> quantizable;
    std::vector<size_t> quantizable_indices;
    for (size_t i = 0; i < tensor_names.size(); ++i) {
        ggml_tensor * tensor = base.tensors.at(tensor_names[i]).tensor;
        result[i] = tensor->type;
        if (tensor_can_decode(tensor) && llama_quant_tensor_allows_quantization(quant_state.get(), tensor)) {
            quantizable.push_back(tensor);
            quantizable_indices.push_back(i);
        }
    }

    std::vector<ggml_type> planned(quantizable.size());
    if (!quantizable.empty()) {
        llama_quant_compute_types(quant_state.get(), target_ftype, quantizable.data(), planned.data(), planned.size());
    }
    for (size_t i = 0; i < planned.size(); ++i) {
        result[quantizable_indices[i]] = planned[i];
    }
    return result;
}

struct evo_candidate {
    std::vector<float> genes;
    double fitness = INFINITY;
};

static void normalize_genes(evo_candidate & candidate, size_t n_tensors, size_t n_inputs) {
    for (size_t tensor = 0; tensor < n_tensors; ++tensor) {
        float sum = 0.0f;
        for (size_t input = 0; input < n_inputs; ++input) {
            float & gene = candidate.genes[tensor*n_inputs + input];
            gene = std::max(0.0f, gene);
            sum += gene;
        }
        if (sum == 0.0f) {
            for (size_t input = 0; input < n_inputs; ++input) {
                candidate.genes[tensor*n_inputs + input] = 1.0f / n_inputs;
            }
        } else {
            for (size_t input = 0; input < n_inputs; ++input) {
                candidate.genes[tensor*n_inputs + input] /= sum;
            }
        }
    }
}

static size_t write_backend_tensor_streaming(
        ggml_tensor * tensor,
        ggml_type output_type,
        const ggml_tensor * shape,
        std::ofstream & output) {
    const int64_t n_per_row = shape->ne[0];
    const int64_t n_rows = ggml_nelements(shape) / n_per_row;
    const size_t float_row_size = n_per_row * sizeof(float);
    const size_t rows_per_chunk = std::max<size_t>(1, (64ull * 1024 * 1024) / float_row_size);
    const size_t output_row_size = ggml_row_size(output_type, n_per_row);
    std::vector<float> decoded;
    std::vector<uint8_t> encoded;
    size_t written = 0;
    for (int64_t first_row = 0; first_row < n_rows; first_row += rows_per_chunk) {
        const int64_t chunk_rows = std::min<int64_t>(n_rows - first_row, rows_per_chunk);
        const size_t chunk_elements = chunk_rows * n_per_row;
        decoded.resize(chunk_elements);
        ggml_backend_tensor_get(
                tensor, decoded.data(), first_row * float_row_size, chunk_elements * sizeof(float));
        if (output_type == GGML_TYPE_F32) {
            output.write((const char *) decoded.data(), chunk_elements * sizeof(float));
            written += chunk_elements * sizeof(float);
            continue;
        }
        encoded.resize(output_row_size * chunk_rows);
        if (ggml_is_quantized(output_type)) {
            const size_t quantized = ggml_quantize_chunk(
                    output_type, decoded.data(), encoded.data(), 0, chunk_rows, n_per_row, nullptr);
            if (quantized != encoded.size()) {
                throw std::runtime_error("unexpected quantized size for tensor '" + std::string(shape->name) + "'");
            }
        } else {
            const ggml_type_traits * traits = ggml_get_type_traits(output_type);
            if (!traits || !traits->from_float_ref) {
                throw std::runtime_error("cannot encode output type " + std::string(ggml_type_name(output_type)));
            }
            traits->from_float_ref(decoded.data(), encoded.data(), chunk_elements);
        }
        output.write((const char *) encoded.data(), encoded.size());
        written += encoded.size();
    }
    const size_t padded = GGML_PAD(written, GGUF_DEFAULT_ALIGNMENT);
    write_zeros(output, padded - written);
    return padded;
}

static size_t merge_weighted_gpu_streaming(
        ggml_backend_t backend,
        const ggml_tensor * shape,
        const std::vector<std::unique_ptr<gguf_input>> & inputs,
        const std::string & name,
        const float * weights,
        ggml_type output_type,
        std::ofstream & output) {
    const size_t max_nodes = 8;
    ggml_init_params init_params = {
        /*.mem_size   = */ max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false),
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        throw std::runtime_error("failed to create GPU merge graph context");
    }
    ggml_tensor * accumulator = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, shape->ne);
    ggml_tensor * source = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, shape->ne);
    ggml_tensor * weighted = ggml_scale(ctx, source, 0.0f);
    ggml_tensor * merged = ggml_add(ctx, accumulator, weighted);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, max_nodes, false);
    ggml_build_forward_expand(graph, merged);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        throw std::runtime_error("failed to allocate GPU merge tensor buffer");
    }

    try {
        ggml_backend_buffer_clear(buffer, 0);
        const int64_t n_per_row = shape->ne[0];
        const int64_t n_rows = ggml_nelements(shape) / n_per_row;
        const size_t float_row_size = n_per_row * sizeof(float);
        const size_t rows_per_chunk = std::max<size_t>(1, (64ull * 1024 * 1024) / float_row_size);
        std::vector<uint8_t> bytes;
        std::vector<float> decoded;
        for (size_t input = 0; input < inputs.size(); ++input) {
            const ggml_tensor * input_tensor = inputs[input]->tensors.at(name).tensor;
            const size_t input_row_size = ggml_row_size(input_tensor->type, n_per_row);
            const ggml_type_traits * traits = ggml_get_type_traits(input_tensor->type);
            for (int64_t first_row = 0; first_row < n_rows; first_row += rows_per_chunk) {
                const int64_t chunk_rows = std::min<int64_t>(n_rows - first_row, rows_per_chunk);
                const size_t chunk_elements = chunk_rows * n_per_row;
                const size_t chunk_input_size = chunk_rows * input_row_size;
                decoded.resize(chunk_elements);
                if (input_tensor->type == GGML_TYPE_F32) {
                    inputs[input]->read_tensor_part(
                            name, first_row * input_row_size, decoded.data(), chunk_input_size);
                } else {
                    bytes.resize(chunk_input_size);
                    inputs[input]->read_tensor_part(
                            name, first_row * input_row_size, bytes.data(), bytes.size());
                    if (!traits || !traits->to_float) {
                        throw std::runtime_error("tensor '" + name + "' has unsupported type " +
                                ggml_type_name(input_tensor->type));
                    }
                    traits->to_float(bytes.data(), decoded.data(), chunk_elements);
                }
                ggml_backend_tensor_set(
                        source, decoded.data(), first_row * float_row_size, chunk_elements * sizeof(float));
            }
            const float scale_params[2] = { weights[input], 0.0f };
            memcpy(weighted->op_params, scale_params, sizeof(scale_params));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("GPU merge graph computation failed for tensor '" + std::string(shape->name) + "'");
            }
            ggml_backend_tensor_copy(merged, accumulator);
        }
        ggml_backend_synchronize(backend);
        std::vector<uint8_t>().swap(bytes);
        std::vector<float>().swap(decoded);
        const size_t written = write_backend_tensor_streaming(accumulator, output_type, shape, output);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return written;
    } catch (...) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
}

struct evo_output {
    gguf_context * gguf = nullptr;
    ggml_context * ctx = nullptr;
    std::ofstream file;
    std::string path;
    size_t metadata_size = 0;
    uint64_t expected_data_size = 0;

    evo_output(
            const std::string & path,
            const gguf_input & base,
            const std::vector<std::string> & tensor_names,
            const std::vector<ggml_type> & output_types,
            llama_ftype target_ftype) : path(path) {
        gguf = gguf_init_empty();
        gguf_set_kv(gguf, base.gguf);
        gguf_remove_key(gguf, GGUF_KEY_GENERAL_ALIGNMENT);
        gguf_set_val_u32(gguf, "general.file_type", target_ftype);
        gguf_remove_key(gguf, "split.no");
        gguf_remove_key(gguf, "split.count");
        gguf_remove_key(gguf, "split.tensors.count");

        ggml_init_params init_params = {
            /*.mem_size   = */ base.tensors.size() * ggml_tensor_overhead(),
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ true,
        };
        ctx = ggml_init(init_params);
        if (!ctx) {
            throw std::runtime_error("failed to create Evo output tensor context");
        }
        for (size_t i = 0; i < tensor_names.size(); ++i) {
            const ggml_tensor * source = base.tensors.at(tensor_names[i]).tensor;
            ggml_tensor * tensor = ggml_new_tensor(ctx, output_types[i], GGML_MAX_DIMS, source->ne);
            ggml_set_name(tensor, source->name);
            gguf_add_tensor(gguf, tensor);
            expected_data_size += GGML_PAD(ggml_nbytes(tensor), GGUF_DEFAULT_ALIGNMENT);
        }

        metadata_size = gguf_get_meta_size(gguf);
        file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        try {
            file.open(path, std::ios::binary);
            write_zeros(file, metadata_size);
        } catch (const std::ios_base::failure & error) {
            throw std::runtime_error("failed to create Evo candidate '" + path + "': " + error.what());
        }
    }

    evo_output(const evo_output &) = delete;
    evo_output & operator=(const evo_output &) = delete;

    ~evo_output() {
        file.exceptions(std::ios::goodbit);
        file.close();
        if (ctx) ggml_free(ctx);
        if (gguf) gguf_free(gguf);
    }

    size_t append(const std::vector<uint8_t> & encoded) {
        try {
            write_encoded_tensor(file, encoded, GGUF_DEFAULT_ALIGNMENT);
        } catch (const std::ios_base::failure & error) {
            throw std::runtime_error("failed to write Evo candidate '" + path + "': " + error.what());
        }
        return GGML_PAD(encoded.size(), GGUF_DEFAULT_ALIGNMENT);
    }

    void finish() {
        try {
            std::vector<uint8_t> metadata(metadata_size);
            gguf_get_meta_data(gguf, metadata.data());
            file.seekp(0);
            file.write((const char *) metadata.data(), metadata.size());
            file.close();
        } catch (const std::ios_base::failure & error) {
            throw std::runtime_error("failed to finish Evo candidate '" + path + "': " + error.what());
        }
    }
};

static std::string format_duration(double seconds) {
    const uint64_t total = seconds > 0.0 ? (uint64_t) seconds : 0;
    const uint64_t hours = total / 3600;
    const uint64_t minutes = total / 60 % 60;
    const uint64_t secs = total % 60;
    char result[32];
    snprintf(result, sizeof(result), "%02llu:%02llu:%02llu",
            (unsigned long long) hours, (unsigned long long) minutes, (unsigned long long) secs);
    return result;
}

static int select_evo_worker_count(
        const merge_params & params,
        const ggml_tensor * tensor,
        size_t n_candidates) {
    const size_t elements = ggml_nelements(tensor);
    const size_t bytes_per_worker = elements > SIZE_MAX / (3 * sizeof(float))
        ? SIZE_MAX
        : elements * 3 * sizeof(float);
    size_t available_memory = SIZE_MAX;
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    size_t value_kib;
    std::string unit;
    while (meminfo >> key >> value_kib >> unit) {
        if (key == "MemAvailable:") {
            available_memory = value_kib > SIZE_MAX / 1024 ? SIZE_MAX : value_kib * 1024;
            break;
        }
    }
    const size_t effective_budget = std::min(params.memory_budget, available_memory - available_memory / 3);
    const size_t memory_workers = bytes_per_worker == 0
        ? 1
        : std::max<size_t>(1, effective_budget / bytes_per_worker);
    return std::max(1, std::min<int>(params.n_threads, std::min(n_candidates, memory_workers)));
}

static void write_evo_candidates(
        const std::vector<std::string> & paths,
        const merge_params & params,
        const std::vector<std::unique_ptr<gguf_input>> & inputs,
        const std::vector<std::string> & tensor_names,
        const std::vector<evo_candidate> & candidates,
        const std::vector<size_t> & gene_offsets,
        const std::vector<ggml_type> & output_types,
        llama_ftype target_ftype,
        ggml_backend_t merge_backend) {
    if (paths.size() != candidates.size() || output_types.size() != tensor_names.size() ||
            gene_offsets.size() != tensor_names.size()) {
        throw std::runtime_error("internal Evo output size mismatch");
    }
    const gguf_input & base = *inputs[0];
    std::vector<std::unique_ptr<evo_output>> outputs;
    outputs.reserve(paths.size());
    for (const std::string & path : paths) {
        outputs.emplace_back(new evo_output(path, base, tensor_names, output_types, target_ftype));
    }
    uint64_t total_bytes = 0;
    for (const auto & output : outputs) {
        total_bytes += output->expected_data_size;
    }
    uint64_t required_bytes = total_bytes;
    for (const auto & output : outputs) {
        required_bytes += output->metadata_size;
    }
    const std::filesystem::path output_dir = std::filesystem::path(paths[0]).parent_path().empty()
        ? std::filesystem::current_path()
        : std::filesystem::path(paths[0]).parent_path();
    std::error_code space_error;
    const std::filesystem::space_info space = std::filesystem::space(output_dir, space_error);
    if (!space_error) {
        const uint64_t reserve_bytes = std::max<uint64_t>(2ull * 1024 * 1024 * 1024, required_bytes / 20);
        if (space.available < required_bytes || space.available - required_bytes < reserve_bytes) {
            throw std::runtime_error(
                    "insufficient space in Evo temp directory '" + output_dir.string() + "': need " +
                    std::to_string(required_bytes / 1000000000.0) + " GB plus " +
                    std::to_string(reserve_bytes / 1000000000.0) + " GB reserve, have " +
                    std::to_string(space.available / 1000000000.0) +
                    " GB; reduce --population or select a larger --temp-dir");
        }
    }
    std::atomic<uint64_t> completed_bytes{0};
    std::mutex progress_mutex;
    const auto progress_start = std::chrono::steady_clock::now();
    auto last_progress = progress_start - std::chrono::seconds(5);
    auto report_progress = [&](size_t written, size_t tensor_index, bool force) {
        const uint64_t completed = completed_bytes.fetch_add(written) + written;
        std::lock_guard<std::mutex> lock(progress_mutex);
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_progress < std::chrono::seconds(5)) {
            return;
        }
        last_progress = now;
        const double elapsed = std::chrono::duration<double>(now - progress_start).count();
        const double completed_gb = completed / 1e9;
        const double total_gb = total_bytes / 1e9;
        const double gb_per_minute = elapsed > 0.0 ? completed_gb * 60.0 / elapsed : 0.0;
        const double eta = completed > 0 && completed < total_bytes
            ? elapsed * (total_bytes - completed) / completed
            : 0.0;
        printf("evo merge: tensors %zu/%zu, %.1f%%, %.2f/%.2f GB, %.2f GB/min, elapsed %s, ETA %s\n",
                std::min(tensor_index + 1, tensor_names.size()), tensor_names.size(),
                total_bytes > 0 ? 100.0 * completed / total_bytes : 100.0,
                completed_gb, total_gb, gb_per_minute,
                format_duration(elapsed).c_str(), format_duration(eta).c_str());
        fflush(stdout);
    };

    const ggml_tensor * largest_tensor = nullptr;
    for (const std::string & name : tensor_names) {
        const ggml_tensor * tensor = base.tensors.at(name).tensor;
        if (!largest_tensor || ggml_nelements(tensor) > ggml_nelements(largest_tensor)) {
            largest_tensor = tensor;
        }
    }
    const int n_workers = merge_backend ? 1 : select_evo_worker_count(params, largest_tensor, candidates.size());
    std::unique_ptr<common_jsonl_worker_pool> worker_pool;
    std::unique_ptr<common_jsonl_worker_pool> element_pool;
    const int n_element_workers = candidates.size() == 1 ? params.n_threads : 1;
    if (!merge_backend) {
        worker_pool.reset(new common_jsonl_worker_pool(n_workers));
        printf("evo merge: %d CPU candidate workers\n", n_workers);
        if (n_element_workers > 1) {
            element_pool.reset(new common_jsonl_worker_pool(n_element_workers));
            printf("evo merge: %d CPU tensor workers\n", n_element_workers);
        }
    }
    for (size_t tensor_index = 0; tensor_index < tensor_names.size(); ++tensor_index) {
        const std::string & name = tensor_names[tensor_index];
        const ggml_tensor * shape = base.tensors.at(name).tensor;
        if (!tensor_can_decode(shape)) {
            std::vector<uint8_t> base_bytes;
            base.read_tensor(name, base_bytes);
            for (size_t input = 1; input < inputs.size(); ++input) {
                const ggml_tensor * other_shape = inputs[input]->tensors.at(name).tensor;
                if (other_shape->type != shape->type) {
                    throw std::runtime_error("non-floating tensor type differs for '" + name + "'");
                }
                std::vector<uint8_t> other_bytes;
                inputs[input]->read_tensor(name, other_bytes);
                if (base_bytes != other_bytes) {
                    throw std::runtime_error("non-floating tensor data differs for '" + name + "'");
                }
            }
            for (auto & output : outputs) {
                report_progress(output->append(base_bytes), tensor_index, false);
            }
            continue;
        }

        for (size_t input = 0; input < inputs.size(); ++input) {
            if (!tensor_can_decode(inputs[input]->tensors.at(name).tensor)) {
                throw std::runtime_error("floating tensor type differs for '" + name + "'");
            }
        }

        if (merge_backend) {
            for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
                const float * weights = candidates[candidate].genes.data() + gene_offsets[tensor_index];
                try {
                    report_progress(
                            merge_weighted_gpu_streaming(
                                    merge_backend, shape, inputs, name, weights,
                                    output_types[tensor_index], outputs[candidate]->file),
                            tensor_index, false);
                } catch (const std::ios_base::failure & error) {
                    throw std::runtime_error(
                            "failed to write Evo candidate '" + outputs[candidate]->path + "': " + error.what());
                }
            }
        } else {
            worker_pool->parallel_for(candidates.size(), [&](size_t candidate, size_t) {
                const float * weights = candidates[candidate].genes.data() + gene_offsets[tensor_index];
                std::vector<float> merged(ggml_nelements(shape), 0.0f);
                std::vector<uint8_t> bytes;
                std::vector<float> decoded;
                for (size_t input = 0; input < inputs.size(); ++input) {
                    decode_tensor(*inputs[input], name, bytes, decoded);
                    if (element_pool && merged.size() >= 262144) {
                        element_pool->parallel_for(n_element_workers, [&](size_t task, size_t) {
                            const size_t begin = merged.size() * task / n_element_workers;
                            const size_t end = merged.size() * (task + 1) / n_element_workers;
                            for (size_t element = begin; element < end; ++element) {
                                merged[element] += weights[input] * decoded[element];
                            }
                        });
                    } else {
                        for (size_t element = 0; element < merged.size(); ++element) {
                            merged[element] += weights[input] * decoded[element];
                        }
                    }
                }
                report_progress(
                        outputs[candidate]->append(encode_tensor(
                                output_types[tensor_index], shape, merged, element_pool.get(), n_element_workers)),
                        tensor_index, false);
            });
        }
    }
    for (auto & output : outputs) output->finish();
    report_progress(0, tensor_names.size() - 1, true);
}

struct calibration_sample {
    std::vector<llama_token> tokens;
    size_t loss_begin = 1;
};

static std::vector<llama_token> tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special) {
    std::vector<llama_token> tokens(text.size() + 8);
    int32_t count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (count < 0) {
        tokens.resize(-count);
        count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), add_special, true);
    }
    if (count < 0) {
        throw std::runtime_error("failed to tokenize calibration text");
    }
    tokens.resize(count);
    return tokens;
}

static calibration_sample make_calibration_sample(
        const llama_vocab * vocab,
        const std::string & prompt,
        const std::string & response) {
    calibration_sample sample;
    sample.tokens = tokenize_text(vocab, prompt, true);
    sample.loss_begin = sample.tokens.size();
    std::vector<llama_token> response_tokens = tokenize_text(vocab, response, false);
    sample.tokens.insert(sample.tokens.end(), response_tokens.begin(), response_tokens.end());
    return sample;
}

static std::string calibration_message_payload(const nlohmann::json & message) {
    std::string result;
    const char * reasoning_key = message.contains("reasoning") ? "reasoning" : "reasoning_content";
    if (message.contains(reasoning_key) && message[reasoning_key].is_string() && !message[reasoning_key].get_ref<const std::string &>().empty()) {
        result += message[reasoning_key].get_ref<const std::string &>();
        result += '\n';
    }
    if (message.contains("content") && message["content"].is_string()) {
        result += message["content"].get_ref<const std::string &>();
    } else if (message.contains("content") && message["content"].is_array()) {
        for (const auto & part : message["content"]) {
            if (part.is_object() && part.value("type", "") == "text" && part.contains("text") && part["text"].is_string()) {
                if (!result.empty() && result.back() != '\n') result += '\n';
                result += part["text"].get_ref<const std::string &>();
            }
        }
    }
    if (message.contains("tool_calls") && message["tool_calls"].is_array() && !message["tool_calls"].empty()) {
        if (!result.empty() && result.back() != '\n') {
            result += '\n';
        }
        result += message["tool_calls"].dump();
    }
    return result;
}

static calibration_sample make_template_free_chat_sample(
        const llama_vocab * vocab,
        const nlohmann::json & messages,
        size_t assistant_index) {
    std::string prompt;
    for (size_t i = 0; i < assistant_index; ++i) {
        prompt += messages[i].value("role", "user");
        prompt += ":\n";
        prompt += calibration_message_payload(messages[i]);
        prompt += '\n';
    }
    prompt += "assistant:\n";
    const std::string response = calibration_message_payload(messages[assistant_index]);
    if (response.empty()) {
        throw std::runtime_error("last assistant calibration message has no text or tool calls");
    }
    return make_calibration_sample(vocab, prompt, response);
}

static std::vector<calibration_sample> load_calibration(
        const std::string & path,
        int32_t n_threads,
        const llama_model * metadata_model,
        bool ignore_chat_template) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open calibration file " + path);
    }
    const llama_vocab * vocab = llama_model_get_vocab(metadata_model);
    std::vector<calibration_sample> samples;
    const bool jsonl = path.size() >= 6 && path.substr(path.size() - 6) == ".jsonl";
    if (!jsonl) {
        std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (!text.empty()) samples.push_back(make_calibration_sample(vocab, "", text));
        return samples;
    }

    std::vector<common_jsonl_line> lines = common_jsonl_read_lines(path, COMMON_JSONL_EMPTY_LINE_SKIP);
    std::vector<nlohmann::json> parsed(lines.size());
    std::vector<std::string> errors(lines.size());
    common_jsonl_worker_pool pool(common_jsonl_worker_count(n_threads, lines.size()));
    pool.parallel_for(lines.size(), [&](size_t i, size_t) {
        if (trim(lines[i].text).empty()) return;
        try {
            parsed[i] = nlohmann::json::parse(lines[i].text);
        } catch (const std::exception & error) {
            errors[i] = error.what();
        }
    });
    for (common_jsonl_line & line : lines) std::string().swap(line.text);
    common_chat_templates_ptr templates;
    if (!ignore_chat_template) {
        templates = common_chat_templates_init(metadata_model, "");
    }
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!errors[i].empty()) throw std::runtime_error("invalid calibration JSONL line " + std::to_string(lines[i].number) + ": " + errors[i]);
        const nlohmann::json & item = parsed[i];
        if (item.is_null()) continue;
        if (item.contains("messages") && item["messages"].is_array()) {
            if (ignore_chat_template) {
                const nlohmann::json & raw_messages = item["messages"];
                size_t assistant_index = raw_messages.size();
                for (size_t index = raw_messages.size(); index-- > 0;) {
                    if (raw_messages[index].is_object() && raw_messages[index].value("role", "") == "assistant" &&
                            !calibration_message_payload(raw_messages[index]).empty()) {
                        assistant_index = index;
                        break;
                    }
                }
                if (assistant_index == raw_messages.size()) {
                    throw std::runtime_error("calibration JSONL line " + std::to_string(lines[i].number) +
                            " has no non-empty assistant response");
                }
                samples.push_back(make_template_free_chat_sample(vocab, raw_messages, assistant_index));
                continue;
            }
            std::vector<common_chat_msg> messages;
            for (const auto & data : item["messages"]) {
                messages.push_back(common_jsonl_parse_chat_message(data, COMMON_JSONL_CHAT_PARSE_OPTIONAL_TEXT));
            }
            size_t assistant_index = messages.size();
            for (size_t index = messages.size(); index-- > 0;) {
                if (messages[index].role == "assistant") {
                    assistant_index = index;
                    break;
                }
            }
            if (assistant_index == messages.size()) {
                throw std::runtime_error("calibration JSONL line " + std::to_string(lines[i].number) + " has no assistant response");
            }
            common_chat_templates_inputs chat_inputs;
            chat_inputs.messages.assign(messages.begin(), messages.begin() + assistant_index);
            chat_inputs.add_generation_prompt = true;
            const std::string prompt = common_chat_templates_apply(templates.get(), chat_inputs).prompt;
            chat_inputs.messages.assign(messages.begin(), messages.begin() + assistant_index + 1);
            chat_inputs.add_generation_prompt = false;
            const std::string full = common_chat_templates_apply(templates.get(), chat_inputs).prompt;
            if (full.rfind(prompt, 0) != 0) {
                throw std::runtime_error("chat template output is not prefixed by its generation prompt on calibration JSONL line " +
                        std::to_string(lines[i].number));
            }
            samples.push_back(make_calibration_sample(vocab, prompt, full.substr(prompt.size())));
        } else if (item.contains("prompt") && item.contains("response")) {
            samples.push_back(make_calibration_sample(
                    vocab, item.at("prompt").get<std::string>(), item.at("response").get<std::string>()));
        } else if (item.contains("text")) {
            samples.push_back(make_calibration_sample(vocab, "", item.at("text").get<std::string>()));
        } else {
            throw std::runtime_error("calibration JSONL line " + std::to_string(lines[i].number) + " has no supported text field");
        }
    }
    if (samples.empty()) {
        throw std::runtime_error("calibration file contains no usable samples");
    }
    return samples;
}

static double evaluate_candidate(
        const std::string & path,
        const merge_params & params,
        const std::vector<calibration_sample> & samples,
        ggml_backend_dev_t fitness_device,
        int fitness_threads) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.gpu_layers;
    std::vector<ggml_backend_dev_t> devices;
    if (fitness_device) {
        devices = { fitness_device, nullptr };
        model_params.devices = devices.data();
    }
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(path.c_str(), model_params), llama_model_free);
    if (!model) {
        throw std::runtime_error("failed to load Evo candidate " + path);
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = params.context_size;
    context_params.n_batch = params.context_size;
    context_params.n_ubatch = params.context_size;
    context_params.n_threads = fitness_threads;
    context_params.n_threads_batch = fitness_threads;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        throw std::runtime_error("failed to create Evo fitness context");
    }

    double total_nll = 0.0;
    size_t total_tokens = 0;
    for (const calibration_sample & sample : samples) {
        const std::vector<llama_token> & tokens = sample.tokens;
        for (size_t begin = 0; begin + 1 < tokens.size();) {
            const size_t count = std::min<size_t>(params.context_size, tokens.size() - begin);
            std::vector<llama_token> targets;
            for (size_t i = 0; i + 1 < count; ++i) {
                if (begin + i + 1 >= sample.loss_begin) targets.push_back(tokens[begin + i + 1]);
            }
            if (targets.empty()) {
                begin += count > 1 ? count - 1 : count;
                continue;
            }
            llama_batch batch = llama_batch_init(count, 0, 1);
            batch.n_tokens = count;
            for (size_t i = 0; i < count; ++i) {
                batch.token[i] = tokens[begin + i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i + 1 < count && begin + i + 1 >= sample.loss_begin;
            }
            llama_memory_clear(llama_get_memory(context.get()), true);
            float batch_nll = 0.0f;
            if (llama_decode_sparse_cross_entropy(context.get(), batch, targets.data(), targets.size(), &batch_nll) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("sparse cross-entropy decode failed during Evo fitness evaluation");
            }
            total_nll += (double) batch_nll*targets.size();
            total_tokens += targets.size();
            llama_batch_free(batch);
            begin += count > 1 ? count - 1 : count;
        }
    }
    if (total_tokens == 0) {
        throw std::runtime_error("calibration data produced fewer than two tokens");
    }
    return total_nll / total_tokens;
}

static void run_evo(const merge_params & params, const std::vector<std::unique_ptr<gguf_input>> & inputs) {
    const llama_ftype target_ftype = parse_target_ftype(params.target_type);
    std::vector<std::string> tensor_names;
    tensor_names.reserve(inputs[0]->tensors.size());
    for (const auto & entry : inputs[0]->tensors) {
        tensor_names.push_back(entry.first);
    }
    std::vector<size_t> gene_offsets(tensor_names.size(), SIZE_MAX);
    size_t n_gene_tensors = 0;
    for (size_t i = 0; i < tensor_names.size(); ++i) {
        if (tensor_can_decode(inputs[0]->tensors.at(tensor_names[i]).tensor)) {
            gene_offsets[i] = n_gene_tensors++ * inputs.size();
        }
    }
    const uint32_t seed = params.seed_set ? params.seed : std::random_device{}();
    std::mt19937 rng(seed);
    const size_t gene_count = n_gene_tensors * inputs.size();
    if (gene_count == 0) {
        throw std::runtime_error("input model has no tensors");
    }
    std::vector<float> mean(gene_count, 1.0f / inputs.size());
    std::vector<float> variance(gene_count, 1.0f);
    std::vector<float> recombination(params.elite_count);
    float recombination_sum = 0.0f;
    for (int i = 0; i < params.elite_count; ++i) {
        recombination[i] = std::log(params.elite_count + 0.5f) - std::log(i + 1.0f);
        recombination_sum += recombination[i];
    }
    for (float & weight : recombination) weight /= recombination_sum;
    float sigma = params.mutation;
    const float covariance_rate = std::min(0.25f, 2.0f / (std::sqrt((float) gene_count) + 2.0f));
    std::normal_distribution<float> normal(0.0f, 1.0f);

    evo_candidate best;
    llama_backend_init();
    ggml_backend_t merge_backend = nullptr;
    std::vector<std::string> temporary_paths;
    std::string best_path;
    try {
        llama_model_params metadata_params = llama_model_default_params();
        metadata_params.n_gpu_layers = 0;
        metadata_params.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> metadata_model(
                llama_model_load_from_file(inputs[0]->path.c_str(), metadata_params), llama_model_free);
        if (!metadata_model) {
            throw std::runtime_error("failed to load base model metadata for Evo");
        }

        const std::vector<ggml_type> output_types = plan_evo_types(
                params, *inputs[0], tensor_names, metadata_model.get(), target_ftype);
        if (get_kv_string(inputs[0]->gguf, "tokenizer.ggml.model") == "no_vocab") {
            throw std::runtime_error("Evo fitness requires a model with an initialized tokenizer");
        }
        const std::vector<calibration_sample> calibration = load_calibration(
                params.calibration, params.n_threads, metadata_model.get(), params.ignore_chat_template);
        metadata_model.reset();

        std::vector<ggml_backend_dev_t> fitness_devices;
        std::vector<std::string> fitness_device_names = params.devices;
        if (!params.device.empty()) fitness_device_names.push_back(params.device);
        for (const std::string & name : fitness_device_names) {
            ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
            if (!device) {
                throw std::runtime_error("backend device not found: " + name);
            }
            if (std::find(fitness_devices.begin(), fitness_devices.end(), device) != fitness_devices.end()) {
                throw std::runtime_error("duplicate fitness device: " + name);
            }
            fitness_devices.push_back(device);
        }
        const size_t n_eval_workers = fitness_devices.empty()
            ? 1
            : std::min<size_t>(fitness_devices.size(), params.population);
        const int fitness_threads = std::max(1, params.n_threads / (int) n_eval_workers);
        if (fitness_devices.size() > 1) {
            printf("evo: %zu concurrent fitness devices, %d host threads each\n", n_eval_workers, fitness_threads);
        }

        if (params.merge_gpu) {
            ggml_backend_dev_t device = !fitness_devices.empty()
                ? fitness_devices[0]
                : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (!device || (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU &&
                            ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_IGPU &&
                            ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_ACCEL)) {
                throw std::runtime_error("--merge-gpu requires a GPU or accelerator device");
            }
            merge_backend = ggml_backend_dev_init(device, nullptr);
            if (!merge_backend) {
                throw std::runtime_error("failed to initialize merge device " + std::string(ggml_backend_dev_name(device)));
            }
            printf("evo: GPU merge device %s (%s)\n", ggml_backend_dev_name(device), ggml_backend_dev_description(device));
        }

        const std::filesystem::path temp_dir = params.temp_dir.empty()
            ? std::filesystem::temp_directory_path()
            : std::filesystem::path(params.temp_dir);
        std::filesystem::create_directories(temp_dir);
        const uint32_t run_id = std::random_device{}();
        for (int generation = 0; generation < params.generations; ++generation) {
            std::vector<evo_candidate> population(params.population);
            for (int candidate = 0; candidate < params.population; ++candidate) {
                population[candidate].genes.resize(gene_count);
                for (size_t gene = 0; gene < gene_count; ++gene) {
                    const float sample = candidate == 0 ? 0.0f : normal(rng);
                    population[candidate].genes[gene] = mean[gene] + sigma * std::sqrt(variance[gene]) * sample;
                }
                normalize_genes(population[candidate], n_gene_tensors, inputs.size());
            }

            temporary_paths.clear();
            temporary_paths.reserve(population.size());
            for (size_t candidate = 0; candidate < population.size(); ++candidate) {
                const std::string filename = "llama-merge-" + std::to_string(run_id) + "-g" +
                    std::to_string(generation) + "-c" + std::to_string(candidate) + ".gguf";
                temporary_paths.push_back((temp_dir / filename).string());
            }
            const std::vector<std::string> candidate_paths = temporary_paths;
            if (params.evo_mode == "ram") {
                printf("evo generation %d/%d: one RAM candidate at a time, CPU merge then GPU fitness (%zu candidates)\n",
                        generation + 1, params.generations, population.size());
                const ggml_backend_dev_t fitness_device = fitness_devices.empty() ? nullptr : fitness_devices[0];
                for (size_t candidate = 0; candidate < population.size(); ++candidate) {
                    const std::string candidate_path = candidate_paths[candidate];
                    temporary_paths = { candidate_path };
                    printf("evo generation %d candidate %zu/%zu: CPU merging\n",
                            generation + 1, candidate + 1, population.size());
                    write_evo_candidates(
                            { candidate_path }, params, inputs, tensor_names, { population[candidate] },
                            gene_offsets, output_types, target_ftype, nullptr);
                    printf("evo generation %d candidate %zu/%zu: loading for fitness\n",
                            generation + 1, candidate + 1, population.size());
                    population[candidate].fitness = evaluate_candidate(
                            candidate_path, params, calibration, fitness_device, fitness_threads);
                    if (!std::isfinite(population[candidate].fitness)) {
                        population[candidate].fitness = INFINITY;
                    }
                    printf("evo generation %d candidate %zu: nll %.6f ppl %.6f\n",
                            generation + 1, candidate + 1, population[candidate].fitness,
                            std::exp(population[candidate].fitness));
                    if (population[candidate].fitness < best.fitness) {
                        best = population[candidate];
                        printf("evo generation %d candidate %zu: fitness and genes retained as global winner\n",
                                generation + 1, candidate + 1);
                    }
                    std::remove(candidate_path.c_str());
                    temporary_paths.clear();
                }
            } else {
                printf("evo generation %d/%d: low-mem tensor-major merge of %zu candidates\n",
                        generation + 1, params.generations, population.size());
                write_evo_candidates(
                        temporary_paths, params, inputs, tensor_names, population, gene_offsets, output_types, target_ftype, merge_backend);

                common_jsonl_worker_pool eval_pool(n_eval_workers);
                eval_pool.parallel_for(population.size(), [&](size_t candidate, size_t worker) {
                    const ggml_backend_dev_t device = fitness_devices.empty() ? nullptr : fitness_devices[worker];
                    population[candidate].fitness = evaluate_candidate(
                            temporary_paths[candidate], params, calibration, device, fitness_threads);
                    if (!std::isfinite(population[candidate].fitness)) {
                        population[candidate].fitness = INFINITY;
                    }
                });
                const size_t best_index = std::min_element(
                        population.begin(), population.end(), [](const evo_candidate & a, const evo_candidate & b) {
                            return a.fitness < b.fitness;
                        }) - population.begin();
                if (!std::isfinite(population[best_index].fitness)) {
                    throw std::runtime_error("all Evo candidates produced non-finite fitness");
                }
                if (best.genes.empty() || population[best_index].fitness < best.fitness) {
                    if (!best_path.empty()) std::remove(best_path.c_str());
                    best = population[best_index];
                    best_path = temporary_paths[best_index];
                }
                for (size_t candidate = 0; candidate < population.size(); ++candidate) {
                    printf("evo generation %d candidate %d: nll %.6f ppl %.6f\n",
                            generation + 1, (int) candidate + 1, population[candidate].fitness,
                            std::exp(population[candidate].fitness));
                    if (temporary_paths[candidate] != best_path) std::remove(temporary_paths[candidate].c_str());
                }
                temporary_paths.clear();
            }
            if (best.genes.empty()) {
                throw std::runtime_error("all Evo candidates produced non-finite fitness");
            }
            std::sort(population.begin(), population.end(), [](const evo_candidate & a, const evo_candidate & b) {
                return a.fitness < b.fitness;
            });
            printf("evo generation %d best: nll %.6f ppl %.6f\n",
                    generation + 1, population[0].fitness, std::exp(population[0].fitness));
            if (generation + 1 == params.generations) break;

            const std::vector<float> previous_mean = mean;
            std::fill(mean.begin(), mean.end(), 0.0f);
            for (int elite = 0; elite < params.elite_count; ++elite) {
                for (size_t gene = 0; gene < gene_count; ++gene) {
                    mean[gene] += recombination[elite] * population[elite].genes[gene];
                }
            }
            double normalized_step2 = 0.0;
            for (size_t gene = 0; gene < gene_count; ++gene) {
                float selected_step2 = 0.0f;
                for (int elite = 0; elite < params.elite_count; ++elite) {
                    const float step = (population[elite].genes[gene] - previous_mean[gene]) / std::max(sigma, 1e-6f);
                    selected_step2 += recombination[elite] * step * step;
                }
                variance[gene] = std::max(1e-4f, (1.0f - covariance_rate) * variance[gene] + covariance_rate * selected_step2);
                const float mean_step = (mean[gene] - previous_mean[gene]) /
                    (std::max(sigma, 1e-6f) * std::sqrt(variance[gene]));
                normalized_step2 += mean_step * mean_step;
            }
            const double normalized_step = std::sqrt(normalized_step2 / gene_count);
            sigma = std::max(0.005f, std::min(1.0f, (float) (sigma * std::exp(0.2 * (normalized_step - 1.0)))));
            printf("evo generation %d CMA-ES sigma %.6f\n", generation + 1, sigma);
        }
        if (params.evo_mode == "ram") {
            printf("evo final: regenerating winner to %s (nll %.6f, ppl %.6f, seed %u)\n",
                    params.output.c_str(), best.fitness, std::exp(best.fitness), seed);
            write_evo_candidates(
                    { params.output }, params, inputs, tensor_names, { best }, gene_offsets,
                    output_types, target_ftype, nullptr);
        } else {
            printf("evo final: promoting evaluated candidate to %s (nll %.6f, ppl %.6f, seed %u)\n",
                    params.output.c_str(), best.fitness, std::exp(best.fitness), seed);
            std::error_code rename_error;
            std::filesystem::rename(best_path, params.output, rename_error);
            if (rename_error) {
                std::filesystem::copy_file(
                        best_path, params.output, std::filesystem::copy_options::overwrite_existing);
                std::remove(best_path.c_str());
            }
            best_path.clear();
        }
        if (merge_backend) ggml_backend_free(merge_backend);
        llama_backend_free();
    } catch (...) {
        for (const std::string & path : temporary_paths) std::remove(path.c_str());
        if (!best_path.empty()) std::remove(best_path.c_str());
        if (merge_backend) ggml_backend_free(merge_backend);
        llama_backend_free();
        throw;
    }
}

#ifndef LLAMA_MERGE_NO_MAIN
int main(int argc, char ** argv) {
    try {
        const merge_params params = parse_args(argc, argv);
        std::vector<std::unique_ptr<gguf_input>> inputs;
        inputs.emplace_back(new gguf_input(params.base));
        for (const std::string & path : params.models) {
            inputs.emplace_back(new gguf_input(path));
        }

        const gguf_input & base = *inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            validate_metadata_compatibility(base, *inputs[i], params.ignore_chat_template);
            if (base.tensors.size() != inputs[i]->tensors.size()) {
                throw std::runtime_error("input models have different tensor counts");
            }
            for (const auto & entry : base.tensors) {
                const auto it = inputs[i]->tensors.find(entry.first);
                if (it == inputs[i]->tensors.end() || !same_shape(entry.second.tensor, it->second.tensor)) {
                    throw std::runtime_error("tensor schema differs for '" + entry.first + "'");
                }
            }
        }

        if (params.method == "evo") {
            run_evo(params, inputs);
            return 0;
        }

        const ggml_type precision_output = select_precision_type(inputs);
        const bool preserve_quant = precision_output == GGML_TYPE_COUNT;
        const uint32_t output_ftype = preserve_quant ? base.file_type : type_to_ftype(precision_output);
        if (preserve_quant) {
            for (const auto & entry : base.tensors) {
                for (size_t i = 1; i < inputs.size(); ++i) {
                    if (entry.second.tensor->type != inputs[i]->tensors.at(entry.first).tensor->type) {
                        throw std::runtime_error("same quant model type has different tensor types for '" + entry.first + "'");
                    }
                }
            }
        }

        gguf_context * output_gguf = gguf_init_empty();
        gguf_set_kv(output_gguf, base.gguf);
        gguf_remove_key(output_gguf, GGUF_KEY_GENERAL_ALIGNMENT);
        gguf_set_val_u32(output_gguf, "general.file_type", output_ftype);
        gguf_remove_key(output_gguf, "split.no");
        gguf_remove_key(output_gguf, "split.count");
        gguf_remove_key(output_gguf, "split.tensors.count");

        ggml_init_params init_params = {
            /*.mem_size   = */ base.tensors.size() * ggml_tensor_overhead(),
            /*.mem_buffer = */ nullptr,
            /*.no_alloc   = */ true,
        };
        ggml_context * output_ctx = ggml_init(init_params);
        std::map<std::string, ggml_type> output_types;
        for (const auto & entry : base.tensors) {
            const ggml_tensor * input_tensor = entry.second.tensor;
            const ggml_type output_type = !tensor_can_decode(input_tensor) || preserve_quant ? input_tensor->type : precision_output;
            ggml_tensor * output_tensor = ggml_new_tensor(output_ctx, output_type, GGML_MAX_DIMS, input_tensor->ne);
            ggml_set_name(output_tensor, input_tensor->name);
            gguf_add_tensor(output_gguf, output_tensor);
            output_types[entry.first] = output_type;
        }

        std::ofstream output(params.output, std::ios::binary);
        output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        write_zeros(output, gguf_get_meta_size(output_gguf));

        std::vector<std::string> tensor_names;
        tensor_names.reserve(base.tensors.size());
        for (const auto & entry : base.tensors) {
            tensor_names.push_back(entry.first);
        }

        std::atomic<size_t> next_job { 0 };
        size_t next_to_write = 0;
        bool stop = false;
        std::exception_ptr worker_error;
        std::mutex write_mutex;
        std::condition_variable write_condition;
        const int n_workers = select_worker_count(params, base, inputs.size());
        if (n_workers < params.n_threads) {
            printf("llama-merge: limiting workers from %d to %d to bound TIES working memory\n", params.n_threads, n_workers);
        }
        std::vector<std::thread> workers;
        workers.reserve(n_workers);

        for (int worker = 0; worker < n_workers; ++worker) {
            workers.emplace_back([&] {
                try {
                    std::vector<uint8_t> bytes;
                    std::vector<float> base_values;
                    std::vector<float> model_values;
                    std::vector<float> merged;

                    for (;;) {
                        const size_t tensor_index = next_job.fetch_add(1);
                        if (tensor_index >= tensor_names.size()) {
                            return;
                        }
                        const std::string & name = tensor_names[tensor_index];
                        if (!tensor_can_decode(base.tensors.at(name).tensor)) {
                            std::vector<uint8_t> base_bytes;
                            base.read_tensor(name, base_bytes);
                            for (size_t i = 1; i < inputs.size(); ++i) {
                                if (inputs[i]->tensors.at(name).tensor->type != base.tensors.at(name).tensor->type) {
                                    throw std::runtime_error("non-floating tensor type differs for '" + name + "'");
                                }
                                std::vector<uint8_t> other_bytes;
                                inputs[i]->read_tensor(name, other_bytes);
                                if (base_bytes != other_bytes) {
                                    throw std::runtime_error("non-floating tensor data differs for '" + name + "'");
                                }
                            }
                            std::unique_lock<std::mutex> lock(write_mutex);
                            write_condition.wait(lock, [&] { return stop || tensor_index == next_to_write; });
                            if (stop) return;
                            write_encoded_tensor(output, base_bytes, GGUF_DEFAULT_ALIGNMENT);
                            ++next_to_write;
                            lock.unlock();
                            write_condition.notify_all();
                            continue;
                        }
                        decode_tensor(base, name, bytes, base_values);
                        std::vector<std::vector<float>> tasks;
                        tasks.reserve(inputs.size() - 1);
                        for (size_t i = 1; i < inputs.size(); ++i) {
                            decode_tensor(*inputs[i], name, bytes, model_values);
                            for (size_t j = 0; j < model_values.size(); ++j) {
                                model_values[j] -= base_values[j];
                            }
                            tasks.push_back(std::move(model_values));
                        }
                        ties_merge(base_values, tasks, params.density, merged);

                        std::unique_lock<std::mutex> lock(write_mutex);
                        write_condition.wait(lock, [&] { return stop || tensor_index == next_to_write; });
                        if (stop) {
                            return;
                        }
                        write_tensor(output, output_types.at(name), base.tensors.at(name).tensor, merged, GGUF_DEFAULT_ALIGNMENT);
                        printf("merged %s\n", name.c_str());
                        ++next_to_write;
                        lock.unlock();
                        write_condition.notify_all();
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(write_mutex);
                    if (!worker_error) {
                        worker_error = std::current_exception();
                    }
                    stop = true;
                    write_condition.notify_all();
                }
            });
        }
        for (std::thread & worker : workers) {
            worker.join();
        }
        if (worker_error) {
            std::rethrow_exception(worker_error);
        }

        std::vector<uint8_t> metadata(gguf_get_meta_size(output_gguf));
        gguf_get_meta_data(output_gguf, metadata.data());
        output.seekp(0);
        output.write((const char *) metadata.data(), metadata.size());
        ggml_free(output_ctx);
        gguf_free(output_gguf);
        printf("wrote %s using TIES (%zu models, density %.3f, %d threads)\n", params.output.c_str(), inputs.size(), params.density, n_workers);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "llama-merge: %s\n", error.what());
        return 1;
    }
}
#endif
