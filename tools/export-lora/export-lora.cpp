#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static bool g_verbose = false;

struct tensor_transformation {
    struct ggml_tensor * in;
    struct ggml_tensor * out;
    bool is_copy;
};

static std::string get_kv_str(struct gguf_context * ctx_gguf, const std::string & key) {
    int id = gguf_find_key(ctx_gguf, key.c_str());
    return id < 0 ? "" : std::string(gguf_get_val_str(ctx_gguf, id));
}

static float get_kv_f32(struct gguf_context * ctx_gguf, const std::string & key) {
    int id = gguf_find_key(ctx_gguf, key.c_str());
    return id < 0 ? 0.0f : gguf_get_val_f32(ctx_gguf, id);
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

// ------------------------------------------------------------------------
// Generic ggml_type <-> string lookup, built from the live GGML_TYPE list
// instead of a hand-maintained if/else chain. This means any type ggml
// knows about (including ones added later) is automatically selectable
// from the command line, as long as it passes is_valid_output_type().
// ------------------------------------------------------------------------

static bool ggml_type_from_name(const std::string & name, ggml_type & out_type) {
    std::string needle = name;
    std::transform(needle.begin(), needle.end(), needle.begin(), ::tolower);

    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type   t         = (ggml_type) i;
        const char *       type_name = ggml_type_name(t);
        if (type_name == nullptr) {
            continue; // removed / reserved enum slot
        }
        std::string haystack = type_name;
        std::transform(haystack.begin(), haystack.end(), haystack.begin(), ::tolower);
        if (haystack == needle) {
            out_type = t;
            return true;
        }
    }
    return false;
}

// Only types that can actually be produced from an F32 accumulator are
// valid merge output types: plain floats, or quantized types that expose
// a from_float converter.
static bool is_valid_output_type(ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        return true;
    }
    // ggml_type_traits (backend-agnostic) only exposes the reference
    // converter as from_float_ref; the SIMD-optimized from_float lives in
    // ggml_type_traits_cpu (ggml-cpu.h), which isn't what we need here —
    // ggml_quantize_chunk() dispatches internally per-type regardless.
    const auto * traits = ggml_get_type_traits(type);
    return traits != nullptr && traits->from_float_ref != nullptr;
}

static std::vector<std::string> list_supported_type_names() {
    std::vector<std::string> names;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type t = (ggml_type) i;
        if (!is_valid_output_type(t)) {
            continue;
        }
        const char * n = ggml_type_name(t);
        if (n == nullptr) {
            continue;
        }
        std::string lower = n;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        names.push_back(lower);
    }
    return names;
}

// Best-effort mapping to the legacy LLAMA_FTYPE metadata value. Not every
// ggml_type has a 1:1 llama_ftype counterpart; unmapped types fall back to
// LLAMA_FTYPE_MOSTLY_F16 with a warning (this only affects the informational
// general.file_type key, not the actual tensor data written to disk).
static uint32_t ggml_type_to_llama_ftype(ggml_type type) {
    static const std::map<ggml_type, uint32_t> k_map = {
        { GGML_TYPE_F32,   LLAMA_FTYPE_ALL_F32          },
        { GGML_TYPE_F16,   LLAMA_FTYPE_MOSTLY_F16       },
        { GGML_TYPE_BF16,  LLAMA_FTYPE_MOSTLY_BF16      },
        { GGML_TYPE_Q4_0,  LLAMA_FTYPE_MOSTLY_Q4_0      },
        { GGML_TYPE_Q4_1,  LLAMA_FTYPE_MOSTLY_Q4_1      },
        { GGML_TYPE_Q5_0,  LLAMA_FTYPE_MOSTLY_Q5_0      },
        { GGML_TYPE_Q5_1,  LLAMA_FTYPE_MOSTLY_Q5_1      },
        { GGML_TYPE_Q8_0,  LLAMA_FTYPE_MOSTLY_Q8_0      },
        { GGML_TYPE_Q2_K,  LLAMA_FTYPE_MOSTLY_Q2_K      },
        { GGML_TYPE_Q3_K,  LLAMA_FTYPE_MOSTLY_Q3_K_M    },
        { GGML_TYPE_Q4_K,  LLAMA_FTYPE_MOSTLY_Q4_K_M    },
        { GGML_TYPE_Q5_K,  LLAMA_FTYPE_MOSTLY_Q5_K_M    },
        { GGML_TYPE_Q6_K,  LLAMA_FTYPE_MOSTLY_Q6_K      },
        { GGML_TYPE_MXFP4, LLAMA_FTYPE_MOSTLY_MXFP4_MOE },
    };
    auto it = k_map.find(type);
    if (it != k_map.end()) {
        return it->second;
    }
    fprintf(stderr, "%s: warning: no direct LLAMA_FTYPE mapping for '%s', "
                     "general.file_type metadata will be approximate\n",
            __func__, ggml_type_name(type));
    return LLAMA_FTYPE_MOSTLY_F16;
}

static struct gguf_context * load_gguf(std::string & fname, struct ggml_context ** ctx_ggml) {
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ ctx_ggml,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(fname.c_str(), params);
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load input GGUF from " + fname);
    }
    return ctx_gguf;
}

struct file_input {
    struct ggml_context * ctx_meta = nullptr;
    struct gguf_context * ctx_gguf = nullptr;
    std::ifstream f_in;
    std::map<std::string, ggml_tensor *> tensors;
    float alpha;
    float scale;

    file_input(std::string & fname, float scale) : f_in(fname, std::ios::binary), scale(scale) {
        if (!f_in.is_open()) {
            throw std::runtime_error("failed to open input gguf from " + fname);
        }

        ctx_gguf = load_gguf(fname, &ctx_meta);
        alpha = get_kv_f32(ctx_gguf, "adapter.lora.alpha");
        printf("%s: loaded gguf from %s\n", __func__, fname.c_str());

        for (ggml_tensor * cur = ggml_get_first_tensor(ctx_meta); cur; cur = ggml_get_next_tensor(ctx_meta, cur)) {
            std::string name(cur->name);
            tensors[name] = cur;
            if (g_verbose) {
                printf("%s: %s\n", __func__, cur->name);
            }
        }
    }

    ggml_tensor * get_tensor(std::string name) {
        if (tensors.find(name) == tensors.end()) {
            return nullptr;
        }
        return tensors[name];
    }

    void read_tensor_data(std::string name, std::vector<uint8_t> & buf) {
        if (tensors.find(name) == tensors.end()) {
            throw std::runtime_error("cannot find tensor with name: " + name);
        }
        auto len = ggml_nbytes(tensors[name]);
        if (buf.size() < len) {
            buf.resize(len);
        }
        auto i_tensor_in = gguf_find_tensor(ctx_gguf, name.c_str());
        auto offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i_tensor_in);
        f_in.seekg(offset);
        f_in.read((char *) buf.data(), len);
    }

    ~file_input() {
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
    }
};

struct lora_merge_ctx {
    file_input base_model;
    std::vector<std::unique_ptr<file_input>> adapters;

    int n_threads;
    ggml_type out_type; // requested output tensor type (e.g. Q4_0, F16, F32...)
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    std::vector<uint8_t> read_buf;

    struct gguf_context * ctx_out;
    struct ggml_context * ctx_out_ggml;
    std::ofstream fout;

    lora_merge_ctx(
            std::string & base_fname,
            std::vector<common_adapter_lora_info> & lora_files,
            std::string & outfile,
            ggml_type target_out_type,
            int n_threads) :
            base_model(base_fname, 0), n_threads(n_threads), out_type(target_out_type), fout(outfile, std::ios::binary) {

        fout.exceptions(std::ofstream::failbit);

        if (gguf_find_key(base_model.ctx_gguf, LLM_KV_SPLIT_COUNT) >= 0) {
            throw std::runtime_error("split model is not yet supported");
        }

        for (auto & lora_inp : lora_files) {
            auto fname = lora_inp.path;
            auto scale = lora_inp.scale;
            std::unique_ptr<file_input> adapter(new file_input(fname, scale));
            check_metadata_lora(adapter.get());
            adapters.push_back(std::move(adapter));
        }

        ctx_out = gguf_init_empty();
        struct ggml_init_params params = {
            /*.mem_size   =*/ static_cast<size_t>(gguf_get_n_tensors(base_model.ctx_gguf) * ggml_tensor_overhead()),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx_out_ggml = ggml_init(params);
        backend = ggml_backend_cpu_init();
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }

    void check_metadata_lora(file_input * adapter) {
        auto general_type = get_kv_str(adapter->ctx_gguf, "general.type");
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }
        auto adapter_type = get_kv_str(adapter->ctx_gguf, "adapter.type");
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }
        auto general_arch_base = get_kv_str(base_model.ctx_gguf, "general.architecture");
        auto general_arch_lora = get_kv_str(adapter->ctx_gguf, "general.architecture");
        if (general_arch_base != general_arch_lora) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }
    }

    // F32 base tensors are kept as F32 in the output (matches upstream
    // behavior); everything else follows the user-requested output type.
    ggml_type get_out_tensor_type(struct ggml_tensor * t) {
        if (t->type == GGML_TYPE_F32) {
            return GGML_TYPE_F32;
        }
        return out_type;
    }

    void run_merge() {
        gguf_set_kv(ctx_out, base_model.ctx_gguf);
        gguf_set_val_u32(ctx_out, "general.file_type", ggml_type_to_llama_ftype(out_type));

        if (adapters.size() > 1) {
            for (size_t i = 1; i < adapters.size(); ++i) {
                if (adapters[0]->tensors.size() != adapters[i]->tensors.size()) {
                    throw std::runtime_error("Subset adapters merging not supported.");
                }
            }
        }

        std::vector<tensor_transformation> trans;
        for (auto & it : base_model.tensors) {
            bool t_a = true;
            bool t_b = true;
            for (auto & adapter : adapters) {
                t_a &= nullptr != adapter->get_tensor(it.first + ".lora_a");
                t_b &= nullptr != adapter->get_tensor(it.first + ".lora_b");
            }
            auto base_tensor = it.second;
            if (!t_a && !t_b) {
                struct ggml_tensor * cpy_tensor = ggml_dup_tensor(ctx_out_ggml, base_tensor);
                ggml_set_name(cpy_tensor, base_tensor->name);
                trans.push_back({cpy_tensor, cpy_tensor, true});
                gguf_add_tensor(ctx_out, cpy_tensor);
            } else if (t_a && t_b) {
                struct ggml_tensor * out_tensor = ggml_new_tensor(
                    ctx_out_ggml, get_out_tensor_type(base_tensor), GGML_MAX_DIMS, base_tensor->ne);
                ggml_set_name(out_tensor, base_tensor->name);
                trans.push_back({base_tensor, out_tensor, false});
                gguf_add_tensor(ctx_out, out_tensor);
            } else {
                throw std::runtime_error("tensor " + it.first + " missing lora_a or lora_b");
            }
        }

        {
            size_t meta_size = gguf_get_meta_size(ctx_out);
            zeros(fout, meta_size);
        }

        size_t n_merged = 0;
        for (auto & it : trans) {
            if (!it.is_copy) {
                merge_tensor(it.in, it.out);
                n_merged++;
            } else {
                copy_tensor(it.in);
            }
        }

        {
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
            gguf_get_meta_data(ctx_out, data.data());
            fout.seekp(0);
            fout.write((const char *) data.data(), data.size());
        }

        printf("%s : merged %zu tensors with lora adapters (output type: %s)\n",
               __func__, n_merged, ggml_type_name(out_type));
    }

    void copy_tensor(struct ggml_tensor * base) {
        printf("%s :  %s [%s]\n", __func__, base->name, ggml_ne_string(base).c_str());
        size_t len = ggml_nbytes(base);
        base_model.read_tensor_data(base->name, read_buf);
        fout.write((char *) read_buf.data(), len);
        zeros(fout, GGML_PAD(len, GGUF_DEFAULT_ALIGNMENT) - len);
    }

    void merge_tensor(struct ggml_tensor * base, struct ggml_tensor * out) {
        std::string name_base(base->name);
        std::string name_lora_a = name_base + ".lora_a";
        std::string name_lora_b = name_base + ".lora_b";

        printf("%s : %s [%s]\n", __func__, base->name, ggml_ne_string(base).c_str());

        std::vector<struct ggml_tensor *> inp_a(adapters.size());
        std::vector<struct ggml_tensor *> inp_b(adapters.size());
        struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * (2 + adapters.size() * 2),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * inp_base = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, base->ne);
        for (size_t i = 0; i < adapters.size(); ++i) {
            auto t_a = adapters[i]->get_tensor(name_lora_a);
            auto t_b = adapters[i]->get_tensor(name_lora_b);
            inp_a[i] = ggml_dup_tensor(ctx, t_a);
            inp_b[i] = ggml_dup_tensor(ctx, t_b);
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

        base_model.read_tensor_data(name_base, read_buf);
        if (base->type != GGML_TYPE_F32) {
            auto nels = ggml_nelements(inp_base);
            const auto * base_traits = ggml_get_type_traits(base->type);
            std::vector<uint8_t> dequant_buf(nels * sizeof(float));
            base_traits->to_float(read_buf.data(), (float *) dequant_buf.data(), nels);
            ggml_backend_tensor_set(inp_base, dequant_buf.data(), 0, dequant_buf.size());
        } else {
            ggml_backend_tensor_set(inp_base, read_buf.data(), 0, ggml_nbytes(inp_base));
        }

        for (size_t i = 0; i < adapters.size(); ++i) {
            adapters[i]->read_tensor_data(name_lora_a, read_buf);
            ggml_backend_tensor_set(inp_a[i], read_buf.data(), 0, ggml_nbytes(inp_a[i]));
            adapters[i]->read_tensor_data(name_lora_b, read_buf);
            ggml_backend_tensor_set(inp_b[i], read_buf.data(), 0, ggml_nbytes(inp_b[i]));
        }

        struct ggml_cgraph * gf;
        {
            static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
            static std::vector<uint8_t> buf(buf_size);
            struct ggml_init_params params0 = {
                /*.mem_size   =*/ buf_size,
                /*.mem_buffer =*/ buf.data(),
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * ctx0 = ggml_init(params0);
            gf = ggml_new_graph(ctx0);

            // The accumulation always happens in F32; ggml has no
            // "quantized destination" op, so the graph itself only ever
            // produces F32 or plain-float output. Quantized output types
            // are handled after the graph is computed, see below.
            struct ggml_tensor * cur = inp_base;
            for (size_t i = 0; i < adapters.size(); ++i) {
                struct ggml_tensor * delta;
                bool is_tok_embd = string_starts_with(name_base, "token_embd");
                if (is_tok_embd) {
                    delta = ggml_mul_mat(ctx0,
                        ggml_cast(ctx0, inp_b[i], GGML_TYPE_F32),
                        ggml_cast(ctx0, inp_a[i], GGML_TYPE_F32));
                } else {
                    delta = ggml_mul_mat(ctx0,
                        ggml_cont(ctx0, ggml_transpose(ctx0, ggml_cast(ctx0, inp_a[i], GGML_TYPE_F32))),
                        ggml_cast(ctx0, inp_b[i], GGML_TYPE_F32));
                }
                const float alpha = adapters[i]->alpha;
                const float rank  = (float) inp_b[i]->ne[0];
                const float scale = alpha ? adapters[i]->scale * alpha / rank : adapters[i]->scale;
                delta = ggml_scale(ctx0, delta, scale);
                cur = ggml_add(ctx0, delta, cur);
            }

            // Only cast in-graph for plain float destination types.
            // Quantized destinations are produced explicitly afterwards.
            if (!ggml_is_quantized(out->type)) {
                cur = ggml_cast(ctx0, cur, out->type);
            }

            ggml_build_forward_expand(gf, cur);
            ggml_free(ctx0);
        }

        {
            ggml_gallocr_alloc_graph(allocr, gf);
            ggml_backend_cpu_set_n_threads(backend, n_threads);
            ggml_backend_graph_compute(backend, gf);
        }

        {
            auto * result = ggml_graph_node(gf, -1);
            size_t nels = ggml_nelements(result);

            if (ggml_is_quantized(out->type)) {
                printf("%s :   + quantizing result on-the-fly to %s\n", __func__, ggml_type_name(out->type));

                // 1. pull the F32 result off the backend
                std::vector<float> f32_buf(nels);
                ggml_backend_tensor_get(result, f32_buf.data(), 0, nels * sizeof(float));

                // 2. quantize row-by-row using the *output* tensor's shape,
                //    not just the raw element count. ggml_quantize_chunk
                //    quantizes block-aligned rows, so it needs to know how
                //    many elements make up one row (out->ne[0]) and how
                //    many such rows there are.
                const int64_t n_per_row = out->ne[0];
                const int64_t nrows     = nels / n_per_row;
                GGML_ASSERT(nrows * n_per_row == (int64_t) nels);

                size_t out_len = ggml_row_size(out->type, n_per_row) * nrows;
                std::vector<uint8_t> q_buf(out_len);

                ggml_quantize_chunk(out->type, f32_buf.data(), q_buf.data(), 0, nrows, n_per_row, nullptr);

                // 3. write the quantized bytes to disk
                fout.write((char *) q_buf.data(), out_len);
                zeros(fout, GGML_PAD(out_len, GGUF_DEFAULT_ALIGNMENT) - out_len);
            } else {
                size_t len = ggml_nbytes(result);
                if (read_buf.size() < len) {
                    read_buf.resize(len);
                }
                ggml_backend_tensor_get(result, read_buf.data(), 0, len);
                fout.write((char *) read_buf.data(), len);
                zeros(fout, GGML_PAD(len, GGUF_DEFAULT_ALIGNMENT) - len);
            }
        }

        ggml_free(ctx);
        ggml_backend_buffer_free(buffer);
    }

    ~lora_merge_ctx() {
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        gguf_free(ctx_out);
        ggml_free(ctx_out_ggml);
    }
};

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n  %s -m base-model.gguf --lora lora-file.gguf -o merged-model.gguf --type q4_0\n", argv[0]);
    printf("\n--type accepts any ggml tensor type that can be produced from F32, e.g.:\n  ");
    auto names = list_supported_type_names();
    for (size_t i = 0; i < names.size(); ++i) {
        printf("%s%s", names[i].c_str(), (i + 1 < names.size()) ? ", " : "\n");
    }
    printf("\n");
}

// Pulls "--type <value>" out of argv (if present) and returns argv/argc
// with it stripped, so downstream common_params_parse() doesn't choke on
// an option it doesn't know about. Returns the requested type via out_type.
static std::vector<std::string> extract_type_arg(int argc, char ** argv, ggml_type & out_type) {
    std::vector<std::string> filtered;
    filtered.reserve(argc);

    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            std::string type_str = argv[i + 1];
            if (!ggml_type_from_name(type_str, out_type)) {
                throw std::runtime_error("unknown --type '" + type_str + "', see --help for the supported list");
            }
            if (!is_valid_output_type(out_type)) {
                throw std::runtime_error("--type '" + type_str + "' cannot be produced from F32 data "
                                          "(no from_float converter), see --help for the supported list");
            }
            i++; // skip the value too
            continue;
        }
        filtered.push_back(argv[i]);
    }
    return filtered;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.out_file = "ggml-lora-merged.gguf";
    ggml_type target_out_type = GGML_TYPE_F16; // default kept for backward compatibility

    std::vector<std::string> filtered_args;
    try {
        filtered_args = extract_type_arg(argc, argv, target_out_type);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s\n", err.what());
        print_usage(argc, argv);
        return 1;
    }

    std::vector<char *> filtered_argv;
    filtered_argv.reserve(filtered_args.size());
    for (auto & s : filtered_args) {
        filtered_argv.push_back(s.data());
    }
    int filtered_argc = (int) filtered_argv.size();

    common_init();

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_EXPORT_LORA, print_usage)) {
        return 1;
    }

    g_verbose = (params.verbosity > 1);
    try {
        lora_merge_ctx ctx(params.model.path, params.lora_adapters, params.out_file, target_out_type, params.cpuparams.n_threads);
        ctx.run_merge();
    } catch (const std::exception & err) {
        fprintf(stderr, "%s\n", err.what());
        exit(EXIT_FAILURE);
    }

    printf("done, output file is %s (type: %s)\n", params.out_file.c_str(), ggml_type_name(target_out_type));

    return 0;
}
