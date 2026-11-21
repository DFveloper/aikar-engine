#include "llama-model-loader.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <filesystem>
#include <random>
#include <stdexcept>

#ifdef __linux__
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void write_model(const std::string & path) {
    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture", "llama");
    ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, false}));
    for (int i = 0; i < 4; ++i) {
        auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16384);
        ggml_format_name(tensor, "weight_%d", i);
        for (int j = 0; j < tensor->ne[0]; ++j) {
            static_cast<float *>(tensor->data)[j] = i*16384 + j;
        }
        gguf_add_tensor(metadata.get(), tensor);
    }
    check(gguf_write_to_file(metadata.get(), path.c_str(), false), "cannot write test model");
}

static void test_load(const std::string & path, bool mapped, bool copy_all, bool validate) {
    std::vector<std::string> splits;
    llama_model_loader loader(nullptr, nullptr, nullptr, path, splits, nullptr,
            mapped ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE, validate, false, false, nullptr, nullptr);
    ggml_context_ptr ctx_copy(ggml_init({1024*1024, nullptr, true}));
    ggml_context_ptr ctx_map(ggml_init({1024*1024, nullptr, true}));
    for (int i = 0; i < 4; ++i) {
        const std::string name = "weight_" + std::to_string(i);
        auto * ctx = mapped && !copy_all && i % 2 == 0 ? ctx_map.get() : ctx_copy.get();
        auto * tensor = ggml_dup_tensor(ctx, loader.require_weight(name.c_str()).tensor);
        ggml_set_name(tensor, name.c_str());
    }
    loader.init_mappings(true, nullptr);
    ggml_backend_buffer_ptr copied(ggml_backend_alloc_ctx_tensors_from_buft(ctx_copy.get(), ggml_backend_cpu_buffer_type()));
    check(copied != nullptr, "cannot allocate copied tensors");
    std::vector<ggml_backend_buffer_t> buffers = {copied.get()};
    check(loader.load_all_data(ctx_copy.get(), buffers, nullptr, nullptr, nullptr), "cannot load copied tensors");

    ggml_backend_buffer_ptr mapping;
    if (mapped && !copy_all) {
        mapping.reset(ggml_backend_cpu_buffer_from_ptr(loader.mappings[0]->addr(), loader.mappings[0]->size()));
        check(mapping != nullptr, "cannot wrap mapping");
        buffers = {mapping.get()};
        check(loader.load_all_data(ctx_map.get(), buffers, nullptr, nullptr, nullptr), "cannot load mapped tensors");
    }

    for (int i = 0; i < 4; ++i) {
        const std::string name = "weight_" + std::to_string(i);
        auto * ctx = mapped && !copy_all && i % 2 == 0 ? ctx_map.get() : ctx_copy.get();
        auto * tensor = ggml_get_tensor(ctx, name.c_str());
        for (int j = 0; j < tensor->ne[0]; ++j) {
            check(static_cast<const float *>(tensor->data)[j] == i*16384 + j, "loaded tensor data changed");
        }
    }

#ifdef __linux__
    if (mapped) {
        const size_t page_size = sysconf(_SC_PAGESIZE);
        const auto & weight = loader.require_weight("weight_1");
        const size_t offset = (weight.offs + page_size - 1)/page_size*page_size;
        unsigned char resident;
        errno = 0;
        if (offset + page_size <= weight.offs + ggml_nbytes(weight.tensor)) {
            check(mincore(static_cast<char *>(loader.mappings[0]->addr()) + offset, page_size, &resident) == -1 && errno == ENOMEM,
                    "copied weight is still mapped");
        }
    }
#endif
}

int main() {
    const auto path = std::filesystem::temp_directory_path() /
            ("llama-loader-test-" + std::to_string(std::random_device{}()) + ".gguf");
    write_model(path.string());
    for (bool validate : {false, true}) {
        test_load(path.string(), false, true, validate);
        if (llama_mmap::SUPPORTED) {
            test_load(path.string(), true, true, validate);
            test_load(path.string(), true, false, validate);
        }
    }
    std::filesystem::remove(path);
}
