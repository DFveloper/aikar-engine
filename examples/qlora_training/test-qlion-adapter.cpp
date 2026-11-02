#include "qlion-adapter.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void rejected(const std::function<void()> & action) {
    bool failed = false;
    try { action(); } catch (const std::exception &) { failed = true; }
    check(failed, "invalid input was accepted");
}

struct fixture {
    ggml_context_ptr ctx { ggml_init({ 16 * ggml_tensor_overhead(), nullptr, true }) };
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * a;
    ggml_tensor * b;
    std::string root;
    std::string base;
    std::vector<ggml_tensor *> targets;
    std::vector<unsigned char> original_a, original_b;

    explicit fixture(const std::string & root, ggml_type type = GGML_TYPE_Q4_0) : root(root), base(root + "/base.gguf") {
        a = ggml_new_tensor_3d(ctx.get(), type, ggml_blck_size(type), 4, 3);
        b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 32, 4);
        ggml_set_name(a, "blk.0.ffn_up_exps.weight");
        ggml_set_name(b, "blk.1.attn_k.weight");
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        check(bool(buffer), "buffer allocation failed");
        original_a.assign(ggml_nbytes(a), 0);
        original_b.assign(ggml_nbytes(b), 0);
        reset();
        targets = {a, b};
        write_base(base);
    }

    std::string path(const std::string & name) const { return root + "/" + name + ".gguf"; }

    void reset() {
        ggml_backend_tensor_set(a, original_a.data(), 0, original_a.size());
        ggml_backend_tensor_set(b, original_b.data(), 0, original_b.size());
    }

    void write_base(const std::string & path, const std::string & label = "base") {
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", "gemma4");
        gguf_set_val_str(meta.get(), "general.name", label.c_str());
        gguf_add_tensor(meta.get(), a);
        gguf_add_tensor(meta.get(), b);
        check(gguf_write_to_file(meta.get(), path.c_str(), false), "fixture write failed");
    }

    void change(ggml_tensor * tensor, size_t offset, unsigned char value = 0x11) {
        ggml_backend_tensor_set(tensor, &value, offset, 1);
    }

    void compare(const std::string & path) {
        ggml_context * loaded = nullptr;
        gguf_context_ptr meta(gguf_init_from_file(path.c_str(), { false, &loaded }));
        ggml_context_ptr tensors(loaded);
        check(bool(meta), "merged GGUF cannot be read");
        for (auto * tensor : targets) {
            const auto * actual = ggml_get_tensor(tensors.get(), tensor->name);
            std::vector<unsigned char> expected(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, expected.data(), 0, expected.size());
            check(actual && actual->type == tensor->type && ggml_are_same_shape(actual, tensor) &&
                    memcmp(actual->data, expected.data(), expected.size()) == 0, "merged tensor bytes differ");
        }
    }
};

static void rewrite(const std::string & input, const std::string & output, const std::function<void(gguf_context *)> & edit) {
    ggml_context * loaded = nullptr;
    gguf_context_ptr meta(gguf_init_from_file(input.c_str(), { false, &loaded }));
    ggml_context_ptr ctx(loaded);
    check(bool(meta), "rewrite input failed");
    gguf_context_ptr output_meta(gguf_init_empty());
    gguf_set_kv(output_meta.get(), meta.get());
    for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
        gguf_add_tensor(output_meta.get(), ggml_get_tensor(ctx.get(), gguf_get_tensor_name(meta.get(), i)));
    }
    edit(output_meta.get());
    check(gguf_write_to_file(output_meta.get(), output.c_str(), false), "rewrite failed");
}

static int64_t tensor_count(const std::string & path) {
    gguf_context_ptr meta(gguf_init_from_file(path.c_str(), { true, nullptr }));
    check(bool(meta), "invalid saved adapter");
    return gguf_get_n_tensors(meta.get());
}

static void test_adapter(fixture & f) {
    const auto snapshot = common_qlion_capture(f.base, f.targets);
    common_qlion_save(snapshot, f.targets, f.path("unchanged"));
    check(tensor_count(f.path("unchanged")) == 0, "unchanged tensors saved");
    f.change(f.a, 3);
    const auto adapter = f.path("full");
    const auto merged = f.path("merged");
    common_qlion_save(snapshot, f.targets, adapter);
    check(tensor_count(adapter) == 1, "unselected/unchanged tensor saved");
    common_qlion_merge(f.base, {adapter}, merged);
    f.compare(merged);
    const auto expected_a = std::vector<unsigned char>(static_cast<unsigned char *>(f.a->data), static_cast<unsigned char *>(f.a->data) + ggml_nbytes(f.a));
    f.reset();
    common_qlion_apply(f.base, {adapter}, {f.a, f.b});
    check(memcmp(f.a->data, expected_a.data(), expected_a.size()) == 0, "runtime replacement differs");
    check(memcmp(f.b->data, f.original_b.data(), f.original_b.size()) == 0, "runtime changed unselected tensor");
    rejected([&] { common_qlion_merge(f.base, {adapter, adapter}, merged); });
    f.compare(merged);
    rejected([&] { common_qlion_merge(f.base, {adapter}, f.base); });
    rejected([&] { common_qlion_merge(f.base, {adapter}, adapter); });
    std::filesystem::create_hard_link(f.base, f.path("hardlink"));
    rejected([&] { common_qlion_merge(f.base, {adapter}, f.path("hardlink")); });
    rejected([&] { common_qlion_save(snapshot, f.targets, f.path("missing/adapter")); });
    rejected([&] { common_qlion_apply(f.base, {adapter}, {f.b}); });
    std::filesystem::create_directory(f.path("occupied"));
    std::ofstream(f.path("occupied") + "/keep.txt") << "keep";
    rejected([&] { common_qlion_save(snapshot, f.targets, f.path("occupied")); });
    rejected([&] { common_qlion_merge(f.base, {adapter}, f.path("occupied")); });
    check(std::filesystem::file_size(f.path("occupied") + "/keep.txt") == 4, "failed publication damaged destination");

    f.reset();
    f.write_base(f.path("compatible"), "changed metadata");
    rejected([&] { common_qlion_merge(f.path("compatible"), {adapter}, merged); });
    common_qlion_merge(f.path("compatible"), {adapter}, merged, true);
    f.change(f.a, 3);
    f.compare(merged);
    f.reset();
    f.change(f.a, 5);
    f.write_base(f.path("wrong-base"));
    rejected([&] { common_qlion_merge(f.path("wrong-base"), {adapter}, merged, true); });
    // Even an exact file hash must not bypass the per-tensor hash.
    rewrite(adapter, f.path("bad-hash"), [](gguf_context * meta) {
        const char * hashes[] = { "sha256:0000000000000000000000000000000000000000000000000000000000000000" };
        gguf_set_arr_str(meta, "qlion.adapter.tensor_base_sha256", hashes, 1);
    });
    rejected([&] { common_qlion_merge(f.base, {f.path("bad-hash")}, merged); });
    rewrite(adapter, f.path("bad-payload"), [](gguf_context * meta) {
        const char * hashes[] = { "sha256:0000000000000000000000000000000000000000000000000000000000000000" };
        gguf_set_arr_str(meta, "qlion.adapter.tensor_payload_sha256", hashes, 1);
    });
    rejected([&] { common_qlion_merge(f.base, {f.path("bad-payload")}, merged); });
    rewrite(adapter, f.path("bad-version"), [](gguf_context * meta) { gguf_set_val_u32(meta, "qlion.adapter.version", 99); });
    rejected([&] { common_qlion_merge(f.base, {f.path("bad-version")}, merged); });
    rewrite(adapter, f.path("bad-version-type"), [](gguf_context * meta) { gguf_set_val_str(meta, "qlion.adapter.version", "1"); });
    rejected([&] { common_qlion_merge(f.base, {f.path("bad-version-type")}, merged); });
    rewrite(adapter, f.path("partial-v1"), [](gguf_context * meta) {
        const uint64_t rows[] = { 0 };
        gguf_set_arr_data(meta, "qlion.adapter.tensor.0.rows", GGUF_TYPE_UINT64, rows, 1);
    });
    rejected([&] { common_qlion_merge(f.base, {f.path("partial-v1")}, merged); });
    std::filesystem::copy_file(adapter, f.path("truncated"));
    std::filesystem::resize_file(f.path("truncated"), std::filesystem::file_size(adapter) - 32);
    rejected([&] { common_qlion_merge(f.base, {f.path("truncated")}, merged); });
    f.reset();
    f.change(f.b, 4);
    common_qlion_save(snapshot, f.targets, f.path("disjoint"));
    common_qlion_merge(f.base, {adapter, f.path("disjoint")}, merged);
    f.change(f.a, 3);
    f.compare(merged);
    f.reset();
}

static void test_units(fixture & f, const std::string & units) {
    const auto snapshot = common_qlion_capture(f.base, f.targets);
    f.change(f.a, 2);
    f.change(f.a, 5 * ggml_row_size(f.a->type, f.a->ne[0]) + 2);
    common_qlion_save(snapshot, f.targets, f.path(units), units);
    common_qlion_merge(f.base, {f.path(units)}, f.path("merged-units"));
    f.compare(f.path("merged-units"));
    rejected([&] { common_qlion_merge(f.base, {f.path(units), f.path("full")}, f.path("merged-units")); });
    f.reset();
    f.change(f.a, 10 * ggml_row_size(f.a->type, f.a->ne[0]) + 2);
    common_qlion_save(snapshot, f.targets, f.path(units + "-disjoint"), units);
    common_qlion_merge(f.base, {f.path(units + "-disjoint"), f.path(units)}, f.path("merged-units"));
    f.change(f.a, 2);
    f.change(f.a, 5 * ggml_row_size(f.a->type, f.a->ne[0]) + 2);
    f.compare(f.path("merged-units"));
    f.reset();
    common_qlion_apply(f.base, {f.path(units + "-disjoint"), f.path(units)}, f.targets);
    f.compare(f.path("merged-units"));
    rewrite(f.path(units), f.path("bad-indices"), [&](gguf_context * meta) {
        const uint64_t indices[] = { UINT64_MAX, UINT64_MAX };
        gguf_set_arr_data(meta, ("qlion.adapter.tensor.0." + units).c_str(), GGUF_TYPE_UINT64, indices, 2);
    });
    rejected([&] { common_qlion_merge(f.base, {f.path("bad-indices")}, f.path("merged-units")); });
    rewrite(f.path(units), f.path("bad-shape"), [](gguf_context * meta) {
        const uint64_t shape[] = { UINT64_MAX, 4, 3, 1 };
        gguf_set_arr_data(meta, "qlion.adapter.tensor.0.shape", GGUF_TYPE_UINT64, shape, 4);
    });
    rejected([&] { common_qlion_merge(f.base, {f.path("bad-shape")}, f.path("merged-units")); });
    f.reset();
}

static void test_malformed_directory(fixture & f) {
    const auto path = f.path("duplicate");
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture", "gemma4");
    ggml_tensor a = *f.a, b = *f.b;
    ggml_set_name(&a, "duplicate0.weight");
    ggml_set_name(&b, "duplicate1.weight");
    gguf_add_tensor(meta.get(), &a);
    gguf_add_tensor(meta.get(), &b);
    check(gguf_write_to_file(meta.get(), path.c_str(), false), "duplicate fixture write failed");
    std::ifstream in(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto position = bytes.find("duplicate1.weight");
    check(position != std::string::npos, "tensor directory entry not found");
    bytes[position + 9] = '0';
    in.close();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), bytes.size());
    out.close();
    rejected([&] { common_qlion_capture(path, {}); });
    rewrite(f.base, f.path("split"), [](gguf_context * ctx) { gguf_set_val_u16(ctx, "split.count", 2); });
    rejected([&] { common_qlion_capture(f.path("split"), f.targets); });
    const auto snapshot = common_qlion_capture(f.base, f.targets);
    // A changed base must invalidate a captured training snapshot.
    f.change(f.a, 4);
    f.write_base(f.base);
    rejected([&] { common_qlion_save(snapshot, f.targets, f.path("stale-base")); });
    f.reset();
    f.write_base(f.base);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: test-qlion-adapter SCRATCH_DIRECTORY\n");
        return 1;
    }
    try {
        check(std::filesystem::create_directory(argv[1]), "scratch directory already exists");
        fixture f(argv[1]);
        test_adapter(f);
        test_units(f, "rows");
        test_units(f, "experts");
        test_malformed_directory(f);
        for (auto type : { GGML_TYPE_MXFP4, GGML_TYPE_Q2_0 }) {
            const auto directory = std::string(argv[1]) + "/" + ggml_type_name(type);
            std::filesystem::create_directory(directory);
            fixture other(directory, type);
            test_adapter(other);
            test_units(other, "rows");
            test_units(other, "experts");
        }
        std::filesystem::remove_all(argv[1]);
        puts("QLion adapter tests passed");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
