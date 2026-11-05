#include "logit-distillation.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-hparams.h"
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <filesystem>
#include <chrono>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static double reference(const std::vector<float> & logits, const std::vector<int32_t> & ids,
        const std::vector<float> & logp, const std::vector<float> & weights, float temperature) {
    constexpr int nv = 8, nr = 2, k = 3;
    double loss = 0;
    for (int r = 0; r < nr; ++r) {
        double total = 0;
        const double maximum = *std::max_element(logits.begin() + r*nv, logits.begin() + (r+1)*nv);
        for (int j = 0; j < nv; ++j) total += std::exp((logits[r*nv+j] - maximum) / temperature);
        bool selected[nv] = {};
        for (int j = 0; j < k; ++j) {
            selected[ids[r*k+j]] = true;
            const double student_logp = (logits[r*nv+ids[r*k+j]] - maximum)/temperature - std::log(total);
            loss += weights[r]*std::exp(double(logp[r*(k+1)+j]))*(logp[r*(k+1)+j] - student_logp);
        }
        double tail_max = -INFINITY, tail = 0;
        for (int j = 0; j < nv; ++j) if (!selected[j]) tail_max = std::max(tail_max, double(logits[r*nv+j]));
        for (int j = 0; j < nv; ++j) if (!selected[j]) tail += std::exp((logits[r*nv+j] - tail_max)/temperature);
        const double tail_logp = (tail_max - maximum)/temperature + std::log(tail) - std::log(total);
        loss += weights[r]*std::exp(double(logp[r*(k+1)+k]))*(logp[r*(k+1)+k] - tail_logp);
    }
    return loss*temperature*temperature/nr;
}

static void loss_and_gradient(float temperature, bool masked, bool extreme = false, bool reference_mode = false) {
    ggml_init_params params = {16*1024*1024, nullptr, false};
    auto * ctx = ggml_init(params);
    std::vector<float> values = {1.3f, -.2f, .5f, -1.f, .8f, -.8f, .1f, -.4f,
        -.1f, .4f, -.3f, 1.1f, .2f, -.5f, .7f, -.9f};
    if (extreme) values[0] = 1000.f;
    const std::vector<int32_t> indices = {4, 0, 2, 6, 3, 1};
    const std::vector<float> weights = {1.5f, masked ? 0.f : .5f};
    std::vector<float> targets = {.3f, .25f, .2f, .25f, .4f, .3f, .1f, .2f};
    for (auto & p : targets) p = std::log(p);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 2);
    auto * logp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto * row_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    std::memcpy(logits->data, values.data(), values.size()*4);
    std::memcpy(ids->data, indices.data(), indices.size()*4);
    std::memcpy(logp->data, targets.data(), targets.size()*4);
    std::memcpy(row_weights->data, weights.data(), weights.size()*4);
    ggml_set_param(logits);
    auto loss_function = reference_mode ? ggml_opt_distillation_loss_reference : ggml_opt_distillation_loss;
    auto * loss = loss_function(ctx, logits, ids, logp, row_weights, temperature);
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_graph_reset(graph);
    check(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS, "loss graph failed");
    const double expected = reference(values, indices, targets, weights, temperature);
    check(std::abs(*static_cast<float *>(loss->data) - expected) < 2e-5 + 1e-6*std::abs(expected), "KL differs from reference");
    const auto * gradient = static_cast<const float *>(ggml_graph_get_grad(graph, logits)->data);
    for (size_t i = 0; i < values.size(); ++i) {
        const float original = values[i];
        const float eps = 0.001f;
        values[i] = original + eps;
        const double upper = values[i];
        const double plus = reference(values, indices, targets, weights, temperature);
        values[i] = original - eps;
        const double lower = values[i];
        const double minus = reference(values, indices, targets, weights, temperature);
        values[i] = original;
        check(std::abs(gradient[i] - (plus-minus)/(upper-lower)) < 2e-4, "KL gradient differs from finite difference");
    }
    ggml_free(ctx);
}

static void cache_checks() {
    logit_distillation::header h;
    h.context = 2; h.vocab = 8; h.top_k = 3; h.windows = 1; h.train_windows = 1;
    const float logits[] = {4, 3, 2, -20, -21, -22, -23, -24};
    std::ostringstream out(std::ios::binary);
    h.save(out);
    const double tail = logit_distillation::encode_row(out, logits, h);
    check(tail < 1e-8 && tail > 0, "export clipped or discarded the teacher tail");
    const std::string content = out.str();
    std::ostringstream serial(std::ios::binary), parallel(std::ios::binary);
    std::vector<const float *> batch(5, logits);
    for (const auto * row : batch) logit_distillation::encode_row(serial, row, h);
    logit_distillation::encode_rows(parallel, batch, h, 3);
    check(serial.str() == parallel.str(), "parallel extraction changed record order or values");
    const auto path = std::filesystem::temp_directory_path() /
            ("qlion-distillation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".dat");
    {
        std::ofstream file(path, std::ios::binary);
        file.write(content.data(), content.size());
    }
    {
        // A torn first window must be removed before extraction resumes.
        logit_distillation::header resume_header = h;
        const int32_t tokens[2] = {0, 1}, labels[2] = {1, -1};
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        resume_header.save(file);
        const uint64_t header_size = file.tellp();
        file.write(reinterpret_cast<const char *>(tokens), sizeof(tokens));
        file.close();
        uint64_t rows = 99;
        check(logit_distillation::resume_cache(path.string(), resume_header, tokens, labels, rows) == 0 && rows == 0 &&
                std::filesystem::file_size(path) == header_size, "torn window was not truncated");
        file.open(path, std::ios::binary | std::ios::app);
        file.write(reinterpret_cast<const char *>(tokens), sizeof(tokens));
        file.write(reinterpret_cast<const char *>(labels), sizeof(labels));
        logit_distillation::encode_row(file, logits, h);
        file.close();
        check(logit_distillation::resume_cache(path.string(), resume_header, tokens, labels, rows) == 1 && rows == 1,
                "completed window was not retained");
        std::ofstream restore(path, std::ios::binary | std::ios::trunc);
        restore.write(content.data(), content.size());
    }
    {
        logit_distillation::cache loaded(path.string());
        const uint64_t offset = loaded.in.tellg();
        int32_t ids[3];
        float probabilities[4];
        loaded.row(offset, ids, probabilities);
        check(ids[0] == 0 && ids[1] == 1 && ids[2] == 2, "top-k IDs differ after reload");
        double sum = 0;
        for (float p : probabilities) sum += std::exp(double(p));
        check(std::abs(sum - 1) < 1e-6, "FP16 teacher distribution was not renormalized");
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(offset + 4);
        file.write(reinterpret_cast<const char *>(ids), 4);
        file.close();
    }
    bool duplicate_rejected = false;
    try {
        logit_distillation::cache loaded(path.string());
        int32_t ids[3];
        float probabilities[4];
        loaded.row(loaded.in.tellg(), ids, probabilities);
    } catch (const std::exception &) { duplicate_rejected = true; }
    std::filesystem::remove(path);
    check(duplicate_rejected, "duplicate cache IDs accepted");
    std::istringstream in(content, std::ios::binary);
    logit_distillation::header restored;
    restored.load(in);
    check(restored.top_k == 3 && restored.temperature == 1, "header roundtrip failed");
    bool rejected = false;
    try {
        std::istringstream broken(content.substr(0, 20), std::ios::binary);
        restored.load(broken);
    } catch (const std::exception &) { rejected = true; }
    check(rejected, "truncated header accepted");
    rejected = false;
    try {
        std::istringstream broken("badmagic", std::ios::binary);
        restored.load(broken);
    } catch (const std::exception &) { rejected = true; }
    check(rejected, "invalid cache version accepted");
}

static void benchmark(bool reference_mode) {
    constexpr int nv = 262144, nr = 16, k = 512;
    ggml_init_params params = {size_t(1024)*1024*1024, nullptr, false};
    auto * ctx = ggml_init(params);
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nv, nr);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, nr);
    auto * logp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k+1, nr);
    auto * weights = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nr);
    for (int r = 0; r < nr; ++r) {
        static_cast<float *>(weights->data)[r] = 1;
        for (int i = 0; i < nv; ++i) static_cast<float *>(logits->data)[r*nv+i] = std::sin(float(i%109)*.07f);
        for (int i = 0; i < k; ++i) {
            static_cast<int32_t *>(ids->data)[r*k+i] = (i*487+r*13)%nv;
            static_cast<float *>(logp->data)[r*(k+1)+i] = std::log(.5f/k);
        }
        static_cast<float *>(logp->data)[r*(k+1)+k] = std::log(.5f);
    }
    ggml_set_param(logits);
    auto fn = reference_mode ? ggml_opt_distillation_loss_reference : ggml_opt_distillation_loss;
    auto * loss = fn(ctx, logits, ids, logp, weights, 1);
    ggml_set_loss(loss);
    auto * graph = ggml_new_graph_custom(ctx, 256, true);
    ggml_build_forward_expand(graph, loss);
    ggml_build_backward_expand(ctx, graph, nullptr);
    ggml_graph_reset(graph);
    check(ggml_graph_compute_with_ctx(ctx, graph, 8) == GGML_STATUS_SUCCESS, "benchmark warmup failed");
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) check(ggml_graph_compute_with_ctx(ctx, graph, 8) == GGML_STATUS_SUCCESS, "benchmark failed");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()/3;
    std::printf("distill_bench: implementation=%s vocab=%d rows=%d k=%d threads=8 ms=%.3f tensor_bytes=%zu loss=%.8g\n",
            reference_mode ? "reference" : "fused", nv, nr, k, ms, ggml_used_mem(ctx), *static_cast<float *>(loss->data));
    ggml_free(ctx);
}

static void frozen_q41_moe_gradient() {
    auto * ctx = ggml_init({1024*1024, nullptr, false});
    auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, 96, 3, 2);
    auto * grad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 2);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
    std::vector<float> original(96*3*2), decoded(original.size());
    for (size_t index = 0; index < original.size(); ++index) original[index] = std::sin(float(index)*0.1f);
    ggml_quantize_chunk(GGML_TYPE_Q4_1, original.data(), weight->data, 0, 6, 96, nullptr);
    ggml_get_type_traits(GGML_TYPE_Q4_1)->to_float(weight->data, decoded.data(), decoded.size());
    const int32_t routes[] = {0, 1, 1, 0};
    std::memcpy(ids->data, routes, sizeof(routes));
    auto * gradient = static_cast<float *>(grad->data);
    for (int index = 0; index < 12; ++index) gradient[index] = float(index - 5)*0.2f;
    auto * output = ggml_mul_mat_id_back(ctx, weight, grad, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    for (int threads : {1, 4}) {
        check(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "Q4_1 MoE backward failed");
        for (int route = 0; route < 4; ++route) for (int column = 0; column < 96; ++column) {
            double expected = 0;
            for (int row = 0; row < 3; ++row) expected += double(decoded[(routes[route]*3 + row)*96 + column])*gradient[route*3 + row];
            check(std::abs(static_cast<float *>(output->data)[route*96 + column] - expected) < 1e-5,
                    "Q4_1 MoE gradient differs from dequantized matrix reference");
        }
    }
    ggml_free(ctx);
}

static ggml_backend_meta_split_state test_split_state(const ggml_tensor * tensor, void * userdata) {
    const auto * axes = static_cast<const int *>(userdata);
    int axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (std::strcmp(tensor->name, "left") == 0) axis = axes[0];
    if (std::strcmp(tensor->name, "right") == 0) axis = axes[1];
    if (std::strcmp(tensor->name, "q") == 0 || std::strcmp(tensor->name, "k") == 0 || std::strcmp(tensor->name, "v") == 0) axis = 2;
    if (std::strcmp(tensor->name, "upstream") == 0) axis = 1;
    ggml_backend_meta_split_state state = {ggml_backend_meta_split_axis(axis), {0}, {1}, 1};
    if (axis >= 0 && axis < GGML_MAX_DIMS) {
        state.ne[0] = tensor->ne[axis]/2;
        state.ne[1] = tensor->ne[axis] - state.ne[0];
    }
    return state;
}

static void tensor_parallel_accumulate(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int mode = 0; mode < 14; ++mode) {
        const bool subset = mode == 6 || mode == 7;
        const bool qkv = mode >= 8;
        const bool zero_qkv = mode >= 11;
        const int part = qkv ? (mode - 8)%3 : 0;
        int axes[] = {mode == 1 ? 1 : 0, mode == 0 || subset || qkv ? 1 : 0};
        auto split = [](const ggml_tensor * tensor, void * userdata) {
            auto state = test_split_state(tensor, userdata);
            if (std::strcmp(tensor->name, "cache") == 0) {
                state.axis = GGML_BACKEND_SPLIT_AXIS_0; state.ne[0] = state.ne[1] = tensor->ne[0]/2;
            }
            if (std::strcmp(tensor->name, "left") == 0 && tensor->ne[0] == 64 && tensor->ne[1] == 4) {
                state.nr[0] = 2;
                state.ne[0] /= 2;
                state.ne[1] /= 2;
            }
            if (std::strcmp(tensor->name, "qkv") == 0) {
                state.axis = tensor->ne[0] == 32 ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
                state.n_segments = 2;
                state.nr[1] = 2;
                state.ne[0] = state.ne[1] = 32;
                state.ne[2] = state.ne[3] = 16;
            }
            return state;
        };
        auto * dev = ggml_backend_meta_device(devices, 2, split, axes);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * a = mode == 1 ? ggml_new_tensor_3d(data, GGML_TYPE_F32, 16, 4, 4) :
            ggml_new_tensor_2d(data, GGML_TYPE_F32, qkv ? 128 : 64, subset ? 16 : 4);
        auto * b = mode == 0 || subset || qkv ? ggml_new_tensor_3d(data, GGML_TYPE_F32, 16, qkv && part != 0 ? 2 : 4, 4) :
            ggml_new_tensor_2d(data, GGML_TYPE_F32, mode == 1 ? 64 : 32, 4);
        ggml_set_name(a, "left"); ggml_set_name(b, "right");
        // Whole-cache reshapes use one segment; fused gate/up gradients use two.
        if (mode == 0 || mode >= 4) ggml_set_name(a, "cache");
        if (qkv) ggml_set_name(a, "qkv");
        auto * weight = zero_qkv ? ggml_new_tensor_2d(data, GGML_TYPE_F32, 32, 128) : nullptr;
        auto * input = zero_qkv ? ggml_new_tensor_2d(data, GGML_TYPE_F32, 32, 4) : nullptr;
        if (weight) ggml_set_name(weight, "qkv");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta accumulation allocation failed");
        std::vector<float> av(ggml_nelements(a)), bv(ggml_nelements(b));
        for (size_t i = 0; i < av.size(); ++i) av[i] = float(i % 13)/13;
        for (size_t i = 0; i < bv.size(); ++i) bv[i] = float(i % 7)/7;
        ggml_backend_tensor_set(a, av.data(), 0, av.size()*sizeof(float));
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size()*sizeof(float));
        if (zero_qkv) {
            std::vector<float> zeros(ggml_nelements(weight), 0.0f);
            ggml_backend_tensor_set(weight, zeros.data(), 0, ggml_nbytes(weight));
            ggml_backend_tensor_set(input, zeros.data(), 0, ggml_nbytes(input));
            a = ggml_scale(compute, ggml_mul_mat(compute, weight, input), 0.0f);
            std::fill(av.begin(), av.end(), 0.0f);
        }
        if (mode == 4 || mode == 5) {
            a = ggml_scale(compute, a, 0.0f);
            std::fill(av.begin(), av.end(), 0.0f);
        }
        const size_t offset = qkv ? (part == 0 ? 0 : part == 1 ? 64 : 96)*sizeof(float) :
            mode == 6 ? 3*a->nb[1] : (mode == 3 || mode == 5 ? 32*sizeof(float) : 0);
        auto * result = ggml_acc(compute, a, b, mode < 2 || subset || qkv ? b->ne[0]*sizeof(float) : a->nb[1],
            qkv ? a->nb[1] : mode < 2 || subset ? b->ne[0]*b->ne[1]*sizeof(float) : a->nb[2], ggml_nbytes(a), offset);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, result);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta accumulation graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta accumulation failed");
        std::vector<float> actual(av.size());
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size()*sizeof(float));
        for (size_t i = 0; i < bv.size(); ++i) {
            const size_t width = b->ne[0]*b->ne[1];
            const size_t index = qkv ? (i/width)*128 + i%width : mode < 2 || subset ? i : (i/32)*64 + i%32;
            av[index + offset/sizeof(float)] += bv[i];
        }
        for (size_t i = 0; i < av.size(); ++i) check(std::abs(actual[i] - av[i]) < 1e-6, "meta accumulation differs from reference");
        ggml_gallocr_free(allocator); ggml_backend_buffer_free(buffer);
        ggml_free(compute); ggml_free(data); ggml_backend_free(backend);
    }
}

static void tensor_parallel_copy(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    for (int axis : {0, 1}) for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q8_0}) {
        int axes[] = {axis, GGML_BACKEND_SPLIT_AXIS_MIRRORED};
        auto * dev = ggml_backend_meta_device(devices, 2, test_split_state, axes);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * source = ggml_new_tensor_2d(data, GGML_TYPE_F32, 128, 8);
        auto * destination = ggml_new_tensor_2d(data, type, 256, 4);
        ggml_set_name(source, "left");
        ggml_set_name(destination, "right");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta copy input allocation failed");
        std::vector<float> values(ggml_nelements(source));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 37) - 18)/19;
        ggml_backend_tensor_set(source, values.data(), 0, values.size()*sizeof(float));
        auto * result = ggml_cpy(compute, source, destination);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, result);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta copy graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta copy compute failed");
        std::vector<uint8_t> packed(ggml_nbytes(result));
        std::vector<float> actual(values.size());
        ggml_backend_tensor_get(result, packed.data(), 0, packed.size());
        for (size_t offset : {size_t(3), result->nb[1] - 7, packed.size()/2 - 13}) {
            std::vector<uint8_t> part(67);
            ggml_backend_tensor_get(result, part.data(), offset, part.size());
            check(std::memcmp(part.data(), packed.data() + offset, part.size()) == 0, "meta partial copy read differs from full read");
        }
        if (type == GGML_TYPE_F32) {
            std::memcpy(actual.data(), packed.data(), packed.size());
        } else {
            ggml_get_type_traits(type)->to_float(packed.data(), actual.data(), actual.size());
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            check(std::abs(actual[i] - values[i]) < (type == GGML_TYPE_F32 ? 1e-6f : 0.008f), "meta copy differs from reference");
        }
        ggml_gallocr_free(allocator); ggml_backend_buffer_free(buffer);
        ggml_free(compute); ggml_free(data); ggml_backend_free(backend);
    }
}

static void tensor_parallel_backward(ggml_backend_dev_t first, ggml_backend_dev_t second) {
    ggml_backend_dev_t devices[] = {first, second};
    const int mirrored = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    static int axes[][2] = {{1, 1}, {0, mirrored}, {mirrored, 0}};
    static int routed_axes[][2] = {{1, 0}, {0, mirrored}, {mirrored, 0}};
    for (bool routed : {false, true}) for (size_t index = 0; index < 3; ++index) {
        auto & split = axes[index];
        if (routed && split[0] == mirrored) continue;
        auto * dev = ggml_backend_meta_device(devices, 2, test_split_state, routed ? routed_axes[index] : split);
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        auto * data = ggml_init({1024*1024, nullptr, true});
        auto * compute = ggml_init({1024*1024, nullptr, true});
        auto * left = ggml_new_tensor_3d(data, routed ? GGML_TYPE_Q4_0 : GGML_TYPE_F32, routed ? 64 : 32, 64, routed ? 4 : 1);
        auto * right = routed ? ggml_new_tensor_3d(data, GGML_TYPE_F32, 64, 2, 3) :
                ggml_new_tensor_2d(data, GGML_TYPE_F32, 16, 64);
        auto * ids = routed ? ggml_new_tensor_2d(data, GGML_TYPE_I32, 2, 3) : nullptr;
        ggml_set_name(left, "left");
        ggml_set_name(right, "right");
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
        check(buffer != nullptr, "meta input allocation failed");
        std::vector<float> a(ggml_nelements(left)), b(ggml_nelements(right));
        for (size_t i = 0; i < a.size(); ++i) a[i] = float(int(i % 17) - 8)/17;
        for (size_t i = 0; i < b.size(); ++i) b[i] = float(int(i % 11) - 5)/11;
        if (routed) {
            std::vector<uint8_t> packed(ggml_nbytes(left));
            ggml_quantize_chunk(GGML_TYPE_Q4_0, a.data(), packed.data(), 0, ggml_nrows(left), left->ne[0], nullptr);
            ggml_backend_tensor_set(left, packed.data(), 0, packed.size());
            ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(packed.data(), a.data(), a.size());
        } else {
            ggml_backend_tensor_set(left, a.data(), 0, a.size()*sizeof(float));
        }
        ggml_backend_tensor_set(right, b.data(), 0, b.size()*sizeof(float));
        const int32_t routes[] = {0, 3, 2, 1, 3, 2};
        if (ids) ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
        auto * result = routed ? ggml_mul_mat_id_back(compute, left, right, ids) : ggml_out_prod(compute, left, right);
        result = ggml_scale(compute, result, 1.0f);
        auto * graph = ggml_new_graph(compute);
        ggml_build_forward_expand(graph, result);
        auto * allocator = ggml_gallocr_new(buft);
        check(ggml_gallocr_alloc_graph(allocator, graph), "meta graph allocation failed");
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "meta backward failed");
        std::vector<float> actual(ggml_nelements(result));
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size()*sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            const size_t width = left->ne[0];
            const size_t column = i % width;
            const size_t row = i / width;
            const size_t expert = routed ? routes[row] : 0;
            double expected = 0;
            for (size_t k = 0; k < 64; ++k) {
                expected += double(a[expert*width*64 + k*width + column]) * b[routed ? row*64 + k : k*16 + row];
            }
            check(std::abs(actual[i] - expected) < 2e-5, "tensor parallel backward differs from reference");
        }
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(buffer);
        ggml_free(compute);
        ggml_free(data);
        ggml_backend_free(backend);
    }
}

static std::vector<float> attention_backward(bool parallel) {
    auto * cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    ggml_backend_dev_t devices[] = {cpu, cpu};
    static int axes[] = {2, 2};
    auto * dev = parallel ? ggml_backend_meta_device(devices, 2, test_split_state, axes) : cpu;
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(dev);
    auto * data = ggml_init({1024*1024, nullptr, true});
    auto * compute = ggml_init({1024*1024, nullptr, true});
    ggml_tensor * inputs[] = {
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 8, 4),
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 8, 2),
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 8, 2),
        ggml_new_tensor_3d(data, GGML_TYPE_F32, 32, 4, 8),
    };
    const char * names[] = {"q", "k", "v", "upstream"};
    for (int i = 0; i < 4; ++i) ggml_set_name(inputs[i], names[i]);
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(data, buft);
    for (int i = 0; i < 4; ++i) {
        std::vector<float> values(ggml_nelements(inputs[i]));
        for (size_t j = 0; j < values.size(); ++j) values[j] = float(int((j + i) % 19) - 9)/19;
        ggml_backend_tensor_set(inputs[i], values.data(), 0, values.size()*sizeof(float));
    }
    auto * forward = ggml_flash_attn_ext(compute, inputs[0], inputs[1], inputs[2], nullptr, 0.125f, 0, 0);
    auto * back = ggml_flash_attn_ext_back(compute, forward, inputs[3]);
    auto * graph = ggml_new_graph(compute);
    ggml_tensor * views[3];
    size_t offset = 0;
    for (int i = 0; i < 3; ++i) {
        auto * input = inputs[i];
        views[i] = ggml_view_4d(compute, back, input->ne[0], input->ne[1], input->ne[2], 1,
                input->ne[0]*4, input->ne[0]*input->ne[1]*4, ggml_nelements(input)*4, offset);
        ggml_build_forward_expand(graph, views[i]);
        offset += GGML_PAD(ggml_nelements(input)*4, GGML_MEM_ALIGN);
    }
    auto * allocator = ggml_gallocr_new(buft);
    check(ggml_gallocr_alloc_graph(allocator, graph), "attention gradient allocation failed");
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention gradient compute failed");
    std::vector<float> result(offset/4);
    offset = 0;
    for (auto * view : views) {
        ggml_backend_tensor_get(view, result.data() + offset, 0, ggml_nbytes(view));
        offset += ggml_nelements(view);
    }
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(buffer);
    ggml_free(compute);
    ggml_free(data);
    ggml_backend_free(backend);
    return result;
}

int main(int argc, char ** argv) {
    try {
        llama_hparams hparams;
        hparams.n_layer_all = 30;
        hparams.n_layer_kv_from_start = 11;
        hparams.kv_global_only = true;
        for (uint32_t i = 0; i < 30; ++i) hparams.is_swa_impl[i] = i % 6 != 5;
        for (uint32_t i = 0; i < 30; ++i) {
            const int32_t expected = i % 6 != 5 ? int32_t(i) : 5;
            check(hparams.kv_source(i) == expected, "Lumen global KV source differs from recovery architecture");
            check(hparams.has_kv(i) == (i % 6 != 5 || i == 5), "Lumen must have one global KV owner");
        }
        if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
            benchmark(true);
            benchmark(false);
            return 0;
        }
        for (bool reference_mode : {false, true}) for (float t : {1.f, 2.f}) {
            loss_and_gradient(t, false, false, reference_mode);
            loss_and_gradient(t, true, false, reference_mode);
            loss_and_gradient(t, false, true, reference_mode);
        }
        cache_checks();
        frozen_q41_moe_gradient();
        auto * cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        tensor_parallel_backward(cpu, cpu);
        tensor_parallel_accumulate(cpu, cpu);
        tensor_parallel_copy(cpu, cpu);
        if (argc == 2 && std::strcmp(argv[1], "--cuda") == 0) {
            ggml_backend_load_all();
            auto * first = ggml_backend_dev_by_name("CUDA0");
            auto * second = ggml_backend_dev_by_name("CUDA1");
            check(first && second, "copy regression requires two CUDA devices");
            // The internal collective can round inputs to BF16; this reference checks F32 peer reduction.
            const char * allreduce = std::getenv("GGML_CUDA_ALLREDUCE");
            if (allreduce && std::strcmp(allreduce, "none") == 0) tensor_parallel_backward(first, second);
            tensor_parallel_accumulate(first, second);
            tensor_parallel_copy(first, second);
        }
        const auto serial = attention_backward(false);
        const auto parallel = attention_backward(true);
        for (size_t i = 0; i < serial.size(); ++i) {
            check(std::abs(serial[i] - parallel[i]) < 2e-5, "tensor parallel attention gradient differs from reference");
        }
        std::puts("distillation: KL reference, finite-difference gradients, masks, temperatures and cache checks passed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
