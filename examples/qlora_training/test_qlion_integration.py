#!/usr/bin/env python3
"""Run QLion integration checks with local dense/MoE fixtures; no downloads."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any

import gguf
import numpy as np


def tensor_map(path):
    reader = gguf.GGUFReader(str(path))
    return reader, {tensor.name: tensor for tensor in reader.tensors}


def compare_models(left, right):
    left_reader, a = tensor_map(left)
    right_reader, b = tensor_map(right)
    assert a.keys() == b.keys(), "tensor directories differ"
    for name, tensor in a.items():
        assert tensor.tensor_type == b[name].tensor_type, name
        assert np.array_equal(tensor.shape, b[name].shape), name
        assert np.array_equal(tensor.data, b[name].data), name
    return len(a)


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()



def make_ple_fixture(source, output):
    reader = gguf.GGUFReader(str(source))
    writer = gguf.GGUFWriter(str(output), "gemma4")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = 32 if key == "gemma4.embedding_length_per_layer_input" else field.contents()
        writer.add_key_value(key, value, field.types[0], field.types[-1] if len(field.types) > 1 else None)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data.copy(), raw_dtype=tensor.tensor_type)
    rng = np.random.default_rng(123)

    def weight(name, shape, norm=False):
        values = np.ones(shape, dtype=np.float32) if norm else (rng.standard_normal(shape) * 0.03).astype(np.float32)
        dtype = gguf.GGMLQuantizationType.F32 if norm else gguf.GGMLQuantizationType.Q8_0
        writer.add_tensor(name, values if norm else gguf.quants.quantize(values, dtype), raw_dtype=dtype)

    weight("per_layer_token_embd.weight", (8192, 192))
    weight("per_layer_model_proj.weight", (192, 256))
    weight("per_layer_proj_norm.weight", (32,), norm=True)
    for layer in range(6):
        weight(f"blk.{layer}.inp_gate.weight", (32, 256))
        weight(f"blk.{layer}.proj.weight", (256, 32))
        weight(f"blk.{layer}.post_norm.weight", (256,), norm=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True, help="directory containing both GGUFs and synthetic_chat.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-layers", default="0")
    parser.add_argument("--gpu-index", type=int, default=0, help="physical GPU index for NVML device-memory sampling")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"device": "CPU" if args.gpu_layers == "0" else "GPU (record device separately)", "runs": {}}
    gpu_memory = None
    if args.gpu_layers != "0":
        class MemoryInfo(ctypes.Structure):
            _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        assert nvml.nvmlInit_v2() == 0, "NVML initialization failed"
        handle = ctypes.c_void_p()
        assert nvml.nvmlDeviceGetHandleByIndex_v2(args.gpu_index, ctypes.byref(handle)) == 0, "invalid NVML GPU index"

        def read_gpu_memory():
            memory = MemoryInfo()
            assert nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0, "NVML memory query failed"
            return int(memory.used)

        gpu_memory = read_gpu_memory
        identity = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu_index),
            "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
        report["gpu"] = {"identity": identity, "memory_scope": "sampled whole-device usage, including other processes",
                         "sample_interval_seconds": 0.01}
    report["comparison"] = {
        "warmup_optimizer_steps": 2,
        "time_goal_ratio_moe_to_dense": 1.10,
        "peak_rss_goal_ratio_moe_to_dense": 1.10,
        "peak_device_vram_goal_ratio_moe_to_dense": 1.10,
        "note": "Goals are recorded before measurement; CPU results do not validate P620 goals.",
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report["comparison"], indent=2))
    small_data = args.output_dir / "small.jsonl"
    small_data.write_text("\n".join((args.fixtures / "synthetic_chat.jsonl").read_text().splitlines()[:4]) + "\n")

    def run(name, command, expected=0):
        command = [str(value) for value in command]
        log = args.output_dir / (name + ".log")
        rss = args.output_dir / (name + ".rss")
        timed = ["/usr/bin/time", "-f", "%M", "-o", str(rss)] + command
        start = time.monotonic()
        memory_samples = []
        if gpu_memory:
            memory_samples.append(gpu_memory())
        with log.open("w") as output:
            process = subprocess.Popen(timed, stdout=output, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    if gpu_memory:
                        memory_samples.append(gpu_memory())
                    time.sleep(0.01)
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()
        elapsed = time.monotonic() - start
        text = log.read_text(errors="replace")
        entry = {"command": command, "exit_code": process.returncode, "wall_seconds": elapsed, "log": str(log)}
        entry["peak_rss_kib"] = int(rss.read_text().splitlines()[-1])
        if memory_samples:
            entry["gpu_baseline_bytes"] = memory_samples[0]
            entry["peak_device_vram_bytes"] = max(memory_samples)
            entry["gpu_memory_samples"] = len(memory_samples)
        steps = [(int(n), int(tokens), int(duration)) for n, tokens, duration in
                 re.findall(r"qat_step: step=(\d+) supervised_tokens=(\d+) duration_us=(\d+)", text)]
        measured = steps[2:]
        if measured:
            seconds = [duration / 1e6 for _, _, duration in measured]
            entry["measured_steps"] = len(measured)
            entry["step_seconds_median"] = statistics.median(seconds)
            entry["step_seconds_range"] = [min(seconds), max(seconds)]
            entry["effective_supervised_tokens_per_second"] = sum(tokens for _, tokens, _ in measured) / sum(seconds)
        report["runs"][name] = entry
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        assert process.returncode == expected, f"{name}: expected {expected}, got {process.returncode}; see {log}"
        return text

    trainer = args.bin_dir / "llama-finetune-qlion"
    exporter = args.bin_dir / "llama-export-qlion"
    completion = args.bin_dir / "llama-completion"
    models = {"dense": args.fixtures / "gemma4-tiny-6l-q4.gguf", "moe": args.fixtures / "gemma4-tiny-moe-q4.gguf"}

    def train(name, model, extra=(), data=None, epochs=1):
        full = args.output_dir / (name + ".gguf")
        adapter = args.output_dir / (name + "-adapter.gguf")
        text = run(name, [trainer, "-m", model, "--train-file", data or small_data,
                         "--optimizer", "qlion2", "--quant-type", "q4_0,q8_0", "-c", "256", "-b", "256", "-ub", "256",
                         "-ngl", args.gpu_layers, "-t", "4", "-tb", "4", "--dataset-threads", "2",
                         "--epochs", str(epochs), "--val-split", "0", "--learning-rate", "0.01", "--lr-scheduler", "constant",
                         "--verbosity", "5", "--save-every", "0", "--qat-out", full, "--qat-adapter-out", adapter, "--verbose-loss", *extra])
        return full, adapter, text

    for model_name, model in models.items():
        reader, tensors = tensor_map(model)
        expert_parameters = sum(t.n_elements for t in tensors.values() if "_exps." in t.name)
        total = sum(t.n_elements for t in tensors.values())
        report[model_name] = {"parameters": int(total), "bytes": model.stat().st_size,
                              "tied_embedding_output": "output.weight" not in tensors,
                              "active_parameters_estimate": int(total - expert_parameters / 2),
                              "experts": 4 if model_name == "moe" else 0,
                              "experts_used": 2 if model_name == "moe" else 0,
                              "sha256": file_hash(model)}
        dynamic, adapter, dynamic_text = train(model_name + "-dynamic", model, data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
        static, _, static_text = train(model_name + "-static", model, ["--qat-static-graph", "on"],
                                       data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
        reused = "reusing forward/backward/optimizer graph" in static_text
        report[model_name]["static_graph_reused"] = reused
        compare_models(dynamic, static)
        loss_pattern = r"qat_loss:.*?loss=([0-9.e+-]+)"
        np.testing.assert_allclose([float(x) for x in re.findall(loss_pattern, dynamic_text)],
                                   [float(x) for x in re.findall(loss_pattern, static_text)], rtol=1e-5, atol=1e-5)
        merged = args.output_dir / (model_name + "-merged.gguf")
        run(model_name + "-merge", [exporter, "-m", model, "--adapter", adapter, "-o", merged])
        compare_models(dynamic, merged)
        infer = ["-p", "hi", "-n", "2", "-c", "256", "-b", "256", "-ub", "256", "-ngl", args.gpu_layers,
                 "-t", "4", "--load-mode", "none", "--lazy-mode", "off", "-no-cnv", "--seed", "42", "--verbosity", "0"]
        run(model_name + "-infer-merged", [completion, "-m", merged, *infer])
        run(model_name + "-infer-runtime", [completion, "-m", model, "--qlion-adapter", adapter, *infer])
        merged_output = (args.output_dir / (model_name + "-infer-merged.log")).read_text()
        runtime_output = (args.output_dir / (model_name + "-infer-runtime.log")).read_text()
        assert "hi" in merged_output, "inference did not emit the prompt"
        assert merged_output == runtime_output, "runtime and merged inference output differ"
        for target in ("attn_k", "attn_v"):
            for flash in ("on", "off"):
                name = model_name + "-" + target + "-fa-" + flash
                _, partial, text = train(name, model, ["--qat-targets", "blk.0." + target + ".weight", "-fa", flash,
                                                        "--save-first-at", "1", "--qat-adapter-units", "rows"])
                state = args.output_dir / (name + ".gguf.step1.gguf.qat-state.gguf")
                state_reader, states = tensor_map(state)
                momentum = states["qat.m.000000"]
                values = gguf.quants.dequantize(momentum.data, momentum.tensor_type)
                assert np.isfinite(values).all() and np.any(values != 0), "missing " + name + " gradient"
                original_reader, originals = tensor_map(model)
                trained_reader, trained = tensor_map(args.output_dir / (name + ".gguf"))
                for tensor_name, original in originals.items():
                    if tensor_name != "blk.0." + target + ".weight":
                        assert np.array_equal(original.data, trained[tensor_name].data), tensor_name
                run(name + "-merge", [exporter, "-m", model, "--adapter", partial, "-o", merged])
                compare_models(args.output_dir / (name + ".gguf"), merged)

    if args.gpu_layers != "0":
        partial, _, text = train("moe-partial-gpu", models["moe"],
            ["-ngl", "0", "--no-op-offload", "--override-tensor",
             r"blk\.0\.attn_(k|v)\.weight=CUDA0,blk\.5\.ffn_down_exps\.weight=CUDA0",
             "--qat-targets", "blk.0.attn_k.weight,blk.0.attn_v.weight,blk.5.ffn_down_exps.weight", "--save-first-at", "1"],
            data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
        selected = set(re.findall(r"qat_target: (\S+)", text))
        assert len(selected) == 3 and re.search(r"CPU nodes=[1-9]\d*", text), "partial GPU placement was not exercised"
        state_reader, states = tensor_map(str(partial) + ".step1.gguf.qat-state.gguf")
        for index in range(3):
            momentum = states[f"qat.m.{index:06d}"]
            values = gguf.quants.dequantize(momentum.data, momentum.tensor_type)
            assert np.isfinite(values).all() and np.any(values != 0), "partial GPU gradient is missing"
        base_reader, originals = tensor_map(models["moe"])
        trained_reader, trained = tensor_map(partial)
        for name in originals:
            if name not in selected:
                assert np.array_equal(originals[name].data, trained[name].data), name

    for model_name, model in models.items():
        off, _, _ = train(model_name + "-reuse-reference", model, epochs=4)
        on, _, text = train(model_name + "-reuse", model, ["--qat-static-graph", "on"], epochs=4)
        reused = "reusing forward/backward/optimizer graph" in text
        assert reused, "compatible CPU/CUDA graph was not reused"
        report["runs"][model_name + "-reuse"]["graph_reused"] = reused
        compare_models(off, on)
    dense_gradient, _, _ = train("moe-gradient-reference", models["moe"],
        ["--qat-moe-gradient", "dense", "-ub", "64"], data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
    fused_gradient, _, _ = train("moe-gradient-fused", models["moe"],
        ["--qat-moe-gradient", "fused", "-ub", "64"], data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
    compare_models(dense_gradient, fused_gradient)
    compact_gradient, _, compact_text = train("moe-gradient-compact", models["moe"],
        ["--qat-moe-gradient", "compact", "-ub", "64", "--qat-static-graph", "on"],
        data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
    compare_models(dense_gradient, compact_gradient)
    assert "reusing forward/backward/optimizer graph" not in compact_text, "compact mode reused transient route buffers"

    for case, extra in (("accum", ["--grad-accum", "2"]),
                        ("eval", ["--val-split", "0.5", "--eval-every", "1"])):
        outputs = []
        for mode in ("dense", "compact"):
            full, _, _ = train("moe-compact-" + case + "-" + mode, models["moe"],
                ["--qat-moe-gradient", mode, "-ub", "64", *extra],
                data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
            outputs.append(full)
        compare_models(*outputs)
        if args.gpu_layers != "0":
            for placement, layers, override in (
                    ("hybrid", "0", r"blk\.0\.attn_(k|v)\.weight=CUDA0,blk\.5\.ffn_down_exps\.weight=CUDA0"),
                    ("cpu-expert", "99", r"blk\.5\.ffn_down_exps\.weight=CPU")):
                outputs = []
                for mode in ("dense", "compact"):
                    full, _, text = train("moe-compact-" + placement + "-" + case + "-" + mode, models["moe"],
                        ["--qat-moe-gradient", mode, "-ub", "64", "-ngl", layers, "--no-op-offload",
                         "--override-tensor", override,
                         "--qat-targets", "blk.0.attn_k.weight,blk.0.attn_v.weight,blk.5.ffn_down_exps.weight", *extra],
                        data=args.fixtures / "synthetic_chat.jsonl", epochs=2)
                    if placement == "cpu-expert" and mode == "compact":
                        assert "GPU gradient with CPU expert state" in text, "missing compact placement fallback"
                    outputs.append(full)
                compare_models(*outputs)
    checkpoint = args.output_dir / "dense-attn_k-fa-on.gguf.step1.gguf"
    reference, _, _ = train("resume-reference", models["dense"], ["--qat-targets", "blk.0.attn_k.weight", "-fa", "on"], epochs=2)
    resumed, _, _ = train("resume", checkpoint, ["--qat-resume", checkpoint, "--qat-targets", "blk.0.attn_k.weight", "-fa", "on"], epochs=2)
    compare_models(reference, resumed)
    for target in ("ffn_gate_inp.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_up.weight"):
        name = "moe-gradient-" + target
        train(name, models["moe"], ["--qat-targets", "blk.0." + target, "--save-first-at", "1", "--qat-adapter-units", "experts"])
        state_reader, states = tensor_map(args.output_dir / (name + ".gguf.step1.gguf.qat-state.gguf"))
        momentum = states["qat.m.000000"]
        values = gguf.quants.dequantize(momentum.data, momentum.tensor_type)
        assert np.isfinite(values).all() and np.any(values != 0), "missing " + name + " gradient"
    full, adapter, text = train("frozen", models["dense"], ["--qat-targets", "attn_k,attn_v", "--freeze-layers", "5"])
    names = re.findall(r"qat_target: (\S+)", text)
    assert names and all(name.startswith("blk.5.") for name in names)
    base_reader, originals = tensor_map(models["dense"])
    trained_reader, trained = tensor_map(full)
    for name in originals:
        if name not in names:
            assert np.array_equal(originals[name].data, trained[name].data), name
    train("tied-explicit", models["dense"], ["--qat-targets", "output.weight"])
    for case, extra in (("empty", ["--qat-targets", "does-not-exist"]), ("freeze-negative", ["--freeze-layers", "-1"]),
                        ("freeze-large", ["--freeze-layers", "7"]), ("freeze-all", ["--freeze-layers", "6", "--qat-targets", "attn_k"])):
        run(case, [trainer, "-m", models["dense"], "--train-file", small_data, "--quant-type", "q4_0,q8_0",
                   "-c", "256", "-b", "256", "-ub", "256", "-ngl", "0", *extra], expected=1)
    for case, extra in (("microbatch", ["-ub", "64"]), ("accum", ["--grad-accum", "2"]),
                        ("parallel-last", ["-np", "2"]), ("length-change", ["-c", "512", "-b", "512", "-ub", "512"]),
                        ("periodic-eval", ["--val-split", "0.5", "--eval-every", "1"])):
        off, _, _ = train(case + "-dynamic", models["dense"], extra)
        on, _, _ = train(case + "-static", models["dense"], [*extra, "--qat-static-graph", "on"])
        compare_models(off, on)
    long_data = args.output_dir / "long.jsonl"
    long_data.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"},
        {"role": "assistant", "content": " hello" * 100}]}) + "\n")
    for model_name, model in models.items():
        for context in (256, 512):
            name = model_name + "-micro-reuse-" + str(context)
            extra = ["-c", str(context), "-b", str(context), "-ub", "64"]
            off, _, _ = train(name + "-reference", model, extra, data=long_data)
            on, _, text = train(name, model, [*extra, "--qat-static-graph", "on"], data=long_data)
            assert "reusing forward/backward/optimizer graph" in text, "microbatch graph was not reused"
            compare_models(off, on)

    loader = args.bin_dir / "test-qlion-loader"
    assert loader.exists(), "build test-qlion-loader to check the direct core API"
    report["core_loader"] = []
    for model_name, model in models.items():
        for suffix in ("dynamic", "attn_k-fa-on"):
            reference = args.output_dir / (model_name + "-" + suffix + ".gguf")
            adapter = args.output_dir / (model_name + "-" + suffix + "-adapter.gguf")
            for mode in ("none", "mmap"):
                for lazy in ("off", "on"):
                    for repack in ("off", "on"):
                        outputs = []
                        for base, overlay in ((reference, "-"), (model, adapter)):
                            result = subprocess.run([str(loader), str(base), str(overlay), mode, lazy, repack],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                            outputs.append(result.stdout)
                        assert outputs[0] and outputs[0] == outputs[1], "core-loader logits differ"
                        report["core_loader"].append([model_name, suffix, mode, lazy, repack])

    ple_base = args.output_dir / "ple-base.gguf"
    make_ple_fixture(models["dense"], ple_base)
    ple_hash = file_hash(ple_base)
    ple_full, ple_adapter, _ = train("ple-rows", ple_base,
        ["-ngl", "0", "--qat-targets", "per_layer_token_embd.weight", "--qat-adapter-units", "rows"], epochs=2)
    ple_reader, ple_payload = tensor_map(ple_adapter)
    assert "per_layer_token_embd.weight" in ple_payload, "lazy tensor adapter is empty"
    for mode in ("none", "mmap"):
        for lazy in ("off", "on"):
            for repack in ("off", "on"):
                outputs = []
                for base, overlay in ((ple_full, "-"), (ple_base, ple_adapter)):
                    result = subprocess.run([str(loader), str(base), str(overlay), mode, lazy, repack],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    if lazy == "on":
                        assert b"lazy read enabled" in result.stderr, "lazy tensor was loaded eagerly"
                    outputs.append(result.stdout)
                assert outputs[0] and outputs[0] == outputs[1], "lazy overlay logits differ"
                report["core_loader"].append(["ple", "rows", mode, lazy, repack])
    assert file_hash(ple_base) == ple_hash, "lazy overlay modified the base file"

    dense_run = report["runs"]["dense-dynamic"]
    moe_run = report["runs"]["moe-dynamic"]
    prefix = "cpu" if args.gpu_layers == "0" else "gpu"
    comparison = report["comparison"]
    comparison[prefix + "_time_ratio"] = moe_run["step_seconds_median"] / dense_run["step_seconds_median"]
    comparison[prefix + "_peak_rss_ratio"] = moe_run["peak_rss_kib"] / dense_run["peak_rss_kib"]
    comparison[prefix + "_time_goal_met"] = comparison[prefix + "_time_ratio"] <= 1.10
    comparison[prefix + "_peak_rss_goal_met"] = comparison[prefix + "_peak_rss_ratio"] <= 1.10
    if gpu_memory:
        comparison["gpu_peak_device_vram_ratio"] = moe_run["peak_device_vram_bytes"] / dense_run["peak_device_vram_bytes"]
        comparison["gpu_peak_device_vram_goal_met"] = comparison["gpu_peak_device_vram_ratio"] <= 1.10
        nvml.nvmlShutdown()
    comparison["p620_validated"] = gpu_memory is not None and "P620" in report["gpu"]["identity"]
    report["data_sha256"] = file_hash(args.fixtures / "synthetic_chat.jsonl")
    for name, model in models.items():
        assert file_hash(model) == report[name]["sha256"], "base file was modified"
    report["passed"] = True
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": True, "report": str(args.output_dir / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
