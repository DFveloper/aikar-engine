#!/usr/bin/env python3
"""Check NVFP4 KV QAT using a local 30-layer Gemma 4 fixture and JSONL data."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(ROOT / "scripts"))
import gguf
from convert_gemma4_to_lumen4exp import convert


def tensors(path):
    reader = gguf.GGUFReader(path)
    return reader, {tensor.name: tensor for tensor in reader.tensors}


def check_changes(source, output, shared):
    original, before = tensors(source)
    trained, after = tensors(output)
    assert before.keys() == after.keys(), "tensor directory changed"
    changed = []
    for name, tensor in before.items():
        assert tensor.tensor_type == after[name].tensor_type, name
        np.testing.assert_array_equal(tensor.shape, after[name].shape)
        if not np.array_equal(tensor.data, after[name].data):
            assert ".attn_k" in name, f"frozen tensor changed: {name}"
            changed.append(name)
    arch = original.fields["general.architecture"].contents()
    pattern = original.fields[f"{arch}.attention.sliding_window_pattern"].contents()
    globals_ = [i for i, swa in enumerate(pattern) if not swa]
    for layer in globals_:
        name = f"blk.{layer}.attn_k.weight"
        if shared and layer >= len(pattern) - 13:
            assert name not in changed, f"unused shared projection changed: {name}"
        else:
            assert name in changed, f"KV owner received no weight update: {name}"
    for key, field in original.fields.items():
        if not key.startswith("GGUF."):
            assert trained.fields[key].contents() == field.contents(), key
    return changed


def validate_inference(bin_dir, output_dir, gpu_layers):
    results = {}
    for name in ("lumen4exp", "qlion-full", "qlion2-full"):
        command = [str(bin_dir / "llama-completion"), "-m", str(output_dir / f"{name}.gguf"),
                   "-p", "Hello", "-n", "3", "-c", "256", "-b", "64", "-ub", "64",
                   "-ngl", str(gpu_layers), "-fa", "on", "-no-cnv", "--verbosity", "5"]
        log = output_dir / f"{name}-inference.log"
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text(errors="replace")
        assert "K (nvfp4)" in text and "V (nvfp4)" in text
        if gpu_layers:
            assert "CUDA0 KV buffer" in text
        results[name] = {"command": command, "log": str(log)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-layers", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = gguf.GGUFReader(args.fixture)
    assert fixture.fields["general.architecture"].contents() == "gemma4"
    assert fixture.fields["gemma4.block_count"].contents() == 30
    assert fixture.fields["gemma4.attention.shared_kv_layers"].contents() == 0
    student = args.output_dir / "lumen4exp.gguf"
    convert(args.fixture, student)
    data = args.output_dir / "data.jsonl"
    data.write_text(args.data.read_text().splitlines()[0] + "\n")
    report = {"gpu_layers": args.gpu_layers, "runs": {}}
    if args.gpu_layers:
        report["gpu"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"], text=True).strip()

    def run(name, optimizer, microbatch, source=student, extra=(), shared=True):
        output = args.output_dir / f"{name}.gguf"
        command = [str(args.bin_dir / "llama-finetune-qlion"), "-m", str(source),
                   "--train-file", str(data), "--optimizer", optimizer, "--quant-type", "q4_0,q8_0",
                   "--qat-targets", "attn_k", "-c", "256", "-b", "256", "-ub", str(microbatch),
                   "-ngl", str(args.gpu_layers), "-t", "4", "-tb", "4", "--dataset-threads", "1",
                   "--epochs", "2", "--val-split", "0", "--learning-rate", "0.01",
                   "--lr-scheduler", "constant", "--verbosity", "5", "--save-every", "0",
                   "--qat-out", str(output), "--qat-adapter-out", str(args.output_dir / f"{name}-adapter.gguf"),
                   "--verbose-loss", "-fa", "on", *extra]
        log = args.output_dir / f"{name}.log"
        start = time.monotonic()
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text(errors="replace")
        assert "K (nvfp4)" in text and "V (nvfp4)" in text, "NVFP4 cache was bypassed"
        assert "kv_nvfp4_ste" in text, "NVFP4 STE training graph is absent"
        losses = [float(value) for value in re.findall(r"qat_epoch:.*?loss=([0-9.eE+-]+)", text)]
        assert losses and all(np.isfinite(loss) for loss in losses), "nonfinite or missing loss"
        changed = check_changes(source, output, shared)
        if args.gpu_layers:
            assert "CUDA0 KV buffer" in text, "KV storage fell back to host memory"
        report["runs"][name] = {"command": command, "wall_seconds": time.monotonic() - start,
                                "losses": losses, "changed_tensors": changed, "log": str(log)}
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
        return output

    for optimizer in ("qlion", "qlion2"):
        run(f"{optimizer}-full", optimizer, 256)
        run(f"{optimizer}-micro", optimizer, 128)
    explicit = run("qlion2-explicit", "qlion2", 256, extra=(
        "--kv-cache-training", "-ctgk", "nvfp4", "-ctgv", "nvfp4", "-ctlk", "f32", "-ctlv", "f32"))
    static = run("qlion2-static", "qlion2", 128, extra=("--qat-static-graph", "on"))
    for reference, candidate in ((args.output_dir / "qlion2-full.gguf", explicit),
                                 (args.output_dir / "qlion2-micro.gguf", static)):
        a_reader, a = tensors(reference)
        b_reader, b = tensors(candidate)
        for name in a:
            np.testing.assert_array_equal(a[name].data, b[name].data, err_msg=f"trajectory differs: {name}")
    run("gemma4-nvfp4", "qlion2", 128, source=args.fixture, shared=False, extra=(
        "--kv-cache-training", "-ctgk", "nvfp4", "-ctgv", "nvfp4", "-ctlk", "f32", "-ctlv", "f32"))
    report["mandatory_nvfp4_matches_explicit"] = True
    report["static_matches_dynamic"] = True
    report["inference"] = validate_inference(args.bin_dir, args.output_dir, args.gpu_layers)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": len(report["runs"]), "report": str(args.output_dir / "report.json")}))


if __name__ == "__main__":
    main()
