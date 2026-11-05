#!/usr/bin/env python3
"""Convert a Gemma 4 GGUF to Lumen 3.8 Pulsar Next and preserve weight payloads."""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf


def convert(source, output, shared_layers=19, context_length=1048576):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("output already exists")
    reader = gguf.GGUFReader(str(source))
    if reader.fields["general.architecture"].contents() != "gemma4":
        raise ValueError("source architecture must be gemma4")
    if any(key.startswith("split.") for key in reader.fields):
        raise ValueError("merge split GGUF files before conversion")
    original_context = reader.fields.get("gemma4.rope.scaling.original_context_length")
    if original_context is None:
        original_context = reader.fields.get("gemma4.context_length")
    if original_context is None:
        raise ValueError("source is missing its original context length")
    original_context = int(original_context.contents())
    if not 0 < original_context <= context_length <= 0xFFFFFFFF:
        raise ValueError("context length must be at least the positive original context length and fit uint32")
    count = int(reader.fields["gemma4.block_count"].contents())
    pattern = reader.fields["gemma4.attention.sliding_window_pattern"].contents()
    if len(pattern) != count or not 0 <= shared_layers < count:
        raise ValueError("invalid layer pattern or sharing count")
    if not any(not pattern[i] for i in range(count - shared_layers)):
        raise ValueError("sharing requires an independent global layer")
    version = reader.fields.get("general.quantization_version")
    legacy_nvfp4 = version is None or version.contents() < 3
    names = {tensor.name for tensor in reader.tensors}
    for i in range(count):
        if pattern[i] or i < count - shared_layers:
            prefix = f"blk.{i}.attn_"
            if prefix + "k_norm.weight" not in names or not (
                prefix + "k.weight" in names or prefix + "qkv.weight" in names
            ):
                raise ValueError(f"source lacks independent KV weights for layer {i}")
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    os.close(fd)
    writer = gguf.GGUFWriter(temporary, None, endianess=reader.endianess)
    try:
        writer.data_alignment = int(reader.alignment)
        for key, field in reader.fields.items():
            if key.startswith("GGUF.") or key in ("general.architecture", "general.name", "general.quantization_version"):
                continue
            key = key.replace("gemma4.", "lumen4exp.", 1) if key.startswith("gemma4.") else key
            if key in ("lumen4exp.attention.shared_kv_layers", "lumen4exp.context_length",
                       "lumen4exp.rope.scale_linear") or key.startswith("lumen4exp.rope.scaling."):
                continue
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
        writer.add_string("general.architecture", "lumen4exp")
        writer.add_name("Lumen 3.8 Pulsar Next")
        writer.add_uint32("lumen4exp.attention.shared_kv_layers", shared_layers)
        writer.add_uint32("lumen4exp.context_length", context_length)
        writer.add_string("lumen4exp.rope.scaling.type", gguf.RopeScalingType.YARN.value)
        writer.add_float32("lumen4exp.rope.scaling.factor", context_length / original_context)
        writer.add_uint32("lumen4exp.rope.scaling.original_context_length", original_context)
        writer.add_bool("lumen4exp.rope.scaling.finetuned", False)
        for tensor in reader.tensors:
            nbytes = tensor.n_bytes
            if legacy_nvfp4 and tensor.tensor_type == gguf.GGMLQuantizationType.NVFP4:
                nbytes += 4 * (tensor.n_elements // int(tensor.shape[0]))
            writer.add_tensor_info(tensor.name, list(reversed(tensor.shape.tolist())),
                                   np.dtype(np.float32), nbytes, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            data = tensor.data
            if legacy_nvfp4 and tensor.tensor_type == gguf.GGMLQuantizationType.NVFP4:
                header = np.ones((*data.shape[:-1], 1), dtype=np.float32).view(np.uint8)
                data = np.concatenate((header, data), axis=-1)
            writer.write_tensor_data(data)
        writer.close()
        os.link(temporary, output)
    finally:
        writer.close()
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shared-layers", type=int, default=19)
    parser.add_argument("--context-length", type=int, default=1048576,
                        help="YaRN target context length (default: 1048576 tokens)")
    args = parser.parse_args()
    convert(args.source, args.output, args.shared_layers, args.context_length)


if __name__ == "__main__":
    main()
