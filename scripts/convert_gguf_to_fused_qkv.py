#!/usr/bin/env python3
"""Fuse Gemma4 GGUF Q/K/V rows without dequantizing or requantizing.

Usage: python3 convert_gguf_to_fused_qkv.py source.gguf fused.gguf --verify
Use --missing-v duplicate to fuse layers that reuse K as V.
Requires numpy and the gguf package from this engine's gguf-py directory.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
    import gguf


HEADER_FIELDS = ('GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count')


def plan_conversion(reader, missing_v):
    if reader.fields['general.architecture'].contents() != 'gemma4':
        raise ValueError('Only Gemma4 is supported by this converter')
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise ValueError('Only little-endian GGUF is supported')
    if any(key.startswith('split.') for key in reader.fields):
        raise ValueError('Merge split GGUF files before conversion')
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    fused = {}
    removed = set()
    skipped = []
    for layer in range(reader.fields['gemma4.block_count'].contents()):
        prefix = f'blk.{layer}.attn_'
        q, k, v = [tensors.get(prefix + part + '.weight') for part in ('q', 'k', 'v')]
        if prefix + 'qkv.weight' in tensors:
            raise ValueError(f'Layer {layer} is already fused')
        if q is None:
            raise ValueError(f'Layer {layer} has no Q projection')
        if k is None or (v is None and missing_v == 'keep'):
            skipped.append(layer)
            continue
        parts = [q, k, v if v is not None else k]
        if any(len(t.shape) != 2 or t.shape[0] != q.shape[0] for t in parts):
            raise ValueError(f'Layer {layer}: incompatible Q/K/V dimensions')
        if not np.array_equal(parts[1].shape, parts[2].shape):
            raise ValueError(f'Layer {layer}: K/V dimensions differ')
        if any(t.tensor_type != q.tensor_type for t in parts):
            raise ValueError(f'Layer {layer}: Q/K/V quantization types differ')
        if any(prefix + part + suffix in tensors for part in ('q', 'k', 'v') for suffix in ('.bias', '.scale')):
            raise ValueError(f'Layer {layer}: separate bias/scale fusion is unsupported')
        name = prefix + 'qkv.weight'
        fused[q.name] = (name, parts)
        removed.update(t.name for t in parts if t.name != q.name)
    if not fused:
        raise ValueError('No compatible Q/K/V projections to fuse')
    entries = []
    for tensor in reader.tensors:
        if tensor.name in removed:
            continue
        entries.append(fused.get(tensor.name, (tensor.name, [tensor])))
    return entries, skipped


def verify_conversion(reader, output, entries):
    converted = gguf.GGUFReader(output)
    original_fields = {k: v for k, v in reader.fields.items() if k not in HEADER_FIELDS}
    target_fields = {k: v for k, v in converted.fields.items() if k not in HEADER_FIELDS}
    if original_fields.keys() != target_fields.keys():
        raise ValueError('Metadata keys changed')
    for name, field in original_fields.items():
        other = target_fields[name]
        if field.types != other.types or field.contents() != other.contents():
            raise ValueError(f'Metadata changed: {name}')
    if len(converted.tensors) != len(entries):
        raise ValueError('Tensor count changed unexpectedly')
    for tensor, (name, parts) in zip(converted.tensors, entries):
        shape = parts[0].shape.copy()
        if len(parts) > 1:
            shape[1] = sum(int(t.shape[1]) for t in parts)
        if tensor.name != name or tensor.tensor_type != parts[0].tensor_type or not np.array_equal(tensor.shape, shape):
            raise ValueError(f'Tensor descriptor mismatch: {name}')
        data = tensor.data.view(np.uint8).reshape(-1)
        offset = 0
        for part in parts:
            source = part.data.view(np.uint8).reshape(-1)
            for start in range(0, source.size, 8 * 1024 * 1024):
                end = min(start + 8 * 1024 * 1024, source.size)
                if not np.array_equal(data[offset + start:offset + end], source[start:end]):
                    raise ValueError(f'Tensor bytes changed: {name}')
            offset += source.size
        if offset != data.size:
            raise ValueError(f'Tensor size mismatch: {name}')


def convert(source, output, missing_v, verify, dry_run):
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Output must be a new file, separate from the source')
    reader = gguf.GGUFReader(source)
    entries, skipped = plan_conversion(reader, missing_v)
    summary = {
        'source': str(source), 'output': str(output), 'missing_v': missing_v,
        'fused_layers': sum(len(parts) == 3 for _, parts in entries),
        'skipped_layers': skipped, 'source_tensors': len(reader.tensors),
        'output_tensors': len(entries),
        'extra_weight_bytes': sum(sum(t.n_bytes for t in parts) - sum(t.n_bytes for t in {t.name: t for t in parts}.values()) for _, parts in entries),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output.parent)
    os.close(fd)
    writer = gguf.GGUFWriter(temporary, None, endianess=reader.endianess)
    try:
        writer.data_alignment = int(reader.alignment)
        for name, field in reader.fields.items():
            if name not in HEADER_FIELDS:
                writer.add_key_value(name, field.contents(), field.types[0], field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
        for name, parts in entries:
            shape = list(reversed(parts[0].shape.tolist()))
            if len(parts) > 1:
                shape[0] = sum(int(t.shape[1]) for t in parts)
            # Use a non-byte dtype so the writer keeps the logical shape.
            writer.add_tensor_info(name, shape, np.dtype(np.float32), sum(t.n_bytes for t in parts), raw_dtype=parts[0].tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, (_, parts) in enumerate(entries):
            data = parts[0].data if len(parts) == 1 else np.concatenate([t.data.view(np.uint8).reshape(-1) for t in parts])
            writer.write_tensor_data(data, tensor_endianess=reader.endianess)
            if index % 100 == 0:
                print(f'Written {index + 1}/{len(entries)} tensors', flush=True)
        writer.close()
        if verify:
            verify_conversion(reader, temporary, entries)
            print('Verified all metadata, tensor shapes, types and weight bytes', flush=True)
        # Publish without replacing a file created by another process.
        os.link(temporary, output)
    finally:
        writer.close()
        os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--missing-v', choices=('keep', 'duplicate'), default='keep')
    parser.add_argument('--verify', action='store_true', help='Compare all metadata and weight bytes before publishing')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        convert(args.source, args.output, args.missing_v, args.verify, args.dry_run)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')


if __name__ == '__main__':
    main()
