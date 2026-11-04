#!/usr/bin/env python3
"""Mix all Korean conversations with an equal reservoir sample from a second JSONL."""
import argparse
import json
from pathlib import Path
import random


def mix(ko, other, output, seed):
    sources = [Path(ko).resolve(), Path(other).resolve()]
    output = Path(output).resolve()
    if output in sources:
        raise ValueError('Output must differ from inputs')
    config = {'sources': [{'path': str(path), 'bytes': path.stat().st_size,
                           'mtime_ns': path.stat().st_mtime_ns} for path in sources], 'seed': seed}
    manifest = Path(str(output) + '.json')
    if output.is_file() and manifest.is_file():
        previous = json.loads(manifest.read_text())
        if previous['inputs'] == config and previous['output_bytes'] == output.stat().st_size:
            return
        raise ValueError('Mixed dataset inputs changed; choose a new output path')
    if output.exists():
        raise ValueError('Untracked mixed dataset exists')
    korean = [line for line in sources[0].read_bytes().splitlines() if line.strip()]
    if not korean:
        raise ValueError('Empty Korean dataset')
    rng = random.Random(seed)
    sample = []
    seen = 0
    with sources[1].open('rb') as stream:
        for line in stream:
            if not line.strip():
                continue
            seen += 1
            if len(sample) < len(korean):
                sample.append(line.rstrip(b'\r\n'))
            else:
                index = rng.randrange(seen)
                if index < len(sample):
                    sample[index] = line.rstrip(b'\r\n')
    if len(sample) != len(korean):
        raise ValueError('Second dataset has fewer conversations than Korean dataset')
    records = korean + sample
    for line in records:
        if not isinstance(json.loads(line), dict):
            raise ValueError('Expected JSON objects')
    rng.shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + '.partial')
    with temporary.open('wb') as stream:
        for line in records:
            stream.write(line + b'\n')
    temporary.replace(output)
    manifest.write_text(json.dumps({'inputs': config, 'ko_rows': len(korean),
                                   'other_rows': len(sample), 'other_source_rows': seen,
                                   'output_bytes': output.stat().st_size}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ko', required=True)
    parser.add_argument('--other', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    mix(args.ko, args.other, args.output, args.seed)
