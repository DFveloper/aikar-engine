"""Verify streamed calibration captures use the supplied token IDs."""
import argparse
import json
import pathlib
import struct
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'model', 'manifest', 'tokens'):
    parser.add_argument(name, type=pathlib.Path)
args = parser.parse_args()
ids = json.loads(args.tokens.read_text())
groups = json.loads(args.manifest.read_text())['groups']
with tempfile.TemporaryDirectory(prefix='gsq-capture-test-') as directory:
    root = pathlib.Path(directory)
    report = root/'report.json'
    result = subprocess.run([str(args.binary), 'capture-store', str(args.model), str(args.manifest),
                             str(args.tokens), str(root/'captures'), str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    captured = json.loads(report.read_text())
    assert captured['tokens'] == ids
    assert len(captured['captures']) == len(groups)
    for item, group in zip(captured['captures'], groups):
        payload = pathlib.Path(item['file']).read_bytes()
        magic, columns, rows, samples = struct.unpack_from('<IQQQ', payload)
        assert magic == 0x31515347 and [columns, rows] == group['shape'][:2]
        assert samples == len(ids)
        assert len(payload) == 28 + 4 * columns * (rows + samples)
    inference = root/'inference.json'
    result = subprocess.run([str(args.binary), 'infer-tokens', str(args.model), str(args.tokens),
                             '2', str(inference), '0'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(inference.read_text())['prompt_token_ids'] == ids
    print('PASS: streamed captures preserve supplied calibration IDs and tensor shapes')
