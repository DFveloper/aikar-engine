"""Preserve source text and create immutable, disjoint native-token windows."""
import argparse
import hashlib
import json
import pathlib
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=pathlib.Path, required=True)
parser.add_argument('--model', type=pathlib.Path, required=True)
parser.add_argument('--source', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--context', type=int, default=128)
args = parser.parse_args()
if args.output.exists():
    parser.error('refusing to overwrite data directory')
if not 2 <= args.context <= 512:
    parser.error('context must be 2..512')
raw = args.source.read_bytes()
has_bos = raw.startswith(b'<bos>')
command = [str(args.binary), '-m', str(args.model), '-f', str(args.source), '--ids', '--no-escape']
if has_bos:
    command.append('--no-bos')
result = subprocess.run(command, text=True, capture_output=True, check=True)
tokens = json.loads(result.stdout)
if len(tokens) <= 12513:
    parser.error('source too short for the fixed 12000/512/held-out split')
splits = {'calibration': (0, 12000), 'heldout': (12512, len(tokens))}
manifest = {
    'schema': 1, 'source': str(args.source.resolve()),
    'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_bytes': len(raw),
    'model': str(args.model.resolve()), 'tokenizer_command': command,
    'source_tokens': len(tokens), 'context': args.context, 'boundary_discarded_tokens': 512,
    'token_ids_sha256': hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
    'template': 'source raw text; no additional system/chat template',
    'limitation': 'sequential holdout from one conversation; not independent conversations',
    'splits': {},
}
args.output.mkdir(parents=True)
for name, (start, end) in splits.items():
    ids = tokens[start:end]
    windows = [ids[i:i+args.context] for i in range(0, len(ids), args.context)]
    dropped = sum(len(w) for w in windows if len(w) < 2)
    windows = [w for w in windows if len(w) >= 2]
    output = json.dumps(windows, separators=(',', ':')).encode()
    (args.output / (name + '.json')).write_bytes(output)
    manifest['splits'][name] = {
        'start': start, 'end': end, 'tokens': len(ids), 'windows': len(windows),
        'scored_tokens': sum(len(w)-1 for w in windows), 'dropped_tokens': dropped,
        'windows_sha256': hashlib.sha256(output).hexdigest(),
    }
(args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
