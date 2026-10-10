"""Evaluate sealed models sequentially with identical held-out token IDs."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'config', 'windows', 'output'):
    parser.add_argument(name, type=pathlib.Path)
parser.add_argument('--gpu-layers', type=int, default=99)
args = parser.parse_args()
config = json.loads(args.config.read_text())
config_digest = hashlib.sha256(args.config.read_bytes()).hexdigest()
windows = json.loads(args.windows.read_text())
assert windows and all(2 <= len(window) <= 512 for window in windows)
expected_tokens = sum(len(window)-1 for window in windows)
arms = config['arms']
assert arms[0]['name'] == 'bf16' and arms[0]['model'] == config['source']
assert len({arm['name'] for arm in arms}) == len(arms)
assert all(arm['name'].replace('-', '').isalnum() for arm in arms)
args.output.mkdir(parents=True, exist_ok=True)
prompt = args.output/'decode-prompt.json'
if prompt.exists():
    assert json.loads(prompt.read_text()) == windows[0]
else:
    prompt.write_text(json.dumps(windows[0])+'\n')
wrapper = pathlib.Path(__file__).with_name('run_experiment.py')
teacher = args.output/'teacher'
results = []

def run(prefix, *command):
    subprocess.run([sys.executable, str(wrapper), str(prefix), str(args.binary),
                    *map(str, command)], check=True)

for index, arm in enumerate(arms):
    root = args.output/arm['name']
    root.mkdir(exist_ok=True)
    complete = root/'complete.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        assert saved['arm'] == arm and saved['windows_sha256'] == hashlib.sha256(args.windows.read_bytes()).hexdigest()
        assert saved['config_sha256'] == config_digest and saved['quality']['gpu_layers'] == args.gpu_layers
        results.append(saved)
        continue
    assert not (root/'quality.json').exists(), 'Partial evaluation needs inspection before rerun'
    temporary = 'model' not in arm
    model = root/'temporary.gguf' if temporary else pathlib.Path(arm['model'])
    if temporary:
        assert not model.exists()
        run(root/'publish', 'candidate-publish', config['source'], arm['manifest'],
            arm['assignment'], model, config['budget'], root/'publish.json')
    if 'manifest' in arm:
        run(root/'verify', 'candidate-verify', config['source'], arm['manifest'],
            arm['assignment'], model, root/'verify.json')
    assert arm['name'] == 'bf16' or model.stat().st_size <= config['budget']
    run(root/'evaluation', 'corpus-eval', model, args.windows, teacher, root/'quality.json',
        args.gpu_layers, 'write' if index == 0 else 'read')
    run(root/'decode', 'infer-tokens', model, prompt, 32, root/'decode.json', args.gpu_layers)
    quality = json.loads((root/'quality.json').read_text())
    decode = json.loads((root/'decode.json').read_text())
    assert quality['evaluation_tokens'] == expected_tokens
    assert decode['prompt_token_ids'] == windows[0]
    result = {'arm': arm, 'quality': quality, 'decode': decode,
              'evaluation_resources': json.loads((root/'evaluation.resources.json').read_text()),
              'decode_resources': json.loads((root/'decode.resources.json').read_text()),
              'windows_sha256': hashlib.sha256(args.windows.read_bytes()).hexdigest(),
              'config_sha256': config_digest}
    complete.write_text(json.dumps(result, indent=2)+'\n')
    results.append(result)
    (args.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    if temporary:
        model.unlink()
    print(json.dumps({'arm': arm['name'], 'quality': quality}, default=str), flush=True)
(args.output/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
