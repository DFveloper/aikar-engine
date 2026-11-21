from pathlib import Path
import json, os, re, subprocess, sys
import numpy as np
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'gguf-py'))
import gguf
out = Path('/tmp/aikar-host-async-accumulation')
out.mkdir(exist_ok=True)
data = out / 'samples.jsonl'
data.write_text(Path('/tmp/aikar-final-accepted-integration/full-context.jsonl').read_text() * 64)
reference = json.loads(Path('/tmp/aikar-final-accepted-integration/report.json').read_text())
report = {}
for model in ('dense', 'moe'):
    results = []
    for version in ('baseline', 'host-async'):
        cmd = reference['runs'][f'full-context-{model}-qlion2-ub768']['command'].copy()
        adapter = out / f'{model}-{version}.adapter.gguf'
        changes = {'--train-file': str(data), '--epochs': '2', '--qat-static-graph': 'off', '--device': 'CUDA0,CUDA1', '--split-mode': 'tensor',
                   '--qat-adapter-out': str(adapter), '--qat-out': str(out / f'{model}-{version}.gguf')}
        for key, value in changes.items():
            cmd[cmd.index(key) + 1] = value
        cmd += ['--grad-accum', '6']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0,1', GGML_DISABLE_VULKAN='1')
        libs = '/tmp/aikar-parallel-accepted' if version == 'baseline' else str(root / 'build/bin')
        env['LD_LIBRARY_PATH'] = libs + ':/usr/local/cuda/lib64'
        log = out / f'{model}-{version}.log'
        if version != 'baseline' or not Path(str(adapter) + '.qat-state.gguf').exists():
            with log.open('w') as stream:
                result = subprocess.run(cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            assert result.returncode == 0, log
        text = log.read_text(errors='replace')
        steps = re.findall(r'step: step=(\d+)', text)
        losses = re.findall(r'data=.*?loss=([\d.e+-]+)', text)
        assert len(losses) > len(steps) >= 2, (log, len(losses), len(steps))
        tensors = {}
        for suffix in ('', '.qat-state.gguf'):
            reader = gguf.GGUFReader(str(adapter) + suffix)
            for tensor in reader.tensors:
                tensors[suffix + tensor.name] = tensor.data.copy()
        results.append((tensors, losses))
        report[model + '-' + version] = {'command': cmd, 'windows': len(losses), 'updates': len(steps)}
    a, b = results[0][0], results[1][0]
    assert a.keys() == b.keys()
    changed = [name for name in a if not np.array_equal(a[name], b[name])]
    report[model + '-comparison'] = {'total_tensors': len(a), 'changed': changed,
                                    'losses_equal': results[0][1] == results[1][1]}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert not changed and results[0][1] == results[1][1], report[model + '-comparison']
print(json.dumps(report, indent=2))
