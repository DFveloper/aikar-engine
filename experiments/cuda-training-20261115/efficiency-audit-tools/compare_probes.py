import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'gguf-py'))
import gguf


def describe(directory, name):
    log = (directory / (name + '.log')).read_text()
    phases = [dict(zip(('build', 'alloc', 'inputs', 'eval', 'total'), map(int, row)))
              for row in re.findall(r'build=(\d+)ms alloc=(\d+)ms inputs=(\d+)ms eval=(\d+)ms total=(\d+)ms', log)]
    losses = re.findall(r'I loss:.*?loss=([\d.e+-]+)', log)
    window_losses = re.findall(r'data=\d+/\d+ loss=([\d.e+-]+)', log)[:len(phases)]
    elapsed = re.findall(r'data=\d+/\d+ loss=\S+ acc=\S+ t=(\d+):(\d+):(\d+)', log)[:len(phases)]
    train_seconds = sum(int(v)*scale for v, scale in zip(elapsed[-1], (3600, 60, 1)))
    peaks = {}
    with (directory / (name + '-gpu.csv')).open() as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                continue
            gpu, memory = int(row[1]), int(row[2])
            peaks[gpu] = max(peaks.get(gpu, 0), memory)
    workspace = re.findall(r'compute_workspace=(\d+)', log)
    status_path = directory / (name + '-status.json')
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        'exit_code': None,
        'final_checkpoint_saved': 'adapter_checkpoint: saved final path=' in log,
        'note': 'Runner exit status was not captured; completion is checked from the checkpoint and log.',
    }
    return {
        'status': status,
        'losses': losses,
        'window_losses': window_losses,
        'train_elapsed_seconds': train_seconds,
        'train_seconds_per_window': train_seconds / len(phases),
        'phase_ms': phases,
        'phase_mean_seconds': {key: sum(p[key] for p in phases) / len(phases) / 1000 for key in phases[0]},
        'peak_device_mib': peaks,
        'workspace_bytes': int(workspace[-1]),
    }


directory = Path(sys.argv[1])
names = sys.argv[2:4]
report = {name: describe(directory, name) for name in names}
report['losses_equal'] = report[names[0]]['losses'] == report[names[1]]['losses']
report['window_losses_equal'] = report[names[0]]['window_losses'] == report[names[1]]['window_losses']
changed = []
count = 0
for suffix in ('', '.qat-state.gguf'):
    readers = [gguf.GGUFReader(str(directory / (name + '.adapter.gguf')) + suffix) for name in names]
    tensors = [{t.name: t for t in reader.tensors} for reader in readers]
    for name in sorted(tensors[0].keys() | tensors[1].keys()):
        count += 1
        if name not in tensors[0] or name not in tensors[1]:
            changed.append(suffix + ':' + name)
            continue
        left, right = tensors[0][name], tensors[1][name]
        if left.tensor_type != right.tensor_type or not np.array_equal(left.shape, right.shape) or not np.array_equal(left.data.view(np.uint8), right.data.view(np.uint8)):
            changed.append(suffix + ':' + name)
report['state_tensors_compared'] = count
report['changed_tensors'] = changed
print(json.dumps(report, indent=2))
sys.exit(bool(changed) or not report['losses_equal'] or not report['window_losses_equal'])
