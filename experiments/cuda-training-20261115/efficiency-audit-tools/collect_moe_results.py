import json
from pathlib import Path
import subprocess
import sys

root = Path('/home/user/aikar-engine')
directory = root / 'experiments/cuda-training-20261115/efficiency-audit-tools'
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
names = ('candidate-moe-512', 'candidate-moe-1024', 'candidate-moe-preserve-1024', 'candidate-moe-512-acc-absent')
results = {}
for name in names:
    process = subprocess.run([sys.executable, str(directory / 'compare_probes.py'), str(out), 'candidate-deployment-profile', name], capture_output=True, text=True)
    assert process.returncode in (0, 1), process.stderr
    comparison = json.loads(process.stdout)
    assert comparison[name]['status']['exit_code'] == 0
    results[name] = comparison
    (directory / (name + '-comparison.json')).write_text(json.dumps(comparison, indent=2) + '\n')
report = dict(target_seconds_per_window=216000/6057, validated_seconds_per_window=664/12, accepted=[], comparisons=results,
              notes=['The two-window baseline comparison was profiled; candidate timings are diagnostic, not sustained speedups.',
                     'candidate-moe-512-noacc-isolated set ACC=0, which still enables ACC by variable presence. Exclude it as an ACC ablation.',
                     'candidate-moe-512-noacc failed during an accidentally overlapping launch; exclude it from performance and numerical conclusions.'])
for name, comparison in results.items():
    if comparison['losses_equal'] and comparison['window_losses_equal'] and not comparison['changed_tensors']:
        report['accepted'].append(name)
(directory / 'moe-tile-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({name: dict(losses_equal=value['window_losses_equal'], changed_tensors=value['changed_tensors']) for name, value in results.items()}, indent=2))
