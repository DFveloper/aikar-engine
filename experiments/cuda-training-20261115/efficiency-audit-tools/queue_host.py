import json
from pathlib import Path
import subprocess
import time

out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
while not (out / 'candidate-safe-status.json').exists():
    time.sleep(3)
assert json.loads((out / 'candidate-safe-status.json').read_text())['exit_code'] == 0
subprocess.run(['python3', str(out / 'run_host_probe.py'), 'candidate-host', 'cpu', '2', '--peer', '--p2p', '--profile'], check=True)
