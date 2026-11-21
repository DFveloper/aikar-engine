import subprocess,time,json
from pathlib import Path
out=Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
while not (out/'baseline-steady-status.json').exists():time.sleep(3)
assert json.loads((out/'baseline-steady-status.json').read_text())['exit_code']==0
while not (out/'small-acc-result.json').exists() or 'moe' not in json.loads((out/'small-acc-result.json').read_text()):time.sleep(3)
assert all(not x['changed'] and x['losses_equal'] for x in json.loads((out/'small-acc-result.json').read_text()).values())
subprocess.run(['python3',str(out/'run_probe.py'),'candidate-final','cpu','12','--peer','--acc','--p2p'],check=True)
