import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
run = Path('/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar')
checkpoint = out / 'production-step11.gguf'
assert checkpoint.stat().st_size == (out / 'production-step7.gguf').stat().st_size
assert Path(str(checkpoint) + '.qat-state.gguf').exists()
assert not (out / 'production-step11-resumed.json').exists()

script = (run / 'train.sh').read_text()
shutil.copy2(run / 'train.sh', out / 'train-before-step11.sh')
old = str(out / 'production-step7.gguf')
assert script.count(old) == 2
script = script.replace(old, str(checkpoint))
assert 'moe-tile' not in script and 'GGML_TRAIN_MOE' not in script
reviewed = out / 'train-step11-reviewed.sh'
reviewed.write_text(script)
subprocess.run(['bash', '-n', str(reviewed)], check=True)
shutil.copy2(reviewed, run / 'train.sh')
shutil.copy2(run / 'train.log', out / 'production-before-step11-resume.log')
env = dict(os.environ)
for key in ('LD_PRELOAD', 'GGML_TRAIN_MOE_TILE', 'GGML_TRAIN_MOE_PRESERVE_MM', 'LLAMA_TRAIN_PHASE_TIMING', 'AIKAR_ACTIVITY_TRACE', 'AIKAR_GRAPH_DUMP', 'AIKAR_SPLIT_TIMING'):
    env.pop(key, None)
with (out / 'step11-resume-launch.log').open('w') as log:
    process = subprocess.Popen(['bash', 'train.sh'], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(launcher_pid=process.pid, checkpoint=str(checkpoint), optimizer_step=11, next_window=66, power_limit_watts=[150, 150], status='launched_pending_verification', launched_at=time.time(), libraries=str(out / 'validated-release'))
(out / 'production-step11-resumed.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
