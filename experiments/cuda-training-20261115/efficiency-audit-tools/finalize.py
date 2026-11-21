import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
scripts = Path(__file__).parent


def compare(baseline, candidate, output):
    with output.open('w') as log:
        return subprocess.run([sys.executable, str(scripts / 'compare_probes.py'), str(out), baseline, candidate], stdout=log).returncode == 0


while not (out / 'candidate-alloc-final-status.json').exists():
    time.sleep(3)
variant = 'candidate-alloc-final'
status = json.loads((out / (variant + '-status.json')).read_text())
if status['exit_code'] != 0 or not compare('baseline-steady', variant, out / 'allocator-comparison.json'):
    variant = 'candidate-safe'
print('Validated variant:', variant, flush=True)

env = dict(os.environ, GGML_DISABLE_VULKAN='1', GGML_CUDA_P2P='1', GGML_CUDA_ALLREDUCE='internal')
env.pop('GGML_TRAIN_HOST_CHECKPOINTS', None)
env.pop('GGML_TRAIN_ACC_REUSE', None)
base = out / 'alloc-clean' if variant == 'candidate-alloc-final' else out
env['LD_LIBRARY_PATH'] = f'{base}:{out}/peer:{out}/candidate:{root}/experiments/cuda-training-20261115/live-candidate-bin:{root}/build/bin'
if variant == 'candidate-alloc-final':
    env['GGML_TRAIN_ACC_ALLOC_INPLACE'] = '1'
else:
    env.pop('GGML_TRAIN_ACC_ALLOC_INPLACE', None)
with (out / 'unit-deployment.log').open('w') as log:
    code = subprocess.run([str(out / 'test-logit-distillation'), '--cuda'], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
assert code == 0, 'Deployment CUDA regression failed'
print('Deployment CUDA tests passed', flush=True)

probe = out / ('run_alloc-clean_probe.py' if variant == 'candidate-alloc-final' else 'run_probe.py')
code = subprocess.run([sys.executable, str(probe), 'candidate-deployment-profile', 'cpu', '2', '--peer', '--p2p', '--profile'], env=env).returncode
assert code == 0, 'Deployment profile failed'
assert compare('baseline-fixed', 'candidate-deployment-profile', out / 'deployment-profile-comparison.json'), 'Deployment profile changed losses or state'
print('Clean deployment build matches the reference model and optimizer', flush=True)
assert compare('baseline-steady', variant, out / 'steady-comparison.json')
subprocess.run([sys.executable, str(scripts / 'resume_validated.py'), variant], check=True)
print('Production launched; verify native progress and library maps', flush=True)
