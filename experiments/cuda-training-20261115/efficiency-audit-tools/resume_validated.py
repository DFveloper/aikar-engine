import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
run = Path('/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar')
comparison = json.loads((out / 'steady-comparison.json').read_text())
variant = sys.argv[1] if len(sys.argv) > 1 else 'candidate-safe'
assert variant in ('candidate-safe', 'candidate-alloc-final')
assert comparison['losses_equal'] and comparison['window_losses_equal']
assert not comparison['changed_tensors']
status = comparison[variant]['status']
assert status['exit_code'] == 0 or status.get('final_checkpoint_saved')
assert len(comparison[variant]['window_losses']) == 12
assert 'event reuse passed' in (out / 'unit-peer-final.log').read_text()
assert not (out / 'production-resumed.json').exists(), 'Production was already launched'

release = out / 'validated-release'
release.mkdir(exist_ok=True)
hashes = {}
allocator = variant == 'candidate-alloc-final'
llama_dir = out / 'candidate'
base_dir = out / 'alloc-clean' if allocator else out
sources = [llama_dir / 'libllama.so.0', out / 'peer/libggml-cuda.so.0', base_dir / 'libggml-base.so.0']
if allocator:
    sources.append(base_dir / 'libggml-cpu.so.0')
for source in sources:
    target = release / source.name
    shutil.copy2(source, target)
    hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()

checkpoint = out / 'production-step7.gguf'
assert checkpoint.stat().st_size > 10_000_000_000
assert Path(str(checkpoint) + '.qat-state.gguf').exists()
script = (run / 'train.sh').read_text()
shutil.copy2(run / 'train.sh', out / 'train-before-efficiency.sh')
script = script.replace(
    'export LD_LIBRARY_PATH="',
    'export GGML_CUDA_P2P=1\nexport GGML_CUDA_ALLREDUCE=internal\nexport LD_LIBRARY_PATH="' + str(release) + ':',
    1,
)
if allocator:
    script = script.replace('export GGML_CUDA_P2P=1', 'export GGML_TRAIN_ACC_ALLOC_INPLACE=1\nexport GGML_CUDA_P2P=1')
old_model = '/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/best-step136.gguf'
assert script.count('-m ' + old_model) == 1
script = script.replace('-m ' + old_model, '-m ' + shlex.quote(str(checkpoint)) + ' --qat-resume ' + shlex.quote(str(checkpoint)))
(out / 'train-resume-reviewed.sh').write_text(script)
subprocess.run(['bash', '-n', str(out / 'train-resume-reviewed.sh')], check=True)
shutil.copy2(out / 'train-resume-reviewed.sh', run / 'train.sh')
if (run / 'train.log').exists():
    shutil.copy2(run / 'train.log', out / 'production-before-efficiency.log')

env = dict(os.environ)
for key in ('LD_PRELOAD', 'AIKAR_ACTIVITY_TRACE', 'AIKAR_GRAPH_DUMP', 'AIKAR_SPLIT_TIMING', 'LLAMA_TRAIN_PHASE_TIMING', 'GGML_TRAIN_ACC_REUSE', 'GGML_TRAIN_HOST_CHECKPOINTS', 'GGML_TRAIN_ACC_ALLOC_INPLACE'):
    env.pop(key, None)
with (out / 'resume-launch.log').open('w') as log:
    process = subprocess.Popen(['bash', 'train.sh'], cwd=run, env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record = dict(launcher_pid=process.pid, checkpoint=str(checkpoint), optimizer_step=7,
              variant=variant,
              next_window=42, power_limit_watts=[150, 150], library_sha256=hashes,
              launched_at=time.time(), status='launched_pending_verification')
(out / 'production-resumed.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
