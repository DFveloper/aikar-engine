import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user/aikar-engine')
OUT = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/optimization-60h')
sys.path.insert(0, str(ROOT / 'gguf-py'))
import gguf

ref = json.loads(Path('/tmp/aikar-final-accepted-integration/report.json').read_text())
args = ref['runs']['full-context-dense-qlion2-ub768']['command'].copy()
adapter = OUT / 'checkpoint-test.adapter.gguf'
for key, val in {'--device': 'CUDA2', '--qat-grad-accumulator': 'cpu', '--qat-static-graph': 'off', '--qat-adapter-out': str(adapter), '--qat-out': str(adapter)+'.unused', '--train-file': str(OUT / 'twelve-window.jsonl'), '--epochs': '1', '--save-every': '1'}.items():
    args[args.index(key)+1] = val
args[0] = str(OUT / 'llama-finetune-qlion')
args += ['--grad-accum', '6', '--distill-max-windows', '12']
env = os.environ.copy()
env.update(GGML_DISABLE_VULKAN='1', OMP_NUM_THREADS='2', LD_LIBRARY_PATH=f'{ROOT}/experiments/cuda-training-20261115/live-candidate-bin:{ROOT}/build/bin')
log = OUT / 'checkpoint-test.log'
backup = OUT / 'saved-six-window.adapter.gguf'
with log.open('w') as stream:
    proc = subprocess.Popen(args, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    copied = False
    while proc.poll() is None:
        if not copied and 'checkpoint set saved at step 1 window 6' in log.read_text():
            source = str(adapter)+'.step1.gguf'
            shutil.copy2(source, backup)
            shutil.copy2(source+'.qat-state.gguf', str(backup)+'.qat-state.gguf')
            copied = True
        time.sleep(.02)
assert proc.returncode == 0 and copied
report = {'checkpoint_windows': re.findall(r'checkpoint set saved at step (\d+) window (\d+)', log.read_text())}
assert report['checkpoint_windows'] == [('1', '6'), ('2', '12')]
reader = gguf.GGUFReader(str(backup)+'.qat-state.gguf')
assert int(reader.get_field('training.dataset.window').parts[-1][0]) == 6
merged = OUT / 'checkpoint-test-merged.gguf'
with (OUT / 'checkpoint-merge.log').open('w') as stream:
    subprocess.run([str(ROOT/'build/bin/llama-export-qlion'), '-m', args[args.index('-m')+1], '--adapter', str(backup), '-o', str(merged)], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
shutil.copy2(str(backup)+'.qat-state.gguf', str(merged)+'.qat-state.gguf')
resumed = OUT / 'checkpoint-resumed.adapter.gguf'
args2 = args.copy()
args2[args2.index('--qat-adapter-out')+1] = str(resumed)
args2 += ['--qat-resume', str(merged)]
with (OUT/'checkpoint-resumed.log').open('w') as stream:
    subprocess.run(args2, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
changed = []
count = 0
for suffix in ('', '.qat-state.gguf'):
    a = gguf.GGUFReader(str(adapter)+suffix)
    b = {t.name: t for t in gguf.GGUFReader(str(resumed)+suffix).tensors}
    assert {t.name for t in a.tensors} == b.keys()
    for tensor in a.tensors:
        count += 1
        if tensor.data.tobytes() != b[tensor.name].data.tobytes():
            changed.append(suffix+tensor.name)
assert not changed, changed
report.update(resume_final_tensors=count, different=changed, exact_resume=True)
(OUT/'checkpoint-regression.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
