import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
sys.path.insert(0, str(root / 'gguf-py'))
import gguf

adapter = out / 'production-step11.adapter.gguf'
state = gguf.GGUFReader(str(adapter) + '.qat-state.gguf')
assert int(state.get_field('training.dataset.window').parts[-1][0]) == 66
checkpoint = out / 'production-step11.gguf'
assert not checkpoint.exists()
env = dict(os.environ, LD_LIBRARY_PATH=f'{out}/validated-release:{root}/build/bin')
with (out / 'step11-export.log').open('w') as log:
    subprocess.run([str(root / 'build/bin/llama-export-qlion'), '-m', str(out / 'production-step7.gguf'), '--adapter', str(adapter), '-o', str(checkpoint)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
shutil.copy2(str(adapter) + '.qat-state.gguf', str(checkpoint) + '.qat-state.gguf')
assert checkpoint.stat().st_size == (out / 'production-step7.gguf').stat().st_size
record = json.loads((out / 'next-capture.json').read_text())
pid = record['pid']
if Path(f'/proc/{pid}/cmdline').exists():
    assert b'llama-finetune' in Path(f'/proc/{pid}/cmdline').read_bytes()
    assert '\nState:\tT' in Path(f'/proc/{pid}/status').read_text()
    os.kill(pid, signal.SIGKILL)
record['status'] = 'exported_for_resume_gpu_released'
(out / 'next-capture.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
