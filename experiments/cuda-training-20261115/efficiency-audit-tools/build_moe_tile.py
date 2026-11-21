import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
name = sys.argv[1] if len(sys.argv) > 1 else 'moe-tile'
assert name in ('moe-tile', 'moe-preserve')
dest = out / name
dest.mkdir(exist_ok=True)
commands = json.loads((root / 'build/compile_commands.json').read_text())
entry = next(e for e in commands if e['file'] == str(root / 'src/llama-graph.cpp'))
args = shlex.split(entry['command'])
args[args.index('-o') + 1] = str(dest / 'llama-graph.o')
with (dest / 'graph.build.log').open('w') as log:
    subprocess.run(args, cwd=entry['directory'], stdout=log, stderr=subprocess.STDOUT, check=True)
args = shlex.split((root / 'build/src/CMakeFiles/llama.dir/link.txt').read_text())
overrides = {'llama-graph.cpp.o': dest / 'llama-graph.o',
             'llama-context.cpp.o': out.parent / 'optimization-60h/llama-context-phase.o'}
args = [str(overrides.get(Path(a).name, a)) for a in args if not a.startswith('-Wl,--dependency-file=')]
args[args.index('-o') + 1] = str(dest / 'libllama.so.0')
with (dest / 'llama.link.log').open('w') as log:
    subprocess.run(args, cwd=root / 'build/src', stdout=log, stderr=subprocess.STDOUT, check=True)
probe = (out / 'run_alloc-clean_probe.py').read_text()
probe = probe.replace('start=time.monotonic()', f"env['LD_LIBRARY_PATH']=str(out/'{name}')+':'+env['LD_LIBRARY_PATH']\nstart=time.monotonic()")
(out / f'run_{name.replace("-", "_")}_probe.py').write_text(probe)
print('MoE tile candidate linked', flush=True)
