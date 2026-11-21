import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
variant = sys.argv[1] if len(sys.argv) > 1 else 'host'
dest = out / variant
dest.mkdir(exist_ok=True)
builder = Path(__file__).with_name('build_acc.py').read_text()
builder = builder.replace("efficiency-audit/acc'", "efficiency-audit/" + variant + "'")
subprocess.run([sys.executable, '-c', builder], check=True)
commands = json.loads((root / 'build/compile_commands.json').read_text())
entry = next(e for e in commands if '/Unity/' in e['file'] and 'models/gemma4.cpp' in Path(e['file']).read_text())
args = shlex.split(entry['command'])
args[args.index('-o') + 1] = str(dest / 'gemma4.o')
with (dest / 'gemma4.build.log').open('w') as log:
    subprocess.run(args, cwd=entry['directory'], stdout=log, stderr=subprocess.STDOUT, check=True)
args = shlex.split((root / 'build/src/CMakeFiles/llama.dir/link.txt').read_text())
overrides = {Path(entry['file']).name + '.o': dest / 'gemma4.o', 'llama-graph.cpp.o': out / 'llama-graph.o',
             'llama-context.cpp.o': out.parent / 'optimization-60h/llama-context-phase.o'}
args = [str(overrides.get(Path(a).name, a)) for a in args if not a.startswith('-Wl,--dependency-file=')]
args[args.index('-o') + 1] = str(dest / 'libllama.so.0')
with (dest / 'llama.link.log').open('w') as log:
    subprocess.run(args, cwd=root / 'build/src', stdout=log, stderr=subprocess.STDOUT, check=True)
print('Host checkpoint candidate linked', flush=True)
probe = (out / 'run_probe.py').read_text()
probe += "\n"
probe = probe.replace("start=time.monotonic()", "env['LD_LIBRARY_PATH']=str(out/'" + variant + "')+':'+env['LD_LIBRARY_PATH']\nenv['GGML_TRAIN_HOST_CHECKPOINTS']='1'\nstart=time.monotonic()")
(out / ('run_' + variant + '_probe.py')).write_text(probe)
