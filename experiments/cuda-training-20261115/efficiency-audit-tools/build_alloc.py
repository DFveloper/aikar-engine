import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path('/home/user/aikar-engine')
out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
variant = sys.argv[1] if len(sys.argv) > 1 else 'alloc'
dest = out / variant
builder = Path(__file__).with_name('build_acc.py').read_text()
builder = builder.replace("efficiency-audit/acc'", "efficiency-audit/" + variant + "'")
subprocess.run([sys.executable, '-c', builder], check=True)
commands = json.loads((root / 'build/compile_commands.json').read_text())
for source in ('ggml/src/ggml-alloc.c', 'ggml/src/ggml-cpu/ops.cpp'):
    entry = next(e for e in commands if e['file'] == str(root / source))
    args = shlex.split(entry['command'])
    args[args.index('-o') + 1] = str(dest / (Path(source).name + '.o'))
    with (dest / (Path(source).name + '.build.log')).open('w') as log:
        subprocess.run(args, cwd=entry['directory'], stdout=log, stderr=subprocess.STDOUT, check=True)
for target in ('ggml-base', 'ggml-cpu'):
    args = shlex.split((root / ('build/ggml/src/CMakeFiles/' + target + '.dir/link.txt')).read_text())
    overrides = {name: dest / name for name in ('ggml-alloc.c.o', 'ops.cpp.o', 'ggml-opt.cpp.o', 'ggml-backend.cpp.o')}
    overrides['ggml-backend-meta.cpp.o'] = Path('/tmp/aikar-live-cache/candidate/meta.o')
    args = [str(overrides.get(Path(a).name, a)) for a in args if not a.startswith('-Wl,--dependency-file=')]
    args[args.index('-o') + 1] = str(dest / ('lib' + target + '.so.0'))
    with (dest / (target + '.link.log')).open('w') as log:
        subprocess.run(args, cwd=root / 'build/ggml/src', stdout=log, stderr=subprocess.STDOUT, check=True)
probe = (out / 'run_probe.py').read_text()
probe = probe.replace('start=time.monotonic()', "env['LD_LIBRARY_PATH']=str(out/'" + variant + "')+':'+env['LD_LIBRARY_PATH']\nenv['GGML_TRAIN_ACC_ALLOC_INPLACE']='1'\nstart=time.monotonic()")
(out / ('run_' + variant + '_probe.py')).write_text(probe)
print('Allocator candidate linked', flush=True)
