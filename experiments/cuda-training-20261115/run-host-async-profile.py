from pathlib import Path
import json, os, subprocess, time
out = Path(__file__).resolve().parent
source = json.loads((out / 'production-before.json').read_text())
command = json.loads((out / 'cpu-profile-command.json').read_text())
command[command.index('--qat-adapter-out') + 1] = '/tmp/aikar-host-async.adapter.gguf'
nsys = '/tmp/lumen-nsys-2025/extracted/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys'
env = os.environ.copy()
env.update(source['environment'])
env.update(GGML_DISABLE_VULKAN='1', NSYS_CUPTI_LIBRARY_PATH='/tmp/lumen-cupti-12')
launch = [nsys, 'profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
          '--force-overwrite=true', '-o', '/tmp/aikar-host-async', *command]
start = time.monotonic()
with (out / 'host-async.log').open('w') as log:
    process = subprocess.Popen(launch, cwd=source['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / 'host-async-active.json').write_text(json.dumps({'pid': process.pid, 'command': command, 'launch': launch}, indent=2))
    code = process.wait()
(out / 'host-async-status.json').write_text(json.dumps({'exit_code': code, 'wall_seconds': time.monotonic() - start}, indent=2))
if code:
    raise SystemExit(code)
with (out / 'host-async-export.log').open('w') as log:
    subprocess.run([nsys, 'export', '--type=sqlite', '--force-overwrite=true',
                    '-o', '/tmp/aikar-host-async.sqlite', '/tmp/aikar-host-async.nsys-rep'],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
