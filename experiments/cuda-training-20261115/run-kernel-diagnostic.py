from pathlib import Path
import json, os, subprocess, sys, time
out = Path(__file__).resolve().parent
name = sys.argv[1]
source = json.loads((out/'production-before.json').read_text())
command = json.loads((out/'cpu-profile-command.json').read_text())
command[command.index('--qat-adapter-out')+1] = '/tmp/aikar-'+name+'.adapter.gguf'
env = os.environ.copy()
env.update(source['environment'])
env['GGML_DISABLE_VULKAN'] = '1'
env.pop('NSYS_CUPTI_LIBRARY_PATH', None)
start = time.monotonic()
with (out/(name+'.log')).open('w') as log:
    process = subprocess.Popen(command, cwd=source['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT)
    (out/(name+'-active.json')).write_text(json.dumps({'pid':process.pid,'command':command}, indent=2))
    code = process.wait()
(out/(name+'-status.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.monotonic()-start}, indent=2))
raise SystemExit(code)
