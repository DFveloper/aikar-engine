from pathlib import Path
import json, os, subprocess, time, sys
out=Path(__file__).resolve().parent
source=json.loads((out/'production-before.json').read_text())
version=sys.argv[1]
cmd=source['command'].copy()
cmd[cmd.index('--qat-adapter-out')+1]='/tmp/aikar-cuda-'+version+'.adapter.gguf'
cmd[cmd.index('--warmup-steps')+1]='0'
cmd+=['--distill-max-windows','1','--distill-validation-windows','1']
env=os.environ.copy(); env.update(source['environment'])
env['LD_LIBRARY_PATH']=('/tmp/aikar-uncropped:' if version=='uncropped' else '/tmp/aikar-async-baseline:' if version=='async-before' else '')+'/home/user/aikar-engine/build/bin'
env['GGML_DISABLE_VULKAN']='1'
start=time.monotonic()
with (out/(version+'.log')).open('w') as log:
    proc=subprocess.Popen(cmd,cwd=source['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/(version+'-active.json')).write_text(json.dumps({'pid':proc.pid,'command':cmd,'environment':{'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']}}))
    code=proc.wait()
(out/(version+'-status.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.monotonic()-start}))
raise SystemExit(code)
