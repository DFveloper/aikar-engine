from pathlib import Path
import json, os, subprocess, sys, time
out=Path(__file__).resolve().parent
name=sys.argv[1] if len(sys.argv)>1 else 'kernel-profile-before'
source=json.loads((out/'production-before.json').read_text())
command=json.loads((out/'cpu-profile-command.json').read_text())
command[command.index('--qat-adapter-out')+1]='/tmp/aikar-'+name+'.adapter.gguf'
nsys='/tmp/lumen-nsys-2025/extracted/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys'
env=os.environ.copy();env.update(source['environment'])
env.update(GGML_DISABLE_VULKAN='1',NSYS_CUPTI_LIBRARY_PATH='/tmp/lumen-cupti-12')
launch=[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o','/tmp/aikar-'+name,*command]
start=time.monotonic()
with (out/(name+'.log')).open('w') as log:
 p=subprocess.Popen(launch,cwd=source['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
 (out/(name+'-active.json')).write_text(json.dumps({'pid':p.pid,'command':command,'launch':launch},indent=2))
 code=p.wait()
(out/(name+'-status.json')).write_text(json.dumps({'exit_code':code,'wall_seconds':time.monotonic()-start},indent=2))
if code:raise SystemExit(code)
with (out/(name+'-export.log')).open('w') as log:
 subprocess.run([nsys,'export','--type=sqlite','--force-overwrite=true','-o','/tmp/aikar-'+name+'.sqlite','/tmp/aikar-'+name+'.nsys-rep'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
