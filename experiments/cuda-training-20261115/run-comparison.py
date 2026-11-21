from pathlib import Path
import json,subprocess,os,time
out=Path(__file__).resolve().parent
source=json.loads((out/'production-before.json').read_text())
nsys='/tmp/lumen-nsys-2025/extracted/opt/nvidia/nsight-systems/2025.5.2/target-linux-x64/nsys'
import sys
for version in (sys.argv[1:] or ['baseline','optimized']):
    cmd=source['command'].copy()
    lib='/tmp/aikar-cuda-training-baseline' if version=='baseline' else '/tmp/aikar-async-baseline' if version=='async-before' else '/home/user/aikar-engine/build/bin'
    cmd[0]=lib+'/llama-finetune-qlion'
    cmd[cmd.index('--qat-adapter-out')+1]='/tmp/aikar-cuda-'+version+'.adapter.gguf'
    cmd[cmd.index('--warmup-steps')+1]='0'
    cmd+=['--distill-max-windows','1','--distill-validation-windows','1']
    env=os.environ.copy();env.update(source['environment']);env['LD_LIBRARY_PATH']=lib+':/home/user/aikar-engine/build/bin';env['GGML_DISABLE_VULKAN']='1';env['NSYS_CUPTI_LIBRARY_PATH']='/tmp/lumen-cupti-12'
    profile=[nsys,'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=true','-o','/tmp/aikar-training-'+version,*cmd]
    (out/(version+'-command.json')).write_text(json.dumps(cmd,indent=2))
    started=time.monotonic()
    with (out/(version+'.log')).open('w') as log:
        proc=subprocess.Popen(profile,cwd=source['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
        (out/'comparison-active.json').write_text(json.dumps(dict(version=version,pid=proc.pid)))
        code=proc.wait()
    (out/(version+'-status.json')).write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-started),indent=2))
    print(version,code,flush=True)
    if code:raise SystemExit(code)
