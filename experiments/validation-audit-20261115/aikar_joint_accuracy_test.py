from pathlib import Path
import json,os,subprocess,re
repo=Path('/home/user/aikar-engine');out=repo/'experiments/validation-audit-20261115';root=Path('/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar');old=root/'logit-distillation/interleaved/runs/20261114T142347Z-3313753'
saved=json.loads((old/'command.json').read_text());cmd=saved['command'].copy();cmd[0]=str(repo/'build-validation-runtime/llama-finetune-qo')
for flag,value in [('--train-file','/tmp/aikar-validation-fixture/mixed.json'),('--distill-cache','/tmp/aikar-validation-fixture/mixed.json'),('--qat-adapter-out','/tmp/aikar-validation-fixture/joint-check.adapter.gguf'),('--save-every','1'),('--eval-every','1')]:cmd[cmd.index(flag)+1]=value
cmd+=['--eval-initial']
env={k:v for k,v in os.environ.items() if not k.startswith('QLION_')};env.update(saved['runtime']);env['LD_LIBRARY_PATH']=str(root/'.cache/ga16-runtime')+':'+str(repo/'build-validation-runtime')+':'+str(repo/'build-joint/bin')+':'+str(repo/'build/bin');env['QLION_DIAG_STOP_STEP']='1';env['QLION_JOINT_START_INITIAL']='1'
(out/'joint-accuracy.command.json').write_text(json.dumps(dict(command=cmd,runtime={k:v for k,v in env.items() if k.startswith(('QLION_','GGML_')) or k=='LD_LIBRARY_PATH'}),indent=2))
with (out/'joint-accuracy.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
print('joint test exit',r.returncode,flush=True)
text=(out/'joint-accuracy.log').read_text();rows=re.findall(r'validation: step=(\d+) epoch=\d+ loss=(\S+) acc=(\S+)%',text)
print('validation records',rows,flush=True)
assert r.returncode==0 and len(rows)>=2
serial=float(rows[0][2]);joint=float(rows[-1][2]);assert abs(serial-joint)<.1,(serial,joint)
assert 'supervised=' in text and 'joint_validation: completed' in text
print('PASS: serial and snapshot joint top-1 accuracy differ by less than 0.1 percentage point',flush=True)
