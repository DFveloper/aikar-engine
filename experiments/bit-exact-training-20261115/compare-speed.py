import json,os,subprocess,time,re,statistics
from pathlib import Path
root=Path('/tmp/aikar-bit-exact-speed');root.mkdir(exist_ok=True)
r=json.loads(Path('/tmp/aikar-bit-exact-replay-gpu/report.json').read_text());report={}
for family in ('dense','moe'):
 key=f'full-context-{family}-qlion-ub512'
 base=r['runs'][key]['command']
 for version in ('baseline','current'):
  cmd=base.copy();cmd[0]='/home/user/aikar-engine/'+('build-joint' if version=='baseline' else 'build')+'/bin/llama-finetune-qlion'
  for flag in ('--qat-out','--qat-adapter-out'):
   cmd[cmd.index(flag)+1]=str(root/f'{family}-{version}-{flag[2:]}.gguf')
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
  if version=='baseline':env['LD_LIBRARY_PATH']='/home/user/aikar-engine/build-joint/bin:/home/user/aikar-engine/build-joint/ggml-system/lib'
  log=root/f'{family}-{version}.log';start=time.monotonic()
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
  assert p.returncode==0,(family,version,p.returncode,str(log))
  text=log.read_text(); stamps=[int(a)*60+int(b)+int(c)/1000+int(d)/1000000 for a,b,c,d in re.findall(r'(\d+)\.(\d+)\.(\d+)\.(\d+) I (?:qat_)?epoch: ',text)]; times=[b-a for a,b in zip(stamps,stamps[1:])][-12:]
  report[f'{family}-{version}']={'command':cmd,'wall_seconds':time.monotonic()-start,'epoch_times':times,'median_step_seconds':statistics.median(times) if times else None,'log':str(log)}
  (root/'report.json').write_text(json.dumps(report,indent=2));print(family,version,report[f'{family}-{version}'],flush=True)
