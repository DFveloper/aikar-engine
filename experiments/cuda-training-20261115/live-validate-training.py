import json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
root=Path('/home/user/aikar-engine');out=root/'experiments/cuda-training-20261115'
sys.path.insert(0,str(root/'gguf-py'));import gguf
reference=json.loads(Path('/tmp/aikar-final-accepted-integration/report.json').read_text())
results={}
for model in ('dense','moe'):
 states=[];losses=[]
 for version in ('before','after'):
  args=reference['runs'][f'full-context-{model}-qlion2-ub768']['command'].copy()
  adapter=out/f'live-train-{model}-{version}.adapter.gguf'
  for key,val in {'--device':'CUDA2','--qat-adapter-out':str(adapter),'--qat-out':str(out/f'live-train-{model}-{version}.gguf')}.items():args[args.index(key)+1]=val
  env=os.environ.copy();env.update(GGML_DISABLE_VULKAN='1',OMP_NUM_THREADS='2')
  env['LD_LIBRARY_PATH']=(str(out/'live-candidate-bin')+':' if version=='after' else '/tmp/aikar-live-cache/baseline:')+str(root/'build/bin')
  log=out/f'live-train-{model}-{version}.log'
  with log.open('w') as stream: proc=subprocess.run(args,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
  if proc.returncode:raise RuntimeError((model,version,proc.returncode,str(log)))
  tensor_state={}
  for suffix in ('','.qat-state.gguf'):
   for t in gguf.GGUFReader(str(adapter)+suffix).tensors:tensor_state[suffix+t.name]=t.data.copy()
  states.append(tensor_state)
  losses.append(re.findall(r'data=.*?loss=([\d.e+-]+)',log.read_text()))
  print(model,version,'finished',flush=True)
 a,b=states
 assert a.keys()==b.keys()
 changed=[k for k in a if a[k].dtype!=b[k].dtype or a[k].shape!=b[k].shape or a[k].tobytes()!=b[k].tobytes()]
 results[model]={'state_and_adapter_tensors':len(a),'changed':changed,'losses_equal':losses[0]==losses[1],'loss_count':len(losses[0])}
 (out/'live-training-validation.json').write_text(json.dumps(results,indent=2)+'\n')
 assert not changed and results[model]['losses_equal']
 print(model,'all tensors and losses byte-exact',flush=True)
