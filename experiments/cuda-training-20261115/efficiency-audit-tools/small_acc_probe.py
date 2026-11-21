import json,os,subprocess,re,sys
from pathlib import Path
import numpy as np
root=Path('/home/user/aikar-engine');sys.path.insert(0,str(root/'gguf-py'));import gguf
out=Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit');old=out.parent/'optimization-60h';reference=json.loads((old/'long-accumulation-probe.json').read_text());report={}
variant=sys.argv[1] if len(sys.argv)>1 else 'acc'
for model in ['dense','moe']:
 name=variant+'-small-'+model;args=reference[model+'-cpu']['command'].copy()
 for key,value in {'--qat-adapter-out':str(out/(name+'.adapter.gguf')),'--qat-out':str(out/(name+'.unused')),'--distill-max-windows':'120'}.items():args[args.index(key)+1]=value
 env=dict(os.environ,GGML_DISABLE_VULKAN='1',GGML_TRAIN_ACC_REUSE='1',GGML_TRAIN_ACC_ALLOC_INPLACE='1',LD_LIBRARY_PATH=f'{out}/candidate:{out}/{variant}:{root}/experiments/cuda-training-20261115/live-candidate-bin:{root}/build/bin')
 with (out/(name+'.log')).open('w') as f:r=subprocess.run(args,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,name
 changed=[];total=0
 for suffix in ['', '.qat-state.gguf']:
  a=gguf.GGUFReader(str(old/('long-'+model+'-cpu.adapter.gguf'))+suffix);b=gguf.GGUFReader(str(out/(name+'.adapter.gguf'))+suffix)
  original={n.name:n.data for n in a.tensors}
  for n in b.tensors:
   total+=1
   if n.name not in original or not np.array_equal(original[n.name],n.data):changed.append(suffix+n.name)
 losses=re.findall(r'data=.*?loss=([\d.e+-]+)',(out/(name+'.log')).read_text())
 report[model]=dict(tensors=total,changed=changed,losses_equal=losses==reference[model+'-cpu']['losses'])
 print(model,report[model],flush=True)
 (out/('small-'+variant+'-result.json')).write_text(json.dumps(report,indent=2)+'\n')
 assert not changed and report[model]['losses_equal'],report[model]
