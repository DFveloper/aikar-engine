import importlib.util,json,re,subprocess
from pathlib import Path
spec=importlib.util.spec_from_file_location('integration','/home/user/aikar-engine/examples/qlora_training/test_qlion_integration.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path('/tmp/aikar-bit-exact-kd-offload-tensor');root.mkdir(exist_ok=True);report={}
for family,model,targets in [('dense','/mnt/openwebui/AIKAR/QLion2-test/artifacts/gemma4-tiny-6l-q4.gguf','.attn_q.weight,.attn_k.weight,.attn_v.weight,.attn_output.weight'),('lumen','/tmp/aikar-full-context-lumen/lumen4exp-12l.gguf','blk.5.attn_k.weight')]:
 runs=[]
 for ub in (1024,768,256):
  adapter=root/f'{family}-ub{ub}.adapter.gguf';log=root/f'{family}-ub{ub}.log';data=Path('/tmp/aikar-bit-exact-kd')/family/'mixed.json'
  cmd=['/home/user/aikar-engine/build/bin/llama-finetune-qlion','-m',model,'--train-file',str(data),'--distill-cache',str(data),'-c','1024','-b','1024','-ub',str(ub),'-ngl','99','-dev','CUDA0,CUDA1','-sm','tensor','--no-op-offload','-fa','on','-t','4','-tb','4','--optimizer','qlion2','--quant-type','q4_0,q8_0','--qat-targets',targets,'--activation-recompute','on','--grad-accum','3','--qat-grad-accumulator','cpu','--epochs','2','-lr','1e-6','--lr-scheduler','constant','--qat-adapter-only','--qat-adapter-out',str(adapter),'--save-every','0','--verbose-loss','--verbosity','5']
  with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,(family,ub,p.returncode,str(log))
  text=log.read_text();losses=re.findall(r'epoch:.*?loss=([0-9.e+-]+)',text)
  assert len(losses)==2,(family,ub,losses)
  _,states=module.tensor_map(Path(str(adapter)+'.qat-state.gguf'));import gguf,numpy as np
  norm=sum(float(np.sum(gguf.quants.dequantize(t.data,t.tensor_type).astype(np.float64)**2)) for name,t in states.items() if name.startswith('qat.m.'));assert norm>1e-20,(family,'missing momentum')
  runs.append((adapter,losses))
  report[f'{family}-ub{ub}']={'command':cmd,'loss':losses,'log':str(log)}
 for adapter,losses in runs[1:]:
  assert losses==runs[0][1],family
  module.compare_models(runs[0][0],adapter)
  module.compare_models(Path(str(runs[0][0])+'.qat-state.gguf'),Path(str(adapter)+'.qat-state.gguf'))
 report[family+'-bit-exact']=True
 (root/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
