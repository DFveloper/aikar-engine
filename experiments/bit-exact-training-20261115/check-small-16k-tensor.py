import importlib.util,json,subprocess,re,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('integration','/home/user/aikar-engine/examples/qlora_training/test_qlion_integration.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path('/tmp/aikar-bit-exact-small-16k-tensor');root.mkdir(exist_ok=True);report={}
for family in ('dense','moe'):
 model='/mnt/openwebui/AIKAR/QLion2-test/artifacts/'+('gemma4-tiny-6l-q4.gguf' if family=='dense' else 'gemma4-tiny-moe-q4.gguf')
 reference=None
 for ub in (1024,768,512,256):
  name=f'{family}-ub{ub}';adapter=root/f'{name}.adapter.gguf';log=root/f'{name}.log'
  cmd=['/home/user/aikar-engine/build/bin/llama-finetune-qlion','-m',model,'--train-file','/tmp/aikar-bit-exact-small-16k/cache/mixed.json','--distill-cache','/tmp/aikar-bit-exact-small-16k/cache/mixed.json','-c','16384','-b','16384','-ub',str(ub),'-ngl','99','-dev','CUDA0,CUDA1','-sm','tensor','--qat-grad-accumulator','cpu','-fa','on','--fit','off','-t','2','-tb','2','--optimizer','qlion2','--quant-type','q4_0,q8_0','--qat-targets','.attn_q.weight,.attn_k.weight,.attn_v.weight,.attn_output.weight','--activation-recompute','on','--epochs','2','-lr','1e-6','--lr-scheduler','constant','--qat-adapter-only','--qat-adapter-out',str(adapter),'--save-every','0','--distill-max-windows','1','--verbose-loss','--verbosity','5']
  start=time.monotonic()
  with log.open('w') as out:p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
  assert p.returncode==0,(name,p.returncode,str(log))
  losses=re.findall(r'epoch:.*?loss=([0-9.e+-]+)',log.read_text());assert len(losses)==2,(name,losses)
  state=Path(str(adapter)+'.qat-state.gguf')
  assert any(int(n)>0 for n in re.findall(r'RAM F32 gradient reduction for (\d+)/',log.read_text()))
  import gguf,numpy as np
  _,tensors=m.tensor_map(state);norm=sum(float(np.sum(gguf.quants.dequantize(t.data,t.tensor_type).astype(np.float64)**2)) for n,t in tensors.items() if n.startswith('qat.m.'));assert norm>1e-20,(name,'missing momentum')
  if reference:
   m.compare_models(reference[0],adapter);m.compare_models(reference[1],state);assert reference[2]==losses
  else:reference=(adapter,state,losses)
  report[name]={'command':cmd,'loss':losses,'bit_exact':True,'seconds':time.monotonic()-start}
  (root/'report.json').write_text(json.dumps(report,indent=2));print(name,report[name]['seconds'],flush=True)
