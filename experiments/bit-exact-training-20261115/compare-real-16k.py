import json,importlib.util
from pathlib import Path
import gguf,numpy as np
root=Path('/tmp/aikar-bit-exact-16k')
spec=importlib.util.spec_from_file_location('integration','/home/user/aikar-engine/examples/qlora_training/test_qlion_integration.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
a=root/'tensor-smoke-full.gguf';b=root/'tensor-ub256.gguf'
r={'weights':m.compare_models(a,b),'optimizer_tensors':m.compare_models(Path(str(a)+'.qat-state.gguf'),root/'tensor-ub256.adapter.gguf.qat-state.gguf'),'ubatches':[768,256],'context':16384,'split_mode':'tensor'}
reader,tensors=m.tensor_map(Path(str(a)+'.qat-state.gguf'))
other_reader=gguf.GGUFReader(str(root/'tensor-ub256.adapter.gguf.qat-state.gguf'))
metadata=[k for k in reader.fields if k.startswith('training.')]
for key in metadata:
 assert np.array_equal(reader.fields[key].contents(),other_reader.fields[key].contents()),('state metadata differs',key)
r['optimizer_metadata_fields_equal']=metadata
r['momentum_norm_squared']=sum(float(np.sum(gguf.quants.dequantize(t.data,t.tensor_type).astype(np.float64)**2)) for name,t in tensors.items() if name.startswith('qat.m.'));assert r['momentum_norm_squared']>1e-20
cmd=json.loads((root/'tensor-command.json').read_text());targets=set(cmd[cmd.index('--qat-targets')+1].split(','))
base_reader,base=m.tensor_map(cmd[cmd.index('-m')+1]);trained_reader,trained=m.tensor_map(a)
assert base.keys()==trained.keys()
for name,t in base.items():
 if name not in targets:assert np.array_equal(t.data.view(np.uint8),trained[name].data.view(np.uint8)),('frozen weight changed',name)
r['frozen_weights_unchanged']=True
(root/'real-16k-bit-exact.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
