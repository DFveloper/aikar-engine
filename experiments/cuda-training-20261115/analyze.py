from pathlib import Path
import json,re,statistics,hashlib,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'gguf-py'))
import gguf
out=Path(__file__).resolve().parent
report={'target_hours':60,'train_windows':6057,'validation_windows':73,'production_args_changed':False,'runs':{}}
for name in ('baseline','optimized'):
    log=(out/(name+'.log')).read_text(errors='replace')
    times=re.findall(r'throughput:.*?elapsed_s=([0-9.]+)',log)
    validation=re.findall(r'validation:.*?loss=([0-9.]+)',log)
    losses=re.findall(r'loss: epoch=.*?loss=([0-9.]+)',log)
    report['runs'][name]={'train_seconds':float(times[-1]) if times else None,'training_loss':float(losses[-1]) if losses else None,'validation_loss':float(validation[-1]) if validation else None}
    status=out/(name+'-status.json')
    if status.exists():report['runs'][name].update(json.loads(status.read_text()))
    report['runs'][name]['kernel_trace']=name=='optimized'
    if times:report['runs'][name]['training_only_projection_hours']=float(times[-1])*6057/3600
base=report['runs']['baseline']['train_seconds'];opt=report['runs']['optimized']['train_seconds']
if base and opt:report['training_speedup']=base/opt
report['required_seconds_per_window_without_validation_or_saves']=60*3600/6057
for suffix,label in (('', 'weights'),('.qat-state.gguf','state')):
    paths=[Path('/tmp/aikar-cuda-'+n+'.adapter.gguf'+suffix) for n in ('baseline','optimized')]
    if not all(p.exists() for p in paths):continue
    readers=[gguf.GGUFReader(str(p)) for p in paths]
    tensors=[{t.name:t for t in r.tensors} for r in readers]
    metrics={}
    for key in tensors[0]:
        a,b=tensors[0][key],tensors[1][key]
        x=gguf.quants.dequantize(a.data,a.tensor_type).astype(np.float64).ravel()
        y=gguf.quants.dequantize(b.data,b.tensor_type).astype(np.float64).ravel()
        norm=float(x@x);other=float(y@y)
        metrics[key]={'bytes_equal':bool(np.array_equal(a.data,b.data)),'finite':bool(np.isfinite(y).all()),'relative_l2':float(np.linalg.norm(x-y)/max(np.sqrt(norm),1e-30)),'cosine':float(x@y/np.sqrt(norm*other)) if norm and other else float(norm==other)}
    report[label]=metrics
report['source_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('src/llama-graph.cpp','ggml/src/ggml.c','ggml/src/ggml-cuda/fattn-back.cu')}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('state','weights','source_sha256')},indent=2))
