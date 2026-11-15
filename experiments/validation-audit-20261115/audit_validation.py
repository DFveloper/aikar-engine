import json,struct,hashlib
from pathlib import Path
import numpy as np
root=Path('/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar')
manifest=json.loads((root/'logit-distillation/interleaved/runs/20261112T001619Z-476684/mixed.json').read_text())
report=[]; samples=[]; train_hash=set(); val_hash=[]
for entry in manifest['caches']:
 p=Path(entry['path']); f=p.open('rb')
 magic,c,v,k,temp,tok,n,nt,ml=struct.unpack('<8sIIIfQQQI',f.read(52)); meta=f.read(ml).decode()
 assert magic==b'QLKD0001' and n==entry['windows'] and nt==entry['train_windows']
 header=dict(context=c,vocab=v,k=k,temperature=temp,tokenizer=tok,windows=n,train=nt,metadata=meta)
 stats={part:dict(windows=0,supervised=0,shift_checked=0,shift_mismatch=0,invalid_ids=0,empty_windows=0,sampled_rows=0,teacher_top1_correct=0,target_in_topk=0,bad_teacher_rows=0,max_mass_error=0.,density=[]) for part in ('train','validation')}
 train_pick=set(np.linspace(0,nt-1,min(32,nt),dtype=int).tolist()); rb=k*6+4
 for w in range(n):
  tokens=np.frombuffer(f.read(c*4),dtype='<i4'); labels=np.frombuffer(f.read(c*4),dtype='<i4'); positions=np.flatnonzero(labels>=0); rows=len(positions); offset=f.tell()
  part='train' if w<nt else 'validation'; st=stats[part];st['windows']+=1;st['supervised']+=rows;st['empty_windows']+=rows==0;st['density'].append(rows/c)
  st['invalid_ids']+=int(np.count_nonzero((tokens<0)|(tokens>=v)))+int(np.count_nonzero((labels < -1)|(labels>=v)))
  active=labels[:-1]>=0;st['shift_checked']+=int(active.sum());st['shift_mismatch']+=int(np.count_nonzero(labels[:-1][active]!=tokens[1:][active]))
  h=hashlib.sha256(tokens.tobytes()+labels.tobytes()).hexdigest()
  if part=='train':train_hash.add(h)
  else:val_hash.append((entry['name'],w,h))
  if part=='validation' or w in train_pick:
   for ri in np.unique(np.linspace(0,rows-1,min(rows,12),dtype=int)) if rows else []:
    f.seek(offset+int(ri)*rb); raw=f.read(rb);ids=np.frombuffer(raw[:k*4],dtype='<i4');logp=np.frombuffer(raw[k*4:k*6],dtype='<f2').astype(float);tail=struct.unpack('<f',raw[k*6:])[0]
    bad=not(np.all((ids>=0)&(ids<v)) and len(np.unique(ids))==k and np.isfinite(logp).all() and np.isfinite(tail) and np.all(logp<=0) and tail<=0)
    st['sampled_rows']+=1;st['bad_teacher_rows']+=int(bad)
    mass=abs(float(np.exp(logp).sum()+np.exp(tail))-1);st['max_mass_error']=max(st['max_mass_error'],mass)
    target=int(labels[positions[ri]]);st['teacher_top1_correct']+=int(ids[0]==target);st['target_in_topk']+=int(target in ids)
   if part=='validation' and len([x for x in samples if x['source']==entry['name']])<2:
    samples.append(dict(source=entry['name'],window=w,tokens=tokens[:900].tolist(),supervised_positions=positions[:25].tolist(),labels=labels[positions[:25]].tolist()))
  f.seek(offset+rows*rb)
 assert f.tell()==p.stat().st_size
 for st in stats.values():
  density=st.pop('density'); st['supervised_density_mean']=float(np.mean(density)); st['density_range']=[float(min(density)),float(max(density))]
  st['teacher_top1_accuracy_sample']=st['teacher_top1_correct']/max(1,st['sampled_rows']); st['target_topk_rate_sample']=st['target_in_topk']/max(1,st['sampled_rows'])
 report.append(dict(source=entry['name'],path=str(p),header=header,stats=stats))
 print(entry['name'],json.dumps(stats),flush=True)
out=dict(sources=report,exact_train_validation_duplicates=[dict(source=s,window=w) for s,w,h in val_hash if h in train_hash],validation_duplicate_count=len(val_hash)-len(set(h for _,_,h in val_hash)))
Path('experiments/validation-audit-20261115/cache-audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
Path('experiments/validation-audit-20261115/token-samples.json').write_text(json.dumps(samples,ensure_ascii=False))
