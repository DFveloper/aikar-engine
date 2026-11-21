import collections,json,sys
from pathlib import Path
path=Path(sys.argv[1]); kernels=collections.defaultdict(lambda:[0,0]); copies=collections.defaultdict(lambda:[0,0,0]); apis=collections.defaultdict(lambda:[0,0]); spans=collections.defaultdict(list)
for line in path.open():
 p=line.rstrip().split('\t')
 try:
  kind=p[0]; start,end=int(p[3]),int(p[4]); dt=(end-start)/1e9
  if end<start or not start:continue
  if kind=='K':
   key=p[5];kernels[key][0]+=1;kernels[key][1]+=dt;spans[int(p[1])].append((start,end))
  elif kind=='M':
   key=p[6];copies[key][0]+=1;copies[key][1]+=int(p[5]);copies[key][2]+=dt;spans[int(p[1])].append((start,end))
  elif kind=='A':apis[p[1]][0]+=1;apis[p[1]][1]+=dt
 except (ValueError,IndexError):pass

def union(intervals):
 out=[]
 for start,end in sorted(intervals):
  if out and start<=out[-1][1]:out[-1][1]=max(out[-1][1],end)
  else:out.append([start,end])
 return out

def stats(intervals):
 merged=union(intervals)
 if not merged:return {}
 busy=sum(b-a for a,b in merged)/1e9;elapsed=(merged[-1][1]-merged[0][0])/1e9
 gaps=sorted([[(merged[i][0]-merged[i-1][1])/1e9,merged[i-1][1],merged[i][0]] for i in range(1,len(merged))],reverse=True)
 return dict(busy_seconds=busy,elapsed_seconds=elapsed,busy_fraction=busy/elapsed,largest_gaps=gaps[:25])
result=dict(kernels=sorted([dict(name=k,count=v[0],seconds=v[1]) for k,v in kernels.items()],key=lambda x:-x['seconds'])[:50],copies=dict(copies),runtime=sorted([dict(id=k,count=v[0],seconds=v[1]) for k,v in apis.items()],key=lambda x:-x['seconds'])[:20],devices={k:stats(v) for k,v in spans.items()},any_gpu=stats([x for values in spans.values() for x in values]))
print(json.dumps(result,indent=2))
