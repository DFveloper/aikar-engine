from pathlib import Path
import json, sqlite3, sys
out=Path(__file__).resolve().parent
results={}
for name in sys.argv[1:] or ('async-before','async-after'):
    path=Path('/tmp/aikar-training-'+name+'.sqlite')
    if not path.exists(): continue
    c=sqlite3.connect(path)
    first=c.execute("select min(k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where s.value like '%mul_mat_q<%' ").fetchone()[0]
    last=c.execute('select max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    result={'compute_interval_start_s':first/1e9,'compute_interval_end_s':last/1e9,'compute_interval_s':(last-first)/1e9,'devices':{},'cpu_cuda_waits':{}}
    for dev in (0,1):
        events=c.execute('select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? and end>? union all select start,end from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=? and end>? order by start',(dev,first,dev,first)).fetchall()
        merged=[]
        for start,end in events:
            start=max(first,start);end=min(last,end)
            if end<=start:continue
            if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
            else:merged.append([start,end])
        gaps=sorted([merged[i][0]-merged[i-1][1] for i in range(1,len(merged))],reverse=True)
        result['devices'][dev]={'gpu_active_fraction':sum(b-a for a,b in merged)/(last-first),'largest_idle_gaps_s':[x/1e9 for x in gaps[:8]],'idle_over_100ms_s':sum(x for x in gaps if x>1e8)/1e9}
    for api in ('cudaEventSynchronize','cudaStreamSynchronize'):
        count,ns=c.execute('select count(*),coalesce(sum(r.end-r.start),0) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like ?', (api+'%',)).fetchone()
        result['cpu_cuda_waits'][api]={'calls':count,'seconds':ns/1e9}
    results[name]=result
(out/'async-profile-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
