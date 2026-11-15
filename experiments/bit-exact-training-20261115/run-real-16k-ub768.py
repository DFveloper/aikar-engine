import json,subprocess,resource,time,ctypes
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path('/tmp/aikar-bit-exact-16k')
a=json.loads((root/'tensor-command.json').read_text())
class Memory(ctypes.Structure):
 _fields_=[('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
n=ctypes.CDLL('libnvidia-ml.so.1');assert n.nvmlInit_v2()==0
handles=[]
for i in (0,1):
 h=ctypes.c_void_p();assert n.nvmlDeviceGetHandleByIndex_v2(i,ctypes.byref(h))==0;handles.append(h)
r={'samples':0,'peak_device_bytes':[0,0],'peak_rss_kib':0,'started_at_unix':time.time()}
with (root/'train-tensor.log').open('w') as log:
 p=subprocess.Popen(a,stdout=log,stderr=subprocess.STDOUT);r['pid']=p.pid
 while p.poll() is None:
  status=Path(f'/proc/{p.pid}/status')
  if status.exists():
   try:
    lines=status.read_text().splitlines()
    for key,target in [('VmRSS:','peak_rss_kib'),('VmHWM:','peak_hwm_kib')]:
     v=next((int(x.split()[1]) for x in lines if x.startswith(key)),0);r[target]=max(r.get(target,0),v)
   except FileNotFoundError:pass
  for i,h in enumerate(handles):
   m=Memory();assert n.nvmlDeviceGetMemoryInfo(h,ctypes.byref(m))==0;r['peak_device_bytes'][i]=max(r['peak_device_bytes'][i],int(m.used))
  r['samples']+=1;(root/'memory-tensor.json').write_text(json.dumps(r,indent=2));time.sleep(1)
r['exit_code']=p.returncode;r['elapsed_seconds']=time.time()-r['started_at_unix']
(root/'memory-tensor.json').write_text(json.dumps(r,indent=2));raise SystemExit(p.returncode)
