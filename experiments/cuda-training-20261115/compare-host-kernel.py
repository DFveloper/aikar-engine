from pathlib import Path
import json, re, sys
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root/'gguf-py'))
import gguf
name = sys.argv[1] if len(sys.argv) > 1 else 'host-kernel-dual'
out = Path(__file__).resolve().parent
readers = [gguf.GGUFReader('/tmp/aikar-'+n+'.adapter.gguf.qat-state.gguf') for n in ('cpu-profile', name)]
a,b = [{t.name:t.data for t in r.tensors} for r in readers]
assert a.keys() == b.keys()
changed = [n for n in a if a[n].dtype != b[n].dtype or a[n].shape != b[n].shape or a[n].tobytes() != b[n].tobytes()]
logs = [(out/(n+'.log')).read_text(errors='replace') for n in ('cpu-profile',name)]
times = [float(re.findall(r'throughput:.*?elapsed_s=([\d.]+)',s)[-1]) for s in logs]
losses = [re.findall(r'data=.*?loss=([\d.]+)',s) for s in logs]
validation = [re.findall(r'validation:.*?loss=([\d.]+)',s) for s in logs]
report = {'state_tensors':len(a),'changed':changed,'state_bytes_equal':not changed,
          'train_seconds_before':times[0],'train_seconds_after':times[1],
          'speedup':times[0]/times[1], 'training_losses_equal':losses[0]==losses[1],
          'validation_losses_equal':validation[0]==validation[1], 'production_args_changed':False,
          'required_seconds_per_window':60*3600/6057}
(out/('results-'+name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert not changed and report['training_losses_equal'] and report['validation_losses_equal']
