from pathlib import Path
import datetime, json, os, signal, subprocess, sys, time
out = Path(__file__).resolve().parent
source = json.loads((out/'production-before.json').read_text())
exe = source['command'][0]
def trainers():
    found = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().split(b'\0')
            if args[0].decode() == exe:
                found[int(entry.name)] = [x.decode() for x in args if x]
        except (OSError, UnicodeError):
            pass
    return found
assert not trainers(), 'A trainer is already running'
validation_name = sys.argv[1] if len(sys.argv) > 1 else 'host-kernel-dual'
validation = json.loads((out/('results-'+validation_name+'.json')).read_text())
if '--allow-floating-point-drift' in sys.argv:
    assert validation['floating_point_drift_authorized'] and validation['gradient_validation_passed']
    assert validation['training_losses_equal'] and validation['validation_losses_equal']
else:
    assert validation['state_bytes_equal'] and validation['training_losses_equal'] and validation['validation_losses_equal']
sample_command = ['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used','--format=csv,noheader,nounits','--loop-ms=200']
sample_path = out/'production-host-kernel-gpu.csv'
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        args = [os.fsdecode(arg) for arg in entry.joinpath('cmdline').read_bytes().split(b'\0') if arg]
        if args == sample_command and os.readlink(entry/'fd/1') == str(sample_path):
            os.kill(int(entry.name), signal.SIGTERM)
    except (OSError, UnicodeError):
        pass
env = os.environ.copy()
env.update(source['environment'])
for key in list(env):
    if key.startswith('AIKAR_PROFILE_') or key in ('CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_ALLREDUCE','NSYS_CUPTI_LIBRARY_PATH','GGML_DISABLE_VULKAN'):
        env.pop(key)
with (out/'production-host-kernel-launch.log').open('w') as log:
    launch = subprocess.Popen(['bash',source['cwd']+'/train.sh'],cwd=source['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with sample_path.open('w') as log:
    sampler = subprocess.Popen(sample_command,stdout=log,stderr=subprocess.DEVNULL,start_new_session=True)
record = {'launcher_pid':launch.pid,'telemetry_pid':sampler.pid,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cwd':source['cwd'],'original_script':source['cwd']+'/train.sh','production_args_changed':False,'validated_build':validation_name,'validation':validation}
for _ in range(50):
    running = trainers()
    if running:
        assert len(running) == 1
        pid,command = next(iter(running.items()))
        record.update(trainer_pid=pid,command=command,args_match=command==source['command'])
        assert record['args_match'], command
        break
    assert launch.poll() is None, 'Launcher exited'
    time.sleep(.2)
assert 'trainer_pid' in record
(out/'production-host-kernel-restart.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('command','validation')},indent=2))
