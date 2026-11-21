import json
import math
from pathlib import Path
import re
import subprocess
import time

out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
path = out / 'production-step11-resumed.json'
record = json.loads(path.read_text())
pid = 213937
for _ in range(120):
    assert Path(f'/proc/{pid}/cmdline').exists(), 'Trainer exited'
    log = (out / 'step11-resume-launch.log').read_text()
    window = re.search(r'data=\d+/\d+ loss=([\d.e+-]+) acc=\S+ t=(\d+:\d+:\d+)', log)
    if 'start_step=11' in log and window:
        assert math.isfinite(float(window[1]))
        libraries = sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines() if 'validated-release/' in line})
        assert len(libraries) == 4
        environment = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
        assert not any(value.startswith(b'GGML_TRAIN_MOE') for value in environment)
        limits = subprocess.check_output(['nvidia-smi', '--id=0,1', '--query-gpu=power.limit', '--format=csv,noheader,nounits'], text=True)
        assert [float(value) for value in limits.split()] == [150.0, 150.0]
        record.update(status='training_verified', trainer_pid=pid, loaded_libraries=libraries, observed_schedule_start_step=11,
                      first_completed_window_loss=float(window[1]), first_completed_window_elapsed=window[2], verified_at=time.time())
        path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))
        break
    time.sleep(5)
else:
    raise RuntimeError('Resume verification timed out; trainer left running')
