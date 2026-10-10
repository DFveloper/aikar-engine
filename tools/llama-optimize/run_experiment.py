import json
import os
import pathlib
import subprocess
import sys
import time

prefix = pathlib.Path(sys.argv[1])
command = sys.argv[2:]
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'
project = pathlib.Path(__file__).resolve().parents[2]
env.setdefault('GGML_BACKEND_PATH', str(project / 'build/bin'))
start = time.monotonic()
peak_gpu = 0
peak_gpu_process = 0
peak_rss = 0
samples = 0
with open(str(prefix) + '.log', 'w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    while process.poll() is None:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if result.returncode == 0:
            peak_gpu = max(peak_gpu, int(result.stdout.strip()))
            samples += 1
        processes = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid,used_gpu_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if processes.returncode == 0:
            for line in processes.stdout.splitlines():
                fields = [field.strip() for field in line.split(',')]
                if len(fields) == 2 and fields[0] == str(process.pid) and fields[1].isdigit():
                    peak_gpu_process = max(peak_gpu_process, int(fields[1]))
        try:
            for line in pathlib.Path(f'/proc/{process.pid}/status').read_text().splitlines():
                if line.startswith('VmHWM:'):
                    peak_rss = max(peak_rss, int(line.split()[1]))
        except FileNotFoundError:
            pass
        time.sleep(0.2)
result = dict(command=command, returncode=process.returncode, seconds=time.monotonic()-start,
              peak_gpu_device_mib=peak_gpu, peak_process_rss_kib=peak_rss or None,
              peak_gpu_process_mib=peak_gpu_process,
              polling_seconds=0.2, gpu_samples=samples)
pathlib.Path(str(prefix) + '.resources.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
sys.exit(process.returncode)
