import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get('FLARE_BENCH_OUT', str(ROOT / 'build-flare-p620/results')))
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:18137'
MODEL = '/mnt/openwebui/AIKAR/Lumen-3.2-Flare/Lumen-3.2-Flare-stage4-LD-Q4_0_L.gguf'
BIN = os.environ.get('FLARE_BENCH_BIN', str(ROOT / 'build-flare-p620/bin/llama-server'))

def post(path, data):
    req = urllib.request.Request(URL + path, json.dumps(data).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1800)

def measure(users, predict=512, prompt_len=8192):
    tokens = json.load(post('/tokenize', {'content': '\ud55c\uad6d\uc5b4\ub85c \uc11c\ubc84\uc758 \ub3d9\uc2dc \uc0ac\uc6a9\uc790 \uc131\ub2a5\uacfc \uba54\ubaa8\ub9ac \uad00\ub9ac \ubc29\ubc95\uc744 \uc790\uc138\ud788 \uc124\uba85\ud558\uc138\uc694. ' * 1500, 'add_special': True}))['tokens']
    tokens = tokens[:prompt_len]
    assert len(tokens) == prompt_len
    barrier = threading.Barrier(users)
    def request(i):
        prompt = tokens.copy()
        prompt[1:5] = [1000 + i, 2000 + i, 3000 + i, 4000 + i]
        data = {'prompt': prompt, 'n_predict': predict, 'ignore_eos': True, 'cache_prompt': False, 'stream': True, 'temperature': 0, 'seed': 42}
        barrier.wait()
        start = time.monotonic()
        arrivals = []
        result = None
        pieces = []
        with post('/completion', data) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                event = json.loads(line[6:])
                if event.get('stop'):
                    result = event
                else:
                    arrivals.append(time.monotonic())
                    pieces.append(event.get('content', ''))
        assert result is not None, 'missing final event'
        timings = result['timings']
        assert timings['predicted_n'] == predict, timings
        assert timings['prompt_n'] == prompt_len, timings
        elapsed = arrivals[-1] - arrivals[0]
        gaps = sorted(b - a for a, b in zip(arrivals, arrivals[1:]))
        return {'user': i, 'timings': timings, 'ttft_s': arrivals[0] - start, 'stream_tok_s': (predict - 1) / elapsed, 'p95_gap_ms': gaps[int(.95 * (len(gaps) - 1))] * 1000, 'max_gap_ms': max(gaps) * 1000, 'wall_s': time.monotonic() - start, 'output': ''.join(pieces)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=users) as pool:
        rows = list(pool.map(request, range(users)))
    return rows

def run(tag, users, ngl, threads, batch, ubatch, paged=True, extra=None, predict=512, prompt_len=8192, env_extra=None, profile=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', LD_LIBRARY_PATH=str(pathlib.Path(BIN).parent))
    env.update(env_extra or {})
    args = [BIN, '-m', MODEL, '--host', '127.0.0.1', '--port', '18137', '-fit', 'off', '-ngl', str(ngl), '-dev', 'CUDA0', '-c', str(9216 * users), '-np', str(users), '-b', str(batch), '-ub', str(ubatch), '-ctk', 'q8_kv', '-ctv', 'q8_kv', '-kvu', '-fa', 'on', '-t', str(threads), '-tb', str(threads), '--backend-sampling', '--no-cache-idle-slots', '--cache-ram', '0', '--ctx-checkpoints', '0', '--slots', '--no-context-shift', '-lv', '4']
    if paged:
        args += ['--kv-paged', '--kv-block-size', '16', '--kv-blocks', str(576 * users)]
    args += extra or []
    print('START', tag, flush=True)
    with (OUT / (tag + '.log')).open('w') as log:
        command = ['nvprof', '--csv', '--log-file', str(OUT / (tag + '.nvprof.csv'))] + args if profile else args
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 120
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server exited: ' + str(proc.returncode))
                try:
                    with urllib.request.urlopen(URL + '/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('startup timeout')
                time.sleep(.3)
            rows = measure(users, predict, prompt_len)
            result = {'tag': tag, 'args': args, 'environment': env_extra or {}, 'rows': rows, 'target_tok_s': 19, 'min_tok_s': min(r['stream_tok_s'] for r in rows), 'pass': all(r['stream_tok_s'] >= 19 and r['timings']['predicted_per_second'] >= 19 for r in rows)}
            (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False))
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print('FAIL', tag, str(exc), flush=True)
        finally:
            if profile:
                os.killpg(proc.pid, signal.SIGINT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

if __name__ == '__main__':
    configs = json.loads(sys.argv[1])
    for config in configs:
        run(**config)
