import concurrent.futures
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(os.environ.get('FLARE_BENCH_OUT', str(ROOT/'build-flare-p620/results')))
OUT.mkdir(parents=True, exist_ok=True)
URL = 'http://127.0.0.1:18137'

def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data, {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=1200)

def obj(path, body=None):
    with api(path, body) as response:
        return json.load(response)

def collect(prompt, slot, predict=512):
    started = time.monotonic()
    arrivals, pieces, final = [], [], None
    with api('/completion', {'prompt': prompt, 'n_predict': predict, 'ignore_eos': True, 'cache_prompt': True, 'stream': True, 'temperature': 0, 'seed': 42}) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            event = json.loads(line[6:])
            if event.get('stop'):
                final = event
            else:
                arrivals.append(time.monotonic())
                pieces.append(event.get('content', ''))
    assert final and final['timings']['predicted_n'] == predict, final
    timings = final['timings']
    assert timings['prompt_n'] + timings['cache_n'] == 8192, timings
    gaps = sorted(b-a for a,b in zip(arrivals, arrivals[1:]))
    return {'slot': slot, 'timings': timings, 'ttft_s': arrivals[0]-started, 'first_arrival_s': arrivals[0], 'last_arrival_s': arrivals[-1], 'stream_tok_s': (predict-1)/(arrivals[-1]-arrivals[0]), 'max_gap_ms': max(gaps)*1000, 'output': ''.join(pieces)}

def run(users):
    try:
        obj('/health')
    except OSError:
        pass
    else:
        raise RuntimeError('benchmark port 18137 is already serving; stop that server first')
    cache_type = os.environ.get('FLARE_CACHE_TYPE', 'f16')
    independent = os.environ.get('FLARE_BENCH_INDEPENDENT', '0') == '1'
    tag = ('cache-independent-np' if independent else 'cache-bos-np') + str(users) + '-' + cache_type + '-512'
    env = dict(os.environ, FLARE_SERVER_BIN=os.environ.get('FLARE_SERVER_BIN', str(ROOT/'build-flare-p620/bin/llama-server')), FLARE_PARALLEL=str(users), FLARE_KV_BLOCKS=str(576*users), FLARE_PORT='18137')
    print('START', tag, flush=True)
    with (OUT/(tag+'.log')).open('w') as log:
        proc = subprocess.Popen([str(ROOT/'scripts/serve-flare-p620-paged.sh'), '--slot-save-path', str(OUT), '-lv', '4'], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('server exit '+str(proc.returncode))
                try:
                    if obj('/health').get('status') == 'ok':
                        break
                except OSError:
                    pass
                if time.monotonic()>deadline:
                    raise RuntimeError('startup timeout')
                time.sleep(.3)
            static = []
            for thinking in [False, True]:
                for slot in [0, users-1]:
                    obj('/slots/'+str(slot)+'?action=erase', {})
                    result=obj('/v1/chat/completions', {'messages':[{'role':'user','content':'\uc548\ub155\ud558\uc138\uc694.'}], 'chat_template_kwargs':{'enable_thinking':thinking}, 'id_slot':slot, 'max_tokens':1, 'temperature':0, 'cache_prompt':False})
                    assert result['timings']['cache_n']>100, result
                    static.append({'slot':slot,'thinking':thinking,'timings':result['timings']})
            messages=[{'role':'user','content':'\ub2e4\uc74c \ubb38\uc11c\ub97c \ubc14\ud0d5\uc73c\ub85c \uc11c\ubc84 \ub3d9\uc2dc \ucc98\ub9ac\uc640 \uba54\ubaa8\ub9ac \uad00\ub9ac \ubc29\ubc95\uc744 \ud55c\uad6d\uc5b4\ub85c \uc124\uba85\ud558\uc138\uc694.\n'+('\uc11c\ubc84\ub294 \uc0ac\uc6a9\uc790\ubcc4 \ub300\ud654\ub97c \uad00\ub9ac\ud558\uace0 \uacf5\ud1b5 \uc9c0\uce68\uacfc \ub300\ud654 \uae30\ub85d\uc744 \uce90\uc2dc\uc5d0 \uc800\uc7a5\ud569\ub2c8\ub2e4. ' * 1800)}]
            rendered=obj('/apply-template', {'messages':messages})['prompt']
            tokens=obj('/tokenize', {'content':rendered, 'add_special':True, 'parse_special':True})['tokens']
            assert len(tokens)>8192
            prompt=tokens[:8160]+tokens[-32:]
            prompt[-64:-60]=[1000,2000,3000,4000]
            warm=None if independent else obj('/completion', {'prompt':prompt, 'id_slot':0, 'n_predict':1, 'ignore_eos':True, 'cache_prompt':True, 'temperature':0})
            if warm is not None:
                assert warm['timings']['cache_n']>100, warm['timings']
            prompts = []
            for slot in range(users):
                own = prompt.copy()
                if independent:
                    own[2000:2004] = [1000+slot, 2000+slot, 3000+slot, 4000+slot]
                    seeded = obj('/completion', {'prompt':own, 'id_slot':slot, 'n_predict':1, 'ignore_eos':True, 'cache_prompt':True, 'temperature':0})
                    assert seeded['timings']['prompt_n'] + seeded['timings']['cache_n'] == 8192, seeded['timings']
                    if slot == 0:
                        warm = seeded
                    print('WARM', tag, slot, seeded['timings']['cache_n'], flush=True)
                else:
                    own[-64:-60] = [1000+slot, 2000+slot, 3000+slot, 4000+slot]
                prompts.append(own)
            for slot in range(users):
                obj('/slots/'+str(slot)+'?action=erase', {})
            barrier=threading.Barrier(users)
            def request(slot):
                own=prompts[slot]
                barrier.wait()
                return collect(own,slot)
            with concurrent.futures.ThreadPoolExecutor(max_workers=users) as pool:
                futures=[pool.submit(request,i) for i in range(users)]
                peak=0
                peak_generating=0
                while not all(f.done() for f in futures):
                    slots=obj('/slots')
                    peak=max(peak,sum(s.get('is_processing',False) for s in slots))
                    peak_generating=max(peak_generating,sum(s.get('is_processing',False) and any(t.get('n_decoded',0)>0 for t in s.get('next_token',[])) for s in slots))
                    time.sleep(.25)
                rows=[f.result() for f in futures]
            assert all(r['timings']['cache_n']>8000 for r in rows), rows
            result={'tag':tag,'users':users,'independent_histories':independent,'input_tokens':8192,'output_tokens':512,'target_min_tok_s':8,'meets_target':peak_generating == users and all(r['stream_tok_s'] >= 8 for r in rows),'system_cache_checks':static,'warmup_timings':warm['timings'],'peak_processing_slots':peak,'peak_generating_slots':peak_generating,'rows':rows,'min_tok_s':min(r['stream_tok_s'] for r in rows),'aggregate_decode_tok_s':sum(511 for r in rows)/(max(r['last_arrival_s'] for r in rows)-min(r['first_arrival_s'] for r in rows))}
            (OUT/(tag+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='rows' and k!='system_cache_checks'},ensure_ascii=False),flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()

if __name__=='__main__':
    for n in map(int,sys.argv[1:]):
        run(n)
