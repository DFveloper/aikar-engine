"""Generate exact GSQ payloads from streamed calibration captures."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'model', 'captures', 'manifest', 'upstream', 'output'):
    parser.add_argument('--'+name, type=pathlib.Path, required=True)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--jobs', type=int, default=2)
args = parser.parse_args()
assert 2 <= args.steps <= 10000 and 1 <= args.jobs <= 4
base = json.loads(args.manifest.read_text())
captures = json.loads(args.captures.read_text())
assert captures['source_identity'] == base['source_identity']
assert [g['tensor'] for g in base['groups']] == [c['tensor'] for c in captures['captures']]
args.output.mkdir(parents=True, exist_ok=True)
tool = pathlib.Path(__file__).resolve().parent

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

algorithm = {'native_gsq': digest(tool/'optimize.cpp'), 'prior_adapter': digest(tool/'make_prior.py'),
             'upstream': subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()}
lock = threading.Lock()
completed = {}
started = time.monotonic()

def command(argv, log):
    with log.open('x') as output:
        result = subprocess.run([str(x) for x in argv], stdout=output, stderr=subprocess.STDOUT)
    assert result.returncode == 0, f'{log}: command failed with status {result.returncode}\n'+log.read_text()[-3000:]

def work(index):
    group, capture = base['groups'][index], captures['captures'][index]
    path = pathlib.Path(capture['file'])
    fingerprint = {'source_identity': base['source_identity'], 'tensor': group['tensor'],
                   'capture_sha256': digest(path), 'steps': args.steps, 'algorithm': algorithm, 'backend': 'CPU'}
    choices = []
    for method in ('rtn', 'gptq'):
        prefix = args.output/f'{index}.{method}'
        seal = pathlib.Path(str(prefix)+'.seal.json')
        payload = pathlib.Path(str(prefix)+'.bin')
        report = pathlib.Path(str(prefix)+'.json')
        if seal.exists():
            record = json.loads(seal.read_text())
            assert record['fingerprint'] == fingerprint and record['sha256'] == digest(payload), 'stale GSQ candidate'
        else:
            assert not payload.exists() and not report.exists(), 'unsealed GSQ output exists'
            prior = pathlib.Path(str(prefix)+'.prior.bin')
            if method == 'gptq':
                command([sys.executable, tool/'make_prior.py', '--upstream', args.upstream, '--capture', path,
                         '--output', prior, '--method', 'gptq'], pathlib.Path(str(prefix)+'.prior.log'))
            argv = [args.binary, 'gsq-pack', args.model, path, group['tensor'], payload,
                    str(args.steps), 'CPU', report]
            if method == 'gptq':
                argv.append(prior)
            command(argv, pathlib.Path(str(prefix)+'.log'))
            result = json.loads(report.read_text())
            assert result['tensor'] == group['tensor'] and result['steps'] == args.steps
            assert payload.stat().st_size == group['choices'][0]['bytes']
            record = {'fingerprint': fingerprint, 'sha256': digest(payload), 'report': str(report),
                      'initial_mse': result['initial_mse'], 'best_hard_mse': result['best_hard_mse']}
            seal.write_text(json.dumps(record, indent=2)+'\n')
            if prior.exists():
                prior.unlink()
        choice = dict(group['choices'][0])
        choice.update(producer='gsq-'+method, file=str(payload.resolve()), sha256=record['sha256'])
        choices.append(choice)
        with lock:
            completed[f'{index}.{method}'] = record
            progress = {'completed_candidates': len(completed), 'target_candidates': 2*len(base['groups']),
                        'seconds': time.monotonic()-started, 'latest_tensor': group['tensor'], 'latest_method': method}
            temporary = args.output/'progress.partial.json'
            temporary.write_text(json.dumps(progress, indent=2)+'\n')
            temporary.replace(args.output/'progress.json')
            print(json.dumps(progress), flush=True)
    return index, choices

with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
    results = list(executor.map(work, range(len(base['groups']))))
for index, choices in results:
    base['groups'][index]['choices'].extend(choices)
manifest = args.output/'manifest.json'
assert not manifest.exists(), 'GSQ manifest exists'
manifest.write_text(json.dumps(base, indent=2)+'\n')
subprocess.run([str(args.binary), 'candidate-validate', str(args.model), str(manifest)], check=True)
print('PASS: all GSQ payloads validated without requantization', flush=True)
