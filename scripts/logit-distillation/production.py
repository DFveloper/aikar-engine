#!/usr/bin/env python3
"""Plan and run CPU teacher extraction or QLion2 attention recovery."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def identity(path):
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def available(path):
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def space_check(path, needed, dry):
    free = available(path)
    print(f'Disk: free={free / 2**30:.2f} GiB, required={needed / 2**30:.2f} GiB', flush=True)
    require(dry or free >= needed, 'Insufficient disk space. Free space or set PROD_DIR to another volume; no training was started.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['extract', 'train'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    env = os.environ
    root = Path(env['PROD_DIR']).resolve()
    repo = Path(env['REPO']).resolve()
    binary_dir = Path(env['BIN_DIR']).resolve()
    data = Path(env['DATA']).resolve()
    student = Path(env.get('STUDENT', str(Path(env['MODEL_DIR']) / 'Lumen-3.8-Pulsar-base.gguf'))).resolve()
    teacher = Path(env.get('TEACHER', str(Path(env['MODEL_DIR']) / 'Test.gguf'))).resolve()
    context = int(env['CTX'])
    extract_batch = int(env['EXTRACT_UBATCH'])
    train_batch = int(env['TRAIN_UBATCH'])
    threads = int(env['THREADS'])
    require(context >= 256 and context % 256 == 0, 'CTX must be a positive multiple of 256.')
    require(0 < extract_batch <= context and 0 < train_batch <= context and context % train_batch == 0, 'Invalid microbatch size.')
    require(threads > 0 and int(env['EPOCHS']) > 0 and int(env['GRAD_ACCUM']) > 0, 'Threads, epochs and accumulation must be positive.')
    for path in (student, teacher, data):
        require(path.is_file(), f'Missing input: {path}')
    for name in ('llama-extract-logits', 'llama-finetune-qlion'):
        require((binary_dir / name).is_file(), f'Build {name} first.')
    inputs = {'student': identity(student), 'teacher': identity(teacher), 'data': identity(data), 'data_sha256': digest(data),
              'context': context, 'top_k': int(env['TOP_K']), 'temperature': float(env['TEMPERATURE']),
              'val_split': float(env['VAL_SPLIT']), 'validation_windows': int(env['VAL_WINDOWS']),
              'max_windows': 0, 'preserve_thinking': True, 'train_on_prompt': False}
    require(0 < inputs['val_split'] < 1 and inputs['validation_windows'] > 0, 'Production requires held-out validation.')
    cache = root / f'teacher-top{inputs["top_k"]}-c{context}.dat'
    common = ['--train-file', str(data), '--distill-cache', str(cache), '--distill-max-windows', '0',
              '--distill-validation-windows', str(inputs['validation_windows']), '--val-split', str(inputs['val_split']),
              '-c', str(context), '-b', str(context), '-np', '1', '-ngl', '0',
              '-t', str(threads), '-tb', str(threads), '-dthr', str(threads)]
    extract = [str(binary_dir / 'llama-extract-logits'), '-m', str(teacher), '--distill-student', str(student),
               '--distill-top-k', str(inputs['top_k']), '--distill-temperature', str(inputs['temperature']),
               '-ub', str(extract_batch), *common]
    if args.dry_run:
        plan_log = subprocess.DEVNULL
    else:
        root.mkdir(parents=True, exist_ok=True)
        plan_log = (root / 'plan.log').open('ab')
    try:
        result = subprocess.run([*extract, '--distill-plan'], stdout=subprocess.PIPE, stderr=plan_log, check=True)
        plan = json.loads(result.stdout)
    finally:
        if plan_log != subprocess.DEVNULL:
            plan_log.close()
    print(json.dumps({'stage': args.stage, 'plan': plan, 'cache': str(cache)}, indent=2), flush=True)
    require(plan['train_windows'] > 0 and plan['validation_windows'] > 0, 'No train or validation windows.')
    if args.stage == 'extract':
        command = [*extract, '--distill-resume-extraction']
        if args.dry_run:
            space_check(root, plan['cache_bytes'] + 2**30, True)
            print(shlex.join(command))
            return
        with (root / 'extract.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            input_manifest = root / 'cache.inputs.json'
            if input_manifest.exists():
                require(json.loads(input_manifest.read_text()) == inputs, 'Cache inputs changed. Use a new PROD_DIR.')
            else:
                require(not cache.exists() and not Path(str(cache) + '.partial').exists(), 'Untracked cache exists. Use a new PROD_DIR.')
                save_json(input_manifest, inputs)
            manifest = root / 'cache.json'
            if cache.exists():
                require(cache.stat().st_size == plan['cache_bytes'], 'Completed cache size differs from plan.')
                checksum = digest(cache)
                if manifest.exists():
                    require(json.loads(manifest.read_text())['sha256'] == checksum, 'Cache checksum differs.')
                save_json(manifest, {'inputs': inputs, 'plan': plan, 'sha256': checksum})
                print('Verified complete teacher cache; extraction skipped.')
                return
            partial = Path(str(cache) + '.partial')
            needed = max(0, plan['cache_bytes'] - (partial.stat().st_size if partial.exists() else 0)) + 2**30
            space_check(root, needed, False)
            save_json(root / 'extract.command.json', command)
            print(shlex.join(command), flush=True)
            with (root / 'extract.log').open('ab') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            require(digest(data) == inputs['data_sha256'], 'Data changed during extraction.')
            require(cache.stat().st_size == plan['cache_bytes'], 'Cache size differs from plan.')
            save_json(manifest, {'inputs': inputs, 'plan': plan, 'sha256': digest(cache)})
            print('Teacher cache complete.')
        return
    if cache.is_file() and (root / 'cache.json').is_file():
        cache_manifest = json.loads((root / 'cache.json').read_text())
        require(cache_manifest['inputs'] == inputs and cache_manifest['plan'] == plan, 'Extraction and training inputs differ.')
        require(cache_manifest['sha256'] == digest(cache), 'Teacher cache checksum differs.')
    else:
        require(args.dry_run, 'Run production-extract.sh first.')
        cache_manifest = {'sha256': 'PENDING_EXTRACTION'}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = env.get('RUN_ID', 'recovery-' + stamp)
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id), 'Invalid RUN_ID.')
    run = root / 'runs' / run_id
    epochs, accumulation = int(env['EPOCHS']), int(env['GRAD_ACCUM'])
    steps = math.ceil(plan['train_windows'] / accumulation) * epochs
    save_every, eval_every = int(env['SAVE_EVERY']), int(env['EVAL_EVERY'])
    warmup = int(env['WARMUP_STEPS'])
    require(0 <= warmup < steps and save_every >= 0 and eval_every >= 0, 'Invalid scheduler or checkpoint interval.')
    settings = {key: env[key] for key in ('TRAIN_UBATCH', 'EPOCHS', 'LR', 'LR_MIN', 'WARMUP_STEPS', 'SAVE_EVERY',
                                         'EVAL_EVERY', 'GRAD_ACCUM', 'TARGETS')}
    settings.update({'inputs': inputs, 'cache_sha256': cache_manifest['sha256'], 'binary': identity(binary_dir / 'llama-finetune-qlion')})
    command = [str(binary_dir / 'llama-finetune-qlion'), '-m', str(student), *common,
               '-ub', str(train_batch), '--optimizer', 'qlion2', '--quant-type', 'q4_0,q8_0',
               '--qat-targets', env['TARGETS'], '--freeze-layers', '0', '--qat-static-graph', 'on',
               '--grad-accum', str(accumulation), '--epochs', str(epochs), '-lr', env['LR'],
               '-lr-min', env['LR_MIN'], '--lr-scheduler', 'cosine', '--warmup-steps', str(warmup),
               '--save-every', str(save_every), '--eval-every', str(eval_every), '--eval-initial',
               '--shuffle-dataset', '--verbose-loss', '--qat-out', str(run / 'final.gguf')]
    resume = env.get('RESUME')
    if resume:
        checkpoint = Path(resume).resolve()
        require(checkpoint.is_file() and Path(str(checkpoint) + '.qat-state.gguf').is_file(), 'Resume requires model and optimizer state.')
        require((run / 'run.json').is_file(), 'Resume requires the original RUN_ID and run manifest.')
        previous = json.loads((run / 'run.json').read_text())
        require({k: v for k, v in previous.items() if k != 'binary'} == {k: v for k, v in settings.items() if k != 'binary'}, 'Resume settings differ.')
        command += ['--qat-resume', str(checkpoint)]
    else:
        require(not run.exists(), 'Run already exists. Set a new RUN_ID or explicit RESUME.')
    sys.path.insert(0, str(repo / 'gguf-py'))
    import gguf
    reader = gguf.GGUFReader(str(student))
    targets = env['TARGETS'].split(',')
    elements = sum(math.prod(int(x) for x in t.shape) for t in reader.tensors
                   if any(s in t.name for s in targets) and int(t.tensor_type) in (2, 8))
    require(elements > 0, 'No supported attention parameters selected.')
    checkpoint_bytes = student.stat().st_size + math.ceil(elements * 52 / 32)
    copies = (steps // save_every if save_every else 0) + 1
    space_check(root, checkpoint_bytes * copies + 2 * 2**30, args.dry_run)
    print(f'Optimizer steps={steps}, checkpoint sets <= {copies}; older checkpoints are retained.', flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    run.mkdir(parents=True, exist_ok=bool(resume))
    with (run / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume:
            save_json(run / 'run.json', settings)
        save_json(run / ('command-' + stamp + '.json'), command)
        save_json(run / 'status.json', {'phase': 'running', 'command': command})
        with (run / ('train-' + stamp + '.log')).open('ab') as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            except BaseException:
                save_json(run / 'status.json', {'phase': 'interrupted', 'resume_requires': 'explicit checkpoint model plus .qat-state.gguf'})
                raise
        require((run / 'final.gguf').is_file() and (run / 'final.gguf.qat-state.gguf').is_file(), 'Final model/state pair is missing.')
        save_json(run / 'status.json', {'phase': 'complete', 'model': str(run / 'final.gguf'), 'optimizer_steps': steps})
    print('Recovery training complete.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'production: {error}', file=sys.stderr)
        sys.exit(1)
