#!/usr/bin/env python3
"""Publish completed teacher caches and checkpoint pairs to Hugging Face."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def upload_error(error):
    message = f'{type(error).__name__}: {error}'
    token = os.environ.get('HF_TOKEN')
    if token:
        message = message.replace(token, '[REDACTED]')
    return re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', message)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def completed_checkpoints(run, log_path=None):
    steps = set()
    final = False
    for log in ([log_path] if log_path else run.glob('train-*.log')):
        with log.open(errors='replace') as stream:
            for line in stream:
                match = re.search(r'checkpoint set saved at step (\d+) window ', line)
                if match:
                    steps.add(int(match[1]))
                if 'qat_adapter_checkpoint: saved final path=' in line:
                    final = True
    models = [run / f'final.gguf.step{step}.gguf' for step in sorted(steps)]
    status = run / 'status.json'
    if final or (log_path is None and status.is_file() and json.loads(status.read_text()).get('phase') == 'complete'):
        models.append(run / 'final.gguf')
    return models


class Publisher:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        token = env.get('HF_TOKEN')
        if not token:
            raise RuntimeError('Set HF_TOKEN with repository write permission, or HF_UPLOAD=0 for a local-only run.')
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
            from huggingface_hub.utils import validate_repo_id
        except ImportError as error:
            raise RuntimeError('Install huggingface_hub to upload training artifacts.') from error
        self.api = HfApi(token=token)
        self.add = CommitOperationAdd
        try:
            account = self.api.whoami()['name']
        except Exception as error:
            raise RuntimeError('Hugging Face authentication failed; check HF_TOKEN.') from error
        self.repos = {}
        for kind, variable, default in [('dataset', 'HF_DATASET_REPO', 'lumen-teacher-logits'),
                                         ('model', 'HF_MODEL_REPO', 'lumen-recovery-checkpoints')]:
            name = env.get(variable) or default
            repo = name if '/' in name else f'{account}/{name}'
            validate_repo_id(repo)
            self.repos[kind] = repo
        self.ready = set()

    def prepare(self, kind):
        repo = self.repos[kind]
        if kind not in self.ready:
            self.api.create_repo(repo_id=repo, repo_type=kind, private=False, exist_ok=True)
            if self.api.repo_info(repo_id=repo, repo_type=kind).private:
                raise RuntimeError(f'{repo} is private; choose a public repository with the corresponding HF_*_REPO variable.')
            self.ready.add(kind)
        return repo

    def commit(self, kind, files, receipt_path, message):
        repo = self.prepare(kind)
        signature = {remote: {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                     for remote, path in files.items() if isinstance(path, Path)}
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text())
            if previous.get('repo_id') == repo and previous.get('files') == signature:
                return
        operations = [self.add(path_in_repo=remote, path_or_fileobj=str(path) if isinstance(path, Path) else path)
                      for remote, path in files.items()]
        result = self.api.create_commit(repo_id=repo, repo_type=kind, operations=operations,
                                        commit_message=message)
        save_json(receipt_path, {'repo_id': repo, 'repo_type': kind, 'files': signature,
                                'commit': result.oid, 'url': result.commit_url})
        print(f'HF upload complete: {result.commit_url}', flush=True)

    def dataset(self, root):
        manifest_path = root / 'cache.json'
        manifest = json.loads(manifest_path.read_text())
        inputs = manifest['inputs']
        cache = root / f'teacher-top{inputs["top_k"]}-c{inputs["context"]}.dat'
        data = Path(inputs['data']['path'])
        if sha256(cache) != manifest['sha256'] or sha256(data) != inputs['data_sha256']:
            raise RuntimeError('Cache or dataset checksum differs from its completed extraction manifest.')
        prefix = 'caches/' + manifest['sha256']
        readme = ('---\ntags:\n- knowledge-distillation\n- logits\n---\n'
                  '# Teacher logits for Lumen recovery\n\n'
                  'Completed caches are versioned by SHA-256 under caches/. Each version includes train.jsonl, '
                  'a QLKD0001 teacher.dat cache and cache.json with checksums and packing settings. '
                  'The cache stores top-k token IDs, FP16 log-probabilities and FP32 tail log-mass. '
                  'Use matching student tokenizer, context, train/validation split and supervised masks. '
                  'Each version contains the exact input data; optional provenance.json records data sources and sampling policy. '
                  'Training uses pure forward KL with no ground-truth CE.\n').encode()
        files = {f'{prefix}/teacher.dat': cache, f'{prefix}/train.jsonl': data,
                 f'{prefix}/cache.json': manifest_path, 'README.md': readme}
        provenance = Path(str(data) + '.json')
        if provenance.is_file():
            files[f'{prefix}/provenance.json'] = provenance
        self.commit('dataset', files,
                    root / 'hf-dataset-upload.json', 'Upload completed teacher cache and matching data')

    def checkpoint(self, run, model):
        if model.parent != run or not re.fullmatch(r'final\.gguf(?:\.step\d+\.gguf)?', model.name):
            raise RuntimeError('Invalid checkpoint path.')
        state = Path(str(model) + '.qat-state.gguf')
        suffix = model.name[len('final.gguf'):]
        adapter = run / ('final.adapter.gguf' + suffix)
        if not adapter.is_file() or not state.is_file():
            raise RuntimeError('A completed adapter checkpoint needs both adapter and optimizer-state files.')
        prefix = f'runs/{run.name}/{model.name}'
        readme = ('---\ntags:\n- gguf\n- knowledge-distillation\n---\n'
                  '# Lumen recovery checkpoints\n\n'
                  'All runs and checkpoints are in this one model repository under runs/. '
                  'Each checkpoint directory contains adapter.gguf, model.gguf.qat-state.gguf and run.json. '
                  'The full-model checkpoint stays local. Keep the adapter, matching base and optimizer state for recovery. '
                  'The adapter embeds its exact base SHA-256. After a training resume its base is the resume checkpoint; '
                  'retain earlier adapters and verify their base/tensor hashes when rebuilding that base. '
                  'Attention targets are blocks 11, 17, 23 and 29. '
                  'The objective is teacher top-k plus tail forward KL with no ground-truth CE. '
                  'Checkpoints are intermediate recovery experiments, not a claim of improved model quality.\n').encode()
        self.commit('model', {f'{prefix}/adapter.gguf': adapter, f'{prefix}/model.gguf.qat-state.gguf': state,
                             f'{prefix}/run.json': run / 'run.json', 'README.md': readme},
                    run / (model.name + '.hf-upload.json'), f'Upload adapter {run.name} {adapter.name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['dataset', 'checkpoints'])
    parser.add_argument('--prod-dir', type=Path, default=Path(os.environ.get('PROD_DIR', '.')))
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    publisher = Publisher()
    if args.stage == 'dataset':
        publisher.dataset(args.prod_dir.resolve())
    else:
        if args.run_dir is None:
            parser.error('checkpoints requires --run-dir')
        run = args.run_dir.resolve()
        models = completed_checkpoints(run)
        if not models:
            raise RuntimeError('No completed checkpoint pairs found in training logs/status.')
        for model in models:
            publisher.checkpoint(run, model)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'HF upload failed: {upload_error(error)}', flush=True)
        raise SystemExit(1)
