#!/usr/bin/env python3
"""Prepare a deterministic Korean knowledge/reasoning JSONL from pinned train splits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re

SOURCES = {
    'reasoning': ('Jongsim/claude-opus-4.6-reasoning-12k-ko-filtered-v2', '7f4d46a49d6ace960f10fc0fcf5d576ee2e95e7c', 'apache-2.0'),
    'knowledge': ('MarkrAI/KoCommercial-Dataset', 'c7ae950f2984ab3ff6a2365331bc9fa27cdac185', 'mit'),
}


def korean(text):
    letters = [char for char in text if char.isalpha()]
    return sum('\uac00' <= char <= '\ud7a3' for char in letters) >= max(5, len(letters)*0.2)


def normalize(row, kind, source):
    prompt = row.get('instruction', '')
    answer = row.get('response' if kind == 'reasoning' else 'output', '')
    reasoning, prefix = '', []
    if 'messages' in row:
        try:
            messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
        except (ValueError, TypeError):
            return None
        if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
            return None
        if messages and messages[0].get('role') == 'system':
            prefix = [messages[0]]
        first = messages[len(prefix):len(prefix) + 2]
        if [message.get('role') for message in first] != ['user', 'assistant']:
            return None
        prompt, answer = first[0].get('content'), first[1].get('content')
        reasoning = first[1].get('reasoning_content') or first[1].get('reasoning') or ''
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return None
    extra = row.get('input')
    if isinstance(extra, str) and extra.strip():
        prompt += '\n\n' + extra.strip()
    prompt, answer = prompt.strip(), answer.strip()
    if not isinstance(reasoning, str):
        return None
    if answer.startswith('<think>'):
        if '</think>' not in answer:
            return None
        embedded, answer = answer[len('<think>'):].split('</think>', 1)
        reasoning, answer = (reasoning or embedded).strip(), answer.strip()
    if not korean(prompt) or not korean(answer) or len(prompt) + len(answer) + len(reasoning) > 32000:
        return None
    if not 20 <= len(answer) <= 16000 or len(prompt) < 10:
        return None
    if any(error in answer for error in ('JSON pointer', '요청하신 JSON 포인터', 'As an AI language model')):
        return None
    if kind == 'knowledge':
        if re.search(r'번역|요약|제목을|시를|소설|광고|문장.*순서|MASK', prompt):
            return None
        if not re.search(r'무엇|왜|어떻게|설명|차이|원리|이유|의미|역할|정의|어떤|인가|일까|입니까', prompt):
            return None
    if kind == 'reasoning' and (not reasoning or not korean(reasoning)):
        return None
    assistant = {'role': 'assistant', 'content': answer}
    if reasoning:
        assistant['reasoning_content'] = reasoning
    return {'messages': prefix + [{'role': 'user', 'content': prompt}, assistant],
            'source': source, 'category': kind}


def sample_source(kind, count, seed, cache_dir, max_shards):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    repo, revision, license_name = SOURCES[kind]
    files = sorted(path for path in HfApi(token=os.environ.get('HF_TOKEN', False)).list_repo_files(
        repo_id=repo, repo_type='dataset', revision=revision) if path.endswith('.parquet') and
        (path.startswith('data/train-') or path == 'reasoning_merged_ko_filtered_v2.parquet'))
    random.Random(seed).shuffle(files)
    if max_shards:
        files = files[:max_shards]
    rng, sample, seen, fingerprints = random.Random(seed), [], 0, set()
    for name in files:
        path = hf_hub_download(repo_id=repo, filename=name, revision=revision, repo_type='dataset',
                               cache_dir=cache_dir, token=os.environ.get('HF_TOKEN', False))
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                record = normalize(row, kind, repo)
                if record is None:
                    continue
                prompt = next(message['content'] for message in record['messages'] if message['role'] == 'user')
                fingerprint = hashlib.sha256(prompt.encode()).digest()
                if fingerprint in fingerprints:
                    continue
                fingerprints.add(fingerprint)
                seen += 1
                if len(sample) < count:
                    sample.append(record)
                else:
                    index = rng.randrange(seen)
                    if index < count:
                        sample[index] = record
        print(f'{kind}: scanned {name}; eligible={seen}, sampled={len(sample)}', flush=True)
    if len(sample) < count:
        raise RuntimeError(f'{kind}: only {len(sample)} eligible records; reduce the requested count.')
    return sample, {'repo': repo, 'revision': revision, 'license': license_name,
                    'files': files, 'eligible': seen, 'sampled': len(sample)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/workspace/ko-knowledge-reasoning.jsonl'))
    parser.add_argument('--reasoning-rows', type=int, default=int(os.environ.get('KO_REASONING_ROWS', '6000')))
    parser.add_argument('--knowledge-rows', type=int, default=int(os.environ.get('KO_KNOWLEDGE_ROWS', '4000')))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-dir', default=os.environ.get('HF_HUB_CACHE', '/workspace/.cache/huggingface/hub'))
    parser.add_argument('--max-shards', type=int, default=0)
    args = parser.parse_args()
    if min(args.reasoning_rows, args.knowledge_rows) < 1 or args.max_shards < 0:
        parser.error('Both category counts must be positive and max-shards nonnegative.')
    output = args.output.resolve()
    manifest = Path(str(output) + '.json')
    config = {'reasoning_rows': args.reasoning_rows, 'knowledge_rows': args.knowledge_rows,
              'seed': args.seed, 'max_shards': args.max_shards, 'sources': SOURCES, 'policy_version': 3}
    config = json.loads(json.dumps(config))
    if output.exists():
        if manifest.is_file():
            old = json.loads(manifest.read_text())
            if old['config'] == config and old['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest():
                print(f'Using verified Korean dataset: {output}')
                return
        raise RuntimeError('Output already exists with different inputs; choose a new output path.')
    records, provenance = [], []
    for kind, count in [('reasoning', args.reasoning_rows), ('knowledge', args.knowledge_rows)]:
        selected, source = sample_source(kind, count, args.seed, args.cache_dir, args.max_shards)
        records.extend(selected)
        provenance.append(source)
    unique, fingerprints = [], set()
    for record in records:
        prompt = next(message['content'] for message in record['messages'] if message['role'] == 'user')
        prompt = re.sub(r'\s+', ' ', prompt).strip()
        fingerprint = hashlib.sha256(prompt.encode()).digest()
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique.append(record)
    records = unique
    random.Random(args.seed).shuffle(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + '.partial')
    with temporary.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary.replace(output)
    manifest.write_text(json.dumps({'config': config, 'provenance': provenance, 'rows': len(records),
                                    'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2) + '\n')
    print(f'Prepared {len(records)} Korean knowledge/reasoning records: {output}')


if __name__ == '__main__':
    main()
