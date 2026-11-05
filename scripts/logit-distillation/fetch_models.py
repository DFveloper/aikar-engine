#!/usr/bin/env python3
"""Download a public GGUF bundle from GHCR with verified, resumable chunks."""
import argparse
import hashlib
import os
from pathlib import Path

import requests


def fetch(repo, tag, directory):
    session = requests.Session()
    response = session.get('https://ghcr.io/token', params={'service': 'ghcr.io', 'scope': f'repository:{repo}:pull'}, timeout=30)
    response.raise_for_status()
    session.headers['Authorization'] = 'Bearer ' + response.json()['token']
    base = 'https://ghcr.io/v2/' + repo
    response = session.get(base + '/manifests/' + tag, headers={'Accept': 'application/vnd.oci.image.manifest.v1+json'}, timeout=30)
    response.raise_for_status()
    descriptor = response.json()['config']
    response = session.get(base + '/blobs/' + descriptor['digest'], timeout=60)
    response.raise_for_status()
    if 'sha256:' + hashlib.sha256(response.content).hexdigest() != descriptor['digest']:
        raise ValueError('Bundle config checksum mismatch')
    files = response.json()['files']
    directory.mkdir(parents=True, exist_ok=True)
    for item in files:
        name = item['name']
        if Path(name).name != name or not name.endswith('.gguf'):
            raise ValueError('Invalid model filename')
        target = directory / name
        if target.exists():
            with target.open('rb') as stream:
                checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
            if checksum != item['sha256']:
                raise ValueError(f'Existing model checksum mismatch: {target}')
            print(f'Already verified: {target}', flush=True)
            continue
        partial = target.with_suffix('.gguf.partial')
        with partial.open('r+b' if partial.exists() else 'w+b') as output:
            offset = 0
            for index, part in enumerate(item['parts']):
                output.seek(offset)
                checksum = hashlib.sha256()
                remaining = part['size']
                while remaining:
                    block = output.read(min(remaining, 8*1024**2))
                    if not block:
                        break
                    checksum.update(block)
                    remaining -= len(block)
                if remaining or 'sha256:' + checksum.hexdigest() != part['digest']:
                    for attempt in range(3):
                        output.seek(offset)
                        output.truncate()
                        checksum = hashlib.sha256()
                        received = 0
                        try:
                            with session.get(base + '/blobs/' + part['digest'], stream=True, timeout=(30, 120)) as response:
                                response.raise_for_status()
                                for block in response.iter_content(8*1024**2):
                                    received += len(block)
                                    if received > part['size']:
                                        raise ValueError('Model chunk exceeds expected size')
                                    output.write(block)
                                    checksum.update(block)
                            if received != part['size'] or 'sha256:' + checksum.hexdigest() != part['digest']:
                                raise ValueError('Model chunk checksum mismatch')
                            output.flush()
                            break
                        except (requests.RequestException, ValueError):
                            if attempt == 2:
                                raise
                offset += part['size']
                print(f'{name}: chunk {index + 1}/{len(item["parts"])} verified ({offset}/{item["size"]} bytes)', flush=True)
            output.truncate(offset)
            output.flush()
            os.fsync(output.fileno())
            output.seek(0)
            if offset != item['size'] or hashlib.file_digest(output, 'sha256').hexdigest() != item['sha256']:
                raise ValueError('Complete model checksum mismatch')
        partial.replace(target)
        print(f'Ready: {target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='Public GHCR owner/package')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('MODEL_DIR', '/workspace/Lumen-3.8-Pulsar')))
    args = parser.parse_args()
    fetch(args.repo, args.tag, args.output)
