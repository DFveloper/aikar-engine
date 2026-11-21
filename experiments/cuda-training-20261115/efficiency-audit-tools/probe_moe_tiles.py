import os
from pathlib import Path
import subprocess

out = Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit')
for tile in (512, 1024):
    env = dict(os.environ, GGML_TRAIN_MOE_TILE=str(tile))
    subprocess.run(['python3', str(out / 'run_moe_tile_probe.py'), f'candidate-moe-{tile}', 'cpu', '2', '--peer', '--p2p'], env=env, check=False)
