#!/usr/bin/env python3
"""Size a recovery run from matching local GPU benchmark measurements."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex


def rate(path, marker):
    matches = re.findall(marker + r':[^\n]*\binput_tok_s=([0-9.]+)', path.read_text())
    if not matches or not math.isfinite(float(matches[-1])) or float(matches[-1]) <= 0:
        raise ValueError(f'Missing positive measured throughput: {path}. Inspect this log for model-loading or extraction/training errors; a failed benchmark cannot size a run.')
    return float(matches[-1])


def estimate(settings, extract_rate, train_rate, hours, reserve, epochs, cache_gib):
    context = int(settings['CTX'])
    validation = int(settings.get('VAL_WINDOWS', '32'))
    accumulation = int(settings.get('GRAD_ACCUM', '1'))
    extraction = extract_rate*0.8
    training = train_rate*0.8
    seconds = (hours - reserve)*3600
    if min(context, extract_rate, train_rate, epochs, cache_gib, validation, accumulation) <= 0 or not 0 <= reserve < hours:
        raise ValueError('Positive measurements, cache limit and epochs are required; reserve must be smaller than hours.')
    windows_time = math.floor((seconds - validation*context/extraction) /
                              (context/extraction + epochs*context/training))
    window_bytes = context*(6*int(settings['TOP_K']) + 4 + 8)
    windows_cache = math.floor((cache_gib*2**30 - 2**20)/window_bytes) - validation
    windows = min(windows_time, windows_cache)
    if windows < validation:
        raise ValueError('Budget is too small for this context and validation plan.')
    steps = math.ceil(windows/accumulation)*epochs
    result = {key: value for key, value in settings.items() if key not in ('MODEL_DIR', 'GPU')}
    result.update({'PROD_MAX_WINDOWS': str(windows), 'EPOCHS': str(epochs),
                   'SAVE_EVERY': str(math.ceil(steps/8)), 'EVAL_EVERY': str(math.ceil(steps/6)),
                   'PROD_DIR': settings['MODEL_DIR'] + f'/logit-distillation/production-ko-c{context}-w{windows}'})
    compute_hours = ((windows + validation)*context/extraction + epochs*windows*context/training)/3600
    return result, {'estimated_compute_hours': compute_hours, 'reserved_hours': reserve,
                    'total_budget_hours': hours, 'train_windows_cap': windows,
                    'worst_case_cache_gib': ((windows + validation)*window_bytes + 2**20)/2**30,
                    'checkpoint_sets_cap': steps//int(result['SAVE_EVERY']) + 1,
                    'benchmark_gpu': settings.get('GPU', 'unspecified'),
                    'note': 'Estimate, not a hard deadline. Reservation covers setup, validation, saves and uploads; network delays can exceed it.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-dir', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=10)
    parser.add_argument('--reserve-hours', type=float, default=2)
    parser.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', '3')))
    parser.add_argument('--max-cache-gib', type=float, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    settings = json.loads((args.bench_dir / 'settings.json').read_text())
    values, report = estimate(settings, rate(args.bench_dir / 'extract.log', 'distill_throughput'),
                              rate(args.bench_dir / 'train.log', 'qat_throughput'), args.hours,
                              args.reserve_hours, args.epochs, args.max_cache_gib)
    output = args.output or args.bench_dir / 'budget.env'
    output.write_text(''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items()))
    Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('Apply before extraction and training: source ' + shlex.quote(str(output)))
    print('GPU, data or context changes require a new benchmark and a new production directory.')


if __name__ == '__main__':
    main()
