from pathlib import Path
import os, subprocess, json
import numpy as np
out = Path(__file__).resolve().parent
baseline = out/'q4-register-baseline-bin'
results = []
for dimension, mode in [(256, i) for i in range(6)] + [(512, 0), (512, 1), (512, 3)]:
    values = []
    for variant in ('before', 'after'):
        target = Path('/dev/shm/aikar-fattn-regression-'+variant+'.bin')
        env = os.environ.copy()
        env.update(GGML_DISABLE_VULKAN='1', AIKAR_BENCH_MODE=str(mode))
        if variant == 'before':
            env.update(LD_LIBRARY_PATH=str(baseline), AIKAR_BENCH_BACKEND_PATH=str(baseline))
        else:
            env.pop('AIKAR_BENCH_BACKEND_PATH', None)
        with (out/f'fattn-fusion-regression-{dimension}-{mode}-{variant}.log').open('w') as log:
            subprocess.run(['/tmp/aikar-fattn-fusion-regression', str(dimension), '1280', str(target)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = np.fromfile(target, np.float32)
        q_count = dimension*256*16
        if mode in (2, 3): data = data[:q_count]
        values.append(data)
    a, b = values
    difference = a.astype(np.float64) - b.astype(np.float64)
    relative = float(np.linalg.norm(difference)/max(np.linalg.norm(a.astype(np.float64)), 1e-30))
    report = dict(dimension=dimension, mode=mode, floats=len(a), different=int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32))), relative_l2=relative, max_abs=float(np.max(np.abs(difference))), finite=bool(np.isfinite(b).all()))
    results.append(report)
    print(json.dumps(report), flush=True)
    assert report['finite'] and relative < 1e-4
(out/'fattn-fusion-regressions.json').write_text(json.dumps(results, indent=2)+'\n')
