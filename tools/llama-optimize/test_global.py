"""Verify real-model RCO updates, checkpoint replay and stale-state rejection."""
import argparse
import json
import pathlib
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary', type=pathlib.Path)
parser.add_argument('model', type=pathlib.Path)
parser.add_argument('manifest', type=pathlib.Path)
parser.add_argument('tokens', type=pathlib.Path)
args = parser.parse_args()
store = json.loads(args.manifest.read_text())
budget = store['fixed_gguf_bytes'] + sum(g['choices'][1]['aligned_bytes'] for g in store['groups'])
with tempfile.TemporaryDirectory(prefix='rco-global-test-') as directory:
    root = pathlib.Path(directory)
    ids = json.loads(args.tokens.read_text())
    windows = root/'windows.json'
    windows.write_text(json.dumps([ids]))
    def run(name, extra=None, success=True):
        config = root/(name+'.config.json')
        settings = {'objective': 'kl', 'float32_math': True, 'publish': False,
                    'monitor_window': 0, 'teacher_cache_max_bytes': 32000000}
        settings.update(extra or {})
        config.write_text(json.dumps(settings))
        report = root/(name+'.json')
        command = [str(args.binary), 'rco-global', str(args.model), str(args.manifest), str(windows),
                   str(root/'teacher'), str(root/(name+'.gguf')), str(budget), '3', 'CPU', str(report), str(config)]
        with (root/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if success:
            assert result.returncode == 0, (root/(name+'.log')).read_text()[-2000:]
            return json.loads(report.read_text())
        assert result.returncode != 0, 'stale checkpoint accepted'
    partial = run('partial', {'stop_after': 2})
    assert partial['completed_steps'] == 2 and not partial['complete']
    resumed = run('resumed', {'resume': partial['checkpoint']})
    full = run('full')
    for key in ('precision_logits', 'first_moment', 'second_moment', 'assignment', 'history', 'monitor'):
        assert resumed[key] == full[key], 'checkpoint replay mismatch: '+key
    assert full['expected_file_bytes'] <= budget
    assert any(h['raw_gradient_norm'] > 0 for h in full['history']), 'global objective is disconnected'
    assert max(abs(h['projected_gradient_dot']) for h in full['history']) < 1e-5
    assert full['precision_logits'] != partial['precision_logits'], 'precision logits did not update'
    checkpoint = json.loads(pathlib.Path(partial['checkpoint']).read_text())
    checkpoint['fingerprint']['source_identity'] = 'stale'
    bad = root/'stale.json'
    bad.write_text(json.dumps(checkpoint))
    run('stale', {'resume': str(bad)}, success=False)
    print('PASS: real global KL updates, exact-byte assignment and checkpoint replay')
