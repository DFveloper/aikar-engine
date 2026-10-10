"""Check held-out window scoring and teacher cache validation on a real model."""
import argparse
import json
import pathlib
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
for name in ('binary', 'model', 'tokens'):
    parser.add_argument(name, type=pathlib.Path)
args = parser.parse_args()
ids = json.loads(args.tokens.read_text())
with tempfile.TemporaryDirectory(prefix='rco-corpus-test-') as directory:
    root = pathlib.Path(directory)
    windows = root/'windows.json'
    windows.write_text(json.dumps([ids, ids[:-1]]))
    def run(mode, name, success=True):
        report = root/(name+'.json')
        result = subprocess.run([str(args.binary), 'corpus-eval', str(args.model), str(windows),
                                 str(root/'teacher'), str(report), '0', mode], capture_output=True, text=True)
        assert (result.returncode == 0) == success, result.stderr[-2000:]
        return json.loads(report.read_text()) if success else None
    first = run('write', 'baseline')
    second = run('read', 'reload')
    assert first['evaluation_tokens'] == second['evaluation_tokens'] == 2*len(ids)-3
    assert abs(first['ce_loss']-second['ce_loss']) < 1e-9
    assert abs(second['teacher_kl']) < 1e-9 and second['logits_mse'] == 0
    assert len(second['windows']) == 2
    run('write', 'overwrite', False)
    metadata = root/'teacher/0.json'
    changed = json.loads(metadata.read_text())
    changed['tokens'][0] += 1
    metadata.write_text(json.dumps(changed))
    run('read', 'stale', False)
    print('PASS: all held-out targets scored, identical teacher IDs, stale cache rejected')
