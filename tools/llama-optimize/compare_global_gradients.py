"""Compare CUDA adjoints to a CPU graph validated by finite differences."""
import argparse
import json
import pathlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('cpu', type=pathlib.Path)
parser.add_argument('gpu', type=pathlib.Path)
parser.add_argument('report', type=pathlib.Path)
args = parser.parse_args()
cpu, gpu = [json.loads(path.read_text()) for path in (args.cpu, args.gpu)]
assert not args.report.exists()
assert cpu['passed'], 'CPU finite difference reference did not pass'
for key in ('tokens', 'float32_math', 'candidate_manifest'):
    assert cpu[key] == gpu[key], 'different graph conditions: '+key
assert gpu['cached_vs_training_logits_max_error'] < 5e-3
checks = []
for a, b in zip(cpu['gradient_checks'], gpu['gradient_checks'], strict=True):
    assert a['objective'] == b['objective']
    assert b['forward_recompute_max_error'] == 0
    errors = [abs(x-y) for x, y in zip(a['gradient'], b['gradient'], strict=True)]
    tolerances = [max(1e-5, abs(x)*1e-3) for x in a['gradient']]
    assert all(error < tolerance for error, tolerance in zip(errors, tolerances))
    checks.append({'objective': a['objective'], 'max_adjoint_error': max(errors),
                   'gradient_components': len(errors),
                   'max_error_over_tolerance': max(error/tolerance for error, tolerance in zip(errors, tolerances)),
                   'cpu_loss': a['loss'], 'gpu_loss': b['loss']})
report = {'method': 'CUDA adjoint against CPU finite-difference-validated native graph',
          'cpu_report': str(args.cpu), 'gpu_report': str(args.gpu),
          'gpu_finite_difference_passed': gpu['passed'], 'checks': checks, 'passed': True,
          'limitation': 'GPU finite differences retain float32 forward rounding; inspect the GPU report separately'}
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
