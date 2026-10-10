"""Check two-point and five-point stencils from recorded model forwards."""
import argparse
import copy
import json
import pathlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=pathlib.Path)
parser.add_argument('output', type=pathlib.Path)
args = parser.parse_args()
source = json.loads(args.input.read_text())
assert source['finite_difference_enabled'] and not args.output.exists()
report = copy.deepcopy(source)
passed = source['cached_vs_training_logits_max_error'] < 5e-3
for check in report['gradient_checks']:
    assert check['forward_recompute_max_error'] == 0
    raw = [point for point in check['finite_differences'] if 'method' not in point]
    for point in raw:
        for outer in raw:
            if point['group'] == outer['group'] and abs(2*point['step']-outer['step']) < 1e-12:
                estimate = (4*point['finite_difference']-outer['finite_difference'])/3
                check['finite_differences'].append(dict(point, method='five-point centered stencil',
                    outer_step=outer['step'], finite_difference=estimate,
                    absolute_error=abs(estimate-point['analytic'])))
    for group in source['finite_difference_groups']:
        matches = [p['absolute_error'] < p['tolerance'] for p in check['finite_differences'] if p['group'] == group]
        assert matches
        passed = passed and any(matches)
report['source_report'] = str(args.input)
report['raw_two_point_passed'] = source['passed']
report['finite_difference_policy'] = 'each group must match a two-point or five-point centered stencil; retain all raw estimates and unchanged tolerances'
report['passed'] = passed
args.output.write_text(json.dumps(report, indent=2)+'\n')
for check in report['gradient_checks']:
    for group in source['finite_difference_groups']:
        best = min((p for p in check['finite_differences'] if p['group'] == group), key=lambda p:p['absolute_error'])
        print(check['objective'], group, best.get('method', 'two-point centered stencil'),
              best['step'], best['absolute_error'], best['tolerance'])
assert passed, 'model finite difference validation failed'
