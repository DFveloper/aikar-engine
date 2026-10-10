"""Map sealed producer candidates to reproducible training and evaluation arms."""
import argparse
import copy
import json
import pathlib

parser = argparse.ArgumentParser(description=__doc__)
for name in ('ptq', 'gsq', 'imatrix', 'baselines', 'output', 'rco_report', 'gsq_rco_model'):
    parser.add_argument(name, type=pathlib.Path)
args = parser.parse_args()
assert not args.output.exists()
ptq = json.loads(args.ptq.read_text())
gsq = json.loads(args.gsq.read_text())
imatrix = json.loads(args.imatrix.read_text())
assert ptq['source_identity'] == gsq['source_identity'] == imatrix['source_identity']
assert [g['tensor'] for g in ptq['groups']] == [g['tensor'] for g in gsq['groups']] == [g['tensor'] for g in imatrix['groups']]
assert ptq['fixed_gguf_bytes'] == gsq['fixed_gguf_bytes'] == imatrix['fixed_gguf_bytes']
args.output.mkdir(parents=True)
selected = copy.deepcopy(gsq)
selection = []
for index, group in enumerate(selected['groups']):
    assert [c['producer'] for c in group['choices']] == ['ptq', 'ptq', 'ptq', 'gsq-rtn', 'gsq-gptq']
    losses = [json.loads((args.gsq.parent/f'{index}.{method}.json').read_text())['best_hard_mse'] for method in ('rtn', 'gptq')]
    best = 3 + int(losses[1] < losses[0])
    group['choices'] = [group['choices'][best], group['choices'][1], group['choices'][2]]
    assert [c['aligned_bytes'] for c in group['choices']] == [c['aligned_bytes'] for c in ptq['groups'][index]['choices']]
    selection.append({'tensor': group['tensor'], 'rtn_calibration_mse': losses[0],
                      'gptq_calibration_mse': losses[1], 'producer': group['choices'][0]['producer']})
manifest = args.output/'gsq-rco.manifest.json'
manifest.write_text(json.dumps(selected, indent=2)+'\n')
(args.output/'q2-selection.json').write_text(json.dumps({'policy': 'Lower original-input calibration reconstruction MSE; no held-out selection', 'groups': selection}, indent=2)+'\n')
static = json.loads((args.baselines/'static-mixed.assignment.json').read_text())
q4 = json.loads((args.baselines/'ptq-q4.assignment.json').read_text())
budget = json.loads((args.baselines/'report.json').read_text())['total_byte_budget']
rco = json.loads(args.rco_report.read_text())
assert rco['complete'] and rco['total_byte_budget'] == budget

def assignment(name, values):
    path = args.output/(name+'.assignment.json')
    path.write_text(json.dumps(values)+'\n')
    return str(path.resolve())

arms = [{'name': 'bf16', 'model': ptq['source_model']}]
for name, store, values in (
    ('ptq-q4', args.ptq, q4),
    ('imatrix-q4', args.imatrix, [0]*len(q4)),
    ('ptq-static', args.ptq, static),
    ('gsq-rtn', args.gsq, [3 if k == 0 else k for k in static]),
    ('gsq-gptq', args.gsq, [4 if k == 0 else k for k in static])):
    arms.append({'name': name, 'manifest': str(store.resolve()), 'assignment': assignment(name, values)})
arms.append({'name': 'rco', 'model': rco['output'], 'manifest': str(args.ptq.resolve()),
             'assignment': assignment('rco', rco['assignment'])})
arms.append({'name': 'gsq-rco', 'model': str(args.gsq_rco_model.resolve()), 'manifest': str(manifest.resolve()),
             'assignment': str((args.output/'gsq-rco.assignment.json').resolve())})
(args.output/'suite.json').write_text(json.dumps({'source': ptq['source_model'], 'budget': budget, 'arms': arms}, indent=2)+'\n')
print(manifest)
