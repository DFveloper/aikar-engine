"""Prepare exact-size static precision maps and weighted Q4 calibration."""
import argparse
import hashlib
import json
import pathlib
import struct
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=pathlib.Path, required=True)
parser.add_argument('--captures', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--budget', type=int, required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
captures = json.loads(args.captures.read_text())
assert captures['source_identity'] == manifest['source_identity']
assert [g['tensor'] for g in manifest['groups']] == [c['tensor'] for c in captures['captures']]
assert not args.output.exists()
args.output.mkdir(parents=True)
groups = manifest['groups']
costs = [[c['aligned_bytes'] for c in g['choices']] for g in groups]
assert all([c['type'] for c in g['choices']] == ['q2_0', 'q4_0', 'q8_0'] for g in groups)
parameters = [int(np.prod(g['shape'])) for g in groups]

def file_bytes(assignment):
    return manifest['fixed_gguf_bytes'] + sum(row[k] for row, k in zip(costs, assignment))

uniform = [1]*len(groups)
assert file_bytes(uniform) <= args.budget
static = [0 if g['tensor'].endswith(('ffn_down.weight', 'ffn_gate.weight')) else
          2 if g['tensor'].endswith('ffn_up.weight') else 1 for g in groups]
q2 = [i for i, k in enumerate(static) if k == 0]
q8 = sorted((i for i, k in enumerate(static) if k == 2), key=lambda i:parameters[i])
assert file_bytes(static) <= args.budget
best = None
for count in range(len(q8)+1):
    candidate = static.copy()
    for i in q8[:count]:
        candidate[i] = 1
    remaining = args.budget-file_bytes(candidate)
    states = {0: []}
    for i in q2:
        gain = costs[i][1]-costs[i][0]
        for used, selected in list(states.items()):
            if used+gain <= remaining and used+gain not in states:
                states[used+gain] = selected+[i]
    for gain in sorted(states, reverse=True):
        assignment = candidate.copy()
        for i in states[gain]:
            assignment[i] = 1
        fractions = [sum(p for p, k in zip(parameters, assignment) if k == bit)/sum(parameters) for bit in (0, 2)]
        if fractions[0] < .25 or fractions[1] < .10:
            continue
        score = (file_bytes(assignment), -count, -len(states[gain]))
        if best is None or score > best[0]:
            best = score, assignment, fractions
        break
assert best is not None
static = best[1]
for name, assignment in (('ptq-q4', uniform), ('static-mixed', static)):
    (args.output/(name+'.assignment.json')).write_text(json.dumps(assignment)+'\n')
summary = {'total_byte_budget': args.budget, 'uniform_q4_bytes': file_bytes(uniform),
           'static_mixed_bytes': file_bytes(static), 'static_remaining_bytes': args.budget-file_bytes(static),
           'static_q2_parameter_fraction': best[2][0], 'static_q8_parameter_fraction': best[2][1],
           'policy': 'FFN down/gate Q2 and up Q8; deterministic Q4 exchanges maximize bytes within this static family; retain at least 25% Q2 and 10% Q8 active parameters',
           'captures_tokens': captures['tokens'], 'source_identity': manifest['source_identity']}
weighted = []
for index, (group, capture) in enumerate(zip(groups, captures['captures'])):
    path = pathlib.Path(capture['file'])
    with path.open('rb') as stream:
        magic, columns, rows, samples = struct.unpack('<IQQQ', stream.read(28))
    assert magic == 0x31515347 and [columns, rows] == group['shape']
    assert path.stat().st_size == 28 + 4*columns*(rows+samples)
    inputs = np.memmap(path, dtype='<f4', mode='r', offset=28+4*columns*rows, shape=(samples, columns))
    assert np.isfinite(inputs).all()
    importance = np.mean(np.square(inputs.astype(np.float64)), axis=0).astype('<f4')
    assert np.isfinite(importance).all() and np.any(importance > 0)
    output = args.output/f'{index}.imatrix.f32'
    output.write_bytes(importance.tobytes())
    weighted.append({'tensor': group['tensor'], 'candidates': [{'type': 'q4_0', 'producer': 'imatrix-ptq',
                    'imatrix_file': str(output.resolve()), 'imatrix_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}]})
(args.output/'imatrix.config.json').write_text(json.dumps({'tensors': weighted}, indent=2)+'\n')
(args.output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
