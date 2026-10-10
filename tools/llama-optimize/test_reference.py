"""Independent numerical oracle for the C++ GSQ/RCO PoC."""
import itertools
import json
import pathlib
import subprocess
import sys
import torch

binary = pathlib.Path(sys.argv[1])
assert binary.is_file(), 'GSQ/RCO executable has not been implemented'
r = json.loads(subprocess.check_output([str(binary), 'self-test'], text=True))
assert 'global_loss' in r, 'missing model CE/KL candidate-gradient check'
for case in r['global_loss']:
    alpha = torch.tensor(case['alpha'], dtype=torch.float64, requires_grad=True).reshape(2, 3)
    noise = torch.tensor(case['noise'], dtype=torch.float64).reshape(2, 3)
    inputs = torch.tensor(case['inputs'], dtype=torch.float64).reshape(3, 2)
    candidates = [torch.tensor(x, dtype=torch.float64).reshape(3, 2, 2) for x in case['candidates']]
    tau = case['temperature']
    labels = torch.tensor(case['targets'], dtype=torch.int64)
    teacher = torch.tensor(case['teacher_log_probs'], dtype=torch.float64).reshape(3, 2)
    soft = torch.softmax((alpha + noise) / tau, dim=-1)
    if case['hard']:
        onehot = torch.nn.functional.one_hot(torch.tensor(case['assignment']), 3).double()
        probs = onehot - soft.detach() + soft
    else:
        probs = soft
    weights = [(probs[i, :, None, None] * candidates[i]).sum(0) for i in range(2)]
    logits = torch.tanh(inputs @ weights[0].T) @ weights[1].T
    if case['objective'] == 'ce':
        loss = torch.nn.functional.cross_entropy(logits, labels)
    else:
        loss = (teacher.exp() * (teacher - logits.log_softmax(-1))).sum(-1).mean()
    grad = torch.autograd.grad(loss, alpha)[0]
    actual = torch.tensor(case['gradient'], dtype=torch.float64).reshape(2, 3)
    assert abs(loss.item() - case['loss']) < 1e-6
    assert (grad - actual).abs().max().item() < 1e-5
    assert grad[0].abs().max() > 1e-5, 'early candidate disconnected from global loss'
    if not case['hard']:
        def smooth_loss(a):
            p = torch.softmax((a + noise) / tau, dim=-1)
            w = [(p[i, :, None, None] * candidates[i]).sum(0) for i in range(2)]
            z = torch.tanh(inputs @ w[0].T) @ w[1].T
            if case['objective'] == 'ce':
                return torch.nn.functional.cross_entropy(z, labels)
            return (teacher.exp() * (teacher - z.log_softmax(-1))).sum(-1).mean()
        max_error = 0.0
        for h in (1e-3, 1e-4, 1e-5):
            for index in range(alpha.numel()):
                delta = torch.zeros_like(alpha)
                delta.flatten()[index] = h
                fd = (smooth_loss(alpha.detach() + delta) - smooth_loss(alpha.detach() - delta)) / (2 * h)
                max_error = max(max_error, abs(fd.item() - actual.flatten()[index].item()))
        assert max_error < 1e-5
        print('global', case['objective'], 'smooth finite_difference_max_error', max_error)
    else:
        print('global', case['objective'], 'hard STE VJP max_abs_error', (grad-actual).abs().max().item())
import tempfile
import numpy as np
import gguf
with tempfile.TemporaryDirectory(prefix='rco-store-test-') as directory:
    root = pathlib.Path(directory)
    source = root / 'source.gguf'
    writer = gguf.GGUFWriter(source, 'test')
    writer.add_tensor('linear.weight', np.linspace(-1, 1, 128, dtype=np.float32).reshape(2, 64))
    writer.add_tensor('fixed.weight', np.ones(5, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    config = root / 'config.json'
    config.write_text(json.dumps({'tensors': [{'tensor': 'linear.weight', 'candidates': [
        {'type': 'q2_0', 'producer': 'ptq'}, {'type': 'q4_0', 'producer': 'ptq'},
        {'type': 'q8_0', 'producer': 'ptq'}]}]}))
    command = [str(binary), 'candidate-store', str(source), str(config), str(root/'store'), str(root/'report.json')]
    run = subprocess.run(command, text=True, capture_output=True)
    assert run.returncode == 0, 'candidate store unavailable: ' + run.stderr[-1000:]
    manifest = json.loads((root/'store/manifest.json').read_text())
    choices = manifest['groups'][0]['choices']
    import hashlib
    for choice in choices:
        data = pathlib.Path(choice['file']).read_bytes()
        assert len(data) == choice['bytes']
        assert hashlib.sha256(data).hexdigest() == choice['sha256']
    assert [choice['bytes'] for choice in choices] == [36, 72, 136]
    assert [choice['aligned_bytes'] for choice in choices] == [64, 96, 160]
    assert subprocess.run(command, capture_output=True).returncode != 0, 'candidate store overwritten'
    validate = [str(binary), 'candidate-validate', str(source), str(root/'store/manifest.json')]
    assert subprocess.run(validate, capture_output=True).returncode == 0, 'candidate manifest cannot be verified'
    original = (root/'store/manifest.json').read_text()
    for mutate in ('cost', 'shape', 'duplicate', 'fixed'):
        bad = json.loads(original)
        if mutate == 'cost':
            bad['groups'][0]['choices'][0]['aligned_bytes'] += 32
        elif mutate == 'shape':
            bad['groups'][0]['shape'] = [32, 4]
        elif mutate == 'duplicate':
            bad['groups'][0]['choices'].append(bad['groups'][0]['choices'][0])
        else:
            bad['fixed_gguf_bytes'] -= 32
        (root/'store/manifest.json').write_text(json.dumps(bad))
        assert subprocess.run(validate, capture_output=True).returncode != 0, 'invalid manifest accepted: ' + mutate
    (root/'store/manifest.json').write_text(original)
    assignment = root/'assignment.json'
    assignment.write_text('[1]')
    expected_size = manifest['fixed_gguf_bytes'] + choices[1]['aligned_bytes']
    publish = [str(binary), 'candidate-publish', str(source), str(root/'store/manifest.json'), str(assignment),
               str(root/'mixed.gguf'), str(expected_size), str(root/'publish.json')]
    result = subprocess.run(publish, capture_output=True, text=True)
    assert result.returncode == 0, 'packed GGUF publication failed: ' + result.stderr[-1000:]
    assert (root/'mixed.gguf').stat().st_size == expected_size
    verify = [str(binary), 'candidate-verify', str(source), str(root/'store/manifest.json'), str(assignment),
              str(root/'mixed.gguf'), str(root/'verified.json')]
    verified = subprocess.run(verify, capture_output=True, text=True)
    assert verified.returncode == 0, 'GGUF payload verification failed: '+verified.stderr[-1000:]
    mixed = gguf.GGUFReader(root/'mixed.gguf')
    chosen = next(t for t in mixed.tensors if t.name == 'linear.weight')
    assert chosen.data.tobytes() == pathlib.Path(choices[1]['file']).read_bytes(), 'candidate was requantized'
    too_small = publish.copy()
    too_small[5] = str(root/'over-budget.gguf')
    too_small[6] = str(expected_size-1)
    too_small[7] = str(root/'over-budget.json')
    assert subprocess.run(too_small, capture_output=True).returncode != 0
    assert not (root/'over-budget.gguf').exists(), 'over-budget model was published'
    import struct
    capture = root/'linear.capture'
    weights = np.linspace(-1, 1, 128, dtype=np.float32)
    inputs = np.sin(np.arange(256, dtype=np.float32)*.1)
    capture.write_bytes(struct.pack('<IQQQ', 0x31515347, 64, 2, 4) + weights.tobytes() + inputs.tobytes())
    gsq_payload = root/'gsq.bin'
    gsq_run = subprocess.run([str(binary), 'gsq-pack', str(source), str(capture), 'linear.weight', str(gsq_payload),
                              '2', 'CPU', str(root/'gsq.json')], capture_output=True, text=True)
    assert gsq_run.returncode == 0, 'GSQ direct payload export failed: ' + gsq_run.stderr[-1000:]
    assert gsq_payload.stat().st_size == 36
    config.write_text(json.dumps({'tensors': [{'tensor': 'linear.weight', 'candidates': [
        {'type': 'q2_0', 'producer': 'gsq-rtn', 'packed_file': str(gsq_payload),
         'sha256': hashlib.sha256(gsq_payload.read_bytes()).hexdigest()}]}]}))
    imported = [str(binary), 'candidate-store', str(source), str(config), str(root/'gsq-store'), str(root/'gsq-store.json')]
    imported_run = subprocess.run(imported, capture_output=True, text=True)
    assert imported_run.returncode == 0, 'GSQ candidate import failed: ' + imported_run.stderr[-1000:]
    gsq_manifest = json.loads((root/'gsq-store/manifest.json').read_text())
    gsq_choice = gsq_manifest['groups'][0]['choices'][0]
    assert pathlib.Path(gsq_choice['file']).read_bytes() == gsq_payload.read_bytes(), 'GSQ candidate was requantized'
    gsq_assignment = root/'gsq-assignment.json'
    gsq_assignment.write_text('[0]')
    gsq_size = gsq_manifest['fixed_gguf_bytes'] + gsq_choice['aligned_bytes']
    subprocess.run([str(binary), 'candidate-publish', str(source), str(root/'gsq-store/manifest.json'),
        str(gsq_assignment), str(root/'gsq-mixed.gguf'), str(gsq_size), str(root/'gsq-published.json')], check=True, capture_output=True)
    subprocess.run([str(binary), 'candidate-verify', str(source), str(root/'gsq-store/manifest.json'),
        str(gsq_assignment), str(root/'gsq-mixed.gguf'), str(root/'gsq-verified.json')], check=True, capture_output=True)
    importance = root/'importance.f32'
    importance.write_bytes(np.tile(np.array([1000, .001, .001, .001], dtype=np.float32), 16).tobytes())
    config.write_text(json.dumps({'tensors': [{'tensor': 'linear.weight', 'candidates': [
        {'type': 'q4_0', 'producer': 'imatrix-ptq', 'imatrix_file': str(importance),
         'imatrix_sha256': hashlib.sha256(importance.read_bytes()).hexdigest()}]}]}))
    weighted = [str(binary), 'candidate-store', str(source), str(config), str(root/'weighted'), str(root/'weighted.json')]
    result = subprocess.run(weighted, capture_output=True, text=True)
    assert result.returncode == 0, 'weighted Q4_0 unavailable: '+result.stderr[-1000:]
    wc = json.loads((root/'weighted/manifest.json').read_text())['groups'][0]['choices'][0]
    assert pathlib.Path(wc['file']).read_bytes() != pathlib.Path(choices[1]['file']).read_bytes(), 'IMatrix was ignored'
    captures_report = root/'captures.json'
    captures_report.write_text(json.dumps({'source_identity': manifest['source_identity'],
        'captures': [{'tensor': 'linear.weight', 'file': str(capture)}]}))
    generate = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name('generate_gsq_candidates.py')),
        '--binary', str(binary.resolve()), '--model', str(source), '--captures', str(captures_report),
        '--manifest', str(root/'store/manifest.json'), '--upstream', sys.argv[2], '--output', str(root/'generated'),
        '--steps', '2', '--jobs', '1'], capture_output=True, text=True)
    assert generate.returncode == 0, 'GSQ candidate pipeline failed: '+generate.stderr[-1500:]
    generated = json.loads((root/'generated/manifest.json').read_text())
    assert len(generated['groups'][0]['choices']) == 5
    assert [c['producer'] for c in generated['groups'][0]['choices'][-2:]] == ['gsq-rtn', 'gsq-gptq']
    source_bytes = bytearray(source.read_bytes())
    source_bytes[-32] ^= 1
    source.write_bytes(source_bytes)
    assert subprocess.run(validate, capture_output=True).returncode != 0, 'changed source weights accepted'
    source_bytes[-32] ^= 1
    source.write_bytes(source_bytes)
    payload = pathlib.Path(choices[0]['file'])
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    assert subprocess.run(validate, capture_output=True).returncode != 0, 'corrupt candidate accepted'
    print('PASS: packed candidate store, exact block/alignment costs and no-clobber')
torch.set_num_threads(2)
l = torch.tensor(r['logits'], dtype=torch.float64, requires_grad=True).reshape(-1, 4)
s = torch.tensor(r['scales'], dtype=torch.float64, requires_grad=True)
noise = torch.tensor(r['noise'], dtype=torch.float64).reshape(-1, 4)
values = torch.tensor([-2., -1., 0., 1.], dtype=torch.float64)
p = torch.softmax((l * 2.3 + noise) / 0.7, -1)
out = (p * values).sum(-1) * s.repeat_interleave(64)
dy = torch.tensor(r['dy'], dtype=torch.float64)
gl, gs = torch.autograd.grad((out * dy).sum(), (l, s))
def close(actual, expected, name, tol=2e-6):
    error = (torch.tensor(actual, dtype=torch.float64).reshape(expected.shape) - expected).abs().max().item()
    assert error < tol, (name, error)
    print(name, 'max_abs_error', error)
close(r['forward'], out.detach(), 'GSQ forward')
close(r['grad_logits'], gl, 'GSQ logits backward')
close(r['grad_scales'], gs, 'GSQ scales backward', 1e-5)
for i in [0, 51, 271]:
    flat = l.detach().flatten().clone()
    h = 1e-5
    def objective(v):
        pp = torch.softmax((v.reshape(-1, 4) * 2.3 + noise) / 0.7, -1)
        return (((pp * values).sum(-1) * s.detach().repeat_interleave(64)) * dy).sum()
    flat[i] += h
    hi = objective(flat)
    flat[i] -= 2*h
    lo = objective(flat)
    assert abs(((hi-lo)/(2*h)).item() - gl.flatten()[i].item()) < 1e-7
param = torch.tensor(r['lion_before'], dtype=torch.float64)
moment = torch.tensor(r['moment_before'], dtype=torch.float64)
g = torch.tensor(r['lion_grad'], dtype=torch.float64)
expected = param * (1-0.01*0.2) - 0.01 * (0.9*moment+0.1*g).sign()
close(r['lion_after'], expected, 'Lion parameter')
close(r['moment_after'], .95*moment+.05*g, 'Lion momentum')
from lion_pytorch import Lion
lp=torch.nn.Parameter(param.float().clone())
lion=Lion([lp],lr=.01,betas=(.9,.95),weight_decay=.2)
lion.state[lp]['exp_avg']=moment.float().clone()
lp.grad=g.float()
lion.step()
close(r['lion_after'],lp.detach(),'lion-pytorch update')
ag = torch.tensor(r['adam_gradient'], dtype=torch.float64)
ap = torch.tensor(r['adam_before'], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([ap], lr=.01)
ap.grad = ag
optimizer.step()
close(r['adam_after'], ap.detach(), 'GGML Adam with zero weight decay')
close(r['adam_first'], .1*ag, 'Adam first moment')
close(r['adam_second'], .001*ag*ag, 'Adam second moment')
assert r['replay_equal'] and r['packing_error'] < 0.002
assert abs(r['projection_dot']) < 1e-6
assert abs(r['transport_dot']) < 1e-6
assert abs(r['budget_after']-r['target_budget']) < 1e-5
costs, scores, budget = r['costs'], r['scores'], r['byte_budget']
feasible = [(sum(scores[i][k] for i,k in enumerate(a)),a) for a in itertools.product(range(3),repeat=3) if sum(costs[i][k] for i,k in enumerate(a)) <= budget]
best = max(feasible)[0]
assert abs(r['assignment_score']-best) < 1e-6
assert r['assigned_bytes'] <= budget
print('PASS: gradients, RNG replay, Lion, packing, manifold and exact-byte DP')

if len(sys.argv) > 2:
    import contextlib
    import importlib.util
    from unittest.mock import patch
    path = pathlib.Path(sys.argv[2]) / 'src/quantization/gumbel_quantizer_2bit.py'
    spec = importlib.util.spec_from_file_location('original_gsq_2bit', path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    ql = torch.tensor(r['logits'], dtype=torch.float64).reshape(2,64,4).permute(2,0,1).contiguous().requires_grad_()
    qs = torch.tensor(r['scales'], dtype=torch.float32).reshape(2,1).requires_grad_()
    eps = 1e-8
    fixed_noise = noise.reshape(2,64,4).permute(2,0,1).contiguous()
    # Supply identical noise; the upstream function otherwise requires a CUDA RNG.
    uniform = torch.exp(-torch.exp(-fixed_noise)-eps)-eps
    with patch('torch.cuda.get_rng_state', return_value=None), patch('torch.cuda.set_rng_state'), patch('torch.random.fork_rng', return_value=contextlib.nullcontext()), patch('torch.rand_like', return_value=uniform):
        result = original.GumbelSoftmaxFunction.apply(ql,qs,values,torch.zeros(64,dtype=torch.int64),.7,2.3,'cpu')
        original_grad = torch.autograd.grad((result * dy.reshape(2,64)).sum(),(ql,qs))
    close(r['forward'], result.detach().flatten(), 'upstream GSQ forward')
    close(r['grad_logits'], original_grad[0].permute(1,2,0).flatten(), 'upstream GSQ backward')
    close(r['grad_scales'], original_grad[1].flatten(), 'upstream GSQ scale gradient',1e-5)
    print('PASS: original GSQ custom autograd with identical supplied noise')

if len(sys.argv)>3:
    import importlib.util
    path=pathlib.Path(sys.argv[3])/'src/manifold.py'
    spec=importlib.util.spec_from_file_location('original_rco_manifold',path)
    manifold=importlib.util.module_from_spec(spec);spec.loader.exec_module(manifold)
    costs=torch.tensor(r['costs'],dtype=torch.float64)
    alpha=torch.tensor(r['rco_alpha_before'],dtype=torch.float64).reshape(3,3).requires_grad_()
    alpha.grad=torch.tensor(r['rco_gradient_before'],dtype=torch.float64).reshape(3,3)
    manifold.project_gradient(alpha,costs)
    close(r['rco_projected'],alpha.grad.flatten(),'upstream tangent projection')
    projected=alpha.grad.detach().clone()
    manifold.retraction(alpha,costs,216.,max_iter=100,tol=1e-9)
    close(r['rco_retracted'],alpha.detach().flatten(),'upstream retraction')
    alpha.grad=projected
    manifold.project_gradient(alpha,costs)
    close(r['rco_transported'],alpha.grad.flatten(),'upstream transport projection')
    ra=torch.tensor(r['synthetic_alpha_before'],dtype=torch.float64).reshape(3,3)
    pa=torch.nn.Parameter(ra.float().clone())
    opt=torch.optim.Adam([pa],lr=.05)
    wanted=torch.tensor([[0.,1.,0.]]*3,dtype=torch.float64)
    for iteration in range(20):
        ra=ra.detach().requires_grad_()
        loss=(torch.softmax(ra,-1)-wanted).square().sum()
        loss.backward();manifold.project_gradient(ra,costs)
        pa.data.copy_(ra.float());pa.grad=ra.grad.float();opt.step()
        ra=pa.detach().double().clone()
        manifold.retraction(ra,costs,216.,max_iter=100,tol=1e-9)
        normal=manifold.budget_normal(ra,costs)
        first=opt.state[pa]['exp_avg'].double()
        first-=((first*normal).sum()/(normal.square().sum()+1e-12))*normal
        opt.state[pa]['exp_avg'].copy_(first.float())
    close(r['synthetic_alpha_after'],ra.flatten(),'20-step Riemannian Adam',5e-5)
    assert r['synthetic_budget_residual']<1e-8
    assert loss.item()<r['synthetic_initial_loss']
    print('PASS: upstream RCO primitives and 20-step projected Adam')
