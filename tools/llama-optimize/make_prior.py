"""Run pinned upstream RTN/GPTQ initialization on a captured linear."""
import argparse
import json
import logging
import pathlib
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
from types import ModuleType
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', type=pathlib.Path, required=True)
parser.add_argument('--capture', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--method', choices=['rtn','gptq'], required=True)
args = parser.parse_args()
assert not args.output.exists() and not args.output.with_suffix('.json').exists()
sys.path.insert(0,str(args.upstream))
# Load the pinned prior without unrelated dataset registration.
namespace = ModuleType('src')
namespace.__path__ = [str(args.upstream/'src')]
sys.modules['src'] = namespace
from src.prior.gptq import GPTQ, rtn_quantize
from src.prior.quant import Quantizer
torch.set_num_threads(8)
with args.capture.open('rb') as f:
    magic,columns,rows,samples=struct.unpack('<IQQQ',f.read(28))
    assert magic==0x31515347
    weights=np.frombuffer(f.read(columns*rows*4),dtype='<f4').copy().reshape(rows,columns)
    inputs=np.frombuffer(f.read(columns*samples*4),dtype='<f4').copy().reshape(samples,columns)
layer=torch.nn.Linear(columns,rows,bias=False)
layer.weight.data.copy_(torch.from_numpy(weights))
config=SimpleNamespace(gptq=SimpleNamespace(groupsize=64,wbits=2,sym=True,trits=False),quantization=SimpleNamespace(gsq_bits=2))
start=time.monotonic()
if args.method=='rtn':
    q,scales=rtn_quantize(layer,config,'cpu',torch.float32)
    hessian_shape=None
else:
    optimizer=GPTQ(layer,'captured_linear',config,'cpu',torch.float32)
    optimizer.quantizer=Quantizer()
    optimizer.quantizer.configure(2,perchannel=True,sym=True,mse=True)
    optimizer.add_batch(torch.from_numpy(inputs).unsqueeze(0),None)
    hessian_shape=list(optimizer.H.shape)
    assert optimizer.H.shape==(columns,columns)
    q,scales=optimizer.fasterquant(logging,blocksize=128,percdamp=.01,groupsize=64)
    optimizer.free()
grid=q/scales.repeat_interleave(64,dim=1)
assert torch.isfinite(q).all() and torch.isfinite(scales).all()
assert grid.min()>=-2.00001 and grid.max()<=1.00001
assert (grid-grid.round()).abs().max()<1e-4
with args.output.open('xb') as f:
    f.write(struct.pack('<IQQ',0x31525047,columns,rows))
    f.write(q.detach().contiguous().numpy().astype('<f4').tobytes())
    f.write(scales.detach().contiguous().numpy().astype('<f4').tobytes())
report=dict(method=args.method,upstream=str(args.upstream),revision=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
            hessian_shape=hessian_shape,columns=columns,rows=rows,samples=samples,group_size=64,damping=.01,
            seconds=time.monotonic()-start,reference_implementation=True,full_model_reproduction=False)
args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
