"""Compare GGML forward and weight gradient to PyTorch on real inputs."""
import json
import pathlib
import struct
import sys
import numpy as np
import torch

torch.set_num_threads(8)
with open(sys.argv[1],'rb') as f:
    magic,columns,rows,samples=struct.unpack('<IQQQ',f.read(28))
    assert magic==0x31515347
    w=torch.from_numpy(np.frombuffer(f.read(columns*rows*4),dtype='<f4').copy().reshape(rows,columns)).requires_grad_()
    x=torch.from_numpy(np.frombuffer(f.read(columns*samples*4),dtype='<f4').copy().reshape(samples,columns))
out=x@w.T
dy=torch.sin(torch.arange(samples*rows,dtype=torch.float64)*.017).float().reshape(samples,rows)*.1
(out*dy).sum().backward()
raw=np.fromfile(sys.argv[2],dtype='<f4')
assert raw.size==samples*rows+columns*rows
actual_out=torch.from_numpy(raw[:samples*rows].reshape(samples,rows))
actual_dw=torch.from_numpy(raw[samples*rows:].reshape(rows,columns))
errors=dict(forward_max_abs=(actual_out-out.detach()).abs().max().item(),backward_max_abs=(actual_dw-w.grad).abs().max().item(),
            forward_rms=(actual_out-out.detach()).square().mean().sqrt().item(),backward_rms=(actual_dw-w.grad).square().mean().sqrt().item())
torch.testing.assert_close(actual_out,out.detach(),atol=.003,rtol=.001)
torch.testing.assert_close(actual_dw,w.grad,atol=.003,rtol=.001)
print(json.dumps(errors))
if len(sys.argv)>3:
    p=pathlib.Path(sys.argv[3]);assert not p.exists();p.write_text(json.dumps(errors,indent=2)+'\n')
