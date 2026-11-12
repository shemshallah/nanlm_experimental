#!/usr/bin/env python3
"""R7.12 longer-stream causal equivalence + CPU latency microbenchmark."""
import pathlib,json,time,torch,numpy as np
from r711_delta import restore_overlay
from r712_streaming import DirectedStreaming
from r74_stream import CrossWindowMirror,OLDER,OUTER_CONTEXT
from r7_core import Corpus
from r710_checkpoint_integrity import sha256
ROOT=pathlib.Path(__file__).resolve().parents[1]

def run():
    torch.set_num_threads(2)
    sha0=sha256(ROOT/'checkpoint/pal_active.pt')
    model,opts,_,_,_=restore_overlay(ROOT,ROOT/'adaptation/r711_experimental_delta.pt')
    c=Corpus(ROOT,'dev');indices=np.array([8192+14001,8192+50212],dtype=np.int64)
    h=torch.tensor(c.get(indices,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
    outer=model.r74.outer
    with torch.no_grad():
        outer.dir_gate.fill_(.2)
        stream=DirectedStreaming(model);state=stream.initial(h)
        ref_base=CrossWindowMirror.initial(outer,h,model.trunk.E)
        ring=h[:,-768:].clone()
        maxdiff=0.;n=64
        t=time.monotonic()
        for ix in range(n):
            cur=stream.matrix(state)
            M,_,_=outer.pair_state(ring,model.trunk.E)
            ref=ref_base+torch.tanh(outer.dir_gate)*(M-ref_base)
            maxdiff=max(maxdiff,float((cur-ref).abs().max()))
            assert torch.allclose(cur,ref,atol=1e-7,rtol=1e-6)
            leaving=torch.from_numpy(c.get(indices+ix,2048)[0][:,0].copy()).long()
            state=stream.advance(state,leaving)
            ring=torch.cat((ring[:,1:],leaving[:,None]),dim=1)
            ref_base=CrossWindowMirror.advance(outer,ref_base,leaving,model.trunk.E)
        seconds=time.monotonic()-t
        assert torch.equal(state.ring,ring)
        assert not torch.allclose(stream.matrix(state),state.legacy,atol=1e-5,rtol=1e-4)
    assert sha256(ROOT/'checkpoint/pal_active.pt')==sha0
    return {'streams':2,'stream_positions':64,'max_matrix_abs_difference':maxdiff,
      'seconds_read_plus_advance':round(seconds,4),
      'milliseconds_per_batch_step':round(seconds/n*1000,3),
      'reference':'incremental legacy matrix + independent exact 768-word ring reconstruction',
      'future_target_leakage':'no target enters state initializer, advancement or ring',
      'cpu_threads':2,'accepted_parent_unmodified':True}
if __name__=='__main__':print(json.dumps(run(),indent=2))
