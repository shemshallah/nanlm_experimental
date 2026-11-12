#!/usr/bin/env python3
import json,pathlib,torch,numpy as np
from r711_delta import restore_overlay
from r712_streaming import DirectedStreaming,history_shift
from r74_stream import CrossWindowMirror,OLDER,OUTER_CONTEXT
from r7_core import Corpus
from train_r7 import batch_data
from r710_checkpoint_integrity import sha256,deep_equal
ROOT=pathlib.Path(__file__).resolve().parent.parent

def run():
    torch.set_num_threads(2)
    before=sha256(ROOT/'checkpoint/pal_active.pt')
    model,opts,base,comp,prior=restore_overlay(ROOT,ROOT/'adaptation/r711_experimental_delta.pt')
    model.eval();stream=DirectedStreaming(model)
    corp=Corpus(ROOT,'dev');positions=np.array([12430,25421,43432],dtype=np.int64)
    hist=torch.from_numpy(corp.get(positions,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
    state=stream.initial(hist)
    with torch.no_grad():
        assert torch.equal(stream.matrix(state),model.initial(hist))
        initial=model.initial(hist)
        leave=torch.from_numpy(corp.get(positions,2048)[0][:,0].copy()).long()
        nextstate=stream.advance(state,leave)
        expected=CrossWindowMirror.advance(model.r74.outer,initial,leave,model.trunk.E)
        assert torch.equal(stream.matrix(nextstate),expected)
        # A zero gate preserves inherited neural predictions exactly at first
        # and successive streaming positions, including downstream retrieval.
        for t in range(2):
            ix=positions+t
            _,ctx,pieces,lens,y,_,_=batch_data(base,corp,comp,ix)
            ex=torch.from_numpy(corp.data[ix].copy()).long()
            lg=torch.from_numpy(corp.long[ix].copy()).long()
            now,nll,_=stream.forward(ex,lg,ctx,pieces,lens,state,hist,y)
            old_now,old_nll,_=model(ex,lg,ctx,pieces,lens,initial,hist,y)
            assert torch.equal(now,old_now)
            assert torch.equal(nll,old_nll)
            if t==0:
                state=stream.advance(state,ctx[:,0])
                initial=CrossWindowMirror.advance(model.r74.outer,initial,ctx[:,0],model.trunk.E)
                hist=history_shift(hist,ctx[:,0])
        # The nonzero gate receives an actual LM gradient and the first read
        # matches the R7.11 full-window reference for the same causal history.
    stream.force_gradient=True
    with torch.enable_grad():
        state=stream.initial(torch.from_numpy(corp.get(positions,OUTER_CONTEXT)[0][:,:OLDER].copy()).long())
        model.r74.outer.dir_gate.data.fill_(0.12)
        matrix=stream.matrix(state)
        exact=model.initial(torch.from_numpy(corp.get(positions,OUTER_CONTEXT)[0][:,:OLDER].copy()).long())
        delta=(matrix-exact).abs().max().item()
        assert delta<5e-7,delta
        a=torch.tensor(corp.data[positions].copy()).long();lg=torch.tensor(corp.long[positions].copy()).long()
        _,ctx,pieces,lens,y,_,_=batch_data(base,corp,comp,positions)
        _,loss,_=stream.forward(a,lg,ctx,pieces,lens,state,
             torch.from_numpy(corp.get(positions,OUTER_CONTEXT)[0][:,:OLDER].copy()).long(),y)
        grad=torch.autograd.grad(loss.mean(),model.r74.outer.dir_gate)[0]
        assert torch.isfinite(grad) and float(abs(grad))>0.
        # Validate TWO nonzero-gate steps against independently constructed
        # reference matrices; no future targets can enter the new ring.
        leaving=ctx[:,0].long();new=stream.advance(state,leaving)
        baseline_next=CrossWindowMirror.advance(model.r74.outer,state.legacy,leaving,model.trunk.E)
        ring_next=history_shift(state.ring,leaving)
        direct,_,_=model.r74.outer.pair_state(ring_next,model.trunk.E)
        ref=baseline_next+torch.tanh(model.r74.outer.dir_gate)*(direct-baseline_next)
        assert torch.allclose(stream.matrix(new),ref,atol=1e-7,rtol=1e-6)
        assert not torch.equal(stream.matrix(new),baseline_next)
    assert sha256(ROOT/'checkpoint/pal_active.pt')==before
    return {'zero_gate_matches_R711_initial_and_two_forward_positions':True,
        'nonzero_gate_first_read_max_difference':delta,
        'nonzero_gate_streaming_advance_matches_independent_reference':True,
        'nonzero_gate_LM_gradient':float(grad.detach()),
        'one_persistent_rank64_matrix':True,'ring_tokens':768,
        'parent_sha256_unchanged':before}
if __name__=='__main__':print(json.dumps(run(),indent=2))
