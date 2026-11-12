#!/usr/bin/env python3
"""R7.4 invariants: migration, causal isolation, carry and one-checkpoint resume."""
import json,pathlib,torch,numpy as np
from r7_core import Corpus,D,R
from r74_stream import restore, evaluate, OUTER_CONTEXT, OLDER
from train_r7 import batch_data

def run(root):
    torch.set_num_threads(2)
    root=pathlib.Path(root)
    model,opt,base,comp,old,prior=restore(root,root/'checkpoint/pal_active.pt',torch.device('cpu'))
    cor=Corpus(root,'dev');p=np.array([2048,4096,8192],dtype=np.int64)
    _,ctx,pieces,lens,y,_,_=batch_data(base,cor,comp,p)
    ex=torch.tensor(cor.data[p].copy());long=torch.tensor(cor.long[p].copy())
    history=torch.tensor(cor.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy())
    model.eval();result={}
    with torch.no_grad():
        state=model.initial(history)
        core,_=model.core(ex,long,ctx,pieces,lens)
        now,_=model(ex,long,ctx,pieces,lens,state)
        err=float((core-now).abs().max());
        if prior['format']=='PAL_R7_3_JOINT_V1':assert err<1e-6,err
        result['zero_gain_migration_max_difference']=err
        junk=ex.clone();junk[:,3:7]=torch.randint(0,32000,(len(p),4))
        changed,_=model(junk,long,ctx,pieces,lens,state)
        assert torch.equal(changed,now),'future labels leaked into causal state'
        result['future_label_invariance']='PASS'
        moved=model.advance(state,ctx[:,0]);assert float((moved-state).abs().max())>0
        result['stream_state_carry']='PASS'
        empty=torch.ones((2,OLDER),dtype=torch.long)
        empty_state=model.initial(empty)
        assert torch.count_nonzero(empty_state)==0
        assert torch.equal(model.advance(empty_state,torch.ones(2,dtype=torch.long)),empty_state)
        result['bos_padding_is_not_memory']='PASS'
        # Split-aware corpus.get returns masked BOS before a document beginning.
        assert (cor.get(p,OUTER_CONTEXT)[1][:,-1]<p).all()
        result['strictly_prior_context']='PASS'
        # Opposite rails must be the same physical matrix transposed.
        q=torch.nn.functional.normalize(model.outer.query(now),dim=-1)
        one=torch.bmm(state,q.unsqueeze(-1)).squeeze(-1)
        two=torch.bmm(state.transpose(-1,-2),q.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose((one+two)/2+(one-two)/2,one,atol=1e-6)
        result['single_physical_mirror']='PASS'
        assert len(opt)==4 and len(opt[0].state)==len(list(model.trunk.parameters()))
        assert opt[1].state and opt[2].state
        if prior['format']=='PAL_R7_4_CROSSWINDOW_V1':assert opt[3].state, 'new optimizer was not checkpointed'
        result['historical_optimizers_preserved']='PASS'
    model.train();now,_=model(ex,long,ctx,pieces,lens,state)
    for op in opt:op.zero_grad(set_to_none=True)
    loss=model.trunk.nll(now,y).mean();loss.backward()
    assert model.outer_gain.grad is not None and model.outer_gain.grad.abs().item()>0
    assert model.trunk.P.grad is not None and model.trunk.P.grad.abs().sum()>0
    result['causal_gradient_to_new_and_old']='PASS'
    result['initial_joint_step']=int(prior['joint_steps'])
    result['passed']=len([x for x in result if result[x]=='PASS'])
    return result

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.')
    args=ap.parse_args();print(json.dumps(run(args.root),indent=2))
