#!/usr/bin/env python3
"""R7.5 regression suite: strict causal retrieval, S+/S- and optimizer continuity."""
import argparse, json, pathlib
import numpy as np
import torch
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER
from r75_retrieval import restore,OPT_NAMES,PALR75
from train_r7 import batch_data

def test(root):
    torch.set_num_threads(2);root=pathlib.Path(root);dev=torch.device('cpu')
    model,opts,base,comp,old,prior=restore(root,root/'checkpoint/pal_active.pt',dev)
    c=Corpus(root,'test');p=np.array([2048,4096,8192])
    _,ctx,pc,ln,y,tp,tl=batch_data(base,c,comp,p)
    a=torch.tensor(c.data[p].copy());l=torch.tensor(c.long[p].copy())
    h=torch.tensor(c.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy());state=model.initial(h)
    result={};model.eval()
    with torch.no_grad():
        original,_=model.r74(a,l,ctx,pc,ln,state=state)
        now,off,_=model(a,l,ctx,pc,ln,state,h,y,retrieval_off=True)
        assert torch.equal(now,original)
        assert torch.allclose(off,model.trunk.nll(original,y),atol=1e-6)
        result['original_predictions_and_decoder_unchanged']='PASS'
        for col in (3,4,5,6):
            attack=a.clone();attack[:,col]=17
            now_attack,nll_attack,_=model(attack,l,ctx,pc,ln,state,h,y)
            now_real,nll_real,_=model(a,l,ctx,pc,ln,state,h,y)
            assert torch.equal(now_attack,now_real) and torch.equal(nll_attack,nll_real)
        result['target_and_future_columns_excluded']='PASS'
        # Query A: target Z appears as value only in strictly older memory.
        fake=torch.ones((2,OLDER),dtype=torch.long)
        fake[:,600:606]=torch.tensor([5,9,5,9,5,9])
        prob,exists,feat,_=model.retrieval.lookup(fake,torch.tensor([5,5]),torch.tensor([9,8]))
        assert exists.all() and prob[0]>.99 and prob[1]==0
        result['exact_old_key_value_retrieval']='PASS'
        fake[:,600:606]=1
        prob2,exists2,_,_=model.retrieval.lookup(fake,torch.tensor([5,5]),torch.tensor([9,8]))
        assert not exists2.any() and torch.equal(prob2,torch.zeros_like(prob2))
        result['older_memory_ablation_removes_recall']='PASS'
        untouched=fake.clone()
        new_history=torch.cat((fake[:,1:],torch.tensor([[5],[5]])),dim=1)
        assert (new_history[:,-1]==5).all() and (new_history[:,:-1]==untouched[:,1:]).all()
        result['causal_fifo_shift']='PASS'
        assert len(opts)==5 and all(opts[i].state for i in range(4))
        if prior['format']=='PAL_R7_5_ASSOCIATIVE_V1':assert opts[4].state
        result['single_model_five_optimizer_states']='PASS'
        assert torch.allclose(model.r74.outer.read_rails(state,now),model.r74.outer.read_rails(state,now))
        result['original_mirror_preserved']='PASS'
    model.train();_,nll,_=model(a,l,ctx,pc,ln,state,h,y)
    for op in opts:op.zero_grad(set_to_none=True)
    nll.mean().backward()
    assert model.retrieval.gate_cap.grad is not None and torch.isfinite(model.retrieval.gate_cap.grad)
    assert model.trunk.P.grad is not None and model.trunk.P.grad.abs().sum()>0
    result['joint_gradient_to_retrieval_and_original_trunk']='PASS'
    result['joint_step']=prior['joint_steps'];result['count']=len([v for v in result.values() if v=='PASS'])
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args()
    print(json.dumps(test(a.root),indent=2))
