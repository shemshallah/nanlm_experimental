#!/usr/bin/env python3
"""Non-destructive R7.6 invariance / causality / continuation tests."""
import argparse,json,pathlib
import numpy as np,torch
from r76_pair_training import restore,FORMAT
from r74_stream import OLDER,OUTER_CONTEXT
from r7_core import Corpus
from train_r7 import batch_data

def test(root):
    torch.set_num_threads(2);root=pathlib.Path(root)
    m,opts,base,comp,legacy,prior=restore(root,root/'checkpoint/pal_active.pt',torch.device('cpu'))
    checks={};m.eval()
    assert len(opts)==5 and len(opts[4].param_groups)==2
    assert all(bool(o.state) for o in opts[:4]) and bool(opts[4].state)
    checks['five_optimizer_histories_continuous']='PASS'
    oldids={id(q) for q in m.retrieval.parameters() if q is not m.retrieval.pair.strength and q is not m.retrieval.pair.bias}
    assert oldids=={id(q) for q in opts[4].param_groups[0]['params']}
    assert {id(q) for q in m.retrieval.pair.parameters()}=={id(q) for q in opts[4].param_groups[1]['params']}
    checks['optimizer_updates_live_model_parameters']='PASS'
    fake=torch.ones((2,OLDER),dtype=torch.long)
    fake[:,500:503]=torch.tensor([3,7,9])
    fake[:,900:903]=torch.tensor([3,7,8])
    query=torch.tensor([[3,7],[3,7]])
    prob,exists,features=m.pair_lookup(fake,query,torch.tensor([9,6]))
    assert exists.all() and 0<float(prob[0])<1 and float(prob[1])==0
    checks['pair_key_exact_ordered_retrieval']='PASS'
    query_bad=torch.tensor([[7,3],[7,3]])
    _,exists_bad,_=m.pair_lookup(fake,query_bad,torch.tensor([9,6]))
    assert not exists_bad.any()
    checks['reversing_key_fails_as_expected']='PASS'
    empty=fake.clone();empty[:,500:503]=1;empty[:,900:903]=1
    _,empty_exists,_=m.pair_lookup(empty,query,torch.tensor([9,6]))
    assert not empty_exists.any()
    checks['older_memory_ablation_disables_pair']='PASS'
    # Adding an occurrence after the memory boundary would be unavailable to pair_lookup.
    c=Corpus(root,'test');p=np.array([2048,4096,8192])
    _,ctx,pieces,lens,y,tp,tl=batch_data(base,c,comp,p)
    a=torch.tensor(c.data[p].copy());l=torch.tensor(c.long[p].copy());history=torch.tensor(c.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy());state=m.initial(history)
    with torch.no_grad():
        now,withpair,mt=m(a,l,ctx,pieces,lens,state,history,y)
        oldnow,oldloss,_=m(a,l,ctx,pieces,lens,state,history,y,pair_off=True)
        assert torch.equal(now,oldnow)
        assert torch.equal(oldnow,m.r74(a,l,ctx,pieces,lens,state=state)[0])
        assert torch.isfinite(withpair).all() and torch.isfinite(oldloss).all()
        checks['trained_trunk_and_now_state_identical']='PASS'
        alternate=a.clone();alternate[:,3:]=1234
        _,altloss,_=m(alternate,l,ctx,pieces,lens,state,history,y)
        assert torch.equal(withpair,altloss)
        checks['current_target_and_future_columns_excluded']='PASS'
        _,no_retrieval,_=m(a,l,ctx,pieces,lens,state,history,y,retrieval_off=True)
        assert torch.allclose(no_retrieval,m.trunk.nll(now,y))
        checks['all_retrieval_ablation_exact']='PASS'
    m.train();_,loss,_=m(a,l,ctx,pieces,lens,state,history,y)
    for o in opts:o.zero_grad(set_to_none=True)
    loss.mean().backward()
    assert m.retrieval.pair.strength.grad is not None and torch.isfinite(m.retrieval.pair.strength.grad)
    assert m.trunk.P.grad is not None and float(m.trunk.P.grad.abs().sum())>0
    checks['pair_and_trunk_gradient_flow']='PASS'
    return {'step':prior['joint_steps'],'format':prior['format'],'checks':checks,'passes':len(checks)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');args=p.parse_args()
    print(json.dumps(test(args.root),indent=2))
