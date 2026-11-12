#!/usr/bin/env python3
"""R7.9 exact migration / causal memory / gradient preservation checks."""
import pathlib,json
import torch,numpy as np
from r77_semantic import restore as restore77
from r79_normalized_memory import restore
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER
from train_r7 import batch_data
from r75_retrieval import OPT_NAMES

def run(root='.'):
    torch.set_num_threads(2);root=pathlib.Path(root)
    original,previous,base,comp,old=restore77(root,root/'checkpoint/pal_active.pt',torch.device('cpu'))
    model,opts,base,comp,prior=restore(root,root/'checkpoint/pal_active.pt')
    checks={}
    assert len(opts)==5 and len(OPT_NAMES)==5 and len(opts[3].param_groups)==4
    checks['five_continuous_optimizer_histories']='PASS'
    for n,p in original.state_dict().items():assert torch.equal(p,model.state_dict()[n]),n
    for i in range(5):
        aa=previous[i].state_dict()['state'];bb=opts[i].state_dict()['state']
        assert list(aa)==list(bb)
        for k in aa:
            for field,v in aa[k].items():
                if torch.is_tensor(v):assert torch.equal(v,bb[k][field]),(i,k,field)
                else:assert v==bb[k][field]
    checks['legacy_model_and_adam_tensors_bit_identical']='PASS'
    test=Corpus(root,'test');ix=np.asarray([8192,30000]);_,ctx,pc,ln,y,_,_=batch_data(base,test,comp,ix)
    hist=torch.as_tensor(test.get(ix,OUTER_CONTEXT)[0][:,:OLDER].copy())
    ex=torch.as_tensor(test.data[ix].copy());long=torch.as_tensor(test.long[ix].copy())
    original.eval();model.eval()
    with torch.no_grad():
        st0=original.initial(hist);st=model.initial(hist)
        assert torch.equal(st0,st)
        _,loss0,_=original(ex,long,ctx,pc,ln,st0,hist,y)
        _,loss,_=model(ex,long,ctx,pc,ln,st,hist,y)
        assert torch.equal(loss0,loss)
        checks['zero_gate_exact_prediction_migration']='PASS'
        model.r74.normal_gain.fill_(.1)
        _,new_loss,_=model(ex,long,ctx,pc,ln,st,hist,y)
        assert float((new_loss-loss0).abs().max())>1e-4
        checks['normalized_gate_changes_prediction']='PASS'
        assert all(float(p)==0 for p in (model.r74.outer.pair_gain,model.r74.outer.order_gain))
        reversed_hist=hist.reshape(-1,OLDER//64,64).flip(-1).reshape(-1,OLDER)
        # With order-dependent writes gated OFF, within-block reversal is invisible.
        reversed_state=model.initial(reversed_hist)
        assert float((st-reversed_state).abs().max())<1e-6
        checks['zero_gate_order_invariance']='PASS'
        model.r74.outer.pair_gain.fill_(.2)
        directed=model.initial(hist);reversed_directed=model.initial(reversed_hist)
        assert float((directed-reversed_directed).abs().max())>1e-5
        checks['one_matrix_causal_order_detection']='PASS'
        model.r74.outer.pair_gain.zero_();model.r74.normal_gain.zero_()
    model.train();state=model.initial(hist)
    _,ll,_=model(ex,long,ctx,pc,ln,state,hist,y)
    for opt in opts:opt.zero_grad(set_to_none=True)
    ll.mean().backward()
    assert model.r74.normal_gain.grad is not None and torch.isfinite(model.r74.normal_gain.grad)
    checks['normalized_gate_backpropagation']='PASS'
    return {'passed':len(checks),'checks':checks,'source_step':prior['joint_steps']}
if __name__=='__main__':print(json.dumps(run('.'),indent=2))
