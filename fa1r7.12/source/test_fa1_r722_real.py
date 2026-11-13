#!/usr/bin/env python3
"""R7.22 unified checkpoint audit against original unified R7.21, on actual corpus."""
import argparse,copy,json,pathlib,tempfile,time,os
import numpy as np
import torch
from fa1_r721_unified import load_candidate as load_r721
from fa1_r722_unified import load_unified, memory_trainable, save_unified
from r710_checkpoint_integrity import deep_equal,sha256

def run(a):
    t=time.monotonic();torch.set_num_threads(2)
    old,oldopts,base,comp,meta=load_r721(a.ancestor,a.root)
    model,opts,base2,comp2,meta2=load_unified(a.candidate,a.root)
    old_state=old.state_dict();states=model.state_dict()
    assert all(torch.equal(v,states[k]) for k,v in old_state.items())
    assert all(deep_equal(x.state_dict(),y.state_dict()) for x,y in zip(oldopts[:4],opts[:4]))
    assert deep_equal(oldopts[4].state_dict()['state'],opts[4].state_dict()['state'])
    ps=np.arange(a.start,a.start+a.positions,dtype=np.int64)
    original=old.runtime(a.root,base,comp).score_positions(ps,mode='packed_index',batch_size=a.batch)
    final=model.runtime(a.root,base2,comp2).score_positions(ps,mode='packed_index',batch_size=a.batch,probe_memory=True)
    err={k:max(abs(x-y) for x,y in zip(original[k],final[k])) for k in ['baseline_nll','near_nll','far_nll']}
    exact_sem=all(float(x)==float(y) for x,y in zip(final['far_nll'],final['semantic_nll']))
    assert all(e<1e-6 for e in err.values()) and exact_sem
    # Ensure the active model is trainable via the existing fifth optimizer,
    # including checkpoint save/load, without touching inherited tensors.
    memory_trainable(model,semantic_only=True)
    model.train()  # gate uses differentiable zero-init path during training
    old_states=copy.deepcopy([o.state_dict() for o in opts]);w=copy.deepcopy(model.state_dict())
    batch={'base':torch.full((12,),5.),'near_features':torch.randn(12,60),
           'near_prob':torch.rand(12,8)*.1,'near_count':torch.ones(12,8),
           'far_features':torch.randn(12,32),'far_prob':torch.rand(12,4)*.1,
           'far_count':torch.ones(12,4),'semantic_features':torch.randn(12,16),
           'semantic_prob':torch.rand(12)*.25,'semantic_count':torch.ones(12)}
    opts[4].zero_grad(set_to_none=True)
    loss=model.joint_nll(batch)[0].mean();loss.backward()
    grad=float(model.semantic.open_fraction.grad.abs())
    assert grad>0;opts[4].step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in w.items() if not k.startswith('semantic.'))
    assert all(deep_equal(o.state_dict(),s) for o,s in zip(opts[:4],old_states[:4]))
    s5=opts[4].state_dict();orig5=old_states[4]
    assert deep_equal(s5['param_groups'][:6],orig5['param_groups'][:6])
    assert all(deep_equal(s5['state'][k],v) for k,v in orig5['state'].items())
    with tempfile.TemporaryDirectory() as td:
        path=pathlib.Path(td)/'r7_candidate.pt'
        save_unified(path,model,opts,a.candidate,train_updates=1,
            selection={'synthetic_gradient_step_only':True,'new_books':False})
        loaded,newopts,_,_,meta3=load_unified(path,a.root)
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
        assert all(deep_equal(o.state_dict(),p.state_dict()) for o,p in zip(opts,newopts))
        assert meta3['semantic_updates']==1
    # No parent files were written.
    assert sha256(a.ancestor)==meta2['r721_ancestor_sha256']
    report={'round':'FA1 R7.22','real_positions':len(ps),'parity_max_error':err,
        'semantic_identity_exact':exact_sem,'semantic_hit_positions':final['semantic_eligible_positions'],
        'final_equal_prior':True,'parameter_tensors':len(states),'optimizers':len(opts),
        'fifth_optimizer_groups':len(opts[4].param_groups),'inherited_adam_states_preserved':True,
        'semantic_grad_nonzero':grad,'synthetic_training_resume':True,
        'training_updates_selected':0,'accepted_checkpoint_mutated':False,
        'new_book_evaluation':False,'promoted':False,'seconds':round(time.monotonic()-t,3)}
    if a.output:pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/fa1_full')
    p.add_argument('--ancestor',default='/mnt/data/fa1r721/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--candidate',default='/mnt/data/fa1r722/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--start',type=int,default=134000);p.add_argument('--positions',type=int,default=128)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--output')
    args=p.parse_args()
    print(json.dumps(run(args),indent=2))