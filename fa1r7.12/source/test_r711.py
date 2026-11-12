#!/usr/bin/env python3
"""R7.11 reproducibility: single-checkpoint overlay, causal directed read, fail-closed hash."""
import json, pathlib, tempfile,shutil,torch, numpy as np
from r711_delta import restore_overlay
from r711_selective_writer import restore as restore_parent, HORIZON
from r710_checkpoint_integrity import verify,deep_equal,sha256
from r7_core import Corpus
from r74_stream import OLDER,OUTER_CONTEXT
from torch.nn import functional as F

def test(root=pathlib.Path('.')):
    torch.set_num_threads(2)
    base=root/'checkpoint/pal_active.pt';overlay=root/'adaptation/r711_experimental_delta.pt'
    parent,info=verify(base,root,expected_step=14416)
    original,oldopts,_,_,_=restore_parent(root,base)
    model,opts,_,_,delta=restore_overlay(root,overlay)
    tests={}
    assert delta['parent_sha256']==info['sha256'] and delta['directed_aux_updates']==80
    tests['full_parent_hash_and_auxiliary_step_pinning']=True
    assert all(torch.equal(model.state_dict()[n],v) for n,v in parent['model'].items())
    tests['inherited_tensors_bitwise_unchanged']=True
    for i,(name,prior) in enumerate(parent['optimizers'].items()):
        actual=opts[i].state_dict()
        if name=='outer_adamw':
            assert deep_equal(actual['param_groups'][:-1],prior['param_groups'])
            assert all(k in actual['state'] and deep_equal(s,actual['state'][k]) for k,s in prior['state'].items())
        else:assert deep_equal(actual,prior)
    tests['all_five_existing_optimizer_moments_unchanged']=True
    c=Corpus(root,'dev');h=torch.tensor(c.get(np.asarray([10000,20000,30000,45000]),OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
    with torch.no_grad():
        assert torch.equal(original.initial(h),model.initial(h))
        M,km,vm=model.r74.outer.pair_state(h,model.trunk.E)
        rev=h[:,-HORIZON:].reshape(4,HORIZON//64,64).flip(-1).reshape(4,HORIZON)
        Mr,_,_=model.r74.outer.pair_state(rev,model.trunk.E)
        assert not torch.allclose(M,Mr,atol=1e-8)
    tests['zero_gate_inherited_memory_bitwise']=True
    g=torch.autograd.grad(model.initial(h).square().sum(),model.r74.outer.dir_gate)[0]
    assert torch.isfinite(g) and g.abs()>0
    tests['zero_gate_has_nonzero_learning_gradient']=True
    with torch.no_grad():
        leaving=h[:,-1].clone()
        assert torch.equal(original.advance(original.initial(h),leaving),
                           model.advance(model.initial(h),leaving))
        model.r74.outer.dir_gate.fill_(0.1)
        try:model.advance(model.initial(h),leaving)
        except RuntimeError:pass
        else:raise AssertionError('unimplemented directed incremental update was silently used')
        model.r74.outer.dir_gate.zero_()
    tests['nonzero_gate_streaming_fails_closed']=True
    tests['directed_memory_sensitive_to_order']=True
    assert M.shape==(4,64,64) and M.isfinite().all()
    tests['exactly_one_rank64_pair_matrix']=True
    try:
        with tempfile.TemporaryDirectory() as t:
            copy=pathlib.Path(t)/'delta.pt';shutil.copyfile(overlay,copy)
            copy.with_suffix(copy.suffix+'.sha256').write_text('0'*64+'  delta.pt\n')
            try:restore_overlay(root,copy)
            except ValueError:pass
            else:raise AssertionError('corrupt overlay accepted')
    except FileNotFoundError:raise AssertionError('overlay artifact missing')
    tests['corrupt_overlay_rejected']=True
    assert sha256(base)==info['sha256']
    tests['accepted_checkpoint_preserved']=True
    result={'passed':len(tests),'tests':tests,'parent_sha256':info['sha256'],
      'auxiliary_updates':delta['directed_aux_updates'],'reproducible_overlay_bytes':overlay.stat().st_size}
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':test()
