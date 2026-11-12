#!/usr/bin/env python3
"""Repo-only FA1 R7.21 gate and single-optimizer tests. No corpus/checkpoint."""
from __future__ import annotations
import json,torch
from torch import nn
from fa1_r721_unified import UnifiedR7, trainable_memory_only
from fa1_r716_router import TrustRouter,nll as near_nll
from fa1_r718_far_residual import FarResidual,loss_far

class TrunkMock(nn.Module):
    def __init__(self):
        super().__init__();self.fake=nn.Linear(3,3)

def main():
    torch.set_num_threads(2);torch.manual_seed(721)
    r=UnifiedR7(TrunkMock(),nn.Linear(4,1),TrustRouter(),FarResidual())
    n=37
    d={
      'base':torch.rand(n)*2+4.,
      'near_features':torch.randn(n,60),
      'near_prob':torch.rand(n,8)*.3,
      'near_count':torch.randint(0,5,(n,8)).float(),
      'far_features':torch.randn(n,32),
      'far_prob':torch.rand(n,4)*.3,
      'far_count':torch.randint(0,5,(n,4)).float(),
    }
    both,close=r.joint_nll(d)
    close_ref,ng=near_nll(r.near,{'base':d['base'],'features':d['near_features'],
         'prob':d['near_prob'],'count':d['near_count']})
    full_ref,fg=loss_far(r.far,{'near':close_ref,'far_features':d['far_features'],
         'far_prob':d['far_prob'],'far_count':d['far_count']})
    assert torch.equal(both,full_ref) and torch.equal(close,close_ref)
    onlyclose,_=r.joint_nll(d,far_enabled=False)
    onlybase,_=r.joint_nll(d,near_enabled=False,far_enabled=False)
    assert torch.equal(onlyclose,close_ref) and torch.equal(onlybase,d['base'])
    empty={k:(v.clone() if torch.is_tensor(v) else v) for k,v in d.items()}
    empty['near_count'].zero_();empty['far_count'].zero_()
    e,_=r.joint_nll(empty)
    assert torch.allclose(e,d['base'],atol=1e-6,rtol=1e-6)
    trainable_memory_only(r)
    opt=torch.optim.AdamW([{'params':list(r.near.parameters())},
                            {'params':list(r.far.parameters())}],lr=1e-4)
    frozen={k:v.detach().clone() for k,v in r.core.state_dict().items()}
    opt.zero_grad();loss=r.joint_nll(d)[0].mean();loss.backward()
    assert all(p.grad is None for p in r.core.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in r.near.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in r.far.parameters())
    opt.step()
    assert all(torch.equal(v,r.core.state_dict()[k]) for k,v in frozen.items())
    print(json.dumps({'series':'R7','tests_passed':8,'unified_model':True,
         'single_optimizer_joint_update':True,'no_parent_or_corpus_required':True,
         'production_checkpoint_promoted':False},indent=2))

if __name__=='__main__':main()