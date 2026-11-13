#!/usr/bin/env python3
"""Full-checkpoint R7.28 -> R7.29 parity and gradient-on-memory probe."""
import argparse
import json
from pathlib import Path

import torch

from fa1_r723_train import forward,prepare
from fa1_r727_checkpoint import load_unified,digest
from fa1_r729_contextual_geometry import attach_contextual,PARAMETER


def run(a):
    torch.set_num_threads(a.threads)
    model,opt,base,comp,metadata=load_unified(a.checkpoint,a.root,a.threads)
    data=prepare(a,model,base,comp,'dev',a.targets,a.seed)
    model.eval()
    with torch.no_grad():before,extra=forward(model,data)
    old_state=opt.state_dict()
    param_ids={id(p) for p in model.parameters()}
    old_tensors={k:v.detach().clone() for k,v in model.state_dict().items()}
    attach_contextual(model,opt)
    with torch.no_grad():after,ex2=forward(model,data)
    if not torch.equal(before,after):
        raise AssertionError(f'R7.29 zero-gain migration changed word loss: max {(before-after).abs().max()}')
    for k,v in old_tensors.items():
        if not torch.equal(v,model.state_dict()[k]):
            raise AssertionError('trained parameter changed during migration: '+k)
    if not all(id(p) in {id(x) for x in model.parameters()} for p in [
               p for g in opt.param_groups[:-1] for p in g['params']]):
        raise AssertionError('migration detached inherited optimizer tensor')
    if len(opt.param_groups)!=len(old_state['groups'])+1:
        raise AssertionError('expected exactly one new group')
    for p in model.parameters():p.requires_grad_(False)
    model.diamond.geometry.context_read_gain.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    branch=forward(model,data)[1]['branch_nll'].mean()
    branch.backward()
    grad=model.diamond.geometry.context_read_gain.grad
    if grad is None or not torch.isfinite(grad).all() or grad.abs().item()<=1.e-12:
        raise AssertionError('geometric context read has no predictive loss gradient')
    report={'round':'FA1 R7.29 function-preserving geometric read migration',
      'checkpoint_sha256':digest(a.checkpoint),'parity_exact':True,
      'inherited_parameter_ids_preserved':len(param_ids),
      'inherited_tensor_values_preserved':True,
      'optimizer_groups_before':len(old_state['groups']),
      'optimizer_groups_after':len(opt.param_groups),
      'single_optimizer':True,
      'context_read_gain_initial':0.,
      'new_gate_parameter':PARAMETER,
      'new_gate_branch_gradient':float(grad),
      'n_dev_samples':len(before),
      'checkpoint_promoted':False}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=16)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=729)
    run(p.parse_args())
