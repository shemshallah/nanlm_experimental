"""Opt-in PPO / actor-only SFT over the existing MirrorLM hidden state.

The adapter is not a second language model. Only a zero-initialized value
projection is appended to the SAME actor and optimizer (new parameter group).
The preexisting actor checkpoint and all Adam slots remain intact.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math
from typing import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from .core import Action, Continuation, tb_gae


class Bridge(nn.Module):
    def __init__(self, actor: nn.Module, dim: int):
        super().__init__()
        self.actor = actor
        self.critic = nn.Linear(dim,1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, ids):
        logits,hidden,_=self.actor(ids)
        return logits,self.critic(hidden).squeeze(-1)


def attach(actor, optimizer, dim, critic_lr=None):
    original={id(p) for p in actor.parameters()}
    if not original or original != {id(p) for g in optimizer.param_groups for p in g['params']}:
        raise ValueError('Optimizer must contain exactly the original actor parameters')
    if dim<=0 or (critic_lr is not None and (not math.isfinite(critic_lr) or critic_lr<=0)):
        raise ValueError('Invalid critic hyperparameter')
    first=next(actor.parameters())
    bridge=Bridge(actor,dim).to(device=first.device,dtype=first.dtype)
    group={'params':list(bridge.critic.parameters()),'weight_decay':0.}
    if critic_lr: group['lr']=critic_lr
    optimizer.add_param_group(group)
    return bridge


def restore(actor, optimizer, parent: Mapping, dim, critic_lr=None):
    if 'model' not in parent or 'optimizer' not in parent:
        raise ValueError('Need complete parent model AND optimizer state')
    actor.load_state_dict(parent['model'],strict=True)
    optimizer.load_state_dict(parent['optimizer'])
    return attach(actor,optimizer,dim,critic_lr)


@dataclass(frozen=True)
class Batch:
    ids: torch.Tensor           # [1,T] prefix input (includes observations)
    labels: torch.Tensor        # [1,T] token emitted from each prefix
    assistant: torch.Tensor     # [1,T] bool, excludes tool observations
    turns: torch.Tensor         # [1,T] int64
    rewards: torch.Tensor       # [1,T] float
    terminated: torch.Tensor    # [1,T] bool
    old_logprobs: torch.Tensor  # [1,T] behavior policy

    def validate(self):
        if self.ids.ndim!=2 or self.ids.shape[0]!=1 or self.ids.dtype!=torch.long:
            raise ValueError('Bounded pilot accepts [1,T] int64 sequence')
        if any(getattr(self,name).shape!=self.ids.shape for name in
               ('labels','assistant','turns','rewards','terminated','old_logprobs')):
            raise ValueError('Fields require aligned byte positions')
        if self.labels.dtype!=torch.long or self.turns.dtype!=torch.long:
            raise ValueError('Token and turn labels must be int64')
        if self.assistant.dtype!=torch.bool or self.terminated.dtype!=torch.bool:
            raise ValueError('Masks must be boolean')
        if not self.assistant.any() or (self.terminated & ~self.assistant).any():
            raise ValueError('Only actor actions may terminate the rollout')
        if (self.turns<0).any() or not torch.isfinite(self.rewards).all() or not torch.isfinite(self.old_logprobs).all():
            raise ValueError('Invalid turn/reward/behavior data')


def logprobs(logits, batch):
    if logits.shape[:2]!=batch.ids.shape: raise ValueError('Bad logits shape')
    selected_logits=logits[batch.assistant]
    labels=batch.labels[batch.assistant]
    if (labels<0).any() or (labels>=logits.shape[-1]).any():
        raise ValueError('Label out of vocabulary')
    log_p=F.log_softmax(selected_logits,-1)
    return log_p.gather(-1,labels[:,None]).squeeze(-1),-(log_p.exp()*log_p).sum(-1).mean()


def sft_loss(bridge, batch):
    """Use only verifier-approved assistant positions during RSR.

    `batch` MUST originate from an adopt_rewrite result and its trusted
    harness replay, not from model-authored reports. This is actor-only.
    """
    batch.validate()
    logits,_,_=bridge.actor(batch.ids)
    lp,_=logprobs(logits,batch)
    return -lp.mean()


def ppo_loss(bridge, batch, continuation: Continuation|None=None,
             gamma_turn=.99, lambda_turn=.95, clip=.2, value_weight=.5,
             entropy_weight=0.):
    batch.validate()
    if not 0<clip<1 or value_weight<0 or entropy_weight<0:
        raise ValueError('Invalid PPO coefficients')
    logits,values=bridge(batch.ids)
    lp,entropy=logprobs(logits,batch)
    action_values=values[batch.assistant]
    tuples=zip(batch.turns[batch.assistant],action_values.detach(),
               batch.rewards[batch.assistant],batch.terminated[batch.assistant])
    actions=[Action(int(t.item()),float(v.item()),float(r.item()),bool(d.item()))
             for t,v,r,d in tuples]
    credit=tb_gae(actions,gamma_turn,lambda_turn,continuation)
    adv=action_values.new_tensor(credit.advantages)
    targets=action_values.new_tensor(credit.targets)
    ratio=(lp-batch.old_logprobs[batch.assistant]).exp()
    actor_loss=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
    value_loss=.5*F.mse_loss(action_values,targets)
    loss=actor_loss+value_weight*value_loss-entropy_weight*entropy
    return loss,{'generated':len(actions),'observations':int((~batch.assistant).sum()),
                 'actor_loss':float(actor_loss.detach()),'critic_loss':float(value_loss.detach())}


def critic_warmup(bridge, optimizer, batch, verified_targets, steps=4):
    """Four (or more) critic-only updates; no actor weights or Adam changed."""
    batch.validate()
    if not 4<=steps<=64 or verified_targets.shape!=batch.ids.shape or not torch.isfinite(verified_targets).all():
        raise ValueError('Require 4..64 updates and verified aligned return targets')
    old_mode=bridge.actor.training
    bridge.actor.eval()
    try:
        with torch.no_grad(): _,h,_=bridge.actor(batch.ids)
        results=[]
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            v=bridge.critic(h.detach()).squeeze(-1)
            loss=F.mse_loss(v[batch.assistant],verified_targets[batch.assistant])
            loss.backward(); optimizer.step()
            results.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        return tuple(results)
    finally: bridge.actor.train(old_mode)


def candidate(bridge, optimizer, parent, stage, verified_runs, warmup):
    """A new *unpromoted* payload; never replace accepted lineage automatically."""
    if 'model' not in parent or 'optimizer' not in parent or verified_runs<1 or not stage:
        raise ValueError('Need verified parent and successful external task runs')
    result=copy.deepcopy(dict(parent))
    result['model']=copy.deepcopy(bridge.actor.state_dict())
    result['optimizer']=copy.deepcopy(optimizer.state_dict())
    result['terminal_rsi']={'schema':1,'stage':stage,'critic':copy.deepcopy(bridge.critic.state_dict()),
                            'warmup':warmup,'verified_runs':verified_runs,'promoted':False}
    return result
