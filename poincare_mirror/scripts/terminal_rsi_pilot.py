#!/usr/bin/env python3
"""Reproducible, honest Terminal-RSI *toy* training on the unified MirrorLM.

The harness verifies one-byte next-letter tasks. This tests RSR->RPR wiring,
critic handoff, and on-policy sampled rewards, NOT terminal-bench capability,
nor a continuation of the inaccessible R38 step-4057 checkpoint.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poincare_frontier import Config,MirrorLM
from terminal_rsi.core import Event,Receipt,Trajectory,adopt_rewrite
from terminal_rsi.torch_adapter import Batch,attach,critic_warmup,ppo_loss,sft_loss


def tiny_model():
    cfg=Config(vocab_size=257,dim=32,layers=2,heads=4,pairs=4,topk_pairs=2,
               expert_hidden=16,context=16,moe_every=2,hierarchy_rank=8,
               symbiotic_rank=8,micro_meta=True,simplex_transfer=True,
               reciprocal_scales=True)
    return MirrorLM(cfg)


def prefix(letter):
    return torch.tensor([[ord(letter),ord(':')]],dtype=torch.long)


def batch_for(letter,action,old_lp=0.,reward=1.):
    ids=prefix(letter)
    labels=torch.tensor([[0,ord(action)]],dtype=torch.long)
    mask=torch.tensor([[False,True]])
    zeros=torch.zeros((1,2),dtype=torch.float32)
    ended=torch.tensor([[False,True]])
    return Batch(ids,labels,mask,torch.tensor([[0,1]]),
                 torch.tensor([[0.,reward]]),ended,
                 torch.tensor([[0.,old_lp]]))


def verify(letter,answer):
    return len(answer)==1 and answer==chr(ord(letter)+1)


def verified_sft(letter):
    action=chr(ord(letter)+1)
    donor=Trajectory(letter,'next-letter','donor',letter+'-donor',
                     (Event('user',letter+':',0),Event('assistant',action,1)),True)
    def replay(task,actions):
        success=verify(task,actions[0])
        return Trajectory(task,'next-letter','target',task+'-fresh',donor.events,success)
    def external_receipt(t):
        outcome=verify(t.task_id,[e.text for e in t.events if e.role=='assistant'][0])
        return Receipt(t.task_id,t.run_id,t.harness_id,t.digest(),outcome)
    sample=adopt_rewrite(donor,replay,external_receipt)
    if sample.loss_mask!=(False,True): raise RuntimeError('SFT mask must exclude prompt')
    return action


@torch.no_grad()
def evaluate(model,letters):
    model.eval()
    total_ce=0.; top1=top5=0
    start=time.perf_counter()
    for letter in letters:
        target=ord(letter)+1
        logits,_,_=model(prefix(letter))
        scores=logits[0,-1].float()
        total_ce+=float(F.cross_entropy(scores[None],torch.tensor([target])))
        top1+=int(scores.argmax().item()==target)
        top5+=int(target in scores.topk(5).indices.tolist())
    count=len(letters)
    ce=total_ce/count
    return dict(positions=count,top1_percent=round(100*top1/count,4),
                top5_percent=round(100*top5/count,4),
                nats_per_byte=round(ce,6),bits_per_byte=round(ce/0.6931471805599453,6),
                bytes_per_token=1.0,bytes_per_second=round(count/(time.perf_counter()-start),2))


def run(seed=2026,sft_steps=80,ppo_steps=32,lr=.003,rpr_actor_lr_scale=.1):
    if not 0 < rpr_actor_lr_scale <= 1:
        raise ValueError('RPR actor learning rate scale must lie in (0,1]')
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    actor=tiny_model().cpu()
    optimizer=torch.optim.AdamW(actor.parameters(),lr=lr)
    bridge=attach(actor,optimizer,32,critic_lr=.002)
    train='abcdef'; holdout='ghij'
    before=evaluate(actor,holdout)
    before_train=evaluate(actor,train)
    verified={letter:verified_sft(letter) for letter in train}
    critic_initial={k:v.detach().clone() for k,v in bridge.critic.state_dict().items()}
    sft_losses=[]
    actor.train()
    for step in range(sft_steps):
        letter=train[step%len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss=sft_loss(bridge,batch_for(letter,verified[letter]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
        optimizer.step(); sft_losses.append(float(loss.detach()))
    if any(not torch.equal(v,bridge.critic.state_dict()[k]) for k,v in critic_initial.items()):
        raise RuntimeError('RSR altered inherited critic')
    after_sft=evaluate(actor,holdout)
    after_sft_train=evaluate(actor,train)
    # Verified RSR action supplies a supervised calibration return for warmup.
    warmup=critic_warmup(bridge,optimizer,batch_for(train[0],verified[train[0]]),
                         torch.tensor([[0.,1.]]),steps=4)
    # Stage-specific PPO learning rate; retain every inherited Adam moment.
    for group in optimizer.param_groups[:-1]:
        group['lr']*=rpr_actor_lr_scale
    sampled_successes=0
    ppo_losses=[]
    for step in range(ppo_steps):
        letter=train[step%len(train)]
        actor.eval()
        with torch.no_grad():
            logits,_,_=actor(prefix(letter))
            distribution=F.softmax(logits[0,-1],-1)
            action_id=int(torch.multinomial(distribution,1).item())
            log_prob=float(distribution[action_id].log())
        # Every action is sampled on-policy, then verified independently.
        success=bool(action_id<256 and verify(letter,chr(action_id)))
        sampled_successes+=int(success)
        ids=prefix(letter)
        mask=torch.tensor([[False,True]])
        batch=Batch(ids,torch.tensor([[0,action_id]]),mask,torch.tensor([[0,1]]),
                    torch.tensor([[0.,float(success)]]),torch.tensor([[False,True]]),
                    torch.tensor([[0.,log_prob]]))
        actor.train()
        optimizer.zero_grad(set_to_none=True)
        loss,stats=ppo_loss(bridge,batch)
        loss.backward();torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.)
        optimizer.step();ppo_losses.append(float(loss.detach()))
    after_rpr=evaluate(actor,holdout)
    after_rpr_train=evaluate(actor,train)
    return {'scope':'toy next-letter task only, not independent books or R38',
            'seed':seed,'train_letters':train,'holdout_letters':holdout,
            'rpr_actor_lr_scale':rpr_actor_lr_scale,
            'sft_updates':sft_steps,'verified_replay_count':len(verified),
            'critic_warmup_updates':len(warmup),'ppo_updates':ppo_steps,
            'onpolicy_verified_successes':sampled_successes,
            'onpolicy_verified_attempts':ppo_steps,
            'mean_sft_loss':sum(sft_losses)/len(sft_losses),
            'mean_ppo_loss':sum(ppo_losses)/len(ppo_losses),
            'metrics_before':before,'metrics_after_rsr':after_sft,
            'metrics_after_rpr':after_rpr,
            'train_metrics_before':before_train,
            'train_metrics_after_rsr':after_sft_train,
            'train_metrics_after_rpr':after_rpr_train,
            'promotion':'FORBIDDEN: no independent-book or R38 checks'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sft-steps',type=int,default=80)
    p.add_argument('--ppo-steps',type=int,default=32)
    p.add_argument('--rpr-actor-lr-scale',type=float,default=.1)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not 1<=args.sft_steps<=1000 or not 1<=args.ppo_steps<=1000:
        p.error('Bounded training steps must be 1..1000')
    report=run(sft_steps=args.sft_steps,ppo_steps=args.ppo_steps,
               rpr_actor_lr_scale=args.rpr_actor_lr_scale)
    payload=json.dumps(report,indent=2,sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(payload+'\n',encoding='utf-8')
    print(payload)


if __name__=='__main__': main()
