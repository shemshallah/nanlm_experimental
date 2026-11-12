#!/usr/bin/env python3
"""FA1 R7.15: CPU, causal query-conditioned S+/S- mirror writer/readout.

Experimental candidate only; accepted step 14,416 is immutable.
All trainable key, value, query, attention and trust parameters get gradients from
next-word LM NLL, not auxiliary recall classification. Reuses fifth optimizer
without altering its pre-existing tensors or Adam states. A source-mismatch
control uses historical tokens from unrelated examples at matching positions.

Historical corpus splits were not independently sealed by book; this is a pilot,
never sufficient on its own for promotion.
"""
from __future__ import annotations
import argparse, pathlib, json, time, copy, hashlib, tempfile, os
import numpy as np, torch
from torch import nn
from torch.nn import functional as F
from r7_core import Corpus
from r74_stream import OLDER, OUTER_CONTEXT
from train_r7 import batch_data
from r712_validate import restore as restore_r712
from r712_streaming import DirectedStreaming
from r710_checkpoint_integrity import sha256

PARENT_SHA='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'

def make_positions(corpus,n,seed):
    end=len(corpus.seq)-1
    if end < OUTER_CONTEXT+n:raise ValueError('short corpus')
    ps=OUTER_CONTEXT + np.arange(n,dtype=np.int64)*(end-OUTER_CONTEXT)//n
    stride=max(1,(end-OUTER_CONTEXT)//n)
    return np.minimum(end,ps+np.random.default_rng(seed).integers(0,stride,size=n))

@torch.no_grad()
def prepare(root,model,base,comp,split,n,seed,k=96,batch=8):
    """Frozen baseline and target-free index selection; no target used in retrieval.

    Snapshot selected (key,value) transitions at positions < target-2048.
    For memory-mismatch control we permute all histories after feature prep,
    keeping each query and inherited baseline identical.
    """
    if k < 8 or k > 512:raise ValueError('unexpected topk')
    corpus=Corpus(root,split)
    ps=make_positions(corpus,n,seed)
    E=model.trunk.E.detach().cpu()
    Eunit=F.normalize(E,dim=-1)
    stream=DirectedStreaming(model)
    out={key:[] for key in ('base','target','query','prev_query','prev_prev','key','val','prev_key','age','valid','exact','exact2','history')}
    for idx in np.array_split(ps,int(np.ceil(n/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,idx)
        allhist=torch.from_numpy(corpus.get(idx,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
        state=stream.initial(allhist)
        ex=torch.from_numpy(corpus.data[idx].copy()).long()
        lng=torch.from_numpy(corpus.long[idx].copy()).long()
        _,loss,_=stream.forward(ex,lng,ctx,pieces,lens,state,allhist,y)
        out['base'].append(loss.cpu());out['target'].append(y.cpu());
        q=ctx[:,-1].long().cpu();p=ctx[:,-2].long().cpu();p2=ctx[:,-3].long().cpu()
        out['query'].append(q);out['prev_query'].append(p);out['prev_prev'].append(p2)
        out['history'].append(allhist.cpu())
        # All candidate transitions are in the *older* historical region; no
        # current next-word target, or 2048-word near context, participates.
        keys=allhist[:,1:-1] # prev at [:,:-2], key [1:-1], value [2:]
        vals=allhist[:,2:]
        preceding=allhist[:,:-2]
        eligible=(keys>1)&(vals>1)&(q[:,None]>1)
        # All scores are computed before the current target is available.
        sim=torch.bmm(Eunit[keys],Eunit[q].unsqueeze(-1)).squeeze(-1)
        prev_sim=torch.bmm(Eunit[preceding],Eunit[p].unsqueeze(-1)).squeeze(-1)
        match=(keys==q[:,None])&eligible
        match2=match&(preceding==p[:,None])
        recency=torch.linspace(-1,0,keys.shape[1])[None,:]
        selector=(.8*sim+.20*prev_sim+1.1*match.float()+1.1*match2.float()+.03*recency)
        selector=selector.masked_fill(~eligible,-1e8)
        ix=selector.topk(k,dim=1).indices
        good=eligible.gather(1,ix)
        # Chosen values are never ranked or filtered by their agreement with y.
        for name,arr in (('key',keys),('val',vals),('prev_key',preceding)):
            out[name].append(arr.gather(1,ix).cpu())
        out['valid'].append(good.cpu())
        out['age'].append((ix.float()/keys.shape[1]-1.).cpu())
        out['exact'].append(match.gather(1,ix).float().cpu())
        out['exact2'].append(match2.gather(1,ix).float().cpu())
    result={k:torch.cat(v,dim=0) for k,v in out.items()}
    result['n']=len(ps)
    return result

@torch.no_grad()
def wrong_history(d,shift,E,k=96,batch=8):
    """Independently reselect wrong-memory transitions with current query."""
    n=d['n'];perm=(torch.arange(n)+shift)%n
    z={key:v for key,v in d.items() if key not in ('key','val','prev_key','age','valid','exact','exact2','history')}
    z['history']=d['history'][perm]
    Eunit=F.normalize(E.detach().cpu(),dim=-1)
    by_key={key:[] for key in ('key','val','prev_key','age','valid','exact','exact2')}
    for sl in torch.arange(n).split(batch):
        hist=z['history'][sl];q=d['query'][sl];p=d['prev_query'][sl]
        keys=hist[:,1:-1];vals=hist[:,2:];preceding=hist[:,:-2]
        eligible=(keys>1)&(vals>1)&(q[:,None]>1)
        sim=torch.bmm(Eunit[keys],Eunit[q].unsqueeze(-1)).squeeze(-1)
        prev_sim=torch.bmm(Eunit[preceding],Eunit[p].unsqueeze(-1)).squeeze(-1)
        match=(keys==q[:,None])&eligible
        match2=match&(preceding==p[:,None])
        recency=torch.linspace(-1,0,keys.shape[1])[None,:]
        selection=.8*sim+.20*prev_sim+1.1*match.float()+1.1*match2.float()+.03*recency
        selection=selection.masked_fill(~eligible,-1e8)
        ix=selection.topk(k,dim=1).indices
        for name,arr in (('key',keys),('val',vals),('prev_key',preceding)):
            by_key[name].append(arr.gather(1,ix))
        by_key['valid'].append(eligible.gather(1,ix))
        by_key['age'].append(ix.float()/keys.shape[1]-1.)
        by_key['exact'].append(match.gather(1,ix).float())
        by_key['exact2'].append(match2.gather(1,ix).float())
    z.update({name:torch.cat(parts,0) for name,parts in by_key.items()})
    return z

class QueryMirrorLM(nn.Module):
    def __init__(self,E,old_key,old_val):
        super().__init__()
        self.register_buffer('E',E.detach().clone(),persistent=False)
        D,R=old_key.weight.shape[1],old_key.weight.shape[0]
        self.key=nn.Linear(D,R,bias=False)
        self.value=nn.Linear(D,R,bias=False)
        with torch.no_grad():
            self.key.weight.copy_(old_key.weight);self.value.weight.copy_(old_val.weight)
        self.context=nn.Linear(2*D,R,bias=False)
        with torch.no_grad():
            self.context.weight.zero_()
            self.context.weight[:,:D]=old_key.weight
        self.scores=nn.Parameter(torch.tensor([11.,1.5,1.0,.20, .35, 1.]))
        self.gate=nn.Sequential(nn.Linear(8,24),nn.Tanh(),nn.Linear(24,1))
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.fill_(-6.)

    def forward(self,b,diagnostics=False):
        key=self.key(F.embedding(b['key'],self.E))
        value=self.value(F.embedding(b['val'],self.E))
        query=self.context(torch.cat((F.embedding(b['query'],self.E),F.embedding(b['prev_query'],self.E)),-1))
        k=F.normalize(key,dim=-1);v=F.normalize(value,dim=-1);q=F.normalize(query,dim=-1)
        key_sim=(k*q[:,None,:]).sum(-1)
        valid=b['valid'];prior=b['age']
        s=self.scores
        score=s[0].clamp(0,25)*key_sim+s[1]*b['exact']+s[2]*b['exact2']+s[3]*prior
        score=score.masked_fill(~valid,-1e4)
        w=F.softmax(score,dim=1)
        w=torch.where(valid,w,torch.zeros_like(w))
        w=w/w.sum(1,keepdim=True).clamp_min(1e-12)
        # ONE rank-64 state, accessed as symmetric and antisymmetric rails.
        mirror=torch.bmm((v*w[:,:,None]).transpose(1,2),k)
        plus=torch.bmm(.5*(mirror+mirror.transpose(-1,-2)),q.unsqueeze(-1)).squeeze(-1)
        minus=torch.bmm(.5*(mirror-mirror.transpose(-1,-2)),q.unsqueeze(-1)).squeeze(-1)
        directed=(v*F.normalize(plus,dim=-1)[:,None,:]).sum(-1)
        skew=(v*F.normalize(minus,dim=-1)[:,None,:]).sum(-1)
        attention=score+s[4]*directed+s[5]*skew
        attention=attention.masked_fill(~valid,-1e4)
        att=F.softmax(attention,dim=1)
        att=torch.where(valid,att,torch.zeros_like(att))
        att=att/att.sum(1,keepdim=True).clamp_min(1e-12)
        p=(att*(b['val']==b['target'][:,None])).sum(1)
        peak=att.max(1).values
        entropy=-(att.clamp_min(1e-30).log()*att).sum(1)/np.log(att.shape[1])
        match=b['exact'].sum(1).float();match2=b['exact2'].sum(1).float()
        gatefeat=torch.stack((key_sim.masked_fill(~valid,-1).max(1).values,
             peak,entropy,torch.log1p(match),torch.log1p(match2),
             valid.float().mean(1),q.norm(dim=-1),
             (score.max(1).values-(score.masked_fill(~valid,-1e4).topk(2).values[:,1])).clamp(0,20)/20),-1)
        trust=(.35*torch.sigmoid(self.gate(gatefeat).squeeze(-1)))*valid.any(1).float()
        log_p=torch.logaddexp(torch.log1p(-trust.clamp(max=1-1e-8))-b['base'],
                torch.log(trust.clamp_min(1e-30))+torch.log(p.clamp_min(1e-30)))
        loss=-log_p
        if diagnostics:return loss,{'memory_target_probability':p.detach(), 'trust':trust.detach(), 'peak':peak.detach(), 'match':match.detach()}
        return loss

def slice_d(d,idx):
    return {k:(v[idx] if torch.is_tensor(v) else len(idx)) for k,v in d.items()}

def judge(candidate,correct,wrong):
    with torch.no_grad():
        lc,dc=candidate(correct,True);lw,dw=candidate(wrong,True)
        return {'base_nll':float(correct['base'].mean()),'correct_nll':float(lc.mean()),
            'wrong_nll':float(lw.mean()),'correct_gain':float((correct['base']-lc).mean()),
            'correct_vs_wrong_advantage':float((lw-lc).mean()),
            'gate_mean':float(dc['trust'].mean()),'correct_memory_hit_rate':float((dc['memory_target_probability']>0).float().mean()),
            'wrong_memory_hit_rate':float((dw['memory_target_probability']>0).float().mean()),'n':correct['n']}

def run(args):
    start=time.monotonic(); torch.set_num_threads(args.threads);torch.manual_seed(715);np.random.seed(715)
    root=pathlib.Path(args.root);accepted=root/'checkpoint/pal_active.pt'
    before=sha256(accepted)
    if before!=PARENT_SHA:raise RuntimeError('accepted parent hash mismatch')
    model,opts,base,comp,oldgate,_=restore_r712(root,root/'adaptation/r712_gate_delta.pt')
    if len(opts)!=5:raise AssertionError('expected 5 optimizers')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    for p in oldgate.parameters():p.requires_grad_(False)
    previous_optim=copy.deepcopy(opts[4].state_dict())
    original_groups=len(opts[4].param_groups)
    print(json.dumps({'phase':'VERIFIED','accepted_step':14416,'five_optimizers':len(opts),'sha256':before}),flush=True)
    # One baseline per example; train and dev have independent positions.
    train=prepare(root,model,base,comp,'train',args.train,7151,args.topk,args.batch)
    dev=prepare(root,model,base,comp,'dev',args.dev,7152,args.topk,args.batch)
    wrong_train=wrong_history(train,max(1,args.train//3),model.trunk.E,args.topk,args.batch)
    wrong_dev=wrong_history(dev,max(1,args.dev//3),model.trunk.E,args.topk,args.batch)
    candidate=QueryMirrorLM(model.trunk.E,model.r74.outer.dir_key,model.r74.outer.dir_value)
    opts[4].add_param_group({'params':list(candidate.parameters()),'lr':args.lr,'weight_decay':0.0})
    optim=opts[4]
    initial=judge(candidate,dev,wrong_dev)
    print(json.dumps({'phase':'INIT','dev':initial,'seconds':round(time.monotonic()-start,2)}),flush=True)
    best={'step':0,'dev_nll':initial['base_nll'],'state':None,'optimizer':None}
    logs=[];rng=np.random.default_rng(715715)
    for step in range(1,args.updates+1):
        ids=torch.tensor(rng.choice(args.train,min(args.train,args.train_batch),replace=False))
        c=slice_d(train,ids);w=slice_d(wrong_train,ids)
        lc=candidate(c).mean();lw=candidate(w).mean()
        loss=lc+args.contrast*F.softplus(lc-lw+.015)
        optim.zero_grad(set_to_none=True);loss.backward()
        gradnorm=nn.utils.clip_grad_norm_(candidate.parameters(),1.0)
        optim.step()
        if step==1 or step%args.eval_every==0 or step==args.updates:
            dv=judge(candidate,dev,wrong_dev)
            if all(np.isfinite(v) for k,v in dv.items() if isinstance(v,float)):
                if dv['correct_nll']<best['dev_nll']-args.min_gain and dv['correct_vs_wrong_advantage']>args.min_adv:
                    best={'step':step,'dev_nll':dv['correct_nll'],
                          'state':copy.deepcopy(candidate.state_dict()),'optimizer':copy.deepcopy(optim.state_dict())}
            logs.append({'step':step,'dev':dv,'grad_norm':float(gradnorm)})
            print(json.dumps({'phase':'TRAIN','step':step,'dev_nll':round(dv['correct_nll'],7),
                'dev_gain':round(dv['correct_gain'],7),'memory_adv':round(dv['correct_vs_wrong_advantage'],7)}),flush=True)
    if best['state'] is not None:
        candidate.load_state_dict(best['state']);optim.load_state_dict(best['optimizer'])
    else:
        # Explicit fallback: accepted baseline; do not promote this candidate.
        with torch.no_grad():candidate.gate[-1].weight.zero_();candidate.gate[-1].bias.fill_(-80.)
    selected=judge(candidate,dev,wrong_dev)
    # Test metrics deliberately collected once after dev-only selection.
    test=prepare(root,model,base,comp,'test',args.test,7153,args.topk,args.batch)
    wrong_test=wrong_history(test,max(1,args.test//3),model.trunk.E,args.topk,args.batch)
    historical=judge(candidate,test,wrong_test)
    current=optim.state_dict()
    if current['param_groups'][:original_groups]!=previous_optim['param_groups']:
        raise RuntimeError('inherited fifth optimizer groups changed')
    for k,vals in previous_optim['state'].items():
        for name,v in vals.items():
            actual=current['state'][k][name]
            if isinstance(v,torch.Tensor):
                if not torch.equal(v,actual):raise RuntimeError('inherited moment changed')
            elif v!=actual:raise RuntimeError('inherited optimizer metadata changed')
    if before!=sha256(accepted):raise RuntimeError('accepted checkpoint changed')
    report={'experiment':'FA1 R7.15 query-conditioned symmetric/antisymmetric mirror LM',
      'parent_checkpoint_sha256':before,'accepted_step':14416,'training_candidate_updates':args.updates,
      'select_step':best['step'],'promoted':False,'full_model_weights_updated':False,
      'new_memory_writer_reader_jointly_trained_from_lm_nll':True,
      'inherited_training_state_preserved':True,'number_inherited_optimizers':len(opts),
      'inherited_fifth_optimizer_groups':original_groups,'additional_fifth_groups':1,
      'book_disjointness_verified':False,'historical_test_only_post_selection':True,
      'topk_target_free':args.topk,'counts':{'train':args.train,'dev':args.dev,'test':args.test},
      'initial_dev':initial,'selected_dev':selected,'historical_test':historical,
      'dev_gate_pass':bool(best['step']>0 and selected['correct_gain']>args.min_gain and selected['correct_vs_wrong_advantage']>args.min_adv),
      'book_level_gate_pass':False,
      'unrelated_history_control':'full 6144-token donor history shuffled, topk recomputed per current query',
      'optimizer_legacy_state_bitwise_unchanged':True,
      'elapsed_s':round(time.monotonic()-start,2),'training_log':logs}
    (root/'reports').mkdir(exist_ok=True)
    report_path=root/'reports/FA1_R7p15_cpu.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    (root/'adaptation').mkdir(exist_ok=True)
    path=root/'adaptation/fa1_r715_candidate.pt'
    group=current['param_groups'][-1]
    torch.save({'format':'FA1_R715_QUERY_MIRROR_CANDIDATE','accepted_parent_sha256':before,
      'accepted_step':14416,'promoted':False,'selected_step':best['step'],
      'new_memory_reader_writer_state':candidate.state_dict(),
      'new_optimizer_group':group,
      'new_optimizer_states':{i:current['state'][i] for i in group['params'] if i in current['state']}},path)
    report['candidate_sha256']=sha256(path);report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','selected_step':best['step'],'dev_gate':report['dev_gate_pass'],
        'selected_dev':selected,'test':historical,'elapsed_s':report['elapsed_s']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--train',type=int,default=192)
    p.add_argument('--dev',type=int,default=96)
    p.add_argument('--test',type=int,default=96)
    p.add_argument('--topk',type=int,default=96)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--train-batch',type=int,default=32)
    p.add_argument('--updates',type=int,default=160)
    p.add_argument('--eval-every',type=int,default=16)
    p.add_argument('--lr',type=float,default=.008)
    p.add_argument('--contrast',type=float,default=.20)
    p.add_argument('--min-gain',type=float,default=.00001)
    p.add_argument('--min-adv',type=float,default=.00001)
    p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())