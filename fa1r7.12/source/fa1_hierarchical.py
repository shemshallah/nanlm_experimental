#!/usr/bin/env python3
"""FA1 R7.13: causal hierarchical selection of 768-token directed mirror memories.

This is a *read-only / auxiliary-router* experiment on PAL R7.12's unchanged
14,416-step checkpoint and selected 80-step R7.11 experimental overlay.
The hierarchy has ONE persistent rank-64 matrix per stream (the inherited state),
not eight independent models. Local directed matrices are transient evaluations
of the same learned writer on two routed historical segments.

Every candidate answer comes from history. The query is a historical key and
no present/future target enters the query->segment routing features.
This is a lexical historical-association probe, NOT semantic paraphrase or
next-token language-model evaluation. Book independence is not established.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,pathlib,random,tempfile,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER
from r711_delta import restore_overlay
from r710_checkpoint_integrity import sha256,deep_equal
from r75_retrieval import OPT_NAMES

SEG=768

class HierarchicalRouter(nn.Module):
    """Lightweight differentiable address selection, not a standalone LM."""
    def __init__(self):
        super().__init__()
        # Feature order: cosine-to-most-similar-key, exact-count, recency.
        self.weights=nn.Parameter(torch.tensor([8.,4.,0.]))
        self.bias=nn.Parameter(torch.zeros(()))

    def logits(self,features):
        if features.shape[-1]!=3: raise ValueError('expected three routing features')
        return torch.einsum('bsc,c->bs',features,self.weights)+self.bias

    def forward(self,features,topk=2):
        scores=self.logits(features)
        k=min(int(topk),scores.shape[-1]);
        if k<1:raise ValueError('topk must be positive')
        chosen=scores.topk(k,dim=-1).indices
        weights=F.softmax(scores.gather(1,chosen),dim=-1)
        return chosen,weights

@torch.no_grad()
def route_features(history,query,E):
    """Compute features from historical key positions, never from answers.

    An eligible pair (k_t,v_t) has both entries >1. A segment stores only
    768-token-local transitions; this deliberately excludes cross-segment edges.
    """
    if history.ndim!=2 or history.shape[1]%SEG or history.shape[1]<SEG:
        raise ValueError('history must be a multiple of 768 causal older tokens')
    B,T=history.shape; S=T//SEG
    chunks=history.reshape(B,S,SEG).long()
    key=chunks[:,:,:-1];value=chunks[:,:,1:]
    valid=(key>1)&(value>1)&(query[:,None,None]>1)
    qe=F.normalize(E[query.long()],dim=-1)
    ke=F.normalize(E[key],dim=-1)
    similarity=(ke*qe[:,None,None,:]).sum(-1).masked_fill(~valid,-2.)
    maxsim=similarity.max(dim=-1).values
    exact=((key==query[:,None,None])&valid).sum(-1).float()
    recency=torch.linspace(-1.,0.,S,device=history.device).repeat(B,1)
    features=torch.stack((maxsim,torch.log1p(exact),recency),dim=-1)
    eligible=valid.any(-1)
    return features,eligible


def pick_query(history,E,seed,segment=None,nneg=31):
    """Unique historical key in the last 512 tokens of a chosen segment;
    mine hard negative successor values from *other* historical keys.
    Ground-truth source segment is only used by auxiliary router training.
    """
    h=np.asarray(history,dtype=np.int64)
    if len(h)%SEG:raise ValueError('history alignment')
    rng=np.random.default_rng(seed)
    nseg=len(h)//SEG
    if segment is None:segment=int(rng.integers(nseg))
    start=segment*SEG+SEG-513;end=(segment+1)*SEG-1
    k=h[start:end];v=h[start+1:end+1]
    count=np.bincount(h.clip(min=0),minlength=len(E))
    available=np.flatnonzero((k>1)&(v>1)&(count[k]==1))
    if not len(available):return None
    t=int(start+available[rng.integers(len(available))]);q=int(h[t]);y=int(h[t+1])
    keys=h[:-1];vals=h[1:]
    other=(keys>1)&(vals>1)&(keys!=q)&(vals!=y)&(vals!=q)
    # Only within-segment pairs are valid; avoids source leakage through a
    # cross-boundary transition the hierarchical writer cannot represent.
    other[np.arange(len(other))%SEG==SEG-1]=False
    ix=np.flatnonzero(other)
    if len(ix)<nneg:return None
    with torch.no_grad():
        qemb=F.normalize(E[q,None],dim=-1).view(-1)
        yemb=F.normalize(E[y,None],dim=-1).view(-1)
        sim=.7*(F.normalize(E[torch.from_numpy(keys[ix].copy())],dim=-1)@qemb)
        sim+=.3*(F.normalize(E[torch.from_numpy(vals[ix].copy())],dim=-1)@yemb)
    order=ix[np.lexsort((ix,-sim.detach().cpu().numpy()))]
    neg=[];seen={q,y}
    for j in order:
        token=int(vals[j])
        if token not in seen:
            neg.append(token);seen.add(token)
            if len(neg)==nneg:break
    if len(neg)<nneg:return None
    return {'query':q,'answers':[y,*neg],'source_segment':segment,
            'source_position':t,'key_unique_in_older_history':True}

@torch.no_grad()
def create_data(root,split,count,seed,E,segment_policy='uniform'):
    corpus=Corpus(root,split);end=len(corpus.seq)-1
    if end<9000:raise ValueError('corpus too small')
    positions=8192+np.arange(count,dtype=np.int64)*(end-8192)//count
    rg=np.random.default_rng(seed)
    entries=[]
    for i,pos in enumerate(positions):
        h=corpus.get(np.asarray([pos]),OUTER_CONTEXT)[0][0,:OLDER].copy()
        seg=int(rg.integers(OLDER//SEG)) if segment_policy=='uniform' else OLDER//SEG-1
        ev=pick_query(h,E,int(seed+pos*7+seg),seg)
        if ev:entries.append((int(pos),h,ev))
    if not entries:raise RuntimeError('no valid associative probes')
    return {'positions':np.asarray([e[0] for e in entries]),
            'history':torch.tensor(np.stack([e[1] for e in entries]),dtype=torch.long),
            'query':torch.tensor([e[2]['query'] for e in entries],dtype=torch.long),
            'answers':torch.tensor([e[2]['answers'] for e in entries],dtype=torch.long),
            'source_segment':torch.tensor([e[2]['source_segment'] for e in entries],dtype=torch.long),
            'source_position':torch.tensor([e[2]['source_position'] for e in entries],dtype=torch.long),
            'count':len(entries)}

@torch.no_grad()
def retrieval(outer,router,E,data,batch=8,k=2,mode='correct',return_details=False):
    H=data['history'];Q=data['query'];A=data['answers'];Y=data['source_segment'];n=len(H)
    out={'ce':[],'accuracy':[],'route_hit':[],'top1_route_hit':[]};per_segment={str(i):[] for i in range(OLDER//SEG)}
    for start in range(0,n,batch):
        sl=slice(start,min(n,start+batch))
        h=H[sl].clone();q=Q[sl];a=A[sl];source=Y[sl]
        if mode=='wrong':
            # Shift by a fixed nonzero offset over all queries, NOT just within
            # a single chunk of the minibatch, to avoid accidental self-controls.
            ids=(torch.arange(start,min(n,start+batch))+max(1,n//3))%n
            h=H[ids].clone()
        if mode=='reversed':
            h=h.reshape(len(h),-1,64).flip(-1).reshape(len(h),-1)
        features,valid=route_features(h,q,E)
        choices,mixweights=router(features,topk=k)
        # Rows are combined after routing; candidate identities never influence
        # routing (they only enter the final 32-way answer scoring).
        chunks=h.reshape(len(h),-1,SEG)
        bidx=torch.arange(len(h))[:,None]
        chosen=chunks[bidx,choices].reshape(-1,SEG)
        M,km,vm=outer.pair_state(chosen,E)
        qq=q[:,None].expand_as(choices).reshape(-1)
        cand=a[:,None,:].expand(-1,choices.shape[1],-1).reshape(-1,a.shape[-1])
        logits=outer.pair_score(M,qq,cand,E,km,vm).reshape(len(h),choices.shape[1],-1)
        logp=torch.logsumexp(F.log_softmax(logits,dim=-1)+mixweights.log().unsqueeze(-1),dim=1)
        ce=-logp[:,0]
        tied=(logp>=logp.max(-1,keepdim=True).values-1.e-7)
        acc=tied[:,0].float()/tied.sum(-1)
        hit=(choices==source[:,None]).any(-1).float() if mode=='correct' else torch.zeros_like(acc)
        out['ce'].extend(ce.tolist());out['accuracy'].extend(acc.tolist());out['route_hit'].extend(hit.tolist());out['top1_route_hit'].extend((choices[:,0]==source).float().tolist())
        for seg in per_segment:
            j=int(seg);per_segment[seg].extend(ce[source==j].tolist())
    result={'ce':float(np.mean(out['ce'])),'top1':float(np.mean(out['accuracy'])),
            'route_topk_recall':float(np.mean(out['route_hit'])) if mode=='correct' else None,'route_top1_recall':float(np.mean(out['top1_route_hit'])) if mode=='correct' else None,'n':n}
    if return_details:result['per_source_segment_ce']={j:float(np.mean(v)) if len(v) else None for j,v in per_segment.items()}
    return result

@torch.no_grad()
def last768_baseline(outer,E,data,batch=8):
    losses=[];acc=[]
    for start in range(0,data['count'],batch):
        s=slice(start,start+batch);h=data['history'][s,-SEG:];q=data['query'][s];a=data['answers'][s]
        M,km,vm=outer.pair_state(h,E)
        logits=outer.pair_score(M,q,a,E,km,vm)
        ce=F.cross_entropy(logits,torch.zeros(len(q),dtype=torch.long),reduction='none')
        same=logits>=logits.max(-1,keepdim=True).values-1e-7
        losses.extend(ce.tolist());acc.extend((same[:,0].float()/same.sum(-1)).tolist())
    return {'ce':float(np.mean(losses)),'top1':float(np.mean(acc)),'n':len(losses)}


def train_router(router,opts,E,train,dev,updates=128,bs=64,eval_every=32):
    trfeat,_=route_features(train['history'],train['query'],E)
    dvfeat,_=route_features(dev['history'],dev['query'],E)
    # Add exactly one new param group to existing fifth optimizer, preserving
    # all five original optimizer histories and not modifying inherited states.
    opt=opts[4]
    original_state=copy.deepcopy(opt.state_dict())
    opt.add_param_group({'params':list(router.parameters()),'lr':.02,'weight_decay':0.0})
    rng=np.random.default_rng(71313)
    best_ce=1e30;best_state=None;best_optim=None;best_step=0
    logs=[]
    for step in range(1,updates+1):
        ids=rng.choice(len(trfeat),min(bs,len(trfeat)),replace=False)
        logits=router.logits(trfeat[ids])
        loss=F.cross_entropy(logits,train['source_segment'][ids])
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(),3.0);opt.step()
        if step==1 or step%eval_every==0 or step==updates:
            with torch.no_grad():
                dloss=F.cross_entropy(router.logits(dvfeat),dev['source_segment']).item()
                tacc=(router.logits(dvfeat).argmax(-1)==dev['source_segment']).float().mean().item()
            logs.append({'step':step,'train_ce':float(loss.detach()),'dev_segment_ce':float(dloss),'dev_top1_segment':float(tacc)})
            if dloss<best_ce-1e-6:
                best_ce=dloss;best_step=step;best_state=copy.deepcopy(router.state_dict());best_optim=copy.deepcopy(opt.state_dict())
    router.load_state_dict(best_state);opt.load_state_dict(best_optim)
    # Ensure all old optimizer moments and groups bit-exact; extra group only.
    new=opt.state_dict()
    for k,v in original_state['state'].items():
        if not deep_equal(v,new['state'][k]):raise AssertionError('accepted optimizer state modified')
    if not deep_equal(original_state['param_groups'],new['param_groups'][:-1]):
        raise AssertionError('accepted parameter groups changed')
    return {'logs':logs,'selected_step':best_step,'best_dev_segment_ce':best_ce,
            'optimizer_extra_group':new['param_groups'][-1],
            'optimizer_extra_states':{k:v for k,v in new['state'].items() if k in new['param_groups'][-1]['params']}}


def durable_torch_save(obj,out):
    out.parent.mkdir(exist_ok=True,parents=True)
    fd,path=tempfile.mkstemp(prefix='.fa1_hier_',suffix='.pending',dir=out.parent);os.close(fd)
    try:
        torch.save(obj,path)
        with open(path,'rb') as f:os.fsync(f.fileno())
        os.replace(path,out)
    finally:
        if os.path.exists(path):os.unlink(path)
    out.with_suffix(out.suffix+'.sha256').write_text(sha256(out)+'  '+out.name+'\n')


def run(a):
    torch.set_num_threads(a.threads);torch.manual_seed(713);np.random.seed(713);random.seed(713)
    root=a.root;start=time.monotonic();p=root/'checkpoint/pal_active.pt';parenthash=sha256(p)
    d11=root/'adaptation/r711_experimental_delta.pt';d12=root/'adaptation/r712_gate_delta.pt'
    model,opts,_,_,_=restore_overlay(root,d11)
    model.eval();E=model.trunk.E.detach().cpu();outer=model.r74.outer
    for par in model.parameters():par.requires_grad_(False)
    router=HierarchicalRouter()
    train=create_data(root,'train',a.train,713,E)
    dev=create_data(root,'dev',a.dev,714,E)
    baseline={'dev':last768_baseline(outer,E,dev,a.batch)}
    init={'dev':retrieval(outer,router,E,dev,batch=a.batch,k=a.topk)}
    metrics=train_router(router,opts,E,train,dev,updates=a.updates,bs=a.train_batch,eval_every=a.eval_every)
    # Selection uses ONLY dev. Test is consulted once, postselection.
    test=create_data(root,'test',a.test,715,E)
    baseline['test']=last768_baseline(outer,E,test,a.batch)
    final={}
    for split,data in (('dev',dev),('test',test)):
        final[split]={name:retrieval(outer,router,E,data,batch=a.batch,k=a.topk,mode=mode,return_details=(name=='correct'))
                 for name,mode in (('correct','correct'),('wrong','wrong'),('reversed','reversed'))}
        final[split]['correct_vs_wrong_ce_advantage']=final[split]['wrong']['ce']-final[split]['correct']['ce']
        final[split]['correct_vs_reversed_ce_advantage']=final[split]['reversed']['ce']-final[split]['correct']['ce']
        final[split]['fixed_last768']=baseline[split]
    report={'codename':'FA1','architecture':'hierarchical causal top-k directed rank64 reader',
        'accepted_checkpoint_sha256':parenthash,'accepted_joint_steps':14416,
        'accepted_checkpoint_promoted':False,'r711_directed_delta_sha256':sha256(d11),
        'r712_gate_delta_sha256':sha256(d12), 'num_segments':OLDER//SEG,
        'segment_tokens':SEG,'topk':a.topk,'query_historical_not_future':True,
        'test_book_disjointness_verified':False,'semantic_paraphrase_evaluated':False,
        'training_task':'supervised lexical historical key->successor segment routing',
        'before':init,'baseline':baseline,'router_training':{k:v for k,v in metrics.items() if k not in ('optimizer_extra_group','optimizer_extra_states')},
        'after':final,'note':'No LM promotion; this experiment does not establish frontier capability.',
        'wall_s':round(time.monotonic()-start,2)}
    # Developer caution: this experiment trains only the router; correct memory
    # must beat wrong and reversed controls on development before promotion.
    gate=(final['dev']['correct_vs_wrong_ce_advantage']>0
          and final['dev']['correct_vs_reversed_ce_advantage']>0
          and final['dev']['correct']['ce']<final['dev']['fixed_last768']['ce'])
    report['auxiliary_probe_passed_memory_controls']=bool(gate)
    report['LM_promotion_approved']=False
    if sha256(p)!=parenthash:raise AssertionError('parent checkpoint changed')
    payload={'format':'FA1_R7_13_HIERARCHICAL_ROUTER_OVERLAY_V1','accepted_parent_sha256':parenthash,
         'r711_delta_sha256':sha256(d11),'r712_gate_delta_sha256':sha256(d12),
         'router_state':router.state_dict(),
         'fifth_optimizer_new_group':metrics['optimizer_extra_group'],
         'fifth_optimizer_new_states':metrics['optimizer_extra_states'],
         'router_updates_selected':metrics['selected_step'],
         'evaluation':report}
    durable_torch_save(payload,a.overlay)
    report['hierarchy_overlay_sha256']=sha256(a.overlay)
    a.report.parent.mkdir(exist_ok=True,parents=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','report':str(a.report),'dev':final['dev']['correct'],
                      'test':final['test']['correct'],'gate':gate,'seconds':report['wall_s']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--train',type=int,default=256);p.add_argument('--dev',type=int,default=128)
    p.add_argument('--test',type=int,default=128);p.add_argument('--topk',type=int,default=2)
    p.add_argument('--updates',type=int,default=128);p.add_argument('--eval-every',type=int,default=32)
    p.add_argument('--train-batch',type=int,default=64);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--report',type=pathlib.Path,default=pathlib.Path('reports/FA1_hierarchical_r713.json'))
    p.add_argument('--overlay',type=pathlib.Path,default=pathlib.Path('adaptation/fa1_hierarchical_r713.pt'))
    run(p.parse_args())
