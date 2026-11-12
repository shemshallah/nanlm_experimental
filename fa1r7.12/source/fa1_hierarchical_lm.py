#!/usr/bin/env python3
"""FA1 R7.13 optional hierarchical directed LM readout: not promoted by default.

One inherited next-word model and its FIVE optimizer histories. The local 768
segment matrices are transient; the inherited 64x64 memory remains the only
persistent mirror state. The LM retrieval head is query-only before the target,
and scores correct and unrelated historic memories with the same frozen gate.

This is a historical split audit, not independent book or semantic validation.
"""
from __future__ import annotations
import argparse,json,pathlib,time
import numpy as np,torch
from torch.nn import functional as F
from fa1_hierarchical import HierarchicalRouter,route_features,SEG
from r712_validate import restore as restore_r712
from r712_lm_bridge import memory_probability,summary
from r74_stream import OUTER_CONTEXT,OLDER
from r7_core import Corpus
from train_r7 import batch_data
from r712_streaming import DirectedStreaming
from r710_checkpoint_integrity import sha256

@torch.no_grad()
def hierarchical_probability(outer,E,history,query,target,router,topk=2):
    if history.shape[1]!=OLDER:raise ValueError('requires causal 6144-token history')
    features,valid=route_features(history,query,E)
    selected,weights=router(features,topk)
    chunks=history.reshape(len(history),OLDER//SEG,SEG)
    rows=torch.arange(len(history))[:,None]
    ring=chunks[rows,selected].reshape(-1,SEG)
    keys=query[:,None].expand_as(selected).reshape(-1)
    targets=target[:,None].expand_as(selected).reshape(-1)
    p,f=memory_probability(outer,E,ring,keys,targets)
    p=(weights*p.reshape(len(history),-1)).sum(-1)
    f=(weights.unsqueeze(-1)*f.reshape(len(history),-1,4)).sum(1)
    return p,f

@torch.no_grad()
def collect(root,model,base,comp,router,split,count,seed,batch=8,wrong=False):
    corpus=Corpus(root,split);end=len(corpus.seq)-1
    positions=8192+np.arange(count,dtype=np.int64)*(end-8192)//count
    if seed:
        stride=max(1,(end-8192)//count)
        positions=np.minimum(end,positions+np.random.default_rng(seed).integers(0,stride,size=count))
    E=model.trunk.E.detach();outer=model.r74.outer;stream=DirectedStreaming(model)
    out={'baseline':[],'prob':[],'features':[]}
    for ids in np.array_split(positions,int(np.ceil(count/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,ids)
        hist=torch.tensor(corpus.get(ids,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
        state=stream.initial(hist)
        ex=torch.tensor(corpus.data[ids].copy()).long();lg=torch.tensor(corpus.long[ids].copy()).long()
        _,nll,_=stream.forward(ex,lg,ctx,pieces,lens,state,hist,y)
        # Match base tokens and targets; wrong-memory ablation only changes
        # historical retrieval, not the inherited model's prediction.
        if wrong:
            # Deterministic different history, even when batch has a single row.
            other=np.minimum(end,ids+max(12000,(end-8192)//7))
            hist=torch.tensor(corpus.get(other,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
        p,f=hierarchical_probability(outer,E,hist,ctx[:,-1],y,router)
        out['baseline'].append(nll.cpu());out['prob'].append(p.cpu());out['features'].append(f.cpu())
    return {k:torch.cat(v) for k,v in out.items()}|{'count':count}


def run(a):
    torch.set_num_threads(a.threads);root=a.root;start=time.monotonic()
    full=root/'checkpoint/pal_active.pt';parenthash=sha256(full)
    model,opts,base,comp,gate,r712=restore_r712(root,root/'adaptation/r712_gate_delta.pt')
    hierarchical=torch.load(root/'adaptation/fa1_hierarchical_r713.pt',map_location='cpu',weights_only=False)
    if hierarchical['accepted_parent_sha256']!=parenthash:raise ValueError('different accepted checkpoint')
    if hierarchical['r712_gate_delta_sha256']!=sha256(root/'adaptation/r712_gate_delta.pt'):
        raise ValueError('hierarchical overlay references different parent')
    router=HierarchicalRouter();router.load_state_dict(hierarchical['router_state'])
    router.eval();model.eval();gate.eval()
    metrics={}
    # Historical dev used repeatedly elsewhere: this is a diagnostic only.
    for split,n,seed in (('dev',a.dev,7313),('test',a.test,7314)):
        correct=collect(root,model,base,comp,router,split,n,seed,batch=a.batch)
        wrong=collect(root,model,base,comp,router,split,n,seed,batch=a.batch,wrong=True)
        metrics[split]={'correct':summary(gate,correct),'wrong':summary(gate,wrong),
           'correct_vs_wrong_nll_advantage':summary(gate,wrong)['nll']-summary(gate,correct)['nll']}
    dev=metrics['dev'];accept=bool(dev['correct']['gain']>0 and dev['correct_vs_wrong_nll_advantage']>0)
    report={'codename':'FA1','stage':'R7.13 hierarchical memory -> frozen R7.12 LM gate',
      'accepted_joint_step':14416,'parent_checkpoint_sha256':parenthash,
      'historical_books_not_sealed':True,'LM_promotion':False,
      'frozen_gate':True,'development_memory_gate_positive':accept,
      'metrics':metrics,'seconds':round(time.monotonic()-start,3)}
    if sha256(full)!=parenthash:raise AssertionError('checkpoint changed')
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--dev',type=int,default=256);p.add_argument('--test',type=int,default=256)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--report',type=pathlib.Path,default=pathlib.Path('reports/FA1_hierarchical_lm.json'))
    run(p.parse_args())
