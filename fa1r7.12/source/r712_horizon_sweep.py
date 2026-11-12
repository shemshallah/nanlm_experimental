#!/usr/bin/env python3
"""R7.12: test directed reader horizons without retraining/target leakage.
Development chooses horizon; repeatedly consulted historical test is postselection.
"""
from __future__ import annotations
import argparse,pathlib,json,time,torch,numpy as np
from torch.nn import functional as F
from r711_delta import restore_overlay
from r711_selective_writer import build_examples,score_batches,HORIZON
from r74_stream import OUTER_CONTEXT,OLDER
from r7_core import Corpus
from r710_checkpoint_integrity import sha256

def pair_horizon(outer,h,E,horizon):
    if horizon==HORIZON:return outer.pair_state(h[:,-HORIZON:],E)
    a=h[:,-horizon:-1];b=h[:,-horizon+1:]
    mask=((a>1)&(b>1)).float()
    k=F.normalize(outer.dir_key(F.embedding(a,E)),dim=-1)
    v=F.normalize(outer.dir_value(F.embedding(b,E)),dim=-1)
    w=mask*torch.linspace(-1.,0.,a.shape[1],device=h.device).exp()[None,:]
    w=w*(2*torch.sigmoid(outer.dir_select(torch.cat((k,v),-1)).squeeze(-1)))
    total=w.sum(1).clamp_min(1)
    km=(k*w.unsqueeze(-1)).sum(1)/total[:,None]
    vm=(v*w.unsqueeze(-1)).sum(1)/total[:,None]
    M=torch.bmm(((v-vm[:,None,:])*w.unsqueeze(-1)).transpose(1,2),k-km[:,None,:])/total[:,None,None]
    return M,km,vm

@torch.no_grad()
def measure(root,outer,E,split,count,seed,horizons,batch):
    d=build_examples(root,split,count,seed,E)
    c=Corpus(root,split)
    full=torch.tensor(c.get(d['positions'],OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
    results={}
    for h in horizons:
        loss=[];acc=[]
        for ix in range(0,len(d['positions']),batch):
            z=slice(ix,ix+batch);hist=full[z]
            M,km,vm=pair_horizon(outer,hist,E,h)
            logits=outer.pair_score(M,d['query'][z],d['answers'][z],E,km,vm)
            ce=F.cross_entropy(logits,torch.zeros(logits.shape[0],dtype=torch.long),reduction='none')
            ties=logits>=logits.max(-1,keepdim=True).values-1e-7
            top=ties[:,0].float()/ties.sum(-1)
            loss.extend(ce.tolist());acc.extend(top.tolist())
        results[str(h)]={'ce':float(np.mean(loss)),'top1':float(np.mean(acc)),
           'examples':len(loss),'candidate_answers':32}
    return results

def main(a):
    torch.set_num_threads(a.threads);t=time.monotonic()
    model,_,_,_,_=restore_overlay(a.root,a.root/'adaptation/r711_experimental_delta.pt')
    m=model.r74.outer;E=model.trunk.E.detach()
    horizons=[768,1536,3072,6144]
    dev=measure(a.root,m,E,'dev',a.count,1913,horizons,a.batch)
    selected=min(horizons,key=lambda h:dev[str(h)]['ce'])
    test=measure(a.root,m,E,'test',a.count,1914,[768,selected] if selected!=768 else [768],a.batch)
    report={'dev':dev,'dev_selected_horizon':selected,'test_postselection':test,
       'eval_positions':a.count,'original_horizon':768,'trained_weights_unchanged':True,
       'selection_only_uses_historical_dev':True,'historical_test_not_sealed':True,
       'elapsed_seconds':round(time.monotonic()-t,2)}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--count',type=int,default=128);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/R7p12_horizon_sweep.json'))
    main(p.parse_args())
