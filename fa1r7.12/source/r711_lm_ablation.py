#!/usr/bin/env python3
"""Historical read-only LM ablation of experimental selected rank64 directed writer.

No promotion; nonzero gate does not yet implement exact incremental advance.
"""
import argparse, json, pathlib,time
import torch,numpy as np
from r711_selective_writer import restore, sha256
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER
from train_r7 import batch_data

def evaluate(root,checkpoint,split,count,batch,alphas):
    corpus=Corpus(root,split)
    model,opts,base,comp,prior=restore(root,checkpoint)
    model.eval();outer=model.r74.outer
    p=8192+np.arange(count,dtype=np.int64)*(len(corpus.seq)-8193)//count
    stats={f'{a:g}':0. for a in alphas};zerocheck=[]
    with torch.no_grad():
        for idx in np.array_split(p,int(np.ceil(count/batch))):
            h=torch.tensor(corpus.get(idx,OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
            _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,idx)
            ex=torch.tensor(corpus.data[idx].copy());lg=torch.tensor(corpus.long[idx].copy())
            for a in alphas:
                outer.dir_gate.fill_(float(np.arctanh(a)) if a<1. else 10.)
                s=model.initial(h)
                _,loss,_=model(ex,lg,ctx,pieces,lens,s,h,y)
                stats[f'{a:g}']+=float(loss.sum())
    return {k:v/count for k,v in stats.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--candidate',type=pathlib.Path,required=True)
    p.add_argument('--count',type=int,default=128);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args();torch.set_num_threads(2);t=time.monotonic()
    alphas=[0.,.05,.2,.5,.9]
    d=evaluate(a.root,a.candidate,'dev',a.count,a.batch,alphas)
    best=min(alphas,key=lambda v:d[f'{v:g}']);best=min(best,.05) if d[f'{best:g}']>=d['0'] else best
    test=evaluate(a.root,a.candidate,'test',a.count,a.batch,[0.,best])
    r={'dev_nll_by_blend':d,'test_baseline_and_dev_selected':test,'dev_selected_gate':best,
       'model_selection_uses_test':False,'historical_test_not_sealed':True,
       'note':'Only evaluates full-window reinitialization. Nonzero-gate streaming advance is not yet correct.',
       'elapsed_seconds':round(time.monotonic()-t,2)}
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
