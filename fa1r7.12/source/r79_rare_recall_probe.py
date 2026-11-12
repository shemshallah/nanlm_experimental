#!/usr/bin/env python3
"""Read-only, same-query correct/wrong rank64 old-history recall diagnostic.

Samples identifiable (once per 6144-token OLD context) key->next pairs in
last 512 old positions. Scores 32 candidates with same query/target.
Not a substitute for independent whole-book evaluation.
"""
import argparse,json,pathlib,random,time
import numpy as np,torch
from torch.nn import functional as F
from r77_semantic import restore as restore77
from r79_normalized_memory import restore as restore79
from r74_stream import OLDER,OUTER_CONTEXT
from r7_core import Corpus

def select(hist,rng,nneg=31,window=512):
    queries=[];positives=[];negatives=[];valid=[]
    for h in hist:
        a=np.asarray(h,dtype=np.int64)
        counts=np.bincount(a.clip(min=0),minlength=32000)
        cand=np.where((a[-window-1:-1]>1)&(a[-window:]>1)&(counts[a[-window-1:-1]]==1))[0]+OLDER-window-1
        if len(cand)==0:
            queries.append(1);positives.append(1);negatives.append([1]*nneg);valid.append(False);continue
        # Fixed RNG for identical query/answer/candidates across models/states.
        j=int(cand[int(rng.integers(len(cand)))])
        q=int(a[j]);y=int(a[j+1])
        pool=np.unique(a[(a>1)&(a!=y)&(a!=q)]);
        if len(pool)<nneg:
            queries.append(1);positives.append(1);negatives.append([1]*nneg);valid.append(False);continue
        queries.append(q);positives.append(y)
        negatives.append(rng.choice(pool,size=nneg,replace=False).tolist());valid.append(True)
    return (torch.as_tensor(queries),torch.as_tensor(positives),torch.as_tensor(negatives),torch.tensor(valid))

@torch.no_grad()
def eval_model(root,checkpoint,fmt,split,n=128,batch=8,seed=1729):
    torch.set_num_threads(2)
    if fmt=='r77':model,opts,b,c,prior=restore77(root,checkpoint,torch.device('cpu'))
    else:model,opts,b,c,prior=restore79(root,checkpoint)
    model.eval();co=Corpus(root,split)
    positions=8192+np.arange(n,dtype=np.int64)*(len(co.seq)-8193)//n
    rng=np.random.default_rng(seed);delta=[];correct=[];wrong=[];order=[];valid_n=0
    E=model.trunk.E
    for i in np.array_split(positions,int(np.ceil(len(positions)/batch))):
        h=co.get(i,OUTER_CONTEXT)[0][:,:OLDER].copy()
        q,y,neg,active=select(h,rng)
        h=torch.as_tensor(h);s=model.initial(h)
        qemb=F.embedding(q.long(),E)
        target=torch.cat((y[:,None],neg),1).long();emb=F.normalize(F.embedding(target,E),dim=-1)
        reverse=h.reshape(-1,OLDER//64,64).flip(-1).reshape(-1,OLDER)
        def ce(st):
            read=F.normalize(model.r74.outer.read_rails(st,qemb),dim=-1)
            logits=torch.einsum('bd,bkd->bk',read,emb)/.15
            return F.cross_entropy(logits,torch.zeros(len(q),dtype=torch.long),reduction='none')
        good=ce(s);shuffled=ce(s.roll(1,0));rev=ce(model.initial(reverse))
        if active.any():
            correct.extend(good[active].tolist());wrong.extend(shuffled[active].tolist());order.extend(rev[active].tolist())
            delta.extend((shuffled-good)[active].tolist());valid_n+=int(active.sum())
    rng=np.random.default_rng(seed+91)
    if delta:
        boot=[float(np.mean(rng.choice(delta,len(delta),replace=True))) for _ in range(2000)]
        lo,hi=np.percentile(boot,[2.5,97.5])
        return {'split':split,'n_positions':n,'valid_unique_key_queries':valid_n,
            'mean_ce_correct':float(np.mean(correct)),'mean_ce_unrelated':float(np.mean(wrong)),
            'mean_ce_within_block_reversed':float(np.mean(order)),
            'memory_advantage_ce':float(np.mean(delta)),'bootstrap_ci95_query_level':[float(lo),float(hi)],
            'bootstrap_limitation':'query-level bootstrap within historically consulted split, not book bootstrap',
            'step':int(prior['joint_steps']),'independent_books':False}
    return {'split':split,'n_positions':n,'valid_unique_key_queries':0,'unavailable':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--accepted',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--candidate',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_r79_normalized_candidate.pt'))
    p.add_argument('--n',type=int,default=128);p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args();t=time.monotonic();r={}
    for name,ck,fmt in [('accepted',a.accepted,'r77'),('r79_trial',a.candidate,'r79')]:
        r[name]={}
        for split in ('dev','test'):
            res=eval_model(a.root,ck,fmt,split,a.n)
            r[name][split]=res
            print(json.dumps({'model':name,**res}),flush=True)
    r['elapsed_seconds']=round(time.monotonic()-t,2)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
