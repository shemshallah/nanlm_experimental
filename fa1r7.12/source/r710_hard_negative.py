#!/usr/bin/env python3
"""PAL R7.10 read-only paired hard-negative rank-64 memory diagnostic.

Historical corpus only, no independent book claims.  Query and answer are
selected wholly inside the older [-8192,-2049] window.  Distractors come from
other causal key->value events in that SAME window.  A fixed list of candidate
answers is scored under correct, unrelated-batch, and within-block-reversed
rank-64 states. Selection depends on the frozen pre-evaluation embeddings and
history, never model predictions. This measures a memory probe rather than LM NLL.
"""
import argparse, json, pathlib, time
import numpy as np
import torch
from torch.nn import functional as F
from r77_semantic import restore
from r74_stream import OLDER,OUTER_CONTEXT,BLOCK
from r7_core import Corpus
from train_r7 import sha

@torch.no_grad()
def pick(history, embeddings, seed, nneg=31, kind='hard'):
    """A single deterministic identifiable key->successor query and distractors.

    Hard distractors: values of competing *different* historical keys,
    ranked by 0.7*cos(candidate key, query)+0.3*cos(candidate value, positive).
    Thus all candidate values exist in the same causal older memory.
    """
    a=np.asarray(history,dtype=np.int64)
    if len(a)!=OLDER: raise ValueError('wrong history size')
    rng=np.random.default_rng(seed)
    counts=np.bincount(a.clip(min=0),minlength=embeddings.size(0))
    eligible=np.where((a[-513:-1]>1)&(a[-512:]>1)&(counts[a[-513:-1]]==1))[0] + OLDER-513
    if not len(eligible): return None
    # Choose query BEFORE constructing either random or hard distractors.
    j=int(eligible[rng.integers(len(eligible))]); q=int(a[j]); y=int(a[j+1]);
    key_ids=a[:-1]; val_ids=a[1:]
    good=(key_ids>1)&(val_ids>1)&(key_ids!=q)&(val_ids!=y)&(val_ids!=q)
    competitors=np.flatnonzero(good)
    if len(competitors)<nneg:return None
    if kind=='random':
        order=rng.permutation(competitors)
    elif kind=='hard':
        # Cosine similarity to query and answer; embeddings are pre-eval frozen.
        k=torch.from_numpy(key_ids[competitors].copy()).long()
        v=torch.from_numpy(val_ids[competitors].copy()).long()
        qe=F.normalize(embeddings[q].view(1,-1),dim=-1)
        ye=F.normalize(embeddings[y].view(1,-1),dim=-1)
        sims=(.7*(F.normalize(embeddings[k],dim=-1)@qe.T).flatten() +
              .3*(F.normalize(embeddings[v],dim=-1)@ye.T).flatten())
        # Deterministic ties; no candidate model output used to mine negatives.
        sort=np.lexsort((competitors,-sims.detach().numpy()))
        order=competitors[sort]
    else:raise ValueError(kind)
    neg=[]; seen={q,y}
    for t in order:
        word=int(val_ids[t])
        if word not in seen:
            neg.append(word);seen.add(word)
            if len(neg)==nneg:break
    if len(neg)<nneg:return None
    return {'query':q,'positive':y,'candidates':[y,*neg],'historic_source_index':j,
            'positive_occurs_in_older_history':True,
            'n_competing_pairs':int(len(competitors))}

@torch.no_grad()
def probe(root,checkpoint,split='dev',count=64,kind='hard',seed=710,batch=4,offset=8192):
    torch.set_num_threads(2)
    root=pathlib.Path(root);checkpoint=pathlib.Path(checkpoint)
    model,opts,base,comp,prior=restore(root,checkpoint,torch.device('cpu'))
    model.eval();E=model.trunk.E.detach().cpu()
    corpus=Corpus(root,split)
    if len(corpus.seq)<=offset+count:raise ValueError('not enough historical corpus rows')
    positions=offset+np.arange(count,dtype=np.int64)*(len(corpus.seq)-offset-1)//count
    stats={condition:{'ce':[],'top1':[],'margin':[]} for condition in ('correct','unrelated','reversed','zero')}
    selection=[];examples=[]; paired=[]
    for begin in range(0,len(positions),batch):
        pp=positions[begin:begin+batch]
        histories=corpus.get(pp,OUTER_CONTEXT)[0][:,:OLDER].copy()
        examples_this=[]
        for j,(pos,h) in enumerate(zip(pp,histories)):
            e=pick(h,E,seed+int(pos)*11,kind=kind)
            if e is not None:examples_this.append((j,int(pos),e))
        if not examples_this:continue
        valid_ix=[t[0] for t in examples_this]
        h=torch.from_numpy(histories[valid_ix].copy()).long()
        chosen=[t[2] for t in examples_this]
        q=torch.tensor([t['query'] for t in chosen],dtype=torch.long)
        candidate=torch.tensor([t['candidates'] for t in chosen],dtype=torch.long)
        query_embeddings=F.embedding(q,E)
        ans_emb=F.normalize(F.embedding(candidate,E),dim=-1)
        states={
          'correct':model.initial(h),
          'reversed':model.initial(h.reshape(-1,OLDER//BLOCK,BLOCK).flip(-1).reshape(-1,OLDER))}
        states['unrelated']=states['correct'].roll(1,0) if len(chosen)>1 else None
        states['zero']=torch.zeros_like(states['correct'])
        # Cross-batch roll cannot be called 'unrelated' for singleton batches.
        for condition,st in states.items():
            if st is None:continue
            read=F.normalize(model.r74.outer.read_rails(st,query_embeddings),dim=-1)
            logits=(torch.einsum('bd,bkd->bk',read,ans_emb)/.15)
            lnprob=F.log_softmax(logits,dim=-1)
            ce=(-lnprob[:,0]).numpy()
            # Fractional top-1 credit for ties; zero state should equal chance.
            ties=(logits>=logits.max(-1,keepdim=True).values-1e-7)
            top1=(ties[:,0].float()/ties.sum(-1)).numpy()
            margin=(logits[:,0]-logits[:,1:].max(-1).values).numpy()
            for k,(j,pos,e) in enumerate(examples_this):
                stats[condition]['ce'].append(float(ce[k]));stats[condition]['top1'].append(float(top1[k]));stats[condition]['margin'].append(float(margin[k]))
        for j,pos,e in examples_this:
            examples.append({'position':pos,**e})
    output={'split':split,'mining':kind,'step':int(prior['joint_steps']),
            'checkpoint_sha256':sha(checkpoint),'candidate_set_size':32,
            'attempted_positions':count,'valid_queries':len(examples),
            'independent_books':False,'query_sampling_seed':seed,
            'note':'Unsealed historic split. Query-level bootstrap is NOT book-level confidence.',
            'conditions':{},'oracle_symbolic_lookup_accuracy':1.0 if examples else None}
    for cond,dat in stats.items():
        if not dat['ce']:continue
        output['conditions'][cond]={k+'_mean':float(np.mean(v)) for k,v in dat.items()}
    # Conditions use the exact same selected queries and candidate answers.
    if len(stats['correct']['ce'])==len(stats['unrelated']['ce']) and examples:
        delta=np.asarray(stats['unrelated']['ce'])-np.asarray(stats['correct']['ce'])
        rng=np.random.default_rng(seed+999)
        bootstrap=[float(np.mean(rng.choice(delta,len(delta),replace=True))) for _ in range(1000)]
        output['correct_vs_unrelated_ce_advantage']=float(delta.mean())
        output['query_bootstrap_ci95']=list(map(float,np.percentile(bootstrap,[2.5,97.5])))
    output['chance_ce']=float(np.log(32))
    output['chance_top1']=1/32
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--count',type=int,default=64);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--seed',type=int,default=710);p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args();start=time.monotonic();report={}
    for split in ('dev','test'):
        report[split]={}
        for kind in ('random','hard'):
            res=probe(a.root,a.checkpoint,split,a.count,kind,a.seed,a.batch)
            report[split][kind]=res
            print(json.dumps({'split':split,'negative_type':kind,'n':res['valid_queries'],
              'mean_ce':res['conditions'].get('correct',{}).get('ce_mean'),
              'correct_minus_unrelated_ce_benefit':res.get('correct_vs_unrelated_ce_advantage')}),flush=True)
    report['wall_seconds']=round(time.monotonic()-start,3)
    if a.output:
        a.output.parent.mkdir(exist_ok=True,parents=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':report['wall_seconds']}),flush=True)
