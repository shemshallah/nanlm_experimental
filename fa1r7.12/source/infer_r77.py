#!/usr/bin/env python3
"""Causal R7.7 generation from the *same trained mixture* used for word NLL.

Draw from a four-way hierarchical mixture: original tree, exact one-word,
exact ordered two-word, semantic non-exact old key. This avoids the incorrect
practice of rescoring only the top-k tree words and dropping retrieval values.
"""
from __future__ import annotations
import argparse,json,pathlib,re,collections
import numpy as np,torch
from torch.nn import functional as F
from r77_semantic import restore
from r74_stream import OLDER,OUTER_CONTEXT
from r7_memory import WORD_CONTEXT,LOCAL
from infer_unified_now import RGX
from infer_longctx2048 import context_features

@torch.no_grad()
def tree_sample(model,base,h,rng,temperature=1.0):
    """Exact hierarchical tree draw when temperature=1; others scale node logits."""
    node=2*len(base.vocab)-2
    while node>=len(base.vocab):
        j=node-len(base.vocab)
        logit=float(torch.dot(model.trunk.W[j],h)+model.trunk.B[j])
        right=rng.random()<1/(1+np.exp(-np.clip(logit/max(temperature,1e-3),-50,50)))
        node=int(base.children[j,int(right)])
    return node

@torch.no_grad()
def run(root,checkpoint,prompt,n=60,seed=17,temperature=.9,
        semantic_off=False,pair_off=False,retrieval_off=False,outer_off=False):
    root=pathlib.Path(root);torch.set_num_threads(2)
    m,opts,base,comp,d=restore(root,checkpoint,torch.device('cpu'));m.eval()
    rng=np.random.default_rng(seed);lookup={s:i for i,s in enumerate(base.vocab)}
    observed=RGX.findall(prompt.lower());known=[lookup.get(s,0) for s in observed]
    previous=[1]*OUTER_CONTEXT+known
    h_old=torch.tensor([previous[-OUTER_CONTEXT:-WORD_CONTEXT]],dtype=torch.long)
    state=m.initial(h_old)
    generated=[];sources=[];gates=[]
    for step in range(n):
        ctx=torch.tensor([[*([1]*WORD_CONTEXT+known)[-WORD_CONTEXT:]]],dtype=torch.long)
        octx=h_old
        ex=torch.tensor([[*([1,1,1]+known)[-3:],0,1,1,1]],dtype=torch.long)
        long=torch.from_numpy(context_features(known).copy())[None,:]
        latest=(['<bos>']*LOCAL+observed)[-LOCAL:]
        pc,ll=comp.encode_words(latest)
        now,_=m.r74(ex,long,ctx,pc[None,:,:],ll[None,:],state,outer_off=outer_off)
        query=ctx[:,-1]
        pg,exists,feats,one_weights=m.retrieval.lookup(octx,query)
        one_gate=m.retrieval.gate(now,query,feats,m.trunk.E,exists)
        _,haspair,pairfeat=m.pair_lookup(octx,ctx)
        pair_gate=m.pair_gate(now,ctx,pairfeat,haspair)
        _,hassem,semfeat,semids,semmass=m.retrieval.semantic.lookup(octx,ctx,m.trunk.E)
        sem_gate=m.retrieval.semantic.gate(now,ctx,semfeat,m.trunk.E,hassem)
        g1=float(one_gate.item()) if not retrieval_off else 0.
        g2=float(pair_gate.item()) if not (pair_off or retrieval_off) else 0.
        g3=float(sem_gate.item()) if not (semantic_off or retrieval_off) else 0.
        # Sequential interpolation exactly matches the training loss.
        components=[('semantic',g3),('pair',(1-g3)*g2),('exact',(1-g3)*(1-g2)*g1),
                    ('tree',(1-g3)*(1-g2)*(1-g1))]
        names,ps=zip(*components);p=np.asarray(ps,dtype=np.float64);p/=p.sum()
        choice=int(rng.choice(len(p),p=p));mode=names[choice]
        if mode=='semantic':
            ids=semids[0].cpu().numpy();weights=semmass[0].cpu().numpy().astype(float)
            weights/=weights.sum();wid=int(rng.choice(ids,p=weights))
        elif mode=='pair':
            prev2=ctx[:,-2:]
            match=(octx[:,:-2]==prev2[:,:1])&(octx[:,1:-1]==prev2[:,1:]) & (octx[:,:-2]>1)&(octx[:,1:-1]>1)&(octx[:,2:]>1)
            ages=torch.arange(OLDER-2,dtype=torch.float32)
            ages=(ages-(OLDER-3))/(OLDER-2)
            w=match.float()*torch.exp(F.softplus(m.retrieval.recency)*ages)
            probs=w[0].cpu().numpy().astype(float);probs/=probs.sum()
            wid=int(rng.choice(octx[0,2:].numpy(),p=probs))
        elif mode=='exact':
            prob=one_weights[0].cpu().numpy().astype(float);prob/=prob.sum()
            wid=int(rng.choice(octx[0,1:].numpy(),p=prob))
        else:
            wid=tree_sample(m,base,now[0],rng,temperature)
            if wid==1:wid=0
        if wid==0:
            fallback=m.adapter.spell(now[0],comp,max_len=22,seed=seed+100019*step)
            word=fallback if re.fullmatch(r"[a-z][a-z'\-]{1,32}",fallback or '') else '<unk>'
        else:word=base.vocab[wid]
        leaving=ctx[:,0].clone()
        state=m.advance(state,leaving)
        h_old=torch.cat((h_old[:,1:],leaving[:,None]),dim=1)
        known.append(wid);observed.append(word);generated.append(word);sources.append(mode)
        gates.append((g1,g2,g3))
    return {'joint_step':d['joint_steps'],'prompt':prompt,'generated':' '.join(generated),
            'source_counts':dict(collections.Counter(sources)),
            'mean_mixture_gates':{'one_word':float(np.mean(gates,axis=0)[0]),
                                   'pair':float(np.mean(gates,axis=0)[1]),
                                   'semantic':float(np.mean(gates,axis=0)[2])},
            'seed':seed,'temperature':temperature,'future_tokens_provided':False,
            'note':'Original word-stream normalization; output remains an exploratory fluency probe.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    ap.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    ap.add_argument('--prompt',default='the stars were shining above the old house')
    ap.add_argument('--prompt-file',type=pathlib.Path)
    ap.add_argument('--prompt-words',type=int,default=8192)
    ap.add_argument('--n',type=int,default=60);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--temperature',type=float,default=.9)
    ap.add_argument('--semantic-off',action='store_true');ap.add_argument('--pair-off',action='store_true')
    ap.add_argument('--retrieval-off',action='store_true');ap.add_argument('--outer-off',action='store_true')
    a=ap.parse_args()
    prompt=' '.join(RGX.findall(a.prompt_file.read_text(errors='replace').lower())[-a.prompt_words:]) if a.prompt_file else a.prompt
    print(json.dumps(run(a.root,a.checkpoint,prompt,a.n,a.seed,a.temperature,a.semantic_off,a.pair_off,a.retrieval_off,a.outer_off),indent=2))
