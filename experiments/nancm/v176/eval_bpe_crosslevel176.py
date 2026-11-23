#!/usr/bin/env python3
"""v176 third coding: constrain initial v133-style BPE segment as well as first letter.
TRAIN-only prefix token histograms; select BPE weight on 55 VAL books only.
"""
from __future__ import annotations
import sys,json,math,collections,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from train_long175 import load_models,load_split,verify_corpus
from train_counterfactual176 import CounterfactualGate,features,mix_logprobs
from eval_crosslevel176 import CausalLetter
from third_bpe176 import BPE

class BPEFirst:
    def __init__(self,X,Y,names,merges,prior=24):
        self.bpe=BPE([tuple(a) for a in merges]);self.token=np.asarray([self.bpe.encode(w.encode('ascii'))[0] for w in names],dtype=np.int64)
        self.V=len(self.bpe.vocab);self.prior=prior
        self.global_count=np.bincount(self.token[Y],minlength=self.V).astype('float64')
        # Uniform pseudocount scaled to the 256-byte base alphabet; otherwise
        # 1600 sparse BPE classes overtake rare expected probabilities.
        self.global_p=(self.global_count+1.)/(self.global_count.sum()+self.V)
        self.c1=collections.defaultdict(collections.Counter);self.c2=collections.defaultdict(collections.Counter)
        for x,y in zip(X,Y):
            a=int(x[-1]);b=int(x[-2]);z=int(self.token[y]);self.c1[a][z]+=1;self.c2[(a,b)][z]+=1
    def batch(self,X):
        a=np.empty((len(X),self.V),dtype='float32')
        for i,row in enumerate(X):
            last=int(row[-1]);previous=int(row[-2]);t=self.c1.get(last)
            p1=self.global_p.copy()
            if t:
                p1*=self.prior
                for j,n in t.items():p1[j]+=n
                p1/=self.prior+sum(t.values())
            t2=self.c2.get((last,previous))
            if t2:
                p1*=self.prior
                for j,n in t2.items():p1[j]+=n
                p1/=self.prior+sum(t2.values())
            a[i]=p1
        return torch.from_numpy(a)

def eval(model,gate,letter,first,split,betas,root,bs=72):
    X,Y,B=load_split(split,root);res={str(k):dict(nll=0.,t1=0,t5=0) for k in betas}
    base=dict(nll=0.,t1=0,t5=0);count=len(Y);model.eval();gate.eval();sidx=torch.from_numpy(first.token)
    books=collections.defaultdict(lambda:dict(n=0,parent_loss=0.,new_loss=0.,parent_top1=0,new_top1=0))
    parent=dict(nll=0.,t1=0,t5=0)
    flidx=torch.from_numpy(letter.letters)
    with torch.no_grad():
        for i in range(0,count,bs):
            x=torch.from_numpy(X[i:i+bs]);y=torch.from_numpy(Y[i:i+bs]);
            yes,no,_=model(x,True);g=gate(features(x,no,yes))
            ln=mix_logprobs(no,yes,g)
            parent_log=F.log_softmax(yes,dim=1)
            parent_top=parent_log.topk(5,dim=1).indices
            parent_ce=-parent_log.gather(1,y[:,None])[:,0]
            parent['nll']+=float(parent_ce.sum());parent['t1']+=int((parent_top[:,0]==y).sum());parent['t5']+=int((parent_top==y[:,None]).any(1).sum())
            rootp=letter.batch(X[i:i+bs]);rootbase=torch.zeros((len(y),26))
            rootbase.index_add_(1,flidx,ln.exp())
            bias=(rootp.clamp_min(1e-8).log()-rootbase.clamp_min(1e-8).log())[:,flidx]
            full=F.log_softmax(ln+.5*bias,dim=1)
            pred=full.topk(5,dim=1).indices
            base['nll']-=float(full.gather(1,y[:,None]).sum());base['t1']+=int((pred[:,0]==y).sum());base['t5']+=int((pred==y[:,None]).any(1).sum())
            q=first.batch(X[i:i+bs]);mass=torch.zeros((len(y),first.V))
            mass.index_add_(1,sidx,full.exp())
            adjustment=(q.clamp_min(1e-8).log()-mass.clamp_min(1e-8).log())[:,sidx]
            for beta in betas:
                z=F.log_softmax(full+beta*adjustment,dim=1)
                t=z.topk(5,dim=1).indices
                res[str(beta)]['nll']-=float(z.gather(1,y[:,None]).sum())
                res[str(beta)]['t1']+=int((t[:,0]==y).sum());res[str(beta)]['t5']+=int((t==y[:,None]).any(1).sum())
                if beta==.35:
                    for k,book in enumerate(B[i:i+bs]):
                        rec=books[int(book)];rec['n']+=1;rec['parent_loss']+=float(parent_ce[k]);rec['new_loss']+=float(-z[k,y[k]]);
                        rec['parent_top1']+=int(parent_top[k,0]==y[k]);rec['new_top1']+=int(t[k,0]==y[k])
    base['nll']/=count;parent['nll']/=count
    for d in res.values():d['nll']/=count
    return dict(baseline=base,parent_v175=parent,arms=res,n=count,books=len(np.unique(B)),bookwise_0p35=books)

def bootstrap(books,reps=5000,seed=176):
    arr=np.asarray([[v['n'],v['parent_loss'],v['new_loss'],v['parent_top1'],v['new_top1']] for v in books.values()],dtype=np.float64)
    r=np.random.default_rng(seed);res=[]
    for _ in range(reps):
        a=arr[r.integers(len(arr),size=len(arr))];n=a[:,0].sum()
        res.append([(a[:,1].sum()-a[:,2].sum())/n,(a[:,4].sum()-a[:,3].sum())/n])
    q=np.asarray(res)
    return dict(books=len(books),reps=reps,nll_gain_95=np.quantile(q[:,0],[.025,.5,.975]).tolist(),
                top1_rate_gain_95=np.quantile(q[:,1],[.025,.5,.975]).tolist())

def main():
    start=time.monotonic();torch.set_num_threads(2)
    m,opt,names,fp,old=load_models();ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',weights_only=False,map_location='cpu');m.load_state_dict(ck['model_state_dict'])
    gk=torch.load(ROOT/'checkpoints/v176_counterfactual_word_gate.pt',weights_only=False,map_location='cpu')
    gate=CounterfactualGate();gate.load_state_dict(gk['gate_state_dict'])
    root=ROOT/'results/v175_release3';verify_corpus(root,names);X,Y,B=load_split('train',root)
    merges=json.loads((ROOT/'checkpoints/v176_v133_style_bpe.json').read_text())['merges']
    fl=CausalLetter(X,Y,names);first=BPEFirst(X,Y,names,merges)
    betas=[0.,.05,.1,.15,.25,.35,.5,.75,1.]
    val=eval(m,gate,fl,first,'val',betas,root)
    best=min(betas,key=lambda k:val['arms'][str(k)]['nll'])
    test=eval(m,gate,fl,first,'test',[0.,best],root)
    vb=bootstrap(val['bookwise_0p35']);tb=bootstrap(test['bookwise_0p35'])
    del val['bookwise_0p35'];del test['bookwise_0p35']
    report={'validation_book_bootstrap':vb,'reused_test_book_bootstrap':tb,'architecture':'v176 six-expert v133 ideas: 12k signed echo+noecho, trusted copy, first letter and 1600 BPE chunks, byte-verified escape',
            'source':'v175 TRAIN only','prior':24,'root_letter_beta':.5,'selected_BPE_beta':best,
            'val':val,'reused_test':test,'test_was_previously_exposed':True,'scope':'12k whole-word specialist not native byte open-vocab',
            'elapsed_s':time.monotonic()-start}
    (ROOT/'results/v176_bpe_crosslevel.json').write_text(json.dumps(report,indent=2))
    print('V176 BPE VAL',val,flush=True);print('V176 BPE TEST',test,'SELECTED',best,flush=True)

if __name__=='__main__':main()