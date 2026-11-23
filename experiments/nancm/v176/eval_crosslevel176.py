#!/usr/bin/env python3
"""Full 12k-word coherent hybrid: signed v175, trainable copy trust,
v133-inspired frozen causal first-letter expert, and exact word top5.

All letter histograms from TRAIN books only; gate from TRAIN only. Beta selection
on 55-book VAL only, never 49-book TEST. Inherited pretraining may overlap.
"""
from __future__ import annotations
import sys,math,json,hashlib,time,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nancm/v175'))
sys.path.insert(0,str(ROOT/'experiments/nancm/v176'))
from train_long175 import load_models,load_split,verify_corpus
from train_counterfactual176 import CounterfactualGate,features,mix_logprobs

class CausalLetter:
    """v133-inspired first-letter expert using TRAIN causal word transitions.
    Interpolate counts of next first letter | last two observed words with a
    global Dirichlet prior. Every backoff tier is normalized over 26 letters.
    """
    def __init__(self,X,Y,names,prior=24):
        self.letters=np.array([ord(w[0])-97 for w in names],dtype=np.int32)
        self.global_count=np.bincount(self.letters[Y],minlength=26).astype('float64')
        self.global_p=(self.global_count+1)/(self.global_count.sum()+26)
        self.c1=np.zeros((12001,26),dtype=np.int32)
        self.c2=defaultdict(lambda:np.zeros(26,dtype=np.int32))
        # No validation/test labels in any counts.
        for hist,y in zip(X,Y):
            a=int(hist[-1]);b=int(hist[-2]);ch=int(self.letters[y]);
            self.c1[a+1,ch]+=1
            self.c2[(a,b)][ch]+=1
        self.prior=prior
    def batch(self,X):
        v=np.zeros((len(X),26),dtype=np.float32)
        for j,h in enumerate(X):
            a=int(h[-1]);b=int(h[-2]);
            t=self.c1[a+1]
            p1=(t+self.prior*self.global_p)/(t.sum()+self.prior)
            t2=self.c2.get((a,b))
            if t2 is None:q=p1
            else:q=(t2+self.prior*p1)/(t2.sum()+self.prior)
            v[j]=q
        return torch.from_numpy(v)


def metric(name,loss,top,truth):
    return dict(n=len(truth),nll=float(loss.mean()),top1=int((top[:,0]==truth).sum()),
                top5=int((top==truth[:,None]).any(1).sum()))


def run_split(model,gate,letter,split,betas,data,save_rows=False,batch=72):
    X,Y,B=load_split(split,data)
    summ={str(v):dict(loss=0.,t1=0,t5=0) for v in betas}
    baseline=dict(loss=0.,t1=0,t5=0)
    gate_only=dict(loss=0.,t1=0,t5=0)
    firstletter_correct=dict(teacher=0,other=0,hybrid=0)
    union5=0;n=len(Y)
    books=defaultdict(lambda:dict(n=0,ref_loss=0.,final_loss=0.,ref_t1=0,final_t1=0))
    model.eval();gate.eval()
    with torch.no_grad():
      for st in range(0,n,batch):
        xx=torch.from_numpy(X[st:st+batch]);yy=torch.from_numpy(Y[st:st+batch]);bb=B[st:st+batch]
        full,base,_=model(xx,True)
        zbase=F.log_softmax(full,dim=1)
        g=gate(features(xx,base,full))
        zz=mix_logprobs(base,full,g)
        pletter=letter.batch(X[st:st+batch])
        ref_fl=torch.zeros((len(yy),26),dtype=torch.float32)
        ref_fl.index_add_(1,torch.from_numpy(letter.letters),zz.exp())
        rootbias=(pletter.clamp_min(1e-8).log()-ref_fl.clamp_min(1e-8).log())
        logs={0.0:zz}
        for beta in betas:
          bfloat=float(beta)
          if bfloat:logs[beta]=F.log_softmax(zz+beta*rootbias[:,letter.letters],dim=1)
        ref_top=zbase.topk(5,dim=1).indices
        mix_top=zz.topk(5,dim=1).indices
        ref_ce=-zbase.gather(1,yy[:,None])[:,0]
        mix_ce=-zz.gather(1,yy[:,None])[:,0]
        union5+=int(((ref_top==yy[:,None])|(mix_top==yy[:,None])).any(1).sum())
        baseline['loss']+=float(ref_ce.sum());baseline['t1']+=int((ref_top[:,0]==yy).sum());baseline['t5']+=int((ref_top==yy[:,None]).any(1).sum())
        gate_only['loss']+=float(mix_ce.sum());gate_only['t1']+=int((mix_top[:,0]==yy).sum());gate_only['t5']+=int((mix_top==yy[:,None]).any(1).sum())
        for beta in betas:
          z=logs[beta];top=z.topk(5,dim=1).indices;ce=-z.gather(1,yy[:,None])[:,0]
          d=summ[str(beta)];d['loss']+=float(ce.sum());d['t1']+=int((top[:,0]==yy).sum());d['t5']+=int((top==yy[:,None]).any(1).sum())
          # Bookwise paired statistics only for beta 0.5 (diagnostic default).
          if beta==.5:
            for j,bi in enumerate(bb):
              v=books[int(bi)];v['n']+=1;v['ref_loss']+=float(ref_ce[j]);v['final_loss']+=float(ce[j]);
              v['ref_t1']+=int(ref_top[j,0]==yy[j]);v['final_t1']+=int(top[j,0]==yy[j])
    for d in [baseline,gate_only,*summ.values()]:d['nll']=d.pop('loss')/n
    return dict(split=split,n=n,books=len(np.unique(B)),baseline=baseline,gate_only=gate_only,
                root_weights=summ,top5_gate_union=union5,bookwise_0p5=books)


def book_block_bootstrap(books,seed=176,reps=5000):
    """Conditional uncertainty, whole books resampled; no claims of pristine TEST."""
    vals=list(books.values())
    if not vals:return None
    arr=np.array([[x['n'],x['ref_loss'],x['final_loss'],x['ref_t1'],x['final_t1']] for x in vals],dtype=np.float64)
    rng=np.random.default_rng(seed);delta=[]
    for _ in range(reps):
        a=arr[rng.integers(len(arr),size=len(arr))]
        n=a[:,0].sum();delta.append([(a[:,1].sum()-a[:,2].sum())/n,(a[:,4].sum()-a[:,3].sum())/n])
    d=np.array(delta)
    return {'book_count':len(vals),'resamples':reps,'nll_gain_95':np.quantile(d[:,0],[.025,.5,.975]).tolist(),
            'strict_top1_rate_gain_95':np.quantile(d[:,1],[.025,.5,.975]).tolist()}

def main():
    a=argparse.ArgumentParser();a.add_argument('--test',action='store_true');a.add_argument('--output',default='results/v176_crosslevel.json')
    args=a.parse_args();torch.set_num_threads(2)
    model,opt,names,fp,old=load_models()
    ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['model_state_dict']);model.eval()
    gk=torch.load(ROOT/'checkpoints/v176_counterfactual_word_gate.pt',weights_only=False,map_location='cpu')
    assert gk['inherited_checkpoint_sha256']==hashlib.sha256((ROOT/'checkpoints/v175_signed_episodic_48.pt').read_bytes()).hexdigest()
    gate=CounterfactualGate();gate.load_state_dict(gk['gate_state_dict'])
    source=ROOT/'results/v175_release3';verify_corpus(source,names)
    X,Y,B=load_split('train',source);letter=CausalLetter(X,Y,names)
    start=time.monotonic();betas=[0.0,.1,.25,.5,.75,1.,1.5]
    val=run_split(model,gate,letter,'val',betas,source)
    print('V176 CROSSLEVEL VAL',json.dumps({k:v for k,v in val.items() if k!='bookwise_0p5'}),flush=True)
    best=min(betas,key=lambda k:val['root_weights'][str(k)]['nll'])
    report=dict(validation={k:v for k,v in val.items() if k!='bookwise_0p5'},selected_letter_beta=best,
                gate_step=gk['step'],first_letter_train_only=True,source='NaNROUTER Release3',test_reused=True,
                caveats=['v175 inherited parent pretrained on TRAIN; incomplete ancestral corpus audit',
                         'v175 TEST previously explored; final result diagnostic not pristine',
                         '12k in-vocabulary word accuracy only; original v133 online T5 not transferable'])
    report['validation_book_bootstrap_beta_0p5']=book_block_bootstrap(val['bookwise_0p5'])
    if args.test:
      test=run_split(model,gate,letter,'test',[0.0,best],source)
      report['reused_test']={k:v for k,v in test.items() if k!='bookwise_0p5'}
      report['reused_test_book_bootstrap_beta_0p5']=book_block_bootstrap(test['bookwise_0p5'])
      print('V176 CROSSLEVEL TEST',json.dumps(report['reused_test']),flush=True)
    report['elapsed_seconds']=time.monotonic()-start
    (ROOT/args.output).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()