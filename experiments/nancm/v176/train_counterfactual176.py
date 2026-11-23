#!/usr/bin/env python3
"""v176 counterfactual trust gate for v175 signed episodic copy.

Frozen trained v175 cloud and all its optimizer moments remain untouched.
Train ONLY the new gate's parameters from 396-book TRAIN examples. Validation
uses 55 disjoint books; the 49-book previously explored TEST is never used for
model selection. Gate is an exact normalized mixture over 12000 whole words.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nancm/v175'))
from train_long175 import load_models,load_split,verify_corpus


def features(x, base, echo, names=None):
    """Features known from the causal past and *model predictions only*."""
    # X: integer word histories with sentinel -1; never use target labels.
    H=x.detach().cpu().numpy()
    f=np.zeros((len(H),14),dtype=np.float32)
    for i,row in enumerate(H):
        last=int(row[-1]);positions=[];weights={};ct2=ct3=0
        if last>=0:
            for j in range(len(row)-1):
                if row[j]!=last or row[j+1]<0:continue
                positions.append(j)
                ct2+=int(j>0 and row[j-1]==row[-2] and row[-2]>=0)
                ct3+=int(j>1 and row[j-1]==row[-2] and row[j-2]==row[-3] and row[-3]>=0)
                key=int(row[j+1]);weights[key]=weights.get(key,0)+1
        cnt=len(positions);distinct=len(weights)
        f[i,:7]=[math.log1p(cnt),math.log1p(ct2),math.log1p(ct3),
                 (max(positions)+1)/48. if cnt else 0.,
                 distinct/max(1,cnt),max(weights.values(),default=0)/max(1,cnt),
                 float(np.count_nonzero(row>=0))/48.]
    with torch.no_grad():
        a=F.log_softmax(base,dim=-1);b=F.log_softmax(echo,dim=-1)
        topa,idxa=torch.topk(a,2,dim=-1)
        topb,idxb=torch.topk(b,2,dim=-1)
        f[:,7]=(topa[:,0]-topa[:,1]).detach().cpu().numpy()
        f[:,8]=(topb[:,0]-topb[:,1]).detach().cpu().numpy()
        f[:,9]=torch.exp(topa[:,0]).detach().cpu().numpy()
        f[:,10]=torch.exp(topb[:,0]).detach().cpu().numpy()
        f[:,11]=(idxa[:,0]==idxb[:,0]).detach().cpu().numpy().astype('float32')
        f[:,12]=(b.gather(1,idxa[:,0:1])[:,0]-a.gather(1,idxa[:,0:1])[:,0]).detach().cpu().numpy()
        f[:,13]=(a.gather(1,idxb[:,0:1])[:,0]-b.gather(1,idxb[:,0:1])[:,0]).detach().cpu().numpy()
    f=np.clip(f,-15,15)
    return torch.from_numpy(f)


class CounterfactualGate(nn.Module):
    """Bounded gate, init=1 (exact parent), conditioned on short causal stats."""
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(14))
        self.bias=nn.Parameter(torch.tensor(3.0))
    def forward(self,f):
        return (.01+.98*torch.sigmoid(f@self.weight+self.bias))


def mix_logprobs(a,b,gate):
    """Exact probability simplex: (1-g)P_without_echo + gP_with_echo."""
    la=F.log_softmax(a,dim=-1)
    lb=F.log_softmax(b,dim=-1)
    g=gate[:,None]
    return torch.logaddexp(la+torch.log1p(-g),lb+torch.log(g))


def collect(model,split,source,count,seed,batch=96):
    X,Y,B=load_split(split,source)
    if count and count<len(Y):
        # Fixed per-book stratified subset (not a contiguous author/topic slice).
        rng=np.random.default_rng(seed)
        indices=[]
        u=np.unique(B)
        per=max(1,math.ceil(count/len(u)))
        for bi in u:
            ids=np.flatnonzero(B==bi);indices.extend(rng.choice(ids,min(per,len(ids)),replace=False).tolist())
        # Avoid selecting only the lowest book IDs when per-book quota exceeds count.
        chosen=np.asarray(sorted(rng.choice(np.asarray(indices),size=min(count,len(indices)),replace=False)))
        X,Y,B=X[chosen],Y[chosen],B[chosen]
    rows=[];model.eval()
    with torch.no_grad():
        for k in range(0,len(Y),batch):
            x=torch.from_numpy(X[k:k+batch]);y=torch.from_numpy(Y[k:k+batch]);book=B[k:k+batch]
            out,base,_=model(x,True)
            feats=features(x,base,out)
            logp0=F.log_softmax(base,dim=1);logp1=F.log_softmax(out,dim=1)
            rows.append(dict(f=feats,lp0=logp0.gather(1,y[:,None]).flatten(),lp1=logp1.gather(1,y[:,None]).flatten(),
                             top0=base.topk(5,dim=1).indices,top1=out.topk(5,dim=1).indices,
                             y=y,book=torch.from_numpy(book.astype('int32'))))
    ret={k:torch.cat([r[k] for r in rows],dim=0) for k in rows[0]}
    return ret


def calc_loss(data,gate):
    g=gate(data['f'])
    return -torch.logaddexp(torch.log1p(-g)+data['lp0'],torch.log(g)+data['lp1'])


def report(data,gate):
    with torch.no_grad():
        loss=calc_loss(data,gate)
        parent=-data['lp1'];off=-data['lp0']
        # This gate interpolates probabilities; top1 among 12000 needs full
        # vectors. The observed-label log probability still gives exact NLL.
        return dict(n=len(loss),gated_nll=float(loss.mean()),parent_nll=float(parent.mean()),
                    parent_off_nll=float(off.mean()),mean_gate=float(gate(data['f']).mean()),
                    oracle_union_top5=int(((data['top0']==data['y'][:,None])|(data['top1']==data['y'][:,None])).any(1).sum()),
                    parent_top5=int((data['top1']==data['y'][:,None]).any(1).sum()),
                    parent_top1=int((data['top1'][:,0]==data['y']).sum()))


def fit(train,val,steps=400):
    model=CounterfactualGate();opt=torch.optim.AdamW(model.parameters(),lr=.025,weight_decay=.001)
    best=None;hist=[]
    for step in range(steps+1):
        if step:
            model.train();opt.zero_grad(set_to_none=True)
            loss=calc_loss(train,model).mean()+.005*(model.weight@model.weight)
            loss.backward();opt.step()
        if step%20==0 or step==steps:
            model.eval();va=report(val,model);tr=report(train,model)
            hist.append(dict(step=step,train_nll=tr['gated_nll'],val_nll=va['gated_nll'],val_gate=va['mean_gate']))
            if best is None or va['gated_nll']<best['val_nll']:
                best=dict(step=step,val_nll=va['gated_nll'],state={k:v.detach().clone() for k,v in model.state_dict().items()},opt=opt.state_dict())
    model.load_state_dict(best['state']);return model,best,hist


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-count',type=int,default=12000)
    ap.add_argument('--val-count',type=int,default=13200);ap.add_argument('--test-count',type=int,default=0)
    ap.add_argument('--steps',type=int,default=300);ap.add_argument('--seed',type=int,default=176)
    args=ap.parse_args();torch.set_num_threads(2);np.random.seed(args.seed);torch.manual_seed(args.seed)
    m,opt,names,fp,old=load_models()
    oldck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',weights_only=False,map_location='cpu')
    m.load_state_dict(oldck['model_state_dict']);m.echo_enabled=True
    for p in m.parameters():p.requires_grad_(False)
    source=ROOT/'results/v175_release3';manifest=verify_corpus(source,names)
    st=time.monotonic();train=collect(m,'train',source,args.train_count,args.seed)
    val=collect(m,'val',source,args.val_count,args.seed+1)
    gate,best,history=fit(train,val,args.steps)
    vr=report(val,gate);tr=report(train,gate)
    ck={'gate_state_dict':best['state'],'optimizer_state_dict':best['opt'],
        'inherited_checkpoint':'checkpoints/v175_signed_episodic_48.pt',
        'inherited_checkpoint_sha256':hashlib.sha256((ROOT/'checkpoints/v175_signed_episodic_48.pt').read_bytes()).hexdigest(),
        'inherited_optimizer_group_count':4,'inherited_optimizer_state_entries':19,
        'step':best['step'],'feature_order':'14 causal counts, recency, diversity, both predictive distributions (documented in features())',
        'source_manifest_sha256':hashlib.sha256((source/'manifest175.json').read_bytes()).hexdigest(),
        'status':'experimental not promoted'}
    path=ROOT/'checkpoints/v176_counterfactual_word_gate.pt';torch.save(ck,path)
    out={'model':'v176 counterfactual trust gate + v175 signed-temporal pointers',
         'training_data':'NaNROUTER Release 0003 TRAIN only','train':tr,'val':vr,'selected_step':best['step'],'history':history,
         'test_uninspected':not args.test_count,'note':'word distribution train gate: NLL only until full-vector top1 re-evaluation',
         'elapsed_s':time.monotonic()-st}
    if args.test_count:
        test=collect(m,'test',source,args.test_count,args.seed+2);out['reused_test']=report(test,gate)
    (ROOT/'results/v176_gate_train.json').write_text(json.dumps(out,indent=2))
    print('V176 GATE',json.dumps({k:v for k,v in out.items() if k!='history'}),flush=True)

if __name__=='__main__':main()