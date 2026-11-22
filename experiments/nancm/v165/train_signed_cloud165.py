#!/usr/bin/env python3
"""v165: learn a successor *signed* 128+128 concept cloud from v164 TRAIN-only
conditional word counts. Original cloud/model/optimizer files never overwritten.
This is a distinct trainable scorer, NOT evidence of downstream performance.
"""
import argparse,copy,hashlib,json,math,random,struct,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'parent/src'))
from native_cloud_adjoint155 import read_cloud

def word_hash(word):
    h=2166136261
    for b in word.encode('ascii'):
        h=((h^b)*16777619)&0xffffffff
    return h&127

def load_index(path, limit=0):
    examples=[]; key=None; group=[]; word_counts={};all_rows=0
    def flush():
        if key is not None and len(group)>=2:
            k,words=key
            if len(words)==k and 1<=k<=3:
                examples.append((words,group[:]))
    with open(path) as f:
        header=f.readline().strip().split('\t')
        assert header[0]=='N164DIST1' and len(header)==4,header
        for line in f:
            fields=line.strip().split('\t');assert len(fields)==5
            k,ctx,word,n,total=fields;k=int(k);n=int(n);total=int(total)
            all_rows+=1
            word_counts[word]=word_counts.get(word,0)+n if k==1 else word_counts.get(word,0)
            target=(k,tuple(ctx.split(' ')))
            if target!=key:
                flush();key=target;group=[]
            group.append((word,n,total))
        flush()
    if limit:examples=examples[:limit]
    return header,examples,word_counts,all_rows

class Cloud165(torch.nn.Module):
    def __init__(self,native):
        super().__init__()
        orig=read_cloud(native)
        # inherit the original oriented S+/S- cross-hemisphere edges exactly
        ids=[]
        for t in range(128):
            for b in range(128):
                a=t; z=128+b
                ids.append(a*(511-a)//2+(z-a-1))
        base_plus=orig['plus'][ids].reshape(128,128,8)
        base_minus=orig['minus'][ids].reshape(128,128,8)
        self.plus=torch.nn.Parameter(torch.from_numpy(base_plus.copy()))
        self.minus=torch.nn.Parameter(torch.from_numpy(base_minus.copy()))
        self.lower=torch.nn.Parameter(torch.tensor(orig['bscale'],dtype=torch.float32)*.05)
        self.upper=torch.nn.Parameter(torch.tensor(orig['tbias'],dtype=torch.float32)*.5)
        self.register_buffer('original_plus',self.plus.detach().clone())
        self.register_buffer('original_minus',self.minus.detach().clone())

    def forward(self,ctx,candidates):
        # ctx [N,3] with 128 as padding (0 contribution); candidates [N,8]
        c=ctx.clamp(max=127);t=candidates.clamp(max=127)
        valid=(ctx<128).float();recency=torch.tensor([.2,.3,.5],device=ctx.device)
        norm=(valid*recency).sum(1).clamp_min(1e-8)
        li=self.lower[c][:,None,:,:]  # [N,1,3,8]
        ui=self.upper[t][:,:,None,:]  # [N,8,1,8]
        plus=self.plus[t[:,:,None],c[:,None,:]]
        minus=self.minus[t[:,:,None],c[:,None,:]]
        # Non-equivalent plus/minus channels; two temporal directions.
        excit=(plus*torch.tanh(li+ui)).sum(-1)
        inhibit=(minus*torch.tanh(li-ui)).sum(-1)
        dot=(li*ui).sum(-1)*.35355339059
        weighted=((dot+excit-inhibit)*valid[:,None,:]*recency).sum(-1)/norm[:,None]
        return weighted

def encode(examples,word_counts):
    n=len(examples);ct=np.full((n,3),128,dtype=np.int64);tar=np.zeros((n,8),dtype=np.int64)
    mask=np.zeros((n,8),dtype=bool);prob=np.zeros((n,8),dtype=np.float32)
    prior=np.zeros((n,8),dtype=np.float32)
    nwords=sum(word_counts.values());v=len(word_counts)
    for i,(past,words) in enumerate(examples):
        for z,w in enumerate(past[-3:]):ct[i,3-len(past[-3:])+z]=word_hash(w)
        summ=sum(x[1] for x in words)
        for j,(w,c,tot) in enumerate(words[:8]):
            tar[i,j]=word_hash(w);mask[i,j]=True;prob[i,j]=c/max(1,summ)
            prior[i,j]=math.log((word_counts.get(w,0)+.5)/(nwords+.5*v))
    # per-context relative prior: no absent candidate gets positive mass.
    return tuple(torch.from_numpy(x) for x in (ct,tar,mask,prob,prior))

def score(m,dat,ids,scale=1.0,batch=512):
    m.eval();total=0.;n=0;acc=0
    with torch.no_grad():
        for start in range(0,len(ids),batch):
            ix=ids[start:start+batch];c,t,mask,p,prior=(a[ix] for a in dat)
            energy=m(c,t)
            logits=(prior+scale*energy).masked_fill(~mask,-1e7)
            logp=F.log_softmax(logits,1)
            total+=float(-(p*logp).sum());n+=len(ix)
            acc+=int((p.argmax(1)==logits.argmax(1)).sum())
    return {'cross_entropy_nats':round(total/max(1,n),7),'modal_agreement_pct':round(100*acc/max(1,n),4),'contexts':n}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--steps',type=int,default=400)
    ap.add_argument('--batch',type=int,default=256)
    ap.add_argument('--limit',type=int,default=80000)
    ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--lr',type=float,default=.01)
    a=ap.parse_args();t0=time.time()
    torch.set_num_threads(a.threads);torch.manual_seed(165);np.random.seed(165);random.seed(165)
    header,ex,counts,nrows=load_index(ROOT/'checkpoints/v164_native_distribution.tsv',a.limit)
    print('loaded',len(ex),'groups',nrows,'rows',flush=True)
    data=encode(ex,counts);n=len(ex)
    # Group-isolated diagnostic. Not a book-disjoint validation set.
    idx=np.random.default_rng(165).permutation(n)
    cut=int(n*.9);tr=idx[:cut];val=idx[cut:]
    model=Cloud165(ROOT/'parent/checkpoints/native_cloud155/v155_joint_trained.cloud')
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    base=score(model,data,val,0.)
    init=score(model,data,val,1.)
    best=init['cross_entropy_nats'];best_step=0;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    best_optimizer=copy.deepcopy(opt.state_dict())
    hist=[]
    for step in range(1,a.steps+1):
        sample=np.random.choice(tr,a.batch,replace=True)
        c,t,mask,p,prior=(x[sample] for x in data)
        model.train();opt.zero_grad(set_to_none=True)
        logits=(prior+model(c,t)).masked_fill(~mask,-1e7)
        loss=-(p*F.log_softmax(logits,1)).sum(1).mean()
        regularize=1e-4*((model.plus-model.original_plus).square().mean()+(model.minus-model.original_minus).square().mean())
        (loss+regularize).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if step%50==0 or step==a.steps:
            evaluation=score(model,data,val)
            hist.append({'step':step,'train_loss':round(float(loss.detach()),6),'validation':evaluation})
            print('EVAL',hist[-1],flush=True)
            if evaluation['cross_entropy_nats']<best:
                best=evaluation['cross_entropy_nats'];best_step=step
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                best_optimizer=copy.deepcopy(opt.state_dict())
    model.load_state_dict(best_state)
    opt.load_state_dict(best_optimizer)
    # Gated research sidecar: never replace production tensors.
    def save_native(path):
        with path.open('wb') as f:
            f.write(b'N165SCL1')
            for x in header[1:]:f.write(struct.pack('<Q',int(x)))
            f.write(struct.pack('<II',128,8))
            for key in ('lower','upper','plus','minus'):
                f.write(model.state_dict()[key].cpu().contiguous().numpy().astype('<f4').tobytes())
    out=ROOT/'checkpoints/v165_signed_cloud.bin'
    if best_step>0:save_native(out)
    report={'status':'experimental_train_index_only','training_data':'v164 TRAIN-only conditional counts, no DEV/TEST',
        'examples':n,'train_contexts':len(tr),'within_train_context_validation':len(val),
        'cross_entropy_before_zero_cloud':base,'inherited_cloud_untuned':init,
        'learned_cloud_validation':score(model,data,val),'best_step':best_step,'history':hist,
        'plus_mean_abs_update':float((model.plus-model.original_plus).abs().mean()),
        'minus_mean_abs_update':float((model.minus-model.original_minus).abs().mean()),
        'elapsed_seconds':round(time.time()-t0,2),
        'sidecar':str(out.relative_to(ROOT)) if best_step else None,
        'sidecar_sha256':hashlib.sha256(out.read_bytes()).hexdigest() if best_step else None,
        'caveat':'TRAIN-index context split is NOT book-disjoint; native DEV gate required; no production promotion'}
    (ROOT/'results/v165_train.json').write_text(json.dumps(report,indent=2))
    torch.save({'cloud':model.state_dict(),'optimizer':opt.state_dict(),'history':hist,
                'training_index_sha':hashlib.sha256((ROOT/'checkpoints/v164_native_distribution.tsv').read_bytes()).hexdigest(),
                'report':report,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()},ROOT/'checkpoints/v165_signed_cloud_train.pt')
    print('FINAL',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
