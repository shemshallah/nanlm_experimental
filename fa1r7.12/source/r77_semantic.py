#!/usr/bin/env python3
"""PAL R7.7: learned semantic old-memory values, one unified state and five optimizers.

The original R7.6 result and all optimizer slots migrate without changes. A
new low-rank similarity head reads only the same pre-2048-window historical
FIFO and contributes a small *third mixture*, which is exactly removable.
"""
from __future__ import annotations
import os, sys, time, json, argparse, pathlib, random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r7_core import R6, Corpus, D
from r7_memory import TokenCompositor, WORD_CONTEXT, LOCAL
from r73_joint import UnifiedPAL
from r74_stream import PALR74, OUTER_CONTEXT, OLDER, build_optimizers
from r76_pair_training import PALR76, restore as restore76, evaluate as eval76
from r75_retrieval import OPT_NAMES
from train_r7 import batch_data, future_labels, stream_targets, sha
from r77_outer_aux import contrastive_loss

FORMAT='PAL_R7_7_SEMANTIC_ASSOCIATIVE_V1'
OLD_FORMAT='PAL_R7_6_PAIR_ASSOCIATIVE_V1'

class SoftAssociation(nn.Module):
    def __init__(self):
        super().__init__()
        # Identity-residual projections at migration, with trainable rank-16
        # additive updates. Novel semantic matches are explicitly non-exact.
        self.q_down=nn.Linear(D,16,bias=False)
        self.q_up=nn.Linear(16,D,bias=False)
        self.k_down=nn.Linear(D,16,bias=False)
        self.k_up=nn.Linear(16,D,bias=False)
        nn.init.normal_(self.q_down.weight,std=.01)
        nn.init.normal_(self.k_down.weight,std=.01)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.k_up.weight)
        self.log_temperature=nn.Parameter(torch.tensor(2.2))
        self.second_weight=nn.Parameter(torch.tensor(-2.5))
        self.recency=nn.Parameter(torch.tensor(-.5))
        self.cap_logit=nn.Parameter(torch.tensor(-4.0))
        self.gate_bias=nn.Parameter(torch.tensor(-2.0))
        self.gate_net=nn.Sequential(nn.Linear(2*D+3,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def lookup(self,history,ctx,E,target=None,topk=32):
        """Probability under a top-k nearest, *different-key* causal value table.

        Key history[j] => value history[j+1]; previous-key history[j-1].
        The current target is used only for loss measurement, never selection.
        """
        if history.ndim!=2 or history.shape[1]!=OLDER or ctx.shape[1]<2:
            raise ValueError('wrong historical context dimensions')
        B=history.shape[0];q=ctx[:,-1].long();q0=ctx[:,-2].long()
        keys=history[:,:-1].long();vals=history[:,1:].long()
        qv=F.embedding(q,E); qv=F.normalize(qv+self.q_up(self.q_down(qv)),dim=-1)
        kv=F.embedding(keys,E);kv=F.normalize(kv+self.k_up(self.k_down(kv)),dim=-1)
        logits=torch.bmm(kv,qv.unsqueeze(-1)).squeeze(-1)
        # Tiny causal preceding-word sketch: cannot access a newer word.
        valid_prev=history[:,:-2].long()
        prev=F.embedding(valid_prev,E)
        qprev=F.normalize(F.embedding(q0,E),dim=-1)
        prev=F.normalize(prev,dim=-1)
        prefix=(prev*qprev[:,None,:]).sum(-1)
        logits=torch.cat((logits[:,:1],logits[:,1:]+torch.sigmoid(self.second_weight)*prefix),dim=-1)
        age=torch.linspace(-1.,0.,logits.size(1),device=logits.device,dtype=logits.dtype)
        logits=logits*torch.exp(self.log_temperature).clamp(max=30)+F.softplus(self.recency)*age
        valid=(keys>1)&(vals>1)&(q[:,None]>1)&(keys!=q[:,None])
        logits=logits.masked_fill(~valid,-1.e5)
        selected=logits.topk(min(topk,logits.shape[1]),dim=1)
        vi=vals.gather(1,selected.indices)
        chosen=valid.gather(1,selected.indices)
        has=chosen.any(dim=1)
        scores=selected.values.masked_fill(~chosen,-1.e5)
        mass=torch.softmax(scores,dim=1)*chosen.float()
        mass=mass/mass.sum(1,keepdim=True).clamp_min(1e-20)
        prob=(mass*(vi==target[:,None])).sum(dim=1) if target is not None else None
        feats=torch.stack((torch.log1p(chosen.sum(1).float()),
                           mass.max(1).values, selected.values[:,0].clamp(min=-30,max=30)/30),dim=1)
        return prob,has,feats,vi,mass

    def gate(self,now,ctx,feats,E,exists):
        q=F.embedding(ctx[:,-1].long(),E)
        logit=self.gate_net(torch.cat((now,q,feats.to(now.dtype)),dim=-1)).squeeze(-1)+self.gate_bias
        cap=.2*torch.sigmoid(self.cap_logit)
        return torch.where(exists,cap*torch.sigmoid(logit),torch.zeros_like(logit))

class PALR77(PALR76):
    def __init__(self,r74):
        super().__init__(r74)
        self.retrieval.semantic=SoftAssociation()

    def forward(self,ex,long,ctx,pieces,lens,state=None,history=None,target=None,
                retrieval_off=False,pair_off=False,semantic_off=False,outer_off=False,mirror_off=False):
        now,nll,metrics=super().forward(ex,long,ctx,pieces,lens,state=state,history=history,
             target=target,retrieval_off=retrieval_off,pair_off=pair_off,outer_off=outer_off,mirror_off=mirror_off)
        metrics=dict(metrics)
        if retrieval_off or semantic_off or history is None:
            metrics['semantic_gate_mean']=now.new_zeros(())
            metrics['semantic_coverage']=now.new_zeros(())
            metrics['semantic_hits']=now.new_zeros(())
            return now,nll,metrics
        prob,exist,feat,_,_=self.retrieval.semantic.lookup(history,ctx,self.trunk.E,target)
        gate=self.retrieval.semantic.gate(now,ctx,feat,self.trunk.E,exist)
        metrics['semantic_gate_mean']=gate.detach().mean()
        metrics['semantic_coverage']=exist.float().mean().detach()
        metrics['semantic_hits']=((prob>0)&exist).float().mean().detach() if prob is not None else now.new_zeros(())
        if target is not None:
            nll=self.retrieval.mixed_nll(nll,prob,gate)
        return now,nll,metrics

def build_model(root,device,lrs):
    base=R6(root);comp=TokenCompositor(root,base)
    model=PALR77(PALR74(UnifiedPAL(base,comp,root))).to(device)
    retrieval_params=dict(model.retrieval.named_parameters())
    groups=[{'params':[p for n,p in retrieval_params.items() if not n.startswith(('pair.','semantic.'))]},
            {'params':list(model.retrieval.pair.parameters()),'weight_decay':0.0},
            {'params':list(model.retrieval.semantic.parameters()),'weight_decay':0.0}]
    opts=(*build_optimizers(model.r74,*lrs[:4]),torch.optim.AdamW(groups,lr=lrs[4],weight_decay=.0005))
    return model,opts,base,comp

def restore(root,checkpoint,device,lrs=(2e-6,6e-5,1e-4,8e-5,.002)):
    checkpoint=pathlib.Path(checkpoint);root=pathlib.Path(root)
    d=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if d['format']==OLD_FORMAT:
        original,opts,base,comp,old,prior=restore76(root,checkpoint,device,*lrs)
        model=PALR77(original.r74).to(device)
        model.retrieval=original.retrieval
        model.retrieval.semantic=SoftAssociation().to(device)
        opts[4].add_param_group({'params':list(model.retrieval.semantic.parameters()),'lr':lrs[4],'weight_decay':0.0})
        # Preserve legacy RNG: module creation consumed torch RNG.
        torch.set_rng_state(d['torch_rng']);np.random.set_state(d['numpy_rng']);random.setstate(d['python_rng'])
        return model,opts,base,comp,d
    if d['format']!=FORMAT:raise ValueError('unsupported checkpoint format: '+str(d['format']))
    if sha(root/'tokenizer/gutenberg_r7.model')!=d['tokenizer_sha256'] or sha(root/'corpus/train_temporal.i32')!=d['train_split_sha256']:
        raise ValueError('training corpus or tokenizer changed')
    model,opts,base,comp=build_model(root,device,lrs)
    model.load_state_dict(d['model'],strict=True)
    for name,opt,lr in zip(OPT_NAMES,opts,lrs):
        opt.load_state_dict(d['optimizers'][name]);
        for group in opt.param_groups:group['lr']=lr
    torch.set_rng_state(d['torch_rng']);np.random.set_state(d['numpy_rng']);random.setstate(d['python_rng'])
    return model,opts,base,comp,d

@torch.no_grad()
def evaluate(model,base,comp,corpus,count=512,batch=8,offset=0):
    model.eval();dev=next(model.parameters()).device
    positions=offset+np.arange(count,dtype=np.int64)*(len(corpus.seq)-offset-1)//count
    sums={k:0. for k in ('all','semantic_off','pair_off','retrieval_off','outer_off')}
    sums.update(n=len(positions),semantic_positive=0,semantic_candidates=0)
    for idx in np.array_split(positions,int(np.ceil(len(positions)/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,idx)
        h=torch.as_tensor(corpus.get(idx,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=dev)
        state=model.initial(h)
        ex=torch.as_tensor(corpus.data[idx].copy(),device=dev)
        long=torch.as_tensor(corpus.long[idx].copy(),device=dev)
        ctx,pieces,lens,y=[v.to(dev) for v in (ctx,pieces,lens,y)]
        now,nll,m=model(ex,long,ctx,pieces,lens,state,h,y)
        _,nll1,_=model(ex,long,ctx,pieces,lens,state,h,y,semantic_off=True)
        _,nll2,_=model(ex,long,ctx,pieces,lens,state,h,y,semantic_off=True,pair_off=True)
        _,nll3,_=model(ex,long,ctx,pieces,lens,state,h,y,retrieval_off=True)
        _,nll4,_=model(ex,long,ctx,pieces,lens,state,h,y,outer_off=True)
        for k,v in zip(('all','semantic_off','pair_off','retrieval_off','outer_off'),(nll,nll1,nll2,nll3,nll4)):sums[k]+=float(v.sum())
        prob,exist,_,_,_=model.retrieval.semantic.lookup(h,ctx,model.trunk.E,y)
        sums['semantic_positive']+=int((prob>0).sum());sums['semantic_candidates']+=int(exist.sum())
    for k in ('all','semantic_off','pair_off','retrieval_off','outer_off'):sums[k+'_nll']=sums.pop(k)/len(positions)
    sums['semantic_gain']=sums['semantic_off_nll']-sums['all_nll']
    sums['pair_gain']=sums['pair_off_nll']-sums['semantic_off_nll']
    sums['retrieval_gain']=sums['retrieval_off_nll']-sums['all_nll']
    return sums

def save(out,model,opts,prior,root,step,rng,metrics):
    root=pathlib.Path(root);out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    payload={**{k:prior[k] for k in ('base_legacy_steps','adapter_legacy_steps','legacy_r6_sha256','legacy_adapter_sha256','legacy_aux')},
      'format':FORMAT,'model':model.state_dict(),
      'optimizers':{k:o.state_dict() for k,o in zip(OPT_NAMES,opts)},
      'joint_steps':int(step),'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
      'train_split_sha256':sha(root/'corpus/train_temporal.i32'),
      'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
      'batch_rng_state':rng.bit_generator.state,
      'architecture':'R7.6 continuous mirror/context/retrieval + R7.7 novel-key trainable semantic top-k retrieval',
      'metrics':metrics}
    tmp=out.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,out);return sha(out)

def train(args):
    start=time.monotonic();torch.set_num_threads(args.threads);dev=torch.device(args.device)
    root=args.root;model,opts,base,comp,prior=restore(root,args.resume or root/'checkpoint/pal_active.pt',dev)
    original=int(prior['joint_steps']);rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    c=Corpus(root,'train');cv=Corpus(root,'dev');ct=Corpus(root,'test')
    stats={'from_step':original,'updates':args.steps,'continuity_verified':True,
           'all_five_optimizers_preserved':True,'initial_checkpoint_sha256':sha(args.resume or root/'checkpoint/pal_active.pt')}
    if args.eval:stats['before_dev']=evaluate(model,base,comp,cv,args.eval,args.eval_batch)
    for i in range(args.steps):
        model.train()
        starts=rng.integers(WORD_CONTEXT,len(c.seq)-args.seq_len,size=args.streams,dtype=np.int64)
        hist=torch.as_tensor(c.get(starts,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=dev)
        state=model.initial(hist);obj=[];aux=[];mt=[]
        for t in range(args.seq_len):
            p=starts+t
            _,ctx,pieces,lens,y,tp,tl=batch_data(base,c,comp,p)
            ex=torch.as_tensor(c.data[p].copy(),device=dev)
            long=torch.as_tensor(c.long[p].copy(),device=dev)
            ctx,pieces,lens,y,tp,tl=[v.to(dev) for v in (ctx,pieces,lens,y,tp,tl)]
            now,nll,m=model(ex,long,ctx,pieces,lens,state,hist,y)
            obj.append(nll.mean());mt.append(m)
            if args.subword_weight:aux.append(args.subword_weight*model.adapter.subword_loss(now,tp,tl))
            if args.future_weight:
                fy,valid=future_labels(c,p)
                aux.append(args.future_weight*model.r74.core.future_loss(now,fy.to(dev),valid.to(dev)))
            if args.align_weight:
                known=ctx[:,-LOCAL:]>1;composed=model.adapter.compose(pieces,lens);target=F.embedding(ctx[:,-LOCAL:],model.trunk.E)
                if bool(known.any()):aux.append(args.align_weight*F.mse_loss(composed[known],target[known]))
            if args.outer_assoc_weight:
                aux.append(args.outer_assoc_weight*contrastive_loss(model,state,hist,negatives=args.outer_assoc_negatives))
            if args.stream_weight:
                st,sl=stream_targets(c,comp,base.vocab,p,words=args.stream_words)
                aux.append(args.stream_weight*model.adapter.stream_loss(now,st.to(dev),sl.to(dev)))
            if t+1<args.seq_len:
                leaving=ctx[:,0:1].long();state=model.advance(state,leaving[:,0]);hist=torch.cat((hist[:,1:],leaving),dim=1)
        word=torch.stack(obj).mean();loss=word+(torch.stack(aux).sum()/args.seq_len if aux else word*0)
        for opt in opts:opt.zero_grad(set_to_none=True)
        loss.backward()
        assert model.trunk.P.grad is not None and model.trunk.P.grad.abs().sum()>0
        assert model.retrieval.pair.strength.grad is not None
        assert model.retrieval.semantic.cap_logit.grad is not None
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        for opt in opts:opt.step()
        if not torch.isfinite(loss):raise FloatingPointError('training divergence')
        last={'step':original+i+1,'word_nll':float(word.detach()),'loss':float(loss.detach()),
              'semantic_cap':float((.2*torch.sigmoid(model.retrieval.semantic.cap_logit)).detach()),
              'semantic_cap_grad':float(model.retrieval.semantic.cap_logit.grad.detach()),
              'semantic_hit_rate':float(torch.stack([m['semantic_hits'] for m in mt]).mean())}
        if (i+1)%args.log_every==0 or i+1==args.steps:
            print(json.dumps({'phase':'TRAIN',**last,'elapsed_seconds':round(time.monotonic()-start,2)}),flush=True)
        if args.checkpoint_every and (i+1)%args.checkpoint_every==0 and i+1<args.steps:
            rolling={**stats,'final_step':original+i+1,'completed_updates':i+1,'last_training':last}
            digest=save(args.output,model,opts,prior,root,original+i+1,rng,rolling)
            print(json.dumps({'phase':'PERIODIC_SAVE','joint_step':original+i+1,'sha256':digest}),flush=True)
    stats['final_step']=original+args.steps;stats['last_training']=last
    if args.eval:
        stats['after_dev']=evaluate(model,base,comp,cv,args.eval,args.eval_batch)
        stats['after_test']=evaluate(model,base,comp,ct,args.eval,args.eval_batch)
    # An informational comparison rather than validation on historical holdouts.
    stats['no_independent_books']=True
    stats['wall_seconds']=round(time.monotonic()-start,2)
    stats['checkpoint_sha256']=save(args.output,model,opts,prior,root,stats['final_step'],rng,stats)
    (root/'reports/R7p7_semantic_results.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps({'phase':'SAVED','checkpoint_sha256':stats['checkpoint_sha256'],'final_step':stats['final_step']}),flush=True)
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--resume',type=pathlib.Path)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--steps',type=int,default=128);p.add_argument('--streams',type=int,default=2);p.add_argument('--seq-len',type=int,default=2)
    p.add_argument('--eval',type=int,default=128);p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--clip',type=float,default=1.0)
    p.add_argument('--subword-weight',type=float,default=.08);p.add_argument('--future-weight',type=float,default=.03)
    p.add_argument('--align-weight',type=float,default=.1);p.add_argument('--stream-weight',type=float,default=.2)
    p.add_argument('--stream-words',type=int,default=6);p.add_argument('--log-every',type=int,default=25)
    p.add_argument('--outer-assoc-weight',type=float,default=0.)
    p.add_argument('--outer-assoc-negatives',type=int,default=24)
    p.add_argument('--checkpoint-every',type=int,default=500)
    train(p.parse_args())
