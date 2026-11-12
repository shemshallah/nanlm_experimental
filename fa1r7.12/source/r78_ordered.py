#!/usr/bin/env python3
"""PAL R7.8: order-sensitive, losslessly migrated writer into SINGLE rank-64 mirror matrix.

Preserves all R7.7 weights, inherited optimizer state and deterministic batch RNG.
R7.8 extends *only* the initializer of the existing R7.4 physical matrix.
Two 8-bin temporal profiles encode signed order moments before a shared writer.
No second memory, no future leakage, no altered exact/semantic retrieval.
"""
from __future__ import annotations
import argparse,json,os,pathlib,random,time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from r7_core import Corpus,D,R
from r7_memory import WORD_CONTEXT,LOCAL
from r74_stream import CrossWindowMirror,OUTER_CONTEXT,OLDER,BLOCK
from r75_retrieval import OPT_NAMES
from r77_semantic import build_model as build77, restore as restore77
from r77_semantic import evaluate as evaluate77
from train_r7 import batch_data,future_labels,stream_targets,sha

FORMAT='PAL_R7_8_ORDERED_MIRROR_V1'
OLD_FORMAT='PAL_R7_7_SEMANTIC_ASSOCIATIVE_V1'
LRS=(2e-6,6e-5,1e-4,8e-5,.002)

class OrderedCrossWindowMirror(CrossWindowMirror):
    """Same single R x R matrix, new ordered evidence, old blocks exactly at gate=0."""
    def add_order_parameters(self):
        # The two basis profiles are zero-mean. Swapping tokens between bins
        # changes these moments, unlike a block-wide mean.
        self.order_profiles=nn.Parameter(torch.tensor([
            [-1.0,-.7142857,-.4285714,-.1428571,.1428571,.4285714,.7142857,1.0],
            [1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0]
        ],dtype=self.key.weight.dtype,device=self.key.weight.device))
        self.order_project=nn.Linear(2*D,D,bias=False,device=self.key.weight.device,dtype=self.key.weight.dtype)
        with torch.no_grad():
            self.order_project.weight.zero_()
            self.order_project.weight[:,:D].copy_(torch.eye(D,device=self.key.weight.device,dtype=self.key.weight.dtype))
            self.order_project.weight[:,D:].copy_(.5*torch.eye(D,device=self.key.weight.device,dtype=self.key.weight.dtype))
        self.order_gain=nn.Parameter(torch.zeros((),device=self.key.weight.device,dtype=self.key.weight.dtype))

    def block_events(self,history,embeddings):
        if history.ndim!=2 or history.shape[1]!=OLDER:
            raise ValueError(f'expected [batch,{OLDER}]')
        b=history.shape[0]; blocks=OLDER//BLOCK
        ids=history.reshape(b,blocks,BLOCK)
        mask=(ids!=1).to(embeddings.dtype)
        tok=F.embedding(ids.long(),embeddings)
        masked=tok*mask.unsqueeze(-1)
        counts=mask.sum(2).clamp_min(1).unsqueeze(-1)
        mean=masked.sum(2)/counts
        # Eight chronological 8-word bins. Temporal profiles are trainable and
        # shared across the SAME physical rank64 matrix. Two moments are mixed
        # into a *single* event before the existing rank-one write.
        weights=self.order_profiles.repeat_interleave(8,dim=1)
        moments=torch.einsum('bltd,kt->blkd',masked,weights)/counts.unsqueeze(2)
        ordered=self.order_project(moments.flatten(2))
        # Exactly zero at migration; no RNG or checkpoint reset needed.
        events=mean+torch.tanh(self.order_gain)*ordered
        return events,mask.sum(2)>0

    def initial(self,history,embeddings):
        events,valid=self.block_events(history,embeddings)
        state=embeddings.new_zeros(history.shape[0],R,R)
        for i in range(events.shape[1]):
            candidate=self._update(state,events[:,i])
            state=torch.where(valid[:,i,None,None],candidate,state)
        return state

def upgrade(model,opts):
    # Reclass the original module without replacing its existing Parameter
    # objects. Thus the restored Adam moments retain identical param identity.
    outer=model.r74.outer
    assert type(outer) is CrossWindowMirror, 'unexpected original outer class'
    outer.__class__=OrderedCrossWindowMirror
    outer.add_order_parameters()
    opts[3].add_param_group({'params':[outer.order_profiles,*outer.order_project.parameters(),outer.order_gain],
                             'lr':LRS[3], 'weight_decay':.0005})
    return model,opts

def restore(root,path,device='cpu'):
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d['format']==OLD_FORMAT:
        model,opts,base,comp,prior=restore77(root,path,torch.device(device),LRS)
        upgrade(model,opts)
        # preserve RNG from R7.7 because parameter initialization above consumed none
        # (all tensors are explicitly deterministic). This is conservative.
        torch.set_rng_state(prior['torch_rng']); np.random.set_state(prior['numpy_rng']);random.setstate(prior['python_rng'])
        return model,opts,base,comp,prior
    if d['format']!=FORMAT:raise ValueError('unsupported format '+str(d['format']))
    if sha(pathlib.Path(root)/'tokenizer/gutenberg_r7.model')!=d['tokenizer_sha256'] or sha(pathlib.Path(root)/'corpus/train_temporal.i32')!=d['train_split_sha256']:
        raise ValueError('source corpus/tokenizer hash mismatch')
    model,opts,base,comp=build77(root,torch.device(device),LRS)
    upgrade(model,opts)
    model.load_state_dict(d['model'],strict=True)
    for name,opt,lr in zip(OPT_NAMES,opts,LRS):
        opt.load_state_dict(d['optimizers'][name])
        for g in opt.param_groups:g['lr']=lr
    torch.set_rng_state(d['torch_rng']);np.random.set_state(d['numpy_rng']);random.setstate(d['python_rng'])
    return model,opts,base,comp,d

@torch.no_grad()
def evaluate_causal(model,base,comp,corpus,count=64,batch=2,offset=0):
    """Isolate effect of rank64 state, leave exact and semantic histories unchanged."""
    model.eval();dev=next(model.parameters()).device
    positions=offset+np.arange(count,dtype=np.int64)*(len(corpus.seq)-offset-1)//count
    keys=['correct','outer_off','order_off','swapped_batch','within_block_reversed','zero']
    sums={k:0. for k in keys}
    for idx in np.array_split(positions,int(np.ceil(len(positions)/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,idx)
        h=torch.as_tensor(corpus.get(idx,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=dev)
        ex=torch.as_tensor(corpus.data[idx].copy(),device=dev)
        long=torch.as_tensor(corpus.long[idx].copy(),device=dev)
        ctx,pieces,lens,y=[v.to(dev) for v in (ctx,pieces,lens,y)]
        state=model.initial(h)
        # Reverse inside each block: same words, SAME original averaged inputs,
        # but changed sequence moments. Does not touch exact/semantic retrieval.
        reversed_h=h.reshape(-1,OLDER//BLOCK,BLOCK).flip(-1).reshape(-1,OLDER)
        reverse_state=model.initial(reversed_h)
        saved_gain=model.r74.outer.order_gain.detach().clone()
        model.r74.outer.order_gain.zero_()
        old_writer_state=model.initial(h)
        model.r74.outer.order_gain.copy_(saved_gain)
        variants={'correct':state,'outer_off':state,'order_off':old_writer_state,'swapped_batch':state.roll(1,0),
                  'within_block_reversed':reverse_state,'zero':torch.zeros_like(state)}
        for key in keys:
            _,loss,_=model(ex,long,ctx,pieces,lens,variants[key],h,y,outer_off=(key=='outer_off'))
            sums[key]+=float(loss.sum())
    out={k+'_nll':v/count for k,v in sums.items()}
    out.update(outer_gain=float(model.r74.outer_gain),order_gain=float(model.r74.outer.order_gain),
               order_benefit_nll=out['order_off_nll']-out['correct_nll'],
               order_vs_reversed=out['within_block_reversed_nll']-out['correct_nll'],
               order_vs_swapped=out['swapped_batch_nll']-out['correct_nll'],
               outer_gain_nll=out['outer_off_nll']-out['correct_nll'],n=count)
    return out

def save(out,model,opts,prior,root,step,rng,metrics):
    root=pathlib.Path(root);out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    payload={**{k:prior[k] for k in ('base_legacy_steps','adapter_legacy_steps','legacy_r6_sha256','legacy_adapter_sha256','legacy_aux')},
       'format':FORMAT,'model':model.state_dict(),
       'optimizers':{k:o.state_dict() for k,o in zip(OPT_NAMES,opts)},
       'joint_steps':int(step),'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
       'train_split_sha256':sha(root/'corpus/train_temporal.i32'),
       'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
       'batch_rng_state':rng.bit_generator.state,
       'architecture':'R7.7 unified + rank64 order-sensitive eight-bin pair moment writer, one mirror matrix',
       'metrics':metrics}
    temp=out.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,out)
    return sha(out)

def train(args):
    start=time.monotonic();torch.set_num_threads(args.threads)
    dev=torch.device(args.device); root=args.root
    model,opts,base,comp,prior=restore(root,args.resume,dev)
    original=int(prior['joint_steps']);rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    c=Corpus(root,'train');cv=Corpus(root,'dev');ct=Corpus(root,'test')
    stats={'source_sha256':sha(args.resume),'source_step':original,'migration':OLD_FORMAT if prior['format']==OLD_FORMAT else FORMAT,
           'five_optimizer_histories_preserved':True,'from_step':original,'updates':args.steps}
    if args.eval:
        stats['before_dev']=evaluate_causal(model,base,comp,cv,args.eval,args.eval_batch)
    last=None
    for i in range(args.steps):
        model.train()
        starts=rng.integers(WORD_CONTEXT,len(c.seq)-args.seq_len,size=args.streams,dtype=np.int64)
        hist=torch.as_tensor(c.get(starts,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=dev)
        state=model.initial(hist); obj=[];aux=[]
        for t in range(args.seq_len):
            p=starts+t
            _,ctx,pieces,lens,y,tp,tl=batch_data(base,c,comp,p)
            ex=torch.as_tensor(c.data[p].copy(),device=dev);long=torch.as_tensor(c.long[p].copy(),device=dev)
            ctx,pieces,lens,y,tp,tl=[v.to(dev) for v in (ctx,pieces,lens,y,tp,tl)]
            now,nll,m=model(ex,long,ctx,pieces,lens,state,hist,y)
            obj.append(nll.mean())
            if args.subword_weight:aux.append(args.subword_weight*model.adapter.subword_loss(now,tp,tl))
            if args.future_weight:
                fy,valid=future_labels(c,p)
                aux.append(args.future_weight*model.r74.core.future_loss(now,fy.to(dev),valid.to(dev)))
            if args.align_weight:
                known=ctx[:,-LOCAL:]>1;composed=model.adapter.compose(pieces,lens);target=F.embedding(ctx[:,-LOCAL:],model.trunk.E)
                if bool(known.any()):aux.append(args.align_weight*F.mse_loss(composed[known],target[known]))
            if args.stream_weight:
                st,sl=stream_targets(c,comp,base.vocab,p,words=args.stream_words)
                aux.append(args.stream_weight*model.adapter.stream_loss(now,st.to(dev),sl.to(dev)))
            if t+1<args.seq_len:
                leaving=ctx[:,0].long();state=model.advance(state,leaving);hist=torch.cat((hist[:,1:],leaving[:,None]),dim=1)
        word=torch.stack(obj).mean();loss=word+(torch.stack(aux).sum()/args.seq_len if aux else word*0)
        for opt in opts:opt.zero_grad(set_to_none=True)
        loss.backward()
        assert model.trunk.P.grad is not None and model.retrieval.semantic.cap_logit.grad is not None
        assert model.r74.outer.order_gain.grad is not None
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        for opt in opts:opt.step()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('training divergence')
        last={'step':original+i+1,'word_nll':float(word.detach()),'loss':float(loss.detach()),
              'order_gain':float(model.r74.outer.order_gain.detach()),
              'order_gain_gradient':float(model.r74.outer.order_gain.grad.detach())}
        if (i+1)%args.log_every==0 or i+1==args.steps:
            print(json.dumps({'phase':'TRAIN',**last,'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
    stats['final_step']=original+args.steps;stats['last_train']=last
    if args.eval:
        stats['after_dev']=evaluate_causal(model,base,comp,cv,args.eval,args.eval_batch)
        stats['after_test']=evaluate_causal(model,base,comp,ct,args.eval,args.eval_batch)
    stats['independent_books_evaluated']=False
    stats['wall_seconds']=round(time.monotonic()-start,2)
    if args.output:
        stats['checkpoint_sha256']=save(args.output,model,opts,prior,root,stats['final_step'],rng,stats)
        print(json.dumps({'phase':'SAVED','step':stats['final_step'],'sha256':stats['checkpoint_sha256']}),flush=True)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(stats,indent=2)+'\n')
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--resume',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--output',type=pathlib.Path)
    p.add_argument('--report',type=pathlib.Path)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--streams',type=int,default=2)
    p.add_argument('--seq-len',type=int,default=2)
    p.add_argument('--eval',type=int,default=64)
    p.add_argument('--eval-batch',type=int,default=2)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--clip',type=float,default=1.)
    p.add_argument('--subword-weight',type=float,default=.08)
    p.add_argument('--future-weight',type=float,default=.03)
    p.add_argument('--align-weight',type=float,default=.1)
    p.add_argument('--stream-weight',type=float,default=.2)
    p.add_argument('--stream-words',type=int,default=6)
    p.add_argument('--log-every',type=int,default=8)
    train(p.parse_args())
