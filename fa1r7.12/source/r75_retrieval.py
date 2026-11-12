#!/usr/bin/env python3
"""PAL R7.5 — exact causal older-window associations and one jointly trained PAL.

Migration loads *all* R7.4 model/optimizer/RNG state and attaches an exact
(key=previous word, value=next word) older-memory path to its loss. The 6144
older tokens are runtime state (not parameters, not another checkpoint).
Only tokens at or before p-2049 are available for target at p.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r7_core import R6, Corpus, D
from r7_memory import TokenCompositor, WORD_CONTEXT, LOCAL
from r73_joint import UnifiedPAL
from r74_stream import PALR74, OLDER, OUTER_CONTEXT, OPT_NAMES as OLD_OPT_NAMES, build_optimizers, restore as restore_r74
from train_r7 import batch_data, future_labels, stream_targets, sha

CHECKPOINT_FORMAT='PAL_R7_5_ASSOCIATIVE_V1'
OPT_NAMES=(*OLD_OPT_NAMES,'retrieval_adamw')

class ExactAssociation(nn.Module):
    """Sparse exact older bigram retrieval; learned recency and calibrated gate.

    Uses no target labels to construct keys/candidates. No separate prediction
    model or trainable per-document table. Values are historical token IDs.
    """
    def __init__(self):
        super().__init__()
        self.recency=nn.Parameter(torch.tensor(-0.5))
        self.gate_net=nn.Sequential(nn.Linear(2*D+3,48),nn.Tanh(),nn.Linear(48,1))
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias,-1.0)
        self.gate_cap=nn.Parameter(torch.tensor(-3.5))

    def lookup(self,history,query,target=None):
        """Return target probability, candidate existence, and gate features.

        Candidate pairs come from history[:,0:6143] -> history[:,1:6144],
        entirely before the 2048-token exact recent window. Works with BOS.
        """
        if history.ndim!=2 or history.size(1)!=OLDER:raise ValueError('older history shape')
        if query.ndim!=1 or query.size(0)!=history.size(0):raise ValueError('query shape')
        match=(history[:,:-1]==query[:,None]) & (query[:,None]>1) & (history[:,:-1]>1) & (history[:,1:]>1)
        nmatch=match.sum(-1)
        age=torch.arange(OLDER-1,device=history.device,dtype=torch.float32)
        age=(age-(OLDER-2))/(OLDER-1)
        score=F.softplus(self.recency)*age
        weight=match.to(torch.float32)*torch.exp(score).unsqueeze(0)
        total=weight.sum(-1)
        if target is None:
            target_weight=None
        else:
            if target.ndim!=1 or target.shape!=query.shape:raise ValueError('target shape')
            target_weight=(weight*(history[:,1:]==target[:,None])).sum(-1)
        probability=target_weight/total.clamp_min(1e-20) if target is not None else None
        newest=torch.where(match,torch.arange(OLDER-1,device=history.device).unsqueeze(0),-1).max(-1).values
        features=torch.stack((torch.log1p(nmatch.float()),total.log1p(),((newest.float()+1)/OLDER)),dim=-1)
        return probability,nmatch>0,features,weight

    def gate(self,now,query,features,embeddings,exists):
        embedding=F.embedding(query.long(),embeddings)
        logits=self.gate_net(torch.cat((now,embedding,features.to(now.dtype)),-1)).squeeze(-1)
        # Small nonzero cap gives a measurable training gradient immediately;
        # ablated path exactly reproduces the R7.4 forward predictions.
        cap=torch.sigmoid(self.gate_cap)
        return torch.where(exists,cap*torch.sigmoid(logits),torch.zeros_like(logits))

    def mixed_nll(self,base_nll,probability,gate):
        if base_nll.shape!=gate.shape or probability.shape!=gate.shape:raise ValueError('NLL shape')
        log_base=torch.log1p(-gate.clamp(max=1-1e-7))-base_nll
        log_retrieval=torch.log(gate.clamp_min(1.e-30))+torch.log(probability.clamp_min(1.e-30))
        return -torch.logaddexp(log_base,log_retrieval)

    def candidate_distribution(self,history,query,vocab_size):
        """Sparse exact values folded into full vocabulary probability for inference."""
        _,exists,features,weight=self.lookup(history,query)
        result=weight.new_zeros((history.size(0),vocab_size))
        result.scatter_add_(1,history[:,1:].long(),weight)
        result=result/result.sum(1,keepdim=True).clamp_min(1e-20)
        return result,exists,features

class PALR75(nn.Module):
    def __init__(self,r74):
        super().__init__();self.r74=r74;self.retrieval=ExactAssociation()
    @property
    def trunk(self):return self.r74.trunk
    @property
    def adapter(self):return self.r74.adapter
    def initial(self,history):return self.r74.initial(history)
    def advance(self,state,leaving):return self.r74.advance(state,leaving)
    def forward(self,ex,long,ctx,pieces,lens,state=None,history=None,
                target=None,retrieval_off=False,outer_off=False,mirror_off=False):
        now,features=self.r74(ex,long,ctx,pieces,lens,state=state,outer_off=outer_off,mirror_off=mirror_off)
        base_nll=self.trunk.nll(now,target) if target is not None else None
        if retrieval_off or history is None:
            return now,base_nll,{'gate_mean':0.,'key_coverage':0.}
        query=ctx[:,-1].long()
        prob,exists,aux,_=self.retrieval.lookup(history,query,target)
        gate=self.retrieval.gate(now,query,aux,self.trunk.E,exists)
        metrics={'gate_mean':gate.mean().detach(),'key_coverage':exists.float().mean().detach()}
        if target is not None:
            metrics['hit_rate']=((prob>0)&exists).float().mean().detach()
            nll=self.retrieval.mixed_nll(base_nll,prob,gate)
        else:nll=None
        return now,nll,metrics


def restore(root,checkpoint,device,trunk_lr=2e-6,adapter_lr=6e-5,mirror_lr=1e-4,
            outer_lr=8e-5,retrieval_lr=0.002):
    root=pathlib.Path(root);checkpoint=pathlib.Path(checkpoint)
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if data['format']=='PAL_R7_4_CROSSWINDOW_V1':
        r74,oldopts,base,comp,old,previous=restore_r74(root,checkpoint,device,trunk_lr,adapter_lr,mirror_lr,outer_lr)
        model=PALR75(r74).to(device)
        retopt=torch.optim.AdamW(model.retrieval.parameters(),lr=retrieval_lr,weight_decay=0.0005)
        return model,(*oldopts,retopt),base,comp,old,previous
    if data['format']!=CHECKPOINT_FORMAT:raise ValueError('not an R7.4/R7.5 checkpoint')
    if sha(root/'tokenizer/gutenberg_r7.model')!=data['tokenizer_sha256']:raise ValueError('tokenizer changed')
    if sha(root/'corpus/train_temporal.i32')!=data['train_split_sha256']:raise ValueError('train data changed')
    base=R6(root);comp=TokenCompositor(root,base)
    model=PALR75(PALR74(UnifiedPAL(base,comp,root))).to(device)
    opts=(*build_optimizers(model.r74,trunk_lr,adapter_lr,mirror_lr,outer_lr),
          torch.optim.AdamW(model.retrieval.parameters(),lr=retrieval_lr,weight_decay=.0005))
    model.load_state_dict(data['model'],strict=True)
    for name,opt,lr in zip(OPT_NAMES,opts,(trunk_lr,adapter_lr,mirror_lr,outer_lr,retrieval_lr)):
        opt.load_state_dict(data['optimizers'][name])
        for group in opt.param_groups:group['lr']=lr
    torch.set_rng_state(data['torch_rng']);np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])
    old={'base_step':data['base_legacy_steps'],'adapter_updates':data['adapter_legacy_steps'],
         'base_sha256':data['legacy_r6_sha256'],'legacy_adapter_sha256':data['legacy_adapter_sha256'],
         'legacy_aux':data.get('legacy_aux',{})}
    return model,opts,base,comp,old,data

@torch.no_grad()
def evaluate(model,base,comp,corpus,count=128,batch=8):
    model.eval();device=next(model.parameters()).device
    positions=np.arange(count,dtype=np.int64)*(len(corpus.seq)-1)//count
    stats={'active':0.,'retrieval_off':0.,'outer_off':0.,'key_positions':0,'target_hits':0,'n':count}
    for indexes in np.array_split(positions,int(np.ceil(count/batch))):
        _,ctx,pc,ln,target,_,_=batch_data(base,corpus,comp,indexes)
        history=torch.as_tensor(corpus.get(indexes,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=device)
        state=model.initial(history)
        a=torch.as_tensor(corpus.data[indexes].copy(),device=device);l=torch.as_tensor(corpus.long[indexes].copy(),device=device)
        ctx,pc,ln,target=(t.to(device) for t in (ctx,pc,ln,target))
        now,nll,metrics=model(a,l,ctx,pc,ln,state,history,target)
        base_nll=model.trunk.nll(now,target)
        _,no_outer,_=model(a,l,ctx,pc,ln,state,history,target,outer_off=True)
        stats['active']+=float(nll.sum());stats['retrieval_off']+=float(base_nll.sum());stats['outer_off']+=float(no_outer.sum())
        p,has,_,_=model.retrieval.lookup(history,ctx[:,-1],target)
        stats['key_positions']+=int(has.sum());stats['target_hits']+=int((p>0).sum())
    for k in ('active','retrieval_off','outer_off'):stats[k+'_nll']=stats.pop(k)/count
    stats['retrieval_gain_nats']=stats['retrieval_off_nll']-stats['active_nll']
    return stats


def save(out,model,opts,prior,root,updates,rng,metrics):
    root=pathlib.Path(root);out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':CHECKPOINT_FORMAT,'model':model.state_dict(),
        'optimizers':{k:o.state_dict() for k,o in zip(OPT_NAMES,opts)},
        'base_legacy_steps':int(prior['base_legacy_steps']),'adapter_legacy_steps':int(prior['adapter_legacy_steps']),
        'joint_steps':int(updates),'legacy_r6_sha256':prior['legacy_r6_sha256'],
        'legacy_adapter_sha256':prior['legacy_adapter_sha256'],
        'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
        'train_split_sha256':sha(root/'corpus/train_temporal.i32'),
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'batch_rng_state':rng.bit_generator.state,'legacy_aux':prior.get('legacy_aux',{}),
        'architecture':'R7.4 joint PAL + single exact, learned-gated older-bigram retrieval over 6144 words',
        'metrics':metrics}
    tmp=out.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,out)
    return sha(out)


def train(args):
    start=time.monotonic();torch.set_num_threads(args.threads)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    model,opts,base,comp,old,prior=restore(args.root,args.resume or args.root/'checkpoint/pal_active.pt',device,
        args.trunk_lr,args.adapter_lr,args.mirror_lr,args.outer_lr,args.retrieval_lr)
    if sha(args.root/'corpus/train_temporal.i32')!=prior['train_split_sha256']:raise ValueError('unexpected training split')
    current=prior['joint_steps'];rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    traincor=Corpus(args.root,'train');devcor=Corpus(args.root,'dev');testcor=Corpus(args.root,'test')
    reports={'start_joint_step':int(current),'new_updates':args.steps,'initial_format':prior['format']}
    if args.eval:
        for name,cor in (('dev',devcor),('test',testcor)):
            reports['before_'+name]=evaluate(model,base,comp,cor,args.eval,args.eval_batch)
    for update in range(args.steps):
        model.train()
        starts=rng.integers(WORD_CONTEXT,len(traincor.seq)-args.seq_len,size=args.streams,dtype=np.int64)
        if traincor.doc_start is not None:
            for j,s in enumerate(starts):
                if traincor.doc_start[s]!=traincor.doc_start[s+args.seq_len-1]:
                    for trial in range(512):
                        nxt=int(rng.integers(WORD_CONTEXT,len(traincor.seq)-args.seq_len))
                        if traincor.doc_start[nxt]==traincor.doc_start[nxt+args.seq_len-1]:starts[j]=nxt;break
                    else:raise RuntimeError('no valid same-document stream')
        history=torch.as_tensor(traincor.get(starts,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=device)
        state=model.initial(history)
        objectives=[];auxloss=[];train_metrics=[]
        for t in range(args.seq_len):
            p=starts+t
            _,ctx,pieces,lens,target,tp,tl=batch_data(base,traincor,comp,p)
            a=torch.as_tensor(traincor.data[p].copy(),device=device)
            l=torch.as_tensor(traincor.long[p].copy(),device=device)
            ctx,pieces,lens,target,tp,tl=(x.to(device) for x in (ctx,pieces,lens,target,tp,tl))
            now,nll,metrics=model(a,l,ctx,pieces,lens,state,history,target)
            objectives.append(nll.mean());train_metrics.append(metrics)
            if args.subword_weight:auxloss.append(args.subword_weight*model.adapter.subword_loss(now,tp,tl))
            if args.future_weight:
                fy,valid=future_labels(traincor,p)
                auxloss.append(args.future_weight*model.r74.core.future_loss(now,fy.to(device),valid.to(device)))
            if args.align_weight:
                known=ctx[:,-LOCAL:]>1
                composed=model.adapter.compose(pieces,lens)
                tgt=F.embedding(ctx[:,-LOCAL:],model.trunk.E)
                if bool(known.any()):auxloss.append(args.align_weight*F.mse_loss(composed[known],tgt[known]))
            if args.stream_weight:
                st,sl=stream_targets(traincor,comp,base.vocab,p,words=args.stream_words)
                auxloss.append(args.stream_weight*model.adapter.stream_loss(now,st.to(device),sl.to(device)))
            if t+1<args.seq_len:
                leaving=ctx[:,0:1].long()
                state=model.advance(state,leaving[:,0])
                history=torch.cat((history[:,1:],leaving),dim=1)
        word=torch.stack(objectives).mean()
        loss=word+(torch.stack(auxloss).sum()/args.seq_len if auxloss else word*0)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        for o in opts:o.zero_grad(set_to_none=True)
        loss.backward()
        if model.trunk.P.grad is None or float(model.trunk.P.grad.abs().sum())==0:raise AssertionError('trunk gradient missing')
        if model.retrieval.gate_cap.grad is None:raise AssertionError('retrieval gradient missing')
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
        for o in opts:o.step()
        last={'joint_step':int(current+update+1),'word_nll':float(word.detach()),'loss':float(loss.detach()),
              'retrieval_cap':float(torch.sigmoid(model.retrieval.gate_cap).detach()),
              'retrieval_gate_grad':float(model.retrieval.gate_cap.grad.detach()),
              'key_coverage':float(torch.stack([m['key_coverage'] for m in train_metrics]).mean())}
        if (update+1)%args.log_every==0 or update+1==args.steps:
            print(json.dumps({'phase':'TRAIN',**last,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    if args.eval:
        for name,cor in (('dev',devcor),('test',testcor)):
            reports['after_'+name]=evaluate(model,base,comp,cor,args.eval,args.eval_batch)
    reports.update({'end_joint_step':int(current+args.steps),'last_training_step':last if args.steps else {},
        'historical_splits_only':True,'new_documents_evaluated':False,'device':str(device),
        'optimizer_names':OPT_NAMES,'single_checkpoint':str(args.output),
        'wall_time_s':round(time.monotonic()-start,1)})
    reports['checkpoint_sha256']=save(args.output,model,opts,prior,args.root,current+args.steps,rng,reports)
    out=args.root/'reports/R7p5_associative.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps({'phase':'DONE',**reports}),flush=True)
    return reports

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    ap.add_argument('--resume',type=pathlib.Path)
    ap.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    ap.add_argument('--steps',type=int,default=8);ap.add_argument('--streams',type=int,default=2)
    ap.add_argument('--seq-len',type=int,default=2);ap.add_argument('--eval',type=int,default=128)
    ap.add_argument('--eval-batch',type=int,default=8);ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--trunk-lr',type=float,default=2e-6);ap.add_argument('--adapter-lr',type=float,default=6e-5)
    ap.add_argument('--mirror-lr',type=float,default=1e-4);ap.add_argument('--outer-lr',type=float,default=8e-5)
    ap.add_argument('--retrieval-lr',type=float,default=.002);ap.add_argument('--clip',type=float,default=1.0)
    ap.add_argument('--subword-weight',type=float,default=.08);ap.add_argument('--future-weight',type=float,default=.03)
    ap.add_argument('--align-weight',type=float,default=.1);ap.add_argument('--stream-weight',type=float,default=.2)
    ap.add_argument('--stream-words',type=int,default=6);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--log-every',type=int,default=5)
    train(ap.parse_args())
