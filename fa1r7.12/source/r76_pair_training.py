#!/usr/bin/env python3
"""PAL R7.6 — one continuous five-optimizer model with causal two-word associations.

Migrate the R7.5 checkpoint *without* modifying any pre-existing parameter,
optimizer state, or RNG state. The fifth optimizer receives a new pair parameter
*group*, not a sixth optimizer. Both one- and two-word keys retrieve from the
same fixed 6,144-token older causal memory; the full R7.4 model remains trainable.
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT, OLDER
from r75_retrieval import PALR75, restore as restore75, evaluate as evaluate75, OPT_NAMES
from r75_retrieval import CHECKPOINT_FORMAT as OLD_FORMAT
from train_r7 import batch_data, future_labels, stream_targets, sha
from r7_memory import WORD_CONTEXT, LOCAL

FORMAT = 'PAL_R7_6_PAIR_ASSOCIATIVE_V1'

class PairHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Smooth positive gate: unlike a zero-initialized hard clamp,
        # this continues to learn when early gradients reduce pair use.
        # The original R7.5 prediction is exactly recovered by pair_off=True.
        self.strength = nn.Parameter(torch.tensor(-3.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

class PALR76(PALR75):
    def __init__(self, r74):
        super().__init__(r74)
        self.retrieval.pair = PairHead()

    def pair_lookup(self, history, ctx, target=None):
        if history.ndim != 2 or history.size(1) != OLDER:
            raise ValueError('wrong causal older-memory shape')
        if ctx.ndim != 2 or ctx.size(1) < 2:
            raise ValueError('requires ordered previous two words')
        left, right = ctx[:,-2].long(), ctx[:,-1].long()
        h0, h1, hv = history[:,:-2], history[:,1:-1], history[:,2:]
        valid = (h0>1)&(h1>1)&(hv>1)&(left[:,None]>1)&(right[:,None]>1)
        match = valid&(h0==left[:,None])&(h1==right[:,None])
        ages = torch.arange(OLDER-2,device=history.device,dtype=torch.float32)
        ages = (ages-(OLDER-3))/(OLDER-2)
        scores = torch.exp(F.softplus(self.retrieval.recency)*ages).unsqueeze(0)
        weights=match.float()*scores
        total=weights.sum(-1)
        nmatch=match.sum(-1)
        newest=torch.where(match,torch.arange(OLDER-2,device=history.device).unsqueeze(0),-1).max(-1).values
        feats=torch.stack((torch.log1p(nmatch.float()),torch.log1p(total),(newest.float()+1)/OLDER),dim=-1)
        prob = None if target is None else ((weights*(hv==target[:,None])).sum(-1)/total.clamp_min(1e-20))
        return prob,(nmatch>0),feats

    def pair_gate(self, now, ctx, features, exists):
        right=ctx[:,-1].long()
        emb=F.embedding(right,self.trunk.E)
        logit=self.retrieval.gate_net(torch.cat((now,emb,features.to(now.dtype)),-1)).squeeze(-1)
        # The only new parameters are in the original retrieval optimizer.
        cap=0.20*torch.sigmoid(self.retrieval.pair.strength)
        return torch.where(exists,cap*torch.sigmoid(logit+self.retrieval.pair.bias),torch.zeros_like(logit))

    def forward(self,ex,long,ctx,pieces,lens,state=None,history=None,target=None,
                retrieval_off=False,pair_off=False,outer_off=False,mirror_off=False):
        now,nll,metrics=super().forward(ex,long,ctx,pieces,lens,state=state,history=history,
                target=target,retrieval_off=retrieval_off,outer_off=outer_off,mirror_off=mirror_off)
        metrics=dict(metrics)
        if retrieval_off or pair_off or history is None:
            metrics.update(pair_gate_mean=now.new_zeros(()),pair_coverage=now.new_zeros(()),pair_hits=now.new_zeros(()))
            return now,nll,metrics
        prob,exists,features=self.pair_lookup(history,ctx,target)
        gate=self.pair_gate(now,ctx,features,exists)
        metrics.update(pair_gate_mean=gate.mean().detach(), pair_coverage=exists.float().mean().detach(),
                       pair_hits=((prob>0)&exists).float().mean().detach() if prob is not None else now.new_zeros(()))
        if target is not None:
            nll=self.retrieval.mixed_nll(nll,prob,gate)
        return now,nll,metrics


def restore(root,checkpoint,device,trunk_lr=2e-6,adapter_lr=6e-5,mirror_lr=1e-4,
            outer_lr=8e-5,retrieval_lr=0.002):
    data=torch.load(checkpoint,map_location='cpu',weights_only=False)
    lrs=(trunk_lr,adapter_lr,mirror_lr,outer_lr,retrieval_lr)
    if data['format']==OLD_FORMAT:
        original,opts,base,comp,old,prior=restore75(root,checkpoint,device,*lrs)
        model=PALR76(original.r74).to(device)
        # Rebind the EXACT existing retrieval module referenced by fifth optimizer;
        # otherwise the optimizer would silently update orphaned tensors.
        model.retrieval=original.retrieval
        model.retrieval.pair=PairHead().to(device)
        torch.set_rng_state(prior['torch_rng'])
        np.random.set_state(prior['numpy_rng']);random.setstate(prior['python_rng'])
        # Exactly four old optimizer states plus original retrieval state preserved.
        opts[4].add_param_group({'params':list(model.retrieval.pair.parameters()),'lr':retrieval_lr,'weight_decay':0.0})
        return model,opts,base,comp,old,prior
    if data['format']!=FORMAT:
        raise ValueError(f'Unexpected format {data["format"]}')
    from r7_core import R6
    from r7_memory import TokenCompositor
    from r73_joint import UnifiedPAL
    from r74_stream import PALR74,build_optimizers
    root=pathlib.Path(root)
    if sha(root/'tokenizer/gutenberg_r7.model')!=data['tokenizer_sha256'] or sha(root/'corpus/train_temporal.i32')!=data['train_split_sha256']:
        raise ValueError('Tokenizer or train corpus changed')
    base=R6(root);comp=TokenCompositor(root,base)
    model=PALR76(PALR74(UnifiedPAL(base,comp,root))).to(device)
    opts=(*build_optimizers(model.r74,*lrs[:4]),
          torch.optim.AdamW([{'params':list(model.retrieval.parameters())[:-2]},
                              {'params':list(model.retrieval.pair.parameters()),'weight_decay':0.0}],
                              lr=retrieval_lr,weight_decay=0.0005))
    model.load_state_dict(data['model'],strict=True)
    for name,opt,lr in zip(OPT_NAMES,opts,lrs):
        opt.load_state_dict(data['optimizers'][name])
        for group in opt.param_groups:group['lr']=lr
    torch.set_rng_state(data['torch_rng']);np.random.set_state(data['numpy_rng']);random.setstate(data['python_rng'])
    old={'base_step':data['base_legacy_steps'],'adapter_updates':data['adapter_legacy_steps'],
         'base_sha256':data['legacy_r6_sha256'],'legacy_adapter_sha256':data['legacy_adapter_sha256'],
         'legacy_aux':data.get('legacy_aux',{})}
    return model,opts,base,comp,old,data

@torch.no_grad()
def evaluate(model,base,comp,corpus,count=512,batch=8):
    model.eval();device=next(model.parameters()).device
    positions=np.arange(count,dtype=np.int64)*(len(corpus.seq)-1)//count
    keys=['active','pair_off','retrieval_off','outer_off']
    stats={k:0. for k in keys}
    stats.update(n=count, key_positions=0,target_hits=0,pair_positions=0,pair_target_hits=0)
    for index in np.array_split(positions,int(np.ceil(count/batch))):
        _,ctx,pc,ln,target,_,_=batch_data(base,corpus,comp,index)
        history=torch.as_tensor(corpus.get(index,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=device)
        state=model.initial(history)
        a=torch.as_tensor(corpus.data[index].copy(),device=device)
        l=torch.as_tensor(corpus.long[index].copy(),device=device)
        ctx,pc,ln,target=(z.to(device) for z in (ctx,pc,ln,target))
        now,nll,mt=model(a,l,ctx,pc,ln,state,history,target)
        _,pair_loss,_=model(a,l,ctx,pc,ln,state,history,target,pair_off=True)
        _,outer_loss,_=model(a,l,ctx,pc,ln,state,history,target,outer_off=True)
        stats['active']+=float(nll.sum())
        stats['pair_off']+=float(pair_loss.sum())
        stats['retrieval_off']+=float(model.trunk.nll(now,target).sum())
        stats['outer_off']+=float(outer_loss.sum())
        p,has,_,_=model.retrieval.lookup(history,ctx[:,-1],target)
        q,hasq,_=model.pair_lookup(history,ctx,target)
        stats['key_positions']+=int(has.sum());stats['target_hits']+=int((p>0).sum())
        stats['pair_positions']+=int(hasq.sum());stats['pair_target_hits']+=int((q>0).sum())
    for k in keys:stats[k+'_nll']=stats.pop(k)/count
    stats['one_word_gain']=stats['retrieval_off_nll']-stats['pair_off_nll']
    stats['incremental_pair_gain']=stats['pair_off_nll']-stats['active_nll']
    stats['total_retrieval_gain']=stats['retrieval_off_nll']-stats['active_nll']
    return stats


def save(out,model,opts,prior,root,updates,rng,metrics):
    out=pathlib.Path(out);root=pathlib.Path(root);out.parent.mkdir(parents=True,exist_ok=True)
    data={'format':FORMAT,'model':model.state_dict(),
        'optimizers':{k:o.state_dict() for k,o in zip(OPT_NAMES,opts)},
        'base_legacy_steps':int(prior['base_legacy_steps']),
        'adapter_legacy_steps':int(prior['adapter_legacy_steps']),
        'joint_steps':int(updates),'legacy_r6_sha256':prior['legacy_r6_sha256'],
        'legacy_adapter_sha256':prior['legacy_adapter_sha256'],
        'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
        'train_split_sha256':sha(root/'corpus/train_temporal.i32'),
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
        'python_rng':random.getstate(),'batch_rng_state':rng.bit_generator.state,
        'legacy_aux':prior.get('legacy_aux',{}),
        'architecture':'R7.5 single PAL + one-word and two-word exact causal older associative retrieval',
        'metrics':metrics}
    tmp=out.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,out)
    return sha(out)


def train(args):
    start=time.monotonic();torch.set_num_threads(args.threads)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    model,opts,base,comp,old,prior=restore(args.root,args.resume or args.root/'checkpoint/pal_active.pt',device,
                        args.trunk_lr,args.adapter_lr,args.mirror_lr,args.outer_lr,args.retrieval_lr)
    prior_state_format=prior['format'];current=int(prior['joint_steps']);rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    traincor=Corpus(args.root,'train');devcor=Corpus(args.root,'dev');testcor=Corpus(args.root,'test')
    stats={'start_joint_step':current,'planned_updates':args.steps,'migration_from':prior_state_format,
           'optimizer_names':OPT_NAMES,'checkpoint':str(args.output),'historical_splits_only':True}
    if args.eval:
        for name,cor in [('dev',devcor),('test',testcor)]: stats['before_'+name]=evaluate(model,base,comp,cor,args.eval,args.eval_batch)
    log_path=pathlib.Path(args.root)/'reports/R7p6_training_metrics.jsonl';log_path.parent.mkdir(exist_ok=True)
    with log_path.open('a') as logfile:
        for i in range(args.steps):
            model.train()
            starts=rng.integers(WORD_CONTEXT,len(traincor.seq)-args.seq_len,size=args.streams,dtype=np.int64)
            if traincor.doc_start is not None:
                for j,s in enumerate(starts):
                    if traincor.doc_start[s]!=traincor.doc_start[s+args.seq_len-1]:
                        for trial in range(512):
                            nxt=int(rng.integers(WORD_CONTEXT,len(traincor.seq)-args.seq_len))
                            if traincor.doc_start[nxt]==traincor.doc_start[nxt+args.seq_len-1]: starts[j]=nxt;break
                        else:raise RuntimeError('no valid same-document stream')
            history=torch.as_tensor(traincor.get(starts,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=device)
            state=model.initial(history);objectives=[];auxloss=[];metrics=[]
            for t in range(args.seq_len):
                p=starts+t
                _,ctx,pieces,lens,target,tp,tl=batch_data(base,traincor,comp,p)
                a=torch.as_tensor(traincor.data[p].copy(),device=device)
                l=torch.as_tensor(traincor.long[p].copy(),device=device)
                ctx,pieces,lens,target,tp,tl=(x.to(device) for x in (ctx,pieces,lens,target,tp,tl))
                now,nll,mt=model(a,l,ctx,pieces,lens,state,history,target)
                objectives.append(nll.mean());metrics.append(mt)
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
                    state=model.advance(state,leaving[:,0]);history=torch.cat((history[:,1:],leaving),dim=1)
            word=torch.stack(objectives).mean()
            loss=word+(torch.stack(auxloss).sum()/args.seq_len if auxloss else word*0)
            if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {current+i}')
            for o in opts:o.zero_grad(set_to_none=True)
            loss.backward()
            if model.trunk.P.grad is None or float(model.trunk.P.grad.abs().sum())==0:raise AssertionError('trunk gradient missing')
            if model.retrieval.gate_cap.grad is None or model.retrieval.pair.strength.grad is None:raise AssertionError('retrieval or pair gradient missing')
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip)
            for o in opts:o.step()
            step=current+i+1
            last={'joint_step':step,'word_nll':float(word.detach()),'loss':float(loss.detach()),
                  'retrieval_cap':float(torch.sigmoid(model.retrieval.gate_cap).detach()),
                  'pair_strength':float((0.2*torch.sigmoid(model.retrieval.pair.strength)).detach()),
                  'pair_gate_grad':float(model.retrieval.pair.strength.grad.detach()),
                  'pair_coverage':float(torch.stack([m['pair_coverage'] for m in metrics]).mean()),
                  'pair_hits':float(torch.stack([m['pair_hits'] for m in metrics]).mean())}
            if step%args.log_every==0 or i+1==args.steps:
                line={'phase':'TRAIN',**last,'elapsed_s':round(time.monotonic()-start,1)}
                print(json.dumps(line),flush=True);logfile.write(json.dumps(line)+'\n');logfile.flush()
            if step%args.checkpoint_every==0 or i+1==args.steps:
                if args.eval and (step%args.eval_every==0 or i+1==args.steps):
                    point={'joint_step':step,'elapsed_s':round(time.monotonic()-start,1),
                           'dev':evaluate(model,base,comp,devcor,args.eval,args.eval_batch),
                           'test':evaluate(model,base,comp,testcor,args.eval,args.eval_batch)}
                    print(json.dumps({'phase':'EVAL',**point}),flush=True)
                    logfile.write(json.dumps({'phase':'EVAL',**point})+'\n');logfile.flush()
                    stats['latest_eval']=point
                stats['completed_updates']=i+1;stats['end_joint_step']=step;stats['last_training_step']=last
                stats['wall_time_s']=round(time.monotonic()-start,1)
                stats['checkpoint_sha256']=save(args.output,model,opts,prior,args.root,step,rng,stats)
                print(json.dumps({'phase':'SAVED','joint_step':step,'checkpoint_sha256':stats['checkpoint_sha256'],'elapsed_s':stats['wall_time_s']}),flush=True)
    output=args.root/'reports/R7p6_10k_results.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps({'phase':'DONE',**stats}),flush=True)
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--resume',type=pathlib.Path)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--streams',type=int,default=2)
    p.add_argument('--seq-len',type=int,default=2);p.add_argument('--eval',type=int,default=512)
    p.add_argument('--eval-batch',type=int,default=8);p.add_argument('--eval-every',type=int,default=2000)
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--trunk-lr',type=float,default=2e-6);p.add_argument('--adapter-lr',type=float,default=6e-5)
    p.add_argument('--mirror-lr',type=float,default=1e-4);p.add_argument('--outer-lr',type=float,default=8e-5)
    p.add_argument('--retrieval-lr',type=float,default=.002);p.add_argument('--clip',type=float,default=1.0)
    p.add_argument('--subword-weight',type=float,default=.08);p.add_argument('--future-weight',type=float,default=.03)
    p.add_argument('--align-weight',type=float,default=.1);p.add_argument('--stream-weight',type=float,default=.2)
    p.add_argument('--stream-words',type=int,default=6);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--log-every',type=int,default=250)
    train(p.parse_args())
