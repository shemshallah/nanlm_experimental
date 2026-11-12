#!/usr/bin/env python3
"""R7.8 stage B: causal adjacent-token key/value events in SAME mirror matrix.

A stronger order writer than position-weighted means: key(history[t]) ->
value(history[t+1]) is folded as an asymmetric outer product, using the very
same learned key/value projections and the very same rank-64 physical state.
The new additive write gate is exactly zero when migrating the step-14640 state.
All pre-existing model/optimizer tensors and RNG states are retained.
"""
from __future__ import annotations
import argparse,pathlib,json,time,random,os
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from r78_ordered import restore as restore78,OrderedCrossWindowMirror,upgrade as upgrade78,FORMAT as V1, LRS
from r77_semantic import build_model as build77
from r74_stream import OUTER_CONTEXT,OLDER
from r75_retrieval import OPT_NAMES
from r7_memory import WORD_CONTEXT,LOCAL
from r7_core import Corpus
from train_r7 import batch_data,future_labels,stream_targets,sha

FORMAT='PAL_R7_8_CAUSAL_EVENT_PAIR_V2'

class EventPairMirror(OrderedCrossWindowMirror):
    def initial(self,history,embeddings):
        state=super().initial(history,embeddings)
        # This is a causal adjacency sketch; NOT a separate memory bank.
        # The operator stores directed edge key(t) -> value(t+1) in same S+/S-
        # matrix, supporting asymmetric lookup by transposition.
        a=history[:,:-1].long();b=history[:,1:].long()
        mask=((a>1)&(b>1)).to(embeddings.dtype)
        if not bool(mask.any()):return state
        keys=F.normalize(self.key(F.embedding(a,embeddings)),dim=-1)
        values=torch.tanh(self.value(F.embedding(b,embeddings)))
        age=torch.linspace(-2.0,0.0,mask.size(1),device=history.device,dtype=embeddings.dtype).exp()
        weight=mask*age[None,:]
        pair=torch.bmm(values.transpose(1,2),keys*weight.unsqueeze(-1))/weight.sum(1)[:,None,None].clamp_min(1)
        return state+2.0*torch.tanh(self.pair_gain)*pair

def upgrade_pair(model,opts):
    old=model.r74.outer
    assert type(old) is OrderedCrossWindowMirror
    old.__class__=EventPairMirror
    old.pair_gain=nn.Parameter(old.order_gain.detach().new_zeros(()))
    opts[3].add_param_group({'params':[old.pair_gain],'lr':.004,'weight_decay':0.})
    return model,opts

def restore(root,checkpoint,device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['format']==V1:
        model,opts,base,comp,prior=restore78(root,checkpoint,device)
        upgrade_pair(model,opts)
        torch.set_rng_state(prior['torch_rng']);np.random.set_state(prior['numpy_rng']);random.setstate(prior['python_rng'])
        return model,opts,base,comp,prior
    if ck['format']!=FORMAT:raise ValueError('Unsupported source checkpoint '+ck['format'])
    if sha(pathlib.Path(root)/'tokenizer/gutenberg_r7.model')!=ck['tokenizer_sha256'] or sha(pathlib.Path(root)/'corpus/train_temporal.i32')!=ck['train_split_sha256']:
        raise ValueError('incompatible training corpus/tokenizer')
    model,opts,base,comp=build77(root,torch.device(device),LRS)
    upgrade78(model,opts);upgrade_pair(model,opts)
    model.load_state_dict(ck['model'],strict=True)
    for name,opt,lr in zip(OPT_NAMES,opts,LRS):
        opt.load_state_dict(ck['optimizers'][name])
        for k,g in enumerate(opt.param_groups):g['lr']=(.004 if name=='outer_adamw' and k==2 else lr)
    torch.set_rng_state(ck['torch_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    return model,opts,base,comp,ck

def recall_objective(model,state,history,nneg=24,first=None):
    """Causal supervision from real *past* directed transitions, no current targets.

    Exact same selected query, positive and negatives for correct and wrong states.
    The loss explicitly measures whether the current model knows the right old
    history instead of merely getting better irrespective of historical state.
    """
    from r77_outer_aux import choose_recall
    q,y,neg,active,mask=choose_recall(history,nneg=nneg)
    E=model.trunk.E.detach()
    qemb=F.embedding(q.long(),E)
    cand=torch.cat((y[:,None],neg),dim=1).long()
    target=F.normalize(F.embedding(cand,E),dim=-1)
    def score(st):
        read=F.normalize(model.r74.outer.read_rails(st,qemb),dim=-1)
        logits=torch.einsum('bd,bkd->bk',read,target)/.15
        logits[:,1:]=logits[:,1:].masked_fill(~mask,-1e4)
        return F.cross_entropy(logits[active],torch.zeros(int(active.sum()),device=st.device,dtype=torch.long))
    if not bool(active.any()):return state.sum()*0,state.sum()*0
    return score(state),score(state.roll(1,0))

@torch.no_grad()
def evaluate(model,base,comp,corpus,count=128,batch=8,offset=0):
    model.eval();dev=next(model.parameters()).device
    positions=offset+np.arange(count,dtype=np.int64)*(len(corpus.seq)-offset-1)//count
    totals={k:0. for k in ('correct','pair_off','both_order_off','outer_off','reversed','unrelated')}
    for ii in np.array_split(positions,int(np.ceil(len(positions)/batch))):
        _,ctx,pieces,lens,y,_,_=batch_data(base,corpus,comp,ii)
        h=torch.as_tensor(corpus.get(ii,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=dev)
        ex=torch.as_tensor(corpus.data[ii].copy(),device=dev)
        long=torch.as_tensor(corpus.long[ii].copy(),device=dev)
        ctx,pieces,lens,y=[v.to(dev) for v in (ctx,pieces,lens,y)]
        st=model.initial(h)
        save_p=model.r74.outer.pair_gain.detach().clone();save_o=model.r74.outer.order_gain.detach().clone()
        model.r74.outer.pair_gain.zero_();oldpair=model.initial(h)
        model.r74.outer.order_gain.zero_();oldboth=model.initial(h)
        model.r74.outer.order_gain.copy_(save_o);model.r74.outer.pair_gain.copy_(save_p)
        shuffled=h.reshape(-1,OLDER//64,64).flip(-1).reshape(-1,OLDER)
        reversed_state=model.initial(shuffled)
        unrelated=st.roll(1,0)
        states={'correct':st,'pair_off':oldpair,'both_order_off':oldboth,
                'outer_off':st,'reversed':reversed_state,'unrelated':unrelated}
        for key,state in states.items():
            _,loss,_=model(ex,long,ctx,pieces,lens,state,h,y,outer_off=key=='outer_off')
            totals[key]+=float(loss.sum())
    vals={k+'_nll':v/count for k,v in totals.items()}
    vals.update(pair_gain=float(model.r74.outer.pair_gain),order_gain=float(model.r74.outer.order_gain),
                pair_benefit=vals['pair_off_nll']-vals['correct_nll'],
                all_order_benefit=vals['both_order_off_nll']-vals['correct_nll'],
                correct_over_reversed=vals['reversed_nll']-vals['correct_nll'],
                correct_over_unrelated=vals['unrelated_nll']-vals['correct_nll'],n=count)
    return vals

def save(out,model,opts,prior,root,step,rng,metrics):
    root=pathlib.Path(root);out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    payload={**{k:prior[k] for k in ('base_legacy_steps','adapter_legacy_steps','legacy_r6_sha256','legacy_adapter_sha256','legacy_aux')},
      'format':FORMAT,'model':model.state_dict(),
      'optimizers':{k:o.state_dict() for k,o in zip(OPT_NAMES,opts)},
      'joint_steps':int(step),'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
      'train_split_sha256':sha(root/'corpus/train_temporal.i32'),
      'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
      'batch_rng_state':rng.bit_generator.state,
      'architecture':'single rank64 mirror + order bin and directed old-adjacent-event writer; R7.7 retrieval unchanged',
      'metrics':metrics}
    tmp=out.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,out)
    return sha(out)

def train(args):
    start=time.monotonic();torch.set_num_threads(args.threads);root=args.root
    model,opts,base,comp,prior=restore(root,args.resume,args.device)
    original=int(prior['joint_steps']);rng=np.random.default_rng();rng.bit_generator.state=prior['batch_rng_state']
    train=Corpus(root,'train')
    stats={'source_step':original,'source_sha256':sha(args.resume),'updates':args.steps,'five_optimizer_histories_preserved':True,
           'source_format':prior['format'],'target_format':FORMAT, 'recall_weight':args.recall_weight,'distinguish_weight':args.distinguish_weight}
    if args.eval:
        stats['before_dev']=evaluate(model,base,comp,Corpus(root,'dev'),args.eval,args.eval_batch)
    for step in range(args.steps):
        model.train();pos=rng.integers(WORD_CONTEXT,len(train.seq)-1,size=args.streams,dtype=np.int64)
        hist=torch.as_tensor(train.get(pos,OUTER_CONTEXT)[0][:,:OLDER].copy())
        state=model.initial(hist)
        _,ctx,pieces,lens,y,tp,tl=batch_data(base,train,comp,pos)
        ex=torch.as_tensor(train.data[pos].copy());long=torch.as_tensor(train.long[pos].copy())
        now,nll,_=model(ex,long,ctx,pieces,lens,state,hist,y)
        extras=[]
        if args.subword_weight:extras.append(args.subword_weight*model.adapter.subword_loss(now,tp,tl))
        if args.future_weight:
            fy,valid=future_labels(train,pos)
            extras.append(args.future_weight*model.r74.core.future_loss(now,fy,valid))
        if args.align_weight:
            known=ctx[:,-LOCAL:]>1;composed=model.adapter.compose(pieces,lens);target=F.embedding(ctx[:,-LOCAL:],model.trunk.E)
            if bool(known.any()):extras.append(args.align_weight*F.mse_loss(composed[known],target[known]))
        if args.stream_weight:
            st,sl=stream_targets(train,comp,base.vocab,pos,words=args.stream_words)
            extras.append(args.stream_weight*model.adapter.stream_loss(now,st,sl))
        loss=nll.mean()+sum(extras,nll.mean()*0)
        if args.recall_weight or args.distinguish_weight:
            recall_correct,recall_wrong=recall_objective(model,state,hist)
            loss=loss+args.recall_weight*recall_correct+args.distinguish_weight*F.softplus(3*(.1+recall_correct-recall_wrong))/3
        for opt in opts:opt.zero_grad(set_to_none=True)
        loss.backward()
        assert model.r74.outer.pair_gain.grad is not None
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for opt in opts:opt.step()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('loss diverged')
        if (step+1)%args.log_every==0 or step+1==args.steps:
            print(json.dumps({'phase':'TRAIN','step':original+step+1,'loss':float(loss.detach()),
             'nll':float(nll.mean().detach()),'pair_gate':float(model.r74.outer.pair_gain.detach()),
             'pair_grad':float(model.r74.outer.pair_gain.grad.detach()),'seconds':round(time.monotonic()-start,2)}),flush=True)
    stats['final_step']=original+args.steps
    if args.eval:
        stats['after_dev']=evaluate(model,base,comp,Corpus(root,'dev'),args.eval,args.eval_batch)
        stats['after_test']=evaluate(model,base,comp,Corpus(root,'test'),args.eval,args.eval_batch)
    stats['independent_books_evaluated']=False
    stats['wall_seconds']=round(time.monotonic()-start,2)
    stats['checkpoint_sha256']=save(args.output,model,opts,prior,root,stats['final_step'],rng,stats)
    if args.report:pathlib.Path(args.report).write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps({'phase':'SAVED','step':stats['final_step'],'sha256':stats['checkpoint_sha256']}),flush=True)
    return stats

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
 p.add_argument('--resume',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_r78_candidate.pt'))
 p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_r78_event_candidate.pt'))
 p.add_argument('--report',type=pathlib.Path)
 p.add_argument('--steps',type=int,default=64)
 p.add_argument('--streams',type=int,default=2)
 p.add_argument('--eval',type=int,default=0)
 p.add_argument('--eval-batch',type=int,default=8)
 p.add_argument('--threads',type=int,default=2)
 p.add_argument('--device',default='cpu')
 p.add_argument('--subword-weight',type=float,default=.08)
 p.add_argument('--future-weight',type=float,default=.03)
 p.add_argument('--align-weight',type=float,default=.1)
 p.add_argument('--stream-weight',type=float,default=.2)
 p.add_argument('--stream-words',type=int,default=6)
 p.add_argument('--log-every',type=int,default=16)
 p.add_argument('--recall-weight',type=float,default=0.)
 p.add_argument('--distinguish-weight',type=float,default=0.)
 train(p.parse_args())
