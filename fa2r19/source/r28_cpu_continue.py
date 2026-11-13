#!/usr/bin/env python3
"""FA2 R28 true 4x12-word-chunk CPU BPTT from immutable unified R27.

Trains only a new zero-initialized recurrent residual THROUGH the existing
recurrent cell; never recreates a detached independent language model.
Experimental selection uses real-book development windows, never test books.
The selected R28 checkpoint is NOT accepted without 132K replay and fresh-author
validation; R27 remains the immutable recovery head.
"""
import argparse, copy, hashlib, json, math, os, pathlib, random, time
import numpy as np
import torch
from torch.nn import functional as F
from r27_restore_unified import restore
from r13_corpus_cache import read_books_cached
from r23_stream_carry import ContinuousPALState, contiguous_book_picks
from r28_recurrent import MirrorRecurrentResidual, bind_recurrent


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def metrics(model,books,chunks,span):
    model.eval(); total=other=0.; bytes_=tokens=other_n=0; perbook={}
    with torch.no_grad():
        for book in books:
            carry=bind_recurrent(ContinuousPALState(book.book_id),model)
            plans=list(contiguous_book_picks(book,span))[:chunks]
            book_nll=0.; book_bytes=0
            for seg in plans:
                row=model.forward_chunk(book,seg,carry)
                y=row['target']; nll=F.cross_entropy(row['logits'],y,reduction='sum')
                bad=y.ne(32)
                total+=float(nll); book_nll+=float(nll)
                other+=float(F.cross_entropy(row['logits'][bad],y[bad],reduction='sum')) if bool(bad.any()) else 0.
                other_n+=int(bad.sum()); tokens+=len(y); bytes_+=row['bytes'];book_bytes+=row['bytes']
            perbook[str(book.book_id)]=book_nll/(book_bytes*math.log(2))
    return {'bpb':total/(bytes_*math.log(2)),'nonspace_nll':other/max(1,other_n),
            'bytes':bytes_,'tokens':tokens,'books':perbook}


def equal_nested(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal_nested(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
    return a==b


def run(args):
    root=pathlib.Path(args.root).resolve();torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    ancestor=root/'reports/fa2_r27_unified_candidate.pth';r27_digest=sha(ancestor)
    model,opt,comp,r6,meta,inherited=restore(root,threads=args.threads)
    inherited_models={k:v.detach().clone() for k,v in model.state_dict().items()}
    inherited_optimizer=copy.deepcopy(opt.state_dict())
    original_carry=copy.deepcopy(inherited['selected']['carry'])
    model.add_module('r28',MirrorRecurrentResidual(dim=model.model.memory.subword_cell.hidden_size,rank=args.rank))
    opt.add_param_group({'params':list(model.r28.parameters()),'lr':args.lr,'weight_decay':1e-5})
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('r28.'))
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,args.max_books)
    training=books['train'][:args.train_books]; dev=books['dev'][:args.dev_books]
    assert training and dev and not {b.book_id for b in training}&{b.book_id for b in dev}
    # Correct initial parity is stronger than approximate BPB parity: old model
    # and zero-initialized R28 must produce identical per-token logits.
    probe=dev[0]; segment=next(contiguous_book_picks(probe,args.chunk_span))
    model.eval()
    with torch.no_grad():
        old=model.forward_chunk(probe,segment,ContinuousPALState(probe.book_id))['logits']
        new=model.forward_chunk(probe,segment,bind_recurrent(ContinuousPALState(probe.book_id),model))['logits']
        assert torch.equal(old,new),'inherited R27 logit parity failed'
    baseline=metrics(model,dev,args.dev_chunks,args.chunk_span)
    initial_r28=copy.deepcopy(model.r28.state_dict()); initial_opt=copy.deepcopy(opt.state_dict())
    best={'step':0,'metrics':baseline,'state':initial_r28,'optimizer':initial_opt}
    trace=[]; start=time.monotonic()
    positions=[0]*len(training); carriers=[bind_recurrent(ContinuousPALState(b.book_id),model) for b in training]
    plans=[list(contiguous_book_picks(b,args.chunk_span)) for b in training]
    for step in range(1,args.steps+1):
        which=(step-1)%len(training);book=training[which];carry=carriers[which]
        if positions[which]+args.unroll>len(plans[which]):
            positions[which]=0;carry.reset(book.book_id)
        losses=[];n=0;model.train()
        for j in range(args.unroll):
            segment=plans[which][positions[which]]
            row=model.forward_chunk(book,segment,carry)
            count=len(row['target']);n+=count
            losses.append(F.cross_entropy(row['logits'],row['target'],reduction='sum'))
            positions[which]+=1
        loss=torch.stack(losses).sum()/n
        opt.zero_grad(set_to_none=True);loss.backward()
        gn=float(torch.nn.utils.clip_grad_norm_(model.r28.parameters(),args.clip))
        if not math.isfinite(gn) or not torch.isfinite(loss):raise ValueError('R28 nonfinite gradient')
        opt.step();carry.detach()
        event={'step':step,'tokens':n,'train_nll':float(loss.detach()),'gradient_norm':gn}
        if step%args.eval_every==0 or step==args.steps:
            current=metrics(model,dev,args.dev_chunks,args.chunk_span)
            event['dev']=current
            # Dev-only selection. Never promote to accepted until unseen-author
            # and 132K replay guards are available.
            if current['bpb']<best['metrics']['bpb']-args.min_gain and current['nonspace_nll']<=baseline['nonspace_nll']+args.nonspace_slack:
                best={'step':step,'metrics':current,'state':copy.deepcopy(model.r28.state_dict()),'optimizer':copy.deepcopy(opt.state_dict())}
        trace.append(event)
        print('R28_UPDATE '+json.dumps(event),flush=True)
    model.r28.load_state_dict(best['state']);opt.load_state_dict(best['optimizer'])
    if not equal_nested({k:v for k,v in model.state_dict().items() if not k.startswith('r28.')},inherited_models):
        raise AssertionError('R27 weights were modified')
    state=opt.state_dict()
    if state['param_groups'][:12]!=inherited_optimizer['param_groups'] or not equal_nested({k:state['state'][k] for k in inherited_optimizer['state']},inherited_optimizer['state']):
        raise AssertionError('inherited R27 Adam moments changed')
    if not equal_nested(inherited['selected']['carry'],original_carry):raise AssertionError('inherited document streams changed')
    if sha(ancestor)!=r27_digest:raise AssertionError('R27 original checkpoint changed')
    summary={'format':'FA2_R28_CPU_LONG_BPTT_V1','status':'candidate_not_promoted',
       'r27_sha256':r27_digest,'r28_selected_step':best['step'], 'attempted_steps':args.steps,
       'bptt_chunks':args.unroll,'chunk_span_words':args.chunk_span,
       'selection_dev':best['metrics'],'initial_dev':baseline,'trace':trace,
       'inherited_tensor_count':len(inherited_models),'inherited_adam_states':len(inherited_optimizer['state']),
       'inherited_optimizer_groups':12,'inherited_stream_snapshots':len(original_carry),
       'needs_132k_replay':True,'needs_unseen_author':True,'args':vars(args),'elapsed_s':time.monotonic()-start}
    (root/'reports/r28_cpu_longbptt.json').write_text(json.dumps(summary,indent=2)+'\n')
    if best['step']>0:
        payload={'format':'FA2_R28_UNIFIED_CANDIDATE_V1','parent_r27_sha256':r27_digest,
          'accepted_model_promoted':False,'selected':{'r28_step':best['step'],
           'model':model.state_dict(),'optimizer':state,
           'carry':original_carry},'r27_inheritance':{'tensors':len(inherited_models),
           'Adam_states':len(inherited_optimizer['state'])},'selection_dev':best['metrics']}
        path=root/'reports/fa2_r28_unified_candidate.pth';temp=path.with_suffix('.tmp.pth')
        torch.save(payload,temp);os.replace(temp,path)
        summary['candidate_checkpoint_sha256']=sha(path)
        (root/'reports/r28_cpu_longbptt.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('R28_FINAL '+json.dumps({k:summary[k] for k in ['status','r28_selected_step','selection_dev','initial_dev','inherited_adam_states']}),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--steps',type=int,default=6);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--train-books',type=int,default=2);p.add_argument('--dev-books',type=int,default=2)
    p.add_argument('--max-books',type=int,default=2)
    p.add_argument('--chunk-span',type=int,default=12);p.add_argument('--unroll',type=int,default=4)
    p.add_argument('--dev-chunks',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=2);p.add_argument('--seed',type=int,default=28028)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--rank',type=int,default=64)
    p.add_argument('--clip',type=float,default=2.);p.add_argument('--min-gain',type=float,default=1e-6)
    p.add_argument('--nonspace-slack',type=float,default=1e-4)
    run(p.parse_args())
