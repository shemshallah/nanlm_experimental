#!/usr/bin/env python3
"""FA2 R29: causally replay 33K/65K/132K prefixes with R27 and R28.

Both variants process ALL previous actual subword tokens through their own
continuously carried recurrent states and causal local-attention histories.
Only scored windows invoke R24..R27 external memory. No training or model
selection occurs in this probe. Supports optional frozen R29 adapter.
"""
import argparse, json, math, pathlib, time
import numpy as np
import torch
from torch.nn import functional as F
from r27_restore_unified import restore as restore_r27
from r28_restore_unified import restore as restore_r28
from r28_recurrent import MirrorRecurrentResidual, bind_recurrent
from r23_stream_carry import ContinuousPALState,contiguous_book_picks
from r13_corpus_cache import read_books_cached


def warmed_piece(model, book, seg, carry, hidden, tags, targets):
    """Exact state/KV-prefix replay without computing unused historical logits.

    In the inherited PAL recurrence the attention read and FFN affect only
    the current output, never the next recurrent state or the stored K/V.
    Thus the past query, top-k, FFN and decoder can safely be skipped for
    segments whose predictions are not scored. Never skip state/KV writes.
    """
    begin,end,t0,t1=seg
    if carry.book_id!=book.book_id or carry.cursor!=t0:
        raise ValueError('noncontiguous replay or cross-book state')
    m=model.model.memory
    if carry.previous is None:
        carry.previous=torch.full((len(targets),),m.start_piece,dtype=torch.long,device=targets.device)
        carry.state=hidden[:,0]
        carry.last_tag=tags[:,0]+begin
    retain=torch.sigmoid(model.bridge.retain_logit)
    for t in range(targets.shape[1]):
        tag=tags[:,t];glob=tag+begin
        nxt=hidden.gather(1,tag[:,None,None].expand(-1,1,hidden.shape[-1]))[:,0]
        changed=glob!=carry.last_tag
        carry.state=torch.where(changed[:,None],retain*carry.state+
             (1.-retain)*model.bridge.context_gain*nxt,carry.state)
        carry.state=m.subword_cell(m.piece_emb(carry.previous),carry.state)
        refiner=getattr(carry,'recurrent_refiner',None)
        if refiner is not None:carry.state=refiner(carry.state)
        adapter=model.adapter
        if adapter is not None:
            carry.key_history.append(adapter.key(carry.state))
            carry.value_history.append(adapter.value(carry.state))
            if len(carry.key_history)>adapter.capacity:
                del carry.key_history[0];del carry.value_history[0]
        carry.previous=targets[:,t]
        carry.last_tag=glob
    carry.cursor+=targets.shape[1]
    carry.detach()


def pair_replay(book, models, targets, span, max_targets=None):
    if not targets or targets!=sorted(set(targets)):
        raise ValueError('targets must be ordered unique')
    carries=[bind_recurrent(ContinuousPALState(book.book_id),m) for m in models]
    result=[];chosen=0;start=time.monotonic()
    with torch.no_grad():
        for i,seg in enumerate(contiguous_book_picks(book,span)):
            begin,end,t0,t1=seg
            if chosen>=len(targets):break
            # Hidden input is exactly shared because all inherited components
            # except the R28 residual are bitwise identical.
            e=torch.as_tensor(book.examples[begin:end].astype(np.int64))[None]
            lexical=torch.as_tensor(book.context[begin:end].astype(np.float32))[None]
            hidden=models[0].model.core.hidden(e.reshape(-1,7),lexical.reshape(-1,72)).reshape(1,end-begin,-1)
            tags=torch.as_tensor(book.token_tag[t0:t1].astype(np.int64))[None]-begin
            toks=torch.as_tensor(book.tokens[t0:t1].astype(np.int64))[None]
            if t0>=targets[chosen]:
                measured=[]
                for model,carry in zip(models,carries):
                    row=model.forward_chunk(book,seg,carry)
                    y=row['target'];not_space=y.ne(32)
                    score=F.nll_loss(row['logits'],y,reduction='sum')
                    nonspace=(F.nll_loss(row['logits'][not_space],y[not_space],reduction='mean') if bool(not_space.any()) else torch.tensor(float('nan')))
                    measured.append({'bpb':float(score)/(row['bytes']*math.log(2)),
                        'nonspace_nll':float(nonspace),'bytes':int(row['bytes']),
                        'tokens':len(y),'nonspace_tokens':int(not_space.sum()),'state_cursor':carry.cursor})
                    carry.detach()
                record={'book_id':str(book.book_id),'requested_offset':targets[chosen],
                    'actual_offset':t0,'comparisons':measured,'bpb_gain':measured[0]['bpb']-measured[1]['bpb'],
                    'elapsed_s':round(time.monotonic()-start,2)}
                result.append(record)
                print('R29_LONG_REPLAY '+json.dumps(record),flush=True)
                chosen+=1
            else:
                for m,c in zip(models,carries):warmed_piece(m,book,seg,c,hidden,tags,toks)
            if i%250==0 and i>0:print('R29_REPLAY_PROGRESS '+json.dumps({'book':str(book.book_id),
                    'token_offset':t0,'target_offset':targets[chosen] if chosen<len(targets) else None,
                    'elapsed_s':round(time.monotonic()-start,2)}),flush=True)
    if not result:raise ValueError('book too short for requested probe')
    return result


def run(args):
    root=pathlib.Path(args.root).resolve();torch.set_num_threads(args.threads)
    r27,_,comp,r6,_,base=restore_r27(root,threads=args.threads)
    r27.add_module('r28',MirrorRecurrentResidual(dim=r27.model.memory.subword_cell.hidden_size))
    r28,_,_,_,_,_,_=restore_r28(root,threads=args.threads)
    if args.overlay:
        delta=torch.load(root/args.overlay,weights_only=False,map_location='cpu')
        if delta['format']!='FA2_R29_R28_CONTINUATION_DELTA_V1':raise ValueError('R29 overlay format')
        r28.r28.load_state_dict(delta['r28_state'])
    for m in (r27,r28):m.eval()
    all_books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,args.max_books)
    group=all_books[args.split]
    b=next((x for x in group if str(x.book_id)==str(args.book)),None)
    if b is None:raise ValueError('selected book absent from corpus cache')
    before=[k for k,v in r27.state_dict().items() if not k.startswith('r28.')
            and not torch.equal(v,r28.state_dict()[k])]
    if before:raise ValueError('inherited R27 weights differ: '+str(before[:3]))
    targets=sorted({int(v) for v in args.targets.split(',')})
    rows=pair_replay(b,(r27,r28),targets,args.warm_span)
    report={'format':'FA2_R29_CAUSAL_LONG_PREFIX_AUDIT_V1','status':'historically_consulted_not_blind',
        'source':'full causal document replay, shared inherited PAL core, distinct R28 residual',
        'book':b.book_id,'split':args.split,'targets':targets,'overlay':args.overlay,
        'windows':rows,'all_scored_windows_r28_win':all(x['bpb_gain']>0 for x in rows),
        'no_training_or_model_selection':True}
    dest=root/'reports'/f'r29_long_{args.split}_{args.book}.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print('R29_LONG_FINAL '+json.dumps({'book':b.book_id,'split':args.split,'windows':len(rows),
        'wins':sum(x['bpb_gain']>0 for x in rows)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--split',choices=['train','dev','test'],default='dev')
    p.add_argument('--book',default='1444');p.add_argument('--targets',default='33000,65000,132000')
    p.add_argument('--warm-span',type=int,default=96);p.add_argument('--max-books',type=int,default=3)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--overlay',default='')
    run(p.parse_args())
