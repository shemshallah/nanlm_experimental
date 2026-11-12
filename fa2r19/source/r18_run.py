#!/usr/bin/env python3
"""Reproducible PAL R18 continuation: preserve R15 optimizer moments and gates.

Run learned key retrieval with hard negatives on inherited NaNROUTER text,
then fit a single copy-evidence gain on frozen R15 logits and gate PROMOTION
on matched short and long book-disjoint dev loss, plus recall. All candidate
calibration and architecture decisions use development, never test split.
"""
import argparse,copy,hashlib,json,math,os,pathlib,random,sys,time
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r11_raw import batch,windows
from r13_corpus_cache import read_books_cached
from r18_long_memory import (R18LongMemory,collect_hard_negatives,batch_examples,
                             test_retrieval,build_copy_queries,apply_copy_boost)

@torch.no_grad()
def frozen_base_logits(model,bridge,adapter,ex,ctx,targets,tags):
    """Identical causal recurrence to R13, retaining logits for one sequence."""
    assert ex.shape[0]==1 and targets.shape[0]==1
    W=ex.shape[1]
    h=model.core.hidden(ex.reshape(-1,7),ctx.reshape(-1,72)).reshape(1,W,-1)
    state=h[:,0];previous=torch.full((1,),model.memory.start_piece,dtype=torch.long)
    last=tags[:,0];retain=bridge.retain_logit.sigmoid();kh=[];vh=[];outs=[]
    for t in range(targets.shape[1]):
        valid=targets[:,t]!=-100
        ix=tags[:,t].clamp(0,W-1)
        nxt=h.gather(1,ix[:,None,None].expand(-1,1,h.shape[-1]))[:,0]
        changed=(ix!=last)&valid
        state=torch.where(changed[:,None],retain*state+(1-retain)*bridge.context_gain*nxt,state)
        state=model.memory.subword_cell(model.memory.piece_emb(previous),state)
        out=state
        if kh:
            past_k=torch.stack(kh[-adapter.capacity:],dim=1)
            past_v=torch.stack(vh[-adapter.capacity:],dim=1)
            scores=(adapter.query(state).unsqueeze(1)*past_k).sum(-1)/math.sqrt(state.shape[-1])
            k=min(adapter.topk,scores.shape[1]);inds=scores.topk(k,dim=1).indices
            selected=past_v.gather(1,inds[:,:,None].expand(-1,-1,state.shape[-1]))
            w=F.softmax(scores.gather(1,inds),dim=-1)
            out=out+adapter.attn_gain*adapter.attn_out((selected*w[:,:,None]).sum(1))
        out=out+adapter.ffn_gain*adapter.ffn(state)
        kh.append(adapter.key(state));vh.append(adapter.value(state))
        outs.append(model.memory.subword_out(out)[0])
        previous=torch.where(valid,targets[:,t],torch.zeros_like(previous))
        last=torch.where(valid,ix,last)
    return torch.stack(outs).detach()

def make_windows(docs,split,span,n,max_tokens):
    # Maintain a deterministic selection separated by whole Gutenberg book.
    w=windows(docs,span,n,max_tokens)[split]
    return w

def prepare_cache(model,bridge,adapter,docs,choices,span):
    cache=[]
    for i,(bi,start) in enumerate(choices):
        ex,lc,target,tag,bytes_=batch(docs,[(bi,start)],span)
        logits=frozen_base_logits(model,bridge,adapter,ex,lc,target,tag)
        t=target[0]
        q=build_copy_queries(t,min(2048,len(t)),65)
        cache.append({'logits':logits,'tokens':t.clone(),'queries':q,'bytes':bytes_,
                      'book':docs[bi].book_id,'chunk':start})
    return cache

def summarize_cache(net,cache):
    total=0.;other=0.;nt=0;no=0;bytes_=0;eligible=0;copyable=0;eligible_loss=0.;
    correct=0;other_correct=0;space_correct=0;space_n=0;by_book=defaultdict(lambda:[0.,0,0])
    with torch.no_grad():
        for row in cache:
            logits=apply_copy_boost(net,row['logits'],row['queries'])
            t=row['tokens'];loss=F.cross_entropy(logits,t,reduction='none')
            other_mask=t!=32
            total+=float(loss.sum());other+=float(loss[other_mask].sum());nt+=len(t);no+=int(other_mask.sum());bytes_+=row['bytes']
            pred=logits.argmax(-1)
            correct+=int((pred==t).sum());other_correct+=int(((pred==t)&other_mask).sum())
            space_n+=int((~other_mask).sum());space_correct+=int(((pred==t)&~other_mask).sum())
            x=by_book[row['book']];x[0]+=float(loss.sum());x[1]+=row['bytes'];x[2]+=len(t)
            if row['queries'] is not None:
                pos=row['queries']['positions'];eligible+=len(pos)
                eligible_loss+=float(loss[pos].sum())
                for i,p in enumerate(pos.tolist()):
                    q=row['queries'];vals=q['values'][i][q['mask'][i]]
                    if bool((vals==t[p]).any()):copyable+=1
    return {'bits_per_byte':total/bytes_/math.log(2),'other_nll':other/max(1,no),
            'other_top1':other_correct/max(1,no),'space_top1':space_correct/max(1,space_n),
            'tokens':nt,'raw_bytes':bytes_,'nonspace_tokens':no,'eligible_queries':eligible,
            'candidate_has_true_target':copyable,'eligible_nll':eligible_loss/max(1,eligible),
            'book_bpb':{b:v[0]/v[1]/math.log(2) for b,v in by_book.items()}}

def main(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    root=pathlib.Path(args.root).resolve();path=root/'checkpoint/r12_active.pal12'
    parent=pal.sha(path)
    assert parent=='dc61500ec3abd424409f136ffff517242134af77435e66f135f4363cb7ec450f',('unexpected parent SHA',parent)
    saved,r6,comp,model,bridge,adapter,old_optimizer,rng=pal.load_for_arm(root,path,True)
    assert saved['r12_updates']==10756 and len(old_optimizer.param_groups)==4
    model.eval();bridge.eval();adapter.eval()
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    net=R18LongMemory(model.memory.piece_emb.weight,max_memory=2048,min_gap=65,topk=8)
    old_rng_state=copy.deepcopy(saved['stream_rng'])
    assert float(net.logit_gain)==0.0
    net.eval()
    train=collect_hard_negatives(docs['train'],args.train_pairs,args.seed)
    dev=collect_hard_negatives(docs['dev'],args.dev_pairs,args.seed+1)
    assert train and dev
    start=time.monotonic();old_probe=test_retrieval(net,dev)
    # Subword embeddings are copied from the old checkpoint and frozen. Train
    # only learned retrieval Q/K and temperature; preserve old 4-group moments.
    retrieval_params=[p for n,p in net.named_parameters() if n not in ('logit_gain','recency')]
    new_opt=torch.optim.AdamW(retrieval_params,lr=args.lr,weight_decay=1e-4)
    best_probe=old_probe;best_net=copy.deepcopy(net.state_dict());track=[]
    for step in range(1,args.retrieval_steps+1):
        minibatch=random.choices(train,k=args.batch)
        q,c=batch_examples(minibatch)
        net.train();loss,_=net.contrastive_loss(q,c)
        new_opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(retrieval_params,1.0);new_opt.step()
        if step%args.probe_every==0 or step==args.retrieval_steps:
            probe=test_retrieval(net,dev)
            track.append({'step':step,'train_loss':float(loss),'dev':probe})
            print(json.dumps({'phase':'RETRIEVAL','step':step,'loss':float(loss),'dev_top1':probe['top1'],'dev_nll':probe['nll']}),flush=True)
            if probe['top1']>best_probe['top1'] and probe['nll']<best_probe['nll']:
                best_probe=probe;best_net=copy.deepcopy(net.state_dict())
    net.load_state_dict(best_net);net.eval()
    # Optimize only gain, retaining trained memory Q/K exactly at chosen step.
    for p in retrieval_params:p.requires_grad_(False)
    cache_train=prepare_cache(model,bridge,adapter,docs['train'],
             make_windows(docs,'train',args.long_span,args.train_windows,args.max_long_tokens),args.long_span)
    cache_short=prepare_cache(model,bridge,adapter,docs['dev'],
             make_windows(docs,'dev',24,args.short_dev_windows,180),24)
    cache_long=prepare_cache(model,bridge,adapter,docs['dev'],
             make_windows(docs,'dev',args.long_span,args.long_dev_windows,args.max_long_tokens),args.long_span)
    orig_short=summarize_cache(net,cache_short);orig_long=summarize_cache(net,cache_long)
    # Test exact inherited-logit identity on arbitrary cache rows.
    assert all(torch.equal(row['logits'],apply_copy_boost(net,row['logits'],row['queries']))
               for row in cache_short[:2]+cache_long[:2]),'zero-gate must reproduce original logits bit-exact'
    print(json.dumps({'phase':'BASE','short':orig_short,'long':orig_long,'retrieval':old_probe,'learned':best_probe}),flush=True)
    opt_gain=torch.optim.AdamW([{'params':[net.logit_gain,net.recency],'lr':args.gain_lr}],weight_decay=0.)
    gain_trace=[]
    for step in range(1,args.gain_steps+1):
        row=cache_train[(step-1)%len(cache_train)]
        net.train();pred=apply_copy_boost(net,row['logits'],row['queries'])
        loss=F.cross_entropy(pred,row['tokens'])
        opt_gain.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([net.logit_gain,net.recency],1.)
        opt_gain.step()
        # Default prevents huge additive-bias excursions on tiny corpus.
        with torch.no_grad():net.logit_gain.clamp_(0.,args.max_gain);net.recency.clamp_(-1.,1.)
        if step%20==0: gain_trace.append({'step':step,'loss':float(loss),'gain':float(net.logit_gain),'recency':float(net.recency)})
    net.eval()
    dev_short=summarize_cache(net,cache_short);dev_long=summarize_cache(net,cache_long)
    probe=test_retrieval(net,dev)
    # Gate intentionally stringent: probe win + both dev scopes improve BOTH
    # overall BPB and nonspace NLL, and long eligible NLL improves. A tiny
    # short-window score is not enough to promote the continuation.
    good_retrieval=(probe['top1']>=old_probe['top1']+.05 and probe['nll']<old_probe['nll']-.05)
    improves_short=all(dev_short[x]<orig_short[x]-args.min_gain for x in ('bits_per_byte','other_nll'))
    improves_long=all(dev_long[x]<orig_long[x]-args.min_gain for x in ('bits_per_byte','other_nll'))
    improves_recall=(dev_long['eligible_queries']>0 and dev_long['eligible_nll']<orig_long['eligible_nll']-.001)
    promotes=bool(args.promote and good_retrieval and improves_short and improves_long and improves_recall)
    report={'round':'PAL R18','source_sha256':parent,'r12_updates':saved['r12_updates'],
            'architecture':'Unified R15 plus causal 2-piece learned long-distance retrieval, max 2048, min gap 65',
            'new_parameters':sum(p.numel() for p in net.parameters()),'old_optimizer_groups_retained':4,
            'old_optimizer_moments_retained':len(saved['stream_optimizer']['state']),
            'train_documents':len(docs['train']),'dev_documents':len(docs['dev']),
            'train_pair_examples':len(train),'dev_pair_examples':len(dev),
            'probe_untrained':old_probe,'probe_trained':probe,'probe_curve':track,
            'language_before':{'short':orig_short,'long':orig_long},
            'language_after':{'short':dev_short,'long':dev_long},
            'gain':float(net.logit_gain.detach()),'recency':float(net.recency.detach()),
            'gain_curve':gain_trace,'gates':{'retrieval':good_retrieval,'short':improves_short,'long':improves_long,'eligible':improves_recall},
            'promoted':promotes,'caveats':['No new GitHub release assets were accessible; inherited NaNROUTER books were used',
            'Development is historical, not sealed, and not a frontier-comparable benchmark',
            'Lexical associative copy is a narrow ability; no semantic generalization demonstrated',
            'Long-memory retrieval within each window only; not persistent across arbitrary segment breaks'],
            'elapsed_seconds':time.monotonic()-start}
    if promotes:
        # Preserve *one* on-disk checkpoint, all historical moments and weights.
        # Separate new optimizer namespace permits older R12 loader to resume
        # the old 4 param groups without losing the R18 auxiliary history.
        upgraded=dict(saved);upgraded['r18_memory']=net.state_dict();upgraded['r18_config']={
          'max_memory':2048,'min_gap':65,'topk':8};upgraded['r18_retrieval_optimizer']=new_opt.state_dict();
        upgraded['r18_gain_optimizer']=opt_gain.state_dict();upgraded['r18_parent_sha256']=parent;
        upgraded['r18_train_pairs']=len(train);upgraded['r18_gain_steps']=args.gain_steps
        tmp=path.with_suffix('.pal12.tmp');torch.save(upgraded,tmp);os.replace(tmp,path)
        report['promoted_sha256']=pal.sha(path)
    else:
        assert pal.sha(path)==parent
        # Useful rejected candidate for diagnostics, but never package it as
        # a second trained checkpoint.
    (root/'reports/r18_run.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','promoted':promotes,'probe_before':old_probe['top1'],
                      'probe_after':probe['top1'],'short_before':orig_short['bits_per_byte'],
                      'short_after':dev_short['bits_per_byte'],'long_before':orig_long['bits_per_byte'],
                      'long_after':dev_long['bits_per_byte'],'time':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=18017)
    p.add_argument('--train-pairs',type=int,default=3000)
    p.add_argument('--dev-pairs',type=int,default=800)
    p.add_argument('--retrieval-steps',type=int,default=180)
    p.add_argument('--probe-every',type=int,default=30)
    p.add_argument('--batch',type=int,default=96)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--gain-lr',type=float,default=.02)
    p.add_argument('--gain-steps',type=int,default=100)
    p.add_argument('--max-gain',type=float,default=5.)
    p.add_argument('--short-dev-windows',type=int,default=32)
    p.add_argument('--long-span',type=int,default=400)
    p.add_argument('--long-dev-windows',type=int,default=8)
    p.add_argument('--train-windows',type=int,default=10)
    p.add_argument('--max-long-tokens',type=int,default=2000)
    p.add_argument('--min-gain',type=float,default=1e-5)
    p.add_argument('--promote',action='store_true')
    main(p.parse_args())
