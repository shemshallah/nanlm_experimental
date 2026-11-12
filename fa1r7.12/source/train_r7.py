#!/usr/bin/env python3
"""R7 additive ordered-memory/subword training on ONE frozen 43.85M R6 model.

R6 original bytes and optimizer untouched; train only progressively added R7 tensors.
No future labels, dev, or test tokens are inputs to the training objective.
"""
import argparse,hashlib,json,os,random,time,pathlib,sys
import numpy as np,torch
import torch.nn.functional as F
from r7_core import R6,Corpus
from r7_memory import R7Memory,TokenCompositor,WORD_CONTEXT,LOCAL
from build_lossless_subwords import encode_bytes

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def batch_data(r6,c,comp,positions):
    positions=np.asarray(positions,dtype=np.int64)
    ctx,indices=c.get(positions,context=WORD_CONTEXT)
    old_h=r6.hidden(c.data[positions],c.long[positions])
    lp,ll=comp.local_pieces(c,indices[:,-LOCAL:],ctx[:,-LOCAL:],r6.vocab)
    y=c.data[positions,3].copy()
    words=comp.strings(c,positions,r6.vocab)
    tp,tl=comp.encode_words(words)
    return torch.from_numpy(old_h),torch.from_numpy(ctx.astype('int64')),lp,ll,torch.from_numpy(y.astype('int64')),tp,tl

def future_labels(corpus,positions):
    """Return +1/2/3 labels, masked at document/split boundaries.

    The legacy corpus uses BOS=1 as future padding. Imported corpora also
    supply exact doc starts so no future supervision crosses a book boundary.
    """
    positions=np.asarray(positions,dtype=np.int64)
    targets=corpus.data[positions,4:7].astype(np.int64).copy()
    mask=targets!=1
    if corpus.doc_start is not None:
        for lag in (1,2,3):
            nxt=positions+lag
            legal=nxt<len(corpus.seq)
            safe=np.minimum(nxt,len(corpus.seq)-1)
            mask[:,lag-1] &= legal & (corpus.doc_start[safe]==corpus.doc_start[positions])
    return torch.from_numpy(targets),torch.from_numpy(mask)

def stream_targets(corpus,comp,vocab,positions,words=6,max_pieces=96):
    """Reconstruct continuous future text, anchored to each strictly causal NOW.

    The historical corpus stores normalized word targets rather than original
    byte offsets. We therefore train a canonical single-space word stream;
    punctuation/novel spellings are retained, but original whitespace is not.
    """
    positions=np.asarray(positions,dtype=np.int64)
    rows=[]
    for pos in positions:
        lo=int(pos);hi=min(len(corpus.seq),lo+words)
        if corpus.doc_start is not None:
            while hi>lo+1 and corpus.doc_start[hi-1]!=corpus.doc_start[lo]:hi-=1
        data=' '.join(corpus.spelling(i,vocab) for i in range(lo,hi)).encode('utf8')
        ids=encode_bytes(comp.sp,data)
        if not ids:raise ValueError('empty causal sequence for position '+str(pos))
        if len(ids)>max_pieces:ids=ids[:max_pieces]
        rows.append(ids)
    maximum=max(map(len,rows));batch=np.zeros((len(rows),maximum),dtype=np.int64)
    lens=np.array([len(row) for row in rows],dtype=np.int64)
    for j,row in enumerate(rows):batch[j,:len(row)]=row
    return torch.from_numpy(batch),torch.from_numpy(lens)

def load_extension(model,optim,prev):
    """Migrate R7 model and AdamW state without reinitializing old slots."""
    original=prev['adapter'];new=model.state_dict()
    old_missing=set(new)-set(original)
    unexpected=set(original)-set(new)
    if unexpected:raise ValueError('checkpoint has unknown parameters: '+str(unexpected))
    allow={'old_token_pos','future_rail','future_rail_gain'}
    if old_missing-allow:raise ValueError('unexpected missing checkpoint tensors '+str(old_missing-allow))
    # Every existing tensor, including all trained parameters, is copied intact.
    model.load_state_dict(original,strict=False)
    oldopt=prev['optimizer'];newopt=optim.state_dict()
    old_ids=oldopt['param_groups'][0]['params'];new_ids=newopt['param_groups'][0]['params']
    if len(new_ids)<len(old_ids):raise ValueError('optimizer parameter list shrank')
    # Torch registers direct nn.Parameters before submodule parameters.
    # Added direct parameters can shift numeric optimizer slots. Map the old
    # slots by ORIGINAL parameter names, preserving all Adam moments exactly.
    ordered=list(model.named_parameters())
    new_by_name={name:new_id for (name,_),new_id in zip(ordered,new_ids)}
    old_names=[name for name,param in ordered if name in original]
    if len(old_names)!=len(old_ids):
        raise ValueError('old model parameter count changed; cannot safely migrate optimizer')
    new_param_by_name=dict(ordered)
    for name,old_id in zip(old_names,old_ids):
        slot=oldopt['state'].get(old_id,{})
        for v in slot.values():
            if isinstance(v,torch.Tensor) and v.ndim>0 and v.shape!=new_param_by_name[name].shape:
                raise ValueError('optimizer slot shape mismatch for '+name)
        if slot:newopt['state'][new_by_name[name]]=slot
    for key,val in oldopt['param_groups'][0].items():
        if key!='params':newopt['param_groups'][0][key]=val
    optim.load_state_dict(newopt)
    return sorted(old_missing)

def measure(r6,model,corpus,comp,n=512,batch=16,ablate=False,**forward_options):
    model.eval();device=next(model.parameters()).device;positions=np.arange(n,dtype=np.int64)*(len(corpus.seq)-1)//n
    total=old_total=0.;known=added=oov=0.;kn=an=un=0
    with torch.no_grad():
        for p in np.array_split(positions,max(1,(len(positions)+batch-1)//batch)):
            h,context,lp,ll,y,tp,tl=batch_data(r6,corpus,comp,p)
            h,context,lp,ll,y,tp,tl=(v.to(device) for v in (h,context,lp,ll,y,tp,tl))
            new,_=model(h,context,lp,ll,ablate=ablate,**forward_options)
            loss=r6.nll(new,y).cpu().numpy();old_loss=r6.nll(h,y).cpu().numpy()
            total+=float(loss.sum());old_total+=float(old_loss.sum())
            ym=y.cpu().numpy();is_oov=ym==0;is_orig=(ym>0)&(ym<4501);is_added=ym>=4501
            known+=float(loss[is_orig].sum());kn+=int(is_orig.sum())
            added+=float(loss[is_added].sum());an+=int(is_added.sum())
            oov+=float(loss[is_oov].sum());un+=int(is_oov.sum())
    return {'nll':total/n,'old_nll':old_total/n,'original_word_nll':known/max(1,kn),'added_word_nll':added/max(1,an),'unk_nll':oov/max(1,un),'samples':n,'original_word_count':kn,'added_word_count':an,'unk_count':un}

def run(args):
    torch.set_num_threads(args.threads);np.random.seed(args.seed);random.seed(args.seed);torch.manual_seed(args.seed)
    root=args.root;cp=root/'checkpoint';reports=root/'reports';reports.mkdir(exist_ok=True)
    r6=R6(root);tc=TokenCompositor(root,r6)
    model=R7Memory(r6,tc)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    model.to(device)
    for attr in ('Wt','Bt','pnodest','pbitst','pmaskt'):setattr(r6,attr,getattr(r6,attr).to(device))
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0005)
    old_hash=sha(cp/'weights.raw');updates=0
    if args.resume:
        prev=torch.load(args.resume,map_location='cpu',weights_only=False)
        if prev['base_sha256']!=old_hash:raise ValueError('checkpoint base checksum mismatch')
        if prev.get('tokenizer_sha256')!=sha(root/'tokenizer/gutenberg_r7.model'):raise ValueError('checkpoint tokenizer checksum mismatch')
        corpus_origin=pathlib.Path(args.data_dir or root/'corpus')/'train_temporal.i32'
        if prev.get('train_split_sha256') != sha(corpus_origin) and not args.allow_new_corpus:
            raise ValueError('new training corpus requires --allow-new-corpus for auditable continuation')
        if prev['tokenizer_sha256']!=sha(root/'tokenizer/gutenberg_r7.model'):raise ValueError('checkpoint tokenizer checksum mismatch')
        # Do not erroneously reject explicitly authorized new whole-document corpora.
        added_tensors=load_extension(model,optim,prev);updates=int(prev['adapter_updates'])
        torch.set_rng_state(prev['torch_rng']);np.random.set_state(prev['numpy_rng'])
        random.setstate(prev['python_rng'])
        for group in optim.param_groups:group['lr']=args.lr
    train=Corpus(root,'train',args.data_dir);dev=Corpus(root,'dev',args.data_dir);test=Corpus(root,'test',args.data_dir)
    t=time.time();baseline_dev=measure(r6,model,dev,tc,args.eval,args.batch,ablate=True)
    baseline_test=measure(r6,model,test,tc,args.eval,args.batch,ablate=True)
    cold_dev=measure(r6,model,dev,tc,args.eval,args.batch)
    assert abs(cold_dev['nll']-baseline_dev['nll'])<1.e-5 if updates==0 else True
    print(json.dumps({'phase':'BASE','step':updates,'dev':cold_dev,'control_dev':baseline_dev,'control_test':baseline_test,'model_params':sum(p.numel() for p in model.parameters()),'base_sha256':old_hash},sort_keys=True),flush=True)
    # Training positions are drawn independently, with replacement, from the train split.
    rg=np.random.default_rng(args.seed+updates)
    if args.resume and 'batch_rng_state' in prev:
        rg.bit_generator.state=prev['batch_rng_state']
    def save_checkpoint(iteration):
        # Atomic replace: exactly one active adapter checkpoint, no competing lineage.
        value={
            'format':'PAL_R7_ORDERED_2048_V2','base_step':43850000,'base_sha256':old_hash,
            'adapter_updates':iteration,'adapter':model.state_dict(),'optimizer':optim.state_dict(),
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'python_rng':random.getstate(),'batch_rng_state':rg.bit_generator.state,
            'batch_size':args.batch,'device':str(device),
            'model_desc':'one R7 memory + causal rank-64 future rails on preserved PAL6',
            'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
            'train_split_sha256':sha(pathlib.Path(args.data_dir or root/'corpus')/'train_temporal.i32'),
            'seq_len':args.seq_len,'future_weight':args.future_weight,'stream_weight':args.stream_weight,'stream_words':args.stream_words}
        tmp=cp/'r7_adapter.pt.tmp'
        torch.save(value,tmp);os.replace(tmp,cp/'r7_adapter.pt')
        print(json.dumps({'phase':'SAVE','adapter_updates':iteration,'checkpoint':str(cp/'r7_adapter.pt')}),flush=True)
    losses=[];auxes=[];sub_losses=[]
    for k in range(args.steps):
        model.train()
        if args.seq_len>1:
            # Draw complete causal runs from the same document; contiguous target
            # supervision is deterministic given the batch RNG in the checkpoint.
            starts=rg.integers(WORD_CONTEXT,len(train.seq)-args.seq_len,size=max(1,args.batch//args.seq_len),dtype=np.int64)
            if train.doc_start is not None:
                valid=np.array([train.doc_start[s]==train.doc_start[s+args.seq_len-1] for s in starts])
                for j in np.flatnonzero(~valid):
                    for attempt in range(128):
                        candidate=int(rg.integers(WORD_CONTEXT,len(train.seq)-args.seq_len))
                        if train.doc_start[candidate]==train.doc_start[candidate+args.seq_len-1]:
                            starts[j]=candidate;break
                    else:raise RuntimeError('no full-length document run found in 128 samples')
            p=(starts[:,None]+np.arange(args.seq_len)[None,:]).reshape(-1)
        else:p=rg.integers(WORD_CONTEXT,len(train.seq),size=args.batch,dtype=np.int64)
        h,context,lp,ll,y,tp,tl=batch_data(r6,train,tc,p)
        h,context,lp,ll,y,tp,tl=(v.to(device) for v in (h,context,lp,ll,y,tp,tl))
        pred,_=model(h,context,lp,ll)
        l_word=r6.nll(pred,y).mean()
        l_sub=model.subword_loss(pred,tp,tl)
        fy,fvalid=future_labels(train,p)
        l_future=model.future_loss(pred,r6,fy.to(device),fvalid.to(device)) if args.future_weight else l_word*0
        # Keep known semantic embeddings as a stable target for the new compositional channel.
        tail=context[:,-LOCAL:];known=tail>1
        compword=model.compose(lp,ll)
        target=F.embedding(tail,model.word_embedding)
        l_align=F.mse_loss(compword[known],target[known]) if bool(known.any()) else l_word*0
        if args.stream_weight:
            stream_n=min(len(p),args.stream_batch)
            st,sl=stream_targets(train,tc,r6.vocab,p[:stream_n],words=args.stream_words)
            l_stream=model.stream_loss(pred[:stream_n],st.to(device),sl.to(device))
        else:l_stream=l_word*0
        loss=l_word+args.subword_weight*l_sub+args.align_weight*l_align+args.future_weight*l_future+args.stream_weight*l_stream
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite R7 loss')
        optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        losses.append(l_word.item());auxes.append(l_align.item());sub_losses.append(l_sub.item())
        if args.save_every and (k+1)%args.save_every==0:save_checkpoint(updates+k+1)
        if (k+1)%args.log_every==0 or k+1==args.steps:
            print(json.dumps({'phase':'TRAIN','adapter_updates':updates+k+1,'word_nll_recent':float(np.mean(losses)),'subword_nll_recent':float(np.mean(sub_losses)),'align_recent':float(np.mean(auxes)),'future_nll_recent':round(float(l_future.item()),6),'stream_nll_recent':round(float(l_stream.item()),6),'gamma':float(model.residual_scale.item()),'elapsed_s':round(time.time()-t,2)},sort_keys=True),flush=True)
            losses.clear();sub_losses.clear();auxes.clear()
    updates+=args.steps
    dev_res=measure(r6,model,dev,tc,args.eval,args.batch)
    test_res=measure(r6,model,test,tc,args.eval,args.batch)
    disabled_dev=measure(r6,model,dev,tc,args.eval,args.batch,ablate=True)
    disabled_test=measure(r6,model,test,tc,args.eval,args.batch,ablate=True)
    print(json.dumps({'phase':'EVAL','updates':updates,'dev':dev_res,'test':test_res,'disabled_dev':disabled_dev,'disabled_test':disabled_test,'seconds':round(time.time()-t,2)},sort_keys=True),flush=True)
    save_checkpoint(updates)
    out=cp/'r7_adapter.pt'
    report={'source_checkpoint_steps':43850000,'adapter_updates':updates,'effective_total_update_events':43850000+updates,'note':'R6 trunk remains frozen; optimizer state is preserved without fabricated updates to its header.','dev':dev_res,'test':test_res,'dev_memory_ablation':disabled_dev,'test_memory_ablation':disabled_test,'legacy_base_dev':baseline_dev,'legacy_base_test':baseline_test,'tokenizer_vocab_size':tc.npieces,'parameters_added':sum(p.numel() for p in model.parameters()),'future_weight':args.future_weight,'seq_len':args.seq_len,'stream_weight':args.stream_weight,'stream_words':args.stream_words,'checkpoint_sha256':sha(out),'base_sha256':old_hash,'elapsed_s':round(time.time()-t,2)}
    (reports/'R7_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'));p.add_argument('--data-dir',type=pathlib.Path,default=None);p.add_argument('--allow-new-corpus',action='store_true');p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=16);p.add_argument('--eval',type=int,default=512);p.add_argument('--lr',type=float,default=.002);p.add_argument('--seed',type=int,default=20261111);p.add_argument('--stream-weight',type=float,default=.12);p.add_argument('--stream-words',type=int,default=6);p.add_argument('--stream-batch',type=int,default=4);p.add_argument('--seq-len',type=int,default=4);p.add_argument('--future-weight',type=float,default=.02);p.add_argument('--subword-weight',type=float,default=.08);p.add_argument('--align-weight',type=float,default=.1);p.add_argument('--threads',type=int,default=2);p.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu');p.add_argument('--save-every',type=int,default=100);p.add_argument('--log-every',type=int,default=50);p.add_argument('--resume',type=pathlib.Path,default=None)
    a=p.parse_args();print(json.dumps(run(a),indent=2),flush=True)
