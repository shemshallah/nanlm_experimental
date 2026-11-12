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
        if prev['train_split_sha256']!=sha(root/'corpus/train_temporal.i32'):raise ValueError('checkpoint training split checksum mismatch')
        model.load_state_dict(prev['adapter']);optim.load_state_dict(prev['optimizer']);updates=int(prev['adapter_updates'])
        torch.set_rng_state(prev['torch_rng']);np.random.set_state(prev['numpy_rng'])
        random.setstate(prev['python_rng'])
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
    losses=[];auxes=[];sub_losses=[]
    for k in range(args.steps):
        model.train();p=rg.integers(WORD_CONTEXT,len(train.seq),size=args.batch,dtype=np.int64)
        h,context,lp,ll,y,tp,tl=batch_data(r6,train,tc,p)
        h,context,lp,ll,y,tp,tl=(v.to(device) for v in (h,context,lp,ll,y,tp,tl))
        pred,_=model(h,context,lp,ll)
        l_word=r6.nll(pred,y).mean()
        l_sub=model.subword_loss(pred,tp,tl)
        # Keep known semantic embeddings as a stable target for the new compositional channel.
        tail=context[:,-LOCAL:];known=tail>1
        compword=model.compose(lp,ll)
        target=F.embedding(tail,model.word_embedding)
        l_align=F.mse_loss(compword[known],target[known]) if bool(known.any()) else l_word*0
        loss=l_word+args.subword_weight*l_sub+args.align_weight*l_align
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite R7 loss')
        optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
        losses.append(l_word.item());auxes.append(l_align.item());sub_losses.append(l_sub.item())
        if (k+1)%args.log_every==0 or k+1==args.steps:
            print(json.dumps({'phase':'TRAIN','adapter_updates':updates+k+1,'word_nll_recent':float(np.mean(losses)),'subword_nll_recent':float(np.mean(sub_losses)),'align_recent':float(np.mean(auxes)),'gamma':float(model.residual_scale.item()),'elapsed_s':round(time.time()-t,2)},sort_keys=True),flush=True)
            losses.clear();sub_losses.clear();auxes.clear()
    updates+=args.steps
    dev_res=measure(r6,model,dev,tc,args.eval,args.batch)
    test_res=measure(r6,model,test,tc,args.eval,args.batch)
    disabled_dev=measure(r6,model,dev,tc,args.eval,args.batch,ablate=True)
    disabled_test=measure(r6,model,test,tc,args.eval,args.batch,ablate=True)
    print(json.dumps({'phase':'EVAL','updates':updates,'dev':dev_res,'test':test_res,'disabled_dev':disabled_dev,'disabled_test':disabled_test,'seconds':round(time.time()-t,2)},sort_keys=True),flush=True)
    checkpoint={
        'format':'PAL_R7_ORDERED_2048_V1','base_step':43850000,'base_sha256':old_hash,
        'adapter_updates':updates,'adapter':model.state_dict(),'optimizer':optim.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'batch_rng_state':rg.bit_generator.state,'batch_size':args.batch,'device':str(device),
        'model_desc':'one residual adapter on preserved PAL5/PAL7/PAL9/PAL11 rank64 model',
        'tokenizer_sha256':sha(root/'tokenizer/gutenberg_r7.model'),
        'train_split_sha256':sha(pathlib.Path(args.data_dir or root/'corpus')/'train_temporal.i32')}
    out=cp/'r7_adapter.pt';torch.save(checkpoint,out)
    report={'source_checkpoint_steps':43850000,'adapter_updates':updates,'effective_total_update_events':43850000+updates,'note':'R6 trunk remains frozen; optimizer state is preserved without fabricated updates to its header.','dev':dev_res,'test':test_res,'dev_memory_ablation':disabled_dev,'test_memory_ablation':disabled_test,'legacy_base_dev':baseline_dev,'legacy_base_test':baseline_test,'tokenizer_vocab_size':tc.npieces,'parameters_added':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(out),'base_sha256':old_hash,'elapsed_s':round(time.time()-t,2)}
    (reports/'R7_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'));p.add_argument('--data-dir',type=pathlib.Path,default=None);p.add_argument('--allow-new-corpus',action='store_true');p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=16);p.add_argument('--eval',type=int,default=512);p.add_argument('--lr',type=float,default=.002);p.add_argument('--seed',type=int,default=20261111);p.add_argument('--subword-weight',type=float,default=.08);p.add_argument('--align-weight',type=float,default=.1);p.add_argument('--threads',type=int,default=2);p.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu');p.add_argument('--log-every',type=int,default=50);p.add_argument('--resume',type=pathlib.Path,default=None)
    a=p.parse_args();print(json.dumps(run(a),indent=2),flush=True)
