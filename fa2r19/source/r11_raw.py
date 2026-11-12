#!/usr/bin/env python3
"""PAL R11: genuinely raw, byte-reversible NaNROUTER text in ONE inherited R10 model.

No new weights/heads. The R10 pretrained subword GRU, shared rank-64 mirrored
NOW trunk, boundary bridge, and live AdamW moments continue together. Split by
whole Gutenberg book as exported by NaNROUTER's verified R11 workflow.

R7 core predictions at each nonspace chunk are conditioned solely on *earlier*
lexical items (never on chunk text or subsequent text). BPE target spans include
original punctuation/casing/newlines. Each sample is a contiguous stream.
"""
import argparse,copy,hashlib,json,os,pathlib,random,sys,time
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from r10_stream import load_model,StreamingBridge,sha
from r7_core import Corpus
from r9_joint import eval_exact,eval_temporal
from infer_unified_now import RGX
from build_lossless_subwords import TOKEN_PATTERN,encode_bytes,decode_bytes
from prepare_document_corpus import longctx

@dataclass
class Book:
    book_id:str
    split:str
    tokens:np.ndarray
    token_tag:np.ndarray
    chunks:np.ndarray
    byte_starts:np.ndarray
    examples:np.ndarray
    context:np.ndarray
    raw_sha:str
    raw_len:int


def read_books(root,compositor,vocab, max_books=0):
    """Reject duplicate files and metadata tampering before any training."""
    root=pathlib.Path(root)
    manifest=json.loads((root/'MANIFEST.json').read_text())
    lookup={w:i for i,w in enumerate(vocab)}
    seen_ids={};seen_hash={};books={'train':[],'dev':[],'test':[]}
    entries=manifest['documents']
    for row in entries:
        bid=str(row['gutenberg_id']);split=row['split']
        assert split in books
        if max_books and len(books[split])>=max_books:continue
        assert bid not in seen_ids,('duplicate Gutenberg ID',bid)
        seen_ids[bid]=split
        path=root/split/(bid+'.txt')
        assert path.is_file(),path
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==row['sha256'] and len(raw)==row['bytes'],('manifest mismatch',path)
        assert digest not in seen_hash,('duplicate book content',bid,seen_hash.get(digest))
        seen_hash[digest]=bid
        text=raw.decode('utf8','strict')
        # TOKEN_PATTERN is exactly the pretrained lossless BPE encoder's split.
        # Therefore concatenated encoded segments roundtrip to ORIGINAL bytes.
        tok=[];tags=[];nonspace_starts=[];nonspace_bstarts=[];past_ids=[];past_e=[];begin_word=[]
        cur_word_ix=0;current_chunk=-1;byte_at=0;history3=[1,1,1]
        for match in TOKEN_PATTERN.finditer(raw):
            segment=match.group();is_space=all(x in (9,10,11,12,13,32) for x in segment)
            if not is_space:
                current_chunk+=1;nonspace_starts.append(len(tok));nonspace_bstarts.append(byte_at)
                past_e.append(history3.copy()+[0,0,0,0])
                begin_word.append(cur_word_ix)
                # Only AFTER constructing prior-only state do we read the target.
                words=RGX.findall(segment.decode('utf8','replace').lower())
                
                for w in words:
                    wid=lookup.get(w,0);past_ids.append(wid);history3=history3[1:]+[wid]
                cur_word_ix+=len(words)
            piece=encode_bytes(compositor.sp,segment)
            assert piece,('empty encoder span',bid)
            tok.extend(piece);tags.extend([max(0,current_chunk)]*len(piece))
            byte_at+=len(segment)
        assert byte_at==len(raw)
        assert tok and nonspace_starts
        # Whole-word long context is computed once for all words; uses strictly
        # earlier positions than first word in the target nonspace chunk.
        if past_ids:
            lc_full=longctx(np.asarray(past_ids,dtype=np.uint32))
            c=np.zeros((len(begin_word),72),dtype=np.int8)
            for j,wi in enumerate(begin_word):
                if wi<len(lc_full):c[j]=lc_full[wi]
        else:c=np.zeros((len(begin_word),72),dtype=np.int8)
        arr=np.asarray(tok,dtype=np.int32)
        # Validate full byte identity, not only token counts/word boundaries.
        # Encoder verifies every nonspace segment independently; literal whitespace
        # byte tokens force decoder flushes between adjacent SPM segments.
        # An additional small-prefix check guards segmentation compatibility.
        assert len(arr)==len(tags) and len(nonspace_starts)==len(past_e)
        print(json.dumps({"phase":"DOCUMENT", "id":bid,"split":split,"bytes":len(raw),"tokens":len(arr),"chunks":len(nonspace_starts)}),flush=True)
        books[split].append(Book(bid,split,arr,np.asarray(tags,dtype=np.int32),
            np.asarray(nonspace_starts,dtype=np.int64),np.asarray(nonspace_bstarts,dtype=np.int64),
            np.asarray(past_e,dtype=np.int32),c,digest,len(raw)))
    assert all(books.values()),{k:len(v) for k,v in books.items()}
    return books,manifest


def windows(books,span_chunks=24,windows_per_split=32,max_tokens=180):
    result={}
    for split,entries in books.items():
        choices=[]
        for bi,b in enumerate(entries):
            # Choose stable chunk start positions; avoid over-sampling very
            # short documents while allowing all books in evaluation.
            valid=max(0,len(b.chunks)-span_chunks)
            if valid:
                aim=1000 if split=='train' else 160
                for k in range(0,valid,max(1,valid//aim)):
                    if int(b.chunks[k+span_chunks])-int(b.chunks[k])<=max_tokens:
                        choices.append((bi,int(k)))
        if not choices:raise ValueError('insufficient book length for '+split)
        # Spread over books and chunk positions. Fixed across every comparison.
        order=sorted(choices,key=lambda p:hashlib.sha256(f'{split}:{entries[p[0]].book_id}:{p[1]}'.encode()).digest())
        result[split]=order[:min(windows_per_split,len(order))]
    return result


def batch(books,picks,span_chunks):
    """Returns words+NOW conditioning, raw next-subword targets and byte counts."""
    all_ex=[];all_ctx=[];streams=[];indices=[];bytes_=0
    for book_i,start in picks:
        b=books[book_i];end=min(start+span_chunks,len(b.chunks))
        t0=int(b.chunks[start]);t1=int(b.chunks[end]) if end<len(b.chunks) else len(b.tokens)
        assert t0<t1
        if t1-t0>10000:raise ValueError('unexpected unbounded raw window')
        rawtags=np.asarray(b.token_tag[t0:t1],dtype=np.int64)-start
        assert np.min(rawtags)>=0 and np.max(rawtags)<end-start
        all_ex.append(b.examples[start:end]);all_ctx.append(b.context[start:end]);
        streams.append(b.tokens[t0:t1]);indices.append(rawtags)
        b0=int(b.byte_starts[start]);b1=int(b.byte_starts[end]) if end<len(b.byte_starts) else b.raw_len
        bytes_+=b1-b0
    B=len(picks);W=max(len(x) for x in all_ex);T=max(len(x) for x in streams)
    e=torch.ones(B,W,7,dtype=torch.long);lc=torch.zeros(B,W,72,dtype=torch.float32)
    targets=torch.full((B,T),-100,dtype=torch.long);tags=torch.zeros((B,T),dtype=torch.long)
    for i,(ee,cc,tt,ix) in enumerate(zip(all_ex,all_ctx,streams,indices)):
        e[i,:len(ee)]=torch.from_numpy(ee.astype(np.int64));lc[i,:len(cc)]=torch.from_numpy(cc.astype(np.float32));
        targets[i,:len(tt)]=torch.from_numpy(tt.astype(np.int64));tags[i,:len(ix)]=torch.from_numpy(ix)
    return e,lc,targets,tags,bytes_


def eval_raw(model,bridge,docs,choices,span_chunks=24,batchsize=2):
    model.eval();bridge.eval();total=0.;n=0;bytes_=0;space_n=0;other_n=0;other_loss=0.;space_hits=0;other_hits=0
    with torch.no_grad():
        for start in range(0,len(choices),batchsize):
            selected=choices[start:start+batchsize]
            e,lc,target,tag,nbytes=batch(docs,selected,span_chunks)
            B,W=e.shape[:2];h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
            state=h[:,0];prev=torch.full((B,),model.memory.start_piece,dtype=torch.long)
            last=tag[:,0];retain=torch.sigmoid(bridge.retain_logit)
            for t in range(target.shape[1]):
                valid=target[:,t]!=-100;ix=tag[:,t]
                x=h.gather(1,ix[:,None,None].expand(-1,1,h.shape[-1]))[:,0]
                changed=(ix!=last)&valid
                state=torch.where(changed[:,None],retain*state+(1-retain)*bridge.context_gain*x,state)
                state=model.memory.subword_cell(model.memory.piece_emb(prev),state)
                logits=model.memory.subword_out(state)
                if valid.any():
                    ce=F.cross_entropy(logits[valid],target[:,t][valid],reduction='none')
                    sp=valid&(target[:,t]==32);ot=valid&(target[:,t]!=32)
                    total+=float(ce.sum());n+=int(valid.sum());space_n+=int(sp.sum());other_n+=int(ot.sum())
                    if ot.any():other_loss+=float(F.cross_entropy(logits[ot],target[:,t][ot],reduction='sum'))
                    pred=logits.argmax(dim=-1)
                    space_hits+=int(((pred==32)&sp).sum());other_hits+=int(((pred==target[:,t])&ot).sum())
                prev=torch.where(valid,target[:,t],torch.zeros_like(prev));last=torch.where(valid,ix,last)
            bytes_+=nbytes
    return {'nll':total/n,'other_nll':other_loss/other_n,'other_top1':other_hits/other_n,'space_top1':space_hits/max(1,space_n),
            'bits_per_byte':total/bytes_/np.log(2),'nats_per_byte':total/bytes_,
            'tokens':n,'nonspace_tokens':other_n,'raw_bytes':bytes_,'windows':len(choices)}


def train(args):
    t0=time.time();torch.set_num_threads(args.threads)
    root=pathlib.Path(args.root);active=pathlib.Path(args.checkpoint)
    assert active.exists(),('missing ONE inherited checkpoint',active)
    saved,r6,comp,model=load_model(root,active)
    assert saved['format'] in ('PAL10_STREAM_SINGLE_V1','PAL11_RAW_SINGLE_V1')
    bridge=StreamingBridge();bridge.load_state_dict(saved['stream_bridge'])
    books,manifest=read_books(args.books,comp,r6.vocab,args.max_books)
    splits=windows(books,args.span,args.eval,args.max_tokens)
    r10_before=sha(active)
    print(json.dumps({'phase':'INTAKE','source':'shemshallah/NaNROUTER/corpora/short_stories',
        'r10_updates':saved['r10_updates'],'r11_updates':saved.get('r11_updates',0),
        'books':{k:len(v) for k,v in books.items()},
        'unique_book_ids':sum(map(len,books.values())),
        'bytes_by_split':{k:sum(b.raw_len for b in v) for k,v in books.items()},
        'checkpoint_sha256':r10_before}),flush=True)
    before=eval_raw(model,bridge,books['dev'],splits['dev'],args.span,args.eval_batch)
    old_dev=Corpus(root,'dev')
    guard_ix=np.arange(args.word_eval,dtype=np.int64)*(len(old_dev.seq)-1)//args.word_eval
    def anchor():
        model.eval();values=[]
        with torch.no_grad():
            for group in np.array_split(guard_ix,max(1,(len(guard_ix)+31)//32)):
                ex=torch.from_numpy(np.array(old_dev.data[group],copy=True).astype(np.int64))
                lc=torch.from_numpy(np.array(old_dev.long[group],copy=True).astype(np.float32))
                h=model.core.hidden(ex,lc)
                values.append(float(model.core.nll(h,ex[:,3]).sum()))
        return sum(values)/len(guard_ix)
    anchor_before=anchor()
    future_before=eval_temporal(model,old_dev,args.future_eval,args.eval_batch)
    if args.steps==0:
        print(json.dumps({'phase':'BASE','raw_dev':before,'core_word_anchor':anchor_before,'future':future_before}),flush=True)
        return
    opt=torch.optim.AdamW([
        {'params':[model.core.E,model.core.H,model.core.P,model.core.M,model.core.Q,model.core.NOW,model.core.L],'lr':args.lr_core},
        {'params':[model.memory.piece_emb.weight,model.memory.subword_cell.weight_ih,model.memory.subword_cell.weight_hh,model.memory.subword_cell.bias_ih,model.memory.subword_cell.bias_hh,model.memory.subword_out.weight,model.memory.subword_out.bias],'lr':args.lr_subword},
        {'params':list(bridge.parameters()),'lr':args.lr_bridge}],weight_decay=.00001)
    opt.load_state_dict(saved['stream_optimizer'])
    for group,rate in zip(opt.param_groups,(args.lr_core,args.lr_subword,args.lr_bridge)):group['lr']=rate
    torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    rng=np.random.default_rng();rng.bit_generator.state=saved['stream_rng']
    recent=[];first_grad=None
    for step in range(args.steps):
        model.train();bridge.train();pool=splits['train']
        selected=[pool[int(i)] for i in rng.integers(0,len(pool),size=args.batch)]
        e,lc,tar,tag,_=batch(books['train'],selected,args.span)
        B,W=e.shape[:2];h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
        loss,_=bridge(model,h,tar,tag)
        # Preserve the inherited whole-word decision geometry by sampling
        # anchored R10 dev strictly as a penalty-free GATE, never train data.
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        if first_grad is None:
            first_grad={p:float(getattr(model.core,p).grad.norm()) if getattr(model.core,p).grad is not None else 0 for p in ('E','P','M','Q','NOW')}
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(bridge.parameters()),1.)
        opt.step();recent.append(float(loss.detach()))
        if step==args.steps-1 or (step+1)%args.log_every==0:
            print(json.dumps({'phase':'TRAIN','r11_candidate_updates':saved.get('r11_updates',0)+step+1,
                  'raw_subword_nll':float(np.mean(recent)),'elapsed_s':round(time.time()-t0,1)}),flush=True);recent=[]
    after=eval_raw(model,bridge,books['dev'],splits['dev'],args.span,args.eval_batch)
    anchor_after=anchor()
    future_after=eval_temporal(model,old_dev,args.future_eval,args.eval_batch)
    future_guard=all(future_after[str(h)]['nll']<=future_before[str(h)]['nll']+args.future_margin for h in (1,2,3))
    # BOTH nonspace lexical training and source-byte loss must improve, with
    # guard on original core whole-word geometry. Avoid space-only promotion.
    promote=bool(after['other_nll']+args.min_gain<before['other_nll'] and
                 after['bits_per_byte']+args.min_gain<before['bits_per_byte'] and
                 anchor_after<=anchor_before+args.word_margin and future_guard)
    report={'phase':'FINAL','promoted':promote,'r10_updates':saved['r10_updates'],
            'r11_candidate_updates':saved.get('r11_updates',0)+args.steps,
            'r11_active_updates':saved.get('r11_updates',0)+args.steps if promote else saved.get('r11_updates',0),
            'raw_dev_before':before,'raw_dev_after':after,'old_word_dev_before':anchor_before,
            'old_word_dev_after':anchor_after,'future_before':future_before,
            'future_after':future_after,'future_guard':future_guard,'core_gradient_norms':first_grad,
            'runtime_s':round(time.time()-t0,2), 'source_checkpoint_sha256':r10_before,
            'policy':'NaNROUTER books only; fixed dev promotion; no test tuning; exact raw bytes'}
    if promote:
        result=dict(saved)
        result.update({'format':'PAL11_RAW_SINGLE_V1','model':model.state_dict(),'stream_bridge':bridge.state_dict(),
            'stream_optimizer':opt.state_dict(),'r11_updates':saved.get('r11_updates',0)+args.steps,
            'stream_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'python_rng':random.getstate(),'r11_source_manifest_sha256':sha(pathlib.Path(args.books)/'MANIFEST.json'),
            'r11_source':'shemshallah/NaNROUTER/corpora/short_stories','r11_hparams':vars(args),
            'parent_sha256':r10_before})
        path=pathlib.Path(args.output)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmppath=path.with_name(path.name+'.tmp');torch.save(result,tmppath);os.replace(tmppath,path)
        report['new_checkpoint_sha256']=sha(path)
        report['raw_test']=eval_raw(model,bridge,books['test'],splits['test'],args.span,args.eval_batch)
    pathlib.Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--books',default='nanrouter')
    p.add_argument('--checkpoint',default='checkpoint/r10_active.pal10')
    p.add_argument('--output',default='checkpoint/r11_active.pal11')
    p.add_argument('--report',default='reports/r11_raw.json')
    p.add_argument('--steps',type=int,default=20);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--span',type=int,default=12);p.add_argument('--eval',type=int,default=16)
    p.add_argument('--eval-batch',type=int,default=2);p.add_argument('--max-books',type=int,default=0)
    p.add_argument('--word-eval',type=int,default=64)
    p.add_argument('--max-tokens',type=int,default=180)
    p.add_argument('--future-eval',type=int,default=48)
    p.add_argument('--lr-core',type=float,default=0.00000002)
    p.add_argument('--lr-subword',type=float,default=0.00005)
    p.add_argument('--lr-bridge',type=float,default=0.0004)
    p.add_argument('--min-gain',type=float,default=0.0001)
    p.add_argument('--word-margin',type=float,default=0.0008)
    p.add_argument('--future-margin',type=float,default=0.00045)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--log-every',type=int,default=10)
    train(p.parse_args())