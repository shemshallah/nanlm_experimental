#!/usr/bin/env python3
"""Train R25 causally from R24 step220, one unified network and optimizer.
Frozen R24 forward cache avoids repeating 12M parameter backprop for a small
adapter pilot; legacy tensor values and all legacy Adam histories are retained.
"""
import argparse,bisect,copy,hashlib,inspect,json,math,pathlib,random,time,types
import numpy as np,torch
from torch.nn import functional as F
from r24_unified_cpu import initialize
from r13_corpus_cache import read_books_cached
from r23_stream_carry import ContinuousPALState,contiguous_book_picks
from r25_episodic_cache import AdaptiveNGramMixer,CausalNGramIndex

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def initialize_compatible(root):
    """GH master has an args-based R24 initializer; local release uses a seed.

    These constructor values match the exact locally executed R24 model.
    All selected weights and all optimizer moments are loaded afterwards.
    """
    if list(inspect.signature(initialize).parameters)[-1]!='args':
        return initialize(root)
    args=types.SimpleNamespace(seed=24024,local_history=256,
        expert_width=384,experts=4,expert_topk=2,
        core_lr=2e-8,subword_lr=1e-5,bridge_lr=5e-5,local_lr=5e-5,
        r18_lr=1e-4,r21_lr=5e-5,r22_lr=5e-5,r23_lr=3e-4,expert_lr=2e-4)
    u,opt,comp,r6,meta=initialize(root,args)
    if 'parent_sha256' not in meta:meta=dict(meta,parent_sha256=meta['accepted_sha256'])
    return u,opt,comp,r6,meta

def prepare(root,threads=2):
    torch.set_num_threads(threads)
    unified,optimizer,comp,r6,meta=initialize_compatible(root)
    legacy_path=root/'reports/fa2_unified_candidate.pth'
    if legacy_path.is_file():
        incoming=torch.load(legacy_path,map_location='cpu',weights_only=False)
        if incoming['format']!='FA2_R24_UNIFIED_CANDIDATE_V1' or incoming['selected']['step']!=220 or incoming['accepted_model_promoted']:
            raise ValueError('expected unpromoted unified R24 step220')
        unified.load_state_dict(incoming['selected']['model']);optimizer.load_state_dict(incoming['selected']['optimizer'])
        original_states=len(optimizer.state)
        unified.add_module('r25',AdaptiveNGramMixer())
        optimizer.add_param_group({'params':list(unified.r25.parameters()),'lr':.015,'weight_decay':1e-4})
    else:
        # The latest R25 recovery is self-contained relative to immutable R19;
        # no separate R24 binary is necessary to resume this same model.
        incoming=torch.load(root/'reports/fa2_r25_unified_candidate.pth',map_location='cpu',weights_only=False)
        if incoming['format']!='FA2_R25_UNIFIED_CANDIDATE_V1' or incoming['accepted_model_promoted']:
            raise ValueError('expected latest R25 unified candidate')
        if incoming['parent']['parent_sha256']!=meta['parent_sha256']:
            raise ValueError('R18 parent mismatch')
        unified.add_module('r25',AdaptiveNGramMixer())
        optimizer.add_param_group({'params':list(unified.r25.parameters()),'lr':.015,'weight_decay':1e-4})
        unified.load_state_dict(incoming['selected']['model']);optimizer.load_state_dict(incoming['selected']['optimizer'])
        original_states=sum(len(g['params']) for g in optimizer.param_groups[:9] if g['params'])
        # State count, not parameter count; inherited legacy Adam moments are 123.
        legacy_ids={k for g in incoming['selected']['optimizer']['param_groups'][:9] for k in g['params']}
        original_states=sum(k in incoming['selected']['optimizer']['state'] for k in legacy_ids)
    if len(optimizer.param_groups)!=10 or original_states!=123:raise ValueError('legacy optimizer not preserved')
    for name,p in unified.named_parameters():p.requires_grad_(name.startswith('r25.'))
    return unified,optimizer,comp,r6,meta,incoming,original_states

def source_warmup(model,book,seg,carry):
    begin,end,t0,t1=seg
    ex=torch.as_tensor(book.examples[begin:end].astype(np.int64))[None]
    lc=torch.as_tensor(book.context[begin:end].astype(np.float32))[None]
    tags=torch.as_tensor(book.token_tag[t0:t1].astype(np.int64))[None]-begin
    tok=torch.as_tensor(book.tokens[t0:t1].astype(np.int64))[None]
    h=model.model.core.hidden(ex.reshape(-1,7),lc.reshape(-1,72)).reshape(1,end-begin,-1)
    carry.forward_chunk(model.model,model.bridge,model.adapter,h,tok,tags,book_id=book.book_id,token_offset=t0,chunk_tag_offset=begin,detach_after=True)

def extract(model,book,index,donor_index,targets,chunk_span,windows):
    segments=list(contiguous_book_picks(book,chunk_span))
    starts=[v[2] for v in segments]
    selected=set()
    for offset in targets:
        ix=bisect.bisect_left(starts,offset)
        if ix>=len(starts):continue
        for j in range(ix,min(len(starts),ix+windows)):selected.add(j)
    if not selected:return []
    carry=ContinuousPALState(book.book_id);out=[]
    with torch.no_grad():
        for i in range(max(selected)+1):
            segment=segments[i]
            if i not in selected:
                source_warmup(model,book,segment,carry)
                continue
            row=model.forward_chunk(book,segment,carry,apply_r25=False)
            t0=segment[2];nt=len(row['target']);v=row['logits'].shape[-1]
            cand,feat=index.distributions(book.tokens,t0,nt,v)
            cutoff=min(t0,len(donor_index.tokens)-nt)
            wrong,wfeat=donor_index.distributions(book.tokens,t0,nt,v,source_cutoff=cutoff)
            out.append({'base':row['logits'].detach().clone(), 'cand':torch.from_numpy(cand),
                        'feat':torch.from_numpy(feat),'wrong':torch.from_numpy(wrong),'wfeat':torch.from_numpy(wfeat),
                        'target':row['target'].clone(),'byte_count':row['bytes'],'book_id':book.book_id,
                        'start':t0,'label':str(segment)})
            carry.detach()
    return out

def extract_split(model,books,targets,span,windows,label):
    start=time.monotonic();indexes=[CausalNGramIndex(b.tokens) for b in books];results=[]
    for i,b in enumerate(books):
        chunks=extract(model,b,indexes[i],indexes[(i+1)%len(books)],targets,span,windows)
        results.extend(chunks)
        print('CACHED',json.dumps({'split':label,'book_id':b.book_id,'windows':len(chunks),
            'tokens':sum(len(c['target']) for c in chunks),'sec':round(time.monotonic()-start,2)}),flush=True)
    return results

def metrics(unified,items):
    unified.eval();loss=baseline=donor=nonspace=0.;count=nb=bytes_=0;per_book={};alphas=[]
    with torch.no_grad():
        for item in items:
            y=item['target'];b=item['base'];p,info=unified.r25(b,item['cand'],item['feat'],return_info=True)
            w=unified.r25(b,item['wrong'],item['wfeat']);correct=F.nll_loss(p,y,reduction='sum')
            d=F.nll_loss(w,y,reduction='sum');raw=F.cross_entropy(b,y,reduction='sum')
            count+=len(y);bytes_+=item['byte_count'];loss+=float(correct);donor+=float(d);baseline+=float(raw)
            mask=y.ne(32);nb+=int(mask.sum());nonspace+=float(F.nll_loss(p[mask],y[mask],reduction='sum')) if bool(mask.any()) else 0.
            book=per_book.setdefault(item['book_id'],[0.,0.,0.,0])
            book[0]+=float(correct);book[1]+=float(d);book[2]+=float(raw);book[3]+=len(y)
            alphas.append(float(info['alpha'].mean()))
    return {'bpb':loss/(bytes_*math.log(2)), 'baseline_bpb':baseline/(bytes_*math.log(2)),
            'delta_bpb':(baseline-loss)/(bytes_*math.log(2)),
            'correct_vs_wrong_nats':(donor-loss)/count,'nonspace_nll':nonspace/max(nb,1),
            'bytes':bytes_,'tokens':count,'mean_alpha':float(np.mean(alphas)),
            'book_wins':sum(1 for x in per_book.values() if x[1]>x[0]), 'books':len(per_book),
            'per_book':[{'book':k,'n':v[3],'delta_nats':(v[2]-v[0])/v[3],'donor_adv_nats':(v[1]-v[0])/v[3]} for k,v in per_book.items()]}

def run(args):
    start=time.monotonic();root=pathlib.Path(args.root).resolve();torch.manual_seed(25);random.seed(25);np.random.seed(25)
    model,opt,comp,r6,meta,incoming,legacy=prepare(root,args.threads)
    r24_hash=(sha(root/'reports/fa2_unified_candidate.pth') if (root/'reports/fa2_unified_candidate.pth').is_file()
              else incoming['r24_sha256'])
    start_step=0
    if args.resume_r25:
        existing=torch.load(root/'reports/fa2_r25_unified_candidate.pth',map_location='cpu',weights_only=False)
        if existing['format']!='FA2_R25_UNIFIED_CANDIDATE_V1' or existing['r24_sha256']!=r24_hash:
            raise ValueError('R25 resume lineage changed')
        model.load_state_dict(existing['selected']['model'])
        opt.load_state_dict(existing['selected']['optimizer'])
        start_step=int(existing['selected']['r25_step'])
        opt.param_groups[-1]['lr']=args.lr
        if args.steps<=start_step:raise ValueError('step target must extend current R25')
        print('RESUME_R25',json.dumps({'from_step':start_step,'ten_groups':len(opt.param_groups),
             'preserved_legacy_states':legacy,'r25_lr':args.lr}),flush=True)
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    tr=books['train'][:args.train_books];dev=books['dev'][:args.dev_books]
    reserve=books['dev'][args.dev_books:args.dev_books+args.challenge_books]
    hold=books['test'][:args.test_books]
    offset=[int(s) for s in args.context_targets.split(',')]
    model.eval()
    train=extract_split(model,tr,offset,args.chunk_span,args.windows,'train')
    development=extract_split(model,dev,offset,args.chunk_span,args.windows,'development')
    challenge=extract_split(model,reserve,offset,args.chunk_span,args.windows,'challenge')
    historical=extract_split(model,hold,offset,args.chunk_span,args.windows,'nonblind-test')
    prior_amp=model.r25.amplitude.detach().clone()
    with torch.no_grad():model.r25.amplitude.zero_()
    base=metrics(model,development);extra0=metrics(model,challenge);test0=metrics(model,historical)
    with torch.no_grad():model.r25.amplitude.copy_(prior_amp)
    before=metrics(model,development)
    print('R25_BASELINE '+json.dumps({'dev':base,'challenge':extra0,'test':test0,
             'resumed_dev':{k:v for k,v in before.items() if k!='per_book'}}),flush=True)
    if abs(base['delta_bpb'])>1e-6 or abs(base['correct_vs_wrong_nats'])>1e-7:raise ValueError('initial prediction parity failed')
    best=(before['bpb'],start_step,copy.deepcopy(model.r25.state_dict()),copy.deepcopy(opt.state_dict()))
    trainable=[x for x in model.r25.parameters()]
    for step in range(start_step+1,args.steps+1):
        model.r25.train();chosen=random.sample(train,min(len(train),args.batch_windows));losses=[]
        for x in chosen:
            p=model.r25(x['base'],x['cand'],x['feat'])
            loss=F.nll_loss(p,x['target'])
            if args.donor_penalty:
                w=model.r25(x['base'],x['wrong'],x['wfeat'])
                # A tiny anti-frequency-shortcut regularizer: make correct history
                # favorable on TRAINING, without leakage into retrieval keys.
                loss=loss+args.donor_penalty*F.softplus(.01+loss-F.nll_loss(w,x['target']))
            losses.append(loss)
        opt.zero_grad(set_to_none=True);loss=torch.stack(losses).mean();loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(trainable,2))
        if not math.isfinite(grad):raise ValueError('nonfinite gradient')
        opt.step()
        if step%args.eval_every==0 or step==args.steps:
            current=metrics(model,development)
            print('R25_EVAL '+json.dumps({'step':step,'train_loss':float(loss),'grad':grad,
                    'dev':{k:v for k,v in current.items() if k!='per_book'}}),flush=True)
            if current['bpb']<best[0]-1e-6 and current['nonspace_nll']<base['nonspace_nll']+1e-4 and current['correct_vs_wrong_nats']>0:
                best=(current['bpb'],step,copy.deepcopy(model.r25.state_dict()),copy.deepcopy(opt.state_dict()))
                print('R25_SELECT '+str(step),flush=True)
    model.r25.load_state_dict(best[2]);opt.load_state_dict(best[3]);final=metrics(model,development)
    extra=metrics(model,challenge);test=metrics(model,historical)
    if (root/'reports/fa2_unified_candidate.pth').is_file() and sha(root/'reports/fa2_unified_candidate.pth')!=r24_hash:
        raise ValueError('R24 parent mutated')
    if best[1] and len(opt.state)<legacy:raise ValueError('legacy moment loss')
    report={'architecture':'one unified FA2 R25, causal n-gram adaptive residual','parent':'FA2 R24 step220',
       'r24_sha256':r24_hash,'r18_parent_sha256':meta['parent_sha256'],
       'selected_r25_updates':best[1],'attempted_r25_updates':args.steps,'resumed_from_step':start_step,'optimizer_groups':len(opt.param_groups),
       'inherited_optimizer_states':legacy,'development_initial':base,'development_selected':final,
       'challenge_initial':extra0,'challenge_selected':extra,
       'historical_initial':test0,'historical_selected':test,
       'settings':vars(args),'elapsed_seconds':time.monotonic()-start,
       'accepted_model_promoted':False,'training_contract':'Offline exact R24 frozen logits, causal document-history counts, training-only gate. Legacy tensor/optimizer histories retained.'}
    if best[1]>start_step:
        parent=incoming['selected'];selected={k:copy.deepcopy(v) for k,v in parent.items() if k!='model' and k!='optimizer'}
        selected.update(model=model.state_dict(),optimizer=opt.state_dict(),r25_step=best[1])
        torch.save({'format':'FA2_R25_UNIFIED_CANDIDATE_V1','selected':selected,'parent':meta,
                    'r24_sha256':r24_hash,'selected_dev':final,'initial_dev':base,
                    'accepted_model_promoted':False,'config':vars(args)},root/'reports/fa2_r25_unified_candidate.pth')
        report['candidate_sha256']=sha(root/'reports/fa2_r25_unified_candidate.pth')
    else:
        report['candidate_sha256']=sha(root/'reports/fa2_r25_unified_candidate.pth') if args.resume_r25 else None
    (root/'reports/r25_cpu_trial.json').write_text(json.dumps(report,indent=2)+'\n')
    print('R25_FINAL '+json.dumps({k:v for k,v in report.items() if k not in ('settings','development_initial','challenge_initial','historical_initial')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--steps',type=int,default=60)
    p.add_argument('--train-books',type=int,default=10);p.add_argument('--dev-books',type=int,default=6)
    p.add_argument('--challenge-books',type=int,default=6);p.add_argument('--test-books',type=int,default=4)
    p.add_argument('--chunk-span',type=int,default=20);p.add_argument('--windows',type=int,default=2)
    p.add_argument('--context-targets',default='768,4096');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch-windows',type=int,default=8);p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--donor-penalty',type=float,default=.05)
    p.add_argument('--resume-r25',action='store_true');p.add_argument('--lr',type=float,default=.004);run(p.parse_args())
