#!/usr/bin/env python3
"""FA2 R26: single-checkpoint CPU continuation from accepted R25 step120.

No recurrence/optimizer state reset: append ONE new optimizer group. Train an
R26 high-order causal memory adapter on frozen, exact R25 model probabilities.
An unconsulted group of old historical test books is only evaluated AFTER
selection. It is not an independent new-author validation.
"""
import argparse,copy,hashlib,json,math,pathlib,random,time,gc
import numpy as np,torch
from torch.nn import functional as F
from r25_unified_cpu import initialize_compatible,extract,metrics as r25_metrics
from r25_episodic_cache import AdaptiveNGramMixer,CausalNGramIndex
from r13_corpus_cache import read_books_cached
from r26_span_memory import CausalHighOrderIndex,AdaptiveSpanMixer,CHANNELS


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()


def prepare(root,threads=2,lr=.012):
    torch.set_num_threads(threads)
    u,opt,comp,r6,meta=initialize_compatible(root)
    p=root/'reports/fa2_r25_unified_candidate.pth'
    incoming=torch.load(p,map_location='cpu',weights_only=False)
    if incoming['format']!='FA2_R25_UNIFIED_CANDIDATE_V1' or incoming['selected']['r25_step']!=120 or incoming['accepted_model_promoted']:
        raise ValueError('cannot migrate unexpected R25 parent')
    if meta['parent_sha256']!=incoming['parent']['parent_sha256']:
        raise ValueError('accepted R18 ancestor changed')
    u.add_module('r25',AdaptiveNGramMixer())
    opt.add_param_group({'params':list(u.r25.parameters()),'lr':.004,'weight_decay':1e-4})
    u.load_state_dict(incoming['selected']['model'],strict=True)
    opt.load_state_dict(incoming['selected']['optimizer'])
    if len(opt.param_groups)!=10 or len(opt.state)!=130 or len(incoming['selected']['carry'])!=12:
        raise ValueError('R25 checkpoint/optimizer not intact')
    u.add_module('r26',AdaptiveSpanMixer())
    opt.add_param_group({'params':list(u.r26.parameters()),'lr':lr,'weight_decay':1e-4})
    for name,p in u.named_parameters():
        p.requires_grad_(name.startswith('r26.'))
    return u,opt,comp,r6,meta,incoming


def freq_matched_donor(book,books):
    a=np.bincount(book.tokens[:4096],minlength=5000).astype(np.float64)+.1
    a/=a.sum()
    ds=[]
    for j,x in enumerate(books):
        if x.book_id==book.book_id:continue
        b=np.bincount(x.tokens[:4096],minlength=5000).astype(np.float64)+.1
        b/=b.sum();ds.append((float(np.square(np.sqrt(a)-np.sqrt(b)).sum()),j))
    if not ds:raise ValueError('at least 2 books for donor control')
    return books[min(ds)[1]],min(ds)[0]


def extract_split(u,books,offsets,span,windows,name):
    results=[];start=time.monotonic()
    for b in books:
        d,dist=freq_matched_donor(b,books)
        # Every retrieval must precede its query position. Indexing a 160K
        # whole-book suffix when queries stop at 8K wastes CPU and RAM without
        # changing a single eligible candidate. Safety margin covers all
        # selected contiguous word chunks after the target offset.
        cap=max(offsets)+4096
        source_prefix=b.tokens[:cap]
        donor_prefix=d.tokens[:cap]
        baseix=CausalNGramIndex(source_prefix)
        donorix=CausalNGramIndex(donor_prefix)
        raw=extract(u,b,baseix,donorix,offsets,span,windows)
        high=CausalHighOrderIndex(source_prefix)
        wrong=CausalHighOrderIndex(donor_prefix)
        with torch.no_grad():
            for row in raw:
                r25=u.r25(row['base'],row['cand'],row['feat'])
                r25_wrong=u.r25(row['base'],row['wrong'],row['wfeat'])
                targ=row['target'].long()
                t0=row['start'];nt=len(targ)
                idx=torch.arange(nt)
                p_y,feats=high.features(b.tokens,t0,nt,r25.exp().numpy(),targ.numpy())
                cutoff=min(t0,len(d.tokens)-nt)
                wp_y,wfeats=wrong.features(b.tokens,t0,nt,r25_wrong.exp().numpy(),targ.numpy(),source_cutoff=cutoff)
                results.append({'base_y':r25[idx,targ].detach().clone(),
                    'wrong_y':r25_wrong[idx,targ].detach().clone(),
                    'p_y':torch.from_numpy(p_y),'features':torch.from_numpy(feats),
                    'wp_y':torch.from_numpy(wp_y),'wfeatures':torch.from_numpy(wfeats),
                    'targets':targ,'byte_count':row['byte_count'],
                    'book_id':str(b.book_id),'start':int(t0),'donor_id':str(d.book_id),
                    'frequency_distance':dist})
        print('EXTRACT',json.dumps({'split':name,'book_id':str(b.book_id),'donor':str(d.book_id),
            'windows':len(raw),'tokens':sum(len(x['target']) for x in raw),
            'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
        del baseix,donorix,high,wrong,raw;gc.collect()
    return results


def metrics(model,items):
    sums={'loss':0.,'baseline':0.,'wrong':0.,'nonspace':0.,'bytes':0,'tokens':0,'nonspace_n':0}
    by={};coverage=[]
    with torch.no_grad():
        for r in items:
            log=model.r26.target_logprob(r['base_y'],r['p_y'],r['features'])
            wrong=model.r26.target_logprob(r['wrong_y'],r['wp_y'],r['wfeatures'])
            sums['loss']-=float(log.sum());sums['baseline']-=float(r['base_y'].sum())
            sums['wrong']-=float(wrong.sum())
            sums['bytes']+=r['byte_count'];sums['tokens']+=len(log)
            nonspace=r['targets'].ne(32);sums['nonspace']-=float(log[nonspace].sum());sums['nonspace_n']+=int(nonspace.sum())
            v=by.setdefault(r['book_id'],[0.,0.,0.,0,0]);v[0]-=float(log.sum());v[1]-=float(r['base_y'].sum());v[2]-=float(wrong.sum());v[3]+=len(log);v[4]+=r['byte_count']
            coverage.append(float(r['features'][:,:,0].gt(0).any(-1).float().mean()))
    z={
        'bpb':sums['loss']/(sums['bytes']*math.log(2)),
        'r25_bpb':sums['baseline']/(sums['bytes']*math.log(2)),
        'r26_gain_bpb':(sums['baseline']-sums['loss'])/(sums['bytes']*math.log(2)),
        'correct_vs_frequency_matched_donor_nats':(sums['wrong']-sums['loss'])/sums['tokens'],
        'nonspace_nll':sums['nonspace']/sums['nonspace_n'],
        'tokens':sums['tokens'],'bytes':sums['bytes'],'books':len(by),'windows':len(items),
        'coverage_fraction':float(np.mean(coverage)),
        'book_improvements':sum(v[0]<v[1] for v in by.values()),
        'correct_vs_donor_book_wins':sum(v[0]<v[2] for v in by.values()),
        'per_book':[{'book':k,'tokens':v[3],'r25_bpb':v[1]/v[4]/math.log(2),
            'r26_bpb':v[0]/v[4]/math.log(2),
            'correct_vs_wrong_nats':(v[2]-v[0])/v[3]} for k,v in by.items()]}
    return z


def assert_inheritance(saved,current,selected_step):
    orig=saved['selected']['model'];updated=current.state_dict()
    for k,v in orig.items():
        if not torch.equal(updated[k].cpu(),v.cpu()):
            raise ValueError('legacy model changed: '+k)
    # Every inherited Adam tensor and scalar must match *bitwise*.
    inherited=saved['selected']['optimizer']
    now=current._r26_optimizer.state_dict() if hasattr(current,'_r26_optimizer') else None
    if now is None:
        raise ValueError('optimizer not attached for lineage comparison')
    if len(now['param_groups'])!=11 or len(inherited['param_groups'])!=10:
        raise ValueError('optimizer group count changed unexpectedly')
    for ga,gb in zip(inherited['param_groups'],now['param_groups'][:10]):
        if ga!=gb:raise ValueError('inherited Adam hyperparameters/IDs changed')
    for k,original in inherited['state'].items():
        if k not in now['state']:raise ValueError('missing inherited Adam state')
        new=now['state'][k]
        if original.keys()!=new.keys():raise ValueError('Adam state structure changed')
        for name,a in original.items():
            b=new[name]
            if torch.is_tensor(a):
                if not torch.equal(a,b):raise ValueError('inherited Adam moment mutated: '+str((k,name)))
            elif a!=b:raise ValueError('inherited Adam scalar mutated')
    return {'inherited_tensors':len(orig),'inherited_optimizer_states':len(inherited['state']),
            'r25_step':saved['selected']['r25_step'],'r26_step':selected_step}


def run(args):
    started=time.monotonic()
    root=pathlib.Path(args.root).resolve()
    torch.manual_seed(26);np.random.seed(26);random.seed(26)
    u,opt,comp,r6,meta,old=prepare(root,args.threads,args.lr)
    parent_sha=sha(root/'reports/fa2_r25_unified_candidate.pth')
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    offsets=[int(x) for x in args.offsets.split(',')]
    u.eval()
    train=extract_split(u,books['train'][:args.train_books],offsets,args.chunk_span,args.windows,'train')
    dev=extract_split(u,books['dev'][:args.dev_books],offsets,args.chunk_span,args.windows,'dev_selection')
    reserved=extract_split(u,books['dev'][args.dev_books:args.dev_books+args.reserve_books],offsets,args.chunk_span,args.windows,'reserved_dev')
    baseline=metrics(u,dev)
    before_reserve=metrics(u,reserved)
    print('R26_BASE',json.dumps({'dev':{k:v for k,v in baseline.items() if k!='per_book'},
                                'reserve':{k:v for k,v in before_reserve.items() if k!='per_book'}}),flush=True)
    if not train or not dev:raise ValueError('no matching windows')
    param=list(u.r26.parameters())
    best=(baseline['bpb'],0,copy.deepcopy(u.r26.state_dict()),copy.deepcopy(opt.state_dict()))
    for step in range(1,args.steps+1):
        u.train();batch=random.sample(train,min(len(train),args.batch))
        errs=[]
        for row in batch:
            p=u.r26.target_logprob(row['base_y'],row['p_y'],row['features'])
            wrong=u.r26.target_logprob(row['wrong_y'],row['wp_y'],row['wfeatures'])
            loss=-p.mean()
            if args.wrong_penalty:
                loss+=args.wrong_penalty*F.softplus(args.margin +wrong.mean()-p.mean())
            errs.append(loss)
        opt.zero_grad(set_to_none=True)
        loss=torch.stack(errs).mean();loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(param,2.))
        if not math.isfinite(grad):raise ValueError('nonfinite gradients')
        opt.step()
        if step%args.eval_every==0 or step==args.steps:
            u.eval();m=metrics(u,dev)
            print('R26_EVAL',json.dumps({'step':step,'loss':float(loss.detach()),'grad':grad,
                     'bpb':m['bpb'],'r25_bpb':m['r25_bpb'],'gain_bpb':m['r26_gain_bpb'],
                     'nonspace':m['nonspace_nll'],'correct_vs_donor':m['correct_vs_frequency_matched_donor_nats'],
                     'improved_books':m['book_improvements']}),flush=True)
            if (m['bpb']<best[0]-1e-6 and m['nonspace_nll']<=baseline['nonspace_nll']+1e-4
                    and m['correct_vs_frequency_matched_donor_nats']>0):
                best=(m['bpb'],step,copy.deepcopy(u.r26.state_dict()),copy.deepcopy(opt.state_dict()))
                print('R26_SELECT',step,flush=True)
    u.r26.load_state_dict(best[2]);opt.load_state_dict(best[3]);u.eval()
    selected=metrics(u,dev);extra=metrics(u,reserved)
    print('R26_LOCKED',json.dumps({'selected_step':best[1],
       'dev_bpb':selected['bpb'],'reserved_bpb':extra['bpb'],
       'reserved_r25':extra['r25_bpb'],'reserved_delta':extra['r26_gain_bpb'],
       'reserved_books_improved':extra['book_improvements']}),flush=True)
    # Only after the selection is locked is any historical test data used.
    fresh=books['test'][args.test_offset:args.test_offset+args.test_books]
    untuned=extract_split(u,fresh,offsets,args.chunk_span,args.windows,'once_only_unused_test')
    test=metrics(u,untuned)
    if len(opt.state)<130:raise ValueError('lost inherited optimizer state')
    # The parameter registration is unchanged; store the already-loaded
    # optimizer for exact moments verification without serializing it twice.
    u._r26_optimizer=opt
    ancestor=assert_inheritance(old,u,best[1])
    del u._r26_optimizer
    snap={'format':'FA2_R26_UNIFIED_CANDIDATE_V1',
          'selected':{**{k:v for k,v in old['selected'].items() if k not in ('model','optimizer')},
                      'model':u.state_dict(),'optimizer':opt.state_dict(),'r26_step':best[1]},
          'parent':old['parent'],'r24_sha256':old['r24_sha256'],
          'r25_sha256':parent_sha,'r18_parent_sha256':meta['parent_sha256'],
          'selected_dev':selected,'reserved_dev':extra,'historical_test':test,
          'accepted_model_promoted':False,'settings':vars(args)}
    report={'format':snap['format'],'accepted_model_promoted':False,
            'parent_r25_sha256':parent_sha,'attempted_updates':args.steps,
            'selected_updates':best[1],'optimizer_groups':len(opt.param_groups),
            'optimizer_states':len(opt.state),'preserved_r25_legacy_moments':130,
            'inherited_weights':ancestor,'development_r25':baseline,
            'development_r26':selected,'reserved_r25':before_reserve,
            'reserved_r26':extra,'previously_unused_test':test,
            'elapsed_s':time.monotonic()-started,'settings':vars(args),
            'scope_note':'Reserved books are previously used in overall PAL program; historical unseen subset is R25-unconsulted only. No semantic QA or backbone scaling measured.'}
    ck=root/'reports/fa2_r26_unified_candidate.pth'
    torch.save(snap,ck)
    report['candidate_sha256']=sha(ck)
    (root/'reports/r26_cpu_trial.json').write_text(json.dumps(report,indent=2)+'\n')
    print('R26_FINAL',json.dumps({'step':best[1],'dev':selected['bpb'],
        'reserve':extra['bpb'],'historical_r26':test['bpb'],
        'historical_r25':test['r25_bpb'],'candidate_sha256':report['candidate_sha256']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='.')
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--lr',type=float,default=.012)
    p.add_argument('--train-books',type=int,default=12)
    p.add_argument('--dev-books',type=int,default=6)
    p.add_argument('--reserve-books',type=int,default=6)
    p.add_argument('--test-offset',type=int,default=4)
    p.add_argument('--test-books',type=int,default=8)
    p.add_argument('--offsets',default='4096,8192,33000')
    p.add_argument('--chunk-span',type=int,default=20)
    p.add_argument('--windows',type=int,default=2)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=12)
    p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--wrong-penalty',type=float,default=.025)
    p.add_argument('--margin',type=float,default=.01)
    run(p.parse_args())
