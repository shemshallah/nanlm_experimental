#!/usr/bin/env python3
"""FA2.3 R22: CPU train *only* new 32k document-memory adapter.

Explicitly nonpromoting. Accepted R18 checkpoint / five optimizer histories and
frozen R21 gate remain untouched. Only train R22 414-param residual; development
selects, book-disjoint historical test is read AFTER selection. The R18 baseline
and R21 predecessor are always measured on identical windows.
"""
import argparse, copy, hashlib, json, math, pathlib, random, sys, time
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import r14_slim_base as pal
from r18_long_memory import R18LongMemory
from r13_corpus_cache import read_books_cached
from r11_raw import batch, windows
from r18_joint_continue import integrated_logits
from r21_relevance_novelty import RelevanceNoveltyGate, causal_proposal
from r22_multiscale_memory import R22Residual, hierarchical_proposals, BANDS


def build_cache(split, docs, choices, span, model, bridge, adapter, net, r21, cache_dir=None):
    caches=[]
    if cache_dir is not None: pathlib.Path(cache_dir).mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for i,pick in enumerate(choices):
            book=docs[pick[0]]; start=pick[1]
            t0=int(book.chunks[start])
            if cache_dir is not None:
                file=pathlib.Path(cache_dir)/f'{split}_{span}_{i:04d}.pth'
                if file.is_file():
                    caches.append({'cache_path':str(file),'split':split,
                                   'book_id':book.book_id,'token_offset':t0})
                    continue
            ex,lc,target,tags,bytes_=batch(docs,[pick],span)
            n=int((target[0]!=-100).sum()); y=target[0,:n].clone()
            assert np.array_equal(y.numpy(),book.tokens[t0:t0+n])
            base=integrated_logits(model,bridge,adapter,net,ex,lc,tags,target)[0,:n].detach().clone()
            r21_delta,r21_features,_=causal_proposal(net,base,y)
            r21_logits,_=r21(base,r21_delta,r21_features)
            new_delta,new_features,meta=hierarchical_proposals(net,r21_logits,book.tokens,t0)
            item={'split':split,'book_id':book.book_id,'token_offset':t0,
                          'y':y,'r18':base,'r21':r21_logits,'proposals':new_delta,
                          'features':new_features,'bytes':bytes_,'meta':meta,
                          'token_sha256':hashlib.sha256(book.tokens.tobytes()).hexdigest()}
            if cache_dir is None:
                caches.append(item)
            else:
                file=pathlib.Path(cache_dir)/f'{split}_{span}_{i:04d}.pth'
                torch.save(item,file)
                caches.append({'cache_path':str(file),'split':split,'book_id':book.book_id,
                               'token_offset':t0,'bytes':int(bytes_)})
                del item,base,r21_delta,r21_logits,new_delta,new_features
    return caches

def load_item(c):
    return torch.load(c['cache_path'],map_location='cpu',weights_only=False) if 'cache_path' in c else c


@torch.no_grad()
def evaluate(net, caches):
    net.eval()
    totals={kind:{'loss':0.,'other':0.,'bytes':0,'nonspace':0,'right':0,'count':0}
            for kind in ('r18','r21','r22')}
    covered={band:0 for band in BANDS}; max_distance={band:0 for band in BANDS}
    for item in caches:
        c=load_item(item)
        r22,_=net(c['r21'],c['proposals'],c['features'])
        y=c['y']; mask=y!=32
        for kind,out in (('r18',c['r18']),('r21',c['r21']),('r22',r22)):
            row=totals[kind];ce=F.cross_entropy(out,y,reduction='none')
            row['loss']+=float(ce.sum());row['other']+=float(ce[mask].sum());
            row['bytes']+=c['bytes'];row['nonspace']+=int(mask.sum());
            row['right']+=int(((out.argmax(-1)==y)&mask).sum());row['count']+=len(y)
        for band in BANDS:
            covered[band]+=c['meta'][band]['eligible']
            max_distance[band]=max(max_distance[band],c['meta'][band]['maximum_read_distance'])
    return {'scores':{k:{'bpb':v['loss']/v['bytes']/math.log(2),
                         'nonspace_nll':v['other']/max(1,v['nonspace']),
                         'nonspace_top1':v['right']/max(1,v['nonspace']),
                         'tokens':v['count'],'bytes':v['bytes']} for k,v in totals.items()},
            'query_coverage':covered,'maximum_retrieval_distance':max_distance,'windows':len(caches)}


def main(args):
    t0=time.perf_counter();root=pathlib.Path(args.root).resolve();torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    checkpoint=root/'checkpoint/r12_active.pal12';orig=pal.sha(checkpoint)
    r21_path=root/'reports/fa2_2_r21_gate_candidate.pth'
    r21_hash=pal.sha(r21_path)
    state=torch.load(r21_path,map_location='cpu',weights_only=False)
    assert state['parent_checkpoint_sha256']==orig and state['step']==1000
    r21=RelevanceNoveltyGate();r21.load_state_dict(state['gate']);r21.eval().requires_grad_(False)
    saved,r6,comp,model,bridge,adapter,opt,_=pal.load_for_arm(root,checkpoint,True)
    assert len(opt.param_groups)==4 and 'r18_memory' in saved
    net=R18LongMemory(model.memory.piece_emb.weight);net.load_state_dict(saved['r18_memory'])
    for x in (model,bridge,adapter,net):x.eval().requires_grad_(False)
    docs,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    spl=windows(docs,args.span,max(96,args.train_windows,args.dev_windows,args.test_windows),args.max_tokens)
    cache_dir=root/'.cache'/f'r22_s{args.span}_seed{args.seed}'
    train=build_cache('train',docs['train'],spl['train'][:args.train_windows],args.span,model,bridge,adapter,net,r21,cache_dir=cache_dir/'train')
    dev=build_cache('dev',docs['dev'],spl['dev'][:args.dev_windows],args.span,model,bridge,adapter,net,r21,cache_dir=cache_dir/'dev')
    short=windows(docs,24,8000,180)
    short_dev=build_cache('dev',docs['dev'],short['dev'][:args.short_windows],24,model,bridge,adapter,net,r21,cache_dir=cache_dir/'short_dev')
    model22=R22Residual();optimizer=torch.optim.AdamW(model22.parameters(),lr=args.lr,weight_decay=1e-4)
    resume_step=0
    if args.resume_state:
        prior=torch.load(args.resume_state,map_location='cpu',weights_only=False)
        assert prior['parent_checkpoint_sha256']==orig and prior['frozen_r21_gate_sha256']==r21_hash
        assert not prior['accepted_model_promoted'] and 0<prior['steps']<args.steps
        model22.load_state_dict(prior['model']);optimizer.load_state_dict(prior['optimizer'])
        resume_step=prior['steps']
    initial={'dev':evaluate(model22,dev),'short_dev':evaluate(model22,short_dev)}
    # No historical test split used for architecture or threshold decisions.
    best=copy.deepcopy(model22.state_dict());bestopt=copy.deepcopy(optimizer.state_dict())
    beststep=resume_step
    existing=initial['dev']['scores']['r22']
    baseline=initial['dev']['scores']['r21']
    bestkey=min(baseline['bpb'],existing['bpb']+0.04*max(0.,existing['nonspace_nll']-baseline['nonspace_nll']))
    trace=[]
    for step in range(resume_step+1,args.steps+1):
        model22.train()
        indices=[(step*7 + k*13 + args.seed)%len(train) for k in range(args.batch)]
        losses=[]
        for i in indices:
            c=load_item(train[i]);new,_=model22(c['r21'],c['proposals'],c['features'])
            # The LR-regularized residual must help non-space data, not only
            # whitespace; short windows remain exactly unchanged by design.
            losses.append(F.cross_entropy(new,c['y']))
        loss=torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model22.parameters(),5.);optimizer.step()
        if step%args.eval_every==0 or step==args.steps:
            devscores=evaluate(model22,dev)['scores']
            shortscore=evaluate(model22,short_dev)['scores']
            r=devscores['r22'];base=devscores['r21']
            objective=r['bpb']+0.04*max(0.,r['nonspace_nll']-base['nonspace_nll'])
            short_ok=(abs(shortscore['r22']['bpb']-shortscore['r21']['bpb'])<1e-10 and
                      abs(shortscore['r22']['nonspace_nll']-shortscore['r21']['nonspace_nll'])<1e-10)
            if short_ok and objective<bestkey-1e-9:
                bestkey=objective;best=copy.deepcopy(model22.state_dict())
                bestopt=copy.deepcopy(optimizer.state_dict());beststep=step
            trace.append({'step':step,'train_ce':float(loss.detach()),
                          'dev_bpb':r['bpb'],'dev_nonspace_nll':r['nonspace_nll'],
                          'short_context_safe':short_ok,
                          'amplitude':{b:float(model22.branches[b].amplitude.detach()) for b in BANDS}})
            print(json.dumps({'phase':'DEV','step':step,'dev_bpb':r['bpb'],
                              'r21_dev_bpb':base['bpb'],'best_step':beststep,
                              'short_safe':short_ok}),flush=True)
    model22.load_state_dict(best);optimizer.load_state_dict(bestopt);model22.eval()
    chosen={'dev':evaluate(model22,dev),'short_dev':evaluate(model22,short_dev)}
    save=root/'reports/r22_multiscale_candidate.pth'
    torch.save({'parent_checkpoint_sha256':orig,'frozen_r21_gate_sha256':r21_hash,
                'model':best,'optimizer':bestopt,'steps':beststep,
                'attempted_steps':args.steps,'accepted_model_promoted':False},save)
    print(json.dumps({'phase':'SAVED_BEST_DEV','step':beststep,'candidate_file':str(save),
                      'dev_bpb':chosen['dev']['scores']['r22']['bpb']}),flush=True)
    # Historical test is read ONCE only after all selection is frozen.
    test=build_cache('test',docs['test'],spl['test'][:args.test_windows],args.span,model,bridge,adapter,net,r21,cache_dir=cache_dir/'test')
    short_test=build_cache('test',docs['test'],short['test'][:args.short_windows],24,model,bridge,adapter,net,r21,cache_dir=cache_dir/'short_test')
    historical={'long':evaluate(model22,test),'short':evaluate(model22,short_test)}
    assert pal.sha(checkpoint)==orig and pal.sha(r21_path)==r21_hash
    save=root/'reports/r22_multiscale_candidate.pth'
    torch.save({'parent_checkpoint_sha256':orig,'frozen_r21_gate_sha256':r21_hash,
                'model':best,'optimizer':bestopt,'steps':beststep,
                'attempted_steps':args.steps,'accepted_model_promoted':False},save)
    report={'round':'FA2.3 R22 32k-addressed multiscale retrieval, adapter only',
            'parent_checkpoint_sha256':orig,'frozen_r21_gate_sha256':r21_hash,
            'new_parameters':sum(x.numel() for x in model22.parameters()),
            'original_checkpoint_unchanged':True,'r21_unchanged':True,'promoted':False,
            'train_windows':len(train),'dev_windows':len(dev),'historical_test_windows':len(test),
            'steps_attempted':args.steps,'resumed_from_step':resume_step,'best_step_dev':beststep,
            'initial':initial,'selected':chosen,'historical_test':historical,
            'train_book_ids':sorted(set(x['book_id'] for x in train)),
            'dev_book_ids':sorted(set(x['book_id'] for x in dev)),
            'test_book_ids':sorted(set(x['book_id'] for x in test)),
            'trace':trace,'seconds':time.perf_counter()-t0,
            'limitations':['Historical test books already consulted; not pristine new-book evaluation',
                           '32k-addressed sparse retrieval, not 32k autoregressive training',
                           'Byte-pair similarity and exact-prefix rescue are not semantic comprehension',
                           'No blind paraphrase or multi-hop improvement shown',
                           'No accepted weight or inherited optimizer changed']}
    (root/'reports/r22_multiscale_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','best_step':beststep,'dev':chosen['dev']['scores'],
                      'historical':historical['long']['scores'],'promoted':False,
                      'seconds':report['seconds']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.')
    ap.add_argument('--span',type=int,default=120);ap.add_argument('--max-tokens',type=int,default=700)
    ap.add_argument('--train-windows',type=int,default=20);ap.add_argument('--dev-windows',type=int,default=8)
    ap.add_argument('--test-windows',type=int,default=8);ap.add_argument('--short-windows',type=int,default=8)
    ap.add_argument('--steps',type=int,default=250);ap.add_argument('--eval-every',type=int,default=25)
    ap.add_argument('--batch',type=int,default=2);ap.add_argument('--lr',type=float,default=.003)
    ap.add_argument('--threads',type=int,default=2);ap.add_argument('--seed',type=int,default=22022)
    ap.add_argument('--resume-state',default='')
    main(ap.parse_args())
