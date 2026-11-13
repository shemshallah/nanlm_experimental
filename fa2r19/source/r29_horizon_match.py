#!/usr/bin/env python3
"""FA2 R29: same-token/same-update matched 2,4,8-chunk truncated BPTT.

All arms continue the SAME inherited R28/R27 PAL weights and the SAME Adam
moments, not parallel deployable models. Every update processes identical
contiguous book spans (8 x 12-word chunks) and has ONE Adam step; the only
experimental difference is the detach boundary between microgroups.

Never uses test books to select; never modifies parent archives. Saves only
selected existing r28 adapter and its 9 optimizer states as a compact overlay.
"""
import argparse,copy,hashlib,json,math,pathlib,random,time,os
import numpy as np
import torch
from torch.nn import functional as F
from r28_restore_unified import restore
from r28_recurrent import bind_recurrent
from r28_cpu_continue import metrics,equal_nested,sha
from r13_corpus_cache import read_books_cached
from r23_stream_carry import ContinuousPALState,contiguous_book_picks


def make_groups(plans, offset, horizon, total):
    if total % horizon:
        raise ValueError('microgroup size must divide fixed training workload')
    return [plans[offset+j:offset+j+horizon] for j in range(0,total,horizon)]


def immutable_reference(model,opt,root):
    model_state={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('r28.')}
    optim=copy.deepcopy(opt.state_dict())
    return model_state,optim,sha(root/'reports/fa2_r27_unified_candidate.pth'),sha(root/'reports/fa2_r28_adapter_overlay.pt')


def audit(model,opt,immutable):
    parent,previous,r27_digest,r28_digest=immutable
    current=model.state_dict()
    if not all(torch.equal(v,current[k]) for k,v in parent.items()):
        raise AssertionError('inherited R27 tensors modified')
    now=opt.state_dict()
    if now['param_groups'][:12]!=previous['param_groups'][:12]:
        raise AssertionError('inherited optimizer groups changed')
    old_ids=[idx for group in previous['param_groups'][:12] for idx in group['params'] if idx in previous['state']]
    if not equal_nested({k:now['state'][k] for k in old_ids},{k:previous['state'][k] for k in old_ids}):
        raise AssertionError('inherited Adam moments modified')
    if len(opt.param_groups)!=13 or len(opt.state)!=153:
        raise AssertionError('R28 optimizer state count mutated')
    return {'inherited_tensor_count':len(parent),'inherited_adam_states':len(old_ids),
            'total_adam_states':len(opt.state),'r27_digest_unchanged':r27_digest,
            'r28_digest_unchanged':r28_digest}


def book_safe(baseline,current,slack=0.0):
    return all(current['books'].get(k,float('inf'))<=v+slack for k,v in baseline['books'].items())


def run(args):
    start=time.monotonic();root=pathlib.Path(args.root).resolve();torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    model,opt,comp,r6,meta,r27,r28=restore(root,threads=args.threads)
    if r28.get('r28_step')!=16 or len(r27['selected']['carry'])!=12:
        raise ValueError('starting checkpoint is not the verified R28 CPU16/R27 lineage')
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('r28.'))
    immutable=immutable_reference(model,opt,root)
    initial_model=copy.deepcopy(model.state_dict())
    initial_opt=copy.deepcopy(opt.state_dict())
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,args.max_books)
    training=books['train'][:args.train_books];development=books['dev'][:args.dev_books]
    if not training or not development or {b.book_id for b in training}&{b.book_id for b in development}:
        raise ValueError('training and development book groups overlap or empty')
    fixed_plans=[list(contiguous_book_picks(b,args.chunk_span)) for b in training]
    if min(map(len,fixed_plans))<args.steps*args.microchunks//len(training):
        raise ValueError('training book too short')
    with torch.no_grad():
        baseline=metrics(model,development,args.dev_chunks,args.chunk_span)
        saved_amplitude=model.r28.amplitude.detach().clone()
        model.r28.amplitude.zero_()
        r27_reference=metrics(model,development,args.dev_chunks,args.chunk_span)
        model.r28.amplitude.copy_(saved_amplitude)
    report={'format':'FA2_R29_MATCHED_HORIZON_V1','status':'unpromoted_experiment',
            'parent_r27_sha256':immutable[2], 'parent_r28_sha256':immutable[3],
            'r28_parent_step':16,'training_books':[str(b.book_id) for b in training],
            'development_books':[str(b.book_id) for b in development],
            'initial_dev':baseline,'r27_reference':r27_reference,'arms':[],'settings':vars(args)}
    print('R29_BASELINE '+json.dumps({'initial_dev':baseline,'training_books':report['training_books'],'development_books':report['development_books']}),flush=True)
    global_best=None
    for horizon in args.horizons:
        model.load_state_dict(initial_model,strict=True)
        opt.load_state_dict(initial_opt)
        if args.lr is not None:opt.param_groups[-1]['lr']=args.lr
        carriers=[bind_recurrent(ContinuousPALState(b.book_id),model) for b in training]
        pos=[0]*len(training)
        trajectory=[];best=None;seen=0
        print('R29_ARM_START '+json.dumps({'horizon':horizon,'updates':args.steps,'microchunks':args.microchunks}),flush=True)
        for step in range(1,args.steps+1):
            i=(step-1)%len(training);b=training[i];carry=carriers[i]
            groups=make_groups(fixed_plans[i],pos[i],horizon,args.microchunks)
            if len(groups)!=args.microchunks//horizon or any(len(g)!=horizon for g in groups):
                raise ValueError('incorrect matched training segment count')
            total_tokens=sum(seg[3]-seg[2] for group in groups for seg in group)
            weighted_tokens=sum(float((b.tokens[seg[2]:seg[3]]!=32).sum())+
                args.space_weight*float((b.tokens[seg[2]:seg[3]]==32).sum())
                for group in groups for seg in group)
            model.train();opt.zero_grad(set_to_none=True)
            nll=0.0
            for group in groups:
                losses=[]
                for seg in group:
                    row=model.forward_chunk(b,seg,carry)
                    if args.space_weight == 1.:
                        losses.append(F.cross_entropy(row['logits'],row['target'],reduction='sum'))
                    else:
                        per_token=F.cross_entropy(row['logits'],row['target'],reduction='none')
                        weights=torch.where(row['target']==32,args.space_weight,1.0)
                        losses.append((weights*per_token).sum())
                local=torch.stack(losses).sum()/weighted_tokens
                local.backward()
                nll+=float(local.detach())
                carry.detach() # only difference across 2,4,8 arms
            grad=float(torch.nn.utils.clip_grad_norm_(model.r28.parameters(),args.clip))
            if not math.isfinite(grad):raise ValueError('nonfinite R29 gradient')
            opt.step();pos[i]+=args.microchunks;seen+=total_tokens
            event={'step':step,'trained_tokens':total_tokens,'mean_nll':nll,'grad_norm':grad}
            if step%args.eval_every==0 or step==args.steps:
                model.eval();current=metrics(model,development,args.dev_chunks,args.chunk_span)
                admissible=(current['bpb']<baseline['bpb']-args.min_gain and
                    current['nonspace_nll']<=baseline['nonspace_nll']+args.nonspace_slack and
                    (not args.r27_gate or (current['nonspace_nll']<=r27_reference['nonspace_nll']+args.nonspace_slack and
                      current['bpb']<r27_reference['bpb']-args.min_gain)) and
                    book_safe(baseline,current,args.perbook_slack))
                event.update({'dev':current,'passes_guards':admissible})
                if admissible and (best is None or current['bpb']<best['metrics']['bpb']-args.min_gain):
                    best={'step':step,'metrics':current,
                        'r28_state':copy.deepcopy(model.r28.state_dict()),
                        'optimizer':copy.deepcopy(opt.state_dict())}
            trajectory.append(event)
            print('R29_STEP '+json.dumps({'horizon':horizon,**event}),flush=True)
        record={'horizon_chunks':horizon,'attempted_steps':args.steps,'tokens_seen':seen,'trajectory':trajectory,
            'best_step':best['step'] if best else 0,'best_dev':best['metrics'] if best else baseline}
        report['arms'].append(record)
        if best:
            model.r28.load_state_dict(best['r28_state']);opt.load_state_dict(best['optimizer'])
            record['audit']=audit(model,opt,immutable)
            if global_best is None or best['metrics']['bpb']<global_best['metrics']['bpb']-args.min_gain:
                global_best=dict(best,horizon=horizon)
        print('R29_ARM_DONE '+json.dumps({'horizon':horizon,'best_step':record['best_step'],
            'best_dev':record['best_dev'],'seconds_elapsed':round(time.monotonic()-start,2)}),flush=True)
    report['winner_horizon']=global_best['horizon'] if global_best else None
    report['winner_step']=global_best['step'] if global_best else 0
    report['selection_dev']=global_best['metrics'] if global_best else baseline
    report['elapsed_seconds']=time.monotonic()-start
    if global_best:
        model.r28.load_state_dict(global_best['r28_state']);opt.load_state_dict(global_best['optimizer'])
        report['audit']=audit(model,opt,immutable)
        original_group=initial_opt['param_groups'][-1]
        opt_state=opt.state_dict()
        new_ids=opt_state['param_groups'][-1]['params']
        if len(new_ids)!=9 or original_group['params']!=new_ids:raise AssertionError('R28 adapter group IDs changed')
        payload={'format':'FA2_R29_R28_CONTINUATION_DELTA_V1',
          'accepted_model_promoted':False,'parent_r27_sha256':immutable[2],
          'parent_r28_sha256':immutable[3],'r28_inherited_step':16,
          'additional_updates':global_best['step'],'effective_horizon_chunks':global_best['horizon'],
          'r28_state':{k:v.detach().clone() for k,v in model.r28.state_dict().items()},
          'r28_optimizer_group':copy.deepcopy(opt_state['param_groups'][-1]),
          'r28_optimizer_states':{k:copy.deepcopy(opt_state['state'][k]) for k in new_ids},
          'inherited_adam_states':144,'inherited_stream_count':12,'selection_dev':global_best['metrics']}
        path=root/f'reports/fa2_r29_{args.label}_adapter_overlay.pt';tmp=path.with_suffix('.tmp')
        torch.save(payload,tmp);os.replace(tmp,path)
        report['overlay_sha256']=sha(path)
        report['overlay_bytes']=path.stat().st_size
    dest=root/f'reports/r29_{args.label}_horizons.json';dest.write_text(json.dumps(report,indent=2)+'\n')
    print('R29_FINAL '+json.dumps({'winner':report['winner_horizon'],'step':report['winner_step'],
      'initial_bpb':baseline['bpb'],'selected_bpb':report['selection_dev']['bpb'],
      'elapsed_seconds':round(report['elapsed_seconds'],2)}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--steps',type=int,default=16);p.add_argument('--horizons',type=int,nargs='+',default=[2,4,8]);p.add_argument('--microchunks',type=int,default=8)
    p.add_argument('--train-books',type=int,default=2);p.add_argument('--dev-books',type=int,default=3);p.add_argument('--max-books',type=int,default=3)
    p.add_argument('--chunk-span',type=int,default=12);p.add_argument('--dev-chunks',type=int,default=20);p.add_argument('--eval-every',type=int,default=4)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=29029);p.add_argument('--lr',type=float,default=None)
    p.add_argument('--label',default='matched')
    p.add_argument('--space-weight',type=float,default=1.0)
    p.add_argument('--r27-gate',action='store_true')
    p.add_argument('--clip',type=float,default=2.);p.add_argument('--min-gain',type=float,default=1e-6)
    p.add_argument('--nonspace-slack',type=float,default=1e-4);p.add_argument('--perbook-slack',type=float,default=1e-4)
    a=p.parse_args()
    if not(0.05<=a.space_weight<=1.0):raise ValueError('bad space loss weight')
    run(a)
