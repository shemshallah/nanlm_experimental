#!/usr/bin/env python3
"""Resume FA2 R26 from its SINGLE checkpoint and existing AdamW histories.

Requires only original immutable R19 assets + R26 recovery overlay; no older
R24/R25 candidate file needed. Training selects on dev, never historical test.
"""
import argparse,copy,json,math,os,pathlib,random,time
import numpy as np,torch
from torch.nn import functional as F
from r26_restore_unified import restore,sha
from r26_unified_cpu import extract_split,metrics
from r13_corpus_cache import read_books_cached


def run(args):
    root=pathlib.Path(args.root).resolve()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    u,opt,comp,r6,meta,old=restore(root)
    previous=int(old['selected']['r26_step'])
    if args.steps<=previous:raise ValueError('steps must extend stored R26 step')
    for name,p in u.named_parameters():p.requires_grad_(name.startswith('r26.'))
    opt.param_groups[-1]['lr']=args.lr
    books,_=read_books_cached(root/'nanrouter',comp,r6.vocab,0)
    offsets=[int(x) for x in args.offsets.split(',')]
    train=extract_split(u,books['train'][:args.train_books],offsets,args.chunk_span,args.windows,'train_resume')
    dev=extract_split(u,books['dev'][:args.dev_books],offsets,args.chunk_span,args.windows,'dev_selection_resume')
    if not train or not dev:raise ValueError('no matched windows')
    u.eval();baseline=metrics(u,dev)
    orig_params={k:v.clone() for k,v in u.state_dict().items() if not k.startswith('r26.')}
    orig_opt=copy.deepcopy(old['selected']['optimizer'])
    best=(baseline['bpb'],previous,copy.deepcopy(u.r26.state_dict()),copy.deepcopy(opt.state_dict()))
    trainable=list(u.r26.parameters())
    for step in range(previous+1,args.steps+1):
        u.train();batch=random.sample(train,min(args.batch,len(train)))
        losses=[]
        for r in batch:
            correct=u.r26.target_logprob(r['base_y'],r['p_y'],r['features'])
            wrong=u.r26.target_logprob(r['wrong_y'],r['wp_y'],r['wfeatures'])
            losses.append(-correct.mean()+args.wrong_penalty*
                F.softplus(args.margin+wrong.mean()-correct.mean()))
        opt.zero_grad(set_to_none=True)
        loss=torch.stack(losses).mean();loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(trainable,2.))
        if not math.isfinite(grad):raise ValueError('nonfinite gradient')
        opt.step()
        if step%args.eval_every==0 or step==args.steps:
            u.eval();m=metrics(u,dev)
            print('R26_RESUME_EVAL',json.dumps({'step':step,'bpb':m['bpb'],
                'r25_bpb':m['r25_bpb'],'matched_donor_nats':m['correct_vs_frequency_matched_donor_nats']}),flush=True)
            if m['bpb']<best[0]-1e-6 and m['nonspace_nll']<=baseline['nonspace_nll']+1e-4 and m['correct_vs_frequency_matched_donor_nats']>0:
                best=(m['bpb'],step,copy.deepcopy(u.r26.state_dict()),copy.deepcopy(opt.state_dict()))
                print('R26_RESUME_SELECT',step,flush=True)
    u.r26.load_state_dict(best[2]);opt.load_state_dict(best[3]);u.eval()
    for k,v in orig_params.items():
        assert torch.equal(v,u.state_dict()[k]),'inherited tensor changed: '+k
    now=opt.state_dict()
    for a,b in zip(orig_opt['param_groups'][:10],now['param_groups'][:10]):
        assert a==b,'legacy optimizer hyperparameters changed'
    for k,vals in orig_opt['state'].items():
        if k in orig_opt['param_groups'][-1]['params']:continue
        for name,v in vals.items():
            w=now['state'][k][name]
            assert torch.equal(v,w) if torch.is_tensor(v) else v==w,(k,name)
    result=metrics(u,dev)
    report={'format':'FA2_R26_RESUME_V1','initial_step':previous,
        'selected_step':best[1],'attempted_step':args.steps,
        'initial_dev':baseline,'selected_dev':result,
        'inherited_adam_states_unchanged':130,'one_checkpoint':True,
        'accepted_model_promoted':False,'settings':vars(args)}
    path=root/'reports/fa2_r26_unified_candidate.pth'
    report['prior_checkpoint_sha256']=sha(path)
    if best[1]>previous:
        updated=dict(old)
        selected=dict(old['selected'])
        selected.update(model=u.state_dict(),optimizer=opt.state_dict(),r26_step=best[1])
        updated.update(selected=selected,selected_dev=result,settings=vars(args))
        tmp=path.with_suffix('.temp.pth')
        torch.save(updated,tmp);os.replace(tmp,path)
    report['selected_checkpoint_sha256']=sha(path)
    (root/'reports/r26_resume_trial.json').write_text(json.dumps(report,indent=2)+'\n')
    print('R26_RESUME_FINAL',json.dumps({'from':previous,'to':best[1],
        'r25_bpb':result['r25_bpb'],'selected_bpb':result['bpb'],
        'checkpoint_sha256':report['selected_checkpoint_sha256']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.')
    p.add_argument('--steps',type=int,default=110);p.add_argument('--lr',type=float,default=.004)
    p.add_argument('--seed',type=int,default=26026)
    p.add_argument('--train-books',type=int,default=12);p.add_argument('--dev-books',type=int,default=6)
    p.add_argument('--offsets',default='4096,8192,33000')
    p.add_argument('--chunk-span',type=int,default=20);p.add_argument('--windows',type=int,default=2)
    p.add_argument('--batch',type=int,default=12);p.add_argument('--eval-every',type=int,default=10)
    p.add_argument('--wrong-penalty',type=float,default=.025);p.add_argument('--margin',type=float,default=.01)
    run(p.parse_args())
