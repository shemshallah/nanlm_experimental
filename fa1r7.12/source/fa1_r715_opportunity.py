#!/usr/bin/env python3
"""Target-free historical exact 1/2-gram trust diagnostics for R7.15."""
import argparse,json,pathlib,time
import numpy as np,torch
from r712_validate import restore as restore_r712
from fa1_r715_mirror import prepare,wrong_history

def metrics(D,W,gmax=0.1,backoff=2,coverage=1):
    def one(dat):
        eq=dat['exact'] if backoff==1 else dat['exact2']
        n=eq.sum(1)
        vals=dat['val']
        valmatches=(vals==dat['target'][:,None]).float()
        p=(eq*valmatches).sum(1)/n.clamp_min(1)
        # target-free trust; never computed from target hits
        gate=gmax*(n>=coverage).float()
        nll=-torch.logaddexp(torch.log1p(-gate)-dat['base'],torch.log(gate.clamp_min(1e-30))+torch.log(p.clamp_min(1e-30)))
        return nll,p,gate
    lc,pc,gc=one(D);lw,pw,gw=one(W)
    return {'base_nll':round(float(D['base'].mean()),7), 'correct_nll':round(float(lc.mean()),7),
        'gain':round(float((D['base']-lc).mean()),7),'wrong_nll':round(float(lw.mean()),7),
        'correct_vs_wrong':round(float((lw-lc).mean()),7),
        'correct_retrieval_coverage':round(float((gc>0).float().mean()),4),
        'correct_hit_when_gated':round(float(((pc>0)&(gc>0)).float().sum()/((gc>0).float().sum().clamp_min(1))),4),
        'wrong_hit_when_gated':round(float(((pw>0)&(gw>0)).float().sum()/((gw>0).float().sum().clamp_min(1))),4)}

def run(args):
    torch.set_num_threads(args.threads);t=time.monotonic();root=pathlib.Path(args.root)
    model,_,base,comp,_,_=restore_r712(root,root/'adaptation/r712_gate_delta.pt');model.eval()
    results={}
    for split,n,seed in [('train',args.train,7151),('dev',args.dev,7152)]:
        D=prepare(root,model,base,comp,split,n,seed,k=args.topk,batch=args.batch)
        W=wrong_history(D,max(1,n//3),model.trunk.E,k=args.topk,batch=args.batch)
        cases={}
        for g in [0.005,0.02,0.05,0.10,0.20]:
            for b in [1,2]:
                for cov in [1,2]:
                    cases[f'g{g:.3f}_order{b}_min{cov}']=metrics(D,W,g,b,cov)
        results[split]={'n':n,'candidates':cases,'query_coverage_1gram':round(float((D['exact'].sum(1)>0).float().mean()),4),
           'query_coverage_2gram':round(float((D['exact2'].sum(1)>0).float().mean()),4)}
    out=root/'reports/FA1_R7p15_opportunity.json';out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'phase':'OPPORTUNITY','seconds':round(time.monotonic()-t,2),
        'train':results['train']['n'],'dev':results['dev']['n'],
        'dev_1gram':results['dev']['query_coverage_1gram'],
        'dev_2gram':results['dev']['query_coverage_2gram'],
        'best_dev_by_gain':sorted([(v['gain'],v['correct_vs_wrong'],k) for k,v in results['dev']['candidates'].items()],reverse=True)[:6],
        'best_dev_by_memory_control':sorted([(v['correct_vs_wrong'],v['gain'],k) for k,v in results['dev']['candidates'].items()],reverse=True)[:6]}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--train',type=int,default=256)
 p.add_argument('--dev',type=int,default=256);p.add_argument('--topk',type=int,default=96)
 p.add_argument('--batch',type=int,default=8);p.add_argument('--threads',type=int,default=2);run(p.parse_args())