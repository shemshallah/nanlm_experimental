#!/usr/bin/env python3
"""Read-only paired historical comparison. Test split used only as reported science, not as training signal."""
import argparse,json,pathlib,time
import numpy as np,torch
from r77_semantic import restore as restore77
from r79_normalized_memory import restore as restore79, eval_causal
from eval_r77_fast import evaluate_positions
from r7_core import Corpus

a=argparse.ArgumentParser()
a.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
a.add_argument('--candidate',type=pathlib.Path,required=True)
a.add_argument('--baseline',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
a.add_argument('--count',type=int,default=512)
a.add_argument('--batch',type=int,default=8)
a.add_argument('--output',type=pathlib.Path)
args=a.parse_args();torch.set_num_threads(2)
t=time.monotonic();results={}
for spl in ('dev','test'):
    cor=Corpus(args.root,spl)
    positions=np.arange(args.count,dtype=np.int64)*(len(cor.seq)-1)//args.count
    m,opts,b,c,d=restore77(args.root,args.baseline,torch.device('cpu'))
    initial=evaluate_positions(m,b,c,cor,positions,batch=args.batch)
    del m,opts
    m,opts,b,c,d=restore77(args.root,args.candidate,torch.device('cpu'))
    cand=evaluate_positions(m,b,c,cor,positions,batch=args.batch)
    results[spl]={'accepted_14416':initial,'continued_14928':cand,
       'continued_advantage':initial['active_nll']-cand['active_nll'],
       'outer_utility':cand['outer_gain']}
    print(json.dumps({'split':spl,'accepted_nll':initial['active_nll'],'continued_nll':cand['active_nll'],
      'improvement':results[spl]['continued_advantage']}),flush=True)
    del m,opts
results['elapsed_seconds']=round(time.monotonic()-t,2)
if args.output:args.output.write_text(json.dumps(results,indent=2)+'\n')
