#!/usr/bin/env python3
import argparse,json,pathlib,time
import numpy as np,torch
from r79_normalized_memory import restore,eval_causal
from r77_semantic import restore as restore77
from eval_r77_fast import evaluate_positions
from r7_core import Corpus
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--count',type=int,default=512);p.add_argument('--out',type=pathlib.Path,required=True)
a=p.parse_args();torch.set_num_threads(2);t=time.monotonic();m,opts,b,c,prior=restore(a.root,a.candidate)
out={'step':prior['joint_steps'],'source':str(a.candidate),'same_original_model':True,'results':{}}
for spl in ('dev','test'):
  cor=Corpus(a.root,spl);positions=np.arange(a.count,dtype=np.int64)*(len(cor.seq)-1)//a.count
  result=evaluate_positions(m,b,c,cor,positions,batch=8)
  causal=eval_causal(m,b,c,cor,count=128,batch=8)
  out['results'][spl]={'matched':result,'causal_128':causal}
  print(json.dumps({'split':spl,'nll':result['active_nll'],'outer_utility':result['outer_gain'],
     'normal_gain':causal['normal_gain'],'normalized_utility_128':causal['normalized_advantage'],
     'correct_over_unrelated_128':causal['correct_over_unrelated'],'correct_over_reversed_128':causal['correct_over_reversed']}),flush=True)
out['wall_seconds']=round(time.monotonic()-t,2)
a.out.write_text(json.dumps(out,indent=2)+'\n')
