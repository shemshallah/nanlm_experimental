#!/usr/bin/env python3
"""Frozen native v176 third-code word-start probability diagnostic.

Each word target's true first letter comes from the original word label; its
preceding byte history is *reconstructed* from 48 word IDs (not original book
punctuation). This is a word-start-only surrogate, NOT contiguous byte BPB.
Alpha selected VAL, final on already reused TEST; inherited v170 byte law
remains fully reversible and default unchanged.
"""
from __future__ import annotations
import sys,math,time,json,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from bench_native176 import init_lib,prefix
from train_long175 import load_models,load_split,verify_corpus

ALPHAS=[0.,.05,.1,.2,.35,.5,.7,1.]

def run(split,names,source,num,seed=176):
 X,Y,B=load_split(split,source)
 rng=np.random.default_rng(seed)
 # Balanced across books; sampling without target-dependent selection.
 groups={int(k):np.flatnonzero(B==k) for k in np.unique(B)}
 ids=[];maxeach=max(1,math.ceil(num/len(groups)))
 for k in sorted(groups):
  arr=groups[k];ids+=rng.choice(arr,min(maxeach,len(arr)),replace=False).tolist()
 idx=rng.choice(ids,size=min(num,len(ids)),replace=False)
 l,h=init_lib(full=True)
 data=[]
 try:
  for j in idx:
   hist=X[j];raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in hist)+' ').encode('ascii')
   assert l.wf176_hybrid_begin(h,__import__('ctypes').c_char_p(raw),len(raw))==0
   q,p=prefix(l,h,b'');truth=ord(names[int(Y[j])][0]);assert 97<=truth<=122
   data.append((int(B[j]),truth,float(p[truth]),float(q[truth]),int(np.argmax(p)==truth),int(np.argmax(q)==truth)))
 finally:l.wf176_hybrid_close(h)
 out={'split':split,'n':len(data),'books':len(groups),'alpha_metrics':{},'surrogate_byte_context':True,
       'word_firstletter_target':'real label','prior_contiguous_byte_bpb':False}
 for alpha in ALPHAS:
  loss=sum(-math.log(max(1e-12,(1-alpha)*r[2]+alpha*r[3])) for r in data)/len(data)
  out['alpha_metrics'][str(alpha)]={'nll_nats_per_firstletter':loss,'bits_per_firstletter':loss/math.log(2)}
 out['firstletter_byte_baseline_top1']=sum(r[4] for r in data)
 out['firstletter_word_top5_proposal_top1']=sum(r[5] for r in data)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--val',type=int,default=280);p.add_argument('--test',type=int,default=220)
 args=p.parse_args();start=time.monotonic()
 import torch;torch.set_num_threads(2)
 m,o,names,fp,old=load_models();source=ROOT/'results/v175_release3';verify_corpus(source,names)
 val=run('val',names,source,args.val)
 best=min(ALPHAS,key=lambda a:val['alpha_metrics'][str(a)]['nll_nats_per_firstletter'])
 test=run('test',names,source,args.test,177)
 out={'name':'v176 word top5 speculative first-letter entropy under native byte oracle',
       'validation':val,'selected_mix_alpha':best,'reused_test':test,
       'test_previously_exposed':True,'status':'diagnostic not production',
       'source':'NaNROUTER Release0003 real next-word labels + reconstructed 48-word raw histories',
       'elapsed_s':time.monotonic()-start}
 (ROOT/'results/v176_wordstart_byte_coding.json').write_text(json.dumps(out,indent=2))
 print('V176 WORDSTART BYTE',json.dumps({k:v for k,v in out.items() if k!='reused_test'}),flush=True)
 print('V176 REUSED TEST',test,flush=True)

if __name__=='__main__':main()