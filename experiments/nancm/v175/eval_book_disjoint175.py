#!/usr/bin/env python3
"""Frozen v174 12/48-word vs continued v175 +/- pointer on Release3 book splits.

Fresh book-disjoint to new v175 weights, test was first used after step280 pilot;
later test runs must be labeled reused. Bootstrap resamples *whole books*.
"""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_long175 import load_models,load_split,verify_corpus

def eval_split(base,latest,cohort,source,batch=72):
 X,Y,B=load_split(cohort,source)
 agg=defaultdict(lambda:{'n':0,'old12_loss':0.,'old48_loss':0.,'new_noecho_loss':0.,'new_echo_loss':0.,
                     'old12_top1':0,'old48_top1':0,'new_noecho_top1':0,'new_echo_top1':0,
                     'old12_top5':0,'old48_top5':0,'new_noecho_top5':0,'new_echo_top5':0})
 base.eval();latest.eval()
 with torch.no_grad():
  for start in range(0,len(Y),batch):
   x=torch.from_numpy(X[start:start+batch]);y=torch.from_numpy(Y[start:start+batch]);bk=B[start:start+batch]
   base.echo_enabled=False
   z12=base(x[:,-12:]);z48=base(x)
   latest.echo_enabled=True
   new,without,_=latest(x,True)
   for key,z in [('old12',z12),('old48',z48),('new_noecho',without),('new_echo',new)]:
    l=F.cross_entropy(z,y,reduction='none').numpy();top=z.topk(5,dim=1).indices.numpy();yy=y.numpy()
    for i in range(len(y)):
     a=agg[int(bk[i])];a[key+'_loss']+=float(l[i]);a[key+'_top1']+=int(top[i,0]==yy[i]);a[key+'_top5']+=int(yy[i] in top[i])
   for k in bk:agg[int(k)]['n']+=1
 metrics=defaultdict(float);metrics['n']=len(Y);metrics['books']=len(agg)
 for a in agg.values():
  for k,v in a.items():
   if k!='n':metrics[k]+=v
 for key in ('old12','old48','new_noecho','new_echo'):
  metrics[key+'_nll']=metrics.pop(key+'_loss')/len(Y)
 return dict(metrics),{str(i):j for i,j in agg.items()}

def bootstrap(agg,seed=175,reps=3000):
 arr=list(agg.values());rng=np.random.default_rng(seed)
 vals=[]
 for _ in range(reps):
  ss=[arr[i] for i in rng.integers(len(arr),size=len(arr))]
  n=sum(x['n'] for x in ss)
  old=sum(x['old12_loss'] for x in ss)/n
  new=sum(x['new_echo_loss'] for x in ss)/n
  top=(sum(x['new_echo_top1'] for x in ss)-sum(x['old12_top1'] for x in ss))/n
  echo=(sum(x['new_noecho_loss'] for x in ss)-sum(x['new_echo_loss'] for x in ss))/n
  vals.append([old-new,top,echo])
 z=np.asarray(vals)
 return {'parent12_to_v175_nll_gain_95':list(map(float,np.quantile(z[:,0],[.025,.5,.975]))),
         'parent12_to_v175_top1_rate_gain_95':list(map(float,np.quantile(z[:,1],[.025,.5,.975]))),
         'echo_ablation_nll_gain_95':list(map(float,np.quantile(z[:,2],[.025,.5,.975]))),
         'bootstrap_reps':reps,'unit':'book'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--reps',type=int,default=3000)
 args=p.parse_args();torch.set_num_threads(2)
 frozen,_,names,fp,old=load_models();latest,_,nn,ff,_=load_models()
 ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',map_location='cpu',weights_only=False)
 assert ck['word_order']==names and tuple(ck['fingerprints'])==tuple(fp)
 latest.load_state_dict(ck['model_state_dict'])
 source=ROOT/'results/v175_release3';man=verify_corpus(source,names)
 assert len(man['books'])==500 and man['release_tag']=='corpora-short-stories-0003'
 metrics,books=eval_split(frozen,latest,args.split,source)
 result={'version':'v175','split':args.split,'status':'diagnostic_not_promoted',
         'test_previously_seen_after_step280':args.split=='test',
         'prior_architecture':'original v174 step800 full teacher+signed GRU 12 word history',
         'new_architecture':'48 word GRU plus causal next-token episodic pointer',
         'metrics':metrics,'book_block_bootstrap':bootstrap(books,reps=args.reps),
         'books':books,'source_release_sha256':man['archive_sha256'],'independent_v175_book_split':man['disjoint_checked'],
         'caveat':'Inherited older parent can have earlier text overlap not fully known; strict in-vocabulary 12k word accuracy only.'}
 path=ROOT/f'results/v175_{args.split}_frozen_comparison.json';path.write_text(json.dumps(result,indent=2))
 print('FROZEN COMPARE',args.split,json.dumps(metrics),flush=True)
 print('BOOTSTRAP',json.dumps(result['book_block_bootstrap']),flush=True)

if __name__=='__main__':main()