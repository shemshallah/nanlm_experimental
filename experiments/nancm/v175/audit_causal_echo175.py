#!/usr/bin/env python3
"""Mechanistic v175 causal pointer audit and unrelated-history counterfactual.
Uses ONLY observed past word IDs. Never trains on TEST labels.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_long175 import load_models,load_split,verify_corpus

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--cf-samples',type=int,default=1200);a=p.parse_args()
 torch.set_num_threads(2)
 m,opt,names,fp,old=load_models()
 ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model_state_dict']);m.eval();m.echo_enabled=True
 root=ROOT/'results/v175_release3';manifest=verify_corpus(root,names);X,Y,B=load_split(a.split,root)
 q=X[:,-1:];match=(X[:,:-1]==q)&(q>=0)&(X[:,1:]>=0)
 eligible=match.any(1)
 goldcopy=(match&(X[:,1:]==Y[:,None])).any(1)
 assert np.all(goldcopy<=eligible)
 groups={k:{'n':0,'old_loss':0.,'new_loss':0.,'old_top1':0,'new_top1':0} for k in ['eligible','not_eligible','gold_copyable','eligible_wrong_copy']}
 with torch.no_grad():
  for start in range(0,len(Y),96):
   x=torch.from_numpy(X[start:start+96]);y=torch.from_numpy(Y[start:start+96]);new,base,copy=m(x,True)
   assert copy is not None
   ce_old=F.cross_entropy(base,y,reduction='none').numpy();ce_new=F.cross_entropy(new,y,reduction='none').numpy();oldtop=base.argmax(1).numpy();newtop=new.argmax(1).numpy()
   for j in range(len(y)):
    k=start+j
    tags=['eligible' if eligible[k] else 'not_eligible']
    if goldcopy[k]:tags.append('gold_copyable')
    elif eligible[k]:tags.append('eligible_wrong_copy')
    for tag in tags:
     agg=groups[tag];agg['n']+=1;agg['old_loss']+=float(ce_old[j]);agg['new_loss']+=float(ce_new[j]);agg['old_top1']+=int(oldtop[j]==Y[k]);agg['new_top1']+=int(newtop[j]==Y[k])
 for g in groups.values():
  g['nll_gain']=(g['old_loss']-g['new_loss'])/max(g['n'],1)
 # Older-history shuffled while keeping immediately preceding three words fixed.
 rng=np.random.default_rng(175);indices=rng.choice(len(Y),size=min(a.cf_samples,len(Y)),replace=False)
 by_book={int(b):np.where(B==b)[0] for b in np.unique(B)}
 keys=np.array(list(by_book));Xorig=X[indices].copy();Xshuf=Xorig.copy()
 for j,i in enumerate(indices):
  otherbooks=keys[keys!=int(B[i])];other=int(rng.choice(otherbooks));loc=int(rng.choice(by_book[other]));Xshuf[j,:-3]=X[loc,:-3]
 gold=Y[indices]
 with torch.no_grad():
  ori,shuffled=[],[]
  for i in range(0,len(gold),72):
   y=torch.from_numpy(gold[i:i+72]);x=torch.from_numpy(Xorig[i:i+72]);cf=torch.from_numpy(Xshuf[i:i+72]);
   ori.append(F.cross_entropy(m(x),y,reduction='sum').item());shuffled.append(F.cross_entropy(m(cf),y,reduction='sum').item())
 result={'cohort':a.split,'books':len(np.unique(B)),'examples':len(Y),'repeat_match_count':int(eligible.sum()),
         'repeat_match_rate':float(eligible.mean()),'correct_word_in_causal_copy_pool_count':int(goldcopy.sum()),
         'pointer_eligible_subsets':groups,
         'counterfactual':{'setup':'replace history words 1..45 by another TEST/VAL book while keeping last3, predict same unseen target',
                           'n':len(gold),'true_history_nll':sum(ori)/len(gold),'unrelated_older_history_nll':sum(shuffled)/len(gold),
                           'correct_history_advantage_nats':(sum(shuffled)-sum(ori))/len(gold)},
         'no_target_leakage':'copy candidates are history[i+1] only for i<47; no future/test labels accessed by prediction',
         'source_sha256':manifest['archive_sha256']}
 out=ROOT/f'results/v175_causal_echo_audit_{a.split}.json';out.write_text(json.dumps(result,indent=2))
 print('V175 CAUSAL ECHO',json.dumps({k:v for k,v in result.items() if k!='pointer_eligible_subsets'}),flush=True)
 print('GROUPS',json.dumps(groups),flush=True)
if __name__=='__main__':main()