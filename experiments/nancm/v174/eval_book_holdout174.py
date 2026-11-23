#!/usr/bin/env python3
"""Book-block bootstrap and strict top1/5 evaluation of frozen temporal word expert.
Note: val books excluded from v174 training, but included in inherited pretraining.
"""
import json,math,sys,collections
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_temporal_signed174 import ROOT,load_parent,CloudNCE,load_sign_geometry,DualPolarTemporal,make_dataset
names,C,O,B,spatial,gain,fp=load_parent();te=CloudNCE(C,O,B,spatial,gain,names)
par=torch.load(ROOT/'checkpoints/v173_exact_ce_word.pt',map_location='cpu',weights_only=False);te.load_state_dict(par['state_dict'])
gp,gn=load_sign_geometry(names);m=DualPolarTemporal(te,names,gp,gn)
ck=torch.load(ROOT/'checkpoints/v174_temporal_signed_now.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model_state_dict']);m.eval()
train,val,books,hold=make_dataset(names);X,Y,booklist=val;torch.set_num_threads(2)
agg=collections.defaultdict(lambda:[0,0.,0,0,0.,0,0])
for i in range(0,len(Y),128):
 x=torch.tensor(X[i:i+128],dtype=torch.long);y=torch.tensor(Y[i:i+128],dtype=torch.long)
 with torch.no_grad():z,p,_=m(x,True);zi=z.topk(5,dim=-1).indices;pi=p.topk(5,dim=-1).indices
 for j,(b,target) in enumerate(zip(booklist[i:i+128],Y[i:i+128])):
  a=agg[b];a[0]+=1;a[1]+=-float(torch.log_softmax(z[j],0)[target]);a[2]+=int(zi[j,0]==target);a[3]+=int((zi[j]==target).any())
  a[4]+=-float(torch.log_softmax(p[j],0)[target]);a[5]+=int(pi[j,0]==target);a[6]+=int((pi[j]==target).any())
A=np.asarray(list(agg.values()),dtype=float);v=A.sum(axis=0)
print('TOTAL n',int(v[0]),'new_nll',v[1]/v[0],'old_nll',v[4]/v[0],'new1',int(v[2]),'old1',int(v[5]),'new5',int(v[3]),'old5',int(v[6]),flush=True)
rng=np.random.default_rng(174);rep=[]
for i in range(5000):
 z=A[rng.integers(len(A),size=len(A))].sum(axis=0)
 rep.append(((z[4]-z[1])/z[0],(z[2]-z[5])/z[0],(z[3]-z[6])/z[0]))
rep=np.array(rep)
interval=lambda k:list(map(float,np.quantile(rep[:,k],[.025,.5,.975])))
report={'status':'RESEARCH_ONLY','split':'held out 20 TRAIN books for v174 new parameters, but inherited teacher had seen TRAIN aggregates',
        'books':len(A),'examples':int(v[0]),'new':{'nll':v[1]/v[0],'top1':int(v[2]),'top5':int(v[3])},'frozen_parent':{'nll':v[4]/v[0],'top1':int(v[5]),'top5':int(v[6])},
        'gain_nll':(v[4]-v[1])/v[0],'gain_top1_rate':(v[2]-v[5])/v[0],
        'book_bootstrap_95_nll_gain':interval(0),'book_bootstrap_95_top1_rate_gain':interval(1),'book_bootstrap_95_top5_rate_gain':interval(2),'bootstrap_samples':len(rep),'by_book':{k:{'examples':int(a[0]),'nll':a[1]/a[0],'top1':a[2],'top5':a[3],'parent_nll':a[4]/a[0],'parent_top1':a[5],'parent_top5':a[6]} for k,a in agg.items()}}
(ROOT/'results/v174_book_holdout.json').write_text(json.dumps(report,indent=2));print('BOOTSTRAP',interval(0),interval(1),interval(2),flush=True)