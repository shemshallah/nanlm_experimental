#!/usr/bin/env python3
"""Train-side-only top1-oriented calibration, normalized simplex remains exact.
Separate hypothesis from NLL-first v180: do not replace accepted default unless it
improves both top1 and NLL on independent books without TEST inspection.
"""
from normalized_mix180 import get,calculate,R
import numpy as np,json,time
st=time.monotonic();t=get('train');v=get('val');sel=[]
patterns=[[1,1,1,1],[0,1,1,1],[1,1,2,4],[4,2,1,0],[0,0,1,1],[0,0,0,1],[0,1,0,0],[1,0,0,0],[0,0,1,0],[1,2,3,4],[1,4,2,1],[0,1,2,1],[1,0,1,0],[0,1,0,1]]
for p in patterns:
 p=np.asarray(p,dtype=float);p/=p.sum()
 for s in np.linspace(0,1,41):
  g=(p*s).tolist();r=calculate(*t,g);sel.append((r['top1'],-r['nll'],g,r))
sel.sort(reverse=True)
chosen=sel[0];report={'source':'TRAIN-only top1 tuning','train_best':chosen[3],
 'gamma':chosen[2],'val':calculate(*v,chosen[2]),'val_parent':calculate(*v,[0,0,0,0]),
 'val_nll_best_gamma':[.1,.1,.1,.1], 'val_nll_best':calculate(*v,[.1,.1,.1,.1]),
 'val_had_previously_been_explored':True,'elapsed':time.monotonic()-st}
print(json.dumps(report,indent=2));(R/'v180_top1_calibration.json').write_text(json.dumps(report,indent=2)+'\n')
