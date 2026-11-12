#!/usr/bin/env python3
"""Contrastive old bigram recall: correct rank64 state vs batch-shuffled state."""
import argparse,json,pathlib
import torch,numpy as np
from r77_semantic import restore
from r77_outer_aux import contrastive_loss
from r74_stream import OUTER_CONTEXT,OLDER
from r7_core import Corpus

@torch.no_grad()
def measure(model,corpus,count=128,batch=8):
    device=next(model.parameters()).device
    positions=np.linspace(9000,len(corpus.seq)-1,count,dtype=np.int64)
    total={'true':0.,'swapped':0.};n=0
    for j,idx in enumerate(np.array_split(positions,int(np.ceil(count/batch)))):
        old=torch.as_tensor(corpus.get(idx,OUTER_CONTEXT)[0][:,:OLDER].copy(),device=device)
        state=model.initial(old)
        # Exactly matched random queries and negative candidates for the two tests.
        gen=torch.Generator(device=device).manual_seed(110011+j)
        ltrue=contrastive_loss(model,state,old,negatives=24,generator=gen)
        gen=torch.Generator(device=device).manual_seed(110011+j)
        lshuffle=contrastive_loss(model,state,old,negatives=24,generator=gen,swapped=True)
        sz=len(idx);total['true']+=float(ltrue)*sz;total['swapped']+=float(lshuffle)*sz;n+=sz
    return {'positions':n,'correct_state_CE':total['true']/n,
            'shuffled_state_CE':total['swapped']/n,
            'correct_state_advantage':(total['swapped']-total['true'])/n}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--split',default='dev');p.add_argument('--count',type=int,default=128)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads)
    m,o,b,c,d=restore(a.root,a.root/'checkpoint/pal_active.pt',torch.device('cpu'));m.eval()
    result={'step':d['joint_steps'],'split':a.split,'historical_only':True,
            'outer_assoc':measure(m,Corpus(a.root,a.split),a.count,a.batch)}
    print(json.dumps(result,indent=2))
