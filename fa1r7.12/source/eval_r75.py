#!/usr/bin/env python3
"""Matched deterministic R7.5 whole-word NLL vs retrieval-ablated baseline.

For large audits, run --total 4096 --start {0,1024,2048,3072} --count 1024;
then combine the four raw sums (not the mean of rounded outputs).
"""
import argparse,json,pathlib,time
import torch,numpy as np
from r7_core import Corpus
from r74_stream import OUTER_CONTEXT,OLDER
from r75_retrieval import restore
from train_r7 import batch_data

@torch.no_grad()
def evaluate(root,checkpoint,total=4096,start=0,count=1024,batch=8):
    torch.set_num_threads(2);t=time.monotonic();root=pathlib.Path(root)
    model,opts,base,comp,old,prior=restore(root,checkpoint,torch.device('cpu'))
    model.eval();r={'checkpoint_format':prior['format'],'joint_step':prior['joint_steps'],
        'total_positions':total,'slice_start':start,'slice_count':count,'historical_splits_only':True,
        'new_documents_evaluated':False,'split_sums':{}}
    for split in ('dev','test'):
        corpus=Corpus(root,split)
        positions=np.arange(total,dtype=np.int64)*(len(corpus.seq)-1)//total
        positions=positions[start:start+count]
        sums={'active':0.,'retrieval_off':0.,'outer_off':0.,'key_positions':0,'target_hits':0,'n':len(positions)}
        for pp in np.array_split(positions,int(np.ceil(len(positions)/batch))):
            _,ctx,pc,ln,target,_,_=batch_data(base,corpus,comp,pp)
            hist=torch.as_tensor(corpus.get(pp,OUTER_CONTEXT)[0][:,:OLDER].copy())
            state=model.initial(hist)
            a=torch.as_tensor(corpus.data[pp].copy());l=torch.as_tensor(corpus.long[pp].copy())
            now,nll,_=model(a,l,ctx,pc,ln,state,hist,target)
            noouter=model.r74(a,l,ctx,pc,ln,state=state,outer_off=True)[0]
            prob,exists,features,_=model.retrieval.lookup(hist,ctx[:,-1],target)
            gate=model.retrieval.gate(noouter,ctx[:,-1],features,model.trunk.E,exists)
            noouter_nll=model.retrieval.mixed_nll(model.trunk.nll(noouter,target),prob,gate)
            sums['active']+=float(nll.sum());sums['retrieval_off']+=float(model.trunk.nll(now,target).sum())
            sums['outer_off']+=float(noouter_nll.sum());sums['key_positions']+=int(exists.sum())
            sums['target_hits']+=int((prob>0).sum())
        r['split_sums'][split]=sums
    r['elapsed_s']=round(time.monotonic()-t,1)
    p=root/'reports'/f'R7p5_slice_{total}_{start}_{count}.json';p.write_text(json.dumps(r,indent=2)+'\n')
    return r

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    ap.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    ap.add_argument('--total',type=int,default=4096);ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--count',type=int,default=1024);ap.add_argument('--batch',type=int,default=8)
    args=ap.parse_args()
    print(json.dumps(evaluate(args.root,args.checkpoint,args.total,args.start,args.count,args.batch),indent=2))
