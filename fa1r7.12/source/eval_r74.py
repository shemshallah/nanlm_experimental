#!/usr/bin/env python3
"""Matched PAL R7.4 8192-context evaluation; historical split, NOT independent."""
import argparse,json,pathlib,time
import torch,numpy as np
from r7_core import Corpus
from r74_stream import restore, OUTER_CONTEXT, OLDER
from train_r7 import batch_data

@torch.no_grad()
def evaluate(root,checkpoint,n=4096,batch=8):
    torch.set_num_threads(2);start=time.monotonic();root=pathlib.Path(root)
    model,opts,base,comp,old,data=restore(root,checkpoint,torch.device('cpu'))
    model.eval()
    report={'checkpoint_format':data['format'],'joint_step':data['joint_steps'],
      'context':'2048 exact + 6144 older recurrent mirror, with streaming carry',
      'historical_holdout_only':True,'new_documents_not_screened':True,
      'caveat':'Original historical corpus has no explicit document boundary map; long-context metrics are exploratory.',
      'positions_per_split':n,'results':{}}
    for split in ('dev','test'):
        corpus=Corpus(root,split)
        positions=np.arange(n,dtype=np.int64)*(len(corpus.seq)-1)//n
        losses={'active':0.,'outer_off':0.,'inwindow_mirror_off':0.}
        for p in np.array_split(positions,int(np.ceil(n/batch))):
            _,ctx,lp,ll,targets,_,_=batch_data(base,corpus,comp,p)
            hist=torch.as_tensor(corpus.get(p,OUTER_CONTEXT)[0][:,:OLDER].copy())
            state=model.initial(hist)
            ex=torch.as_tensor(corpus.data[p].copy())
            long=torch.as_tensor(corpus.long[p].copy())
            # Compare with identical trained weights and identical target positions.
            core,_=model.core(ex,long,ctx,lp,ll)
            no_inwindow,_=model.core(ex,long,ctx,lp,ll,mirror_off=True)
            read=model.outer.read_rails(state,core)
            active=torch.tanh(torch.atanh(core.clamp(-.999999,.999999))+model.outer_gain*read)
            losses['active']+=float(model.trunk.nll(active,targets).sum())
            losses['outer_off']+=float(model.trunk.nll(core,targets).sum())
            losses['inwindow_mirror_off']+=float(model.trunk.nll(no_inwindow,targets).sum())
        report['results'][split]={key+'_nll':value/n for key,value in losses.items()}
        report['results'][split]['outer_memory_benefit_nats']=(losses['outer_off']-losses['active'])/n
    report['outer_gain']=float(model.outer_gain.detach())
    report['model_parameter_count']=sum(p.numel() for p in model.parameters())
    report['elapsed_s']=round(time.monotonic()-start,1)
    output=root/'reports/R7p4_matched.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--count',type=int,default=4096);p.add_argument('--batch',type=int,default=8)
    a=p.parse_args()
    print(json.dumps(evaluate(a.root,a.checkpoint,a.count,a.batch),indent=2))
