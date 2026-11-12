#!/usr/bin/env python3
"""Frozen synthetic copy-trainer transfer audit on held-out historic book dev."""
import argparse,json,pathlib,time
import torch
from r712_validate import restore as restore_r712
from fa1_r715_mirror import QueryMirrorLM,prepare,wrong_history,judge
from r710_checkpoint_integrity import sha256

def run(a):
    t=time.monotonic();torch.set_num_threads(2);root=pathlib.Path(a.root)
    ck=sha256(root/'checkpoint/pal_active.pt')
    model,opts,base,comp,gate,_=restore_r712(root,root/'adaptation/r712_gate_delta.pt');model.eval()
    cache=prepare(root,model,base,comp,'dev',a.dev,7154,k=a.topk,batch=8)
    wrong=wrong_history(cache,max(1,a.dev//3),model.trunk.E,k=a.topk,batch=8)
    head=QueryMirrorLM(model.trunk.E,model.r74.outer.dir_key,model.r74.outer.dir_value)
    synth=torch.load(root/'adaptation/fa1_r715_synthetic_diagnostic.pt',map_location='cpu',weights_only=False)
    if synth['parent']!=ck:raise ValueError('synthetic adapter parent mismatch')
    head.load_state_dict(synth['parameters'])
    result={'experiment':'Synthetic key-value writer transfer to natural-language historical dev',
       'accepted_parent':ck,'synthetic_training_only':True,'real_book_data_in_training':False,
       'counts':{'dev':a.dev},'dev':judge(head,cache,wrong),
       'book_disjointness_verified':False,'status':'diagnostic only, not promotion',
       'elapsed_s':round(time.monotonic()-t,2)}
    (root/'reports/FA1_R7p15_transfer.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--dev',type=int,default=256);p.add_argument('--topk',type=int,default=96);run(p.parse_args())