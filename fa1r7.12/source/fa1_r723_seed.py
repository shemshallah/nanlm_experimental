#!/usr/bin/env python3
"""Migrate into the one unified R7.23 Diamond identity checkpoint, no training."""
import argparse,copy,json,pathlib,time,torch
from fa1_r723_diamond import extend_r722,save_diamond,load_diamond
from r710_checkpoint_integrity import sha256,deep_equal

def run(a):
 t=time.monotonic();torch.set_num_threads(2);torch.manual_seed(a.seed)
 m,opts,base,comp,meta=extend_r722(a.parent,a.root)
 assert len(opts)==5 and len(opts[-1].param_groups)==8
 inherited_model={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('diamond.')}
 inherited_mom=[copy.deepcopy(o.state_dict()) for o in opts]
 result=save_diamond(a.out,m,opts,meta['parent_sha256'],training_step=0,
              metrics={'status':'identity migration only, no training or promotion'})
 restored,ropts,_,_,rmeta=load_diamond(a.out,a.root)
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in m.state_dict().items())
 assert all(deep_equal(x.state_dict(),y.state_dict()) for x,y in zip(opts,ropts))
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in inherited_model.items())
 assert all(deep_equal(old,new.state_dict()) for old,new in zip(inherited_mom,ropts))
 assert rmeta['new_diamond_updates']==0 and rmeta['promoted'] is False
 report={'format':'FA1_R7_23_DIAMOND_IDENTITY_CHECKPOINT_VERIFIED','sha256':result['sha256'],
         'bytes':result['bytes'],'accepted_parent_step':14416,'optimization_updates':0,
         'optimizer_histories_preserved':5,'all_original_weight_tensors_bitwise_preserved':True,
         'all_original_optimizer_history_bitwise_preserved':True,
         'r722_ancestor_sha256':meta['parent_sha256'],
         'learnable_triples':2763520,'meta_specialists':256,'roundtrip_verified':True,
         'promoted':False,'elapsed_s':time.monotonic()-t}
 a.report.parent.mkdir(parents=True,exist_ok=True)
 a.report.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
 p.add_argument('--parent',default='/mnt/data/files/FA1_R7p22_UNIFIED_CANDIDATE.pt')
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('/mnt/data/diamond_r723/FA1_R723_DIAMOND_IDENTITY.pt'))
 p.add_argument('--report',type=pathlib.Path,default=pathlib.Path('/mnt/data/diamond_r723/identity_roundtrip.json'))
 p.add_argument('--seed',type=int,default=72347)
 run(p.parse_args())