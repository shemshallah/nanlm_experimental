#!/usr/bin/env python3
"""End-to-end FA1 R7.19 read-only parity on the verified real parent + R7.18 gate."""
import argparse, json, pathlib
import numpy as np
import torch
from r7_core import Corpus
from train_r7 import batch_data
from r712_validate import restore
from r710_checkpoint_integrity import sha256
from fa1_r716_router import PARENT, TrustRouter
from fa1_r718_far_residual import FarResidual, gather, loss_far, make_feature
from fa1_r719_streaming_index import StreamingFarIndex

NEAR='436ff87fae69ef1f44b53e2ffde92313da408342f65a1897c6eaff1876f9c09c'
FAR='f4d9175ff202d9ec1ff3a06dcddfd3d262cea25f8441fa5dab972a313309aa9f'

def run(a):
    torch.set_num_threads(2)
    root=pathlib.Path(a.root);nearpath=pathlib.Path(a.near);farpath=pathlib.Path(a.far)
    assert sha256(root/'checkpoint/pal_active.pt')==PARENT
    assert sha256(nearpath)==NEAR and sha256(farpath)==FAR
    model,opts,base,comp,_,_=restore(root,root/'adaptation/r712_gate_delta.pt')
    assert len(opts)==5
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    blob=torch.load(nearpath,map_location='cpu',weights_only=False)
    near=TrustRouter();near.load_state_dict(blob['gate']);near.eval()
    farblob=torch.load(farpath,map_location='cpu',weights_only=False)
    far=FarResidual();far.load_state_dict(farblob['far_gate']);far.eval()
    old=gather(root,model,base,comp,near,a.split,a.examples,71927,batch=a.examples)
    corpus=Corpus(root,a.split)
    feature=[];prob=[];counts=[]
    for i,pos in enumerate(old['positions']):
        inds=np.array([pos],dtype=np.int64)
        _,ctx,_,_,y,_,_=batch_data(base,corpus,comp,inds)
        query=ctx[0].numpy(); target=int(y[0])
        idx=StreamingFarIndex(corpus.seq,pos)
        rec=idx.query(query,target)
        prior=[float(old['far_features'][i,7*j+5]) for j in range(4)]
        f=make_feature(rec,prior,query,far=True)
        feature.append(f);prob.append([r['target_probability'] for r in rec]);counts.append([r['n'] for r in rec])
        assert np.allclose(counts[-1],old['far_count'][i].numpy(),atol=0,rtol=0)
        assert np.allclose(prob[-1],old['far_prob'][i].numpy(),atol=6e-5,rtol=6e-5)
    new=dict(old)
    new['far_features']=torch.tensor(np.asarray(feature),dtype=torch.float32)
    new['far_prob']=torch.tensor(np.asarray(prob),dtype=torch.float32)
    new['far_count']=torch.tensor(np.asarray(counts),dtype=torch.float32)
    ll1,g1=loss_far(far,old);ll2,g2=loss_far(far,new)
    gate_error=float((g1-g2).abs().max().detach())
    nll_error=float((ll1-ll2).abs().max().detach())
    feature_error=float((old['far_features']-new['far_features']).abs().max().detach())
    assert torch.allclose(ll1,ll2,atol=2e-6,rtol=2e-6),(ll1,ll2)
    assert torch.allclose(g1,g2,atol=2e-6,rtol=2e-6)
    report={'round':'FA1 R7.19 actual R7.13+R7.16+R7.18 parity',
        'accepted_parent_step':14416,'parent_sha256':PARENT,'five_optimizer_histories':len(opts),
        'r716_overlay_sha256':NEAR,'r718_overlay_sha256':FAR,'split':a.split,
        'historical_examples':a.examples,'positions':old['positions'],
        'maximum_abs_feature_error':feature_error,'maximum_abs_gate_error':gate_error,
        'maximum_abs_NLL_error_nats_per_word':nll_error,
        'mean_r718_reference_NLL':float(ll1.mean().detach()),'mean_r719_indexed_NLL':float(ll2.mean().detach()),
        'parity_passed':True,'heldout_books':False,'promoted':False}
    assert sha256(root/'checkpoint/pal_active.pt')==PARENT
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/mnt/data/fa1_full')
    ap.add_argument('--near',default='/mnt/data/fa1r719/adaptation/fa1_r716_225.pt')
    ap.add_argument('--far',default='/mnt/data/fa1r719/adaptation/fa1_r718_full_candidate.pt')
    ap.add_argument('--split',choices=('train','dev','test'),default='test')
    ap.add_argument('--examples',type=int,default=4)
    run(ap.parse_args())