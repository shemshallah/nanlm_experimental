#!/usr/bin/env python3
"""R10 correctness smoke tests; no actual training updates and no checkpoint changes."""
import argparse,json,pathlib,sys,torch,numpy as np
import torch.nn.functional as F
from r10_stream import load_model,StreamingBridge,Corpus,compatible_starts,build_batch
from r10_generate import generate
from r7_core import D,R

def run(root,checkpoint):
    torch.set_num_threads(2);checks={}
    s,r6,comp,m=load_model(root,checkpoint)
    checks['one_checkpoint_format']= s['format']=='PAL10_STREAM_SINGLE_V1' and s['r10_updates']>0
    checks['single_live_optimizer']= 'stream_optimizer' in s and 'optimizer' not in s and 'legacy_optimizer_bytes' not in s
    checks['shared_rank64_mirror_and_time']=tuple(m.core.P.shape)==(32640,R) and tuple(m.core.M.shape)==(32640,R) and tuple(m.core.T.shape)==(4,R)
    c=Corpus(root,'dev');p=compatible_starts(c,6)[:2];e,lc,pieces,tags,nb=build_batch(c,comp,r6.vocab,p,6)
    checks['stream_has_literal_spaces']=int((pieces==32).sum())==len(p)*6
    checks['stream_has_no_interword_eos']=int((pieces==m.memory.start_piece).sum())==0
    checks['causal_word_history']=all(int(e[i,j,2])==int(e[i,j-1,3]) for i in range(2) for j in range(1,6))
    h=m.core.hidden(e.reshape(-1,7),lc.reshape(-1,72));q=e.clone();q[:,:,3:]=0
    h2=m.core.hidden(q.reshape(-1,7),lc.reshape(-1,72))
    checks['target_future_not_in_now']=bool(torch.equal(h,h2))
    bridge=StreamingBridge();bridge.load_state_dict(s['stream_bridge'])
    with torch.no_grad():
        loss,n=bridge(m,h.reshape(2,6,-1),pieces,tags)
        checks['finite_stream_loss']=bool(torch.isfinite(loss)) and n==int((pieces!=-100).sum())
        # Prefix logits depend only on tokens before that position, not suffix.
        def prefix_logits(piece,limit=3):
            state=h.reshape(2,6,-1)[:,0];prev=torch.full((2,),m.memory.start_piece,dtype=torch.long)
            last=tags[:,0];keep=torch.sigmoid(bridge.retain_logit)
            for t in range(limit+1):
                valid=piece[:,t]!=-100;idx=tags[:,t]
                x=h.reshape(2,6,-1).gather(1,idx[:,None,None].expand(-1,1,D))[:,0,:]
                state=torch.where(((idx!=last)&valid)[:,None],keep*state+(1-keep)*bridge.context_gain*x,state)
                state=m.memory.subword_cell(m.memory.piece_emb(prev),state)
                logits=m.memory.subword_out(state)
                prev=piece[:,t];last=idx
            return logits
        altered=pieces.clone();altered[:,4:]=0
        checks['suffix_cannot_change_prefix_logits']=bool(torch.equal(prefix_logits(pieces,3),prefix_logits(altered,3)))
    sample=generate(root,checkpoint,'the river was quiet',words=7,seed=4)
    checks['free_autoregressive_generation']=sample['generated_words']==7 and sample['invalid_utf8_or_empty_segments']==0
    checks['optimizer_has_live_moments']=len(s['stream_optimizer']['state'])>=10
    result={'passed':sum(checks.values()),'total':len(checks),'checks':checks,'checkpoint_updates':s['r10_updates'],'generation':sample}
    assert all(checks.values()), result
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'));p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('./checkpoint/r10_active.pal10'))
    a=p.parse_args();print(json.dumps(run(a.root,a.checkpoint),indent=2))
