#!/usr/bin/env python3
"""Matched, causal streaming-BPE R9-vs-R10 evaluation with lexical/space split."""
import argparse,json,pathlib,sys
import numpy as np,torch
import torch.nn.functional as F
from r7_core import Corpus
from r10_stream import load_model,StreamingBridge,compatible_starts,build_batch

def detail(model,bridge,cor,comp,starts,window,batchsize=4,ablation="none"):
    model.eval();bridge.eval()
    stats={c:{'loss':0.,'n':0,'correct':0} for c in ['all','space','nonspace','first_word','later_words']}
    with torch.no_grad():
        for ix in np.array_split(starts,max(1,(len(starts)+batchsize-1)//batchsize)):
            e,lc,pieces,tags,bytes_=build_batch(cor,comp,model._r10_vocab,ix,window)
            B,W=e.shape[:2]
            h=model.core.hidden(e.reshape(-1,7),lc.reshape(-1,72)).reshape(B,W,-1)
            if ablation=="zero_now": h=torch.zeros_like(h)
            elif ablation=="shuffled_now": h=h.roll(1,dims=0)
            current=torch.full((B,),model.memory.start_piece,dtype=torch.long)
            state=h[:,0,:];last=tags[:,0];retain=torch.sigmoid(bridge.retain_logit)
            for t in range(pieces.shape[1]):
                valid=pieces[:,t]!=-100;ind=tags[:,t].clamp(0,W-1)
                context=h.gather(1,ind[:,None,None].expand(-1,1,h.shape[-1]))[:,0,:]
                state=torch.where(((ind!=last)&valid)[:,None],retain*state+(1-retain)*bridge.context_gain*context,state)
                state=model.memory.subword_cell(model.memory.piece_emb(current),state)
                logits=model.memory.subword_out(state)
                targets=pieces[:,t]
                for key,mask in [('all',valid),('space',valid&(targets==32)),('nonspace',valid&(targets!=32)),('first_word',valid&(ind==0)),('later_words',valid&(ind>0))]:
                    if not mask.any():continue
                    truth=targets[mask];pred=logits[mask]
                    stats[key]['loss']+=float(F.cross_entropy(pred,truth,reduction='sum'))
                    stats[key]['correct']+=int((pred.argmax(-1)==truth).sum())
                    stats[key]['n']+=int(mask.sum())
                current=torch.where(valid,targets,torch.zeros_like(current));last=torch.where(valid,ind,last)
    return {k:{'nll':v['loss']/v['n'],'accuracy':v['correct']/v['n'],'count':v['n']} for k,v in stats.items()}

def run(root,checkpoint,n=160,window=8,which='test'):
    torch.set_num_threads(2)
    legacy_path=root/'checkpoint/r9_active.pal9'
    old,r6,comp,baseline=load_model(root,legacy_path) if legacy_path.exists() else (None,None,None,None)
    new,r6,comp,model=load_model(root,checkpoint)
    assert new['format']=='PAL10_STREAM_SINGLE_V1'
    if baseline is not None:baseline._r10_vocab=r6.vocab
    model._r10_vocab=r6.vocab
    bridge=StreamingBridge();bridge.load_state_dict(new['stream_bridge'])
    corp=Corpus(root,which);possible=compatible_starts(corp,window)
    starts=possible[np.linspace(0,len(possible)-1,n,dtype=np.int64)]
    base=detail(baseline,StreamingBridge(),corp,comp,starts,window) if baseline is not None else None
    curr=detail(model,bridge,corp,comp,starts,window)
    zero=detail(model,bridge,corp,comp,starts,window,ablation='zero_now')
    shuffled=detail(model,bridge,corp,comp,starts,window,ablation='shuffled_now')
    out={'source':'same sample starts; BPE from original TRAIN-only tokenizer; normalized word streams, not original raw text','split':which,'windows':n,'window_words':window,'base_r9':base,'r10':curr,'zero_now_ablation':zero,'shuffled_now_ablation':shuffled,'steps':new['r10_updates']}
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--checkpoint',default='./checkpoint/r10_active.pal10');p.add_argument('--split',default='test',choices=['test','dev']);p.add_argument('--n',type=int,default=160);p.add_argument('--out',default='./reports/r10_diagnostics.json')
    a=p.parse_args();x=run(pathlib.Path(a.root),pathlib.Path(a.checkpoint),a.n,which=a.split)
    pathlib.Path(a.out).write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
