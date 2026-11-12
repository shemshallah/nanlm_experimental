#!/usr/bin/env python3
"""Free autoregressive PAL R7.3 inference, using ONE consolidated checkpoint.

All NOW inputs are derived from the observed prompt and earlier generated tokens.
No teacher labels or future words are accessed. Produces genuine model samples;
no claim of fluency is inferred from held-out teacher-forced loss.
"""
from __future__ import annotations
import argparse,json,pathlib,re
import numpy as np,torch
from r7_core import R6,D,LONGF
from r7_memory import TokenCompositor,WORD_CONTEXT,LOCAL
from r73_joint import restore_joint
from infer_unified_now import topk,candidate_score,RGX
from infer_longctx2048 import context_features
from infer_temporal_subword import sample_char_word,CH,PREV

def historical_char_model(state):
    data=state.get('legacy_aux',{}).get('weights.raw.charv1')
    if not data:return None
    h=np.frombuffer(data,dtype='<i4',count=8)
    if h[:5].tolist()!=[0x50414c43,32000,D,CH,PREV]:raise ValueError('historical char sidecar has incompatible header')
    raw=np.frombuffer(data,dtype='<f4',offset=32)
    i=0
    def take(n,shape):
        nonlocal i
        x=raw[i:i+n].reshape(shape);i+=n;return x
    return {'W':take(CH*D,(CH,D)),'B':take(CH,(CH,)),'G1':take(PREV*CH,(PREV,CH)),'G2':take(PREV*CH,(PREV,CH))}

@torch.no_grad()
def run(root,checkpoint,prompt,n=60,temperature=.88,seed=17,top_k=50,mode='word'):
    torch.set_num_threads(2);root=pathlib.Path(root);rng=np.random.default_rng(seed)
    model,_,base,comp,old,state=restore_joint(root,checkpoint,torch.device('cpu'),.00006,.000002,.0001)
    model.eval()
    cmap={v:i for i,v in enumerate(base.vocab)};wordset=set(base.vocab)
    words=RGX.findall(prompt.lower());history=[cmap.get(w,0) for w in words]
    recent=[];output=[];fallbacks=0
    table={'W':model.trunk.W.detach().cpu().numpy(),
           'B':model.trunk.B.detach().cpu().numpy(),
           'children':base.children,'vocab':base.vocab}
    char=historical_char_model(state)
    for t in range(max(1,n)):
        ex=torch.tensor([[*([1,1,1]+history)[-3:],0,1,1,1]],dtype=torch.long)
        long=torch.from_numpy(context_features(history).copy())[None,:]
        ctx=torch.tensor(([1]*WORD_CONTEXT+history)[-WORD_CONTEXT:],dtype=torch.long)[None,:]
        text=(['<bos>']*LOCAL+words)[-LOCAL:]
        pieces,lens=comp.encode_words(text)
        h,_=model(ex,long,ctx,pieces[None,:,:],lens[None,:])
        if mode=='stream':
            result=model.adapter.generate_stream(h,comp,max_pieces=n,temperature=temperature,seed=seed)
            return {'mode':'continuous_subwords','prompt':prompt,'generated':result,'joint_steps':state['joint_steps'],
                    'future_labels_seen':False,'warning':'Historical corpus normalized words; original punctuation/whitespace not reconstructed'}
        now=h[0].cpu().numpy()
        candidates=[(wid,score) for wid,score in topk(table,now,max(40,top_k)+4) if wid>1][:top_k]
        if all(wid!=0 for wid,_ in candidates):candidates.append((0,candidate_score(table,now,0)))
        logit=np.asarray([score for wid,score in candidates],dtype=np.float64)/max(.06,temperature)
        for j,(wid,_) in enumerate(candidates):
            if wid>6 and wid in recent[-16:]:logit[j]-=.7
        logit-=logit.max();probs=np.exp(logit);probs/=probs.sum()
        wid=candidates[int(rng.choice(len(candidates),p=probs))][0]
        if wid==0:
            candidate=None
            for k in range(20):
                textpiece=model.adapter.spell(h[0],comp,max_len=22,seed=seed+100007*t+11*k)
                if re.fullmatch(r"[a-zA-Z][a-zA-Z'\-]{1,32}",textpiece) and textpiece.lower() not in wordset:
                    candidate=textpiece.lower();break
            if candidate is None and char is not None:candidate=sample_char_word(now,char,rng,wordset,temp=temperature)
            if candidate is None:candidate='<unk>'
            fallbacks+=1
            word=candidate
        else:word=base.vocab[wid]
        output.append(word);words.append(word);history.append(wid);recent.append(wid)
    return {'mode':'word','prompt':prompt,'generated':re.sub(r'\s+([.,!?;:])',r'\1',' '.join(output)),
            'joint_steps':state['joint_steps'],'legacy_r6_steps':state['base_legacy_steps'],
            'legacy_r7_adapter_steps':state['adapter_legacy_steps'],'fallbacks':fallbacks,'future_labels_seen':False}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--prompt',default='the stars were shining above the old house')
    p.add_argument('--n',type=int,default=50);p.add_argument('--temperature',type=float,default=.88)
    p.add_argument('--seed',type=int,default=16);p.add_argument('--top-k',type=int,default=50)
    p.add_argument('--mode',choices=('word','stream'),default='word')
    a=p.parse_args();print(json.dumps(run(a.root,a.checkpoint,a.prompt,a.n,a.temperature,a.seed,a.top_k,a.mode),indent=2))
