#!/usr/bin/env python3
"""Single-checkpoint PAL R7.4 autoregressive generation with persistent outer memory.

The cross-window rank64 state is initialized from previously OBSERVED prompt
words and advanced ONLY after a new word has been generated. No future data.
"""
from __future__ import annotations
import argparse,json,pathlib,re
import numpy as np,torch
from r7_memory import WORD_CONTEXT,LOCAL
from r74_stream import restore,OLDER,OUTER_CONTEXT
from infer_unified_now import topk,candidate_score,RGX
from infer_longctx2048 import context_features
from infer_r73 import historical_char_model
from infer_temporal_subword import sample_char_word

@torch.no_grad()
def run(root,checkpoint,prompt,n=60,temperature=.88,seed=17,top_k=50,outer_off=False,mode='word'):
    torch.set_num_threads(2);root=pathlib.Path(root);rng=np.random.default_rng(seed)
    model,_,base,comp,old,checkpoint_meta=restore(root,checkpoint,torch.device('cpu'))
    model.eval()
    cmap={word:i for i,word in enumerate(base.vocab)}
    observed=RGX.findall(prompt.lower());history=[cmap.get(w,0) for w in observed]
    older=([1]*OUTER_CONTEXT+history)[-OUTER_CONTEXT:-WORD_CONTEXT]
    state=model.initial(torch.tensor([older],dtype=torch.long))
    table={'W':model.trunk.W.detach().cpu().numpy(),
           'B':model.trunk.B.detach().cpu().numpy(),
           'children':base.children,'vocab':base.vocab}
    char=historical_char_model(checkpoint_meta);recent=[];output=[];fallbacks=0
    for t in range(n):
        ex=torch.tensor([[*([1,1,1]+history)[-3:],0,1,1,1]],dtype=torch.long)
        long=torch.from_numpy(context_features(history).copy())[None,:]
        ctx=torch.tensor(([1]*WORD_CONTEXT+history)[-WORD_CONTEXT:],dtype=torch.long)[None,:]
        text=(['<bos>']*LOCAL+observed)[-LOCAL:]
        pieces,lens=comp.encode_words(text)
        now,_=model(ex,long,ctx,pieces[None,:,:],lens[None,:],state=state,outer_off=outer_off)
        if mode=='stream':
            sample=model.adapter.generate_stream(now,comp,max_pieces=n,temperature=temperature,seed=seed)
            return {'mode':'continuous_subwords','generated':sample,'joint_steps':checkpoint_meta['joint_steps'],
                    'older_memory_active':not outer_off,'future_labels_seen':False}
        h=now[0].numpy()
        candidates=[(w,score) for w,score in topk(table,h,max(40,top_k)+4) if w>1][:top_k]
        if all(w!=0 for w,_ in candidates):candidates.append((0,candidate_score(table,h,0)))
        logits=np.asarray([sc for wid,sc in candidates],dtype=np.float64)/max(.06,temperature)
        for j,(wid,_) in enumerate(candidates):
            if wid>6 and wid in recent[-16:]:logits[j]-=.7
        logits-=logits.max();probs=np.exp(logits);probs/=probs.sum()
        wid=candidates[int(rng.choice(len(candidates),p=probs))][0]
        if wid==0:
            fallback=None
            for k in range(20):
                candidate=model.adapter.spell(now[0],comp,max_len=22,seed=seed+100007*t+11*k)
                if re.fullmatch(r"[a-zA-Z][a-zA-Z'\-]{1,32}",candidate) and candidate.lower() not in cmap:
                    fallback=candidate.lower();break
            if fallback is None and char is not None:
                fallback=sample_char_word(h,char,rng,set(base.vocab),temp=temperature)
            word=fallback or '<unk>';fallbacks+=1
        else:word=base.vocab[wid]
        leaving=int(ctx[0,0])
        state=model.advance(state,torch.tensor([leaving]))
        output.append(word);observed.append(word);history.append(wid);recent.append(wid)
    return {'mode':'word','prompt':prompt[-140:],'generated':' '.join(output),
            'joint_steps':checkpoint_meta['joint_steps'],'fallbacks':fallbacks,
            'older_memory_active':not outer_off,'future_labels_seen':False,
            'warning':'PAL corpus has normalized word streams, not original punctuation and whitespace.'}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('checkpoint/pal_active.pt'))
    p.add_argument('--prompt',default='the stars were shining above the old house')
    p.add_argument('--prompt-file',type=pathlib.Path)
    p.add_argument('--prompt-words',type=int,default=8192)
    p.add_argument('--n',type=int,default=60);p.add_argument('--temperature',type=float,default=.88)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--top-k',type=int,default=50)
    p.add_argument('--outer-off',action='store_true')
    p.add_argument('--mode',choices=('word','stream'),default='word')
    a=p.parse_args()
    prompt=' '.join(RGX.findall(a.prompt_file.read_text(errors='replace').lower())[-a.prompt_words:]) if a.prompt_file else a.prompt
    print(json.dumps(run(a.root,a.checkpoint,prompt,a.n,a.temperature,a.seed,a.top_k,a.outer_off,a.mode),indent=2))
