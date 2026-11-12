#!/usr/bin/env python3
"""Autoregressive generated BPE stream using ONE PAL10 checkpoint.

Causal: only observed prompt words and generated completed words contribute to
new shared rank-64 NOW conditioning. Previous subwords persist in one GRU state.
"""
import argparse,json,pathlib,re,sys
import numpy as np,torch
import torch.nn.functional as F
from r10_stream import load_model,StreamingBridge
from infer_longctx2048 import context_features
from infer_unified_now import RGX
from build_lossless_subwords import decode_bytes

def generate(root,checkpoint,prompt,words=30,seed=19,temperature=.88,topk=60):
    saved,r6,comp,model=load_model(root,checkpoint)
    assert saved['format']=='PAL10_STREAM_SINGLE_V1'
    br=StreamingBridge();br.load_state_dict(saved['stream_bridge'])
    model.eval();br.eval();rng=torch.Generator().manual_seed(seed)
    lookup={w:i for i,w in enumerate(r6.vocab)}
    history=[lookup.get(w,0) for w in RGX.findall(prompt.lower())]
    def now():
        past=([1,1,1]+history)[-3:]
        e=torch.tensor([past+[1,1,1,1]],dtype=torch.long)
        sketch=torch.from_numpy(context_features(history)[None].astype('float32'))
        return model.core.hidden(e,sketch)
    with torch.no_grad():
        state=now();previous=torch.tensor([model.memory.start_piece]);retain=torch.sigmoid(br.retain_logit)
        out=[];this=[];raw_tokens=[];oov=0;invalid=0
        for _ in range(max(500,words*35)):
            state=model.memory.subword_cell(model.memory.piece_emb(previous),state)
            logits=model.memory.subword_out(state)[0].clone()
            logits[:32]=-1e9 # reject binary controls in human-readable demo
            logits[33:127]=-1e9 # reject unusual non-space raw bytes
            logits[model.memory.start_piece]=-1e9
            if not this:logits[32]=-1e9 # no empty words
            if len(this)>=26:logits[32]+=8.
            values,ids=torch.topk(logits,min(topk,logits.numel()))
            pick=ids[torch.multinomial(F.softmax(values/temperature,dim=-1),1,generator=rng)]
            tok=int(pick.item());raw_tokens.append(tok)
            if tok==32:
                try:spelling=decode_bytes(comp.sp,this).decode('utf8')
                except UnicodeDecodeError:spelling='';invalid+=1
                spelling=spelling.strip()
                if not spelling:spelling='<?>';invalid+=1
                out.append(spelling);history.append(lookup.get(spelling.lower(),0))
                oov+=int(history[-1]==0)
                if len(out)>=words:break
                state=retain*state+(1-retain)*br.context_gain*now()
                this=[]
            else:this.append(tok)
            previous=torch.tensor([tok],dtype=torch.long)
        return {'prompt':prompt,'text':' '.join(out),'generated_words':len(out),'novel_or_unrecognized_words':oov,
                'invalid_utf8_or_empty_segments':invalid,'generated_stream_tokens':len(raw_tokens),
                'core_updates':saved['source_core_steps'],'stream_updates':saved['r10_updates'],
                'note':'No constrained vocabulary or teacher forcing; stream uses its own generated previous pieces.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--checkpoint',type=pathlib.Path,default=pathlib.Path('./checkpoint/r10_active.pal10'))
    p.add_argument('--prompt',default='the river was quiet');p.add_argument('--words',type=int,default=30)
    p.add_argument('--seed',type=int,default=19);p.add_argument('--temperature',type=float,default=.88)
    a=p.parse_args();torch.set_num_threads(2)
    print(json.dumps(generate(a.root,a.checkpoint,a.prompt,a.words,a.seed,a.temperature),indent=2))