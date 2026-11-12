#!/usr/bin/env python3
"""PAL11 open-vocabulary generation from ONE rank-64 model with an attached character fallback.
The PAL5 hierarchical decoder chooses a word or <unk>. The attached PAL11
causal character head spells a novel word only when the shared decoder chooses
<unk>. No future tokens are supplied to either branch at inference time.
"""
import argparse,sys,re
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from infer_longctx2048 import read_long, context_features, hidden_long
from infer_unified_now import load, topk, candidate_score, RGX
D,CH,PREV=96,34,35
ALPHABET="abcdefghijklmnopqrstuvwxyz'.,!?;:"
EOS=CH-1;SOS=CH

def read_char(path, updates, vocab_size):
    raw=path.read_bytes();h=np.frombuffer(raw,dtype='<i4',count=8)
    if h.tolist()[:7]!=[0x50414c43,vocab_size,D,CH,PREV,updates,1]:
        raise ValueError(f'PAL11 sidecar shape/step mismatch: {h}')
    data=np.frombuffer(raw,dtype='<f4',offset=32)
    n=CH*D+CH+2*PREV*CH
    if data.size!=2*n:raise ValueError(f'PAL11 sidecar truncated: {data.size} != {2*n}')
    i=0
    def take(count,shape):
        nonlocal i
        out=data[i:i+count].reshape(shape);i+=count
        return out
    W=take(CH*D,(CH,D));B=take(CH,(CH,));G1=take(PREV*CH,(PREV,CH));G2=take(PREV*CH,(PREV,CH))
    return dict(W=W,B=B,G1=G1,G2=G2)

def sample_char_word(h,char,rng,wordset,temp=1.0,max_chars=36,only_novel=True,retries=140):
    base=char['W']@h+char['B']
    for attempt in range(retries):
        p1=p2=SOS;out=[]
        # Higher temperature on retry helps get outside the known vocabulary.
        t=max(.4,temp+(min(attempt,70)*.007))
        for _ in range(max_chars):
            logits=(base+char['G1'][p1]+char['G2'][p2]).astype(np.float64)/t
            if not out:logits[EOS]=-1e9
            if len(out)>=22:logits[EOS]+=min(8.,(len(out)-21)*1.5)
            logits-=np.max(logits);p=np.exp(logits);p/=p.sum()
            ix=int(rng.choice(CH,p=p))
            if ix==EOS:break
            out.append(ALPHABET[ix]);p2,p1=p1,ix
        word=''.join(out)
        if word and (not only_novel or word not in wordset):return word
    return None

def generate(m,L,char,prompt,n=50,seed=42,temp=.9,top_k=60,force_first_character=False,full_tree=False):
    rng=np.random.default_rng(seed)
    prompt_words=RGX.findall(prompt.lower())
    hist=[m['lookup'].get(x,0) for x in prompt_words]
    wordset=set(m['vocab'])
    out=[];fallbacks=0;recent=[]
    children=m['children']
    for step in range(n):
        h=hidden_long(m,L,hist)
        if force_first_character and step==0:y=0
        elif full_tree:
            # Exact pathwise hierarchical sampling. Branch temperature is optional.
            idx=2*len(m['vocab'])-2
            while idx>=len(m['vocab']):
                j=idx-len(m['vocab']);z=float(m['W'][j]@h+m['B'][j])/temp
                p=1/(1+np.exp(-np.clip(z,-60,60)))
                idx=int(children[j,1 if rng.random()<p else 0])
            y=idx
            if y==1:continue
        else:
            # Top-k lexical sampling + actual unknown-leaf probability, for a
            # controllable continuation similar to the original R5 generator.
            ids=[(idx,score) for idx,score in topk(m,h,max(top_k+4,18)) if idx>1]
            ids=ids[:top_k]
            if all(idx!=0 for idx,_ in ids):ids.append((0,candidate_score(m,h,0)))
            logits=np.array([score for idx,score in ids],dtype=np.float64)/temp
            for k,(idx,_) in enumerate(ids):
                if idx in recent[-16:] and idx>6:logits[k]-=.7
            logits-=logits.max();p=np.exp(logits);p/=p.sum()
            y=ids[int(rng.choice(len(ids),p=p))][0]
        if y==0:
            text=sample_char_word(h,char,rng,wordset,temp=temp,only_novel=True)
            if text is None:
                text='<unk>'
            else:
                fallbacks+=1
            hist.append(0)
        else:
            text=m['vocab'][y];hist.append(y)
        recent.append(y)
        out.append(text)
    return re.sub(r'\s+([.,!?;:])',r'\1',' '.join(out)),fallbacks

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--char',type=Path,required=True);p.add_argument('--prompt',default='the little house beside the sea')
    p.add_argument('--n',type=int,default=65);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--temp',type=float,default=.9);p.add_argument('--top-k',type=int,default=60)
    p.add_argument('--force-first-character',action='store_true');p.add_argument('--full-tree',action='store_true')
    a=p.parse_args();m=load(a.weights,a.root);L=read_long(a.context,m['updates'],len(m['vocab']))
    char=read_char(a.char,m['updates'],len(m['vocab']))
    text,n=generate(m,L,char,a.prompt,a.n,a.seed,a.temp,a.top_k,a.force_first_character,a.full_tree)
    print(f'CHECKPOINT step={m["updates"]} vocabulary={len(m["vocab"])} context=2048 char_fallback_active=true sampled_novel_words={n}')
    print(text)

if __name__=='__main__':main()
