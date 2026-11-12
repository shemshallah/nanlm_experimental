#!/usr/bin/env python3
"""PAL5 autoregressive NOW generation; exactly one shared forward/infill decoder.

Causal mode reads only past words. Speculative mode predicts its own future via
causal rollouts and re-scores candidates using that *imagined* future; no actual
future tokens are read from held-out documents during generation.
"""
import argparse, heapq, json, re
from pathlib import Path
import numpy as np
D,R,NP,NH=96,64,32640,256
RGX=re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]",re.I)

def load(path:Path,root:Path):
    v=json.loads((root/'vocab.json').read_text())
    raw=np.fromfile(path,dtype=np.uint8)
    hdr=np.frombuffer(raw,dtype='<i4',count=6)
    if hdr.tolist()[:5]!=[0x50414c35,len(v),D,R,NH]:raise ValueError(f'wrong PAL5: {hdr}')
    x=np.frombuffer(raw,dtype='<f4',offset=24);at=0
    def get(n,shape):
        nonlocal at
        a=x[at:at+n].reshape(shape);at+=n;return a
    V=len(v);m=dict(vocab=v, lookup={w:i for i,w in enumerate(v)}, updates=int(hdr[5]))
    for k,n,shape in [('E',V*D,(V,D)),('H',NH*D,(NH,D)),('Splus',NP*R,(NP,R)),('Sminus',NP*R,(NP,R)),('Q',D*R,(D,R)),('NOW',D,(D,)),('W',(V-1)*D,(V-1,D)),('B',V-1,(V-1,))]:m[k]=get(n,shape)
    assert at==len(x)
    m['heads']=np.fromfile(root/'token_heads.u16',dtype='<u2').reshape(V,2)
    m['children']=np.fromfile(root/'tree.i32',dtype='<i4').reshape(V-1,2)
    return m

def pair_index(a,b):
    if a>b:a,b=b,a
    return a*(511-a)//2+b-a-1

def hidden(m,past,future=None,rank=R):
    assert len(past)==3
    E,H,Q,NOW=m['E'],m['H'],m['Q'],m['NOW']
    f=np.zeros(R,dtype=np.float32)
    z=NOW.copy()
    for side,context in enumerate((past,future)):
        if context is None:continue
        ids=np.array(context,dtype=np.int64)
        if len(ids)!=3:raise ValueError('context must have three tokens')
        a,b=map(int,(m['heads'][ids[2],0],m['heads'][ids[1],1]))
        if a==b:b=(b+1)&255
        p=pair_index(a,b)
        sign=(1 if a<b else -1)*(-1 if side else 1)
        primary,other=(m['Splus'],m['Sminus']) if side==0 else (m['Sminus'],m['Splus'])
        f[:rank]+=np.float32(.4*sign)*(primary[p,:rank]-.2*other[p,:rank])
        weight=1. if side==0 else .32
        z+=weight*(.58*E[ids[2]]+.28*E[ids[1]]+.14*E[ids[0]])
        z+=weight*(.18*(H[a]-H[b])+.04*(H[(a+1)&255]+H[(a+2)&255]-H[(b+255)&255]-H[(b+254)&255]))
    z+=Q@f
    return np.tanh(z).astype(np.float32)

def topk(m,h,k=40):
    children,W,B=m['children'],m['W'],m['B'];V=len(m['vocab'])
    heap=[(0.,2*V-2)];out=[]
    while heap and len(out)<k:
        neg,id=heapq.heappop(heap)
        if id<V:
            out.append((id,-neg));continue
        n=id-V;score=float(W[n]@h+B[n]);l,r=children[n]
        pl=-float(np.logaddexp(0.,score));pr=-float(np.logaddexp(0.,-score))
        heapq.heappush(heap,(neg-pl,int(l)));heapq.heappush(heap,(neg-pr,int(r)))
    return out

def candidate_score(m,h,y):
    # Exact hierarchical log probability, no separate reverse head.
    V=len(m['vocab']);nodes=m['children']; parent=np.zeros(2*V-1,dtype=np.int32);bit=np.zeros(2*V-1,dtype=np.int32)
    for j,(l,r) in enumerate(nodes):parent[l]=V+j;parent[r]=V+j;bit[r]=1
    logp=0.;k=y
    while k!=2*V-2:
        node=int(parent[k])-V;score=float(m['W'][node]@h+m['B'][node]);b=int(bit[k]);
        logp-=float(np.logaddexp(0.,-score if b else score));k=node+V
    return logp

def predict_future(m,past,y):
    ctx=list(past)+[y];f=[]
    for _ in range(3):
        nxt=next((i for i,_ in topk(m,hidden(m,ctx[-3:]),12) if i>1),3)
        f.append(nxt);ctx.append(nxt)
    return f

def generate(m,prompt,n=80,temp=.86,k=50,seed=17,imagined=False,beta=.1):
    rng=np.random.default_rng(seed);w=RGX.findall(prompt.lower());ctx=([1,1,1]+[m['lookup'].get(t,0) for t in w])[-3:]
    out=w[:];recent=[]
    for t in range(n):
        h=hidden(m,ctx);c=[(i,s) for i,s in topk(m,h,max(60,k)+3) if i>1][:max(60,k)]
        if imagined:
            amended=[]
            for i,score in c:
                f=predict_future(m,ctx,i)
                hf=hidden(m,ctx,future=f[::-1])
                score+=beta*(candidate_score(m,hf,i)-candidate_score(m,h,i))
                amended.append((i,score))
            c=sorted(amended,key=lambda item:-item[1])
        c=c[:k];logits=np.array([s for _,s in c],dtype=np.float64)/temp
        for j,(i,_) in enumerate(c):
            if i in recent[-16:] and i>6:logits[j]-=.7
        logits-=logits.max();probs=np.exp(logits);probs/=probs.sum()
        y=c[int(rng.choice(len(c),p=probs))][0]
        out.append(m['vocab'][y]);recent.append(y);ctx=ctx[1:]+[y]
    return re.sub(r'\s+([.,!?;:])',r'\1',' '.join(out))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).parent)
    parser.add_argument('--weights',type=Path,default=Path(__file__).parent/'weights_now64_13m.raw')
    parser.add_argument('--prompt',default='she looked at')
    parser.add_argument('--n',type=int,default=80);parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--temp',type=float,default=.85);parser.add_argument('--top-k',type=int,default=50)
    parser.add_argument('--imagined',action='store_true')
    a=parser.parse_args();m=load(a.weights,a.root)
    print(generate(m,a.prompt,a.n,a.temp,a.top_k,a.seed,a.imagined))