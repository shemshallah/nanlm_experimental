"""v17 disposable bounded donor-style NWP diagnostic.
Reference-only minibatch pilot; no signed parameter changes or production promotion.
Full-support OOV escape and normalized mixture, exact ranking and frozen DEV."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="2"
os.environ["OMP_NUM_THREADS"]="2"
import base as B, foundation6 as N, foundation8 as E, hybrid15 as H
import numpy as np
import argparse, collections as C, functools, heapq, json, math, pathlib, time

class Neural:
    def __init__(self,docs,vocab=2048,dim=32,seed=2718):
        self.rng=np.random.default_rng(seed);self.dim=dim
        counts=C.Counter(w for d in docs for w in d)
        self.words=sorted(counts,key=lambda w:(-counts[w],w))[:vocab]
        self.ids={w:i for i,w in enumerate(self.words)}
        self.V=len(self.words);self.oov=self.V;self.bos=self.V+1;self.H=4096
        r=self.rng
        self.p={"E1":r.normal(0,.12,(self.V+2,dim)).astype(np.float32),
                "E2":r.normal(0,.12,(self.V+2,dim)).astype(np.float32),
                "E3":r.normal(0,.12,(self.H,dim)).astype(np.float32),
                "T":r.normal(0,.06,(dim,dim)).astype(np.float32),
                "bias":np.zeros(dim,np.float32),
                "U":r.normal(0,.04,(dim,self.V+1)).astype(np.float32),
                "out":np.log(np.array([counts[w]+1 for w in self.words]+[sum(n for w,n in counts.items() if w not in self.ids)+1],np.float64)).astype(np.float32)}
        self.p["out"]-=self.p["out"].mean()
        self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.step=0
    def wid(self,w):return self.ids.get(w,self.oov)
    @functools.lru_cache(maxsize=65536)
    def topicword(self,w):
        a=np.zeros(self.dim,np.float32)
        for j in range(4):
            z=B.addr("topic:"+str(j)+":"+w);a[z%self.dim]+=1 if z>>32&1 else -1
        return a/2
    def features(self,h):
        a=self.wid(h[-1]) if h else self.bos;b=self.wid(h[-2]) if len(h)>=2 else self.bos
        c=B.addr(str(b)+":"+str(a))%self.H
        t=np.zeros(self.dim,np.float32)
        for w in h[-64:]:t=.95*t+.05*self.topicword(w)
        return a,b,c,t
    def data(self,docs):
        rows=[]
        for d in docs:
            for i,w in enumerate(d):
                a,b,c,t=self.features(d[max(0,i-64):i]);rows.append((a,b,c,t,self.wid(w)))
        return (np.asarray([x[0] for x in rows]),np.asarray([x[1] for x in rows]),np.asarray([x[2] for x in rows]),np.stack([x[3] for x in rows]),np.asarray([x[4] for x in rows]))
    def forward(self,a,b,c,t):
        p=self.p;h=np.tanh(p["E1"][a]+p["E2"][b]+p["E3"][c]+t@p["T"]+p["bias"])
        z=h@p["U"]+p["out"];z-=z.max(axis=1,keepdims=True)
        q=np.exp(z);q/=q.sum(axis=1,keepdims=True)
        return h,q
    def loss(self,data):
        a,b,c,t,y=data;_,q=self.forward(a,b,c,t)
        return float(-np.log(np.maximum(q[np.arange(len(y)),y],1e-30)).mean())
    def train(self,data,steps=512,batch=32):
        a,b,c,t,y=data;trace=[];start=time.monotonic()
        for j in range(steps):
            ix=self.rng.integers(len(y),size=batch);aa,bb,cc,tt,yy=a[ix],b[ix],c[ix],t[ix],y[ix]
            h,q=self.forward(aa,bb,cc,tt)
            loss=float(-np.log(np.maximum(q[np.arange(batch),yy],1e-30)).mean())
            q[np.arange(batch),yy]-=1;q/=batch
            dh=(q@self.p["U"].T)*(1-h*h)
            g={k:np.zeros_like(v) for k,v in self.p.items()}
            g["U"]=h.T@q;g["out"]=q.sum(axis=0);g["bias"]=dh.sum(axis=0);g["T"]=tt.T@dh
            for name,idx in (("E1",aa),("E2",bb),("E3",cc)):np.add.at(g[name],idx,dh)
            self.step+=1
            for k in self.p:
                self.m[k]=.9*self.m[k]+.1*g[k];self.v[k]=.999*self.v[k]+.001*g[k]*g[k]
                self.p[k]-=.003*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
            if j%64==0 or j+1==steps:trace.append([j+1,loss])
        return {"steps":steps,"sampled_targets":steps*batch,"batch":batch,"trace":trace,"seconds":time.monotonic()-start}
    def probs(self,h):
        a,b,c,t=self.features(h)
        _,q=self.forward(np.array([a]),np.array([b]),np.array([c]),t[None,:])
        q=q[0].astype(np.float64);q/=q.sum()
        return q

class Escape:
    """Known-word probabilities plus OOV mass on the background conditioned
    on NOT being a member of the neural vocabulary. Prefix mass is exact."""
    def __init__(self,base,head,h):
        self.ref=base.ref;self.b=H.BasePrefix(base);self.head=head
        self.q=head.probs(h);self.s={w:float(self.q[i]) for i,w in enumerate(head.words)}
        self.knownbase={w:base.complete(w) for w in head.words}
        self.z=1-math.fsum(self.knownbase.values());assert self.z>0
        self.scale=float(self.q[-1])/self.z
        self.delta={w:p-self.scale*self.knownbase[w] for w,p in self.s.items()}
        self.dp=C.Counter()
        for w,v in self.delta.items():
            for i in range(len(w)+1):self.dp[w[:i]]+=v
        assert abs(self.mass("")-1)<1e-8
    def mass(self,p):return max(0.,self.scale*self.b.mass(p)+self.dp[p])
    def complete(self,w):return self.s[w] if w in self.s else self.scale*self.b.complete(w)

class Pilot:
    def __init__(self,ref,head):self.ref=ref;self.head=head
    def branches(self,state):
        h=state.h;_,_,s,_=self.ref.hypotheses(h[-64:],state.ep,h[-128:],1.)
        n=Escape(s,self.head,h)
        return s,n
    def fit_gate(self,docs):
        scores=C.Counter();n=0
        for d in docs:
            s=H.Replay()
            for i,w in enumerate(d[:48]):
                if i>=16:
                    base,neural=self.branches(s)
                    p=base.complete(w);q=neural.complete(w)
                    for weight in (0.,.05,.15,.30,.50,.75,1.):scores[str(weight)]-=math.log(max((1-weight)*p+weight*q,1e-300))
                    n+=1
                s.observe(w)
        chosen=min(scores,key=lambda k:(scores[k],float(k)))
        return float(chosen),{"targets":n,"nll":{k:v/n for k,v in scores.items()}}
    def evaluate(self,docs,limit,weight):
        modes=("signed","neural","fixed_15","calibrated","causal")
        ms={k:C.Counter() for k in modes};start=time.monotonic()
        for doc in docs:
            state=H.Replay();odds=0.
            for i,w in enumerate(doc[:limit+16]):
                base,neural=self.branches(state);bp=H.BasePrefix(base)
                causal=min(.75,max(.02,weight if weight else .05)*math.exp(odds)/(1-max(.02,weight if weight else .05)+max(.02,weight if weight else .05)*math.exp(odds)))
                weights=(0.,1.,.15,weight,causal)
                if i>=16:
                    for mode,q in zip(modes,weights):
                        l=H.Mixture((bp,neural),(1-q,q))
                        if q==0:rank,cert,n=base.rank()
                        else:rank,cert,n=l.rank(budget=600)
                        m=ms[mode];m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                        m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1;m["oov"]+=w not in self.head.ids
                        p="";prod=1.
                        for c in w+B.EOS:
                            d=l.next(p);prod*=d[c]
                            if c!=B.EOS:p+=c
                        assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                odds=max(-4.,min(4.,.94*odds+.35*math.log(max(neural.complete(w),1e-300)/max(base.complete(w),1e-300))))
                state.observe(w)
        return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"seconds":time.monotonic()-start}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];cal=books[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(cal)==12 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr]
    # v11 signed field is retained; no gradients flow into it in this probe.
    import continuation11 as P
    ref=P.Continuation(docs,lex,N.Native(a.lib));before=ref.params.payload()
    head=Neural(docs);data=head.data([d[:512] for d in docs]);caldocs=[B.load(p,200) for _,p in cal]
    cd=head.data([d[:96] for d in caldocs]);pre=head.loss(cd);train=head.train(data);post=head.loss(cd)
    print("PILOT",json.dumps({"calibration_head_nll_before":pre,"after":post,**train}),flush=True)
    p=Pilot(ref,head);weight,gate=p.fit_gate(caldocs)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5]
    out={"scope":"disposable neural-head pilot; signed parameters frozen; no full training","reference_ids":[i for i,p in tr],"calibration_ids":[i for i,p in cal],"head_vocab":head.V,"parameters":sum(x.size for x in head.p.values()),"pilot":train,"head_calibration_nll_before":pre,"head_calibration_nll_after":post,"gate":gate,"gate_weight":weight,"fresh":{}}
    out["stories"]=p.evaluate([B.load(path,300) for _,path in picked],72,weight);print("STORIES",json.dumps(out["stories"]),flush=True)
    used={i for files in groups.values() for i,path in [x for x in files if split(x[0]) in (7,8)][:5]};ids={}
    for k,files in groups.items():
        chosen=[]
        for i,path in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with path.open("rb") as f:
                f.seek(min(path.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;ids[k]=[i for i,w in chosen];out["fresh"][k]=p.evaluate([w for i,w in chosen],32,weight)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert ref.params.payload()==before;out["signed_unchanged"]=True;out["fresh_ids"]=ids;out["graph"]=dict(ref.diag)
    np.savez_compressed(a.out.with_name("disposable-head.npz"),words=np.array(head.words),step=np.array(head.step),**head.p,**{"m_"+k:v for k,v in head.m.items()},**{"v_"+k:v for k,v in head.v.items()})
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
