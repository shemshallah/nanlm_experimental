"""v23 signed latent class learning diagnostic, full-vocabulary contextual readout.
A separate bounded pilot, no mutation of inherited signed gains.
128 lower addresses x 128 upper points x 8 signed rails, trained on reference classes."""
import hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N
import numpy as np
import argparse, collections as C, functools, json, math, pathlib, time

RAILS=8
class Field:
    def __init__(self,sem):
        self.sem=sem;self.rng=np.random.default_rng(2301);self.step=0
        r=self.rng
        self.p={"plus":r.uniform(.005,.035,(8,128,128)).astype(np.float32),
                "minus":r.uniform(.005,.035,(8,128,128)).astype(np.float32),
                "read":r.normal(0,.05,(128,256)).astype(np.float32),
                "bias":np.log(np.maximum(sem.prior,1e-12)).astype(np.float32)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.rails=np.arange(8)[None,:]
    @staticmethod
    @functools.lru_cache(maxsize=131072)
    def address(rail,key):
        return B.addr(str(rail)+":"+repr(key))%128
    def features(self,h):
        c=[self.sem.classof(w) for w in h[-6:]]
        def w(n):return h[-n] if len(h)>=n else ""
        def a(n):return c[-n] if len(c)>=n else -1
        keys=(a(1),(a(2),a(1)),(a(3),a(2),a(1)),w(1),(w(2),w(1)),
              (w(3),w(1)),(a(4),a(2),a(1)),(a(6),a(3),w(1)))
        return [self.address(i,k) for i,k in enumerate(keys)]
    def data(self,docs,cap=None):
        x=[];y=[]
        for doc in docs:
            for i,w in enumerate(doc if cap is None else doc[:cap]):
                x.append(self.features(doc[max(0,i-6):i]));y.append(self.sem.classof(w))
        return np.asarray(x,np.uint8),np.asarray(y,np.int64)
    def forward(self,x):
        p=self.p
        # Addressed cross-half pair rails. Reverse traversal has negative
        # orientation; support and refutation remain separate parameters.
        edge=p["plus"][self.rails,x]-p["minus"][self.rails,x]
        v=np.tanh(edge)
        h=np.tanh(v.sum(axis=1)/math.sqrt(8))
        logits=h@p["read"]+p["bias"]
        logits[:,~self.sem.class_active]=-1e9
        logits-=logits.max(axis=1,keepdims=True)
        q=np.exp(logits);q/=q.sum(axis=1,keepdims=True)
        return q,h,v
    def loss(self,data):
        x,y=data;total=0.
        for i in range(0,len(y),512):
            yy=y[i:i+512];q,_,_=self.forward(x[i:i+512])
            total-=float(np.log(np.maximum(q[np.arange(len(yy)),yy],1e-30)).sum())
        return total/len(y)
    def gradients(self,x,y):
        q,h,v=self.forward(x);n=len(y)
        loss=float(-np.log(np.maximum(q[np.arange(n),y],1e-30)).mean())
        q[np.arange(n),y]-=1;q/=n
        g={"read":h.T@q,"bias":q.sum(axis=0)}
        dh=(q@self.p["read"].T)*(1-h*h)
        dv=dh[:,None,:]*(1-v*v)/math.sqrt(8)
        gp=np.zeros_like(self.p["plus"])
        for j in range(8):np.add.at(gp[j],x[:,j],dv[:,j])
        g["plus"]=gp;g["minus"]=-gp
        return loss,g
    def check(self,data):
        x,y=data;x=x[:16];y=y[:16];loss,g=self.gradients(x,y)
        # Pick the strongest nonzero entry of each parameter group.
        errors={}
        for k in self.p:
            idx=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape)
            a=float(self.p[k][idx]);eps=.002
            self.p[k][idx]=a+eps;hi=self.loss((x,y))
            self.p[k][idx]=a-eps;lo=self.loss((x,y))
            self.p[k][idx]=a;numeric=(hi-lo)/(2*eps);analytic=float(g[k][idx])
            errors[k]={"numeric":numeric,"analytic":analytic,"abs_error":abs(numeric-analytic)}
            assert abs(numeric-analytic)<.002
        # Map cross-half edges into the original 32640 pair address space.
        addresses={B.pair(i,128+j) for i in range(128) for j in range(128)}
        assert len(addresses)==16384
        return errors
    def train(self,data,steps=1024,batch=128):
        x,y=data;trace=[];start=time.monotonic()
        for j in range(steps):
            ix=self.rng.integers(len(y),size=batch);loss,g=self.gradients(x[ix],y[ix])
            self.step+=1
            for k in self.p:
                self.m[k]=.9*self.m[k]+.1*g[k];self.v[k]=.999*self.v[k]+.001*g[k]*g[k]
                self.p[k]-=.004*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
                if k in ("plus","minus"):np.clip(self.p[k],0.,3.,out=self.p[k])
                assert np.isfinite(self.p[k]).all()
            if j%128==0 or j==steps-1:trace.append([j+1,loss])
        return {"steps":steps,"batch":batch,"sampled_targets":steps*batch,"available_targets":len(y),"trace":trace,"seconds":time.monotonic()-start}
    def probs(self,h):
        q,_,_=self.forward(np.asarray([self.features(h)],np.uint8))
        p=q[0].astype(np.float64);p/=p.sum()
        return p
    def save(self,path):
        # Save both projected Adam histories and the actual pair-rail address mapping.
        pair=np.array([[B.pair(i,128+j) for j in range(128)] for i in range(128)],np.uint16)
        np.savez_compressed(path,step=np.array(self.step),pairs=pair,words=np.array(self.sem.words),classes=self.sem.cls,centers=self.sem.centers,
          **self.p,**{"m_"+k:v for k,v in self.m.items()},**{"v_"+k:v for k,v in self.v.items()})

def branches(ref,small,large,sem,field,state,cal,oldprior):
    orig,key=V.branches(ref,small,large,state,cal)
    old=H.Mixture(orig,oldprior[1].get(key,oldprior[0]));base=orig[1].law
    count=sem.law(state.h,base,True)
    neural=sem.law(state.h,base,True,field.probs(state.h))
    return (old,count,neural),key

def fit(ref,small,large,sem,field,cal,oldprior,docs):
    rows=C.defaultdict(list)
    for doc in docs:
        s=H.Replay()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                laws,k=branches(ref,small,large,sem,field,s,cal,oldprior)
                p=[l.complete(w) for l in laws];rows[("all",)].append(p);rows[k].append(p)
            s.observe(w)
    q=U.em(rows[("all",)],[.70,.15,.15],pseudocount=24)
    return q,{k:U.em(v,q,pseudocount=64) for k,v in rows.items() if k!=("all",)}

def evaluate(ref,small,large,sem,field,cal,oldprior,prior,docs,limit):
    ms=C.defaultdict(C.Counter);oracle=C.Counter();t=time.monotonic()
    for doc in docs:
        state=H.Replay()
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                q,k=branches(ref,small,large,sem,field,state,cal,oldprior)
                for name,ws in (("v20",(1,0,0)),("count_class",(0,1,0)),("learned_field",(0,0,1)),("hybrid",prior[1].get(k,prior[0])),("fixed",(.65,.15,.20))):
                    l=H.Mixture(q,ws);rank,cert,n=l.rank(budget=2000)
                    m=ms[name];m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.
                    for c in w+B.EOS:
                        d=l.next(p);prod*=d[c]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                # Count distinct discoveries from the learned field.
                top=[]
                for a in q:
                    rank,cert,n=H.Mixture((a,),(1.,)).rank(32,budget=5000);top.append(set(rank));oracle["cert"]+=cert
                oracle["n"]+=1;oracle["old32"]+=w in top[0];oracle["field32"]+=w in top[2]
                oracle["novel32"]+=w in top[2] and w not in top[0]|top[1]
                oracle["union32"]+=w in top[0]|top[1]|top[2]
            state.observe(w)
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"oracle":dict(oracle),"seconds":time.monotonic()-t}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];calbooks=books[96:108];extra=books[108:396];lex=set()
    assert len(tr)==96 and len(calbooks)==12 and len(extra)==288
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];expanded=docs+[B.load(p,6000) for _,p in extra]
    ref=U.Unified(docs,lex,N.Native(a.lib));before=ref.params.payload()
    pilot=ref.head.train(ref.head.data([d[:512] for d in docs]))
    cdocs=[B.load(p,200) for _,p in calbooks];cal=U.fit(ref,cdocs)
    small=V.Memory(docs);large=V.Memory(expanded);oldprior=V.fit(ref,small,large,cal,cdocs)
    sem=Z.Semantic(expanded,large);field=Field(sem)
    # Bounded pilot across every expanded reference book, not full training.
    data=field.data(expanded,cap=768);valid=field.data(cdocs)
    gradcheck=field.check(data);initial={k:v.copy() for k,v in field.p.items()}
    pre=field.loss(valid);train=field.train(data);post=field.loss(valid)
    movement={k:float(np.linalg.norm(v-initial[k])) for k,v in field.p.items()}
    print("FIELD",json.dumps({"before":pre,"after":post,"movement":movement,"train":train,"gradient_check":gradcheck}),flush=True)
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"class_cal_nll_before":pre,"class_cal_nll_after":post,"movement":movement,"gradient_check":gradcheck,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
    out["stories"]=evaluate(ref,small,large,sem,field,cal,oldprior,prior,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]};ids={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;ids[k]=[i for i,w in chosen]
        out["fresh"][k]=evaluate(ref,small,large,sem,field,cal,oldprior,prior,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert before==ref.params.payload() and frozen==json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    out["signed_unchanged"]=True;out["graph"]=dict(ref.diag);out["fresh_ids"]=ids
    field.save(a.out.with_name("disposable-class-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
