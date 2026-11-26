"""v22: 256-class latent semantic grammar, full-vocabulary emission law.
Corpus-derived distributional geometry; no extra neural gradient updates.
v20 baseline reproduced, new class-only and contextual emission routes compared."""
import hybrid20 as V, hybrid18 as U, hybrid15 as H, base as B, foundation6 as N
import numpy as np
import argparse, bisect, collections as C, functools, json, math, pathlib, time

K=256
SCHEMAS=(
 ("c1",((1,1),)),("c2",((2,1),(1,1))),("c3",((3,1),(2,1),(1,1))),
 ("w1",((1,0),)),("cw",((2,1),(1,0))),("wc",((2,0),(1,1))),
 ("w2",((2,0),(1,0))),("gap",((3,1),(1,0))))
class Semantic:
    def __init__(self,docs,mem):
        t=time.monotonic();self.mem=mem;self.words=mem.words;self.ids=mem.ids;self.V=len(self.words)
        counts=C.Counter(w for d in docs for w in d)
        freq=np.asarray([0]+[counts[w] for w in self.words],np.float64)
        self.freq=freq
        seq=[np.asarray([self.ids[w] for w in d],np.uint32) for d in docs]
        # Distributional point coordinates: 64 left and 64 right anchor words.
        anchors=[self.ids[w] for w,_ in sorted(counts.items(),key=lambda p:(-p[1],p[0]))[:64]]
        lookup=np.zeros(self.V+1,np.int32)
        for i,v in enumerate(anchors,1):lookup[v]=i
        hist=np.zeros((self.V+1,128),np.float32)
        for d in seq:
            for off,word,anchor in ((0,d[1:],lookup[d[:-1]]),(64,d[:-1],lookup[d[1:]])):
                ok=anchor>0;np.add.at(hist,(word[ok],off+anchor[ok]-1),1)
        base=(hist.sum(axis=0,dtype=np.float64)+1)/(sum(map(len,seq))+128)
        # Smoothed positive PMI, with shrinkage towards background for rare types.
        prob=(hist+8*base)/(freq[:,None]+8)
        x=np.maximum(0,np.log(prob/base)).astype(np.float32)
        x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-6)
        del hist,prob
        # 64 singleton high-frequency words, 176 learned role clusters,
        # 16 hashed spelling-role fallback classes for very rare words.
        singleton=sorted(range(1,self.V+1),key=lambda i:(-freq[i],self.words[i-1]))[:64]
        mask=np.ones(self.V+1,bool);mask[0]=False;mask[singleton]=False
        candidates=np.flatnonzero(mask&(freq>=5))
        assert len(candidates)>=176
        rng=np.random.default_rng(2201)
        seeds=rng.choice(candidates,176,replace=False)
        centers=x[seeds].copy()
        # Weighted spherical k-means, deterministic full assignment of common
        # vocabulary, bounded square-root frequency weights.
        w=np.minimum(16,np.sqrt(freq[candidates])).astype(np.float32)
        xx=x[candidates];trace=[]
        for iteration in range(8):
            scores=xx@centers.T;labels=scores.argmax(axis=1)
            trace.append(float(np.average(scores[np.arange(len(xx)),labels],weights=w)))
            sums=np.zeros_like(centers);np.add.at(sums,labels,xx*w[:,None])
            norm=np.linalg.norm(sums,axis=1,keepdims=True)
            centers=np.where(norm>1e-6,sums/np.maximum(norm,1e-6),centers)
        cls=np.zeros(self.V+1,np.uint32)
        for k,i in enumerate(singleton):cls[i]=k
        cls[candidates]=64+(xx@centers.T).argmax(axis=1)
        for i in np.flatnonzero(mask&(freq<5)):
            cls[i]=240+B.addr("v22shape:"+U.S.shape(self.words[i-1]))%16
        self.cls=cls;self.centers=centers;self.trace=trace
        self.class_total=np.bincount(cls[1:],weights=freq[1:],minlength=K)
        self.prior=(self.class_total+.5)/(self.class_total.sum()+.5*K)
        self.emission=(freq[1:]+.25)/np.bincount(cls[1:],weights=freq[1:]+.25,minlength=K)[cls[1:]]
        self.class_active=np.bincount(cls[1:],minlength=K)>0
        self.tables={};self.entries=0
        for name,schema in SCHEMAS:
            depth=max(l for l,r in schema);keys=[];vals=[]
            for d in seq:
                if len(d)<=depth:continue
                n=len(d)-depth
                # class+1 reserves zero for unknown lexical identity.
                a=[(cls[d]+1 if r else d)[depth-l:len(d)-l] for l,r in schema]
                keys.append(V.vector_hash(a,n));vals.append(cls[d[depth:]].astype(np.uint16))
            k=np.concatenate(keys);v=np.concatenate(vals)
            order=np.lexsort((v,k));k=k[order];v=v[order]
            change=np.empty(len(k),bool);change[0]=True
            change[1:]=(k[1:]!=k[:-1])|(v[1:]!=v[:-1])
            ix=np.flatnonzero(change);c=np.diff(np.append(ix,len(k))).astype(np.uint32)
            self.tables[name]=(k[ix],v[ix],c);self.entries+=len(ix)
        self.seconds=time.monotonic()-t
    def classof(self,w):
        i=self.ids.get(w)
        return int(self.cls[i]) if i else 240+B.addr("v22shape:"+U.S.shape(w))%16
    @functools.lru_cache(maxsize=32768)
    def row(self,name,key):
        k,v,c=self.tables[name]
        a=int(np.searchsorted(k,key,"left"));b=int(np.searchsorted(k,key,"right"))
        if a==b:return None
        q=np.zeros(K,np.float64);q[v[a:b]]=c[a:b]
        return q
    def classlaw(self,h):
        p=self.prior.copy()
        for name,schema in SCHEMAS:
            if len(h)<max(l for l,r in schema):continue
            values=[self.classof(h[-l])+1 if r else self.ids.get(h[-l],0) for l,r in schema]
            if not all(values):continue
            q=self.row(name,V.scalar_hash(values))
            if q is None:continue
            n=q.sum();d=np.maximum(0,q-.5);s=d.sum()
            # Broad class contexts back off gently; rare exact contexts
            # must retain uncertainty rather than collapse to a singleton.
            mass=min(.85,s/(n+12))
            if s:p=(1-mass)*p+mass*d/s
        p[~self.class_active]=0;p/=p.sum()
        return p
    def law(self,h,base,contextual,classprob=None):
        p=self.classlaw(h) if classprob is None else np.asarray(classprob,dtype=np.float64)
        assert abs(p.sum()-1)<1e-9 and np.all(p[~self.class_active]==0)
        d=p[self.cls[1:]]*self.emission
        if contextual:
            # Within-class contextual emission: allocate each class's probability
            # among its words using actual observed lexical/role follow counts.
            # This preserves class marginals and avoids multiplying correlated laws.
            evidence=np.zeros(self.V,np.float64)
            for name,schema,weight in V.SCHEMAS:
                if len(h)<max(l for l,r in schema):continue
                values=[self.mem.roleof(h[-l]) if r else self.ids.get(h[-l],0) for l,r in schema]
                if not all(values):continue
                row,total,distinct=self.mem.row(name,V.scalar_hash(values))
                if not total:continue
                for word,n in row.items():
                    j=self.ids[word]-1
                    evidence[j]+=weight*max(0,n-.65)/(total+8+distinct)
            # Mix within each class, keeping the class probability exactly.
            csum=np.bincount(self.cls[1:],weights=evidence,minlength=K)
            s=csum[self.cls[1:]]
            lam=np.minimum(.80,s/(s+.10))
            d=(1-lam)*d+lam*p[self.cls[1:]]*evidence/np.maximum(s,1e-300)
            actual=np.bincount(self.cls[1:],weights=d,minlength=K)
            assert np.max(np.abs(actual-p))<1e-9
        assert abs(d.sum()-1)<1e-9 and d.min()>=0
        return Dense(self,d,base)
    def report(self):
        return {"classes":K,"active_classes":int(self.class_active.sum()),"vocabulary":self.V,"singleton_classes":64,"cluster_trace":self.trace,"class_pairs":self.entries,"index_bytes":sum(x.nbytes for row in self.tables.values() for x in row),"build_seconds":self.seconds}

class Dense:
    """Exact dense lexical prefix masses with a full-support signed escape."""
    def __init__(self,sem,d,base):
        self.sem=sem;self.d=.94*d;self.cum=np.r_[0.,np.cumsum(self.d)];self.base=H.BasePrefix(base)
    def complete(self,w):
        i=self.sem.ids.get(w)
        return .06*self.base.complete(w)+(float(self.d[i-1]) if i else 0.)
    def mass(self,p):
        a=bisect.bisect_left(self.sem.words,p);b=bisect.bisect_right(self.sem.words,p+"{")
        return .06*self.base.mass(p)+max(0.,float(self.cum[b]-self.cum[a]))

def laws(ref,small,large,sem,state,cal,oldprior):
    original,key=V.branches(ref,small,large,state,cal)
    old=H.Mixture(original,oldprior[1].get(key,oldprior[0]))
    # reuse the signed branch from the expanded memory rather than rerunning the field.
    base=original[1].law
    return (old,sem.law(state.h,base,False),sem.law(state.h,base,True)),key

def fit(ref,small,large,sem,cal,oldprior,docs):
    rows=C.defaultdict(list)
    for doc in docs:
        state=H.Replay()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                q,k=laws(ref,small,large,sem,state,cal,oldprior)
                p=[l.complete(w) for l in q];rows[("all",)].append(p);rows[k].append(p)
            state.observe(w)
    prior=U.em(rows[("all",)],[.70,.10,.20],pseudocount=24)
    return prior,{k:U.em(v,prior,pseudocount=64) for k,v in rows.items() if k!=("all",)}
def evaluate(ref,small,large,sem,cal,oldprior,prior,docs,limit):
    ms=C.defaultdict(C.Counter);oracle=C.Counter();t=time.monotonic()
    for doc in docs:
        state=H.Replay()
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                q,k=laws(ref,small,large,sem,state,cal,oldprior)
                weights=prior[1].get(k,prior[0])
                for name,ws in (("v20",(1,0,0)),("class_only",(0,1,0)),("class_context",(0,0,1)),("hybrid",weights),("fixed",(0.70,0.05,0.25))):
                    l=H.Mixture(q,ws);rank,cert,n=l.rank(budget=2000)
                    m=ms[name];m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.
                    for c in w+B.EOS:
                        d=l.next(p);prod*=d[c]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                # Oracle of three top-32 lists, evaluated separately from output.
                union=set()
                for j,a in enumerate(q):
                    rank,cert,n=H.Mixture((a,),(1.,)).rank(32,budget=5000)
                    oracle["cert"+str(j)]+=cert;oracle["hit"+str(j)]+=w in rank;union.update(rank)
                oracle["n"]+=1;oracle["union32"]+=w in union
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
    sem=Semantic(expanded,large);print("SEMANTIC",json.dumps(sem.report()),flush=True)
    prior=fit(ref,small,large,sem,cal,oldprior,cdocs)
    frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(data):
        s=H.Replay()
        for w in data[:16]:s.observe(w)
        q,k=laws(ref,small,large,sem,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"pilot":pilot,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
    out["stories"]=evaluate(ref,small,large,sem,cal,oldprior,prior,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
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
        out["fresh"][k]=evaluate(ref,small,large,sem,cal,oldprior,prior,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert before==ref.params.payload() and frozen==json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    out["signed_unchanged"]=True;out["graph"]=dict(ref.diag);out["fresh_ids"]=ids
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
