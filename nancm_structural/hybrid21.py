"""v21 causal long-document associative memory and paired history counterfactual.
Same prediction targets with short, correct long, and unrelated old history.
No additional gradient training beyond deterministic v17 bounded head replay."""
import hybrid18 as U, hybrid15 as H, foundation8 as E, foundation7 as S
import base as B, foundation6 as N
import argparse, collections as C, math, pathlib, json, time
import numpy as np

class Document(H.Replay):
    def __init__(self,ref):
        super().__init__();self.ref=ref;self.all=[];self.post=C.defaultdict(list);self.rolepost=C.defaultdict(list)
    def observe(self,w):
        # At target position i, the context ending at i-1 is already observed.
        # Add the address only AFTER the target at i has been observed.
        i=len(self.all)
        if i:
            last=self.all[-1];self.post[last].append(i);self.rolepost[S.shape(last)].append(i)
        self.all.append(w);super().observe(w)
    def retrieve(self,base):
        h=self.all;n=len(h)
        if n<2:return E.FastLattice(base.ref,dict(base.s),base.bg),0.,0
        candidates=set(self.post.get(h[-1],[])[-128:])
        # Anchor in a discontinuous history position, then propose the word
        # at the corresponding historical target boundary.
        for lag in (2,3,4,6):
            if n<lag:continue
            anchor=h[-lag]
            for i in self.post.get(anchor,[])[-48:]:
                j=i+lag-1
                if j<n:candidates.add(j)
        if len(candidates)<16:
            candidates.update(self.rolepost.get(S.shape(h[-1]),[])[-48:])
        scored=[]
        for i in candidates:
            assert 0<i<n
            depth=min(12,i,n);sim=0.;exact=0
            for lag in range(1,depth+1):
                a,b=h[-lag],h[i-lag]
                weight=math.exp(-(lag-1)/4)
                if a==b:
                    rare=min(2.5,1+3/math.sqrt(1+self.ref.uni[a]))
                    sim+=weight*rare;exact+=1
                elif S.shape(a)==S.shape(b) and S.shape(a)!="W":
                    sim+=.12*weight
            if exact<1:continue
            recency=1/(1+(n-i)/512)**.25
            scored.append((sim*recency,exact,i))
        best=sorted(scored,reverse=True)[:32];q=C.Counter()
        for score,exact,i in best:
            # One exact token alone does not authorize strong prediction.
            q[h[i]]+=math.expm1(min(5.,score))*min(1.,exact/2)
        strength=math.fsum(q.values());peak=best[0][0] if best else 0.
        lam=min(.75,strength/(strength+20)*min(1.,peak/3))
        s={w:(1-lam)*p for w,p in base.s.items()}
        B.add(s,B.norm(q),lam);bg=(1-lam)*base.bg
        return E.FastLattice(base.ref,s,bg),lam,len(best)

def seed(ref,history):
    s=Document(ref)
    for w in history:s.observe(w)
    return s
def predictions(ref,cal,state):
    six,key,base=ref.six(state);old=H.Mixture(six,cal.weights(key))
    d,lam,n=state.retrieve(base)
    return old,H.Mixture((H.BasePrefix(d),),(1.,)),lam,n
def mix(a,b,p):return H.Mixture((a,b),(1-p,p))

def fit(ref,cal,docs):
    rows=C.defaultdict(list)
    for d in docs:
        if len(d)<544:continue
        s=seed(ref,d[:512])
        for w in d[512:544]:
            a,b,lam,_=predictions(ref,cal,s);p=[a.complete(w),b.complete(w)]
            rows[("all",)].append(p);rows[(int(lam>.25),)].append(p)
            s.observe(w)
    assert rows[("all",)]
    prior=U.em(rows[("all",)],[.75,.25],pseudocount=24)
    local={k:U.em(v,prior,pseudocount=48) for k,v in rows.items() if k!=("all",)}
    return prior,local,len(rows[("all",)])
def evaluate(ref,cal,prior,docs,limit=32,warm=512):
    global_q,local,_=prior;metrics=C.defaultdict(C.Counter);diag=C.Counter();start=time.monotonic()
    for j,doc in enumerate(docs):
        other=docs[(j+1)%len(docs)];assert len(doc)>=warm+limit and len(other)>=warm
        full=seed(ref,doc[:warm]);wrong=seed(ref,other[:warm-16]+doc[warm-16:warm])
        for i,w in enumerate(doc[warm:warm+limit]):
            short=seed(ref,doc[warm+i-16:warm+i])
            a,b,lam,n=predictions(ref,cal,full)
            c,d,wl,wn=predictions(ref,cal,wrong)
            e,_,_,_=predictions(ref,cal,short)
            q=local.get((int(lam>.25),),global_q)[1]
            wq=local.get((int(wl>.25),),global_q)[1]
            laws={"short16":e,"full512":a,"retrieval":b,"hybrid":mix(a,b,q),"unrelated_old":c,"hybrid_unrelated_old":mix(c,d,wq)}
            diag["n"]+=1;diag["neighbors"]+=n;diag["lambda"]+=lam;diag["wrong_neighbors"]+=wn
            for mode,l in laws.items():
                m=metrics[mode];rank,cert,nodes=l.rank(budget=1000)
                m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                p="";prod=1.
                for c0 in w+B.EOS:
                    v=l.next(p);prod*=v[c0]
                    if c0!=B.EOS:p+=c0
                assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            full.observe(w);wrong.observe(w)
    return {"modes":{k:dict(v,word_top1=v["t1"]/v["n"],word_top5=v["t5"]/v["n"],spelling_bps=v["bits"]/v["symbols"]) for k,v in metrics.items()},"diag":dict(diag),"seconds":time.monotonic()-start}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];calbooks=books[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(calbooks)==12 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=U.Unified(docs,lex,N.Native(a.lib));before=ref.params.payload()
    pilot=ref.head.train(ref.head.data([d[:512] for d in docs]))
    caldocs=[B.load(p,1024) for _,p in calbooks];cal=U.fit(ref,caldocs);prior=fit(ref,cal,caldocs)
    frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()},prior[2]],sort_keys=True)
    # New 18 distinct book IDs. Avoid the historical five per group and the
    # v20 fresh subset to obtain a different long-window probe.
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]}
    for k,files in groups.items():
        count=0
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);count+=1
            if count==6:break
    out={"pilot":pilot,"reference_ids":[i for i,p in tr],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"fresh":{}}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v21-long:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(36000).decode("utf-8","replace"))
            if len(w)<544:continue
            used.add(i);chosen.append((i,w[:544]))
            if len(chosen)==6:break
        assert len(chosen)==6
        # Suffix changes must not affect the distribution at boundary 512.
        d=chosen[0][1]
        def at(data):
            state=seed(ref,data[:512]);x,y,lam,n=predictions(ref,cal,state)
            return x.rank(),y.rank(),lam,n
        assert at(d)==at(d[:512]+["other","future"])
        out["fresh"][k]=evaluate(ref,cal,prior,[w for i,w in chosen])
        out["fresh"][k]["ids"]=[i for i,w in chosen]
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert before==ref.params.payload() and frozen==json.dumps([prior[0],{str(k):v for k,v in prior[1].items()},prior[2]],sort_keys=True)
    assert not {i for i,p in tr+calbooks}&used
    out["signed_unchanged"]=True;out["suffix_invariance"]=True;out["graph"]=dict(ref.diag)
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
