"""v18 unified generative word hybrid: signed/BPE/mirror/replay/neural/skip-role.
Bounded deterministic v17 diagnostic replay; no inherited signed updates.
Mixture likelihood calibrated on disjoint reference-side books, never DEV."""
import hybrid17 as J
import base as B, foundation6 as N, foundation7 as S, foundation8 as E, hybrid15 as H
import argparse, collections as C, heapq, math, pathlib, json, time
import numpy as np

NAMES=("signed","pieces","mirror","replay","neural","skip_role")
def context_keys(h):
    if len(h)>=2:
        yield ("rolelast",S.shape(h[-2]),h[-1]),.45
        yield ("lastrole",h[-2],S.shape(h[-1])),.35
    if len(h)>=3:
        yield ("gap1",h[-3],h[-1]),.75
        yield ("role2",S.shape(h[-3]),h[-2],h[-1]),.65
    if len(h)>=4:yield ("gap2",h[-4],h[-1]),.55
    if len(h)>=5:yield ("gap3",h[-5],h[-1]),.40

class Skip:
    def __init__(self,docs):
        self.rows=C.defaultdict(C.Counter)
        for doc in docs:
            h=[]
            for w in doc:
                for k,_ in context_keys(h):self.rows[k][w]+=1
                h=(h+[w])[-5:]
        self.rows={k:v for k,v in self.rows.items() if sum(v.values())>=2}
    def law(self,h,base):
        dist={};strength=0.
        for k,weight in context_keys(h):
            row=self.rows.get(k)
            if not row:continue
            n=sum(row.values());distinct=len(row)
            # Evidence must survive discount and have repeated observations.
            q={w:max(0.,c-.75) for w,c in row.items()}
            support=math.fsum(q.values())
            if not support:continue
            trust=weight*support/(n+8+distinct)
            B.add(dist,B.norm(q),trust);strength+=trust
        lam=min(.65,strength/(1+strength))
        s={w:p*(1-lam) for w,p in base.s.items()}
        B.add(s,B.norm(dist),lam)
        return E.FastLattice(base.ref,s,(1-lam)*base.bg),lam

class Unified(H.Hybrid):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel)
        self.head=J.Neural(docs);self.skip=Skip(docs)
    def six(self,state):
        branches,key,base=self.branches(state)
        skip,strength=self.skip.law(state.h,base)
        neural=J.Escape(base,self.head,state.h)
        return (*branches,neural,H.BasePrefix(skip)),(*key,int(strength>.15)),base

def em(rows,prior,steps=40,pseudocount=12):
    q=np.asarray(prior,dtype=np.float64).copy()
    if not len(rows):return q.tolist()
    x=np.asarray(rows,dtype=np.float64)
    # rescale per observation to avoid underflow; relative likelihood invariant
    x=x/np.maximum(x.max(axis=1,keepdims=True),1e-300)
    for _ in range(steps):
        a=x*q;z=a.sum(axis=1,keepdims=True)
        assert np.all(z>0)
        q=((a/z).sum(axis=0)+pseudocount*np.asarray(prior))/(len(x)+pseudocount)
    q=.97*q+.03/len(q)
    return (q/q.sum()).tolist()

class Cal:
    def __init__(self):self.rows=C.defaultdict(list);self.global_weights=None;self.local={}
    def observe(self,key,probs):
        self.rows[("all",)].append(probs);self.rows[key].append(probs)
    def fit(self):
        self.global_weights=em(self.rows[("all",)],[.45,.12,.05,.18,.12,.08],pseudocount=18)
        self.local={k:em(v,self.global_weights,pseudocount=48) for k,v in self.rows.items() if k!=("all",)}
    def weights(self,key):return self.local.get(key,self.global_weights)
    def export(self):return {"global":self.global_weights,"buckets":{str(k):{"n":len(self.rows[k]),"weights":v} for k,v in self.local.items()}}

def fit(ref,docs):
    cal=Cal()
    for doc in docs:
        state=H.Replay()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                branches,key,_=ref.six(state);cal.observe(key,[l.complete(w) for l in branches])
            state.observe(w)
    cal.fit();return cal

def modes(branches,cal,key,logs):
    # Frozen baseline v15 mixture exactly matches previous static weights.
    old=[.65,.15,.10,.10,0.,0.]
    prior=cal.weights(key);z=B.norm({i:prior[i]*math.exp(logs[i]) for i in range(6)})
    adaptive=[.01+.94*z[i] for i in range(6)]
    no_skip=np.array(prior);no_skip[5]=0;no_skip/=no_skip.sum()
    no_neural=np.array(prior);no_neural[4]=0;no_neural/=no_neural.sum()
    return {"v15_static":old,"six_static":[.45,.12,.05,.13,.15,.10],"six_em":prior,
            "six_causal":adaptive,"no_skip":no_skip.tolist(),"no_neural":no_neural.tolist()}

def evaluate(ref,cal,docs,limit):
    ms=C.defaultdict(C.Counter);oracle=C.Counter();start=time.monotonic()
    for doc in docs:
        state=H.Replay();logs=[0.]*6
        for i,w in enumerate(doc[:limit+16]):
            branches,key,base=ref.six(state)
            if i>=16:
                # Independent candidate oracle, never used in the predictor.
                union=set()
                for j,branch in enumerate(branches):
                    rank,cert,n=H.Mixture((branch,),(1.,)).rank(budget=800)
                    oracle[NAMES[j]+"_top5"]+=w in rank;oracle[NAMES[j]+"_cert"]+=cert
                    union.update(rank)
                oracle["n"]+=1;oracle["union5"]+=w in union;oracle["union_size"]+=len(union)
                for mode,weights in modes(branches,cal,key,logs).items():
                    l=H.Mixture(branches,weights);rank,cert,n=l.rank(budget=800)
                    m=ms[mode];m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert;m["expanded"]+=n
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.;a1=a5=True
                    for c in w+B.EOS:
                        d=l.next(p);prod*=d[c];order=sorted(d,key=lambda x:(-d[x],x))
                        a1=a1 and c==order[0];a5=a5 and c in order[:5]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                    m["spell1"]+=a1;m["spell5"]+=a5
            probs=[l.complete(w) for l in branches]
            logs=[.94*v+.35*math.log(max(p,1e-300)) for v,p in zip(logs,probs)]
            mx=max(logs);logs=[max(-15.,v-mx) for v in logs];state.observe(w)
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"oracle":dict(oracle),"seconds":time.monotonic()-start}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];calbooks=books[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(calbooks)==12 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=Unified(docs,lex,N.Native(a.lib));before=ref.params.payload()
    pilot=ref.head.train(ref.head.data([d[:512] for d in docs]))
    cal=fit(ref,[B.load(p,200) for _,p in calbooks]);frozen=cal.export()
    print("CALIBRATION",json.dumps(frozen),flush=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(data,k):
        s=H.Replay()
        for w in data[:k]:s.observe(w)
        branches,key,_=ref.six(s);l=H.Mixture(branches,cal.weights(key))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0],16)==at(story[0][:16]+["other","future"],16)
    out={"scope":"bounded v17 neural-head replay; signed weights frozen","pilot":pilot,"reference_ids":[i for i,p in tr],"calibration_ids":[i for i,p in calbooks],"skip_contexts":len(ref.skip.rows),"calibration":frozen,"suffix_invariance":True,"fresh":{}}
    out["stories"]=evaluate(ref,cal,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
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
        assert len(chosen)==6;ids[k]=[i for i,w in chosen];out["fresh"][k]=evaluate(ref,cal,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert before==ref.params.payload() and frozen==cal.export()
    out["signed_unchanged"]=True;out["fresh_ids"]=ids;out["graph"]=dict(ref.diag)
    out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    h=ref.head
    np.savez_compressed(a.out.with_name("disposable-head.npz"),words=np.array(h.words),step=np.array(h.step),**h.p,**{"m_"+k:v for k,v in h.m.items()},**{"v_"+k:v for k,v in h.v.items()})
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
