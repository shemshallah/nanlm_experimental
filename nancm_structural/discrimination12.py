"""v12 reference-only reliability calibration and bounded contrastive competition.
No gradient training. Calibration is count/statistic fitting on disjoint reference-side books."""
import base as B, foundation4 as F, foundation6 as N, foundation8 as E, continuation11 as P
import argparse, collections as C, heapq, json, math, pathlib, time

NAMES=("kn","aligned","signed","contrast","contrast_signed")
def mix(ref,laws,weights):
    s={};bg=0.
    for l,q in zip(laws,weights):
        if q:B.add(s,l.s,q);bg+=q*l.bg
    return E.FastLattice(ref.background_kn,s,bg)

class Discrimination(P.Continuation):
    def contrast(self,l,h,q,trust):
        if not l.s or not q:return l
        # Compare aligned hypotheses against the same continuation background.
        # Preserve the total conditional mass in the competing subspace.
        top=set(heapq.nlargest(24,l.s,key=lambda w:(l.complete(w),w)))
        top.update(w for w in heapq.nlargest(24,q,key=lambda w:(q[w],w)) if w in l.s)
        top=sorted(top)
        budget=math.fsum(l.s[w] for w in top)
        if budget<=0:return l
        base=self.background_kn
        scores={}
        for w in top:
            prior=base.background(w)
            # Residual evidence is bounded and discounted when alignment is weak.
            # Smoothing prevents absence from the retrieved contexts being a veto.
            ratio=math.log((q.get(w,0.)+.50*prior+1e-15)/(1.50*prior+1e-15))
            residual=max(-1.5,min(1.5,ratio))
            scores[w]=l.s[w]*math.exp(.45*trust*residual)
        r=dict(l.s)
        for w,p in B.norm(scores).items():r[w]=budget*p
        return E.FastLattice(base,r,l.bg)
    def laws(self,h,ep):
        kn,a,g,lam=self.hypotheses(h,ep,h,1.)
        q,trust=self.alignment(tuple(h[-8:]))
        laws=(kn,a,g,self.contrast(a,h,q,trust),self.contrast(g,h,q,trust))
        count=sum(self.tables[2].get(tuple(h[-2:]),{}).values()) if len(h)>=2 else 0
        # Context bucket is defined before the target is seen.
        countbin=0 if count==0 else 1 if count<4 else 2 if count<16 else 3
        strength=0 if not q else 1 if trust<.40 else 2
        return laws,(countbin,strength),lam

class Ledger:
    def __init__(self):self.rows=C.defaultdict(lambda:{"n":0,"loss":[0.]*len(NAMES),"hit":[0]*len(NAMES)})
    def observe(self,key,laws,w):
        for bucket in (("all",),key):
            row=self.rows[bucket];row["n"]+=1
            for j,l in enumerate(laws):
                row["loss"][j]-=math.log(max(l.complete(w),1e-300))
                rank,cert,_=l.rank(1);assert cert
                row["hit"][j]+=rank[0]==w
    def weights(self,key,rankaware):
        root=self.rows[("all",)];row=self.rows.get(key)
        n=root["n"];assert n
        scores=[]
        for j in range(len(NAMES)):
            loss=root["loss"][j]/n;hit=root["hit"][j]/n
            if row:
                # Empirical-Bayes shrinkage to the global calibration average.
                loss=(row["loss"][j]+32*loss)/(row["n"]+32)
                hit=(row["hit"][j]+32*hit)/(row["n"]+32)
            scores.append(-4*loss+(8*hit if rankaware else 0))
        mx=max(scores);p=B.norm({j:math.exp(v-mx) for j,v in enumerate(scores)})
        return [.85*p[j]+.15/len(scores) for j in range(len(scores))]
    def export(self):return {str(k):v for k,v in sorted(self.rows.items(),key=lambda p:str(p[0]))}
def observe(ep,h,w):
    for n in range(1,min(8,len(h))+1):ep[tuple(h[-n:])][w]+=1
def calibrate(ref,docs):
    ledger=Ledger();start=time.monotonic()
    for doc in docs:
        h=[];ep=C.defaultdict(C.Counter)
        for i,w in enumerate(doc[:48]):
            if i>=16:
                laws,key,lam=ref.laws(h,ep);ledger.observe(key,laws,w)
            observe(ep,h,w);h=(h+[w])[-64:]
    return ledger,time.monotonic()-start
def evaluate(ref,ledger,docs,limit):
    modes=(*NAMES,"uniform","calibrated_nll","calibrated_rank","causal_rank")
    ms={k:C.Counter() for k in modes};diag=C.Counter();start=time.monotonic()
    for doc in docs:
        h=[];ep=C.defaultdict(C.Counter);logs=[0.]*len(NAMES)
        for i,w in enumerate(doc[:limit+16]):
            laws,key,lam=ref.laws(h,ep)
            nll=ledger.weights(key,False);rank=ledger.weights(key,True)
            z=B.norm({j:rank[j]*math.exp(logs[j]) for j in range(len(NAMES))})
            causal=[.90*z[j]+.10/len(NAMES) for j in range(len(NAMES))]
            if i>=16:
                outputs=(*laws,mix(ref,laws,[1/len(NAMES)]*len(NAMES)),mix(ref,laws,nll),mix(ref,laws,rank),mix(ref,laws,causal))
                diag["n"]+=1;diag["lambda"]+=lam
                for mode,l in zip(modes,outputs):
                    m=ms[mode];top,cert,n=l.rank();m["n"]+=1;m["top1"]+=top[0]==w;m["top5"]+=w in top;m["certified"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    if diag["n"]<=24:
                        p="";b=1.;prod=1.
                        for c in w+B.EOS:
                            d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1));prod*=d[c]
                            if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                        assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            # Online evidence applies only to subsequent predictions.
            logs=[.94*v+.4*math.log(max(l.complete(w),1e-300)) for v,l in zip(logs,laws)]
            mx=max(logs);logs=[max(-15.,v-mx) for v in logs]
            observe(ep,h,w);h=(h+[w])[-64:]
    return {"modes":{k:dict(m,word_top1=m["top1"]/m["n"],word_top5=m["top5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"diag":dict(diag),"seconds":time.monotonic()-start}
def check(ref,ledger,doc):
    saved=json.dumps(ledger.export(),sort_keys=True)
    def at(data,n):
        h=[];ep=C.defaultdict(C.Counter)
        for w in data[:n]:observe(ep,h,w);h=(h+[w])[-64:]
        laws,key,lam=ref.laws(h,ep);l=mix(ref,laws,ledger.weights(key,True))
        return l.s,l.bg,l.rank()
    assert at(doc,16)==at(doc[:16]+["different","unseen","suffix"],16)
    assert saved==json.dumps(ledger.export(),sort_keys=True)
    return {"suffix_invariance":True,"calibration_frozen":True}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7]
    tr=books[:96];cal=books[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(cal)==12 and len(lex)>1000
    assert not {i for i,p in tr}&{i for i,p in cal}
    docs=[B.load(p,3000) for _,p in tr];ref=Discrimination(docs,lex,N.Native(a.lib))
    ledger,seconds=calibrate(ref,[B.load(p,200) for _,p in cal])
    print("CALIBRATION",json.dumps({"seconds":seconds,"ledger":ledger.export()}),flush=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    out={"gradient_updates":0,"calibration_kind":"reference-side held-out count/statistic fitting","reference_words":sum(map(len,docs)),"reference_ids":[i for i,_ in tr],"calibration_ids":[i for i,_ in cal],"calibration_seconds":seconds,"calibration":ledger.export(),"checks":check(ref,ledger,story[0]),"fresh":{}}
    out["stories"]=evaluate(ref,ledger,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
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
        assert len(chosen)==6;ids[k]=[i for i,w in chosen];out["fresh"][k]=evaluate(ref,ledger,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert out["calibration"]==ledger.export()
    out["fresh_ids"]=ids;out["graph"]=dict(ref.diag);out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
