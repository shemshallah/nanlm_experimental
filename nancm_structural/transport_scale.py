"""Frozen reference-scale diagnostic for v9 alignment, without graph confounds."""
import base as B, foundation4 as F, foundation8 as E, foundation9 as T
import argparse, collections as C, gc, heapq, json, math, pathlib, time

class Alignment(F.Foundation):
    affinity=T.Transport.affinity
    neighbors=T.Transport.neighbors
    alignment=T.Transport.alignment
    def __init__(self,docs,lex):
        super().__init__(docs,lex)
        self.bg_rank=sorted(self.bgword,key=lambda w:(self.background(w),w),reverse=True)
        self.rows=[];self.postings=C.defaultdict(list);self.transport_stats=C.Counter()
        for j,doc in enumerate(docs):
            for i,w in enumerate(doc):
                if not i:continue
                h=tuple(reversed(doc[max(0,i-8):i]));r=len(self.rows)
                self.rows.append((j,h,w))
                for lag,v in enumerate(h):self.postings[lag,v].append(r)
    def laws(self,h,ep):
        exact=self.predict(h,ep,"exact")
        baseline=E.FastLattice(self,exact.s,exact.bg)
        q,t=self.alignment(tuple(h[-8:]))
        s=F.blend_sparse(baseline.s,1-t,q,t)
        return baseline,E.FastLattice(self,s,baseline.bg*(1-t)),q,t

def score(ref,docs,limit):
    m={mode:C.Counter() for mode in ("exact","aligned")};stats=C.Counter();start=time.monotonic()
    for doc in docs:
        h=[];ep=C.defaultdict(C.Counter)
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                a,b,q,t=ref.laws(h,ep)
                stats["n"]+=1;stats["trust"]+=t
                stats["alignment_available"]+=bool(q)
                stats["alignment_top5"]+=w in heapq.nlargest(5,q,key=lambda v:(q[v],v))
                stats["alignment_support"]+=w in q
                for mode,l in (("exact",a),("aligned",b)):
                    d=m[mode];top,cert,n=l.rank()
                    d["n"]+=1;d["top1"]+=w==top[0];d["top5"]+=w in top;d["certified"]+=cert
                    d["bits"]-=math.log2(max(l.complete(w),1e-300));d["symbols"]+=len(w)+1
                    d["shortlist128"]+=w in heapq.nlargest(128,l.s,key=lambda v:(l.complete(v),v))
                if stats["n"]<=12:
                    assert abs(b.mass("",1)-1)<1e-9
                    p="";v=1.;prod=1.
                    for c in w+B.EOS:
                        d=b.next(p,v);assert abs(math.fsum(d.values())-1)<1e-8;prod*=d[c]
                        if c!=B.EOS:v*=ref.charprob(p)[c];p+=c
                    assert math.isclose(prod,b.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            B.observe(ep,h,w);h=(h+[w])[-64:]
    return {"modes":{mode:dict(d,word_top1=d["top1"]/d["n"],word_top5=d["top5"]/d["n"],spelling_bps=d["bits"]/d["symbols"]) for mode,d in m.items()},"alignment":dict(stats),"seconds":time.monotonic()-start}
def counterfactual(ref,docs):
    m=C.Counter();n=0
    for i,doc in enumerate(docs):
        other=docs[(i+1)%len(docs)]
        for j in range(16,28):
            w=doc[j]
            for label,h in (("correct",doc[:j]),("unrelated",other[:j])):
                a,b,q,t=ref.laws(h,None)
                for name,l in (("exact",a),("aligned",b)):
                    m[name+"_"+label]-=math.log2(max(l.complete(w),1e-300))
            n+=1
    return {k:v/n for k,v in m.items()}
def clear():
    for method in (B.Ref.charprob,B.Ref.baseword,F.Foundation.background,F.Foundation.semantic,F.Foundation.analog,Alignment.affinity,Alignment.neighbors,Alignment.alignment):method.cache_clear()
    gc.collect()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:96];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(lex)>1000
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]};fresh={};ids={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;ids[k]=[i for i,w in chosen];fresh[k]=[w for i,w in chosen]
    out={"training_updates":0,"reference_scale":{},"fresh_ids":ids,"story_ids":[i for i,_ in picked],"scope":"isolated v9 alignment operator blended into exact full-support law; no signed graph in this diagnostic"}
    for count in (24,96):
        start=time.monotonic()
        docs=[B.load(p,3000) for _,p in tr[:count]];ref=Alignment(docs,lex)
        x={"reference_words":sum(map(len,docs)),"reference_ids":[i for i,_ in tr[:count]],"context_rows":len(ref.rows),"build_seconds":time.monotonic()-start}
        x["stories"]=score(ref,story,72)
        print(count,"stories",json.dumps(x["stories"]),flush=True)
        x["fresh"]={}
        for k,ds in fresh.items():
            x["fresh"][k]=score(ref,ds,32);print(count,k,json.dumps(x["fresh"][k]),flush=True)
        x["counterfactual"]=counterfactual(ref,story)
        x["transport_stats"]=dict(ref.transport_stats);out["reference_scale"][str(count)]=x
        del ref,docs;clear()
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
