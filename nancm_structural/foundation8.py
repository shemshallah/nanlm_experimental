"""v8: exact finite-support ranking + lazy prefix materialization; paired shared evaluation."""
import base as B, foundation4 as F, foundation5 as V, foundation6 as N, foundation7 as S
import argparse, collections as C, heapq, json, math, pathlib, time

class FastLattice(F.FullLattice):
    def __init__(self,ref,s,bg):
        self.ref=ref;self.s={w:v for w,v in s.items() if v>0};self.bg=bg
        assert abs(math.fsum(self.s.values())+bg-1)<1e-8
        self.p=self.s;self.e=bg*ref.char_weight;self._pref=None
    @property
    def pref(self):
        if self._pref is None:
            self._pref=C.Counter()
            for w,v in self.s.items():
                for i in range(len(w)+1):self._pref[w[:i]]+=v
        return self._pref
    def rank(self,k=5,budget=2500):
        if k<1:raise ValueError("k must be positive")
        # Every explicit conditional word is evaluated. Among known background
        # words outside that set, only the top k background words can enter the
        # global top k. Unknown spellings have no discrete background mass.
        known=self.ref.bg_rank
        seeds=set(self.s)|set(known[:k])
        best=heapq.nlargest(k,((self.complete(w),w) for w in seeds))
        heap=[(-self.e,"",1.)] if self.e>0 else [];n=0
        while heap and n<budget:
            if len(best)>=k and -heap[0][0]<best[-1][0]:break
            _,p,b=heapq.heappop(heap);n+=1
            for c,q in self.ref.charprob(p).items():
                if c==B.EOS:
                    if p not in self.s and p not in self.ref.bgword:
                        best=heapq.nlargest(k,best+[(self.e*b*q,p)])
                else:
                    t=p+c;nb=b*q;ub=self.e*nb
                    if ub>0 and (len(best)<k or ub>=best[-1][0]):heapq.heappush(heap,(-ub,t,nb))
        cert=not heap or (len(best)>=k and -heap[0][0]<best[-1][0])
        return [w for _,w in best],cert,n

class FastCompetition(S.Competition):
    lattice=FastLattice
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel)
        # Tie ordering must match the exact word ranking.
        self.bg_rank=sorted(self.bgword,key=lambda w:(self.background(w),w),reverse=True)
    def predict(self,h,episode=None,mode="signed"):
        l=super().predict(h,episode,mode)
        if isinstance(l,F.FullLattice):return FastLattice(self,l.s,l.bg)
        return l

def compare(ref,docs):
    n=0;maxerr=0.;oldnodes=newnodes=0;oldsecs=newsecs=0.;cert=0
    for doc in docs:
        h=[];ep=C.defaultdict(C.Counter)
        for i,w in enumerate(doc[:40]):
            if i>=16:
                l=ref.predict(h,ep,"signed");a=F.FullLattice(ref,l.s,l.bg)
                for k in (1,5,16):
                    t=time.monotonic();x,xc,xn=a.rank(k);oldsecs+=time.monotonic()-t
                    t=time.monotonic();y,yc,yn=l.rank(k);newsecs+=time.monotonic()-t
                    assert xc and yc and x==y,(x,y,xc,yc)
                    oldnodes+=xn;newnodes+=yn;cert+=yc
                p="";b=1.
                for c in w+B.EOS:
                    d=a.next(p,b);e=l.next(p,b)
                    err=max(abs(d[k]-e[k]) for k in d);maxerr=max(maxerr,err);assert err<1e-12
                    if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                n+=1
            B.observe(ep,h,w);h=(h+[w])[-64:]
    return {"boundaries":n,"rank_checks":cert,"probability_max_error":maxerr,"old_search_nodes":oldnodes,"new_search_nodes":newnodes,"old_rank_seconds":oldsecs,"new_rank_seconds":newsecs}

def weigh(logs,mode):
    n=len(logs)
    if mode=="static":return [1/n]*n
    count=n-1 if mode=="adaptive_nosign" else n
    z=B.norm({i:math.exp(max(-30.,v)) for i,v in enumerate(logs[:count])})
    return [.9*z[i]+.1/count for i in range(count)]+[0.]*(n-count)
def shared(ref,docs,limit=72):
    modes=("static","adaptive","adaptive_nosign")
    metrics={mode:C.Counter() for mode in modes};weights={mode:C.Counter() for mode in modes};start=time.monotonic()
    for doc in docs:
        session=S.Session(ref)
        for i,w in enumerate(doc[:limit+16]):
            session.predict()
            if i>=16:
                for mode in modes:
                    ws=weigh(session.logs,mode)
                    s={};bg=0.
                    for l,q in zip(session.cache,ws):
                        if q:
                            B.add(s,l.s,q);bg+=q*l.bg
                    l=FastLattice(ref,s,bg);top,cert,n=l.rank();m=metrics[mode]
                    m["n"]+=1;m["top1"]+=top[0]==w;m["top5"]+=w in top;m["certified"]+=cert
                    m["shortlist128"]+=w in heapq.nlargest(128,l.s,key=lambda v:(l.complete(v),v))
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";b=1.;prod=1.
                    for c in w+B.EOS:
                        d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1));prod*=d[c]
                        if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                    for name,q in zip(ref.names,ws):weights[mode][name]+=q
            session.observe(w)
    out={}
    for mode,m in metrics.items():
        n=m["n"];out[mode]=dict(m,word_top1=m["top1"]/n,word_top5=m["top5"]/n,spelling_bps=m["bits"]/m["symbols"],mean_weights={k:v/n for k,v in weights[mode].items()})
    return out,time.monotonic()-start

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:24];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=FastCompetition(docs,lex,N.Native(a.lib));raw=B.RawByte([p.read_bytes()[:36000] for _,p in tr])
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5]
    dev=[B.load(p,300) for _,p in picked]
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"parity":compare(ref,dev),"matched":{},"fresh_dev":{},"shared_seconds":{}}
    print("PARITY",json.dumps(out["parity"]),flush=True)
    def run(k,ds,dest,limit):
        v,sec=shared(ref,ds,limit);dest[k]={"retrieval":F.evaluate(ref,ds,"retrieval",limit),"exact":F.evaluate(ref,ds,"exact",limit),**v};out["shared_seconds"][k]=sec
        print(k,json.dumps(dest[k]),flush=True)
    for k,files in groups.items():
        selected=[(i,p) for i,p in files if split(i) in (7,8)][:5];run(k,[B.load(p,300) for _,p in selected],out["matched"],72)
    for k,p in (("wikitext",r/"wiki"/"wiki.valid.raw"),("chemistry",r/"domain"/"chemistry.txt")):
        with p.open(encoding="utf-8",errors="replace") as f:text=f.read(40000)
        run(k,[B.words(text[i:i+8000])[:400] for i in range(0,len(text),8000)],out["matched"],72)
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]};fresh={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                f.seek(min(p.stat().st_size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6;fresh[k]=[i for i,w in chosen];run("fresh_"+k,[w for i,w in chosen],out["fresh_dev"],96)
    out["fresh_ids"]=fresh
    rawtests=[p.read_bytes()[:768] for _,p in picked]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:rawtests.append(f.read(768))
    out["checks"]=S.checks(ref,raw,rawtests[0])
    out["raw"]={mode:S.raw_eval(ref,raw,rawtests,mode) for mode in ("v6","adaptive","adaptive_nosign")}
    out["graph"]=dict(ref.diag);out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
