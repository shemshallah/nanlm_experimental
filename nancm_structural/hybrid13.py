"""v13 hybrid: signed continuation + mirrored causal spelling + finite PoE readout.
Mechanisms inspired by NaNCM v130 and palindromic substrate; no imported weights.
All reference statistics from real NaNROUTER words. No gradient training."""
import base as B, foundation4 as F, foundation6 as N, foundation8 as E, continuation11 as P
import argparse, collections as C, functools, heapq, json, math, pathlib, time, numpy as np

class MirrorSpelling:
    """256 palindromic head identities; 32640 unordered pairs x 8 rails.
    Lower heads encode causal multi-scale features, upper heads encode symbols.
    Directed lower->upper evidence is positive; reverse edge is its negative.
    Separate support and opposition are derived from conditional vs global counts.
    """
    def __init__(self,docs):
        self.head_ids=[(i<<8)|int(f"{i:08b}"[::-1],2) for i in range(256)]
        assert len(set(self.head_ids))==256
        self.count=np.zeros((8,128,27),dtype=np.float64)
        self.char=C.defaultdict(C.Counter);self.globalcount=np.ones(27)
        for doc in docs:
            prev=""
            for w in doc:
                p=""
                for c in w+B.EOS:
                    j=B.SYM.index(c);self.globalcount[j]+=1
                    for rail,key in enumerate(self.keys(prev,p)):
                        self.count[rail,key,j]+=1
                    for k in self.contexts(prev,p):self.char[k][c]+=1
                    p+=c
                prev=w
        self.globalp=self.globalcount/self.globalcount.sum()
        totals=self.count.sum(axis=2,keepdims=True)
        self.posterior=(self.count+16*self.globalp)/(totals+16)
        delta=np.log(self.posterior/self.globalp)
        # Antisymmetric pair field with separately inspectable support/opposition.
        self.plus=np.zeros((32640,8),dtype=np.float32)
        self.minus=np.zeros_like(self.plus)
        for rail in range(8):
            for i in range(128):
                for j in range(27):
                    slot=B.pair(i,128+j);v=delta[rail,i,j]
                    self.plus[slot,rail]=max(v,0)
                    self.minus[slot,rail]=max(-v,0)
        self.energy=self.plus-self.minus
        self.forward=np.empty((8,128,27))
        for rail in range(8):
            for i in range(128):
                for j in range(27):
                    self.forward[rail,i,j]=self.energy[B.pair(i,128+j),rail]
    @staticmethod
    @functools.lru_cache(maxsize=65536)
    def keys(prev,p):
        return tuple(B.addr(repr((i,k)))%128 for i,k in enumerate((p[-1:],p[-2:],p[-3:],p[:2],(min(len(p),8),p[-1:]),(prev[-2:],p[-1:]),(prev[-3:],p[:1]),(prev,p[-2:]))))
    @staticmethod
    def contexts(prev,p):
        return (("u",p[-1:]),("bi",p[-2:]),("tri",p[-3:]),("cross",prev[-2:],p[-1:]),("word",prev,p[-2:]))
    @functools.lru_cache(maxsize=65536)
    def probs(self,prev,p,mirror):
        n=26 if not p else 27
        d=self.globalp[:n].copy();d/=d.sum()
        for k in self.contexts(prev,p):
            row=self.char.get(k)
            if not row:continue
            counts=np.array([row.get(c,0) for c in B.SYM[:n]],dtype=float);total=counts.sum()
            if total:
                lam=total/(total+12+np.count_nonzero(counts))
                d=(1-lam)*d+lam*counts/total
        if mirror:
            evidence=np.mean([self.forward[r,k,:n] for r,k in enumerate(self.keys(prev,p))],axis=0)
            d*=np.exp(.45*np.clip(evidence,-2.,2.));d/=d.sum()
        return d
    @functools.lru_cache(maxsize=65536)
    def logword(self,prev,w,mirror):
        s=0.;p=""
        for c in w+B.EOS:
            s+=math.log(max(1e-300,float(self.probs(prev,p,mirror)[B.SYM.index(c)])))
            p+=c
        return s
    def checks(self):
        for i,j,rail in ((0,128,0),(55,133,4),(127,154,7)):
            v=float(self.energy[B.pair(i,j),rail])
            def read(a,b):return float(self.energy[B.pair(min(a,b),max(a,b)),rail])*(1 if a<b else -1)
            assert read(i,j)==v and read(i,j)==-read(j,i)
        assert all(int(f"{x:016b}"[::-1],2)==x for x in self.head_ids)
        return {"palindromic_heads":256,"pair_addresses":32640,"active_pair_rails":int(np.count_nonzero(self.energy)),"antisymmetry":True}

class Tilt:
    """Exact finite modification of a normalized open word law.
    The entire probability mass of selected completed words is conserved.
    All unselected words, including unknown spellings, retain exact probability."""
    def __init__(self,base,weights):
        self.base=base;self.ref=base.ref;self.s=base.s;self.bg=base.bg
        budget=math.fsum(base.complete(w) for w in weights)
        scores=B.norm({w:base.complete(w)*math.exp(max(-3.,min(3.,v))) for w,v in weights.items()})
        self.delta={w:budget*scores[w]-base.complete(w) for w in scores}
        self.dp=C.Counter()
        for w,v in self.delta.items():
            for i in range(len(w)+1):self.dp[w[:i]]+=v
        assert abs(math.fsum(self.delta.values()))<1e-9
    def complete(self,w):return self.base.complete(w)+self.delta.get(w,0.)
    def mass(self,p,b):return self.base.mass(p,b)+self.dp[p]
    def next(self,p,b):
        m=self.mass(p,b);assert m>0
        return {c:(self.complete(p) if c==B.EOS else self.mass(p+c,b*q))/m for c,q in self.ref.charprob(p).items()}
    def rank(self,k=5,budget=2500):
        seeds=set(self.s)|set(self.delta)|set(self.ref.bg_rank[:k+len(self.delta)])
        best=heapq.nlargest(k,((self.complete(w),w) for w in seeds))
        heap=[(-self.base.e,"",1.)];n=0
        while heap and n<budget:
            if len(best)>=k and -heap[0][0]<best[-1][0]:break
            _,p,b=heapq.heappop(heap);n+=1
            for c,q in self.ref.charprob(p).items():
                if c==B.EOS:
                    if p not in seeds and p not in self.ref.bgword:
                        best=heapq.nlargest(k,best+[(self.complete(p),p)])
                else:
                    t=p+c;nb=b*q;ub=self.base.e*nb
                    if ub>0 and (len(best)<k or ub>=best[-1][0]):heapq.heappush(heap,(-ub,t,nb))
        cert=not heap or (len(best)>=k and -heap[0][0]<best[-1][0])
        return [w for _,w in best],cert,n

class Hybrid(P.Continuation):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel);self.spell=MirrorSpelling(docs)
    def correction(self,base,h,recent,mirror,weight):
        prev=h[-1] if h else ""
        # Candidates are proposed using only the past. Candidate spellings
        # are evaluated hypothetically, never using the actual future word.
        top=set(heapq.nlargest(32,base.s,key=lambda w:(base.complete(w),w)))
        top.update(base.ref.bg_rank[:16]);top.update(recent[-16:])
        if not top:return base
        cache=C.Counter(recent[-128:]);weights={}
        for w in sorted(top):
            if not w:continue
            raw=self.spell.logword(prev,w,mirror)
            generic=math.log(max(self.baseword(w),1e-300))
            evidence=max(-2.,min(2.,(raw-generic)/(1+len(w)/4)))
            recurrence=math.log1p(cache[w])/(1+math.log1p(self.uni[w]))
            weights[w]=weight*evidence+.18*recurrence
        return Tilt(base,weights)
    def laws(self,h,ep,recent):
        kn,a,s,lam=self.hypotheses(h,ep,recent,1.)
        return (kn,s,self.correction(s,h,recent,False,.55),self.correction(s,h,recent,True,.55))

MODES=("continuation","signed","spelling_poe","mirror_poe")
def evaluate(ref,docs,limit):
    ms={k:C.Counter() for k in MODES};start=time.monotonic()
    for doc in docs:
        h=[];recent=[];ep=C.defaultdict(C.Counter)
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                for mode,l in zip(MODES,ref.laws(h,ep,recent)):
                    m=ms[mode];top,cert,n=l.rank();m["n"]+=1;m["t1"]+=w==top[0];m["t5"]+=w in top;m["cert"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";b=1.;prod=1.;all1=all5=True
                    for c in w+B.EOS:
                        d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1));prod*=d[c]
                        rank=sorted(d,key=lambda x:(-d[x],x))
                        all1=all1 and c==rank[0];all5=all5 and c in rank[:5]
                        if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                    m["spelled1"]+=all1;m["spelled5"]+=all5
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            for n in range(1,min(8,len(h))+1):ep[tuple(h[-n:])][w]+=1
            h=(h+[w])[-64:];recent=(recent+[w])[-128:]
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],teacher_forced_spelled_top1=m["spelled1"]/m["n"],teacher_forced_spelled_top5=m["spelled5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"seconds":time.monotonic()-start}

def causal_check(ref,doc):
    def at(data,k):
        h=[];ep=C.defaultdict(C.Counter)
        for w in data[:k]:
            for n in range(1,min(8,len(h))+1):ep[tuple(h[-n:])][w]+=1
            h.append(w)
        laws=ref.laws(h,ep,h)
        return [(l.rank(),[l.complete(w) for w in ("the","a","unseenword","garden")]) for l in laws]
    assert at(doc,16)==at(doc[:16]+["other","future"],16)
    return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:96];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=Hybrid(docs,lex,N.Native(a.lib))
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    out={"gradient_updates":0,"reference_words":sum(map(len,docs)),"reference_ids":[i for i,_ in tr],"mirror":ref.spell.checks(),"causal":causal_check(ref,story[0]),"fresh":{}}
    out["stories"]=evaluate(ref,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
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
        assert len(chosen)==6;ids[k]=[i for i,w in chosen];out["fresh"][k]=evaluate(ref,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    out["fresh_ids"]=ids;out["graph"]=dict(ref.diag);out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    np.savez_compressed(a.out.with_name("mirror-counts.npz"),plus=ref.spell.plus,minus=ref.spell.minus)
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
