"""v7: competing causal context laws with prequential reliability, no gradient training."""
import base as B, foundation4 as F, foundation5 as V, foundation6 as N
import argparse, collections as C, functools, heapq, json, math, pathlib, time

def normalized(c):return B.norm(c)
def update(s,bg,p,lam):
    return F.blend_sparse(s,1-lam,p,lam),bg*(1-lam)

def shape(w):
    # Context classes are observed spelling features, not target labels or external tags.
    if w in ("the","a","an","this","that","these","those"):return "DET"
    if w in ("i","he","she","it","we","they","you","me","him","her","them","us"):return "PRON"
    if w in ("of","in","to","from","with","on","at","by","for","as","into","over","under"):return "PREP"
    if w in ("is","are","was","were","be","been","am","have","has","had","do","does","did"):return "AUX"
    if w in ("and","or","but","if","so","than","though"):return "CONJ"
    if len(w)>4:
        for suffix in ("tion","ment","ness","able","ing","est","ous","ive","ed","ly","er","s"):
            if w.endswith(suffix):return "S:"+suffix
    return "W"

class Competition(N.NativeFoundation):
    names=("local","phrase","syntax","topic","episode","analog","signed")
    lattice=F.FullLattice
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel)
        self.long=C.defaultdict(C.Counter);self.pattern=C.defaultdict(C.Counter)
        self.cont=C.defaultdict(set)
        for doc in docs:
            for i,w in enumerate(doc):
                if i:self.cont[w].add(doc[i-1])
                for n in (5,6,7,8):
                    if i>=n:self.long[tuple(doc[i-n:i])][w]+=1
                if i>=2:
                    self.pattern[(shape(doc[i-2]),doc[i-1])][w]+=1
                    self.pattern[(doc[i-2],shape(doc[i-1]))][w]+=1
                    self.pattern[(shape(doc[i-2]),shape(doc[i-1]))][w]+=1
                if i>=3:self.pattern[(shape(doc[i-3]),shape(doc[i-2]),doc[i-1])][w]+=1
        c=normalized({w:len(v) for w,v in self.cont.items()})
        self.continuation=self.lattice(self,{w:.75*q for w,q in c.items()},.25)
    def law(self,parts,base=None):
        s,bg=({},1.) if base is None else (dict(base.s),base.bg)
        for c,lam in parts:
            if c and lam>0:s,bg=update(s,bg,normalized(c),lam)
        return self.lattice(self,s,bg)
    def discount(self,c):
        if not c:return {},0.
        n=sum(c.values());d=.6
        p={w:max(0.,v-d) for w,v in c.items()}
        return p,sum(p.values())/n
    def experts(self,h,ep,recency):
        # Seven alternative normalized laws. None reads the target.
        loc=[]
        for n in (1,2):
            if len(h)>=n:loc.append(self.discount(self.idx.get(("local"+str(n),tuple(h[-n:])))))
        local=self.law(loc,self.continuation)
        ph=[]
        for n in range(3,9):
            if len(h)>=n:
                c=self.idx.get(("local"+str(n),tuple(h[-n:]))) if n<=4 else self.long.get(tuple(h[-n:]))
                ph.append(self.discount(c))
        phrase=self.law(ph,local)
        syn=[]
        if len(h)>=2:
            for k in ((shape(h[-2]),shape(h[-1])),(h[-2],shape(h[-1])),(shape(h[-2]),h[-1])):
                c=self.pattern.get(k)
                if c:syn.append((c,min(.48,sum(c.values())/(sum(c.values())+16))))
        if len(h)>=3:
            c=self.pattern.get((shape(h[-3]),shape(h[-2]),h[-1]))
            if c:syn.append((c,min(.65,sum(c.values())/(sum(c.values())+12))))
        syntax=self.law(syn,local)
        ds={}
        for w in sorted(set(h[-32:-4])):
            c=self.idx.get(("discourse",(w,)))
            if c:B.add(ds,normalized(c),1/math.sqrt(1+self.uni[w]))
        topic=self.law([(ds,.35)],local)
        epis=[]
        if recency:
            # Bounded recency cache boosts document entities, not unseen target words.
            c=C.Counter()
            for j,w in enumerate(reversed(recency[-128:])):c[w]+=math.exp(-j/48)
            epis.append((c,.20))
        for n in range(1,9):
            if len(h)>=n:
                c=ep.get(tuple(h[-n:]))
                if c:
                    count=sum(c.values());epis.append((c,min(.85,count/(count+(3 if n==1 else 1)))))
        episode=self.law(epis,phrase)
        a=self.analog(h[-1]) if h else {}
        analog=self.law([(a,.35 if h and self.uni[h[-1]]<8 else .16)],local)
        # Single coherent conditional pool for signed support/refutation.
        # Competing experts supply distinct context evidence; the graph
        # can move mass only within this pool, preserving the background.
        sources=[]
        for name,l in zip(self.names,(local,phrase,syntax,topic,episode,analog)):
            if l.s:sources.append((("expert",name,tuple(h[-4:])),normalized(l.s),1/6))
        s={};bg=0.
        for l in (local,phrase,syntax,topic,episode,analog):
            B.add(s,l.s,1/6);bg+=l.bg/6
        s,bg=self.field(h,sources,s,bg,"signed") if sources else (s,bg)
        signed=self.lattice(self,s,bg)
        return (local,phrase,syntax,topic,episode,analog,signed)

class Session:
    def __init__(self,ref,mode="adaptive"):
        self.ref=ref;self.mode=mode;self.h=[];self.recent=[];self.ep=C.defaultdict(C.Counter)
        self.logs=[0.]*len(ref.names);self.cache=None;self.prob=None;self.weights=None
    def predict(self):
        if self.prob is not None:return self.prob
        self.cache=self.ref.experts(self.h,self.ep,self.recent)
        if self.mode in self.ref.names:
            self.weights=[float(n==self.mode) for n in self.ref.names]
        elif self.mode=="static":self.weights=[1/len(self.cache)]*len(self.cache)
        elif self.mode=="adaptive_nosign":
            z=B.norm({i:math.exp(max(-30.,v)) for i,v in enumerate(self.logs[:-1])})
            self.weights=[.90*z[i]+.10/len(z) for i in range(len(z))]+[0.]
        else:
            z=B.norm({i:math.exp(max(-30.,v)) for i,v in enumerate(self.logs)})
            self.weights=[.90*z[i]+.10/len(z) for i in range(len(z))]
        s={};bg=0.
        for l,q in zip(self.cache,self.weights):
            B.add(s,l.s,q);bg+=q*l.bg
        self.prob=self.ref.lattice(self.ref,s,bg)
        return self.prob
    def observe(self,w):
        self.predict()
        # All scores are evaluated before updating; the state is document-local.
        # Forgetting prevents an early mismatch from permanently disabling a route.
        logs=[.94*v+.45*math.log(max(l.complete(w),1e-300)) for v,l in zip(self.logs,self.cache)]
        mx=max(logs);self.logs=[max(-30.,v-mx) for v in logs]
        for n in range(1,9):
            if len(self.h)>=n:self.ep[tuple(self.h[-n:])][w]+=1
        self.h=(self.h+[w])[-64:];self.recent=(self.recent+[w])[-128:]
        self.cache=None;self.prob=None

def evaluate(ref,docs,mode,limit=96):
    if mode in ("retrieval","exact","analog_parent","signed_parent"):
        return F.evaluate(ref,docs,{"analog_parent":"analog","signed_parent":"signed"}.get(mode,mode),limit)
    m=C.Counter();start=time.monotonic();weights=C.Counter()
    for doc in docs:
        session=Session(ref,mode)
        for i,w in enumerate(doc[:limit+16]):
            l=session.predict()
            if i>=16:
                top,cert,n=l.rank();m["n"]+=1;m["top1"]+=top[0]==w;m["top5"]+=w in top;m["certified"]+=cert
                m["shortlist128"]+=w in heapq.nlargest(128,l.s,key=lambda v:(l.complete(v),v))
                m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                p="";b=1.;prod=1.
                for c in w+B.EOS:
                    d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1));prod*=d[c]
                    if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                for name,q in zip(ref.names,session.weights):weights[name]+=q
            session.observe(w)
    n=m["n"];return dict(m,word_top1=m["top1"]/n,word_top5=m["top5"]/n,spelling_bps=m["bits"]/m["symbols"],mean_weights={k:v/n for k,v in weights.items()},seconds=time.monotonic()-start)

class AdaptiveBytes:
    def __init__(self,ref,raw,mode="adaptive"):
        self.session=Session(ref,mode);self.raw=raw;self.tail=b"";self.prefix="";self.bp=1.;self.lat=None
        self.prior=.5;self.posterior=.5;self.cached=None
    def probs(self):
        if self.cached is not None:return self.cached[0]
        base=self.raw.probs(self.tail)
        if self.lat is None:self.lat=self.session.predict()
        d=self.lat.next(self.prefix,self.bp);non=math.fsum(base[i] for i in B.NONLETTERS)
        if not d:word=base
        else:
            word=[0.]*256;end=d[B.EOS] if self.prefix else non
            for i in B.NONLETTERS:word[i]=end*base[i]/non
            for c in B.A:
                lo,hi=ord(c),ord(c.upper());z=base[lo]+base[hi]
                m=d[c] if self.prefix else (1-end)*d[c]
                word[lo]=m*base[lo]/z;word[hi]=m*base[hi]/z
        p=[(1-self.posterior)*a+self.posterior*b for a,b in zip(base,word)]
        self.cached=(p,word);return p
    def observe(self,b):
        p=self.probs();self.posterior=self.posterior*self.cached[1][b]/p[b]
        if b in B.LETTERS:
            c=chr(b).lower();self.bp*=self.session.ref.charprob(self.prefix)[c];self.prefix+=c
        else:
            if self.prefix:self.session.observe(self.prefix);self.prefix="";self.bp=1.;self.lat=None
            self.posterior=self.prior
        self.tail=(self.tail+bytes([b]))[-3:];self.cached=None

def raw_eval(ref,raw,docs,mode):
    if mode=="v6":return V.byte_eval(ref,raw,docs,"bayes")
    n=one=five=0;bits=err=0.;start=time.monotonic()
    for doc in docs:
        s=AdaptiveBytes(ref,raw,mode)
        for i,b in enumerate(doc[:768]):
            p=s.probs();err=max(err,abs(math.fsum(p)-1))
            assert len(p)==256 and min(p)>=0
            if i>=64:
                order=sorted(range(256),key=lambda j:(-p[j],j))[:5]
                n+=1;one+=b==order[0];five+=b in order;bits-=math.log2(max(p[b],1e-300))
            s.observe(b)
    return {"bytes":n,"raw_bpb":bits/n,"byte_top1":one/n,"byte_top5":five/n,"mass_error":err,"seconds":time.monotonic()-start}

def checks(ref,raw,doc):
    h=B.words(doc.decode("utf-8","replace"))[:24]
    def at(words,k):
        s=Session(ref)
        for w in words[:k]:s.observe(w)
        l=s.predict();return l.bg,l.s,l.rank(),s.weights
    assert at(h,16)==at(h[:16]+["arbitrary","future","suffix"],16)
    s=Session(ref);p=s.predict();assert abs(p.mass("",1)-1)<1e-9
    for w in h:s.observe(w)
    fresh=Session(ref);assert fresh.predict().s==p.s and fresh.logs==[0.]*len(ref.names)
    def bat(d,k):
        s=AdaptiveBytes(ref,raw)
        for b in d[:k]:s.observe(b)
        return s.probs()
    for k in (8,32,64,128):assert bat(doc,k)==bat(doc[:k]+b"different future",k)
    return {"word_future_invariance":True,"byte_future_invariance":True,"episode_reset":True,"normalized_expert_mixture":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")}
    split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:24]
    lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=Competition(docs,lex,N.Native(a.lib));raw=B.RawByte([p.read_bytes()[:36000] for _,p in tr])
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"long_context_keys":len(ref.long),"syntax_keys":len(ref.pattern),"matched":{},"fresh_dev":{}}
    modes=("retrieval","exact","static","adaptive","adaptive_nosign")
    def run(k,ds,dest,limit):
        dest[k]={mode:evaluate(ref,ds,mode,limit) for mode in modes}
        print(k,json.dumps(dest[k]),flush=True)
    for k,files in groups.items():
        picked=[(i,p) for i,p in files if split(i) in (7,8)][:5]
        run(k,[B.load(p,300) for _,p in picked],out["matched"],72)
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
        assert len(chosen)==6;fresh[k]=[i for i,w in chosen]
        run(k,[w for i,w in chosen],out["fresh_dev"],96)
    out["fresh_ids"]=fresh
    rawtests=[p.read_bytes()[:768] for i,p in groups["stories"] if split(i) in (7,8)][:5]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:rawtests.append(f.read(768))
    out["checks"]=checks(ref,raw,rawtests[0])
    out["raw"]={mode:raw_eval(ref,raw,rawtests,mode) for mode in ("v6","static","adaptive","adaptive_nosign")}
    out["graph"]=dict(ref.diag);out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
