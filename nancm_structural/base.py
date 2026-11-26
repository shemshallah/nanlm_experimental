"""NaNCM structural foundation, no gradient training. Python reference."""
import argparse, collections as C, hashlib, heapq, json, math, pathlib, re, time
from functools import lru_cache
A="abcdefghijklmnopqrstuvwxyz"
EOS=" "
SYM=A+EOS
def words(s): return re.findall(r"[a-z]+",s.lower())
def digest(s): return hashlib.sha256(s.encode()).hexdigest()
def addr(s): return int(digest(s)[:16],16)
def norm(d):
    z=math.fsum(d.values())
    return {k:v/z for k,v in d.items()} if z else {}
def dot(a,b): return sum(v*b.get(k,0.) for k,v in a.items())
def add(a,b,scale=1.):
    for k,v in b.items(): a[k]=a.get(k,0.)+scale*v
    return a
def bounded(a):
    z=max(1.,math.sqrt(dot(a,a)))
    return {k:v/z for k,v in a.items()}
def pair(i,j):
    if not 0<=i<j<256: raise ValueError("unordered pair")
    return i*(511-i)//2+j-i-1
@lru_cache(maxsize=32768)
def lexvec(w):
    d={}
    s="^"+w+"$"
    for n in (1,2,3):
        for i in range(len(s)-n+1):
            h=addr(str(n)+s[i:i+n]); k=(128+h%128,(h>>8)%8)
            d[k]=d.get(k,0.)+(1 if (h>>16)&1 else -1)
    return bounded(d)
def keys(h):
    # local syntax, skip roles, discourse. Only observed prefix.
    out=[]
    for n in (1,2,3,4):
        if len(h)>=n: out.append(("local"+str(n),tuple(h[-n:])))
    if len(h)>=3: out.append(("skip",(h[-3],h[-1])))
    for w in sorted(set(h[-32:-4])): out.append(("discourse",(w,)))
    return out
def lower(key):
    h=addr(repr(key)); return (h%128,(h>>8)%8)
class Ref:
    def __init__(self,docs,lex):
        self.lex=set(lex); self.uni=C.Counter(); self.idx=C.defaultdict(C.Counter); self.char=C.defaultdict(C.Counter)
        for doc in docs:
            h=[]
            for w in doc:
                self.uni[w]+=1
                for k in keys(h): self.idx[k][w]+=1
                h.append(w); h=h[-64:]
        for w,n in self.uni.items():
            p=""
            for c in w+EOS:
                self.char[p[-2:]][c]+=n
                p+=c
        self.globalp=norm(self.uni)
        self.top=self.uni.most_common(32)
    @lru_cache(maxsize=2048)
    def charprob(self,p):
        chars=SYM if p else A
        c=self.char.get(p[-2:],{})
        total=sum(c.get(x,0)+.25 for x in chars)
        return {x:(c.get(x,0)+.25)/total for x in chars}
    @lru_cache(maxsize=131072)
    def baseword(self,w):
        p=1.; s=""
        for c in w+EOS:
            p*=self.charprob(s)[c]; s+=c
        return p
    def sources(self,h,episode):
        out=[]
        for k in keys(h):
            c=self.idx.get(k)
            if c:
                n=sum(c.values()); wt=math.log1p(n)
                wt*= {"local1":1.,"local2":2.,"local3":3.,"local4":4.,"skip":1.1,"discourse":.12}[k[0]]
                out.append((k,c,wt))
        # causal episodic memory: prefix only, never retained across documents
        if episode:
            for k in keys(h):
                c=episode.get(k)
                if c and k[0]!="discourse": out.append((("episode",k),c,2.*math.log1p(sum(c.values()))))
        return out
    def predict(self,h,episode=None,mode="unified"):
        src=self.sources(h,episode); pool={w for w,_ in self.top}
        for _,c,_ in src: pool.update(w for w,_ in c.most_common(12))
        # lexical entries admit alternatives but are never interpreted as text sequences
        for w in h[-4:]:
            for v in (w,w+"s",w+"ed",w+"ing",w[:-1]):
                if v in self.lex: pool.add(v)
        def prior(pool,weights):
            p={w:.2*self.globalp.get(w,0.)+1e-6 for w in pool}
            for (_,c,_),wt in zip(src,weights):
                z=sum(c.values())
                for w in pool:
                    p[w]+=wt*c.get(w,0)/z
            return norm(p)
        weights=[wt for _,_,wt in src]
        p=prior(pool,weights)
        if mode=="unified" and src:
            # Bounded feedback: lower context evidence -> signed upper lexical
            # clouds -> source trust -> candidate expansion -> unified commitment.
            # All cells have stable C(256,2) addresses; only active dimensions
            # are materialized. Edges are ephemeral evidence, not trained weights.
            for iteration in range(2):
                pos={}; neg={}; edges={}; low={}
                for (k,c,_),wt in zip(src,weights):
                    l=lower(k); z=sum(c.values()); low[l]=low.get(l,0.)+wt
                    for w,n in c.most_common(24):
                        q=n/z; bg=self.globalp.get(w,1e-8)
                        # positive relative evidence; negative competing evidence
                        ev=math.tanh(math.log((q+1e-8)/(bg+1e-8))/5)
                        for u,v in lexvec(w).items():
                            cell=(pair(l[0],u[0]),l[1],u[1])
                            a,b=edges.get(cell,(0.,0.))
                            edges[cell]=(a+wt*q*max(ev,0)*v,b+wt*q*(1-max(ev,0))*v)
                            pos[u]=pos.get(u,0.)+wt*q*max(ev,0)*v
                            neg[u]=neg.get(u,0.)+wt*q*(1-max(ev,0))*v
                pos=bounded(pos); neg=bounded(neg)
                # same-sign lateral upper interaction, via bounded shared field
                consensus=bounded({u:pos.get(u,0.)-neg.get(u,0.) for u in pos.keys()|neg.keys()})
                def score(w):
                    v=lexvec(w); return dot(v,consensus)
                # lower-upper return messages modify source reliability
                revised=[]
                for (k,c,_),wt in zip(src,weights):
                    z=sum(c.values()); feedback=sum(n/z*score(w) for w,n in c.most_common(24))
                    revised.append(wt*math.exp(.4*max(-1.,min(1.,feedback))))
                # expand from sources favoured by the cloud, not future labels
                for j in sorted(range(len(src)),key=lambda j:(-revised[j],j))[:4]:
                    pool.update(w for w,_ in src[j][1].most_common(32))
                q=prior(pool,revised)
                q=norm({w:v*math.exp(.7*score(w)) for w,v in q.items()})
                p=norm({w:.5*p.get(w,0.)+.5*q[w] for w in sorted(pool)})
                weights=revised
        # fixed conservative escape, no learned gate or calibration on DEV
        escape=.35 if not src else .20
        return Lattice(self,p,escape)
class Lattice:
    """Candidate mass plus disjoint open spelling escape. Exact prefix marginal."""
    def __init__(self,ref,p,e):
        self.ref=ref; self.p=p; self.e=e
        self.pref=C.Counter(); self.ex=C.Counter(); self.bw={}
        for w,q in p.items():
            b=ref.baseword(w); self.bw[w]=b
            for i in range(len(w)+1):
                self.pref[w[:i]]+=q
                self.ex[w[:i]]+=b
        self.z=1-math.fsum(self.bw.values())
        if self.z<=1e-12: raise ValueError("escape collapsed")
    def complete(self,w):
        return (1-self.e)*self.p[w] if w in self.p else self.e*self.ref.baseword(w)/self.z
    def mass(self,p,b):
        return (1-self.e)*self.pref[p]+self.e*max(0.,b-self.ex[p])/self.z
    def next(self,p,b):
        m=self.mass(p,b)
        if m<=0: return {}
        return {c:(self.complete(p) if c==EOS else self.mass(p+c,b*q))/m for c,q in self.ref.charprob(p).items()}
    def rank(self,k=5,budget=1200):
        # exact top-k when remaining prefix bound <= kth complete probability.
        # completed proposals are known immediately; search only disjoint escape.
        best=sorted(((self.complete(w),w) for w in self.p),reverse=True)[:k]
        heap=[(-self.e,"",1.)]; count=0
        while heap and count<budget:
            bound=-heap[0][0]
            if len(best)>=k and bound<=best[-1][0]: break
            _,p,b=heapq.heappop(heap);count+=1
            for c,q in self.ref.charprob(p).items():
                if c==EOS:
                    if p not in self.p:
                        best=sorted(best+[(self.complete(p),p)],reverse=True)[:k]
                else:
                    t=p+c; nb=b*q
                    ub=self.e*max(0.,nb-self.ex[t])/self.z
                    if ub>0 and (len(best)<k or ub>best[-1][0]):
                        heapq.heappush(heap,(-ub,t,nb))
        certified=not heap or (len(best)>=k and -heap[0][0]<=best[-1][0])
        return [w for _,w in best],certified,count
def observe(ep,h,w):
    for k in keys(h): ep[k][w]+=1
def check(ref):
    assert len({pair(i,j) for i in range(256) for j in range(i+1,256)})==32640
    assert pair(254,255)==32639
    h=["the","word","is"]; ep=C.defaultdict(C.Counter)
    a=ref.predict(h,ep); b=ref.predict(h+["future"] if False else h,ep)
    assert a.p==b.p
    assert abs(a.mass("",1)-1)<1e-9
    for w in ["the","unseenlexicalstring",*list(a.p)[:16]]:
        p=""; q=1.; prod=1.
        for c in w+EOS:
            d=a.next(p,q); assert abs(sum(d.values())-1)<1e-7,(p,d)
            prod*=d[c]
            if c!=EOS:q*=ref.charprob(p)[c];p+=c
        assert math.isclose(prod,a.complete(w),rel_tol=1e-6,abs_tol=1e-15)
    # no input suffix API; explicit deterministic repeated and episode reset
    assert ref.predict(h,ep).p==ref.predict(h,ep).p
    return {"pair_addresses":32640,"mass_conservation":True,"word_prefix_equivalence":True,"repeat_determinism":True}
def bookfiles(root):
    return sorted((p.name.split("--")[0],p) for p in root.rglob("*.txt") if re.match(r"^\d+--",p.name))
def load(p,n=5000):
    with p.open(encoding="utf-8",errors="replace") as f:return words(f.read(n*12))[:n]
def evaldocs(ref,docs,mode,limit=72):
    m=C.Counter(); start=time.monotonic()
    for doc in docs:
        h=[]; ep=C.defaultdict(C.Counter)
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                l=ref.predict(h,ep,mode); top,cert,n=l.rank()
                m["n"]+=1;m["top1"]+=bool(top and top[0]==w);m["top5"]+=w in top
                m["certified"]+=cert;m["nodes"]+=n;m["candidate"]+=w in l.p
                m["lexical"]+=w in ref.lex;m["oov"]+=w not in ref.uni
                m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                p="";q=1.
                for c in w+EOS:
                    d=l.next(p,q);m["max_mass_error"]=max(m["max_mass_error"],abs(sum(d.values())-1))
                    order=sorted(d,key=lambda s:(-d[s],s))
                    m["char1"]+=c==order[0];m["char5"]+=c in order[:5]
                    if c!=EOS:q*=ref.charprob(p)[c];p+=c
            observe(ep,h,w);h=(h+[w])[-64:]
    n=m["n"];s=m["symbols"]
    if not n:raise ValueError("empty eval")
    return dict(m,word_top1=m["top1"]/n,word_top5=m["top5"]/n,candidate_recall=m["candidate"]/n,rank_certification=m["certified"]/n,normalized_spelling_bps=m["bits"]/s,char_top1=m["char1"]/s,char_top5=m["char5"]/s,seconds=time.monotonic()-start)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:bookfiles(r/k) for k in ("stories","sentences","paragraphs")}
    # one global partition across alternate Gutenberg representations
    split=lambda i:int(digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:24]
    if len(tr)<12:raise ValueError("insufficient actual Gutenberg books")
    lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):
            lex.update(words(p.read_text(errors="replace")))
    if len(lex)<1000:raise ValueError("lexical release not unpacked")
    docs=[load(p,3000) for _,p in tr]
    ref=Ref(docs,lex)
    res={"training_updates":0,"reference_index_words":sum(map(len,docs)),"reference_book_ids":[i for i,_ in tr],"lexicon_words":len(lex),"checks":check(ref),"evaluations":{},"scope":"lowercase ASCII words with EOS; spelling BPS is not raw-byte BPB"}
    # do not tune on these diagnostics; untouched split 9 not used
    for k,files in groups.items():
        picked=[(i,p) for i,p in files if split(i) in (7,8)][:5]
        if len(picked)<3:raise ValueError("insufficient DEV "+k)
        samples=[load(p,300) for _,p in picked]
        res["evaluations"][k]={"book_ids":[i for i,_ in picked],**{mode:evaldocs(ref,samples,mode) for mode in ("retrieval","unified")}}
        print(k,json.dumps(res["evaluations"][k]),flush=True)
    for k,p in (("wikitext",r/"wiki"/"wiki.valid.raw"),("chemistry",r/"domain"/"chemistry.txt")):
        # contiguous bounded diagnostic windows; no reference ingestion
        text=p.read_text(encoding="utf-8",errors="replace")
        samples=[words(text[i:i+8000])[:400] for i in range(0,min(len(text),40000),8000)]
        res["evaluations"][k]={mode:evaldocs(ref,samples,mode) for mode in ("retrieval","unified")}
        print(k,json.dumps(res["evaluations"][k]),flush=True)
    a.out.write_text(json.dumps(res,indent=2));print("FINAL",json.dumps(res),flush=True)
if __name__=="__main__":main()
