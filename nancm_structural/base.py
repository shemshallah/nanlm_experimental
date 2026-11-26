"""NaNCM structural foundation, no gradient training. Python reference."""
import argparse, collections as C, hashlib, heapq, json, math, pathlib, re, time, array, struct, sys
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
class SignedParameters:
    """Persistent S+/S- gains: 32640 x 8 per channel, exact identity at zero."""
    N=32640*8
    def __init__(self):
        self.plus=array.array("f",[0.])*self.N
        self.minus=array.array("f",[0.])*self.N
    def gain(self,slot,d,a,b,signs=(True,True)):
        k=slot*8+d
        pa=self.plus[k]; pb=self.minus[k]
        if not math.isfinite(pa) or not math.isfinite(pb):raise ValueError("nonfinite field parameter")
        return math.tanh((a*math.exp(max(-4.,min(4.,pa))) if signs[0] else 0.)-(b*math.exp(max(-4.,min(4.,pb))) if signs[1] else 0.))
    def payload(self):
        values=[]
        for x in (self.plus,self.minus):
            y=array.array("f",x)
            if sys.byteorder!="little":y.byteswap()
            values.append(y.tobytes())
        return b"NCMSIGN1"+struct.pack("<III",256,8,self.N)+b"".join(values)
    def save(self,path):
        payload=self.payload();path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()
    @classmethod
    def load(cls,path):
        data=path.read_bytes()
        if len(data)!=20+8*cls.N or data[:20]!=b"NCMSIGN1"+struct.pack("<III",256,8,cls.N):raise ValueError("bad field checkpoint")
        result=cls()
        for off,channel in ((20,"plus"),(20+4*cls.N,"minus")):
            x=array.array("f");x.frombytes(data[off:off+4*cls.N])
            if sys.byteorder!="little":x.byteswap()
            if not all(map(math.isfinite,x)):raise ValueError("nonfinite field checkpoint")
            setattr(result,channel,x)
        return result
def stable_field(edges,anchor,params,signs=(True,True),tol=1e-5,max_passes=12):
    # Symmetric graph with row-absolute-sum bound >= spectral norm.
    # Projected update x <- P(.65 anchor + .35 W x) has contraction <= .35
    # for a fixed graph. Outer retrieval expansion is separately capped.
    row=C.Counter(); active=[]
    for (slot,i,j,d),(a,b) in sorted(edges.items()):
        v=params.gain(slot,d,a,b,signs)
        if not v:continue
        ki,kj=(i,d),(j,d);row[ki]+=abs(v);row[kj]+=abs(v);active.append((ki,kj,v))
    scale=max(1.,max(row.values(),default=0.))
    x=anchor.copy()
    for _ in range(max_passes):
        msg={}
        for ki,kj,v in active:
            msg[ki]=msg.get(ki,0.)+v*x.get(kj,0.)
            msg[kj]=msg.get(kj,0.)+v*x.get(ki,0.)
        q=bounded(add({k:.65*v for k,v in anchor.items()},msg,.35/scale))
        delta=math.sqrt(sum((q.get(k,0.)-x.get(k,0.))**2 for k in q.keys()|x.keys()))
        if not math.isfinite(delta):return anchor.copy(),False
        x=q
        if delta<=tol:return x,True
    # deterministic rollback if fixed-graph convergence not certified
    return anchor.copy(),False
class Ref:
    def __init__(self,docs,lex):
        self.params=SignedParameters();self.graph_stats=C.Counter();self.lex=set(lex); self.uni=C.Counter(); self.idx=C.defaultdict(C.Counter); self.char=C.defaultdict(C.Counter)
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
        self.total_words=sum(self.uni.values())
        self.unknown=max(.02,min(.35,sum(n==1 for n in self.uni.values())/max(1,self.total_words)))
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
        for _,c,_ in src: pool.update(w for w,_ in c.most_common(12 if mode=="retrieval" else 32))
        # lexical entries admit alternatives but are never interpreted as text sequences
        for w in h[-4:]:
            for v in (w,w+"s",w+"ed",w+"ing",w[:-1]):
                if v in self.lex: pool.add(v)
        def prior(pool,weights):
            p={w:.2*self.globalp.get(w,0.)+1e-6 for w in sorted(pool)}
            for (_,c,_),wt in zip(src,weights):
                z=sum(c.values())
                for w in sorted(pool):
                    p[w]+=wt*c.get(w,0)/z
            return norm(p)
        def hierarchical(pool,weights):
            # The untruncated distribution is normalized by construction.
            # Compute its mass on the admitted set; the remainder is ESC.
            keys_sorted=sorted(pool)
            unknown=self.unknown
            q={w:(1-unknown)*self.globalp.get(w,0.)+unknown*self.baseword(w) for w in keys_sorted}
            totals=[sum(c.values()) for _,c,_ in src]
            def mix(indices,cap):
                if not indices:return
                reliabilities=[]
                for j in indices:
                    _,c,orig=src[j]; n=sum(c.values())
                    reliabilities.append(min(.95,n/(n+len(c)+4)*weights[j]/max(orig,1e-12)))
                z=sum(reliabilities)
                if z<=0:return
                lam=min(cap,z/len(indices))
                for w in keys_sorted:
                    conditional=sum(r*src[j][1].get(w,0)/totals[j] for j,r in zip(indices,reliabilities))/z
                    q[w]=(1-lam)*q[w]+lam*conditional
            for stage,cap in (("discourse",.18),("local1",.90),("skip",.45),("local2",.95),("local3",.95),("local4",.95),("episode",.80)):
                mix([j for j,(key,_,_) in enumerate(src) if key[0]==stage],cap)
            mass=math.fsum(q.values())
            assert -1e-9<=mass<=1+1e-9
            return norm(q),max(.001,min(.999,1-mass))
        weights=[wt for _,_,wt in src]
        if mode=="retrieval":p=prior(pool,weights); escape=.35 if not src else .20
        else:p,escape=hierarchical(pool,weights)
        if mode in ("unified","plus_only","minus_only") and src:
            # Bounded feedback: lower context evidence -> signed upper lexical
            # clouds -> source trust -> candidate expansion -> unified commitment.
            # All cells have stable C(256,2) addresses; only active dimensions
            # are materialized. Edges are ephemeral evidence, not trained weights.
            # Explicit sparse 8D pair cells with S+/S- evidence.
            # Lower/lower agreement, upper/upper lexical coactivation, and
            # bidirectional lower/upper conditional messages share the field.
            def edge(store,i,j,d,v):
                if i==j:return
                i,j=min(i,j),max(i,j)
                k=(pair(i,j),i,j,d)
                a,b=store.get(k,(0.,0.))
                store[k]=(a+max(v,0.),b+max(-v,0.))
            for iteration in range(2):
                edges={}; low={}; up={}; source_vec=[]
                for (k,c,_),wt in zip(src,weights):
                    l=lower(k); z=sum(c.values()); low[l]=low.get(l,0.)+wt
                    vec={}
                    proposals=sorted({w for w,_ in c.most_common(16)}|{w for w,_ in self.top[:8]})
                    for w in proposals:
                        delta=c.get(w,0)/z-self.globalp.get(w,1e-8)
                        ev=math.tanh(4*delta)*math.sqrt(wt)
                        v=lexvec(w); add(vec,v,delta)
                        for u,q in sorted(v.items()):
                            edge(edges,l[0],u[0],l[1],ev*q)
                            ku=(u[0],l[1]); up[ku]=up.get(ku,0.)+delta*q*wt
                    source_vec.append(bounded(vec))
                for i,(k,_,_) in enumerate(src):
                    li=lower(k)
                    for j in range(i+1,len(src)):
                        lj=lower(src[j][0])
                        if li[1]==lj[1]:edge(edges,li[0],lj[0],li[1],dot(source_vec[i],source_vec[j]))
                for w in sorted(pool,key=lambda w:(-p.get(w,0.),w))[:20]:
                    bydim=C.defaultdict(list)
                    for (i,d),v in sorted(lexvec(w).items()):bydim[d].append((i,v))
                    for d,items in bydim.items():
                        for j,(i,x) in enumerate(items):
                            for u,y in items[j+1:]:edge(edges,i,u,d,math.sqrt(p.get(w,0.))*x*y)
                anchor=bounded(add(bounded(low),bounded(up)))
                signs=(mode!="minus_only",mode!="plus_only")
                x,converged=stable_field(edges,anchor,self.params,signs)
                self.graph_stats["attempts"]+=1
                self.graph_stats["converged"]+=converged
                if not converged:
                    self.graph_stats["rollbacks"]+=1
                    break
                def score(w):
                    total=0.
                    for (i,d),q in lexvec(w).items():
                        total+=q*(x.get((i,d),0.)+.125*sum(x.get((i,t),0.) for t in range(8)))
                    return max(-1.,min(1.,total))
                revised=[]
                for (k,c,_),wt in zip(src,weights):
                    z=sum(c.values()); feedback=sum(n/z*score(w) for w,n in c.most_common(16))
                    feedback+=.25*x.get(lower(k),0.)
                    revised.append(wt*math.exp(.4*max(-1.,min(1.,feedback))))
                for j in sorted(range(len(src)),key=lambda j:(-revised[j],j))[:4]:
                    pool.update(w for w,_ in src[j][1].most_common(48))
                q,escape=hierarchical(pool,revised)
                q=norm({w:v*math.exp(.7*score(w)) for w,v in q.items()})
                p=norm({w:.5*p.get(w,0.)+.5*q[w] for w in sorted(pool)})
                weights=revised
        # ESC is fixed for the historical baseline and residual mass for the hierarchy.
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
    def boundary(doc,cut):
        local=C.defaultdict(C.Counter); prefix=[]
        for w in doc[:cut]:
            observe(local,prefix,w);prefix=(prefix+[w])[-64:]
        return ref.predict(prefix,local)
    a=boundary(h+["alpha","beta"],len(h))
    for suffix in (["other"],["unseenlexicalstring"]*5,[]):
        b=boundary(h+suffix,len(h))
        assert a.p==b.p and a.e==b.e and a.rank()==b.rank()
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
    return {"pair_addresses":32640,"mass_conservation":True,"word_prefix_equivalence":True,"repeat_determinism":True,"future_suffix_invariance":True}
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
class RawByte:
    """All 256 bytes, causal case/punctuation projection of word lattice."""
    def __init__(self,docs):
        self.counts=C.defaultdict(C.Counter)
        for doc in docs:
            h=b""
            for b in doc:
                for n in range(min(3,len(h))+1):
                    self.counts[h[-n:] if n else b""][b]+=1
                h=(h+bytes([b]))[-3:]
    @lru_cache(maxsize=65536)
    def probs(self,h):
        c=self.counts[b""]; z=sum(c.values())+25.6
        p=[(c.get(i,0)+.1)/z for i in range(256)]
        for n in range(1,min(3,len(h))+1):
            c=self.counts.get(h[-n:])
            if not c:continue
            z=sum(c.values()); lam=z/(z+len(c)+4)
            p=[(1-lam)*q+lam*c.get(i,0)/z for i,q in enumerate(p)]
        return p
LETTERS=set(range(65,91))|set(range(97,123))
NONLETTERS=sorted(set(range(256))-LETTERS)
class ByteSession:
    def __init__(self,ref,raw,mode):
        self.ref=ref;self.raw=raw;self.mode=mode
        self.h=[];self.ep=C.defaultdict(C.Counter);self.tail=b""
        self.prefix="";self.bp=1.;self.lattice=None
    def probs(self):
        base=self.raw.probs(self.tail)
        if self.mode=="byte_only":return base
        if self.lattice is None:self.lattice=self.ref.predict(self.h,self.ep,self.mode)
        d=self.lattice.next(self.prefix,self.bp)
        if not d:return base
        p=[0.]*256
        non=math.fsum(base[i] for i in NONLETTERS)
        end=d[EOS] if self.prefix else non
        for i in NONLETTERS:p[i]=end*base[i]/non
        for c in A:
            lo,hi=ord(c),ord(c.upper()); z=base[lo]+base[hi]
            mass=d[c] if self.prefix else (1-end)*d[c]
            p[lo]=mass*base[lo]/z;p[hi]=mass*base[hi]/z
        return p
    def observe(self,b):
        if b in LETTERS:
            c=chr(b).lower()
            self.bp*=self.ref.charprob(self.prefix)[c];self.prefix+=c
        elif self.prefix:
            observe(self.ep,self.h,self.prefix);self.h=(self.h+[self.prefix])[-64:]
            self.prefix="";self.bp=1.;self.lattice=None
        self.tail=(self.tail+bytes([b]))[-3:]
def byteeval(ref,raw,docs,mode):
    n=one=five=0;bits=err=0.;start=time.monotonic()
    for doc in docs:
        s=ByteSession(ref,raw,mode)
        for i,b in enumerate(doc[:768]):
            p=s.probs();err=max(err,abs(math.fsum(p)-1))
            assert len(p)==256 and min(p)>=0 and all(math.isfinite(x) for x in p)
            if i>=64:
                order=sorted(range(256),key=lambda j:(-p[j],j))[:5]
                n+=1;one+=b==order[0];five+=b in order
                bits-=math.log2(max(p[b],1e-300))
            s.observe(b)
    return {"bytes":n,"raw_bpb":bits/n,"byte_top1":one/n,"byte_top5":five/n,"max_mass_error":err,"seconds":time.monotonic()-start}
def rawcheck(ref,raw,doc):
    # Full 256-way future suffix invariance over real bytes.
    def at(data,k):
        s=ByteSession(ref,raw,"unified")
        for b in data[:k]:s.probs();s.observe(b)
        return s.probs()
    for k in (1,8,31,64):
        assert at(doc,k)==at(doc[:k]+b"completely different suffix",k)
    return True
def fieldcheck(ref,path):
    import tempfile
    # Known active two-node graph; sign isolation and finite perturbation
    # of each persistent signed channel. No optimizer or updates to final state.
    i,j,d=0,128,3; slot=pair(i,j); index=slot*8+d
    E={(slot,i,j,d):(1.,.4)}
    anchor={(i,d):1.}
    a,ok=stable_field(E,anchor,ref.params);assert ok
    plus,okp=stable_field(E,anchor,ref.params,(True,False))
    minus,okm=stable_field(E,anchor,ref.params,(False,True))
    assert okp and okm and plus!=minus and a!=plus and a!=minus
    diffs=[]
    for channel in (ref.params.plus,ref.params.minus):
        channel[index]=.01
        b,ok=stable_field(E,anchor,ref.params)
        channel[index]=0.
        assert ok and a!=b
        diffs.append((b[(j,d)]-a[(j,d)])/.01)
    sha=ref.params.save(path)
    loaded=SignedParameters.load(path)
    assert loaded.payload()==ref.params.payload()
    assert stable_field(E,anchor,loaded)==stable_field(E,anchor,ref.params)
    return {"signed_parameter_count":2*SignedParameters.N,"zero_checkpoint_sha256":sha,"checkpoint_roundtrip":True,"plus_minus_separate":True,"finite_difference_sensitivities":diffs,"fixed_graph_convergence":True}
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
    rawdocs=[p.read_bytes()[:36000] for _,p in tr]; raw=RawByte(rawdocs)
    res={"training_updates":0,"reference_index_words":sum(map(len,docs)),"reference_book_ids":[i for i,_ in tr],"lexicon_words":len(lex),"checks":check(ref),"evaluations":{},"scope":"lowercase ASCII words with EOS; spelling BPS is not raw-byte BPB"}
    # do not tune on these diagnostics; untouched split 9 not used
    for k,files in groups.items():
        picked=[(i,p) for i,p in files if split(i) in (7,8)][:5]
        if len(picked)<3:raise ValueError("insufficient DEV "+k)
        samples=[load(p,300) for _,p in picked]
        res["evaluations"][k]={"book_ids":[i for i,_ in picked],**{mode:evaldocs(ref,samples,mode) for mode in ("retrieval","hierarchical","unified")}}
        print(k,json.dumps(res["evaluations"][k]),flush=True)
    for k,p in (("wikitext",r/"wiki"/"wiki.valid.raw"),("chemistry",r/"domain"/"chemistry.txt")):
        # contiguous bounded diagnostic windows; no reference ingestion
        text=p.read_text(encoding="utf-8",errors="replace")
        samples=[words(text[i:i+8000])[:400] for i in range(0,min(len(text),40000),8000)]
        res["evaluations"][k]={mode:evaldocs(ref,samples,mode) for mode in ("retrieval","hierarchical","unified")}
        print(k,json.dumps(res["evaluations"][k]),flush=True)
    # Raw byte paths preserve casing, punctuation, whitespace and UTF-8 bytes.
    rawtests=[p.read_bytes()[:768] for i,p in groups["stories"] if split(i) in (7,8)][:5]
    with (r/"wiki"/"wiki.valid.raw").open("rb") as f:rawtests.append(f.read(768))
    with (r/"domain"/"chemistry.txt").open("rb") as f:rawtests.append(f.read(768))
    res["raw_reference_bytes"]=sum(map(len,rawdocs))
    res["checks"]["persistent_signed_field"]=fieldcheck(ref,a.out.with_name("signed-init.bin"))
    res["checks"]["raw_future_invariance"]=rawcheck(ref,raw,rawtests[0])
    res["raw_byte_probe"]={mode:byteeval(ref,raw,rawtests,mode) for mode in ("byte_only","retrieval","hierarchical","unified")}
    print("RAW",json.dumps(res["raw_byte_probe"]),flush=True)
    # Paired directional ablation on real short-story prefixes; not promotion.
    ab=[load(p,150) for i,p in groups["stories"] if split(i) in (7,8)][:3]
    res["signed_ablation"]={mode:evaldocs(ref,ab,mode,24) for mode in ("hierarchical","unified","plus_only","minus_only")}
    res["graph_convergence"]=dict(ref.graph_stats)
    print("ABLATION",json.dumps(res["signed_ablation"]),flush=True)
    a.out.write_text(json.dumps(res,indent=2));print("FINAL",json.dumps(res),flush=True)
if __name__=="__main__":main()
