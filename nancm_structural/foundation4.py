"""v4: full-support word law + analogical signed concept field. No gradient updates."""
import base as B
import argparse, collections as C, functools, heapq, json, math, pathlib, time
from functools import lru_cache

def blend_sparse(s,scale,source,weight):
    return B.add({w:p*scale for w,p in s.items()},source,weight)

class FullLattice:
    """A normalized mixture: fixed full-support background + sparse conditional mass.
    All known words retain probability even when absent from the explicit shortlist.
    Character escape is a latent generative route, not an extra unnormalized score."""
    def __init__(self,ref,s,bg):
        self.ref=ref; self.s={w:v for w,v in s.items() if v>0};self.bg=bg
        assert abs(math.fsum(self.s.values())+bg-1)<1e-8
        self.pref=C.Counter()
        for w,v in self.s.items():
            for i in range(len(w)+1):self.pref[w[:i]]+=v
        self.p=self.s # explicit conditional support; not the full distribution
        self.e=bg*ref.char_weight
    def complete(self,w):return self.s.get(w,0.)+self.bg*self.ref.background(w)
    def mass(self,p,b):
        return self.pref[p]+self.bg*(self.ref.bgprefix[p]+self.ref.char_weight*b)
    def next(self,p,b):
        m=self.mass(p,b)
        if m<=0:return {}
        return {c:(self.complete(p) if c==B.EOS else self.mass(p+c,b*q))/m for c,q in self.ref.charprob(p).items()}
    def rank(self,k=5,budget=2500):
        seed=set(self.ref.bg_top[:max(k,32)])|set(heapq.nlargest(max(k,32),self.s,key=lambda w:(self.s[w],w)))
        best=sorted(((self.complete(w),w) for w in seed),reverse=True)[:k]
        visited=set(seed); heap=[(-1.,"",1.)]; n=0
        while heap and n<budget:
            if len(best)>=k and -heap[0][0]<best[-1][0]:break
            _,p,b=heapq.heappop(heap);n+=1
            for c,q in self.ref.charprob(p).items():
                if c==B.EOS:
                    if p not in visited:
                        visited.add(p); best=sorted(best+[(self.complete(p),p)],reverse=True)[:k]
                else:
                    t=p+c; nb=b*q; ub=self.mass(t,nb)
                    if ub>0 and (len(best)<k or ub>=best[-1][0]):heapq.heappush(heap,(-ub,t,nb))
        cert=not heap or (len(best)>=k and -heap[0][0]<best[-1][0])
        return [w for _,w in best],cert,n

class Foundation(B.Ref):
    def __init__(self,docs,lex):
        super().__init__(docs,lex)
        self.context=C.defaultdict(C.Counter); self.follow=C.defaultdict(C.Counter)
        for doc in docs:
            for i,w in enumerate(doc):
                if i:
                    self.context[w][("L",doc[i-1])]+=1
                    self.follow[doc[i-1]][w]+=1
                if i+1<len(doc):self.context[w][("R",doc[i+1])]+=1
        self.roles={}; self.post=C.defaultdict(list)
        for w,c in sorted(self.context.items()):
            # Distributional role; remove uninformative ubiquitous contexts.
            vals={k:math.log1p(n)/math.sqrt(1+math.log1p(self.uni[k[1]])) for k,n in c.items()}
            vals=dict(sorted(vals.items(),key=lambda x:(-x[1],x[0]))[:20])
            z=math.sqrt(sum(x*x for x in vals.values()))
            if z and self.uni[w]>=3:
                self.roles[w]={k:v/z for k,v in vals.items()}
                for k in vals:self.post[k].append(w)
        self.char_weight=self.unknown
        self.lex_weight=.015
        self.bgword={w:(1-self.char_weight-self.lex_weight)*v for w,v in self.globalp.items()}
        for w in sorted(self.lex):self.bgword[w]=self.bgword.get(w,0.)+self.lex_weight/len(self.lex)
        self.bgprefix=C.Counter()
        for w,v in self.bgword.items():
            for i in range(len(w)+1):self.bgprefix[w[:i]]+=v
        self.bg_top=sorted(self.bgword,key=lambda w:(-self.background(w),w))[:128]
        self.diag=C.Counter()
    @lru_cache(maxsize=65536)
    def background(self,w):return self.bgword.get(w,0.)+self.char_weight*self.baseword(w)
    @lru_cache(maxsize=8192)
    def analog(self,w):
        x=self.roles.get(w)
        if not x:return {}
        candidates=set()
        # Select discriminative context features, not huge postings.
        for k in sorted(x,key=lambda k:(len(self.post[k]),-x[k],k))[:8]:
            if len(self.post[k])<=512:candidates.update(self.post[k])
        near=[]
        for v in sorted(candidates):
            if v==w:continue
            sim=B.dot(x,self.roles[v])
            if sim>=.25:near.append((sim,v))
        near=sorted(near,reverse=True)[:12]
        out={}
        for sim,v in near:
            c=self.follow[v]; n=sum(c.values())
            if n:
                for u,k in c.items():out[u]=out.get(u,0.)+sim**3*k/n
        return B.norm(out)
    @lru_cache(maxsize=32768)
    def semantic(self,w):
        # 8D point projection combines spelling and distributional role, same upper addresses.
        v={k:.30*x for k,x in B.lexvec(w).items()}
        for key,x in self.roles.get(w,{}).items():
            h=B.addr(repr(key));k=(128+h%128,(h>>8)%8)
            v[k]=v.get(k,0.)+(1 if h>>16&1 else -1)*x
        z=math.sqrt(B.dot(v,v))
        return {k:x/z for k,x in v.items()} if z else {}
    def field(self,h,src,s,bg,mode):
        # Independent support/refutation relative to lower-order distribution.
        # Restrict graph work, never truncate probability support.
        top=heapq.nlargest(48,s,key=lambda w:(s[w]+bg*self.background(w),w))
        if not top or not src:return s,bg
        top=sorted(top); edges={};lo={};up={};sv=[]
        def edge(i,j,d,v):
            if i==j:return
            i,j=min(i,j),max(i,j);k=(B.pair(i,j),i,j,d)
            a,b=edges.get(k,(0.,0.));edges[k]=(a+max(v,0.),b+max(-v,0.))
        for key,c,weight in src:
            l=B.lower(key);lo[l]=lo.get(l,0.)+weight
            z=math.fsum(c.values());v={}
            for w in top:
                p=c.get(w,0.)/z
                # shrink context likelihood ratios to avoid one-count certainty
                prior=self.background(w)
                evidence=weight*math.tanh(math.log((p+.25*prior+1e-12)/(1.25*prior+1e-12))/3.)
                if p==0:evidence*=.15
                for (u,d),q in self.semantic(w).items():
                    edge(l[0],u,l[1],evidence*q)
                    k=(u,d);up[k]=up.get(k,0.)+evidence*q
                    v[k]=v.get(k,0.)+evidence*q
            sv.append((l,B.bounded(v)))
        for i,(l,v) in enumerate(sv):
            for r,u in sv[i+1:]:
                if l[1]==r[1]:edge(l[0],r[0],l[1],B.dot(v,u))
        for w in sorted(top,key=lambda w:(-s[w],w))[:12]:
            bydim=C.defaultdict(list)
            for (u,d),v in sorted(self.semantic(w).items()):bydim[d].append((u,v))
            for d,a in bydim.items():
                for i,(u,v) in enumerate(a):
                    for r,q in a[i+1:]:edge(u,r,d,math.sqrt(s[w])*v*q)
        anchor=B.bounded(B.add(B.bounded(lo),B.bounded(up)))
        signs=(mode!="minus",mode!="plus")
        x,ok=B.stable_field(edges,anchor,self.params,signs)
        self.diag["attempts"]+=1;self.diag["converged"]+=ok
        if not ok:return s,bg
        # Change only conditional mass in the selected subspace. All other
        # conditional words and the full background retain exact mass.
        budget=math.fsum(s[w] for w in top)
        def score(w):
            v=self.semantic(w)
            return max(-1.,min(1.,sum(q*(x.get((i,d),0.)+.125*sum(x.get((i,t),0.) for t in range(8))) for (i,d),q in v.items())))
        changed=B.norm({w:s[w]*math.exp(.75*score(w)) for w in top})
        result=dict(s)
        for w in top:result[w]=budget*changed[w]
        return result,bg
    def predict(self,h,episode=None,mode="signed"):
        if mode in ("retrieval","hierarchical","unified"):return super().predict(h,episode,mode)
        s={};bg=1.;src=[]
        def mix(key,c,lam):
            nonlocal s,bg
            if not c or lam<=0:return
            p=B.norm(c);s=blend_sparse(s,1-lam,p,lam);bg*=1-lam
            src.append((key,p,lam))
        def discount(key,c):
            nonlocal s,bg
            if not c:return
            n=sum(c.values());d=.75
            p={w:max(0.,v-d)/n for w,v in c.items()}
            lam=math.fsum(p.values())
            # absolute discounting backs off instead of forcing a singleton
            s=blend_sparse(s,1-lam,p,1.);bg*=1-lam
            src.append((key,B.norm(p),lam))
        if h and mode not in ("exact",):
            a=self.analog(h[-1])
            if a:mix(("analog",h[-1]),a,.24 if self.uni[h[-1]]<8 else .14)
        # Discourse topic distribution is aggregated as one low-trust source.
        if len(h)>4:
            ds={}
            for w in sorted(set(h[-32:-4])):
                c=self.idx.get(("discourse",(w,)))
                if c:B.add(ds,B.norm(c),1/math.sqrt(1+self.uni[w]))
            if ds:mix(("topic",tuple(h[-8:])),ds,.12)
        if len(h)>=3:
            c=self.idx.get(("skip",(h[-3],h[-1])))
            if c:mix(("skip",(h[-3],h[-1])),c,min(.35,sum(c.values())/(sum(c.values())+12)))
        for n in (1,2,3,4):
            if len(h)>=n:
                key=("local"+str(n),tuple(h[-n:]));c=self.idx.get(key)
                if c:discount(key,c)
        # Document-local episodic counts only after observing the target.
        if episode:
            for n in (1,2,3,4):
                if len(h)>=n:
                    k=("local"+str(n),tuple(h[-n:]));c=episode.get(k)
                    if c:
                        count=sum(c.values());mix(("episode",k),c,min(.70,count/(count+3)))
        if mode in ("signed","plus","minus"):s,bg=self.field(h,src[-8:],s,bg,mode)
        return FullLattice(self,s,bg)

def evaluate(ref,docs,mode,limit=72):
    m=C.Counter();start=time.monotonic()
    for doc in docs:
        h=[];ep=C.defaultdict(C.Counter)
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                l=ref.predict(h,ep,mode);top,cert,n=l.rank(5)
                m["n"]+=1;m["top1"]+=top[0]==w;m["top5"]+=w in top
                m["certified"]+=cert;m["conditional_support"]+=w in l.p
                m["shortlist32"]+=w in heapq.nlargest(32,l.p,key=lambda v:(l.complete(v),v))
                m["shortlist128"]+=w in heapq.nlargest(128,l.p,key=lambda v:(l.complete(v),v))
                m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                p="";b=1.;prod=1.
                for c in w+B.EOS:
                    d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1))
                    prod*=d[c]
                    if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            B.observe(ep,h,w);h=(h+[w])[-64:]
    n=m["n"];return dict(m,word_top1=m["top1"]/n,word_top5=m["top5"]/n,spelling_bps=m["bits"]/m["symbols"],seconds=time.monotonic()-start)

def integrity(ref):
    h=["in","the","middle","of","the"]
    for mode in ("exact","analog","signed"):
        l=ref.predict(h,None,mode)
        assert abs(l.mass("",1)-1)<1e-9
        assert l.rank()==ref.predict(h,None,mode).rank()
        # Verify a rare known word outside a dynamic shortlist remains positive.
        w=next(w for w in sorted(ref.globalp) if w not in l.p)
        assert l.complete(w)>=l.bg*ref.background(w)>0
        for w in [w,"xylophonically",*l.rank()[0]]:
            p="";b=1.;prob=1.
            for c in w+B.EOS:
                d=l.next(p,b);assert abs(math.fsum(d.values())-1)<1e-8;prob*=d[c]
                if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
            assert math.isclose(prob,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
    return {"full_support":True,"prefix_identity":True,"repeat_determinism":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics-v4.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")}
    split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:24]
    lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=Foundation(docs,lex)
    rawdocs=[p.read_bytes()[:36000] for _,p in tr];raw=B.RawByte(rawdocs)
    modes=("retrieval","hierarchical","exact","analog","signed")
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"reference_ids":[i for i,_ in tr],"lexicon":len(lex),"roles":len(ref.roles),"checks":integrity(ref),"matched":{},"fresh_dev":{}}
    def run(name,ds,dest,ms=modes,limit=72):
        dest[name]={mode:evaluate(ref,ds,mode,limit) for mode in ms}
        print(name,json.dumps(dest[name]),flush=True)
    for k,files in groups.items():
        picked=[(i,p) for i,p in files if split(i) in (7,8)][:5]
        run(k,[B.load(p,300) for _,p in picked],out["matched"])
    for k,p in (("wikitext",r/"wiki"/"wiki.valid.raw"),("chemistry",r/"domain"/"chemistry.txt")):
        text=p.read_text(encoding="utf-8",errors="replace")
        run(k,[B.words(text[i:i+8000])[:400] for i in range(0,min(len(text),40000),8000)],out["matched"])
    # Fresh windows are disjoint by ID across representations and from matched books.
    used={i for files in groups.values() for i,p in [x for x in files if split(x[0]) in (7,8)][:5]}
    fresh={}
    for k,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used:continue
            with p.open("rb") as f:
                size=p.stat().st_size
                f.seek(min(size//3,100000));w=B.words(f.read(12000).decode("utf-8","replace"))
            if len(w)<112:continue
            used.add(i);chosen.append((i,w[:200]))
            if len(chosen)==6:break
        assert len(chosen)==6
        fresh[k]=[i for i,w in chosen]
        run(k,[w for i,w in chosen],out["fresh_dev"],modes,96)
    out["fresh_ids"]=fresh
    rawtests=[p.read_bytes()[:768] for i,p in groups["stories"] if split(i) in (7,8)][:5]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:rawtests.append(f.read(768))
    out["raw"]={mode:B.byteeval(ref,raw,rawtests,mode) for mode in ("byte_only",*modes)}
    # directional ablation and causal counterfactual on actual reference-disjoint histories
    chosen=[B.load(p,150) for i,p in groups["stories"] if split(i) in (7,8)][:3]
    out["ablation"]={mode:evaluate(ref,chosen,mode,24) for mode in ("analog","signed","plus","minus")}
    # test the prediction's log probability under correct vs unrelated history, no episodic updates
    losses=C.Counter()
    for i,doc in enumerate(chosen):
        other=chosen[(i+1)%len(chosen)]
        for j in range(16,40):
            w=doc[j]
            for mode in ("analog","signed"):
                for label,h in (("correct",doc[:j]),("unrelated",other[:j])):
                    l=ref.predict(h,None,mode);losses[mode+"_"+label]-=math.log2(max(l.complete(w),1e-300))
    out["counterfactual_bits_per_word"]={k:v/72 for k,v in losses.items()}
    out["graph"]=dict(ref.diag);out["checks"]["raw_future_invariance"]=B.rawcheck(ref,raw,rawtests[0])
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
