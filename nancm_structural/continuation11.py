"""Continuation-discount structural diagnostic; frozen reference counts, no optimization."""
import base as B, foundation4 as F, foundation6 as N, foundation8 as E, transport_scale as A
import argparse, collections as C, functools, heapq, json, math, pathlib, time

class Background:
    def __init__(self,ref,cont):
        self.ref=ref
        self.char_weight=.35*ref.char_weight
        self.bgword={w:.35*v for w,v in ref.bgword.items()}
        B.add(self.bgword,B.norm(cont),.65)
        self.bgprefix=C.Counter()
        for w,v in self.bgword.items():
            for i in range(len(w)+1):self.bgprefix[w[:i]]+=v
        self.bg_rank=sorted(self.bgword,key=lambda w:(self.background(w),w),reverse=True)
        assert abs(math.fsum(self.bgword.values())+self.char_weight-1)<1e-9
    def charprob(self,p):return self.ref.charprob(p)
    def baseword(self,w):return self.ref.baseword(w)
    @functools.lru_cache(maxsize=65536)
    def background(self,w):return self.bgword.get(w,0.)+self.char_weight*self.baseword(w)

class Continuation(A.Alignment):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex)
        self.kernel=kernel;self.tables={};self.discounts={}
        # Count distinct preceding words for lower-order context transitions,
        # not occurrences. Highest orders retain actual frequency counts.
        for n in range(5):
            counts=C.defaultdict(C.Counter);seen=set()
            for doc in docs:
                for i in range(n+1,len(doc)):
                    ctx=tuple(doc[i-n:i]);w=doc[i]
                    key=(doc[i-n-1],ctx,w)
                    if key not in seen:
                        seen.add(key);counts[ctx][w]+=1
            if n==0:self.background_kn=Background(self,counts[()])
            else:
                self.tables[n]=counts
                hist=C.Counter(v for c in counts.values() for v in c.values())
                self.discounts[n]=max(.30,min(.90,hist[1]/max(1,hist[1]+2*hist[2])))
            del seen
        for n in (5,6,7,8):
            counts=C.defaultdict(C.Counter)
            for doc in docs:
                for i in range(n,len(doc)):counts[tuple(doc[i-n:i])][doc[i]]+=1
            self.tables[n]=counts
            hist=C.Counter(v for c in counts.values() for v in c.values())
            self.discounts[n]=max(.30,min(.90,hist[1]/max(1,hist[1]+2*hist[2])))
    def solve_field(self,edges,anchor,signs):
        import foundation5 as V
        active,scale=V.operator(edges,self.params,signs)
        return self.kernel.solve(active,scale,{k:.65*v for k,v in anchor.items()})
    def law(self,h,ep,recent):
        s={};bg=1.;src=[]
        for n in range(1,min(8,len(h))+1):
            key=tuple(h[-n:]);c=self.tables[n].get(key)
            if not c:continue
            total=sum(c.values());d=self.discounts[n]
            q={w:(v-d)/total for w,v in c.items() if v>d}
            mass=math.fsum(q.values())
            s=F.blend_sparse(s,1-mass,q,1.);bg*=1-mass
            src.append((("continuation",n,key),B.norm(q),mass))
        # Episode counts only observed earlier in this document.
        for n in range(1,min(8,len(h))+1):
            k=tuple(h[-n:]);c=ep.get(k)
            if c:
                count=sum(c.values());mass=min(.75,count/(count+(5 if n==1 else 2)))
                q=B.norm(c);s=F.blend_sparse(s,1-mass,q,mass);bg*=1-mass
                src.append((("episode",n,k),q,mass))
        return s,bg,src
    def hypotheses(self,h,ep,recent,adaptive):
        s,bg,src=self.law(h,ep,recent);kn=E.FastLattice(self.background_kn,s,bg)
        q,trust=self.alignment(tuple(h[-8:]))
        if not q:return kn,kn,kn,0.
        # Never wholesale-replace the base. Weight contextual evidence by
        # confidence AND agreement with exact evidence, with a small floor.
        overlap=math.fsum(min(v,kn.complete(w)) for w,v in q.items())
        # Fixed causal rules: no DEV calibration.
        lam=min(.28,trust*(.18+.65*overlap)*adaptive)
        a=F.blend_sparse(s,1-lam,q,lam);abg=bg*(1-lam)
        aligned=E.FastLattice(self.background_kn,a,abg)
        src.append((("aligned",tuple(h[-8:])),q,lam))
        # Reuse the signed 128+128 geometry and persistent gain layout.
        # The graph only redistributes selected conditional mass.
        if a and lam>.01:
            g,gbg=self.field(h,src[-6:],a,abg,"signed")
        else:g,gbg=a,abg
        signed=E.FastLattice(self.background_kn,g,gbg)
        assert abs(math.fsum(g.values())+gbg-1)<1e-8
        return kn,aligned,signed,lam

def evaluate(ref,docs,limit):
    modes=("exact","kn","aligned","signed","adaptive")
    ms={k:C.Counter() for k in modes};diag=C.Counter();start=time.monotonic()
    for doc in docs:
        h=[];recent=[];ep=C.defaultdict(C.Counter);old=C.defaultdict(C.Counter);logodds=0.
        for i,w in enumerate(doc[:limit+16]):
            exact0=ref.predict(h,old,"exact");exact=E.FastLattice(ref,exact0.s,exact0.bg)
            kn,a,g,lam=ref.hypotheses(h,ep,recent,1.)
            # Compare fixed aligned and KN via prior observed likelihoods.
            # A bounded document-local mixture; no future target access.
            trust=1/(1+math.exp(-logodds))
            weight=.05+.70*trust
            adaptive=E.FastLattice(ref.background_kn,F.blend_sparse(kn.s,1-weight,g.s,weight),(1-weight)*kn.bg+weight*g.bg)
            laws=(exact,kn,a,g,adaptive)
            if i>=16:
                diag["n"]+=1;diag["lambda"]+=lam;diag["adaptive_weight"]+=weight
                for mode,l in zip(modes,laws):
                    m=ms[mode];rank,cert,n=l.rank();m["n"]+=1;m["top1"]+=w==rank[0];m["top5"]+=w in rank
                    m["certified"]+=cert;m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    m["shortlist128"]+=w in heapq.nlargest(128,l.s,key=lambda v:(l.complete(v),v))
                    if diag["n"]<=24:
                        p="";b=1.;prod=1.
                        for c in w+B.EOS:
                            d=l.next(p,b);m["mass_error"]=max(m["mass_error"],abs(math.fsum(d.values())-1));prod*=d[c]
                            if c!=B.EOS:b*=ref.charprob(p)[c];p+=c
                        assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            logodds=max(-6.,min(6.,.94*logodds+.5*math.log(max(g.complete(w),1e-300)/max(kn.complete(w),1e-300))))
            for n in range(1,min(8,len(h))+1):ep[tuple(h[-n:])][w]+=1
            B.observe(old,h,w);h=(h+[w])[-64:];recent=(recent+[w])[-128:]
    return {"modes":{k:dict(m,word_top1=m["top1"]/m["n"],word_top5=m["top5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"diag":dict(diag),"seconds":time.monotonic()-start}

def checks(ref,docs):
    h=docs[0][:20]
    def at(data,n):
        prefix=[];ep=C.defaultdict(C.Counter)
        for w in data[:n]:
            for k in range(1,min(8,len(prefix))+1):ep[tuple(prefix[-k:])][w]+=1
            prefix.append(w)
        laws=ref.hypotheses(prefix,ep,prefix,1.)[:3]
        return [(l.s,l.bg,l.rank()) for l in laws]
    assert at(h,16)==at(h[:16]+["unrelated","suffix"],16)
    return {"suffix_invariance":True,"background_mass":math.fsum(ref.background_kn.bgword.values())+ref.background_kn.char_weight}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in groups["stories"] if split(i)<7][:96];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=Continuation(docs,lex,N.Native(a.lib))
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"reference_ids":[i for i,_ in tr],"discounts":ref.discounts,"checks":checks(ref,story),"fresh":{}}
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
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
