"""v5 contractive differentiable field and causal Bayesian byte routing; no training."""
import base as B, foundation4 as F
import argparse, collections as C, json, math, pathlib, time

def operator(edges,params,signs):
    # Parameter-independent degree bound avoids differentiating a max over weights.
    degree=C.Counter();active=[]
    for (slot,i,j,d),(a,b) in sorted(edges.items()):
        ki,kj=(i,d),(j,d);degree[ki]+=1;degree[kj]+=1
        v=params.gain(slot,d,a,b,signs)
        active.append((slot*8+d,ki,kj,a,b,v))
    return active,max(1,max(degree.values(),default=0))
def fixed(active,scale,forcing,tol=1e-12):
    x=dict(forcing)
    for _ in range(48):
        q=dict(forcing)
        for _,i,j,_,_,v in active:
            w=.35*v/scale
            q[i]=q.get(i,0.)+w*x.get(j,0.)
            q[j]=q.get(j,0.)+w*x.get(i,0.)
        err=math.sqrt(math.fsum((q.get(k,0.)-x.get(k,0.))**2 for k in q.keys()|x.keys()))
        if err<=tol:return q,True
        x=q
    return dict(forcing),False

class DiffFoundation(F.Foundation):
    def __init__(self,*a):
        super().__init__(*a);self.capture=False;self.trace=None
    def solve_field(self,edges,anchor,signs):
        active,scale=operator(edges,self.params,signs)
        x,ok=fixed(active,scale,{k:.65*v for k,v in anchor.items()})
        assert not ok or math.sqrt(B.dot(x,x))<=1+1e-9
        return x,ok
    def record_field(self,edges,anchor,x,top,s,bg,changed,budget,signs):
        if self.capture:self.trace=(edges,x,top,bg,changed,budget,signs)
    def gradient(self,target):
        edges,x,top,bg,q,budget,signs=self.trace
        if target not in q:return {},{}
        probability=budget*q[target]+bg*self.background(target)
        g={};factor=budget*q[target]/probability
        for w in top:
            v=self.semantic(w)
            raw=sum(a*(x.get((i,d),0.)+.125*sum(x.get((i,t),0.) for t in range(8))) for (i,d),a in v.items())
            if abs(raw)>=1:continue
            coeff=.75*factor*(q[w]-(w==target))
            for (i,d),a in v.items():
                for t in range(8):
                    k=(i,t);g[k]=g.get(k,0.)+coeff*a*((d==t)+.125)
        active,scale=operator(edges,self.params,signs)
        adj,ok=fixed(active,scale,g);assert ok
        gp=C.Counter();gm=C.Counter()
        for k,i,j,a,b,v in active:
            common=.35/scale*(adj.get(i,0.)*x.get(j,0.)+adj.get(j,0.)*x.get(i,0.))*(1-v*v)
            u=self.params.plus[k];t=self.params.minus[k]
            if signs[0] and abs(u)<4:gp[k]+=common*a*math.exp(u)
            if signs[1] and abs(t)<4:gm[k]-=common*b*math.exp(t)
        return dict(gp),dict(gm)

def gradcheck(ref,docs):
    before=ref.params.payload();checks=[];ref.capture=True
    for doc in docs:
        for i in range(16,min(60,len(doc))):
            h=doc[:i];w=doc[i];ref.trace=None
            l=ref.predict(h,None,"signed")
            if not ref.trace or w not in ref.trace[4]:continue
            gp,gm=ref.gradient(w)
            if not gp or not gm:continue
            if min(max(map(abs,gp.values())),max(map(abs,gm.values())))<1e-8:continue
            for name,channel,grad in (("S+",ref.params.plus,gp),("S-",ref.params.minus,gm)):
                k=max(grad,key=lambda k:abs(grad[k]));eps=.01
                try:
                    channel[k]=eps;positive=-math.log(ref.predict(h,None,"signed").complete(w))
                    channel[k]=-eps;negative=-math.log(ref.predict(h,None,"signed").complete(w))
                finally:channel[k]=0.
                numeric=(positive-negative)/(2*eps)
                error=abs(numeric-grad[k]);relative=error/max(abs(numeric),abs(grad[k]),1e-10)
                checks.append({"channel":name,"address":k,"analytic":grad[k],"finite_difference":numeric,"relative_error":relative})
                assert relative<.015 or error<2e-8,checks[-1]
            break
        if len(checks)>=12:break
    ref.capture=False;assert ref.params.payload()==before
    assert len(checks)>=6,checks
    return {"checks":checks,"max_relative_error":max(x["relative_error"] for x in checks),"parameters_unchanged":True}

class BayesianBytes:
    """A latent raw/word route selected per ASCII-word segment.
    Posterior is updated only after actual observed bytes, not from future targets."""
    def __init__(self,ref,raw,mode="signed"):
        self.w=B.ByteSession(ref,raw,mode);self.raw=raw
        self.prior=.5;self.posterior=self.prior;self.cache=None
    def probs(self):
        if self.cache is None:
            a=self.raw.probs(self.w.tail);b=self.w.probs();g=self.posterior
            p=[(1-g)*x+g*y for x,y in zip(a,b)]
            self.cache=(p,b)
        return self.cache[0]
    def observe(self,b):
        p=self.probs();word=self.cache[1]
        self.posterior=self.posterior*word[b]/p[b]
        self.w.observe(b);self.cache=None
        if b not in B.LETTERS:self.posterior=self.prior
def byte_eval(ref,raw,docs,mode):
    bits=0.;one=five=n=0;err=0.;start=time.monotonic()
    for doc in docs:
        s=BayesianBytes(ref,raw,"signed") if mode=="bayes" else B.ByteSession(ref,raw,mode)
        for i,b in enumerate(doc[:768]):
            p=s.probs();assert len(p)==256 and min(p)>=0
            err=max(err,abs(math.fsum(p)-1))
            if i>=64:
                order=sorted(range(256),key=lambda j:(-p[j],j))[:5];n+=1;one+=b==order[0];five+=b in order
                bits-=math.log2(max(p[b],1e-300))
            s.observe(b)
    return {"bytes":n,"raw_bpb":bits/n,"byte_top1":one/n,"byte_top5":five/n,"mass_error":err,"seconds":time.monotonic()-start}
def bytecheck(ref,raw,doc):
    # Within one alphabetic segment, the joint mixture equals its two route sequence laws.
    seq=next(x.encode() for x in B.words(doc.decode("utf-8","replace")) if len(x)>=5)+b" "
    bayes=BayesianBytes(ref,raw);ws=B.ByteSession(ref,raw,"signed");rs=B.ByteSession(ref,raw,"byte_only")
    a=b=c=1.
    for token in seq:
        a*=bayes.probs()[token];b*=ws.probs()[token];c*=rs.probs()[token]
        for s in (bayes,ws,rs):s.observe(token)
    assert math.isclose(a,.5*(b+c),rel_tol=1e-9,abs_tol=1e-30)
    def at(data,k):
        s=BayesianBytes(ref,raw)
        for b in data[:k]:s.observe(b)
        return s.probs()
    for k in (1,8,31,64,127):
        assert at(doc,k)==at(doc[:k]+b"unrelated future",k)
    return {"segment_mixture_identity":True,"future_suffix_invariance":True}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=B.bookfiles(r/"stories");tr=[(i,p) for i,p in books if split(i)<7][:24]
    picked=[(i,p) for i,p in books if split(i) in (7,8)][:5]
    lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=DiffFoundation(docs,lex)
    raw=B.RawByte([p.read_bytes()[:36000] for _,p in tr]);dev=[B.load(p,300) for _,p in picked]
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"gradient_probe":gradcheck(ref,dev),"word":{}}
    for mode in ("retrieval","exact","analog","signed"):
        out["word"][mode]=F.evaluate(ref,dev,mode)
    print("WORD",json.dumps(out["word"]),flush=True)
    rawtests=[p.read_bytes()[:768] for _,p in picked]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:rawtests.append(f.read(768))
    out["bytecheck"]=bytecheck(ref,raw,rawtests[0])
    out["byte"]={mode:byte_eval(ref,raw,rawtests,mode) for mode in ("byte_only","retrieval","analog","signed","bayes")}
    out["convergence"]=dict(ref.diag)
    out["checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
