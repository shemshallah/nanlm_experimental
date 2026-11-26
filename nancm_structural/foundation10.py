"""v10 unified gradient/optimizer checkpoint contract. No corpus optimizer updates."""
import base as B, foundation6 as N, foundation9 as T
import argparse, array, collections as C, hashlib, json, math, pathlib, struct, sys, time

D=9
class Unified(T.Transport):
    def __init__(self,docs,lex,kernel):
        super().__init__(docs,lex,kernel)
        self.gate=[0.]*(len(self.names)*D)
class Session(T.Session):
    def weights_for(self,mode):
        if mode!="signed_aligned":return super().weights_for(mode)
        ids,conditional=T.MODES[mode];bucket=self.bucket()
        local=self.buckets.get(bucket,[0.]*len(self.logs))
        n=self.bucket_counts[bucket];strength=min(.85,n/(n+4))
        last=self.h[-1] if self.h else ""
        freq=min(1.,math.log1p(self.ref.uni[last])/10)
        tri=float(len(self.h)>=3 and bool(self.ref.idx.get(("local3",tuple(self.h[-3:])))))
        features={};scores={}
        for i in ids:
            l=self.cache[i]
            # Features are invariant under signed mass redistribution; this
            # prevents an unaccounted gradient path through routing features.
            x=(1.,freq,tri,len(self.h)/64,l.bg,min(1.,math.log1p(len(l.s))/10),self.logs[i]/30,local[i]/30,strength)
            features[i]=x
            scores[i]=(1-strength)*self.logs[i]+strength*local[i]+sum(self.ref.gate[i*D+j]*v for j,v in enumerate(x))
        mx=max(scores.values());z=B.norm({i:math.exp(scores[i]-mx) for i in ids})
        self.gate_trace=(features,z)
        return [.9*z[i]+.1/len(ids) if i in z else 0. for i in range(len(self.logs))]
    def full_gradient(self,w):
        l=self.predict();prob=l.complete(w)
        features,z=self.gate_trace
        ps={i:self.cache[i].complete(w) for i in z}
        average=math.fsum(z[i]*ps[i] for i in z)
        gate={}
        for i,x in features.items():
            coeff=.9*z[i]*(average-ps[i])/prob
            for j,v in enumerate(x):gate[i*D+j]=coeff*v
        gp,gm={},{}
        if self.ref.trace is not None:
            gp,gm=self.ref.gradient(w)
            # chain rule through the selected signed expert into total mixture
            ratio=self.weights[8]*self.cache[8].complete(w)/prob
            gp={k:v*ratio for k,v in gp.items()};gm={k:v*ratio for k,v in gm.items()}
        return {"gate":gate,"plus":gp,"minus":gm}

class SparseAdam:
    """One optimizer, three parameter groups. Explicit active-coordinate semantics.
    Absent coordinates do not advance their local clock or undergo decay."""
    def __init__(self,lr=.001,beta1=.9,beta2=.999,eps=1e-8,decay=.01):
        self.config=[lr,beta1,beta2,eps,decay];self.step=0;self.moments={}
    def apply(self,groups,gradient):
        lr,b1,b2,eps,decay=self.config
        # validate whole gradient before mutating any state
        for group,items in gradient.items():
            if group not in groups:raise ValueError("unknown group")
            for k,g in items.items():
                if not 0<=k<len(groups[group]) or not math.isfinite(g):raise ValueError("invalid gradient")
        updates=[]
        for group,items in sorted(gradient.items()):
            for k,g in sorted(items.items()):
                if g==0:continue
                key=(group,k);m,v,t=self.moments.get(key,(0.,0.,0));t+=1
                m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
                value=groups[group][k]*(1-lr*decay)-lr*(m/(1-b1**t))/(math.sqrt(v/(1-b2**t))+eps)
                if not all(map(math.isfinite,(value,m,v))):raise ValueError("nonfinite update")
                updates.append((group,k,value,(m,v,t)))
        for group,k,value,mom in updates:
            groups[group][k]=value;self.moments[group,k]=mom
        self.step+=1
    def encode(self):
        return {"algorithm":"active-coordinate-adamw-v1","config":self.config,"step":self.step,"moments":[[g,k,*v] for (g,k),v in sorted(self.moments.items())]}
    @classmethod
    def decode(cls,d):
        if d["algorithm"]!="active-coordinate-adamw-v1":raise ValueError("algorithm")
        o=cls(*d["config"]);o.step=d["step"]
        o.moments={(g,k):(m,v,t) for g,k,m,v,t in d["moments"]}
        return o

def groups(params,gate):return {"plus":params.plus,"minus":params.minus,"gate":gate}
def serialize(params,gate,optimizer,corpus):
    meta=json.dumps({"version":1,"corpus_sha256":corpus,"gate":gate,"optimizer":optimizer.encode()},sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    body=struct.pack("<I",len(meta))+meta+params.payload()
    return b"NCMTRAIN1"+hashlib.sha256(body).digest()+body
def deserialize(data,corpus):
    if len(data)<45 or data[:9]!=b"NCMTRAIN1":raise ValueError("checkpoint header")
    body=data[41:]
    if hashlib.sha256(body).digest()!=data[9:41]:raise ValueError("checkpoint digest")
    n,=struct.unpack("<I",body[:4]);d=json.loads(body[4:4+n]);raw=body[4+n:]
    if d["version"]!=1 or d["corpus_sha256"]!=corpus:raise ValueError("checkpoint identity")
    p=B.SignedParameters()
    if len(raw)!=20+8*p.N or raw[:20]!=b"NCMSIGN1"+struct.pack("<III",256,8,p.N):raise ValueError("geometry")
    for off,channel in ((20,"plus"),(20+4*p.N,"minus")):
        x=array.array("f");x.frombytes(raw[off:off+4*p.N])
        if sys.byteorder!="little":x.byteswap()
        if not all(map(math.isfinite,x)):raise ValueError("invalid weights")
        setattr(p,channel,x)
    gate=d["gate"]
    if len(gate)!=9*D or not all(map(math.isfinite,gate)):raise ValueError("gate geometry")
    o=SparseAdam.decode(d["optimizer"])
    for (g,k),(m,v,t) in o.moments.items():
        if g not in groups(p,gate) or not 0<=k<len(groups(p,gate)[g]) or not math.isfinite(m) or not math.isfinite(v) or v<0 or not 1<=t<=o.step:raise ValueError("invalid moments")
    return p,gate,o

def optimizer_probe(corpus):
    # Synthetic gradients on discarded clones only: no optimizer step on real data.
    p=B.SignedParameters();g=[0.]*(9*D);o=SparseAdam()
    gradients=[{"plus":{42:.2},"minus":{100:-.1},"gate":{3:.5}},
               {"plus":{42:-.4,51:.05},"gate":{3:.25,8:-.2}}]
    o.apply(groups(p,g),gradients[0])
    data=serialize(p,g,o,corpus);p2,g2,o2=deserialize(data,corpus)
    assert data==serialize(p2,g2,o2,corpus)
    o.apply(groups(p,g),gradients[1]);o2.apply(groups(p2,g2),gradients[1])
    assert serialize(p,g,o,corpus)==serialize(p2,g2,o2,corpus)
    bad=bytearray(data);bad[-1]^=1
    try:deserialize(bytes(bad),corpus)
    except ValueError:pass
    else:raise AssertionError("corruption not rejected")
    prior=serialize(p,g,o,corpus)
    try:o.apply(groups(p,g),{"gate":{0:1.,1:float("nan")}})
    except ValueError:pass
    else:raise AssertionError("invalid gradient accepted")
    assert prior==serialize(p,g,o,corpus)
    return {"synthetic_updates_per_clone":2,"resume_exact":True,"corruption_rejected":True,"nonfinite_gradient_atomic_rejection":True,"real_model_updates":0}

def gradients(ref,docs):
    original=ref.params.payload();orig_gate=list(ref.gate);checks=[];parity=0;parity_error=0.
    for doc in docs:
        s=Session(ref);old=T.Session(ref)
        # Replay identical observed prefix and compare zero-init predictions.
        for i,w in enumerate(doc[:50]):
            ref.capture=False
            a=s.predict();b=old.predict()
            err=max(abs(a.complete(v)-b.complete(v)) for v in set(a.s)|set(b.s)|{w})
            parity_error=max(parity_error,err);assert err<1e-12
            parity+=1
            if i>=16:
                s.cache=None;s.prob=None;ref.trace=None;ref.capture=True
                l=s.predict();g=s.full_gradient(w)
                if all(g[k] and max(map(abs,g[k].values()))>1e-8 for k in ("plus","minus","gate")):
                    for group,channel in groups(ref.params,ref.gate).items():
                        grad=g[group];k=max(grad,key=lambda k:abs(grad[k]));eps=.005
                        vals=[]
                        try:
                            for d in (eps,-eps):
                                channel[k]=d;s.prob=None;s.cache=None;ref.trace=None
                                vals.append(-math.log(s.predict().complete(w)))
                        finally:
                            channel[k]=0.;s.prob=None;s.cache=None;ref.trace=None
                        numeric=(vals[0]-vals[1])/(2*eps)
                        error=abs(numeric-grad[k]);rel=error/max(abs(numeric),abs(grad[k]),1e-10)
                        checks.append({"channel":group,"index":k,"analytic":grad[k],"numeric":numeric,"relative_error":rel})
                        assert rel<.02 or error<2e-8,checks[-1]
                    ref.capture=False
                    break
            ref.capture=False;s.observe(w);old.observe(w)
        if len(checks)>=9:break
    assert len(checks)>=6 and ref.params.payload()==original and ref.gate==orig_gate,checks
    return {"checks":checks,"max_relative_error":max(v["relative_error"] for v in checks),"zero_gate_parity_boundaries":parity,"max_parity_error":parity_error,"parameters_unchanged":True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    books=B.bookfiles(r/"stories");split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    tr=[(i,p) for i,p in books if split(i)<7][:24];picked=[(i,p) for i,p in books if split(i) in (7,8)][:5];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==24 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];dev=[B.load(p,300) for _,p in picked]
    corpus=hashlib.sha256(json.dumps({"books":[(i,w) for (i,p),w in zip(tr,docs)],"lex":sorted(lex)},separators=(",",":")).encode()).hexdigest()
    ref=Unified(docs,lex,N.Native(a.lib))
    out={"training_updates":0,"reference_words":sum(map(len,docs)),"corpus_sha256":corpus,"signed_parameters":2*B.SignedParameters.N,"router_parameters":len(ref.gate),"optimizer_probe":optimizer_probe(corpus)}
    out["gradient_probe"]=gradients(ref,dev);print("GRAD",json.dumps(out["gradient_probe"]),flush=True)
    # Actual zero-init model, not the synthetic optimizer clones.
    p=a.out.with_name("unified-init.bin");data=serialize(ref.params,ref.gate,SparseAdam(),corpus);p.write_bytes(data)
    p2,g2,o2=deserialize(data,corpus);assert data==serialize(p2,g2,o2,corpus) and o2.step==0
    out["checkpoint_sha256"]=hashlib.sha256(data).hexdigest();out["checkpoint_bytes"]=len(data)
    out["graph"]=dict(ref.diag);out["signed_checkpoint_sha256"]=ref.params.save(a.out.with_name("signed-init.bin"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
