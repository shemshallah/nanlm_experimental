"""v19 unified six-route word and native byte prediction, segment Bayesian gate.
No inherited signed updates; deterministic bounded v17 head replay."""
import hybrid18 as U, hybrid16 as V, hybrid15 as H
import base as B, foundation6 as N
import argparse, collections as C, math, pathlib, json, time

class Gate:
    def __init__(self):self.rows=C.defaultdict(list);self.local={};self.global_p=.5
    def add(self,k,logs):
        m=max(logs);p=[math.exp(x-m) for x in logs]
        self.rows[("all",)].append(p);self.rows[k].append(p)
    def fit(self):
        q=U.em(self.rows[("all",)],[.5,.5],pseudocount=24)
        self.global_p=q[1]
        self.local={k:U.em(v,q,pseudocount=64)[1] for k,v in self.rows.items() if k!=("all",)}
    def prior(self,k):return max(.05,min(.95,self.local.get(k,self.global_p)))
    def export(self):return {"global":self.global_p,"local":{str(k):{"n":len(self.rows[k]),"word_prior":v} for k,v in self.local.items()}}

class Joint(V.Stream):
    def __init__(self,ref,raw,cal,mode,gate=None):
        super().__init__(ref,raw,cal,mode)
        self.logweights=[0.]*6;self.gate=gate;self.logs=[0.,0.];self.odds=0.;self.segments=0
        self.start_prior=.5;self.key=None;self.record=None
    def begin(self):
        branches,key,_=self.ref.six(self.state);self.branches=branches
        weights=U.modes(branches,self.cal,key,self.logweights)
        self.lat=H.Mixture(branches,weights["six_causal" if self.mode=="six_causal" else "six_em"])
        self.key=(key[0],int(bool(self.tail and self.tail[-1] in (9,10,13,32))))
        self.start_prior=self.gate.prior(self.key) if self.gate else .5
        if self.mode=="six_causal":
            p=self.start_prior
            self.start_prior=1/(1+math.exp(-(math.log(p/(1-p))+self.odds)))
        self.posterior=self.start_prior
        self.logs=[0.,0.]
    def observe(self,b):
        p=self.probs();word=self.cached[1][b];raw=self.raw.probs(self.tail)[b]
        self.logs[0]+=math.log(max(raw,1e-300));self.logs[1]+=math.log(max(word,1e-300))
        self.posterior=self.posterior*word/p[b]
        self.record=None
        if b in B.LETTERS:self.prefix+=chr(b).lower()
        else:
            # Check exact posterior for the entire segment, including delimiter.
            a=math.log(self.start_prior)+self.logs[1]
            z=math.log1p(-self.start_prior)+self.logs[0];mx=max(a,z)
            posterior=math.exp(a-mx)/(math.exp(a-mx)+math.exp(z-mx))
            assert abs(posterior-self.posterior)<1e-8
            self.record=(self.key,tuple(self.logs))
            self.odds=max(-2.,min(2.,.94*self.odds+.15*max(-8.,min(8.,self.logs[1]-self.logs[0]))))
            if self.prefix:
                logs=[.94*v+.35*math.log(max(l.complete(self.prefix),1e-300)) for v,l in zip(self.logweights,self.branches)]
                mx=max(logs);self.logweights=[max(-15.,v-mx) for v in logs]
                self.state.observe(self.prefix)
            self.prefix="";self.lat=None;self.posterior=.5;self.segments+=1
        self.tail=(self.tail+bytes([b]))[-24:];self.cached=None

def fit_gate(ref,raw,cal,docs):
    gate=Gate()
    for doc in docs:
        s=Joint(ref,raw,cal,"six_em")
        for i,b in enumerate(doc[:512]):
            s.observe(b)
            if i>=64 and s.record:gate.add(*s.record)
    gate.fit();return gate

def evaluate(ref,raw,cal,docs,mode,gate):
    # Reuse v16's exact metric implementation with the new stream interface.
    saved=V.Stream
    try:
        if mode!="v16":
            V.Stream=lambda r,m,c,k:Joint(r,m,c,k,gate)
        return V.evaluate(ref,raw,cal,docs,"adaptive" if mode=="v16" else mode)
    finally:V.Stream=saved

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--cm",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    books=B.bookfiles(r/"stories");split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    train=[(i,p) for i,p in books if split(i)<7];tr=train[:96];calbooks=train[96:108];lex=set()
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(tr)==96 and len(calbooks)==12 and len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];ref=U.Unified(docs,lex,N.Native(a.lib));before=ref.params.payload()
    pilot=ref.head.train(ref.head.data([d[:512] for d in docs]))
    cdocs=[B.load(p,200) for _,p in calbooks]
    oldcal=H.fit(ref,cdocs);cal=U.fit(ref,cdocs)
    raw=V.NativeCM(a.cm,[p.read_bytes()[:16000] for _,p in tr])
    gate=fit_gate(ref,raw,cal,[p.read_bytes()[:512] for _,p in calbooks])
    frozen=gate.export();print("GATE",json.dumps(frozen),flush=True)
    dev=[p.read_bytes()[:768] for i,p in books if split(i) in (7,8)][:5]
    for p in (r/"wiki"/"wiki.valid.raw",r/"domain"/"chemistry.txt"):
        with p.open("rb") as f:dev.append(f.read(768))
    def at(data,k):
        s=Joint(ref,raw,cal,"six_causal",gate)
        for b in data[:k]:s.observe(b)
        return s.probs()
    for k in (16,31,64):assert at(dev[0],k)==at(dev[0][:k]+b"different future",k)
    out={"pilot":pilot,"reference_words":sum(map(len,docs)),"raw_reference_bytes":raw.count,"gate":frozen,"suffix_invariance":True,"modes":{}}
    for name,mode,c,g in (("v16_adaptive","v16",oldcal,None),("six_fixed_half","six_em",cal,None),("six_calibrated_gate","six_em",cal,gate),("six_causal_gate","six_causal",cal,gate)):
        out["modes"][name]=evaluate(ref,raw,c,dev,mode,g);print(name,json.dumps(out["modes"][name]),flush=True)
    assert before==ref.params.payload() and frozen==gate.export()
    out["signed_unchanged"]=True;out["graph"]=dict(ref.diag);raw.close()
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
