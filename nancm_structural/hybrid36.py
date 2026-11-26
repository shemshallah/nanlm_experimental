"""Stage 8 v36: document-local long-range attractor memory, conditioned by the
frozen inherited recurrent field. Exact normalized sparse word/prefix mixture."""
import hybrid35 as R5, hybrid34 as Z4, hybrid33 as X, hybrid32 as W, hybrid31 as I, hybrid30 as M, hybrid29 as Q, hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, temporal_phasor as T, subword_geometry as Y, attractor_memory as AM
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib

def load35(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v34_sha
        assert int(ck["step"])==7296
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v35_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    meta=json.loads(path.with_suffix(".token.json").read_text())
    assert meta["split"]==list(Y.SPLIT) and meta["vocabulary"]==len(Y.TOKENS) and meta["parent_sha256"]==f.v34_sha
    f.trust=tuple(meta["trust"]);assert len(f.trust)==2 and all(0<=v<=1 for v in f.trust)
    return f









class Sparse:
    def __init__(self,words,prob):
        self.d=C.defaultdict(float)
        for w,p in zip(words,prob):
            if p>0:
                assert w and all(c in B.A for c in w)
                self.d[w]+=float(p)
        total=math.fsum(self.d.values());assert total>0
        self.d={w:v/total for w,v in self.d.items()}
    def complete(self,w):return self.d.get(w,0.)
    def mass(self,p):return math.fsum(v for w,v in self.d.items() if w.startswith(p))

class MemoryReplay(M.TemporalReplay):
    def __init__(self,field):
        super().__init__(field);self.attractor=field.bank()
    def observe(self,w):
        self.attractor.observe(w);super().observe(w)
    def snapshot(self):
        s=super().snapshot();s["attractor"]=self.attractor.snapshot();return s
    def restore(self,state):
        super().restore(state);self.attractor.restore(state["attractor"])

class MemoryField(R5.Resonance):
    mem_keys=("mem_support","mem_oppose","mem_gate","mem_query")
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,available);load35(self,p35)
        self.parent=load35(R5.Resonance(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,available),p35)
        self.inherited_keys=tuple(self.p);self.memory_scale=0.
        for k,shape in zip(self.mem_keys,((AM.DIM,),(AM.DIM,),(AM.DIM+129,),(128,2*AM.DIM))):
            a=np.zeros(shape,np.float32)
            if k=="mem_gate":a[-1]=-2.
            self.p[k]=a;self.m[k]=np.zeros_like(a);self.v[k]=np.zeros_like(a);self.steps[k]=0
    def bank(self):return AM.Attractor(self.p["word_code"],self.sem.ids,self.sem.classof)
    def session(self):return MemoryReplay(self)
    def field_law(self,state,base):
        # Same arithmetic as the inherited lexical law; expose its hidden state.
        self.stream_override=state.trace
        try:q,hidden,_,_=self.tape(np.asarray([self.features(state.h)+self.events(state.h)+self.ids(state.h)],np.uint32))
        finally:self.stream_override=None
        _,_,mix,e=self.gates(hidden);q=q[0].astype(np.float64);q/=q.sum()
        q=(1-float(mix[0]))*q+float(mix[0])*self.sem.classlaw(state.h);q/=q.sum()
        old=self.sem.law(state.h,base,True,q);z=(hidden@self.proj)[0]
        score=(self.lex@z).astype(np.float64)
        if self.lex_mode!='output_off':score+=(self.p['word_code'][1:]@(hidden@self.p['word_query'])[0]).astype(np.float64)
        d=old.d/.94;classes=self.sem.cls[1:];maxima=np.full(256,-np.inf);np.maximum.at(maxima,classes,score)
        tilted=d*np.exp(score-maxima[classes]);norm=np.bincount(classes,weights=tilted,minlength=256)
        d=tilted*q[classes]/np.maximum(norm[classes],1e-300);assert abs(d.sum()-1)<1e-9
        return F.AdaptiveDense(self.sem,d,base,float(e[0])),hidden[0]
    def predict(self,f,n,v,hidden,grad=False):
        p={k:torch.tensor(self.p[k],requires_grad=grad) for k in self.mem_keys}
        prob,g=AM.law(torch.as_tensor(f),torch.as_tensor(n),torch.as_tensor(v),p["mem_support"],p["mem_oppose"],p["mem_gate"],torch.as_tensor(hidden),p["mem_query"])
        return prob,g,p
    def combine(self,base,hidden,query,mode="full",scale=None):
        f,n,v,words=query
        if not v.any():return base,0.
        if scale is None:scale=self.memory_scale
        if scale==0:return base,0.
        p,g,_=self.predict(f[None],n[None],v[None],hidden[None])
        gate=float(g[0])*scale
        weights=p[0].numpy()
        if mode=="uniform":weights=v.astype(np.float64)/v.sum()
        if mode=="exact":
            weights=v*(f[:,3]>.0);total=weights.sum()
            if not total:return base,0.
            weights=weights/total
        return H.Mixture((base,Sparse(words,weights)),(1-gate,gate)),gate
    def collect(self,ref,small,large,cal,docs,count,seed):
        rng=np.random.default_rng(seed);rows=[]
        for doc in docs:
            if len(doc)<160:continue
            pos=set(rng.choice(np.arange(128,min(len(doc),1200)),min(count,min(len(doc),1200)-128),replace=False).tolist())
            state=self.session()
            for i,w in enumerate(doc[:max(pos)+1]):
                if i in pos:
                    orig,_=V.branches(ref,small,large,state,cal);base,h=self.field_law(state,orig[1].law)
                    f,n,v,words=state.attractor.query()
                    rows.append((f,n,v,h,max(base.complete(w),1e-30),[k==w for k in words]))
                state.observe(w)
        assert rows
        return tuple(np.asarray(a,dtype=dt) for a,dt in zip(zip(*rows),(np.float32,np.float32,bool,np.float32,np.float32,np.float32)))
    def objective(self,ds,grad=False,scale=1.):
        f,n,v,h,base,hit=ds;p,g,params=self.predict(f,n,v,h,grad)
        g=g*scale;actual=(p*torch.as_tensor(hit)).sum(-1)
        prob=(1-g)*torch.as_tensor(base)+g*actual
        loss=-prob.clamp_min(1e-30).log().mean()
        if not grad:return float(loss.detach())
        loss.backward();return float(loss.detach()),{k:a.grad.numpy().copy() for k,a in params.items()}
    def gradient_check(self,ds):
        a=tuple(x[:32] for x in ds);_,g=self.objective(a,True);errors={};norms={}
        for k in self.mem_keys:
            p=self.p[k];ix=np.unravel_index(np.abs(g[k]).argmax(),p.shape);z=float(p[ix]);eps=.003
            p[ix]=z+eps;hi=self.objective(a);p[ix]=z-eps;lo=self.objective(a);p[ix]=z
            errors[k]=abs((hi-lo)/(2*eps)-float(g[k][ix]));norms[k]=float(np.linalg.norm(g[k]))
            assert errors[k]<.002 and norms[k]>0,(k,errors[k],norms[k])
        return {"errors":errors,"norms":norms}
    def train_memory(self,ds,steps=384):
        trace=[]
        for t in range(steps):
            ix=self.rng.integers(len(ds[-1]),size=32);loss,g=self.objective(tuple(a[ix] for a in ds),True)
            self.step+=1
            for k in self.mem_keys:
                self.steps[k]+=1;j=self.steps[k];a=g[k]
                self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
                self.p[k]-=.005*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
                np.clip(self.p[k],-5,5,out=self.p[k]);assert np.isfinite(self.p[k]).all()
            if t%32==0 or t==steps-1:trace.append([t+1,loss])
        return trace
    def calibrate_memory(self,ds):
        grid=(0.,.125,.25,.5,.75,1.)
        scores={str(a):self.objective(ds,scale=a) for a in grid};self.memory_scale=min(grid,key=lambda a:scores[str(a)])
        return {"scale":self.memory_scale,"grid_nll":scores,"candidate_recall":float(ds[-1].any(1).mean()),"targets":len(ds[-1])}
    def save(self,path):
        parent=self.v25_sha
        try:self.v25_sha=self.v35_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent
        path.with_suffix(".token.json").write_text(json.dumps({"split":list(Y.SPLIT),"trust":list(self.trust),"vocabulary":len(Y.TOKENS),"parent_sha256":self.v35_sha},indent=2))
        path.with_suffix(".memory.json").write_text(json.dumps({"scale":self.memory_scale,"capacity":AM.CAP,"candidates":AM.K,"parent_sha256":self.v35_sha},indent=2))

def evaluate(field,ref,small,large,cal,docs,long=False):
    m=C.defaultdict(C.Counter);cand=C.Counter();t=time.monotonic()
    for j,doc in enumerate(docs):
        if not doc:continue
        other=docs[(j+1)%len(docs)];state=field.session();wrong=field.bank()
        positions=set(range(16,min(88,len(doc)))) if not long else set(range(128,min(len(doc),1200),17))
        if not positions:continue
        for i,w in enumerate(doc[:max(positions)+1]):
            if i in positions:
                orig,_=V.branches(ref,small,large,state,cal);parent,h=field.field_law(state,orig[1].law)
                if i==min(positions):
                    # parent parity, session replay, empty memory fallback.
                    old=field.parent.law_stream(state,orig[1].law)
                    for word in ("a","the","garden",w):assert parent.complete(word)==old.complete(word)
                    for prefix in ("","a","th"):assert parent.mass(prefix)==old.mass(prefix)
                    copy=field.session();copy.restore(state.snapshot());assert copy.attractor.query()[3]==state.attractor.query()[3]
                    assert field.combine(parent,h,field.bank().query())[0] is parent
                real=state.attractor.query();bad=wrong.query(state.attractor.h)
                cand["n"]+=1;cand["correct48"]+=w in real[3];cand["wrong48"]+=w in bad[3]
                for horizon in (32,128,512):
                    q=state.attractor.query(min_age=horizon);cand["eligible"+str(horizon)]+=bool(q[2].any());cand["hit"+str(horizon)]+=w in q[3]
                for mode,law,g in (("parent",parent,0.),("full",*field.combine(parent,h,real)),("wrong_memory",*field.combine(parent,h,bad)),("uniform",*field.combine(parent,h,real,"uniform")),("exact",*field.combine(parent,h,real,"exact")),("forced_memory",*field.combine(parent,h,real,scale=1.))):
                    l=H.Mixture((law,),(1.,));rank,cert,_=l.rank(budget=2000);z=m[mode]
                    z["n"]+=1;z["t1"]+=w==rank[0];z["t5"]+=w in rank;z["cert"]+=cert;z["gate"]+=g;z["bits"]-=math.log2(max(l.complete(w),1e-300));z["symbols"]+=len(w)+1
                    prefix="";p=1.
                    for c in w+B.EOS:
                        p*=l.next(prefix)[c]
                        if c!=B.EOS:prefix+=c
                    assert math.isclose(p,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
            state.observe(w)
            if i<len(other):wrong.observe(other[i])
    return {"modes":{k:dict(v,word_top1=v["t1"]/v["n"],word_top5=v["t5"]/v["n"],spelling_bps=v["bits"]/v["symbols"],mean_gate=v["gate"]/v["n"]) for k,v in m.items()},"candidate":dict(cand),"seconds":time.monotonic()-t}

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];calbooks=books[96:108];extra=books[108:396];lex=set()
    assert len(tr)==96 and len(calbooks)==12 and len(extra)==288
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];expanded=docs+[B.load(p,6000) for _,p in extra]
    ref=U.Unified(docs,lex,N.Native(a.lib));ref.head.train(ref.head.data([d[:512] for d in docs]))
    cdocs=[B.load(p,200) for _,p in calbooks];cal=U.fit(ref,cdocs);small=V.Memory(docs);large=V.Memory(expanded);sem=Z.Semantic(expanded,large)
    data=P.Field(sem).data(expanded,cap=768)
    field=MemoryField(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,len(data[1]));field.bind(ref,small,large,cal)
    selected=extra[::12];train=field.collect(ref,small,large,cal,[B.load(p,1200) for _,p in selected],24,3601)
    valid=field.collect(ref,small,large,cal,[B.load(p,1200) for _,p in calbooks],24,3602)
    checks=field.gradient_check(train);before=field.objective(valid);trace=field.train_memory(train);gate=field.calibrate_memory(valid)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) and field.steps[k]==field.parent.steps[k] for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    forbidden={i for i,p in tr+extra+calbooks};assert not forbidden&{i for i,p in picked}
    story=[B.load(p,1200) for _,p in picked]
    result={"parent_sha256":field.v35_sha,"gradient_check":checks,"training_trace":trace,"cal_before":before,"calibration":gate,"train_targets":len(train[-1]),"train_ids":[i for i,p in selected],"calibration_ids":[i for i,p in calbooks],"dev_ids":[i for i,p in picked],"short":evaluate(field,ref,small,large,cal,story),"long":evaluate(field,ref,small,large,cal,story,True),"group_steps":field.steps,"inherited_unchanged":True,"memory_capacity":AM.CAP}
    field.save(a.out.with_name("attractor-field.npz"));a.out.write_text(json.dumps(result,indent=2));print("FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
