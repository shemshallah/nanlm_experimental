"""v24: resume actual v23 field checkpoint and grow recurrent signed upper cloud.
Zero-coupling prediction parity, contractive dynamics, exact unrolled gradients.
Bounded continuation diagnostic, not full system training."""
import hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N
import numpy as np
import argparse, collections as C, json, math, pathlib, time, hashlib

class Recurrent(P.Field):
    passes=4
    alpha=.55
    def __init__(self,sem,path,available):
        super().__init__(sem)
        with np.load(path,allow_pickle=False) as ck:
            assert np.array_equal(ck["words"],np.array(sem.words))
            assert np.array_equal(ck["classes"],sem.cls)
            assert np.allclose(ck["centers"],sem.centers,rtol=1e-5,atol=1e-6)
            assert int(ck["step"])==1024
            pairs=np.array([[B.pair(i,128+j) for j in range(128)] for i in range(128)],np.uint16)
            assert np.array_equal(ck["pairs"],pairs)
            for k in self.p:
                for prefix,group in (("",self.p),("m_",self.m),("v_",self.v)):
                    a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all()
                    group[k]=a.copy()
            self.step=int(ck["step"])
        # v23 did not serialize RNG state: reproduce only the deterministic
        # minibatch-index draws, without recomputing or replacing any weights.
        for _ in range(self.step):self.rng.integers(available,size=128)
        self.parent_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        self.i,self.j=np.triu_indices(128,1)
        self.upper_pairs=np.array([B.pair(128+int(i),128+int(j)) for i,j in zip(self.i,self.j)],np.uint16)
        assert len(set(map(int,self.upper_pairs)))==8128
        self.old_keys=tuple(self.p);self.steps={k:self.step for k in self.p}
        for k in ("rec_plus","rec_minus"):
            self.p[k]=np.zeros(len(self.i),np.float32);self.m[k]=np.zeros_like(self.p[k]);self.v[k]=np.zeros_like(self.p[k]);self.steps[k]=0
        self.frozen=P.Field(sem)
        for k in self.old_keys:
            self.frozen.p[k]=self.p[k].copy();self.frozen.m[k]=self.m[k].copy();self.frozen.v[k]=self.v[k].copy()
        self.frozen.step=self.step
    def coupling(self):
        a=np.tanh(self.p["rec_plus"]-self.p["rec_minus"])
        A=np.zeros((128,128),np.float32);A[self.i,self.j]=a;A[self.j,self.i]=a
        abs_smooth=np.sqrt(A*A+1e-6)
        den=1+abs_smooth.sum(axis=1,keepdims=True)
        W=self.alpha*A/den
        assert np.max(np.abs(W).sum(axis=1))<self.alpha
        return W,A,den,abs_smooth,a
    def tape(self,x):
        p=self.p
        v=np.tanh(p["plus"][self.rails,x]-p["minus"][self.rails,x])
        z=v.sum(axis=1)/math.sqrt(8)
        W,A,den,ab,a=self.coupling()
        states=[np.tanh(z)]
        for _ in range(self.passes):states.append(np.tanh(z+states[-1]@W.T))
        h=states[-1]
        logits=h@p["read"]+p["bias"];logits[:,~self.sem.class_active]=-1e9
        logits-=logits.max(axis=1,keepdims=True);q=np.exp(logits);q/=q.sum(axis=1,keepdims=True)
        return q,h,v,(states,W,A,den,ab,a)
    def forward(self,x):
        q,h,v,_=self.tape(x);return q,h,v
    def back(self,x,v,tape,dh):
        states,W,A,den,ab,a=tape
        dz=np.zeros_like(dh);gW=np.zeros_like(W)
        for k in range(self.passes,0,-1):
            d=dh*(1-states[k]*states[k]);dz+=d
            gW+=d.T@states[k-1];dh=d@W
        dz+=dh*(1-states[0]*states[0])
        dA=self.alpha*gW/den-self.alpha*(gW*A).sum(axis=1,keepdims=True)/den**2*A/ab
        g=(dA[self.i,self.j]+dA[self.j,self.i])*(1-a*a)
        dv=dz[:,None,:]*(1-v*v)/math.sqrt(8)
        gp=np.zeros_like(self.p["plus"])
        for j in range(8):np.add.at(gp[j],x[:,j],dv[:,j])
        return {"plus":gp,"minus":-gp,"rec_plus":g,"rec_minus":-g}
    def gradients(self,x,y):
        q,h,v,tape=self.tape(x);n=len(y)
        loss=float(-np.log(np.maximum(q[np.arange(n),y],1e-30)).mean())
        q[np.arange(n),y]-=1;q/=n
        g={"read":h.T@q,"bias":q.sum(axis=0)}
        g.update(self.back(x,v,tape,q@self.p["read"].T))
        return loss,g
    def check_extension(self,data):
        x,y=data;x=x[:24];y=y[:24]
        q=self.forward(x)[0];old=self.frozen.forward(x)[0]
        assert np.array_equal(q,old)
        # Test nonzero recurrence with a reversible perturbation, not merely
        # the trivial zero matrix. Check all six groups.
        r=np.random.default_rng(2401)
        save={k:self.p[k].copy() for k in ("rec_plus","rec_minus")}
        self.p["rec_plus"][:]=r.uniform(.005,.04,len(self.i))
        self.p["rec_minus"][:]=r.uniform(.005,.04,len(self.i))
        errors=self.check((x,y))
        for k,a in save.items():self.p[k][:]=a
        return {"zero_coupling_exact":True,"gradient_errors":errors}
    def train_phase(self,data,steps,rec_only):
        x,y=data;trace=[];start=time.monotonic()
        keys=("rec_plus","rec_minus") if rec_only else tuple(self.p)
        for j in range(steps):
            ix=self.rng.integers(len(y),size=128);loss,g=self.gradients(x[ix],y[ix]);self.step+=1
            for k in keys:
                self.steps[k]+=1;t=self.steps[k]
                self.m[k]=.9*self.m[k]+.1*g[k];self.v[k]=.999*self.v[k]+.001*g[k]*g[k]
                lr=.003 if k.startswith("rec_") else .0015
                self.p[k]-=lr*(self.m[k]/(1-.9**t))/(np.sqrt(self.v[k]/(1-.999**t))+1e-8)
                if k in ("plus","minus","rec_plus","rec_minus"):np.clip(self.p[k],0.,3.,out=self.p[k])
                assert np.isfinite(self.p[k]).all()
            if j%128==0 or j==steps-1:trace.append([j+1,loss])
        return {"steps":steps,"sampled_targets":steps*128,"rec_only":rec_only,"trace":trace,"seconds":time.monotonic()-start}
    def diagnostics(self,data):
        x=data[0][:128];q,h,v,tape=self.tape(x);states,W,A,den,ab,a=tape
        z=v.sum(axis=1)/math.sqrt(8)
        residual=float(np.max(np.abs(np.tanh(z+h@W.T)-h)))
        return {"row_norm_max":float(np.max(np.abs(W).sum(axis=1))),"four_pass_residual_inf":residual,
                "rec_plus_l2":float(np.linalg.norm(self.p["rec_plus"])),"rec_minus_l2":float(np.linalg.norm(self.p["rec_minus"])),
                "steps":self.steps}
    def save(self,path):
        pairs=np.array([[B.pair(i,128+j) for j in range(128)] for i in range(128)],np.uint16)
        np.savez_compressed(path,step=np.array(self.step),pairs=pairs,upper_pairs=self.upper_pairs,words=np.array(self.sem.words),
          classes=self.sem.cls,centers=self.sem.centers,parent_sha=np.array(self.parent_sha),
          rng_json=np.array(json.dumps(self.rng.bit_generator.state)),group_steps_json=np.array(json.dumps(self.steps)),
          **self.p,**{"m_"+k:v for k,v in self.m.items()},**{"v_"+k:v for k,v in self.v.items()})

def branches(ref,small,large,sem,field,state,cal,oldprior):
    orig,key=V.branches(ref,small,large,state,cal)
    base=orig[1].law
    old=H.Mixture(orig,oldprior[1].get(key,oldprior[0]))
    count=sem.law(state.h,base,True)
    parent=sem.law(state.h,base,True,field.frozen.probs(state.h))
    recurrent=sem.law(state.h,base,True,field.probs(state.h))
    return (old,count,parent,recurrent),key
def fit(ref,small,large,sem,field,cal,oldprior,docs):
    rows=C.defaultdict(list)
    for doc in docs:
        s=H.Replay()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                q,k=branches(ref,small,large,sem,field,s,cal,oldprior);p=[l.complete(w) for l in q];rows[("all",)].append(p);rows[k].append(p)
            s.observe(w)
    q=U.em(rows[("all",)],[.62,.25,.065,.065],pseudocount=24)
    return q,{k:U.em(v,q,pseudocount=64) for k,v in rows.items() if k!=("all",)}
def evaluate(ref,small,large,sem,field,cal,oldprior,prior,docs,limit):
    ms=C.defaultdict(C.Counter);oracle=C.Counter();t=time.monotonic()
    for doc in docs:
        s=H.Replay()
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                q,k=branches(ref,small,large,sem,field,s,cal,oldprior)
                modes={"v23_field":(0,0,1,0),"v24_field":(0,0,0,1),"v23_fixed":(.65,.15,.20,0),"v24_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
                for name,ws in modes.items():
                    l=H.Mixture(q,ws);rank,cert,n=l.rank(budget=2000)
                    m=ms[name];m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.
                    for c in w+B.EOS:
                        d=l.next(p);prod*=d[c]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                a,ac,_=H.Mixture((q[2],),(1.,)).rank(32,budget=5000)
                b,bc,_=H.Mixture((q[3],),(1.,)).rank(32,budget=5000)
                oracle["n"]+=1;oracle["parent32"]+=w in a;oracle["recurrent32"]+=w in b;oracle["new32"]+=w in b and w not in a;oracle["cert"]+=ac+bc
            s.observe(w)
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"oracle":dict(oracle),"seconds":time.monotonic()-t}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
    groups={k:B.bookfiles(r/k) for k in ("stories","sentences","paragraphs")};split=lambda i:int(B.digest("NaNCM-structural-v1:"+i)[:8],16)%10
    books=[(i,p) for i,p in groups["stories"] if split(i)<7];tr=books[:96];calbooks=books[96:108];extra=books[108:396];lex=set()
    assert len(tr)==96 and len(calbooks)==12 and len(extra)==288
    for p in (r/"lex").rglob("*.txt"):
        if "google" in p.name.lower() and not p.name.lower().startswith(("license","readme")):lex.update(B.words(p.read_text(errors="replace")))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];expanded=docs+[B.load(p,6000) for _,p in extra]
    ref=U.Unified(docs,lex,N.Native(a.lib));before=ref.params.payload()
    pilot=ref.head.train(ref.head.data([d[:512] for d in docs]))
    cdocs=[B.load(p,200) for _,p in calbooks];cal=U.fit(ref,cdocs)
    small=V.Memory(docs);large=V.Memory(expanded);oldprior=V.fit(ref,small,large,cal,cdocs)
    sem=Z.Semantic(expanded,large);template=P.Field(sem)
    # Bounded pilot across every expanded reference book, not full training.
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Recurrent(sem,a.parent,len(data[1]))
    gradcheck=field.check_extension(data);initial={k:v.copy() for k,v in field.p.items()}
    pre=field.loss(valid);assert abs(pre-4.370675710042318)<1e-4
    phase1=field.train_phase(data,512,True);mid=field.loss(valid)
    assert all(np.array_equal(field.p[k],field.frozen.p[k]) and np.array_equal(field.m[k],field.frozen.m[k]) and np.array_equal(field.v[k],field.frozen.v[k]) for k in field.old_keys)
    phase2=field.train_phase(data,512,False);post=field.loss(valid);train={"rec_only":phase1,"joint":phase2,"middle_nll":mid};diagnostics=field.diagnostics(valid)
    movement={k:float(np.linalg.norm(v-initial[k])) for k,v in field.p.items()}
    print("FIELD",json.dumps({"before":pre,"after":post,"movement":movement,"train":train,"gradient_check":gradcheck}),flush=True)
    field.save(a.out.with_name("recurrent-class-field.npz"))
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"class_cal_nll_before":pre,"class_cal_nll_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.parent_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
    out["stories"]=evaluate(ref,small,large,sem,field,cal,oldprior,prior,story,72);print("STORIES",json.dumps(out["stories"]),flush=True)
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
        assert len(chosen)==6;ids[k]=[i for i,w in chosen]
        out["fresh"][k]=evaluate(ref,small,large,sem,field,cal,oldprior,prior,[w for i,w in chosen],32)
        print(k,json.dumps(out["fresh"][k]),flush=True)
    assert before==ref.params.payload() and frozen==json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    out["signed_unchanged"]=True;out["graph"]=dict(ref.diag);out["fresh_ids"]=ids
    field.save(a.out.with_name("recurrent-class-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
