"""v25: direct whole-word learning through the resumed recurrent signed field.
Exact hierarchical vocabulary softmax; zero-initialized lexical residual.
Bounded diagnostic continuation from actual v24 artifact, no full-model training."""
import hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N
import numpy as np
import argparse, collections as C, json, math, pathlib, time, hashlib

def load24(field,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(field.sem.words))
        assert np.array_equal(ck["classes"],field.sem.cls)
        assert np.array_equal(ck["upper_pairs"],field.upper_pairs)
        assert str(ck["parent_sha"].item())==field.parent_sha
        assert int(ck["step"])==2048
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert steps=={"plus":1536,"minus":1536,"read":1536,"bias":1536,"rec_plus":1024,"rec_minus":1024}
        for k in field.p:
            for prefix,group in (("",field.p),("m_",field.m),("v_",field.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all()
                group[k]=a.copy()
        field.steps=steps;field.step=int(ck["step"])
        field.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    field.v24_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return field

class Lexical(R.Recurrent):
    def __init__(self,sem,p23,p24,available):
        super().__init__(sem,p23,available);load24(self,p24)
        self.parent=load24(R.Recurrent(sem,p23,available),p24)
        self.groups=[np.flatnonzero(sem.cls[1:]==c) for c in range(256)]
        self.classlocal=np.zeros(sem.V,np.int32)
        for ix in self.groups:self.classlocal[ix]=np.arange(len(ix))
        r=np.random.default_rng(2501)
        self.proj=np.linalg.qr(r.normal(size=(128,32)))[0].astype(np.float32)
        self.lex=np.zeros((sem.V,32),np.float32);self.lm=np.zeros_like(self.lex);self.lv=np.zeros_like(self.lex)
        self.lt=np.zeros(256,np.int64);self.logemit=np.log(sem.emission).astype(np.float32)
    def worddata(self,docs,cap=None):
        x=[];y=[];oov=0
        for doc in docs:
            for i,w in enumerate(doc if cap is None else doc[:cap]):
                j=self.sem.ids.get(w)
                if j is None:oov+=1;continue
                x.append(self.features(doc[max(0,i-6):i]));y.append(j-1)
        return (np.asarray(x,np.uint8),np.asarray(y,np.int64)),oov
    def direct(self,x,y,gradient=False):
        cls=self.sem.cls[y+1].astype(np.int64);q,h,v,tape=self.tape(x);n=len(y)
        lossclass=float(-np.log(np.maximum(q[np.arange(n),cls],1e-30)).mean())
        z=h@self.proj;dz=np.zeros_like(z);lexgrad={};lossemit=0.
        for c in np.unique(cls):
            rows=np.flatnonzero(cls==c);ix=self.groups[c];target=self.classlocal[y[rows]]
            logits=z[rows]@self.lex[ix].T+self.logemit[ix]
            logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
            lossemit-=float(np.log(np.maximum(p[np.arange(len(rows)),target],1e-30)).sum())/n
            if gradient:
                p[np.arange(len(rows)),target]-=1;p/=n
                lexgrad[int(c)]=p.T@z[rows];dz[rows]=p@self.lex[ix]
        if not gradient:return lossclass,lossemit
        q[np.arange(n),cls]-=1;q/=n
        g={"read":h.T@q,"bias":q.sum(axis=0)}
        g.update(self.back(x,v,tape,q@self.p["read"].T+dz@self.proj.T))
        return lossclass+lossemit,g,lexgrad
    def wordloss(self,data):
        x,y=data;tot=np.zeros(2)
        for i in range(0,len(y),128):
            a=x[i:i+128];b=y[i:i+128];tot+=len(b)*np.asarray(self.direct(a,b))
        return {"class_nll":float(tot[0]/len(y)),"emission_nll":float(tot[1]/len(y)),"known_word_nll":float(tot.sum()/len(y)),"known_targets":len(y)}
    def check_word(self,data):
        x,y=data;x=x[:24];y=y[:24]
        loss,g,lg=self.direct(x,y,True);c=max(lg,key=lambda c:np.max(np.abs(lg[c])))
        i,j=np.unravel_index(np.abs(lg[c]).argmax(),lg[c].shape);row=self.groups[c][i]
        analytic=float(lg[c][i,j]);a=float(self.lex[row,j]);eps=.003
        self.lex[row,j]=a+eps;hi=sum(self.direct(x,y))
        self.lex[row,j]=a-eps;lo=sum(self.direct(x,y));self.lex[row,j]=a
        numeric=(hi-lo)/(2*eps);assert abs(numeric-analytic)<.002
        # Check full class+emission derivative back through the recurrence
        # with nonzero lexical readout, not just the zero-residual case.
        self.lex[row,j]=.1
        _,g,_=self.direct(x,y,True);errors={}
        for k in self.p:
            idx=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape);a=float(self.p[k][idx])
            self.p[k][idx]=a+eps;hi=sum(self.direct(x,y))
            self.p[k][idx]=a-eps;lo=sum(self.direct(x,y));self.p[k][idx]=a
            err=abs((hi-lo)/(2*eps)-float(g[k][idx]));assert err<.003;errors[k]=err
        self.lex[row,j]=0.
        return {"lexical_numeric":numeric,"lexical_analytic":analytic,"lexical_abs_error":abs(numeric-analytic),"field_group_abs_errors":errors}
    def train_word(self,data,steps,lex_only):
        x,y=data;trace=[];start=time.monotonic()
        for t in range(steps):
            ix=self.rng.integers(len(y),size=64);loss,g,lg=self.direct(x[ix],y[ix],True);self.step+=1
            for c,grad in lg.items():
                rows=self.groups[c];self.lt[c]+=1;k=int(self.lt[c])
                self.lm[rows]=.9*self.lm[rows]+.1*grad
                self.lv[rows]=.999*self.lv[rows]+.001*grad*grad
                self.lex[rows]-=.005*(self.lm[rows]/(1-.9**k))/(np.sqrt(self.lv[rows]/(1-.999**k))+1e-8)
                assert np.isfinite(self.lex[rows]).all()
            if not lex_only:
                for k in self.p:
                    self.steps[k]+=1;j=self.steps[k]
                    self.m[k]=.9*self.m[k]+.1*g[k];self.v[k]=.999*self.v[k]+.001*g[k]*g[k]
                    self.p[k]-=.001*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
                    if k in ("plus","minus","rec_plus","rec_minus"):np.clip(self.p[k],0.,3.,out=self.p[k])
                    assert np.isfinite(self.p[k]).all()
            if t%64==0 or t==steps-1:trace.append([t+1,loss])
        return {"steps":steps,"sampled_targets":steps*64,"lex_only":lex_only,"trace":trace,"seconds":time.monotonic()-start}
    def law(self,h,base):
        q,hidden,_,_=self.tape(np.asarray([self.features(h)],np.uint8))
        q=q[0].astype(np.float64);q/=q.sum()
        old=self.sem.law(h,base,True,q)
        if not np.any(self.lex):return old
        # Tilt only within each class. Preserve the calibrated class marginal
        # and retain the original signed unknown-word escape.
        z=(hidden@self.proj)[0];score=(self.lex@z).astype(np.float64)
        d=old.d/.94;classes=self.sem.cls[1:]
        maxima=np.full(256,-np.inf);np.maximum.at(maxima,classes,score)
        tilted=d*np.exp(score-maxima[classes])
        norm=np.bincount(classes,weights=tilted,minlength=256)
        d=tilted*q[classes]/np.maximum(norm[classes],1e-300)
        assert np.max(np.abs(np.bincount(classes,weights=d,minlength=256)-q))<1e-9
        assert abs(d.sum()-1)<1e-9
        return Z.Dense(self.sem,d,base)
    def save(self,path):
        pairs=np.array([[B.pair(i,128+j) for j in range(128)] for i in range(128)],np.uint16)
        np.savez_compressed(path,step=np.array(self.step),pairs=pairs,upper_pairs=self.upper_pairs,words=np.array(self.sem.words),
          classes=self.sem.cls,centers=self.sem.centers,parent_sha=np.array(self.v24_sha),
          rng_json=np.array(json.dumps(self.rng.bit_generator.state)),group_steps_json=np.array(json.dumps(self.steps)),
          lexical=self.lex,lex_m=self.lm,lex_v=self.lv,lex_steps=self.lt,projection=self.proj,
          **self.p,**{"m_"+k:v for k,v in self.m.items()},**{"v_"+k:v for k,v in self.v.items()})

def branches(ref,small,large,sem,field,state,cal,oldprior):
    orig,key=V.branches(ref,small,large,state,cal)
    old=H.Mixture(orig,oldprior[1].get(key,oldprior[0]));base=orig[1].law
    count=sem.law(state.h,base,True)
    parent=sem.law(state.h,base,True,field.parent.probs(state.h))
    return (old,count,parent,field.law(state.h,base)),key

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
                modes={"v24_field":(0,0,1,0),"v25_field":(0,0,0,1),"v24_fixed":(.65,.15,.20,0),"v25_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
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
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--recurrent",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
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
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Lexical(sem,a.parent,a.recurrent,len(data[1]))
    # Preserve parent law at zero lexical residual, including prefix masses.
    s=H.Replay()
    for w in cdocs[0][:16]:s.observe(w)
    q,k=branches(ref,small,large,sem,field,s,cal,oldprior)
    for w in ("a","the","unseenword","garden"):
        assert q[2].complete(w)==q[3].complete(w)
    for prefix in ("","t","th","ga"):assert q[2].mass(prefix)==q[3].mass(prefix)
    wd,train_oov=field.worddata(expanded,cap=768);vd,valid_oov=field.worddata(cdocs)
    gradcheck=field.check_word(wd)
    pre=field.wordloss(vd);phase1=field.train_word(wd,512,True);mid=field.wordloss(vd)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) for k in field.p)
    phase2=field.train_word(wd,512,False);post=field.wordloss(vd)
    train={"lex_only":phase1,"joint":phase2,"middle":mid,"train_oov":train_oov,"calibration_oov":valid_oov}
    diagnostics=field.diagnostics(valid);movement={"lexical_l2":float(np.linalg.norm(field.lex)),"classes_updated":int(np.count_nonzero(field.lt))}
    print("DIRECT_WORD",json.dumps({"before":pre,"after":post,"training":train,"movement":movement,"gradient_check":gradcheck}),flush=True)
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"known_word_cal_before":pre,"known_word_cal_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.v24_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
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
    field.save(a.out.with_name("direct-word-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
