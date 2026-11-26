"""v28: optimize the deployed normalized word/prefix law, including OOV.
Train retrieval trust, contextual emission, escape and the entire v27 field
with the same word likelihood. Preserve all prior Adam state."""
import hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N
import numpy as np
import argparse, collections as C, json, math, pathlib, time, hashlib, bisect

def load27(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v26_sha
        assert int(ck["step"])==4608
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v27_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return f

class AdaptiveDense(Z.Dense):
    def __init__(self,sem,d,base,escape):
        self.sem=sem;self.escape=float(escape);self.d=(1-self.escape)*d
        self.cum=np.r_[0.,np.cumsum(self.d)];self.base=H.BasePrefix(base)
        assert 0<self.escape<1 and abs(d.sum()-1)<1e-9
    def complete(self,w):
        i=self.sem.ids.get(w)
        return self.escape*self.base.complete(w)+(float(self.d[i-1]) if i else 0.)
    def mass(self,p):
        a=bisect.bisect_left(self.sem.words,p);b=bisect.bisect_right(self.sem.words,p+"{")
        return self.escape*self.base.mass(p)+max(0.,float(self.cum[b]-self.cum[a]))

class Fusion(S.Symbiotic):
    fusion_keys=("fusion_class","fusion_escape")
    def __init__(self,sem,p23,p24,p25,p26,p27,available):
        super().__init__(sem,p23,p24,p25,p26,available);load27(self,p27)
        self.parent=load27(S.Symbiotic(sem,p23,p24,p25,p26,available),p27)
        self.inherited_keys=tuple(self.p)
        for k in self.fusion_keys:
            self.p[k]=np.zeros(129,np.float32);self.m[k]=np.zeros(129,np.float32);self.v[k]=np.zeros(129,np.float32);self.steps[k]=0
        self.records=[];self.cache={}
    def bind(self,ref,small,large,cal):
        self.runtime=(ref,small,large,cal)
    def worddata(self,docs,cap=None):
        x=[];y=[];oov=0
        # Include unknown words instead of silently removing them.
        for doc in docs:
            for i,w in enumerate(doc if cap is None else doc[:cap]):
                j=self.sem.ids.get(w,0);oov+=j==0
                x.append(self.features(doc[max(0,i-128):i])+[len(self.records)])
                self.records.append((doc,i));y.append(j-1)
        return (np.asarray(x,np.uint32),np.asarray(y,np.int64)),oov
    def evidence(self,h,c):
        ix=self.groups[c];e=np.zeros(len(ix),np.float64)
        for name,schema,weight in V.SCHEMAS:
            if len(h)<max(l for l,r in schema):continue
            vals=[self.sem.mem.roleof(h[-l]) if r else self.sem.ids.get(h[-l],0) for l,r in schema]
            if not all(vals):continue
            row,total,distinct=self.sem.mem.row(name,V.scalar_hash(vals))
            if not total:continue
            for word,n in row.items():
                j=self.sem.ids[word]-1
                if self.sem.cls[j+1]==c:e[self.classlocal[j]]+=weight*max(0,n-.65)/(total+8+distinct)
        s=e.sum();lam=min(.80,s/(s+.10))
        return (1-lam)*self.sem.emission[ix]+lam*e/max(s,1e-300)
    def record(self,i):
        i=int(i)
        if i in self.cache:return self.cache[i]
        doc,pos=self.records[i];w=doc[pos];state=H.Replay()
        for t in doc[:pos]:state.observe(t)
        ref,small,large,cal=self.runtime
        orig,_=V.branches(ref,small,large,state,cal)
        base=float(orig[1].complete(w));assert base>0 and math.isfinite(base)
        j=self.sem.ids.get(w,0)
        if j:
            c=int(self.sem.cls[j]);count=float(self.sem.classlaw(state.h)[c]);prior=self.evidence(state.h,c)
            assert abs(prior.sum()-1)<1e-9 and prior.min()>0
        else:c=-1;count=0.;prior=None
        self.cache[i]=(base,c,count,prior)
        return self.cache[i]
    def gates(self,h):
        z=np.c_[h,np.ones(len(h),np.float32)]
        raw=z@self.p["fusion_class"];mix=np.clip(raw,0,.75)
        e=1/(1+np.exp(-np.clip(z@self.p["fusion_escape"]+math.log(.06/.94),-12,12)))
        return z,raw,mix,e
    def direct(self,x,y,gradient=False):
        q,h,v,tape=self.tape(x[:,:40]);n=len(y)
        z,raw,mix,escape=self.gates(h)
        qg=np.zeros_like(q);dh=np.zeros_like(h);dz=np.zeros((n,32),np.float32)
        gm=np.zeros(n,np.float32);ge=np.zeros(n,np.float32)
        lg={};loss=0.;project=h@self.proj
        for i,(rid,j) in enumerate(zip(x[:,40],y)):
            bp,c,count,prior=self.record(rid);e=float(escape[i]);g=float(mix[i])
            if j>=0:
                ix=self.groups[c];target=self.classlocal[j]
                logits=self.lex[ix]@project[i]+np.log(prior)
                p=np.exp(logits-logits.max());p/=p.sum()
                emission=float(p[target]);qc=float(q[i,c]);cm=(1-g)*qc+g*count
                known=cm*emission;prob=(1-e)*known+e*bp
                if gradient:
                    scale=(1-e)*emission*(1-g)*qc/(n*prob)
                    qg[i]=scale*q[i];qg[i,c]-=scale
                    if 0<=raw[i]<=.75:gm[i]=-(1-e)*emission*(count-qc)/(n*prob)
                    pe=p.copy();pe[target]-=1;pe*=((1-e)*known/(n*prob))
                    pe=pe.astype(np.float32);dz[i]=pe@self.lex[ix]
                    if c not in lg:lg[c]=np.zeros_like(self.lex[ix])
                    lg[c]+=pe[:,None]*project[i][None,:]
            else:
                known=0.;prob=e*bp
            loss-=math.log(max(prob,1e-300))/n
            if gradient:ge[i]=-(bp-known)*e*(1-e)/(n*prob)
        if not gradient:return (loss,0.)
        g={"read":h.T@qg,"bias":qg.sum(axis=0),
           "fusion_class":z.T@gm,"fusion_escape":z.T@ge}
        dh=qg@self.p["read"].T+dz@self.proj.T+gm[:,None]*self.p["fusion_class"][:128]+ge[:,None]*self.p["fusion_escape"][:128]
        g.update(self.back(x[:,:40],v,tape,dh))
        return loss,g,lg
    def wordloss(self,data):
        x,y=data;total=0.;known=0.;oov=0.
        for i in range(0,len(y),32):
            a=x[i:i+32];b=y[i:i+32];total+=len(b)*self.direct(a,b)[0]
        return {"full_word_nll":total/len(y),"targets":len(y),"oov_targets":int((y<0).sum())}
    def law(self,h,base):
        if not any(np.any(self.p[k]) for k in self.fusion_keys):
            return L.Lexical.law(self,h,base)
        q,hidden,_,_=self.tape(np.asarray([self.features(h)],np.uint8))
        _,_,mix,e=self.gates(hidden);q=q[0].astype(np.float64);q/=q.sum()
        q=(1-float(mix[0]))*q+float(mix[0])*self.sem.classlaw(h);q/=q.sum()
        old=self.sem.law(h,base,True,q)
        z=(hidden@self.proj)[0];score=(self.lex@z).astype(np.float64)
        d=old.d/.94;classes=self.sem.cls[1:]
        maxima=np.full(256,-np.inf);np.maximum.at(maxima,classes,score)
        tilted=d*np.exp(score-maxima[classes]);norm=np.bincount(classes,weights=tilted,minlength=256)
        d=tilted*q[classes]/np.maximum(norm[classes],1e-300)
        assert abs(d.sum()-1)<1e-9
        return AdaptiveDense(self.sem,d,base,float(e[0]))
    def check_dual(self,data):
        x,y=data;x=x[:16];y=y[:16]
        assert np.array_equal(self.forward(x[:,:40])[0],self.parent.forward(x[:,:40])[0])
        self.p["fusion_class"][-1]=.12;self.p["fusion_escape"][-1]=.08
        _,g,lg=self.direct(x,y,True);errors={}
        for k in self.p:
            ix=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape);a=float(self.p[k][ix]);eps=.005
            self.p[k][ix]=a+eps;hi=self.direct(x,y)[0];self.p[k][ix]=a-eps;lo=self.direct(x,y)[0];self.p[k][ix]=a
            err=abs((hi-lo)/(2*eps)-float(g[k][ix]));assert err<.006,(k,err);errors[k]=err
        # Check the actual deployed likelihood and its prefix factorization.
        for row,j in zip(x[:4],y[:4]):
            doc,i=self.records[int(row[40])];state=H.Replay()
            for w in doc[:i]:state.observe(w)
            orig,_=V.branches(*self.runtime[:3],state,self.runtime[3]);base=orig[1].law
            law=self.law(state.h,base);w=doc[i]
            pred=-math.log(law.complete(w));loss=self.direct(row[None],np.array([j]))[0]
            assert abs(pred-loss)<2e-5,(pred,loss)
            mix=H.Mixture((law,),(1.,));prefix="";prob=1.
            for c in w+B.EOS:
                prob*=mix.next(prefix)[c]
                if c!=B.EOS:prefix+=c
            assert math.isclose(prob,law.complete(w),rel_tol=1e-6,abs_tol=1e-15)
        for k in self.fusion_keys:self.p[k].fill(0)
        return {"zero_extension_exact":True,"gradient_abs_errors":errors,"deployment_likelihood_match":True,"prefix_factorization":True}
    def train_dual(self,data,steps,new_only):
        x,y=data;trace=[];start=time.monotonic();keys=self.fusion_keys if new_only else tuple(self.p)
        for t in range(steps):
            ix=self.rng.integers(len(y),size=16);loss,g,lg=self.direct(x[ix],y[ix],True);self.step+=1
            for k in keys:
                self.steps[k]+=1;j=self.steps[k];a=g[k]
                self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
                lr=.003 if k in self.fusion_keys else .0005
                self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
                if k not in ("read","bias","field_out",*self.fusion_keys):np.clip(self.p[k],0.,3.,out=self.p[k])
                assert np.isfinite(self.p[k]).all()
            self.p["fusion_class"][-1]=max(0.,float(self.p["fusion_class"][-1]))
            if not new_only:
                for c,grad in lg.items():
                    rows=self.groups[c];self.lt[c]+=1;j=int(self.lt[c])
                    self.lm[rows]=.9*self.lm[rows]+.1*grad;self.lv[rows]=.999*self.lv[rows]+.001*grad*grad
                    self.lex[rows]-=.0015*(self.lm[rows]/(1-.9**j))/(np.sqrt(self.lv[rows]/(1-.999**j))+1e-8)
            if t%32==0 or t==steps-1:trace.append([t+1,loss])
        return {"steps":steps,"sampled_targets":steps*16,"new_only":new_only,"trace":trace,"seconds":time.monotonic()-start}
    def diagnostics(self,data):
        x,y=data;x=x[:256];y=y[:256]
        _,h,_,_=self.tape(x[:,:40]);_,_,m,e=self.gates(h)
        full=self.wordloss((x,y));save={k:self.p[k].copy() for k in self.fusion_keys}
        ab={}
        for k in self.fusion_keys:
            self.p[k].fill(0);ab[k]=self.wordloss((x,y));self.p[k][:]=save[k]
        return {"full":full,"ablations":ab,"retrieval_weight_mean":float(m.mean()),"escape_mean":float(e.mean()),
                "channel_l2":{k:float(np.linalg.norm(self.p[k])) for k in self.fusion_keys},"cached_contexts":len(self.cache),"steps":self.steps}
    def save(self,path):
        parent=self.v25_sha
        try:
            self.v25_sha=self.v27_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent

def branches(ref,small,large,sem,field,state,cal,oldprior):
    orig,key=V.branches(ref,small,large,state,cal);base=orig[1].law
    old=H.Mixture(orig,oldprior[1].get(key,oldprior[0]));count=sem.law(state.h,base,True)
    return (old,count,field.parent.law(state.h,base),field.law(state.h,base)),key

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
                modes={"v27_field":(0,0,1,0),"v28_field":(0,0,0,1),"v27_fixed":(.65,.15,.20,0),"v28_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
                for name,ws in modes.items():
                    l=H.Mixture(q,ws);rank,cert,n=l.rank(budget=2000)
                    m=ms[name];m["n"]+=1;m["t1"]+=w==rank[0];m["t5"]+=w in rank;m["cert"]+=cert
                    m["bits"]-=math.log2(max(l.complete(w),1e-300));m["symbols"]+=len(w)+1
                    p="";prod=1.
                    for c in w+B.EOS:
                        d=l.next(p);prod*=d[c]
                        order=sorted(d,key=lambda k:(-d[k],k));m['symbol_n']+=1;m['symbol_t1']+=c==order[0];m['symbol_t5']+=c in order[:5]
                        if c!=B.EOS:p+=c
                    assert math.isclose(prod,l.complete(w),rel_tol=1e-6,abs_tol=1e-15)
                a,ac,_=H.Mixture((q[2],),(1.,)).rank(32,budget=5000)
                b,bc,_=H.Mixture((q[3],),(1.,)).rank(32,budget=5000)
                oracle["n"]+=1;oracle["parent32"]+=w in a;oracle["recurrent32"]+=w in b;oracle["new32"]+=w in b and w not in a;oracle["cert"]+=ac+bc
            s.observe(w)
    return {"modes":{k:dict(m,word_top1=m["t1"]/m["n"],word_top5=m["t5"]/m["n"],spelling_bps=m["bits"]/m["symbols"]) for k,m in ms.items()},"oracle":dict(oracle),"seconds":time.monotonic()-t}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--recurrent",type=pathlib.Path,required=True);ap.add_argument("--direct",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));ap.add_argument("--dual",type=pathlib.Path,required=True);ap.add_argument("--symbiotic",type=pathlib.Path,required=True);a=ap.parse_args();r=a.data
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
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Fusion(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,len(data[1]));field.bind(ref,small,large,cal)
    # Preserve parent law at zero lexical residual, including prefix masses.
    s=H.Replay()
    for w in cdocs[0][:16]:s.observe(w)
    q,k=branches(ref,small,large,sem,field,s,cal,oldprior)
    for w in ("a","the","unseenword","garden"):
        assert q[2].complete(w)==q[3].complete(w)
    for prefix in ("","t","th","ga"):assert q[2].mass(prefix)==q[3].mass(prefix)
    wd,train_oov=field.worddata(expanded,cap=96);vd,valid_oov=field.worddata(cdocs,cap=96)
    gradcheck=field.check_dual(wd)
    pre=field.wordloss(vd);phase1=field.train_dual(wd,128,True);mid=field.wordloss(vd)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    phase2=field.train_dual(wd,128,False);post=field.wordloss(vd)
    train={"dual_only":phase1,"joint":phase2,"middle":mid,"train_oov":train_oov,"calibration_oov":valid_oov}
    diagnostics=field.diagnostics(vd);movement={"lexical_delta_l2":float(np.linalg.norm(field.lex-field.parent.lex)),"dual":diagnostics["channel_l2"]}
    print("FUSION_FIELD",json.dumps({"before":pre,"after":post,"training":train,"diagnostics":diagnostics,"gradient_check":gradcheck}),flush=True)
    field.save(a.out.with_name("fusion-word-field.npz"))
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"full_word_cal_before":pre,"full_word_cal_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.v27_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
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
    field.save(a.out.with_name("fusion-word-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
