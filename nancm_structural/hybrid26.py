"""v26: full pair-address geometry and distinct support/opposition dynamics.
Actual v25 continuation, zero-residual parity, bounded direct-word diagnostic."""
import hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N
import numpy as np
import argparse, collections as C, json, math, pathlib, time, hashlib

def load25(field,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(field.sem.words))
        assert np.array_equal(ck["classes"],field.sem.cls)
        assert np.array_equal(ck["upper_pairs"],field.upper_pairs)
        assert str(ck["parent_sha"].item())==field.v24_sha
        assert int(ck["step"])==3072
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(field.p)
        for k in field.p:
            for prefix,group in (("",field.p),("m_",field.m),("v_",field.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(field,k).shape and np.isfinite(a).all();setattr(field,k,a.copy())
        field.steps=steps;field.step=int(ck["step"]);field.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    field.v25_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return field

class Dual(L.Lexical):
    names=("dual_lower_plus","dual_lower_minus","dual_upper_plus","dual_upper_minus")
    def __init__(self,sem,p23,p24,p25,available):
        super().__init__(sem,p23,p24,available);load25(self,p25)
        self.parent=load25(L.Lexical(sem,p23,p24,available),p25)
        self.augment()
    def augment(self):
        self.lower_pairs=np.array([B.pair(int(i),int(j)) for i,j in zip(self.i,self.j)],np.uint16)
        cross={B.pair(i,128+j) for i in range(128) for j in range(128)}
        allpairs=cross|set(map(int,self.lower_pairs))|set(map(int,self.upper_pairs))
        assert allpairs==set(range(32640))
        self.inherited_keys=tuple(self.p)
        for k in self.names:
            self.p[k]=np.zeros(8128,np.float32);self.m[k]=np.zeros_like(self.p[k]);self.v[k]=np.zeros_like(self.p[k]);self.steps[k]=0
    def matrix(self,k,scale):
        a=np.tanh(self.p[k]);A=np.zeros((128,128),np.float32)
        A[self.i,self.j]=a;A[self.j,self.i]=a
        ab=np.sqrt(A*A+1e-6);den=1+ab.sum(axis=1,keepdims=True)
        W=scale*A/den
        assert np.max(np.abs(W).sum(axis=1))<scale
        return W,(A,ab,den,a,scale)
    def matrixgrad(self,g,meta):
        A,ab,den,a,scale=meta
        dA=scale*g/den-scale*(g*A).sum(axis=1,keepdims=True)/den**2*A/ab
        return (dA[self.i,self.j]+dA[self.j,self.i])*(1-a*a)
    def tape(self,x):
        p=self.p
        v=np.tanh(p["plus"][self.rails,x]-p["minus"][self.rails,x])
        z0=v.sum(axis=1)/math.sqrt(8)
        u=np.zeros((len(x),128),np.float32)
        for rail in range(8):np.add.at(u,(np.arange(len(x)),x[:,rail]),.125)
        LP,lp=self.matrix(self.names[0],.45);LN,ln=self.matrix(self.names[1],.45)
        SP,sp=self.matrix(self.names[2],.12);SN,sn=self.matrix(self.names[3],.12)
        # Lower support diffuses the excitation itself; opposition responds
        # quadratically to concentrated excitation. They are not interchangeable.
        lower=np.tanh(u@LP.T-(u*u)@LN.T)
        cross=np.tanh(p["plus"]-p["minus"]);Cmat=cross.mean(axis=0)
        z=z0+.4*(lower@Cmat)
        W,A,den,ab,a=self.coupling()
        states=[np.tanh(z)]
        for _ in range(self.passes):
            s=states[-1]
            # Upper support transfers positive activation; opposition suppresses
            # large-amplitude activations of either sign.
            states.append(np.tanh(z+s@W.T+np.maximum(s,0)@SP.T-(s*s)@SN.T))
        h=states[-1];logits=h@p["read"]+p["bias"]
        logits[:,~self.sem.class_active]=-1e9;logits-=logits.max(axis=1,keepdims=True)
        q=np.exp(logits);q/=q.sum(axis=1,keepdims=True)
        return q,h,v,(states,W,A,den,ab,a,z,u,lower,cross,Cmat,LP,LN,SP,SN,lp,ln,sp,sn)
    def back(self,x,v,tape,dh):
        states,W,A,den,ab,a,z,u,lower,cross,Cmat,LP,LN,SP,SN,lp,ln,sp,sn=tape
        dz=np.zeros_like(dh);gW=np.zeros_like(W);gSP=np.zeros_like(SP);gSN=np.zeros_like(SN)
        for k in range(self.passes,0,-1):
            prev=states[k-1];d=dh*(1-states[k]*states[k]);dz+=d
            gW+=d.T@prev;gSP+=d.T@np.maximum(prev,0);gSN-=d.T@(prev*prev)
            dh=d@W+(d@SP)*(prev>0)-2*prev*(d@SN)
        dz+=dh*(1-states[0]*states[0])
        dA=self.alpha*gW/den-self.alpha*(gW*A).sum(axis=1,keepdims=True)/den**2*A/ab
        g=(dA[self.i,self.j]+dA[self.j,self.i])*(1-a*a)
        dl=.4*(dz@Cmat.T)*(1-lower*lower)
        gp=np.zeros_like(self.p["plus"]);dv=dz[:,None,:]*(1-v*v)/math.sqrt(8)
        for j in range(8):np.add.at(gp[j],x[:,j],dv[:,j])
        gp+=(.4*(lower.T@dz)[None,:,:]/8)*(1-cross*cross)
        return {"plus":gp,"minus":-gp,"rec_plus":g,"rec_minus":-g,
                self.names[0]:self.matrixgrad(dl.T@u,lp),self.names[1]:self.matrixgrad(-dl.T@(u*u),ln),
                self.names[2]:self.matrixgrad(gSP,sp),self.names[3]:self.matrixgrad(gSN,sn)}
    def check_dual(self,data):
        x,y=data;x=x[:24];y=y[:24]
        assert np.array_equal(self.forward(x)[0],self.parent.forward(x)[0])
        rng=np.random.default_rng(2601)
        for k in self.names:self.p[k][:]=rng.uniform(.005,.04,len(self.i))
        loss,g,lg=self.direct(x,y,True);errors={}
        for k in self.p:
            idx=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape);a=float(self.p[k][idx]);eps=.003
            self.p[k][idx]=a+eps;hi=sum(self.direct(x,y));self.p[k][idx]=a-eps;lo=sum(self.direct(x,y));self.p[k][idx]=a
            e=abs((hi-lo)/(2*eps)-float(g[k][idx]));assert e<.004,(k,e);errors[k]=e
        c=max(lg,key=lambda k:np.max(np.abs(lg[k])));i,j=np.unravel_index(np.abs(lg[c]).argmax(),lg[c].shape)
        row=self.groups[c][i];a=float(self.lex[row,j]);eps=.003
        self.lex[row,j]=a+eps;hi=sum(self.direct(x,y));self.lex[row,j]=a-eps;lo=sum(self.direct(x,y));self.lex[row,j]=a
        e=abs((hi-lo)/(2*eps)-float(lg[c][i,j]));assert e<.004;errors["lexical"]=e
        independent={"lower":float(np.linalg.norm(g[self.names[0]]+g[self.names[1]])),
                     "upper":float(np.linalg.norm(g[self.names[2]]+g[self.names[3]]))}
        assert min(independent.values())>0
        for k in self.names:self.p[k].fill(0)
        return {"zero_extension_exact":True,"gradient_abs_errors":errors,"non_opposite_gradients_l2":independent}
    def train_dual(self,data,steps,new_only):
        x,y=data;trace=[];start=time.monotonic()
        keys=self.names if new_only else tuple(self.p)
        for t in range(steps):
            ix=self.rng.integers(len(y),size=64);loss,g,lg=self.direct(x[ix],y[ix],True);self.step+=1
            for k in keys:
                self.steps[k]+=1;j=self.steps[k]
                self.m[k]=.9*self.m[k]+.1*g[k];self.v[k]=.999*self.v[k]+.001*g[k]*g[k]
                lr=.003 if k in self.names else .0007
                self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
                if k in self.names or k in ("plus","minus","rec_plus","rec_minus"):np.clip(self.p[k],0.,3.,out=self.p[k])
                assert np.isfinite(self.p[k]).all()
            if not new_only:
                for c,grad in lg.items():
                    rows=self.groups[c];self.lt[c]+=1;j=int(self.lt[c])
                    self.lm[rows]=.9*self.lm[rows]+.1*grad;self.lv[rows]=.999*self.lv[rows]+.001*grad*grad
                    self.lex[rows]-=.0015*(self.lm[rows]/(1-.9**j))/(np.sqrt(self.lv[rows]/(1-.999**j))+1e-8)
                    assert np.isfinite(self.lex[rows]).all()
            if t%64==0 or t==steps-1:trace.append([t+1,loss])
        return {"steps":steps,"sampled_targets":steps*64,"new_only":new_only,"trace":trace,"seconds":time.monotonic()-start}
    def diagnostics(self,data):
        q,h,v,tape=self.tape(data[0][:128])
        states,W,A,den,ab,a,z,u,lower,cross,Cmat,LP,LN,SP,SN,*_=tape
        row=lambda M:float(np.max(np.abs(M).sum(axis=1)))
        bound=row(W)+row(SP)+2*row(SN)
        assert bound<.91
        residual=float(np.max(np.abs(np.tanh(z+h@W.T+np.maximum(h,0)@SP.T-(h*h)@SN.T)-h)))
        return {"addressed_pairs":32640,"new_pair_parameters":32512,"upper_contraction_bound":bound,
                "lower_support_row_norm":row(LP),"lower_opposition_row_norm":row(LN),"four_pass_residual_inf":residual,
                "channel_l2":{k:float(np.linalg.norm(self.p[k])) for k in self.names},"steps":self.steps}
    def save(self,path):
        pairs=np.array([[B.pair(i,128+j) for j in range(128)] for i in range(128)],np.uint16)
        np.savez_compressed(path,step=np.array(self.step),pairs=pairs,upper_pairs=self.upper_pairs,lower_pairs=self.lower_pairs,
          words=np.array(self.sem.words),classes=self.sem.cls,centers=self.sem.centers,parent_sha=np.array(self.v25_sha),
          rng_json=np.array(json.dumps(self.rng.bit_generator.state)),group_steps_json=np.array(json.dumps(self.steps)),
          lexical=self.lex,lex_m=self.lm,lex_v=self.lv,lex_steps=self.lt,projection=self.proj,
          **self.p,**{"m_"+k:v for k,v in self.m.items()},**{"v_"+k:v for k,v in self.v.items()})

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
                modes={"v25_field":(0,0,1,0),"v26_field":(0,0,0,1),"v25_fixed":(.65,.15,.20,0),"v26_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
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
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--recurrent",type=pathlib.Path,required=True);ap.add_argument("--direct",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
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
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Dual(sem,a.parent,a.recurrent,a.direct,len(data[1]))
    # Preserve parent law at zero lexical residual, including prefix masses.
    s=H.Replay()
    for w in cdocs[0][:16]:s.observe(w)
    q,k=branches(ref,small,large,sem,field,s,cal,oldprior)
    for w in ("a","the","unseenword","garden"):
        assert q[2].complete(w)==q[3].complete(w)
    for prefix in ("","t","th","ga"):assert q[2].mass(prefix)==q[3].mass(prefix)
    wd,train_oov=field.worddata(expanded,cap=768);vd,valid_oov=field.worddata(cdocs)
    gradcheck=field.check_dual(wd)
    pre=field.wordloss(vd);phase1=field.train_dual(wd,512,True);mid=field.wordloss(vd)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    phase2=field.train_dual(wd,512,False);post=field.wordloss(vd)
    train={"dual_only":phase1,"joint":phase2,"middle":mid,"train_oov":train_oov,"calibration_oov":valid_oov}
    diagnostics=field.diagnostics(valid);movement={"lexical_delta_l2":float(np.linalg.norm(field.lex-field.parent.lex)),"dual":diagnostics["channel_l2"]}
    print("DUAL_FIELD",json.dumps({"before":pre,"after":post,"training":train,"diagnostics":diagnostics,"gradient_check":gradcheck}),flush=True)
    field.save(a.out.with_name("dual-word-field.npz"))
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"known_word_cal_before":pre,"known_word_cal_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.v25_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
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
    field.save(a.out.with_name("dual-word-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
