"""Stage 2 / v30: signed multi-timescale rotating temporal memory.
Bounded sequence credit assignment; streaming state lives outside weights."""
import hybrid29 as Q, hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, coordinate_exchange as E, temporal_phasor as T
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib, functools

def load29(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v28_sha
        assert int(ck["step"])==5120
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v29_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return f



class TemporalReplay(H.Replay):
    def __init__(self,field):
        super().__init__();self.field=field
        with torch.no_grad():
            self.coeff=T.coefficients(*(torch.from_numpy(field.p[k]) for k in field.temporal_keys[:3]))
            self.trace=torch.zeros(1,2,4,128,2)
    def observe(self,w):
        with torch.no_grad():
            e=torch.tensor(self.field.event(w),dtype=torch.long).reshape(1,2,4)
            self.trace=T.step(self.trace,e,*self.coeff)
        super().observe(w)
    def snapshot(self):
        return {"trace":self.trace.clone(),"h":list(self.h),"ep":{k:v.copy() for k,v in self.ep.items()},"long":{k:v.copy() for k,v in self.long.items()}}
    def restore(self,state):
        self.trace=state["trace"].clone();self.h=list(state["h"])
        self.ep=C.defaultdict(C.Counter,{k:v.copy() for k,v in state["ep"].items()})
        self.long=C.defaultdict(C.Counter,{k:v.copy() for k,v in state["long"].items()})

class Temporal(Q.Coordinate):
    temporal_keys=("time_decay","time_phase","time_amp","time_lower","time_upper")
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,available);load29(self,p29)
        self.parent=load29(Q.Coordinate(sem,p23,p24,p25,p26,p27,p28,available),p29)
        self.inherited_keys=tuple(self.p);self.stream_override=None
        rng=np.random.default_rng(3002)
        for k in self.temporal_keys:
            a=np.zeros((8,128) if k in self.temporal_keys[3:] else (2,4,128),np.float32)
            if k=="time_phase":a[:]=rng.uniform(-.25,.25,a.shape)
            self.p[k]=a;self.m[k]=np.zeros_like(a);self.v[k]=np.zeros_like(a);self.steps[k]=0
    @functools.lru_cache(maxsize=100000)
    def event(self,w):
        c=self.sem.classof(w)
        return tuple(self.address(100+4*s+i,(w if s==0 else c,w[:i+1] if s==0 else len(w)%8)) for s in range(2) for i in range(4))
    def events(self,h):
        tail=h[-T.LENGTH:]
        return [128]*(8*(T.LENGTH-len(tail)))+[a for w in tail for a in self.event(w)]
    def worddata(self,docs,cap=None):
        x=[];y=[];oov=0
        for doc in docs:
            for i,w in enumerate(doc if cap is None else doc[:cap]):
                j=self.sem.ids.get(w,0);oov+=j==0
                history=doc[max(0,i-128):i]
                x.append(self.features(history)+[len(self.records)]+self.events(history))
                self.records.append((doc,i));y.append(j-1)
        return (np.asarray(x,np.uint32),np.asarray(y,np.int64)),oov
    def session(self):return TemporalReplay(self)
    def law_stream(self,state,base):
        self.stream_override=state.trace
        try:return self.law(state.h,base)
        finally:self.stream_override=None

    def _extension(self,x,hidden):
        p={k:torch.tensor(self.p[k],requires_grad=True) for k in (*self.new_keys,*self.coord_keys,*self.temporal_keys)}
        h=torch.tensor(hidden,requires_grad=True)
        def norm(a,scale):
            return scale*a/(1+a.abs().sum(-1,keepdim=True))
        def pair(k,scale):
            a=torch.zeros((8,128,128));v=p[k].tanh()
            a[:,self.ti,self.tj]=v;a[:,self.tj,self.ti]=v
            return norm(a,scale)
        lp,ln=pair("field_lp",.22),pair("field_ln",.10)
        up,un=pair("field_up",.22),pair("field_un",.10)
        cp=p["field_cp"].tanh();cn=p["field_cn"].tanh()
        lu,lv=norm(cp,.22),norm(cn,.10)
        ul,uv=norm(cp.transpose(1,2),.22),norm(cn.transpose(1,2),.10)
        xp,xn,yp,yn=(E.matrices(p[k]) for k in self.coord_keys)
        if self.ablate=="feedback":lu=lu*0;lv=lv*0
        ix=torch.as_tensor(x[:,8:40].astype(np.int64)).reshape(-1,8,4)
        s=torch.zeros((len(x),8,128)).scatter_add_(2,ix,torch.full(ix.shape,.25))
        z=h[:,None,:].expand(-1,8,-1)
        rho,theta,gain=T.coefficients(p["time_decay"],p["time_phase"],p["time_amp"])
        events=torch.as_tensor(x[:,40:].astype(np.int64)).reshape(-1,T.LENGTH,2,4)
        mem=T.scan(events,rho,theta,gain) if self.stream_override is None else self.stream_override
        c=T.contrast(mem)
        s=s+.5*c*p["time_lower"].tanh()
        z=z+.5*c*p["time_upper"].tanh()
        lo=s.tanh();hi=z.tanh()
        def transfer(a,b,positive,negative):
            return torch.einsum("cij,bcj->bci",positive,a.relu())-torch.einsum("cij,bcj->bci",negative,b.square())
        def advance(lo,hi):
            return ((s+transfer(lo,lo,lp,ln)+transfer(hi,hi,lu,lv)+E.exchange(lo,xp,xn)).tanh(),
                    (z+transfer(hi,hi,up,un)+transfer(lo,lo,ul,uv)+E.exchange(hi,yp,yn)).tanh())
        for _ in range(4):lo,hi=advance(lo,hi)
        lnext,hnext=advance(lo,hi)
        residual=max(float((lnext-lo).detach().abs().max()),float((hnext-hi).detach().abs().max()))
        joint=torch.cat((lo,hi),dim=1).transpose(1,2)
        out=h+.35*(joint*p["field_out"]).sum(-1).tanh()
        if self.ablate=="readout":out=h
        return out,h,p,residual


    def direct(self,x,y,gradient=False):
        q,h,v,tape=self.tape(np.concatenate((x[:,:40],x[:,41:]),axis=1));n=len(y)
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
        g.update(self.back(np.concatenate((x[:,:40],x[:,41:]),axis=1),v,tape,dh))
        return loss,g,lg

    def law(self,h,base):
        q,hidden,_,_=self.tape(np.asarray([self.features(h)+self.events(h)],np.uint32))
        _,_,mix,e=self.gates(hidden);q=q[0].astype(np.float64);q/=q.sum()
        q=(1-float(mix[0]))*q+float(mix[0])*self.sem.classlaw(h);q/=q.sum()
        old=self.sem.law(h,base,True,q)
        z=(hidden@self.proj)[0];score=(self.lex@z).astype(np.float64)
        d=old.d/.94;classes=self.sem.cls[1:]
        maxima=np.full(256,-np.inf);np.maximum.at(maxima,classes,score)
        tilted=d*np.exp(score-maxima[classes]);norm=np.bincount(classes,weights=tilted,minlength=256)
        d=tilted*q[classes]/np.maximum(norm[classes],1e-300)
        assert abs(d.sum()-1)<1e-9
        return F.AdaptiveDense(self.sem,d,base,float(e[0]))

    def check_dual(self,data):
        x,y=data;x=x[:16];y=y[:16];inp=np.concatenate((x[:,:40],x[:,41:]),axis=1)
        assert np.array_equal(self.forward(inp)[0],self.parent.forward(x[:,:40])[0])
        self.p["time_lower"].fill(.15);self.p["time_upper"].fill(.15)
        _,g,_=self.direct(x,y,True);errors={};norms={}
        for k in self.temporal_keys:
            ix=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape);a=float(self.p[k][ix]);eps=.005
            self.p[k][ix]=a+eps;hi=self.direct(x,y)[0];self.p[k][ix]=a-eps;lo=self.direct(x,y)[0];self.p[k][ix]=a
            err=abs((hi-lo)/(2*eps)-float(g[k][ix]));assert err<.001,(k,err);errors[k]=err
            norms[k]=float(np.linalg.norm(g[k]));assert norms[k]>0,k
        # Integrated streamed/batch law equivalence at nonzero temporal coupling.
        doc,pos=self.records[int(x[-1,40])];session=self.session()
        for w in doc[:pos]:session.observe(w)
        ref,small,large,cal=self.runtime;orig,_=V.branches(ref,small,large,session,cal);base=orig[1].law
        aa=self.law(session.h,base);bb=self.law_stream(session,base)
        for w in ("a","the","garden",doc[pos]):assert math.isclose(aa.complete(w),bb.complete(w),rel_tol=2e-6,abs_tol=1e-12)
        restored=self.session();restored.restore(session.snapshot());assert torch.equal(restored.trace,session.trace)
        self.p["time_lower"].fill(0);self.p["time_upper"].fill(0)
        assert np.array_equal(self.forward(inp)[0],self.parent.forward(x[:,:40])[0])
        return {"zero_extension_exact":True,"gradient_abs_errors":errors,"gradient_l2":norms,"stream_law_equivalence":True,"session_resume_exact":True}
    def train_dual(self,data,steps,new_only):
        x,y=data;trace=[];start=time.monotonic();keys=self.temporal_keys if new_only else tuple(self.p)
        for t in range(steps):
            ix=self.rng.integers(len(y),size=16);loss,g,lg=self.direct(x[ix],y[ix],True);self.step+=1
            for k in keys:
                self.steps[k]+=1;j=self.steps[k];a=g[k]
                self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
                lr=.003 if k in self.temporal_keys else .0005
                self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
                if k not in ("read","bias","field_out",*self.fusion_keys,*self.temporal_keys):np.clip(self.p[k],0.,3.,out=self.p[k])
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
        full=self.wordloss((x,y));wrong=x.copy();wrong[:,41:]=np.roll(x[:,41:],len(x)//2,axis=0)
        shuffled=self.wordloss((wrong,y));rev=x.copy();rev[:,41:]=x[:,41:].reshape(-1,T.LENGTH,8)[:,::-1].reshape(len(x),-1)
        reversed_loss=self.wordloss((rev,y));save={k:self.p[k].copy() for k in self.temporal_keys[3:]}
        for k in save:self.p[k].fill(0)
        off=self.wordloss((x,y))
        for k,a in save.items():self.p[k][:]=a
        rho,_,_=T.coefficients(*(torch.from_numpy(self.p[k]) for k in self.temporal_keys[:3]))
        return {"full":full,"wrong_history":shuffled,"reversed_history":reversed_loss,"temporal_off":off,
                "rho_min":float(rho.min()),"rho_max":float(rho.max()),"training_horizon":T.LENGTH,
                "channel_l2":{k:float(np.linalg.norm(self.p[k])) for k in self.temporal_keys},"steps":self.steps}
    def save(self,path):
        parent=self.v25_sha
        try:
            self.v25_sha=self.v29_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent
def branches(ref,small,large,sem,field,state,cal,oldprior):
    orig,key=V.branches(ref,small,large,state,cal);base=orig[1].law
    old=H.Mixture(orig,oldprior[1].get(key,oldprior[0]));count=sem.law(state.h,base,True)
    return (old,count,field.parent.law(state.h,base),field.law_stream(state,base)),key

def fit(ref,small,large,sem,field,cal,oldprior,docs):
    rows=C.defaultdict(list)
    for doc in docs:
        s=field.session()
        for i,w in enumerate(doc[:48]):
            if i>=16:
                q,k=branches(ref,small,large,sem,field,s,cal,oldprior);p=[l.complete(w) for l in q];rows[("all",)].append(p);rows[k].append(p)
            s.observe(w)
    q=U.em(rows[("all",)],[.62,.25,.065,.065],pseudocount=24)
    return q,{k:U.em(v,q,pseudocount=64) for k,v in rows.items() if k!=("all",)}
def evaluate(ref,small,large,sem,field,cal,oldprior,prior,docs,limit):
    ms=C.defaultdict(C.Counter);oracle=C.Counter();t=time.monotonic()
    for doc in docs:
        s=field.session()
        for i,w in enumerate(doc[:limit+16]):
            if i>=16:
                q,k=branches(ref,small,large,sem,field,s,cal,oldprior)
                modes={"v29_field":(0,0,1,0),"v30_field":(0,0,0,1),"v29_fixed":(.65,.15,.20,0),"v30_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
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
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--recurrent",type=pathlib.Path,required=True);ap.add_argument("--direct",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));ap.add_argument("--dual",type=pathlib.Path,required=True);ap.add_argument("--symbiotic",type=pathlib.Path,required=True);ap.add_argument("--fusion",type=pathlib.Path,required=True);ap.add_argument("--coordinate",type=pathlib.Path,required=True);a=ap.parse_args();r=a.data
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
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Temporal(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,len(data[1]));field.bind(ref,small,large,cal)
    # Preserve parent law at zero lexical residual, including prefix masses.
    s=field.session()
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
    print("TEMPORAL_FIELD",json.dumps({"before":pre,"after":post,"training":train,"diagnostics":diagnostics,"gradient_check":gradcheck}),flush=True)
    field.save(a.out.with_name("temporal-word-field.npz"))
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=field.session()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"full_word_cal_before":pre,"full_word_cal_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.v29_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
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
    field.save(a.out.with_name("temporal-word-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
