"""Stage 1 / v29: signed cross-coordinate exchanges at every field point."""
import hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, coordinate_exchange as E
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib

def load28(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v27_sha
        assert int(ck["step"])==4864
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v28_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return f


class Coordinate(F.Fusion):
    coord_keys=("coord_lp","coord_ln","coord_up","coord_un")
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,available):
        super().__init__(sem,p23,p24,p25,p26,p27,available);load28(self,p28)
        self.parent=load28(F.Fusion(sem,p23,p24,p25,p26,p27,available),p28)
        self.inherited_keys=tuple(self.p)
        for k in self.coord_keys:
            self.p[k]=np.zeros((128,28),np.float32);self.m[k]=np.zeros_like(self.p[k]);self.v[k]=np.zeros_like(self.p[k]);self.steps[k]=0

    def _extension(self,x,hidden):
        p={k:torch.tensor(self.p[k],requires_grad=True) for k in (*self.new_keys,*self.coord_keys)}
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
        ix=torch.as_tensor(x[:,8:].astype(np.int64)).reshape(-1,8,4)
        s=torch.zeros((len(x),8,128)).scatter_add_(2,ix,torch.full(ix.shape,.25))
        z=h[:,None,:].expand(-1,8,-1)
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

    def check_dual(self,data):
        x,y=data;x=x[:16];y=y[:16]
        assert np.array_equal(self.forward(x[:,:40])[0],self.parent.forward(x[:,:40])[0])
        rng=np.random.default_rng(2902)
        for k in self.coord_keys:self.p[k][:]=rng.uniform(.02,.15,self.p[k].shape)
        _,g,_=self.direct(x,y,True);errors={};norms={}
        for k in self.coord_keys:
            ix=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape);a=float(self.p[k][ix]);eps=.005
            self.p[k][ix]=a+eps;hi=self.direct(x,y)[0];self.p[k][ix]=a-eps;lo=self.direct(x,y)[0];self.p[k][ix]=a
            err=abs((hi-lo)/(2*eps)-float(g[k][ix]));assert err<.001,(k,err);errors[k]=err
            norms[k]=float(np.linalg.norm(g[k]));assert norms[k]>0
        for k in self.coord_keys:self.p[k].fill(0)
        assert np.array_equal(self.forward(x[:,:40])[0],self.parent.forward(x[:,:40])[0])
        return {"zero_extension_exact":True,"gradient_abs_errors":errors,"gradient_l2":norms}
    def train_dual(self,data,steps,new_only):
        x,y=data;trace=[];start=time.monotonic();keys=self.coord_keys if new_only else tuple(self.p)
        for t in range(steps):
            ix=self.rng.integers(len(y),size=16);loss,g,lg=self.direct(x[ix],y[ix],True);self.step+=1
            for k in keys:
                self.steps[k]+=1;j=self.steps[k];a=g[k]
                self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
                lr=.003 if k in self.coord_keys else .0005
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
        full=self.wordloss((x,y));save={k:self.p[k].copy() for k in self.coord_keys}
        for k in self.coord_keys:self.p[k].fill(0)
        ab=self.wordloss((x,y))
        for k in self.coord_keys:self.p[k][:]=save[k]
        # Break only the extended causal addresses, not retrieval or parent intake.
        wrong=x.copy();wrong[:,8:40]=np.roll(x[:,8:40],len(x)//2,axis=0)
        shuffled=self.wordloss((wrong,y))
        return {"full":full,"coordinate_off":ab,"shuffled_extra_history":shuffled,
                "joint_contraction_bound":.905,"four_pass_residual_inf":self.tape(x[:32,:40])[3][-1],
                "channel_l2":{k:float(np.linalg.norm(self.p[k])) for k in self.coord_keys},"steps":self.steps}
    def save(self,path):
        parent=self.v25_sha
        try:
            self.v25_sha=self.v28_sha;D.Dual.save(self,path)
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
                modes={"v28_field":(0,0,1,0),"v29_field":(0,0,0,1),"v28_fixed":(.65,.15,.20,0),"v29_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
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
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--recurrent",type=pathlib.Path,required=True);ap.add_argument("--direct",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));ap.add_argument("--dual",type=pathlib.Path,required=True);ap.add_argument("--symbiotic",type=pathlib.Path,required=True);ap.add_argument("--fusion",type=pathlib.Path,required=True);a=ap.parse_args();r=a.data
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
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Coordinate(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,len(data[1]));field.bind(ref,small,large,cal)
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
    print("COORDINATE_FIELD",json.dumps({"before":pre,"after":post,"training":train,"diagnostics":diagnostics,"gradient_check":gradcheck}),flush=True)
    field.save(a.out.with_name("coordinate-word-field.npz"))
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"full_word_cal_before":pre,"full_word_cal_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.v28_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
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
    field.save(a.out.with_name("coordinate-word-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
