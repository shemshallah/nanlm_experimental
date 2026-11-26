"""v27: bidirectional eight-coordinate signed field, long causal intake.
CPU diagnostic; resumes v26 exactly. Torch autograd for new field, inherited
analytic gradients and Adam histories for the parent. No full training."""
import hybrid26 as D
import hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib
torch.set_num_threads(2)

def load26(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v25_sha
        assert int(ck["step"])==4096
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v26_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return f

class Symbiotic(D.Dual):
    def __init__(self,sem,p23,p24,p25,p26,available):
        super().__init__(sem,p23,p24,p25,available);load26(self,p26)
        self.parent=load26(D.Dual(sem,p23,p24,p25,available),p26)
        self.inherited_keys=tuple(self.p)
        self.new_keys=("field_lp","field_ln","field_up","field_un","field_cp","field_cn","field_out")
        rng=np.random.default_rng(2701)
        for k in self.new_keys:
            shape=(128,16) if k=="field_out" else (8,128,128) if k in ("field_cp","field_cn") else (8,8128)
            a=np.zeros(shape,np.float32) if k=="field_out" else rng.gamma(.4,.12,shape).astype(np.float32)
            self.p[k]=a;self.m[k]=np.zeros_like(a);self.v[k]=np.zeros_like(a);self.steps[k]=0
        self.ti=torch.as_tensor(self.i);self.tj=torch.as_tensor(self.j)
        self.ablate=None
    def features(self,h):
        old=P.Field.features(self,h)
        # Eight different time scales, four causal addresses per coordinate.
        extra=[]
        for channel,n in enumerate((1,2,4,8,16,32,64,128)):
            w=h[-n:] if h else []
            cls=tuple(self.sem.classof(t) for t in w)
            # Separate exact identity, role order, spelling and skip history.
            keys=(tuple(w[-4:]),cls,tuple((t[:2],t[-2:],len(t)) for t in w[-8:]),tuple(w[::max(1,n//8)]))
            extra.extend(self.address(8+4*channel+j,k) for j,k in enumerate(keys))
        return old+extra
    def worddata(self,docs,cap=None):
        x=[];y=[];oov=0
        for doc in docs:
            for i,w in enumerate(doc if cap is None else doc[:cap]):
                j=self.sem.ids.get(w)
                if j is None:oov+=1;continue
                x.append(self.features(doc[max(0,i-128):i]));y.append(j-1)
        return (np.asarray(x,np.uint8),np.asarray(y,np.int64)),oov
    def _extension(self,x,hidden):
        p={k:torch.tensor(self.p[k],requires_grad=True) for k in self.new_keys}
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
        if self.ablate=="feedback":lu=lu*0;lv=lv*0
        ix=torch.as_tensor(x[:,8:].astype(np.int64)).reshape(-1,8,4)
        s=torch.zeros((len(x),8,128)).scatter_add_(2,ix,torch.full(ix.shape,.25))
        z=h[:,None,:].expand(-1,8,-1)
        lo=s.tanh();hi=z.tanh()
        def transfer(a,b,positive,negative):
            return torch.einsum("cij,bcj->bci",positive,a.relu())-torch.einsum("cij,bcj->bci",negative,b.square())
        def advance(lo,hi):
            return ((s+transfer(lo,lo,lp,ln)+transfer(hi,hi,lu,lv)).tanh(),
                    (z+transfer(hi,hi,up,un)+transfer(lo,lo,ul,uv)).tanh())
        for _ in range(4):lo,hi=advance(lo,hi)
        lnext,hnext=advance(lo,hi)
        residual=max(float((lnext-lo).detach().abs().max()),float((hnext-hi).detach().abs().max()))
        joint=torch.cat((lo,hi),dim=1).transpose(1,2)
        out=h+.35*(joint*p["field_out"]).sum(-1).tanh()
        if self.ablate=="readout":out=h
        return out,h,p,residual
    def tape(self,x):
        _,parent_h,v,parent_tape=D.Dual.tape(self,x[:,:8])
        out,h,p,residual=self._extension(x,parent_h)
        hidden=out.detach().numpy()
        logits=hidden@self.p["read"]+self.p["bias"]
        logits[:,~self.sem.class_active]=-1e9;logits-=logits.max(axis=1,keepdims=True)
        q=np.exp(logits);q/=q.sum(axis=1,keepdims=True)
        return q,hidden,v,(parent_tape,out,h,p,residual)
    def back(self,x,v,tape,dh):
        parent_tape,out,h,p,_=tape
        out.backward(torch.from_numpy(dh))
        g=D.Dual.back(self,x[:,:8],v,parent_tape,h.grad.numpy())
        g.update({k:(a.grad.numpy().copy() if a.grad is not None else np.zeros_like(self.p[k])) for k,a in p.items()})
        return g
    def check_dual(self,data):
        x,y=data;x=x[:16];y=y[:16]
        assert np.array_equal(self.forward(x)[0],self.parent.forward(x[:,:8])[0])
        self.p["field_out"].fill(.02)
        _,g,lg=self.direct(x,y,True);errors={}
        for k in self.p:
            idx=np.unravel_index(np.abs(g[k]).argmax(),g[k].shape);a=float(self.p[k][idx]);eps=.006
            self.p[k][idx]=a+eps;hi=sum(self.direct(x,y));self.p[k][idx]=a-eps;lo=sum(self.direct(x,y));self.p[k][idx]=a
            err=abs((hi-lo)/(2*eps)-float(g[k][idx]));assert err<.005,(k,err);errors[k]=err
        self.p["field_out"].fill(0)
        assert np.array_equal(self.forward(x)[0],self.parent.forward(x[:,:8])[0])
        return {"zero_extension_exact":True,"gradient_abs_errors":errors,"new_gradient_l2":{k:float(np.linalg.norm(g[k])) for k in self.new_keys}}
    def train_dual(self,data,steps,new_only):
        x,y=data;trace=[];start=time.monotonic()
        keys=self.new_keys if new_only else tuple(self.p)
        for t in range(steps):
            ix=self.rng.integers(len(y),size=32);loss,g,lg=self.direct(x[ix],y[ix],True);self.step+=1
            for k in keys:
                self.steps[k]+=1;j=self.steps[k];a=g[k]
                self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
                lr=.003 if k in self.new_keys else .0005
                self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
                if k not in ("read","bias","field_out"):np.clip(self.p[k],0.,3.,out=self.p[k])
                assert np.isfinite(self.p[k]).all()
            if not new_only:
                for c,grad in lg.items():
                    rows=self.groups[c];self.lt[c]+=1;j=int(self.lt[c])
                    self.lm[rows]=.9*self.lm[rows]+.1*grad;self.lv[rows]=.999*self.lv[rows]+.001*grad*grad
                    self.lex[rows]-=.0015*(self.lm[rows]/(1-.9**j))/(np.sqrt(self.lv[rows]/(1-.999**j))+1e-8)
            if t%64==0 or t==steps-1:trace.append([t+1,loss])
        return {"steps":steps,"sampled_targets":steps*32,"new_only":new_only,"trace":trace,"seconds":time.monotonic()-start}
    def diagnostics(self,data):
        x,y=data;x=x[:256];y=y[:256]
        full=self.wordloss((x,y));wrong=x.copy();wrong[:,8:]=np.roll(wrong[:,8:],len(x)//2,axis=0)
        unrelated=self.wordloss((wrong,y));ablations={}
        for mode in ("feedback","readout"):
            self.ablate=mode;ablations[mode]=self.wordloss((x,y))
        self.ablate=None
        return {"addressed_pairs":32640,"coordinates":8,"new_edge_parameters":2*32640*8,
                "joint_contraction_bound":.84,"four_pass_residual_inf":self.tape(x[:32])[3][-1],
                "full":full,"shuffled_extra_history":unrelated,"ablations":ablations,
                "channel_l2":{k:float(np.linalg.norm(self.p[k])) for k in self.new_keys},"steps":self.steps}
    def save(self,path):
        parent=self.v25_sha
        try:
            self.v25_sha=self.v26_sha
            D.Dual.save(self,path)
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
                modes={"v26_field":(0,0,1,0),"v27_field":(0,0,0,1),"v26_fixed":(.65,.15,.20,0),"v27_fixed":(.65,.15,0,.20),"hybrid":prior[1].get(k,prior[0])}
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
    ap=argparse.ArgumentParser();ap.add_argument("--data",type=pathlib.Path,required=True);ap.add_argument("--lib",type=pathlib.Path,required=True);ap.add_argument("--parent",type=pathlib.Path,required=True);ap.add_argument("--recurrent",type=pathlib.Path,required=True);ap.add_argument("--direct",type=pathlib.Path,required=True);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));ap.add_argument("--dual",type=pathlib.Path,required=True);a=ap.parse_args();r=a.data
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
    data=template.data(expanded,cap=768);valid=template.data(cdocs);field=Symbiotic(sem,a.parent,a.recurrent,a.direct,a.dual,len(data[1]))
    # Preserve parent law at zero lexical residual, including prefix masses.
    s=H.Replay()
    for w in cdocs[0][:16]:s.observe(w)
    q,k=branches(ref,small,large,sem,field,s,cal,oldprior)
    for w in ("a","the","unseenword","garden"):
        assert q[2].complete(w)==q[3].complete(w)
    for prefix in ("","t","th","ga"):assert q[2].mass(prefix)==q[3].mass(prefix)
    wd,train_oov=field.worddata(expanded,cap=768);vd,valid_oov=field.worddata(cdocs)
    gradcheck=field.check_dual(wd)
    pre=field.wordloss(vd);phase1=field.train_dual(wd,256,True);mid=field.wordloss(vd)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    phase2=field.train_dual(wd,256,False);post=field.wordloss(vd)
    train={"dual_only":phase1,"joint":phase2,"middle":mid,"train_oov":train_oov,"calibration_oov":valid_oov}
    diagnostics=field.diagnostics(vd);movement={"lexical_delta_l2":float(np.linalg.norm(field.lex-field.parent.lex)),"dual":diagnostics["channel_l2"]}
    print("SYMBIOTIC_FIELD",json.dumps({"before":pre,"after":post,"training":train,"diagnostics":diagnostics,"gradient_check":gradcheck}),flush=True)
    field.save(a.out.with_name("symbiotic-word-field.npz"))
    prior=fit(ref,small,large,sem,field,cal,oldprior,cdocs);frozen=json.dumps([prior[0],{str(k):v for k,v in prior[1].items()}],sort_keys=True)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];story=[B.load(p,300) for _,p in picked]
    def at(d):
        s=H.Replay()
        for w in d[:16]:s.observe(w)
        q,k=branches(ref,small,large,sem,field,s,cal,oldprior);l=H.Mixture(q,prior[1].get(k,prior[0]))
        return l.rank(),[l.complete(w) for w in ("a","the","unseenword")]
    assert at(story[0])==at(story[0][:16]+["future","suffix"])
    out={"semantic":sem.report(),"inherited_pilot":pilot,"field_pilot":train,"known_word_cal_before":pre,"known_word_cal_after":post,"movement":movement,"gradient_check":gradcheck,"recurrent":diagnostics,"parent_sha256":field.v26_sha,"reference_ids":[i for i,p in tr+extra],"calibration_ids":[i for i,p in calbooks],"gate":json.loads(frozen),"suffix_invariance":True,"fresh":{}}
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
    field.save(a.out.with_name("symbiotic-word-field.npz"))
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
