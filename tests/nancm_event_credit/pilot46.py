"""v46: frozen v37 recurrent kernel, persistent memory anchor, exact raw-event
credit. No separate recurrent implementation. New optimizer groups only."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'nancm_retention'))
from pilot44 import *
from metric_io import write_metrics
import hybrid38 as P38
from persistent_triangle import TriangleAnchor
from event_credit import factors,probabilities,objective

class PersistentField(P38.Portable):
    def anchored_hidden(self,x,trace,residual,counts,anchor,scale):
        assert getattr(self,"edge_anchor",None) is None and self.stream_override is None
        self.stream_override=trace
        if scale:
            self.edge_anchor=lambda s,z,pairs:anchor(s,z,residual.to(s.device),counts.to(s.device),pairs,scale)
        try:
            _,_,_,tape=self.tape(np.asarray(x,dtype=np.uint32))
            return tape[1],float(tape[4])
        finally:
            self.edge_anchor=None;self.stream_override=None

class Readout46(torch.nn.Module):
    def __init__(self,field):
        super().__init__();self.retention=Retention()
        self.anchor=TriangleAnchor(triangles=len(field.topology[0]))
        self.field=field
    def forward(self,x,trace,hidden,m,c,b,code,scale=1.):
        if not scale:return torch.zeros(code.shape[:2],dtype=code.dtype,device=code.device)
        residual,n=self.retention(m,c,b)
        out,solver_residual=self.field.anchored_hidden(x.detach().numpy(),trace,residual,n,self.anchor,scale)
        self.last_residual=solver_residual
        return self.anchor.lift(out-hidden,code)

class Trial(Adaptive):
    def root_data46(self,root,wrong=None):
        state=self.field.session();g=self.graph()
        def observed(raw):
            words=re.findall(rb'[A-Za-z]+',raw)
            if raw and FE.IS_LETTER[raw[-1]]:words=words[:-1]
            return [w.lower().decode('ascii') for w in words]
        for w in observed(root):state.observe(w);g.observe(w)
        if wrong is not None:
            g=self.graph()
            for w in observed(wrong):g.observe(w)
        with torch.no_grad():
            orig,_=V.branches(self.ref,self.small,self.large,state,self.cal);ex=orig[3];f,h=self.field.field_law(state,orig[1].law)
            mem=state.attractor.query();base,_=self.field.combine(f,h,mem)
        ep=heapq.nlargest(24,ex.law.s,key=ex.law.s.get)
        fw=[self.field.sem.words[k] for k in np.argsort(-f.d,kind='stable')[:24]]
        mw={v for v,ok in zip(mem[3],mem[2]) if ok}
        words=sorted(set(ep)|set(fw)|mw|set(g.propose(state.h)));assert 0<len(words)<=MAX
        x=np.asarray(self.field.features(state.h)+self.field.events(state.h)+self.field.ids(state.h),np.uint32)
        trace=state.trace.detach().numpy()[0].copy()
        m,c,b=g.basis(state.h)
        ids=np.asarray([self.field.sem.ids.get(w,0) for w in words],np.int64)
        code=self.field.p["word_code"][ids].copy()
        prior=np.asarray([base.complete(w) for w in words],np.float64)
        assert prior.sum()<=1+1e-8
        return base,words,(x,trace,h,m,c,b,code),prior
    def check46(self,row):
        x,trace,h,m,c,b,code=(torch.as_tensor(a[None]) for a in row)
        with torch.no_grad():
            out,_=self.field.anchored_hidden(x.numpy(),trace,torch.zeros(1,8,128),torch.zeros(1,8),self.model.anchor,0.)
            legacy=out.numpy().copy()
            assert np.array_equal(legacy,h.numpy())
            empty,_=self.field.anchored_hidden(x.numpy(),trace,torch.zeros(1,8,128),torch.zeros(1,8),self.model.anchor,1.)
            assert np.array_equal(legacy,empty.numpy())
            self.field.configure("legacy")
            try:old,_=self.field.anchored_hidden(x.numpy(),trace,torch.zeros(1,8,128),torch.zeros(1,8),self.model.anchor,0.)
            finally:self.field.configure("cpu")
            e=float(np.max(np.abs(legacy-old.numpy())));assert np.allclose(legacy,old.numpy(),rtol=1e-6,atol=1e-6)
            assert torch.count_nonzero(self.model(x,trace,h,m,c,b,code,0.))==0
        return {"zero_lift_exact":True,"empty_full_field_exact":True,"legacy_portable_hidden_max_error":e}
    def data46(self,files,seed,count):
        rows=[];checks=[];max_factor_error=0.
        for root,event in self.raw_rows(files,seed,count):
            base,words,row,prior=self.root_data46(root);k=len(words);x,trace,h,m,c,b,code=row
            donor=FE.Event(base,self.backoff.prob,root,self.surface,0.)
            member,bm,s,raw,folded,n=factors(event,words,base,donor)
            if len(checks)<4:
                checks.append(self.check46(row))
                z=np.linspace(-.4,.4,k);law=SL.Transport(base,words,z);e=FE.Event(law,self.backoff.prob,root,self.surface,.25)
                literal,_=probabilities(*(torch.as_tensor(np.asarray(a)[None],dtype=torch.float64) for a in (z,prior,member,bm,s,raw,folded)),.25)
                err=abs(float(literal[0])-e.mass(event));max_factor_error=max(max_factor_error,err)
                assert math.isclose(float(literal[0]),e.mass(event),rel_tol=1e-8,abs_tol=1e-25)
            rows.append((x,trace,h,m,c,b,np.pad(code,((0,MAX-k),(0,0))),np.pad(prior,(0,MAX-k)),np.pad(member,(0,MAX-k)),bm,s,raw,folded,n))
        assert rows
        ds=tuple(torch.as_tensor(np.asarray(a),dtype=torch.int64 if i==0 else torch.float32 if i<7 else torch.float64) for i,a in enumerate(zip(*rows)))
        return ds,{"full_field_checks":checks,"event_factor_max_error":max_factor_error}
    def score46(self,ds,scale,weight):
        with torch.no_grad():
            z=torch.cat([self.model(*(a[i:i+8] for a in ds[:7]),scale=scale) for i in range(0,len(ds[-1]),8)])
            literal,folded=probabilities(z,*ds[7:-1],weight)
            return {"raw_bpb":float(-literal.log2().sum()/ds[-1].sum()),"folded_nll":float(-folded.log().mean()),"events":len(ds[-1]),"bytes":int(ds[-1].sum())}
    def fit46(self,fitbooks,selectbooks):
        torch.manual_seed(4602);self.model=Readout46(self.field);self.opt=torch.optim.Adam(self.model.parameters(),lr=.015,weight_decay=.002)
        tr,checks=self.data46(fitbooks,4603,12);va,vchecks=self.data46(selectbooks,4604,6)
        w=float(self.field.bridge_weight);train_weight=w if w>0 else .25
        rng=np.random.default_rng(4605);log=[];norms={};max_residual=0.
        baseline=self.score46(va,0.,train_weight)
        for i in range(128):
            ix=rng.integers(len(tr[-1]),size=8);a=tuple(t[ix] for t in tr)
            self.opt.zero_grad();lift=self.model(*a[:7]);loss=objective(lift,*a[7:],train_weight)
            loss.backward();self.opt.step()
            max_residual=max(max_residual,self.model.last_residual)
            for k,p in self.model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all();norms[k]=max(norms.get(k,0.),float(p.grad.norm()))
            if i==0 or (i+1)%16==0:log.append([i+1,float(loss.detach())])
        grid=(0.,.125,.25,.5,1.);scores={str(s):self.score46(va,s,train_weight) for s in grid}
        self.scale46=min(grid,key=lambda s:scores[str(s)]["raw_bpb"])
        return train_weight,{"train_events":len(tr[-1]),"selection_events":len(va[-1]),"weight":train_weight,"forced_nonzero_weight":w==0,"baseline":baseline,"selection":scores,"scale":self.scale46,"trace":log,"gradient_max_norms":norms,"maximum_observed_solver_residual":max_residual,"train_checks":checks,"validation_checks":vchecks}
    def law46(self,base,words,row,scale):
        if not scale:return base
        args=tuple(torch.as_tensor(a[None],dtype=torch.int64 if i==0 else torch.float32) for i,a in enumerate(row))
        with torch.no_grad():lift=self.model(*args,scale=scale)[0].numpy().astype(np.float64)
        return SL.Transport(base,words,lift)
    def root_laws(self,root):
        base,words,row,_=self.root_data46(root)
        # 'adaptive' key also serves inherited codec, 'forced' exposes learning if calibration rejects it.
        return {"parent":base,"adaptive":self.law46(base,words,row,self.scale46),"forced":self.law46(base,words,row,1.)}
    def raw_eval(self,files,weight):
        out=super().raw_eval(files,weight)
        # A separate matched wrong-graph test; same raw target and true inherited base.
        records=[]
        for bid,p in files:
            rr=self.raw_rows([(bid,p)],4606,4)
            records.append(rr)
        bits=C.Counter();bytes_=0
        for j,rr in enumerate(records):
            other=records[(j+1)%len(records)]
            for i,(root,event) in enumerate(rr):
                wrong=other[i%len(other)][0];base,words,row,_=self.root_data46(root)
                wb,ww,wr,_=self.root_data46(root,wrong)
                assert abs(wb.complete("the")-base.complete("the"))<1e-12
                for name,law in (("parent",base),("true",self.law46(base,words,row,self.scale46)),("wrong",self.law46(base,ww,wr,self.scale46)),("true_forced",self.law46(base,words,row,1.)),("wrong_forced",self.law46(base,ww,wr,1.))):
                    ev=FE.Event(law,self.backoff.prob,root,self.surface,weight)
                    bits[name]-=math.log2(max(ev.mass(event),1e-300))
                bytes_+=len(event)
        out["counterfactual"]={"bytes":bytes_,"bpb":{k:v/bytes_ for k,v in bits.items()}}
        return out
    def save46(self,path,parent,weight):
        torch.save({"parent_sha256":parent,"scale":self.scale46,"event_weight":weight,"model":self.model.state_dict(),"optimizer":self.opt.state_dict(),"updates":128},path)

def main():
    from fit_split import cohorts
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor","unified"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("triangle-event-metrics.json"));a=ap.parse_args();r=a.data
    torch.set_num_threads(4)
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
    field=PersistentField(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,a.attractor,a.unified,len(data[1]))
    field.configure("cpu");field.bind(ref,small,large,cal)
    exp=Trial(field,ref,small,large,cal)
    def digest():
        out={k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
        out.update({"lexical:"+k:hashlib.sha256(getattr(field,k).tobytes()).hexdigest() for k in ("lex","lm","lv","lt","proj")})
        return out
    initial=digest();steps=field.steps.copy();start_step=field.step;rng_state=json.dumps(field.rng.bit_generator.state,sort_keys=True)
    primary,fitbooks,selectbooks,split_report=cohorts(books,tr,calbooks,extra)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    rawdocs=[first(p) for _,p in tr];exp.backoff=A.Backoff(rawdocs);exp.surface=FE.Surface(rawdocs)
    event_weight,fit=exp.fit46(fitbooks,selectbooks)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    out={"parent_sha256":field.v37_sha,"fit":fit,"fresh":{}}
    write_metrics(a.out,out,"calibration")
    out["matched_story_raw"]=exp.raw_eval(picked,event_weight);write_metrics(a.out,out,"stories")
    for name,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used or i in forbidden:continue
            if len(B.load(p,200))<112:continue
            chosen.append((i,p));used.add(i)
            if len(chosen)==6:break
        assert len(chosen)==6
        out["fresh"][name]={"ids":[i for i,p in chosen],"raw":exp.raw_eval(chosen,event_weight)}
        write_metrics(a.out,out,"domain:"+name)
    raw=first(picked[0][1]);out["codec"]=exp.codec(raw[:128],raw[128:144],event_weight)
    assert steps==field.steps and initial==digest() and field.step==start_step and json.dumps(field.rng.bit_generator.state,sort_keys=True)==rng_state
    out["split_integrity"]=split_report
    out["inherited_unchanged"]=True;out["fit_ids"]=[i for i,p in fitbooks];out["selection_ids"]=[i for i,p in selectbooks];out["dev_ids"]=[i for i,p in picked];out["windows"]=list(WINDOWS)
    write_metrics(a.out,out,'evaluated')
    exp.save46(a.out.with_suffix(".pt"),field.v37_sha,event_weight)
    write_metrics(a.out,out)
if __name__=="__main__":main()
