"""v45 exact deployed raw-event credit: continue learned v44 optimizer, not reset.
Compare folded-only vs literal NLL with same checkpoint, examples and updates.
"""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'nancm_retention'))
from pilot44 import *
import event_credit as EC
import copy

class JointEvent(Adaptive):
    def root_data(self,root):
        state=self.field.session();g=self.graph();words=re.findall(rb'[A-Za-z]+',root)
        if root and FE.IS_LETTER[root[-1]]:words=words[:-1]
        for w in words:
            w=w.lower().decode('ascii');state.observe(w);g.observe(w)
        orig,_=V.branches(self.ref,self.small,self.large,state,self.cal);ex=orig[3];f,h=self.field.field_law(state,orig[1].law)
        base=H.Mixture((ex,f),(1-self.mix,self.mix));mem=state.attractor.query()
        ep=heapq.nlargest(24,ex.law.s,key=ex.law.s.get);fw=[self.field.sem.words[k] for k in np.argsort(-f.d,kind='stable')[:24]]
        mw={v for v,ok in zip(mem[3],mem[2]) if ok};row=self.features(base,ex,f,set(ep)|set(fw)|mw,g,state.h,mw)
        return base,row
    def event_data(self,files,seed,count):
        rows=[];checked=0
        for root,event in self.raw_rows(files,seed,count):
            law,row=self.root_data(root);words,x,p,m,c,b,v=row;k=len(p)
            donor=FE.Event(law,self.backoff.prob,root,self.surface,0.)
            member,base,surface,raw,folded,n=EC.factors(event,words,law,donor)
            # Verify training likelihood factors against the actual deployment implementation.
            if checked<16:
                z=np.linspace(-.7,.7,k);testlaw=SL.Transport(law,words,z)
                e=FE.Event(testlaw,self.backoff.prob,root,self.surface,.25)
                l,f=EC.probabilities(*(torch.as_tensor(np.asarray(a)[None],dtype=torch.float64) for a in (z,p,member,base,surface,raw,folded)),.25)
                assert math.isclose(float(l[0]),e.mass(event),rel_tol=1e-8,abs_tol=1e-25)
                checked+=1
            rows.append((np.pad(x,((0,MAX-k),(0,0))),m,c,b,np.pad(v,((0,MAX-k),(0,0))),np.pad(p,(0,MAX-k)),np.pad(member,(0,MAX-k)),base,surface,raw,folded,n))
        assert rows;return tuple(torch.as_tensor(np.asarray(a),dtype=torch.float32 if i<5 else torch.float64) for i,a in enumerate(zip(*rows)))
    def scores(self,model,ds,weight,scale):
        with torch.no_grad():
            z=torch.cat([model(*(a[i:i+16] for a in ds[:5]),adaptive=True) for i in range(0,len(ds[-1]),16)])*scale
            literal,folded=EC.probabilities(z,*ds[5:-1],weight)
            return {"raw_bpb":float(-literal.log2().sum()/ds[-1].sum()),"folded_nll":float(-folded.log().mean()),"events":len(ds[-1]),"bytes":int(ds[-1].sum())}
    def resume(self,path,metrics,fitbooks,selectbooks):
        ck=torch.load(path,map_location="cpu",weights_only=True);m=json.loads(metrics.read_text())
        assert ck["parent_sha256"]==self.field.v36_sha==m["parent_sha256"] and ck["steps"]==384 and m["inherited_unchanged"]
        assert set(ck["models"])==set(ck["optimizers"])=={"full","exact"}
        assert abs(float(ck["mix"])-float(m["fit"]["mix"]))<1e-12
        assert float(ck["scale"]["full"])==float(m["fit"]["controls"]["adaptive"]["scale"])
        self.mix=float(ck["mix"]);self.retention_sha=hashlib.sha256(path.read_bytes()).hexdigest();self.retention_metrics_sha=hashlib.sha256(metrics.read_bytes()).hexdigest()
        self.models={};self.optimizers={};self.scale={"parent":float(ck["scale"]["full"])}
        parent=Readout();parent.load_state_dict(ck["models"]["full"]);self.models["parent"]=parent
        opt0=torch.optim.Adam(parent.parameters(),lr=.015);opt0.load_state_dict(ck["optimizers"]["full"]);self.optimizers["parent"]=opt0
        self.parent_optimizer_steps={k:int(opt0.state[v]["step"]) for k,v in parent.named_parameters() if v in opt0.state and "step" in opt0.state[v]}
        assert self.parent_optimizer_steps and all(v==384 for v in self.parent_optimizer_steps.values())
        parent_weight=float(m["raw_fit"]["weight"]);weight=parent_weight if parent_weight>0 else .25
        train_scale=max(.25,self.scale["parent"])
        ds=self.event_data(fitbooks,4504,24);dv=self.event_data(selectbooks,4505,8)
        result={"parent_weight":parent_weight,"shared_diagnostic_weight":weight,"forced_nonzero_weight":parent_weight==0,"training_scale":train_scale,"fit_events":len(ds[-1]),"selection_events":len(dv[-1]),"parent":self.scores(parent,dv,weight,self.scale["parent"]),"arms":{}}
        for name,spelling in (("folded_only",0.),("adaptive",1.)):
            model=copy.deepcopy(parent);opt=torch.optim.Adam(model.parameters(),lr=.015)
            opt.load_state_dict(copy.deepcopy(ck["optimizers"]["full"]));rng=np.random.default_rng(4506);trace=[]
            for i in range(192):
                ix=rng.integers(len(ds[-1]),size=16);a=tuple(t[ix] for t in ds)
                opt.zero_grad();lift=model(*a[:5],adaptive=True)*train_scale
                loss=EC.objective(lift,*a[5:],weight,spelling);loss.backward();opt.step()
                if i==0 or (i+1)%32==0:trace.append([i+1,float(loss.detach())])
            grid=(0.,.125,.25,.5,1.);scores={str(s):self.scores(model,dv,weight,s) for s in grid}
            scale=min(grid,key=lambda s:scores[str(s)]["raw_bpb"]);self.models[name]=model;self.optimizers[name]=opt;self.scale[name]=scale
            final_steps={k:int(opt.state[v]["step"]) for k,v in model.named_parameters() if v in opt.state and "step" in opt.state[v]}
            assert final_steps=={k:v+192 for k,v in self.parent_optimizer_steps.items()}
            result["arms"][name]={"scale":scale,"calibration":scores,"trace":trace,"optimizer_steps":final_steps}
        assert hashlib.sha256(path.read_bytes()).hexdigest()==self.retention_sha
        return weight,result
    def root_laws(self,root):
        base,row=self.root_data(root);words,x,p,m,c,b,v=row;out={"consensus":base}
        args=tuple(torch.as_tensor(a[None],dtype=torch.float32) for a in (x,m,c,b,v))
        for name,model in self.models.items():
            scale=self.scale[name]
            with torch.no_grad():lift=model(*args,adaptive=True)[0].numpy().astype(np.float64)
            out[name]=SL.Transport(base,words,lift*scale) if scale else base
        return out
    def save45(self,path,parent,weight):
        torch.save({"parent_sha256":parent,"retention_sha256":self.retention_sha,"retention_metrics_sha256":self.retention_metrics_sha,"mix":self.mix,"scale":self.scale,"event_weight":weight,"parent_optimizer_steps":self.parent_optimizer_steps,"updates_added":192,"models":{k:v.state_dict() for k,v in self.models.items()},"optimizers":{k:v.state_dict() for k,v in self.optimizers.items()}},path)

def main():
    from fit_split import cohorts
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--retention",type=pathlib.Path,required=True);ap.add_argument("--retention-metrics",type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("joint-event-metrics.json"));a=ap.parse_args();r=a.data
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
    field=R6.MemoryField(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,len(data[1]))
    U37.load36(field,a.attractor);field.bind(ref,small,large,cal)
    exp=JointEvent(field,ref,small,large,cal)
    digest=lambda: {k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
    initial=digest();steps=field.steps.copy()
    primary,fitbooks,selectbooks,split_report=cohorts(books,tr,calbooks,extra)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    rawdocs=[first(p) for _,p in tr];exp.backoff=A.Backoff(rawdocs);exp.surface=FE.Surface(rawdocs)
    parent_report=json.loads(a.retention_metrics.read_text())
    assert parent_report.get('split_integrity',{}).get('unindexed_fit') is True, 'Refit v44 without indexed fitting targets before continuation'
    assert set(parent_report['fit_ids'])=={i for i,p in primary}
    event_weight,fit=exp.resume(a.retention,a.retention_metrics,fitbooks,selectbooks)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    out={"parent_sha256":field.v36_sha,"retention_sha256":exp.retention_sha,"fit":fit,"matched_story_raw":exp.raw_eval(picked,event_weight),"fresh":{}}
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
    assert steps==field.steps and initial==digest()
    out["split_integrity"]=split_report
    out["inherited_unchanged"]=True;out["fit_ids"]=[i for i,p in fitbooks];out["selection_ids"]=[i for i,p in selectbooks];out["dev_ids"]=[i for i,p in picked];out["windows"]=list(WINDOWS)
    write_metrics(a.out,out,'evaluated')
    exp.save45(a.out.with_suffix(".pt"),field.v36_sha,event_weight)
    write_metrics(a.out,out)
if __name__=="__main__":main()
