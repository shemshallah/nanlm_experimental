"""v47: matched finite tilt vs candidate/complement allocation through v46 solver."""
from pilot46 import *
from partition_credit import Budget,allocate,AllocationLaw,event_probs
K47=160

class PartitionTrial(Trial):
    def root_data47(self,root,wrong=None,scout=True):
        base,words,row,prior=self.root_data46(root,wrong);x,trace,h,m,c,b,code=row
        original=tuple(words)
        if scout and c.sum()>0:
            if not hasattr(self,"scout_vectors"):
                g=self.graph();self.scout_vectors=np.stack([g.vec(w) for w in self.field.sem.words])
            n=c.sum((1,2));res=(m.sum((1,2))/np.maximum(n[:,None],1)-b)*(n/(n+4))[:,None]
            score=self.scout_vectors@res.sum(0)
            order=np.argsort(-score,kind="stable");seen=set(words);added=[]
            for i in order:
                w=self.field.sem.words[int(i)]
                if w not in seen and w.isascii() and w.isalpha() and w.islower():
                    added.append(w);seen.add(w)
                if len(added)==32:break
            words=sorted(seen);assert len(words)<=K47
            ids=np.asarray([self.field.sem.ids.get(w,0) for w in words],np.int64)
            code=self.field.p["word_code"][ids].copy()
            prior=np.asarray([base.complete(w) for w in words],np.float64)
        return base,words,(x,trace,h,m,c,b,code),prior,original
    def data47(self,files,seed,count):
        rows=[];audit=C.Counter();checks=[]
        for root,event in self.raw_rows(files,seed,count):
            base,words,row,p,original=self.root_data47(root);k=len(words);x,trace,h,m,c,b,code=row
            donor=FE.Event(base,self.backoff.prob,root,self.surface,0.)
            member,bm,s,raw,folded,n=factors(event,words,base,donor)
            if len(checks)<4:checks.append(self.check46(row))
            if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                w=FE.folded(event[:-1]);audit["words"]+=1;audit["original"]+=w in original;audit["expanded"]+=w in words
            rows.append((x,trace,h,m,c,b,np.pad(code,((0,K47-k),(0,0))),np.pad(p,(0,K47-k)),np.pad(member,(0,K47-k)),bm,s,raw,folded,n))
        assert rows
        ds=tuple(torch.as_tensor(np.asarray(a),dtype=torch.int64 if i==0 else torch.float32 if i<7 else torch.float64) for i,a in enumerate(zip(*rows)))
        return ds,{"candidate_audit":dict(audit),"parity_checks":checks}
    def distrib(self,name,lift,p,scale):
        if name=="partition":
            a,b=self.budget(p,p>0,lift);return allocate(p,p>0,lift,a,b,scale)
        if scale==0:return p,torch.ones_like(p.sum(-1))
        d=p*torch.exp(lift.double()*scale);den=1+(d-p).sum(-1)
        return d/den[:,None],1/den
    def score47(self,name,ds,scale,weight):
        with torch.no_grad():
            z=torch.cat([self.models47[name](*(a[i:i+8] for a in ds[:7])) for i in range(0,len(ds[-1]),8)])
            q,c=self.distrib(name,z,ds[7],scale)
            literal,folded=event_probs(q,c,*ds[7:-1],weight)
            return {"raw_bpb":float(-literal.log2().sum()/ds[-1].sum()),"folded_nll":float(-folded.log().mean()),"candidate_mass":float(q.sum(-1).mean()),"events":len(literal)}
    def fit47(self,fitbooks,selectbooks):
        torch.manual_seed(4702);self.model=Readout46(self.field)
        tr,checks=self.data47(fitbooks,4703,12);va,vchecks=self.data47(selectbooks,4704,6)
        weight=self.field.bridge_weight or .25
        self.models47={};self.opts47={};self.scales47={};report={}
        for name in ("tilt","partition"):
            torch.manual_seed(4702);model=Readout46(self.field);self.models47[name]=model
            params=list(model.parameters())
            if name=="partition":self.budget=Budget();params+=list(self.budget.parameters())
            opt=torch.optim.Adam(params,lr=.015,weight_decay=.002);self.opts47[name]=opt
            rng=np.random.default_rng(4705);trace=[]
            for i in range(128):
                ix=rng.integers(len(tr[-1]),size=8);a=tuple(t[ix] for t in tr)
                opt.zero_grad();lift=model(*a[:7]);q,c=self.distrib(name,lift,a[7],1.)
                literal,folded=event_probs(q,c,*a[7:-1],weight)
                loss=-literal.log().sum()/a[-1].sum();loss.backward();opt.step()
                if i==0 or (i+1)%16==0:trace.append([i+1,float(loss.detach())])
            scores={str(s):self.score47(name,va,s,weight) for s in (0.,.125,.25,.5,1.)}
            self.scales47[name]=min((0.,.125,.25,.5,1.),key=lambda s:scores[str(s)]["raw_bpb"])
            report[name]={"scale":self.scales47[name],"selection":scores,"trace":trace}
        return weight,{"arms":report,"train":checks,"selection":vchecks,"shared_event_weight":weight}
    def law47(self,name,base,words,row,p,scale):
        if not scale:return base
        args=tuple(torch.as_tensor(a[None],dtype=torch.int64 if i==0 else torch.float32) for i,a in enumerate(row))
        with torch.no_grad():
            z=self.models47[name](*args);q,c=self.distrib(name,z,torch.as_tensor(p[None],dtype=torch.float64),scale)
        return AllocationLaw(base,words,q[0].numpy(),float(c[0]))
    def root_laws(self,root):
        base,words,row,p,old=self.root_data47(root)
        tilt=self.law47("tilt",base,words,row,p,self.scales47["tilt"])
        part=self.law47("partition",base,words,row,p,self.scales47["partition"])
        keep=np.asarray([w in old for w in words],bool);oldrow=(*row[:-1],row[-1][keep])
        restricted=self.law47("partition",base,[w for w,k in zip(words,keep) if k],oldrow,p[keep],self.scales47["partition"])
        return {"parent":base,"tilt":tilt,"adaptive":part,"partition_no_scout":restricted}
    def raw_eval(self,files,weight):
        out=Adaptive.raw_eval(self,files,weight);out["modes"]["partition"]=out["modes"].pop("adaptive")
        counts=C.Counter();bits=C.Counter();n=0
        rr=[self.raw_rows([(bid,p)],4404,4) for bid,p in files]
        for j,rows in enumerate(rr):
            other=rr[(j+1)%len(rr)]
            for i,(root,event) in enumerate(rows):
                base,words,row,p,old=self.root_data47(root)
                wrong=other[i%len(other)][0];_,ww,wr,wp,_=self.root_data47(root,wrong)
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    w=FE.folded(event[:-1]);counts["words"]+=1;counts["original"]+=w in old;counts["expanded"]+=w in words
                for name,law in (("true",self.law47("partition",base,words,row,p,self.scales47["partition"])),("wrong",self.law47("partition",base,ww,wr,wp,self.scales47["partition"])),("true_forced",self.law47("partition",base,words,row,p,1.)),("wrong_forced",self.law47("partition",base,ww,wr,wp,1.))):
                    ev=FE.Event(law,self.backoff.prob,root,self.surface,weight);bits[name]-=math.log2(max(ev.mass(event),1e-300))
                n+=len(event)
        out["candidate_audit"]=dict(counts);out["counterfactual"]={"bytes":n,"bpb":{k:v/n for k,v in bits.items()}}
        return out
    def save47(self,path,parent,weight):
        torch.save({"parent_sha256":parent,"event_weight":weight,"scales":self.scales47,"models":{k:v.state_dict() for k,v in self.models47.items()},"budget":self.budget.state_dict(),"optimizers":{k:v.state_dict() for k,v in self.opts47.items()},"updates":128},path)

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor","unified"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("partition-event-metrics.json"));a=ap.parse_args();r=a.data
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
    exp=PartitionTrial(field,ref,small,large,cal)
    def digest():
        out={k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
        out.update({"lexical:"+k:hashlib.sha256(getattr(field,k).tobytes()).hexdigest() for k in ("lex","lm","lv","lt","proj")})
        return out
    initial=digest();steps=field.steps.copy();start_step=field.step;rng_state=json.dumps(field.rng.bit_generator.state,sort_keys=True)
    fitbooks=extra[12::24];selectbooks=calbooks
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    rawdocs=[first(p) for _,p in tr];exp.backoff=A.Backoff(rawdocs);exp.surface=FE.Surface(rawdocs)
    event_weight,fit=exp.fit47(fitbooks,selectbooks)
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
    out["inherited_unchanged"]=True;out["fit_ids"]=[i for i,p in fitbooks];out["selection_ids"]=[i for i,p in selectbooks];out["dev_ids"]=[i for i,p in picked];out["windows"]=list(WINDOWS)
    write_metrics(a.out,out,'evaluated')
    exp.save47(a.out.with_suffix(".pt"),field.v37_sha,event_weight)
    write_metrics(a.out,out)
if __name__=="__main__":main()
