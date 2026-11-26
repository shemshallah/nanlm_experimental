"""v49: v48 scout with inherited codebook, full vocabulary and exact raw credit.
Frozen v37 model and open word law. Train only new scout/retention parameters.
"""
from pilot46 import *
sys.path.insert(0,str(root/'nancm_scout'))
from dense_scout import DenseScout,fuse,interval,event_prob,event_loss

class DenseTrial(Trial):
    def __init__(self,*args):
        super().__init__(*args)
        self.words=self.field.sem.words
        assert self.words==sorted(set(self.words)) and all(w.isascii() and w.isalpha() and w.islower() for w in self.words)
        g=self.graph();self.code=4*np.stack([g.vec(w) for w in self.words]).astype(np.float32)
        freq=self.field.sem.freq[1:]+.1
        self.log_prior=.25*np.log(freq/freq.sum())
    def observed(self,raw):
        words=re.findall(rb'[A-Za-z]+',raw)
        if raw and FE.IS_LETTER[raw[-1]]:words=words[:-1]
        return [w.lower().decode('ascii') for w in words]
    def context49(self,g,words,h):
        rr=np.zeros((8,128),np.float32)
        for i in range(4):
            if len(words)>i:rr[i]=4*g.vec(words[-1-i])
        for i,(a,b) in enumerate(((0,8),(8,32),(32,128),(128,512)),4):
            if len(words)>a:
                tail=words[max(0,len(words)-b):len(words)-a]
                rr[i]=4*np.mean([g.vec(w) for w in tail],axis=0)
        m,c,b=g.basis(h)
        return rr,m,c,b
    def root49(self,root,wrong=None):
        words=self.observed(root);state=self.field.session();g=self.graph()
        for w in words:state.observe(w);g.observe(w)
        with torch.no_grad():
            orig,_=V.branches(self.ref,self.small,self.large,state,self.cal)
            f,h=self.field.field_law(state,orig[1].law);base,_=self.field.combine(f,h,state.attractor.query())
        if wrong is not None:
            g=self.graph();words=self.observed(wrong)
            for w in words:g.observe(w)
        return base,(*self.context49(g,words,state.h),h)
    def data49(self,files,seed,count):
        rows=[];audit=C.Counter()
        for root,event in self.raw_rows(files,seed,count):
            base,row=self.root49(root);donor=FE.Event(base,self.backoff.prob,root,self.surface,0.)
            _,bm,s,raw,folded,n=factors(event,[],base,donor)
            capped=bool(FE.IS_LETTER[event[-1]]);letters=event if capped else event[:-1]
            lo,hi=interval(self.words,FE.folded(letters),capped) if letters else (0,0)
            audit["events"]+=1;audit["empty"]+=not bool(letters);audit["capped"]+=capped
            if letters and not capped:audit["words"]+=1;audit["known_words"]+=hi>lo
            rows.append((*row,lo,hi,bm,s,raw,folded,n))
        assert rows
        return tuple(torch.as_tensor(np.asarray(a),dtype=torch.float32 if i<5 else torch.int64 if i<7 else torch.float64) for i,a in enumerate(zip(*rows))),dict(audit)
    def score49(self,model,ds,memory,strength,weight):
        with torch.no_grad():
            z=torch.cat([model(*(a[i:i+8] for a in ds[:5]),memory=memory) for i in range(0,len(ds[-1]),8)])
            lit,fold=event_prob(z,*ds[5:-1],weight,strength)
            return {"raw_bpb":float(-lit.log2().sum()/ds[-1].sum()),"folded_nll":float(-fold.log().mean()),"events":len(lit),"bytes":int(ds[-1].sum())}
    def fit49(self,fitbooks,selectbooks):
        tr,ta=self.data49(fitbooks,4903,48);va,sa=self.data49(selectbooks,4904,16)
        weight=self.field.bridge_weight or .25;report={}
        self.models49={};self.opts49={};self.scales49={}
        for name,memory in (("context_only",False),("adaptive",True)):
            torch.manual_seed(4902)
            model=DenseScout(self.code,self.log_prior);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            self.models49[name]=model;self.opts49[name]=opt;rng=np.random.default_rng(4905);trace=[];norms={}
            before=self.score49(model,va,memory,0.,weight)
            for i in range(192):
                ix=rng.integers(len(tr[-1]),size=8);a=tuple(t[ix] for t in tr)
                opt.zero_grad();z=model(*a[:5],memory=memory)
                loss=event_loss(z,*a[5:],weight,.5);loss.backward()
                for k,p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all();norms[k]=max(norms.get(k,0.),float(p.grad.norm()))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                if i==0 or (i+1)%32==0:trace.append([i+1,float(loss.detach())])
            grid=(0.,.125,.25,.5,.75);scores={str(s):self.score49(model,va,memory,s,weight) for s in grid}
            self.scales49[name]=min(grid,key=lambda s:scores[str(s)]["raw_bpb"])
            report[name]={"parent":before,"selection":scores,"scale":self.scales49[name],"trace":trace,"gradient_max_norms":norms}
        return weight,{"arms":report,"fit_audit":ta,"selection_audit":sa,"vocabulary":len(self.words),"parameters":sum(p.numel() for p in self.models49["adaptive"].parameters()),"shared_event_weight":weight,"forced_nonzero_event_weight":self.field.bridge_weight==0}
    def logits49(self,name,row):
        args=tuple(torch.as_tensor(a[None],dtype=torch.float32) for a in row)
        with torch.no_grad():return self.models49[name](*args,memory=name=="adaptive")[0].double()
    def law49(self,name,base,row,strength=None):
        if strength is None:strength=self.scales49[name]
        if not strength:return base
        return fuse(self.words,self.logits49(name,row).softmax(0).numpy(),base,strength)
    def root_laws(self,root):
        base,row=self.root49(root)
        return {"parent":base,"context_only":self.law49("context_only",base,row),"adaptive":self.law49("adaptive",base,row),"forced_dense":self.law49("adaptive",base,row,.5)}
    def raw_eval(self,files,weight):
        out=Adaptive.raw_eval(self,files,weight);audit=C.Counter();bits=C.Counter();n=0
        rr=[self.raw_rows([(bid,p)],4404,4) for bid,p in files]
        for j,rows in enumerate(rr):
            other=rr[(j+1)%len(rr)]
            for i,(root,event) in enumerate(rows):
                base,row=self.root49(root);_,wrong=self.root49(root,other[i%len(other)][0])
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    target=FE.folded(event[:-1]);rank,cert,_=H.Mixture((base,),(1.,)).rank(k=32,budget=2000)
                    logits=self.logits49("adaptive",row).numpy();dense=[self.words[k] for k in np.argsort(-logits,kind="stable")[:32]]
                    audit["words"]+=1;audit["parent32"]+=target in rank;audit["parent32_cert"]+=cert
                    audit["dense32"]+=target in dense;audit["union64"]+=target in rank or target in dense
                    audit["new_target"]+=target not in rank and target in dense
                for name,law in (("parent",base),("true",self.law49("adaptive",base,row)),("wrong_context",self.law49("adaptive",base,wrong)),("true_forced",self.law49("adaptive",base,row,.5)),("wrong_forced",self.law49("adaptive",base,wrong,.5))):
                    e=FE.Event(law,self.backoff.prob,root,self.surface,weight);bits[name]-=math.log2(max(e.mass(event),1e-300))
                n+=len(event)
        out["discovery_audit"]=dict(audit);out["counterfactual"]={"bytes":n,"bpb":{k:v/n for k,v in bits.items()}}
        return out
    def save49(self,path,parent,weight):
        torch.save({"parent_sha256":parent,"words":self.words,"event_weight":weight,"scales":self.scales49,"models":{k:v.state_dict() for k,v in self.models49.items()},"optimizers":{k:v.state_dict() for k,v in self.opts49.items()},"updates":192},path)

def main():
    from fit_split import cohorts
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor","unified"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("dense-scout-metrics.json"));a=ap.parse_args();r=a.data
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
    field.configure("legacy");field.bind(ref,small,large,cal)
    exp=DenseTrial(field,ref,small,large,cal)
    def digest():
        out={k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
        out.update({"lexical:"+k:hashlib.sha256(getattr(field,k).tobytes()).hexdigest() for k in ("lex","lm","lv","lt","proj")})
        return out
    initial=digest();steps=field.steps.copy();start_step=field.step;rng_state=json.dumps(field.rng.bit_generator.state,sort_keys=True)
    primary,fitbooks,selectbooks,split_report=cohorts(books,tr,calbooks,extra)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    rawdocs=[first(p) for _,p in tr];exp.backoff=A.Backoff(rawdocs);exp.surface=FE.Surface(rawdocs)
    event_weight,fit=exp.fit49(primary,selectbooks);fitbooks=primary
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    out={"parent_sha256":field.v37_sha,"fit":fit,"split_integrity":split_report,"fresh":{}}
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
    exp.save49(a.out.with_suffix(".pt"),field.v37_sha,event_weight)
    write_metrics(a.out,out)
if __name__=="__main__":main()
