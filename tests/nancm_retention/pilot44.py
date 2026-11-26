"""v44 differentiable memory retention, signed credit and folded event parity."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'nancm_folded'));sys.path.insert(0,str(root/'nancm_graph'))
from learned_pilot import *
from adaptive_retention import Retention
from causal_basis import AdaptiveGraph
import folded_event as FE
import re
class Readout(torch.nn.Module):
    def __init__(self):
        super().__init__();self.retention=Retention();self.credit=EdgeCredit()
    def forward(self,x,m,c,b,v,adaptive=True):
        r,n=self.retention(m,c,b,adaptive)
        return self.credit(x,r,v,n,True)
class Adaptive(Learned):
    def graph(self):return AdaptiveGraph(self.field.p["word_code"],self.field.sem.ids,self.field.sem.classof)
    def features(self,base,ex,field,common,g,h,mw):
        words,x,p=Experiment.features(self,base,ex,field,common,g,h,mw);x[:,15:23]=0
        m,c,b=g.basis(h);v=np.asarray([g.vec(w) for w in words],np.float32).reshape(-1,128)
        return words,x,p,m,c,b,v
    def dataset(self,examples):
        rows=[]
        for w,ex,f,parent,base,real,bad,*_ in examples:
            words,x,p,m,c,b,v=real;k=len(p)
            rows.append((np.pad(x,((0,MAX-k),(0,0))),m,c,b,np.pad(v,((0,MAX-k),(0,0))),np.pad(p,(0,MAX-k)),np.pad(np.asarray([s==w for s in words]),(0,MAX-k)),max(base.complete(w),1e-300)))
        assert rows
        return tuple(torch.as_tensor(np.asarray(a),dtype=torch.float32 if i<5 else torch.float64) for i,a in enumerate(zip(*rows)))
    def fit(self,fitdocs,validdocs):
        rows=list(self.examples(fitdocs,long=True));grid=(0.,.25,.5,.75,1.)
        scores={str(s):float(np.mean([-math.log(max((1-s)*ex.complete(w)+s*f.complete(w),1e-300)) for w,ex,f,*_ in rows])) for s in grid}
        self.mix=min(grid,key=lambda s:scores[str(s)]);del rows
        ds=self.dataset(self.examples(fitdocs,long=True));dv=self.dataset(self.examples(validdocs,long=True))
        self.models={};self.optimizers={};report={}
        for name,adaptive in (("full",True),("exact",False)):
            model=Readout();opt=torch.optim.Adam(model.parameters(),lr=.015,weight_decay=.005)
            rng=np.random.default_rng(4402);trace=[]
            for i in range(1,385):
                ix=rng.integers(len(ds[-1]),size=16);a=tuple(t[ix] for t in ds)
                opt.zero_grad();loss=likelihood(model(*a[:5],adaptive=adaptive),*a[5:]);loss.backward();opt.step()
                if i==1 or i%64==0:trace.append([i,float(loss.detach())])
            with torch.no_grad():
                z=torch.cat([model(*(a[i:i+16] for a in dv[:5]),adaptive=adaptive) for i in range(0,len(dv[-1]),16)])
                nll={str(s):float(likelihood(z,*dv[5:],scale=s)) for s in (0.,.125,.25,.5,1.)}
                weights=model.retention.weights(adaptive).tolist()
            self.scale[name]=min((0.,.125,.25,.5,1.),key=lambda s:nll[str(s)])
            self.models[name]=model;self.optimizers[name]=opt
            report["adaptive" if adaptive else "fixed"]={"selection_nll":nll,"scale":self.scale[name],"trace":trace,"fit_targets":len(ds[-1]),"selection_targets":len(dv[-1]),"candidate_coverage":float(ds[-2].any(1).double().mean()),"retention_weights":weights}
        return {"mix":self.mix,"mix_fit_nll":scores,"controls":report}
    def law(self,base,row,name,forced=False):
        words,x,p,m,c,b,v=row;scale=1. if forced else self.scale[name]
        if not scale:return base
        args=tuple(torch.as_tensor(a[None],dtype=torch.float32) for a in (x,m,c,b,v))
        with torch.no_grad():lift=self.models[name](*args,adaptive=name=="full")[0].numpy().astype(np.float64)
        return SL.Transport(base,words,lift*scale)
    def evaluate(self,*args,**kwargs):
        out=super().evaluate(*args,**kwargs);out["modes"]["fixed_retention"]=out["modes"].pop("exact_control");out["modes"]["adaptive_retention"]=out["modes"].pop("full");return out
    def root_laws(self,root):
        # Replay the whole observed prefix, not raw_byte_law.words' 128-word tail.
        state=self.field.session();g=self.graph();words=re.findall(rb'[A-Za-z]+',root)
        if root and FE.IS_LETTER[root[-1]]:words=words[:-1]
        for w in words:
            w=w.lower().decode('ascii');state.observe(w);g.observe(w)
        orig,_=V.branches(self.ref,self.small,self.large,state,self.cal);ex=orig[3];f,h=self.field.field_law(state,orig[1].law)
        base=H.Mixture((ex,f),(1-self.mix,self.mix));mem=state.attractor.query()
        ep=heapq.nlargest(24,ex.law.s,key=ex.law.s.get);fw=[self.field.sem.words[k] for k in np.argsort(-f.d,kind='stable')[:24]]
        mw={v for v,ok in zip(mem[3],mem[2]) if ok};row=self.features(base,ex,f,set(ep)|set(fw)|mw,g,state.h,mw)
        return {"consensus":base,"fixed":self.law(base,row,"exact"),"adaptive":self.law(base,row,"full")}
    def raw_rows(self,files,seed,count):
        rng=np.random.default_rng(seed);out=[]
        for bid,p in files:
            with p.open('rb') as f:raw=f.read(8192)
            start=0;events=[]
            for i,b in enumerate(raw):
                if not FE.IS_LETTER[b] or i-start==63:
                    if start>=128:events.append((start,i+1))
                    start=i+1
            for j in sorted(rng.choice(len(events),min(count,len(events)),replace=False)):
                a,b=events[j];out.append((raw[:a],raw[a:b]))
        assert out;return out
    def fit_events(self,files):
        # Shared event weight chosen with fixed-retention control only.
        scores=C.Counter();n=0
        for root,event in self.raw_rows(files,4403,4):
            law=self.root_laws(root)["fixed"];n+=len(event)
            e=FE.Event(law,self.backoff.prob,root,self.surface,0.)
            for w in (0.,.125,.25,.5,.75):
                e.weight=w;scores[w]-=math.log2(max(e.mass(event),1e-300))
        weight=min(scores,key=scores.get);return weight,{"weight":weight,"bytes":n,"fixed_calibration_bpb":{str(k):v/n for k,v in scores.items()}}
    def raw_eval(self,files,weight):
        m=C.defaultdict(C.Counter);checks=0
        for root,event in self.raw_rows(files,4404,4):
            for name,law in self.root_laws(root).items():
                e=FE.Event(law,self.backoff.prob,root,self.surface,weight);p=b'';prod=1.;z=m[name]
                for b in event:
                    q=e.next(p);rank=np.argsort(-q,kind='stable')[:5];z["bytes"]+=1;z["byte_t1"]+=b==rank[0];z["byte_t5"]+=b in rank
                    prod*=q[b];p+=bytes([b])
                assert e.done(p) and math.isclose(prod,e.mass(event),rel_tol=1e-7,abs_tol=1e-30);z["bits"]-=math.log2(max(prod,1e-300))
                p=b'';i=0
                while i<len(event):
                    one=event[i:i+1];size=2 if one[0] in b' etaoinr' and not e.done(p+one) else 1
                    target=event[i:i+size];t=e.tokens(p);rank=heapq.nlargest(5,t,key=t.get);z["tokens"]+=1;z["token_t1"]+=target==rank[0];z["token_t5"]+=target in rank
                    mass=np.zeros(256)
                    for tok,v in t.items():mass[tok[0]]+=v
                    assert np.max(np.abs(mass-e.next(p)))<1e-8;checks+=1;p+=target;i+=size
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    word=event[:-1].lower();rank,cert,_=e.rank(budget=1200)
                    z["words"]+=1;z["word_t1"]+=word.decode('ascii')==rank[0];z["word_t5"]+=word.decode('ascii') in rank;z["word_cert"]+=cert
                    z["word_bits"]-=math.log2(max(e.complete(word),1e-300))
        return {"modes":{k:dict(z,bpb=z["bits"]/z["bytes"],byte_top1=z["byte_t1"]/z["bytes"],byte_top5=z["byte_t5"]/z["bytes"],token_top1=z["token_t1"]/z["tokens"],token_top5=z["token_t5"]/z["tokens"],word_top1=z["word_t1"]/max(1,z["words"]),word_top5=z["word_t5"]/max(1,z["words"])) for k,z in m.items()},"marginal_checks":checks}
    def codec(self,root,target,weight):
        cache={}
        def predict(h):
            start=0
            for i,b in enumerate(h):
                if not FE.IS_LETTER[b] or i-start==63:start=i+1
            key=h[:start]
            if key not in cache:cache[key]=FE.Event(self.root_laws(key)["adaptive"],self.backoff.prob,key,self.surface,weight)
            return cache[key].next(h[start:])
        code=A.encode(root,target,predict);cache.clear();assert A.decode(root,code,len(target),predict)==target
        return {"bytes":len(target),"bits":code[1],"roundtrip":True}
def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("retention-metrics.json"));a=ap.parse_args();r=a.data
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
    exp=Adaptive(field,ref,small,large,cal)
    digest=lambda: {k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
    initial=digest();steps=field.steps.copy()
    fitbooks=extra[::24];selectbooks=calbooks;fit=exp.fit([B.load(p,512) for _,p in fitbooks],[B.load(p,512) for _,p in selectbooks])
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    rawdocs=[first(p) for _,p in tr];exp.backoff=A.Backoff(rawdocs);exp.surface=FE.Surface(rawdocs)
    event_weight,rawfit=exp.fit_events(calbooks)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    story=[B.load(p,1200) for _,p in picked];out={"parent_sha256":field.v36_sha,"fit":fit,"raw_fit":rawfit,"fresh":{}}
    write_metrics(a.out,out,"calibration")
    out["short"]=exp.evaluate(story);write_metrics(a.out,out,"short")
    out["long"]=exp.evaluate(story,True);write_metrics(a.out,out,"long")
    for name,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used or i in forbidden:continue
            if len(B.load(p,200))<112:continue
            chosen.append((i,p));used.add(i)
            if len(chosen)==6:break
        assert len(chosen)==6
        out["fresh"][name]={"ids":[i for i,p in chosen],"metrics":exp.evaluate([B.load(p,300) for _,p in chosen],limit=32),"raw":exp.raw_eval(chosen,event_weight)}
        write_metrics(a.out,out,"domain:"+name)
    raw=first(picked[0][1]);out["codec"]=exp.codec(raw[:128],raw[128:144],event_weight)
    assert steps==field.steps and initial==digest()
    out["inherited_unchanged"]=True;out["fit_ids"]=[i for i,p in fitbooks];out["selection_ids"]=[i for i,p in selectbooks];out["dev_ids"]=[i for i,p in picked];out["windows"]=list(WINDOWS)
    write_metrics(a.out,out,'evaluated')
    exp.save(a.out.with_suffix(".pt"),field.v36_sha)
    write_metrics(a.out,out)
if __name__=="__main__":main()
