"""v41 diagnostic: persistent cross-half graph + exact mass transport.
Uses completed v36 checkpoint, not a new model lineage. No inherited changes.
Fits equal-budget full and exact-only controls on reference-held calibration docs.
"""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'nancm_structural'))
import hybrid36 as R6, hybrid37 as U37, hybrid15 as H, hybrid20 as V, hybrid22 as Z, hybrid23 as P, hybrid18 as U
import base as B, foundation6 as N, symbiotic_law as SL, mass_credit as MC, raw_byte_law as A
from persistent_edges import Graph,WINDOWS
import numpy as np
from metric_io import write_metrics
import torch
import argparse,collections as C,hashlib,heapq,json,math,time
DIM=25
MAX=128

class Experiment:
    def __init__(self,field,ref,small,large,cal):
        self.field=field;self.ref=ref;self.small=small;self.large=large;self.cal=cal
        self.mix=.5;self.scale={};self.params={}
    def graph(self):return Graph(self.field.p["word_code"],self.field.sem.ids,self.field.sem.classof)
    def examples(self,docs,limit=48,long=False):
        for j,doc in enumerate(docs):
            if len(doc)<32:continue
            s=self.field.session();g=self.graph();wrong=self.graph();other=docs[(j+1)%len(docs)]
            pos=set(range(128,min(len(doc),1200),13)) if long else set(range(16,min(len(doc),16+limit)))
            if not pos:continue
            for i,w in enumerate(doc[:max(pos)+1]):
                if i in pos:
                    orig,_=V.branches(self.ref,self.small,self.large,s,self.cal)
                    ex=orig[3];f,h=self.field.field_law(s,orig[1].law)
                    parent,_=self.field.combine(f,h,s.attractor.query())
                    base=H.Mixture((ex,f),(1-self.mix,self.mix))
                    mem=s.attractor.query()
                    ep=heapq.nlargest(24,ex.law.s,key=ex.law.s.get)
                    fw=[self.field.sem.words[k] for k in np.argsort(-f.d,kind='stable')[:24]]
                    mw={v for v,ok in zip(mem[3],mem[2]) if ok}
                    # Both real and wrong graphs use identical target-free base pool.
                    common=set(ep)|set(fw)|mw
                    real=self.features(base,ex,f,common,g,s.h,mw)
                    bad=self.features(base,ex,f,common,wrong,s.h,mw)
                    yield w,ex,f,parent,base,real,bad,w in g.propose(s.h),w in wrong.propose(s.h)
                s.observe(w);g.observe(w)
                if i<len(other):wrong.observe(other[i])
    def features(self,base,ex,field,common,g,h,mw):
        candidates=sorted(common|set(g.propose(h)));assert len(candidates)<=MAX
        p=np.asarray([base.complete(w) for w in candidates],np.float64)
        a=np.asarray([ex.complete(w) for w in candidates],np.float64)
        b=np.asarray([field.complete(w) for w in candidates],np.float64)
        den=np.maximum(p,1e-300);geom=g.features(candidates,h)
        feat=np.column_stack([np.ones(len(p)),np.clip(np.log(np.maximum(a,1e-300)/den),-4,4),
            np.clip(np.log(np.maximum(b,1e-300)/den),-4,4),np.maximum(-16,np.log(den))/16,
            np.asarray([w in mw for w in candidates]),np.asarray([w in common for w in candidates]),np.asarray([w in g.propose(h) for w in candidates]),geom])
        assert feat.shape==(len(p),DIM) and p.sum()<=1+1e-8
        return candidates,feat,p
    def dataset(self,examples,exact=False):
        fs=[];ps=[];hits=[];bs=[]
        for w,ex,f,parent,base,real,bad,*_ in examples:
            words,x,p=real;x=x.copy()
            if exact:x[:,15:23]=0 # only remove 8 signed geometry residuals, keep exact rows
            k=len(p);fs.append(np.pad(x,((0,MAX-k),(0,0))));ps.append(np.pad(p,(0,MAX-k)))
            hits.append(np.pad(np.asarray([v==w for v in words],np.float64),(0,MAX-k)));bs.append(max(base.complete(w),1e-300))
        return tuple(np.asarray(a,np.float64) for a in (fs,ps,hits,bs))
    def fit(self,fitdocs,validdocs):
        rows=list(self.examples(fitdocs,long=True));grid=(0.,.25,.5,.75,1.)
        scores={str(s):float(np.mean([-math.log(max((1-s)*ex.complete(w)+s*f.complete(w),1e-300)) for w,ex,f,*_ in rows])) for s in grid}
        self.mix=min(grid,key=lambda s:scores[str(s)]);del rows
        tr=list(self.examples(fitdocs,long=True));va=list(self.examples(validdocs,long=True));report={}
        for name,exact in (("full",False),("exact",True)):
            ds=self.dataset(tr,exact);dv=self.dataset(va,exact)
            theta=np.zeros(DIM);m=theta.copy();v=theta.copy();rng=np.random.default_rng(4102)
            trace=[]
            for i in range(1,385):
                ix=rng.integers(len(ds[-1]),size=32);loss,g=MC.objective(theta,*(a[ix] for a in ds),reg=.01)
                m=.9*m+.1*g;v=.999*v+.001*g*g
                theta-=.015*(m/(1-.9**i))/(np.sqrt(v/(1-.999**i))+1e-8)
                if i==1 or i%64==0:trace.append([i,loss])
            scales=(0.,.125,.25,.5,1.);nll={str(s):MC.objective(theta,*dv,scale=s)[0] for s in scales}
            self.scale[name]=min(scales,key=lambda s:nll[str(s)]);self.params[name]=(theta,m,v)
            report[name]={"selection_nll":nll,"scale":self.scale[name],"trace":trace,"fit_targets":len(ds[-1]),"selection_targets":len(dv[-1]),"candidate_coverage":float(ds[-2].any(1).mean())}
        return {"mix":self.mix,"mix_fit_nll":scores,"controls":report}
    def law(self,base,row,name,forced=False):
        words,x,p=row;x=x.copy()
        if name=="exact":x[:,15:23]=0
        scale=1. if forced else self.scale[name]
        if not scale:return base
        return SL.Transport(base,words,4*np.tanh(x@self.params[name][0]/4)*scale)
    def evaluate(self,docs,long=False,limit=72):
        m=C.defaultdict(C.Counter);rec=C.Counter();start=time.monotonic()
        for w,ex,f,parent,base,real,bad,hit,wrong in self.examples(docs,limit,long):
            rec["n"]+=1;rec["correct_graph32"]+=hit;rec["wrong_graph32"]+=wrong;rec["union"]+=w in real[0]
            full=self.law(base,real,"full")
            for name,law in (("v36",parent),("expanded",ex),("consensus",base),("exact_control",self.law(base,real,"exact")),("full",full),("forced",self.law(base,real,"full",True)),("wrong_graph",self.law(base,bad,"full"))):
                q=H.Mixture((law,),(1.,));rank,cert,_=q.rank(budget=2000);z=m[name];z["n"]+=1;z["t1"]+=w==rank[0];z["t5"]+=w in rank;z["cert"]+=cert
                p=law.complete(w);z["bits"]-=math.log2(max(p,1e-300));z["symbols"]+=len(w)+1
                if name=="full":
                    pref="";prod=1.
                    for c in w+B.EOS:
                        prod*=q.next(pref)[c]
                        if c!=B.EOS:pref+=c
                    assert math.isclose(prod,p,rel_tol=1e-6,abs_tol=1e-15)
        return {"modes":{k:dict(v,word_top1=v["t1"]/v["n"],word_top5=v["t5"]/v["n"],spelling_bps=v["bits"]/v["symbols"]) for k,v in m.items()},"candidate":dict(rec),"seconds":time.monotonic()-start}
def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("graph-metrics.json"));a=ap.parse_args();r=a.data
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
    exp=Experiment(field,ref,small,large,cal)
    digest=lambda: {k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
    initial=digest();steps=field.steps.copy()
    fitbooks=calbooks[:8];selectbooks=calbooks[8:];fit=exp.fit([B.load(p,1200) for _,p in fitbooks],[B.load(p,1200) for _,p in selectbooks])
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    story=[B.load(p,1200) for _,p in picked];out={"parent_sha256":field.v36_sha,"fit":fit,"fresh":{}}
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
        out["fresh"][name]={"ids":[i for i,p in chosen],"metrics":exp.evaluate([B.load(p,300) for _,p in chosen],limit=32)}
        write_metrics(a.out,out,"domain:"+name)
    assert steps==field.steps and initial==digest()
    out["inherited_unchanged"]=True;out["fit_ids"]=[i for i,p in fitbooks];out["selection_ids"]=[i for i,p in selectbooks];out["dev_ids"]=[i for i,p in picked];out["windows"]=list(WINDOWS)
    write_metrics(a.out,out,'evaluated')
    # Keep mechanism coefficients only; never promote or overwrite lineage.
    np.savez(a.out.with_suffix(".npz"),parent_sha256=np.asarray(field.v36_sha),mix=exp.mix,**{name+"_"+k:v for name,(t,m,z) in exp.params.items() for k,v in (("theta",t),("m",m),("v",z),("scale",np.asarray(exp.scale[name])))})
    write_metrics(a.out,out)
if __name__=="__main__":main()
