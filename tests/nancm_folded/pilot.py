"""Independent v42 case-folded event diagnostic on exact completed v36.
No new checkpoint lineage; fit only surface donor on reference and select
mixture coefficients on calibration. Compare count, field, and unified law.
"""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'nancm_structural'))
import hybrid36 as R6, hybrid37 as U37, hybrid15 as H, hybrid20 as V, hybrid22 as Z, hybrid23 as P, hybrid18 as U
import base as B, foundation6 as N, raw_byte_law as A
import folded_event as FE
import numpy as np
import torch
import argparse,collections as C,hashlib,heapq,json,math,time

class Model:
    def __init__(self,field,ref,small,large,cal,backoff,surface):
        self.field=field;self.ref=ref;self.small=small;self.large=large;self.cal=cal
        self.backoff=backoff;self.surface=surface
    def root(self,raw):
        state=self.field.session()
        for w in A.words(raw):state.observe(w)
        orig,_=V.branches(self.ref,self.small,self.large,state,self.cal)
        field,h=self.field.field_law(state,orig[1].law)
        parent,_=self.field.combine(field,h,state.attractor.query())
        return orig[3],parent,h
    def donor(self,h,scale):
        p=self.field.p;code=p["byte_code"].astype(np.float64);query=h@p["byte_query"]
        local=p["byte_local"][:3].astype(np.float64);bias=p["byte_bias"]
        def predict(s):
            q=self.backoff.prob(s)
            if not scale:return q
            ids=list(s[::-1])+[256]*(3-len(s))
            z=query+np.einsum('ri,rij->j',code[ids],local)
            l=code[:256]@z+bias;t=q*np.exp(l-l.max());t/=t.sum()
            return (1-scale)*q+scale*t
        return predict
    def event(self,root,scale,weight,mix):
        ex,field,h=self.root(root)
        law=H.Mixture((ex,field),(1-mix,mix))
        return FE.Event(law,self.donor(h,scale),root,self.surface,weight)
    def fit(self,rows):
        grid=(0.,.125,.25,.5,.75)
        stats=C.Counter();n=0
        for _,root,event in rows:
            ex,f,h=self.root(root);n+=len(event)
            for mix in (0.,.5,1.):
                law=H.Mixture((ex,f),(1-mix,mix))
                for s in (0.,.25,1.):
                    e=FE.Event(law,self.donor(h,s),root,self.surface,0.)
                    for g in grid:
                        e.weight=g;stats[mix,s,g]-=math.log2(max(e.mass(event),1e-300))
        key=min(stats,key=stats.get)
        return key,{"grid_bpb":{str(k):v/n for k,v in stats.items()},"bytes":n,"selected":list(key)}
    def evaluate(self,rows,selected):
        mix,s,g=selected;m=C.defaultdict(C.Counter);checks=0;start=time.monotonic()
        for _,root,event in rows:
            ex,f,h=self.root(root)
            for name,mx,sc,wt in (("count",0.,0.,0.),("raw_field",mix,s,0.),("lexical_retrieval",0.,s,g),("unified",mix,s,g)):
                e=FE.Event(H.Mixture((ex,f),(1-mx,mx)),self.donor(h,sc),root,self.surface,wt)
                z=m[name];p=b'';prod=1.
                for b in event:
                    q=e.next(p);rank=np.argsort(-q,kind="stable")[:5];z["bytes"]+=1;z["byte_t1"]+=b==rank[0];z["byte_t5"]+=b in rank
                    prod*=q[b];p+=bytes([b])
                assert e.done(p) and math.isclose(prod,e.mass(event),rel_tol=1e-7,abs_tol=1e-30)
                z["bits"]-=math.log2(max(prod,1e-300))
                p=b'';i=0
                while i<len(event):
                    one=event[i:i+1];size=2 if one[0] in b' etaoinr' and not e.done(p+one) else 1
                    target=event[i:i+size];tokens=e.tokens(p);rank=heapq.nlargest(5,tokens,key=tokens.get)
                    z["tokens"]+=1;z["token_t1"]+=target==rank[0];z["token_t5"]+=target in rank
                    if name=="unified":
                        mass=np.zeros(256)
                        for tok,v in tokens.items():mass[tok[0]]+=v
                        assert np.max(np.abs(mass-e.next(p)))<1e-8;checks+=1
                    p+=target;i+=size
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    word=event[:-1].lower();rank,cert,_=e.rank(budget=1200)
                    z["words"]+=1;z["word_t1"]+=word.decode('ascii')==rank[0];z["word_t5"]+=word.decode('ascii') in rank;z["word_cert"]+=cert
                    z["word_bits"]-=math.log2(max(e.complete(word),1e-300))
                    z["uppercase_words"]+=any(65<=b<=90 for b in event[:-1]);z["nonspace_words"]+=event[-1]!=32
        return {"modes":{k:dict(v,bpb=v["bits"]/v["bytes"],byte_top1=v["byte_t1"]/v["bytes"],byte_top5=v["byte_t5"]/v["bytes"],token_top1=v["token_t1"]/v["tokens"],token_top5=v["token_t5"]/v["tokens"],word_top1=v["word_t1"]/max(1,v["words"]),word_top5=v["word_t5"]/max(1,v["words"]),word_cert_rate=v["word_cert"]/max(1,v["words"])) for k,v in m.items()},"marginal_checks":checks,"seconds":time.monotonic()-start}
    def codec(self,root,target,selected):
        mix,s,g=selected;cache={}
        def predict(h):
            start=0
            for i,b in enumerate(h):
                if not FE.IS_LETTER[b] or i-start==63:start=i+1
            key=h[:start]
            if key not in cache:cache[key]=self.event(key,s,g,mix)
            return cache[key].next(h[start:])
        code=A.encode(root,target,predict);cache.clear();assert A.decode(root,code,len(target),predict)==target
        return {"bytes":len(target),"bits":code[1],"roundtrip":True}

def rows(files,seed,count):
    rng=np.random.default_rng(seed);out=[]
    for bid,p in files:
        with p.open('rb') as f:raw=f.read(8192)
        start=0;events=[]
        for i,b in enumerate(raw):
            if not FE.IS_LETTER[b] or i-start==63:
                if start>=128:events.append((start,i+1))
                start=i+1
        for j in sorted(rng.choice(len(events),min(count,len(events)),replace=False)):
            a,b=events[j];out.append((bid,raw[:a],raw[a:b]))
    assert out;return out

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("folded-metrics.json"));a=ap.parse_args();r=a.data
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
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    donor=[first(p) for _,p in tr]
    model=Model(field,ref,small,large,cal,A.Backoff(donor),FE.Surface(donor))
    digest=lambda: {k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
    initial=digest();steps=field.steps.copy()
    selected,fit=model.fit(rows(calbooks,4201,12))
    out={"parent_sha256":field.v36_sha,"fit":fit,"domains":{}}
    forbidden={i for i,p in tr+extra+calbooks};used=set();first_dev=None
    for name,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used or i in forbidden:continue
            chosen.append((i,p));used.add(i)
            if len(chosen)==6:break
        assert len(chosen)==6
        if first_dev is None:first_dev=chosen[0][1]
        out["domains"][name]={"ids":[i for i,p in chosen],"metrics":model.evaluate(rows(chosen,4202,12),selected)}
    raw=first(first_dev);out["codec"]=model.codec(raw[:128],raw[128:144],selected)
    assert steps==field.steps and initial==digest();out["inherited_unchanged"]=True
    out["raw_order"]=3;out["cap"]=64;out["calibration_ids"]=[i for i,p in calbooks]
    a.out.write_text(json.dumps(out,indent=2));print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
