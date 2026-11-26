"""v39: conserved finite word-mass transport and unified event law.
All inherited v37 weights and optimizer state frozen; v38 portability optional."""
import hybrid38 as P38, hybrid36 as R6, hybrid15 as H, hybrid20 as V, hybrid22 as Z, hybrid23 as P, hybrid18 as U
import base as B, foundation6 as N, raw_byte_law as A, subword_geometry as Y, symbiotic_law as SL, mass_credit as MC
import argparse, collections as C, hashlib, heapq, json, math, pathlib, time
import torch
import numpy as np
KMAX=80

class Symbiosis(P38.Portable):
    def __init__(self,*args):
        super().__init__(*args);self.theta=np.zeros(MC.DIM);self.theta_m=self.theta.copy();self.theta_v=self.theta.copy()
        self.mix=.5;self.scale=0.;self.event_weight=0.;self.raw_scale=0.
    def branches39(self,ref,small,large,cal,state):
        orig,_=V.branches(ref,small,large,state,cal)
        field,hidden=self.field_law(state,orig[1].law)
        parent,_=self.combine(field,hidden,state.attractor.query())
        return orig[3],field,hidden,parent
    def consensus(self,expanded,field):
        return H.Mixture((expanded,field),(1-self.mix,self.mix))
    def candidates(self,expanded,field,hidden,query,base,ignore_field=False):
        f,n,valid,words=query
        ep=heapq.nlargest(16,expanded.law.s,key=expanded.law.s.get)
        fi=np.argsort(-field.d,kind='stable')[:16];fw=[self.sem.words[j] for j in fi]
        memory={w for w,v in zip(words,valid) if v};candidates=sorted(set(ep)|set(fw)|memory)
        assert len(candidates)<=KMAX
        mp,_,_=self.predict(f[None],n[None],valid[None],hidden[None]);mp=mp[0].numpy()
        by=C.defaultdict(list)
        for i,w in enumerate(words):
            if valid[i]:by[w].append(i)
        features=[];probs=[]
        for w in candidates:
            p=base.complete(w);a=expanded.complete(w);b=field.complete(w);ix=by[w]
            values=f[ix].max(0) if ix else np.zeros(6)
            def ratio(x):return float(np.clip(math.log(max(x,1e-30)/max(p,1e-30)),-4,4)/4)
            rel=float(np.clip(math.log(max(float(mp[ix].sum())*max(int(valid.sum()),1),1e-30)),-4,4)/4) if ix else 0.
            features.append([1.,ratio(b) if not ignore_field else 0.,ratio(a),max(-16,math.log(max(p,1e-30)))/16,
                float(bool(ix)),*values[:5],math.log1p(len(ix))/math.log(49),rel if not ignore_field else 0.,float(w in fw and not ignore_field),float(w in ep)])
            probs.append(p)
        return candidates,np.asarray(features,np.float64).reshape(-1,MC.DIM),np.asarray(probs,np.float64)
    def transported(self,base,cand,scale=None):
        words,features,_=cand
        if scale is None:scale=self.scale
        if not scale:return base
        return SL.Transport(base,words,4*np.tanh(features@self.theta/4)*scale)
    def visit(self,docs,ref,small,large,cal,limit=64,long=False):
        for j,doc in enumerate(docs):
            state=self.session();wrong=self.bank();other=docs[(j+1)%len(docs)]
            positions=set(range(16,min(len(doc),16+limit))) if not long else set(range(128,min(len(doc),1200),17))
            if not positions:continue
            for i,w in enumerate(doc[:max(positions)+1]):
                if i in positions:
                    expanded,field,h,parent=self.branches39(ref,small,large,cal,state)
                    yield w,expanded,field,h,parent,state.attractor.query(),wrong.query(state.attractor.h)
                state.observe(w)
                if i<len(other):wrong.observe(other[i])
    def fit_mass(self,fitdocs,selectdocs,ref,small,large,cal):
        # Fit only on first eight calibration books; select on other four.
        rows=list(self.visit(fitdocs,ref,small,large,cal,limit=48))
        grid=(0.,.25,.5,.75,1.)
        scores={str(a):float(np.mean([-math.log(max((1-a)*ex.complete(w)+a*f.complete(w),1e-300)) for w,ex,f,*_ in rows])) for a in grid}
        self.mix=min(grid,key=lambda a:scores[str(a)])
        def dataset(rows):
            fs=[];ps=[];hs=[];bs=[]
            for w,ex,f,h,parent,q,_ in rows:
                base=self.consensus(ex,f);words,feat,p=self.candidates(ex,f,h,q,base)
                k=len(p);assert p.sum()<=1+1e-8
                fs.append(np.pad(feat,((0,KMAX-k),(0,0))));ps.append(np.pad(p,(0,KMAX-k)))
                hs.append(np.pad(np.asarray([v==w for v in words],np.float64),(0,KMAX-k)));bs.append(max(base.complete(w),1e-300))
            return tuple(np.asarray(a,np.float64) for a in (fs,ps,hs,bs))
        ds=dataset(rows);del rows
        self.theta,self.theta_m,self.theta_v,trace=MC.fit(ds)
        valid=dataset(self.visit(selectdocs,ref,small,large,cal,limit=72))
        trust=(0.,.125,.25,.5,1.)
        v={str(a):MC.objective(self.theta,*valid,scale=a)[0] for a in trust}
        self.scale=min(trust,key=lambda a:v[str(a)])
        return {"base_mix":self.mix,"base_fit_nll":scores,"transport_scale":self.scale,"selection_nll":v,
                "fit_targets":len(ds[-1]),"selection_targets":len(valid[-1]),"fit_candidate_coverage":float(ds[-2].any(1).mean()),"trace":trace}
    def word_eval39(self,docs,ref,small,large,cal,long=False,limit=72):
        m=C.defaultdict(C.Counter);start=time.monotonic()
        for w,ex,f,h,parent,q,bad in self.visit(docs,ref,small,large,cal,limit,long):
            base=self.consensus(ex,f);cand=self.candidates(ex,f,h,q,base);wrong=self.candidates(ex,f,h,bad,base)
            off=self.candidates(ex,f,h,q,base,True)
            for name,law in (("parent",parent),("expanded",ex),("consensus",base),("transport",self.transported(base,cand)),("forced_transport",self.transported(base,cand,1.)),("wrong_memory",self.transported(base,wrong)),("field_features_off",self.transported(base,off))):
                l=H.Mixture((law,),(1.,));rank,cert,_=l.rank(budget=2000);z=m[name]
                z["n"]+=1;z["top1"]+=w==rank[0];z["top5"]+=w in rank;z["cert"]+=cert
                p=law.complete(w);z["bits"]-=math.log2(max(p,1e-300));z["symbols"]+=len(w)+1
                if name=="transport":
                    pref="";prod=1.
                    for c in w+B.EOS:
                        prod*=l.next(pref)[c]
                        if c!=B.EOS:pref+=c
                    assert math.isclose(prod,p,rel_tol=1e-6,abs_tol=1e-15)
        return {"modes":{k:dict(v,word_top1=v["top1"]/v["n"],word_top5=v["top5"]/v["n"],spelling_bps=v["bits"]/v["symbols"]) for k,v in m.items()},"seconds":time.monotonic()-start}
    def root_raw(self,root,hidden,backoff,scale):
        # Fixed field hidden at event entry; local causal byte interactions and
        # two-byte composition are still evaluated at every prefix. This is a
        # new factorization, not parity with v37's dynamic recurrent raw law.
        p=self.p;code=p["byte_code"].astype(np.float64);local=p["byte_local"].astype(np.float64)
        q1=hidden@p["byte_query"];q2=hidden@p["sub_query"];pair=p["sub_pair"];cond=p["sub_cond"];bias=p["byte_bias"]
        alpha,beta=(scale*a for a in self.trust);cache={}
        def predict(prefix):
            if prefix in cache:return cache[prefix]
            full=root+prefix;prior=backoff.prob(full)
            if scale==0:return prior
            _,pending=Y.parse(prefix)
            if pending:
                a=code[pending[0]];z=q2+a@cond;comp=a+code[:256]+(a*code[:256])@pair;logits=comp@z;g=beta
            else:
                e=code[np.asarray(A.context(full))];z=q1+np.einsum('ri,rij->j',e,local);logits=code[:256]@z+bias;g=alpha
            v=prior*np.exp(logits-logits.max());v/=v.sum();out=(1-g)*prior+g*v;out/=out.sum()
            # bounded cache per event, not across documents.
            if len(cache)<4096:cache[prefix]=out
            return out
        return predict
    def event39(self,root,ref,small,large,cal,backoff,weight=None,raw_scale=None):
        state=self.session()
        for w in A.words(root):state.observe(w)
        ex,f,h,parent=self.branches39(ref,small,large,cal,state);base=self.consensus(ex,f)
        word=self.transported(base,self.candidates(ex,f,h,state.attractor.query(),base))
        x,b,_=self.raw_row(root,backoff);t,_=Y.parse(root)
        self.byte_context=np.asarray([b],np.int64);self.token_context=np.asarray([Y.context(t)],np.int64)
        try:_,hidden,_,_=self.tape(np.asarray([x],np.uint32))
        finally:self.byte_context=None;self.token_context=None
        return SL.Event(word,self.root_raw(root,hidden[0],backoff,self.raw_scale if raw_scale is None else raw_scale),self.event_weight if weight is None else weight)
    def event_rows(self,files,seed,count):
        rng=np.random.default_rng(seed);rows=[]
        for bid,path in files:
            raw=path.read_bytes()[:8192];start=0;pos=[]
            for i,b in enumerate(raw):
                if b not in SL.ALPHA or i-start==SL.CAP-1:
                    if start>=128:pos.append((start,i+1))
                    start=i+1
            if not pos:continue
            for j in sorted(rng.choice(len(pos),min(count,len(pos)),replace=False)):
                a,b=pos[j];rows.append((bid,raw[:a],raw[a:b]))
        return rows
    def fit_event(self,rows,ref,small,large,cal,backoff):
        grid=(0.,.125,.25,.5,.75);tot=C.Counter();n=0
        for _,root,event in rows:
            n+=len(event)
            # Build field once per raw configuration; vary word weight without another field pass.
            for raw_scale in (0.,1.):
                e=self.event39(root,ref,small,large,cal,backoff,0.,raw_scale)
                for weight in grid:
                    e.weight=weight;tot[raw_scale,weight]-=math.log2(max(e.mass(event),1e-300))
        choice=min(tot,key=tot.get);self.raw_scale,self.event_weight=choice
        return {"raw_scale":self.raw_scale,"word_event_weight":self.event_weight,"selection_bytes":n,"grid_bpb":{str(k):v/n for k,v in tot.items()}}
    def raw_eval(self,rows,ref,small,large,cal,backoff):
        m=C.defaultdict(C.Counter);checks=0
        for _,root,event in rows:
            e=self.event39(root,ref,small,large,cal,backoff)
            for name,scale,weight in (("count",0.,0.),("raw_field",self.raw_scale,0.),("unified",self.raw_scale,self.event_weight)):
                raw=e.raw if scale==self.raw_scale else self.root_raw(root,np.zeros(128,np.float32),backoff,0.)
                ev=SL.Event(e.word,raw,weight);z=m[name];p=b'';prod=1.
                for b in event:
                    q=ev.next(p);order=np.argsort(-q,kind='stable')[:5]
                    z["bytes"]+=1;z["byte_t1"]+=order[0]==b;z["byte_t5"]+=b in order
                    prod*=q[b];p+=bytes([b])
                assert math.isclose(prod,ev.mass(event),rel_tol=1e-6,abs_tol=1e-30)
                z["bits"]-=math.log2(max(prod,1e-300))
                # Derive subword scores from this exact same byte law, not another head.
                p=b'';i=0
                while i<len(event):
                    tokens=ev.tokens(p);one=event[i:i+1];size=2 if one[0] in SL.SPLIT and not ev.done(p+one) else 1
                    target=event[i:i+size];rank=heapq.nlargest(5,tokens,key=tokens.get)
                    z["tokens"]+=1;z["token_t1"]+=target==rank[0];z["token_t5"]+=target in rank
                    if name=="unified":
                        q=ev.next(p);mass=np.zeros(256)
                        for tok,prob in tokens.items():mass[tok[0]]+=prob
                        assert np.max(np.abs(mass-q))<1e-7;checks+=1
                    p+=target;i+=size
                if event[-1]==32 and len(event)>1:
                    w=event[:-1].decode('ascii');rank,cert,_=ev.rank_words(budget=2000)
                    z["canonical_words"]+=1;z["word_t1"]+=w==rank[0];z["word_t5"]+=w in rank;z["word_cert"]+=cert
        return {"modes":{k:dict(v,bpb=v["bits"]/v["bytes"],byte_top1=v["byte_t1"]/v["bytes"],byte_top5=v["byte_t5"]/v["bytes"],token_top1=v["token_t1"]/v["tokens"],token_top5=v["token_t5"]/v["tokens"],canonical_word_top1=v["word_t1"]/max(1,v["canonical_words"]),canonical_word_top5=v["word_t5"]/max(1,v["canonical_words"])) for k,v in m.items()},"marginal_checks":checks}
    def codec_check(self,root,target,ref,small,large,cal,backoff):
        cache={}
        def predict(history):
            start=0
            for i,b in enumerate(history):
                if b not in SL.ALPHA or i-start==SL.CAP-1:start=i+1
            key=history[:start];p=history[start:]
            if key not in cache:cache[key]=self.event39(key,ref,small,large,cal,backoff)
            return cache[key].next(p)
        code=A.encode(root,target,predict);cache.clear();assert A.decode(root,code,len(target),predict)==target
        return {"bytes":len(target),"bits":code[1],"roundtrip":True}
    def save39(self,path):
        # Immutable inherited artifact plus a tiny transport state; never clone or reset inherited Adam.
        np.savez(path,theta=self.theta,m=self.theta_m,v=self.theta_v,step=np.asarray(256),parent_sha256=np.asarray(self.v37_sha))
        path.with_suffix(".json").write_text(json.dumps({"parent_sha256":self.v37_sha,"mix":self.mix,"scale":self.scale,"raw_scale":self.raw_scale,"event_weight":self.event_weight,"event_cap":SL.CAP,"token_split":sorted(SL.SPLIT)},indent=2))

def manifest(groups):
    seen={};out={}
    for name,files in groups.items():
        out[name]=[]
        for bid,p in files:
            sha=hashlib.sha256(p.read_bytes()).hexdigest()
            if sha in seen:assert seen[sha]==name,("cross-split exact content overlap",name,seen[sha],bid)
            seen[sha]=name;out[name].append({"id":bid,"sha256":sha})
    return out

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor","unified"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--profile",choices=("cpu","t4","blackwell"),default="cpu");ap.add_argument("--threads",type=int,default=4);ap.add_argument("--smoke-seconds",type=int,default=60);ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
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
    field=Symbiosis(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,a.attractor,a.unified,len(data[1]));field.bind(ref,small,large,cal)

    torch.set_num_threads(max(1,a.threads))
    field.configure(a.profile)

    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    fitbooks=calbooks[:8];selectbooks=calbooks[8:];fitdocs=[B.load(p,1200) for _,p in fitbooks];selectdocs=[B.load(p,1200) for _,p in selectbooks]
    initial={k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()};steps=field.steps.copy()
    fit=field.fit_mass(fitdocs,selectdocs,ref,small,large,cal)
    rawfit=field.fit_event(field.event_rows(selectbooks,3902,16),ref,small,large,cal,backoff)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    fresh={}
    for name,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used or i in forbidden:continue
            if len(B.load(p,200))<112:continue
            chosen.append((i,p));used.add(i)
            if len(chosen)==6:break
        assert len(chosen)==6;fresh[name]=chosen
    mani=manifest({"reference":tr+extra,"cal_fit":fitbooks,"cal_select":selectbooks,"dev_story":picked,**{"dev_fresh_"+k:v for k,v in fresh.items()}})
    a.out.with_name("split_manifest.json").write_text(json.dumps(mani,indent=2))
    story=[B.load(p,1200) for _,p in picked]
    result={"parent_sha256":field.v37_sha,"fit":fit,"raw_fit":rawfit,"short":field.word_eval39(story,ref,small,large,cal),"long":field.word_eval39(story,ref,small,large,cal,True),"fresh":{}}
    for name,files in fresh.items():
        result["fresh"][name]={"word":field.word_eval39([B.load(p,300) for _,p in files],ref,small,large,cal,limit=32),"raw":field.raw_eval(field.event_rows(files,3903,12),ref,small,large,cal,backoff)}
    raw=first(picked[0][1]);result["codec"]=field.codec_check(raw[:128],raw[128:144],ref,small,large,cal,backoff)
    assert steps==field.steps and all(initial[k]==hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items())
    result["inherited_unchanged"]=True;result["manifest_sha256"]=hashlib.sha256(a.out.with_name("split_manifest.json").read_bytes()).hexdigest()
    field.save39(a.out.with_name("symbiotic-transport.npz"));a.out.write_text(json.dumps(result,indent=2));print("FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
