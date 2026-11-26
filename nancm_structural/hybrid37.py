"""v37: conflict-aware transactional joint learning and canonical word-event
bridge. A bounded pilot, not full training or production acceptance."""
import hybrid36 as R6, hybrid35 as R5, hybrid34 as Z4, hybrid33 as X, hybrid32 as W, hybrid31 as I, hybrid30 as M, hybrid29 as Q, hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, temporal_phasor as T, subword_geometry as Y, attractor_memory as AM, raw_byte_law as A, joint_credit as JC, word_byte_bridge as WB
import numpy as np
import argparse, collections as C, json, math, pathlib, time, hashlib

def load36(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v35_sha
        assert int(ck["step"])==7680
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v36_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    meta=json.loads(path.with_suffix(".token.json").read_text())
    assert meta["split"]==list(Y.SPLIT) and meta["vocabulary"]==len(Y.TOKENS) and meta["parent_sha256"]==f.v35_sha
    f.trust=tuple(meta["trust"]);assert len(f.trust)==2 and all(0<=v<=1 for v in f.trust)
    meta=json.loads(path.with_suffix(".memory.json").read_text())
    assert meta["capacity"]==AM.CAP and meta["candidates"]==AM.K and meta["parent_sha256"]==f.v35_sha
    f.memory_scale=float(meta["scale"]);assert 0<=f.memory_scale<=1
    return f










class Unified(R6.MemoryField):
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,available)
        load36(self,p36)
        self.parent=load36(R6.MemoryField(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,available),p36)
        self.bridge_weight=0.
        self.word_mode="full"
    def token_loss(self,ds,grad=False):
        if grad:return self.token_forward(*ds,gradient=True)
        q1,q2=self.token_forward(*ds[:-1]);y=ds[-1];ix=np.arange(len(y));mask=y[:,1]!=256
        return float(-(np.log(q1[ix,y[:,0]]).sum()+np.log(q2[ix[mask],0,y[mask,1]]).sum())/(len(y)+mask.sum()))
    def transaction(self,word,token):
        x,y=word;lw,gw,_=self.direct(x,y,True);lt,gt=self.token_loss(token,True)
        g,stats=JC.merge(gw,gt)
        # Word code frozen so stored lexical keys keep the same representation.
        # Sparse lexical rows and memory parameters are fitted in separate phases.
        keys=[k for k in g if k in self.p and k not in (*self.mem_keys,"word_code") and np.any(g[k])]
        saved={k:(self.p[k].copy(),self.m[k].copy(),self.v[k].copy(),self.steps[k]) for k in keys}
        direction={};moments={}
        for k in keys:
            p,m,v,j=saved[k];j+=1;a=g[k]
            m=.9*m+.1*a;v=.999*v+.001*a*a
            lr=.0002 if k in (*self.res_keys,*self.byte_keys,*self.sub_keys) else .0001
            direction[k]=lr*(m/(1-.9**j))/(np.sqrt(v/(1-.999**j))+1e-8);moments[k]=(m,v,j)
        signed=("read","bias","field_out",*self.fusion_keys,*self.temporal_keys,*self.word_keys,*self.byte_keys,*self.sub_keys,*self.res_keys)
        accepted=0.;after=(lw,lt)
        try:
            for alpha in (1.,.25,.0625):
                for k in keys:
                    self.p[k][:]=saved[k][0]-alpha*direction[k]
                    if k not in signed:np.clip(self.p[k],0.,3.,out=self.p[k])
                    if k=="fusion_class":self.p[k][-1]=max(0.,float(self.p[k][-1]))
                    if k=="byte_code":self.p[k][256].fill(0)
                nw=self.direct(x,y)[0];nt=self.token_loss(token)
                # Compare actual proposed Adam step on both sampled objectives.
                if np.isfinite(nw+nt) and nw<=lw+1e-7 and nt<=lt+1e-7 and nw+nt<lw+lt-1e-8:
                    accepted=alpha;after=(nw,nt);break
        finally:
            if not accepted:
                for k in keys:self.p[k][:]=saved[k][0]
        if accepted:
            self.step+=1
            for k in keys:self.m[k],self.v[k],self.steps[k]=moments[k]
        else:
            for k in keys:
                p,m,v,j=saved[k];assert np.array_equal(p,self.p[k]) and np.array_equal(m,self.m[k]) and np.array_equal(v,self.v[k]) and j==self.steps[k]
        return {"accepted":accepted,"word_before":lw,"token_before":lt,"word_after":after[0],"token_after":after[1],**stats}
    def train_joint(self,word,token,attempts=192):
        trace=[];total=C.Counter()
        for i in range(attempts):
            ix=self.rng.integers(len(word[-1]),size=16);jx=self.rng.integers(len(token[-1]),size=16)
            z=self.transaction(tuple(a[ix] for a in word),tuple(a[jx] for a in token))
            for k in ("shared","conflicts","projected","cosine_sum"):total[k]+=z[k]
            total["accepted"]+=bool(z["accepted"]);total["attempts"]+=1
            if i%16==0 or i==attempts-1:trace.append([i+1,z])
        return {"trace":trace,**dict(total),"mean_shared_cosine":total["cosine_sum"]/max(1,total["shared"])}
    def select_word_mode(self,wd):
        modes={"full":("full","full"),"res_off":("off","full"),"res_and_input_off":("off","input_off")}
        scores={}
        for name,(r,l) in modes.items():
            self.res_mode=r;self.lex_mode=l;scores[name]=self.wordloss(wd)["full_word_nll"]
        self.word_mode=min(scores,key=scores.get);self.res_mode,self.lex_mode=modes[self.word_mode]
        return {"selected":self.word_mode,"nll":scores}
    def bridge_rows(self,ref,small,large,cal,files,backoff,seed,events=12):
        rng=np.random.default_rng(seed);rows=[]
        for bid,path in files:
            with path.open("rb") as f:raw=f.read(8192)
            if len(raw)<256:continue
            starts=[];start=0
            for i,b in enumerate(raw):
                if not 97<=b<=122 or i-start==63:
                    if start>=128:starts.append((start,i+1))
                    start=i+1
            if not starts:continue
            for k in sorted(rng.choice(len(starts),min(events,len(starts)),replace=False)):
                start,end=starts[k];prefix=raw[:start];state=self.session()
                for word in A.words(prefix):state.observe(word)
                orig,_=V.branches(ref,small,large,state,cal);base,hidden=self.field_law(state,orig[1].law);law,_=self.combine(base,hidden,state.attractor.query())
                event=raw[start:end];raw_probs=[self.predict_token_byte(raw[:i],backoff).astype(np.float64) for i in range(start,end)]
                raw_probs=[q/q.sum() for q in raw_probs]
                rows.append((bid,event,law,raw_probs))
        assert rows;return rows
    def bridge_metrics(self,rows,weights):
        results={}
        for weight in weights:
            bits=0.;n=t1=t5=0;canonical_n=0
            for bid,event,law,raw in rows:
                state=WB.Bridge(law,weight);product=1.;rawproduct=1.
                for b,q in zip(event,raw):
                    d=state.probability(q);rank=np.argsort(-d,kind="stable")[:5];n+=1;t1+=int(rank[0]==b);t5+=int(b in rank)
                    p=state.observe(b,q);bits-=math.log2(p);product*=p;rawproduct*=q[b]
                assert state.done
                canon=0.
                if event[-1]==32 and len(event)>1:
                    canon=law.complete(event[:-1].decode("ascii"));canonical_n+=1
                elif len(event)==64 and all(97<=b<=122 for b in event):
                    canon=law.mass(event.decode("ascii"))
                assert math.isclose(product,(1-weight)*rawproduct+weight*canon,rel_tol=1e-7,abs_tol=1e-30)
            results[str(weight)]={"bpb":bits/n,"bytes":n,"byte_top1":t1/n,"byte_top5":t5/n,"events":len(rows),"canonical_events":canonical_n}
        return results
    def save(self,path):
        parent=self.v25_sha
        try:self.v25_sha=self.v36_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent
        path.with_suffix(".token.json").write_text(json.dumps({"split":list(Y.SPLIT),"trust":list(self.trust),"vocabulary":len(Y.TOKENS),"parent_sha256":self.v36_sha},indent=2))
        path.with_suffix(".memory.json").write_text(json.dumps({"scale":self.memory_scale,"capacity":AM.CAP,"candidates":AM.K,"parent_sha256":self.v36_sha},indent=2))
        path.with_suffix(".unified.json").write_text(json.dumps({"word_mode":self.word_mode,"bridge_weight":self.bridge_weight,"parent_sha256":self.v36_sha,"accepted_step":self.step},indent=2))

def word_eval(field,ref,small,large,cal,docs):
    m=C.defaultdict(C.Counter)
    for doc in docs:
        a=field.session();b=field.parent.session()
        for i,w in enumerate(doc[:88]):
            if i>=16:
                oa,_=V.branches(ref,small,large,a,cal);ob,_=V.branches(ref,small,large,b,cal)
                x,hh=field.field_law(a,oa[1].law);y,hp=field.parent.field_law(b,ob[1].law)
                x,_=field.combine(x,hh,a.attractor.query());y,_=field.parent.combine(y,hp,b.attractor.query())
                for mode,law in (("v36",y),("v37",x)):
                    l=H.Mixture((law,),(1.,));rank,cert,_=l.rank(budget=2000);z=m[mode];z["n"]+=1;z["t1"]+=w==rank[0];z["t5"]+=w in rank;z["cert"]+=cert
                    z["bits"]-=math.log2(max(l.complete(w),1e-300));z["symbols"]+=len(w)+1
            a.observe(w);b.observe(w)
    return {k:dict(v,word_top1=v["t1"]/v["n"],word_top5=v["t5"]/v["n"],spelling_bps=v["bits"]/v["symbols"]) for k,v in m.items()}

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("metrics.json"));a=ap.parse_args();r=a.data
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
    field=Unified(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,a.attractor,len(data[1]));field.bind(ref,small,large,cal)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    train,trainids=field.token_data(extra,backoff,24,3701);valid,validids=field.token_data(calbooks,backoff,24,3702,True)
    word,_=field.worddata(expanded,cap=96);wd,_=field.worddata(cdocs,cap=96)
    assert np.array_equal(field.forward(np.concatenate((word[0][:8,:40],word[0][:8,41:]),axis=1))[0],field.parent.forward(np.concatenate((word[0][:8,:40],word[0][:8,41:]),axis=1))[0])
    before=field.wordloss(wd);token_before=field.metrics(valid)
    field.trust=tuple(max(.125,v) for v in field.trust)
    joint=field.train_joint(word,train);after_joint=field.wordloss(wd)
    mode=field.select_word_mode(wd);gate=field.calibrate(valid)
    # Recompute cached field states before fitting memory. Word-code frozen.
    selected=extra[::12];mt=field.collect(ref,small,large,cal,[B.load(p,1200) for _,p in selected],16,3703)
    mv=field.collect(ref,small,large,cal,[B.load(p,1200) for _,p in calbooks],16,3704)
    memory_trace=field.train_memory(mt,128);mgate=field.calibrate_memory(mv)
    assert np.array_equal(field.p["word_code"],field.parent.p["word_code"]) and np.array_equal(field.lex,field.parent.lex)
    grid=(0.,.0625,.125,.25,.5)
    br=field.bridge_rows(ref,small,large,cal,calbooks,backoff,3705,8);scores=field.bridge_metrics(br,grid)
    field.bridge_weight=min(grid,key=lambda k:scores[str(k)]["bpb"])
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5]
    forbidden={i for i,p in tr+extra+calbooks};assert len(picked)==5 and not forbidden&{i for i,p in picked}
    dev={}
    held={i for i,p in picked}
    for name,files in groups.items():
        chosen=picked if name=="stories" else [(i,p) for i,p in files if split(i) in (7,8) and i not in held][:6]
        assert chosen and not forbidden&{i for i,p in chosen};held.update(i for i,p in chosen)
        ds,ids=field.token_data(chosen,backoff,48,3406,True,offset=1024)
        rows=field.bridge_rows(ref,small,large,cal,chosen,backoff,3706,16)
        dev[name]={"ids":sorted(set(ids)),"parent_tokens":field.parent.metrics(ds),"tokens":field.metrics(ds),"bridge":field.bridge_metrics(rows,sorted({0.,field.bridge_weight,.25}))}
    result={"parent_sha256":field.v36_sha,"word_before":before,"word_after_joint":after_joint,"word_mode":mode,"joint":joint,"token_before":token_before,"token_calibration":gate,"memory_trace":memory_trace,"memory_calibration":mgate,"bridge_calibration":scores,"bridge_weight":field.bridge_weight,"dev":dev,"word_dev":word_eval(field,ref,small,large,cal,[B.load(p,300) for _,p in picked]),"group_steps":field.steps,"step":field.step,"raw_canonical_event_consistency":True,"unrestricted_word_marginal_consistency":False}
    field.save(a.out.with_name("unified-field.npz"));a.out.write_text(json.dumps(result,indent=2));print("FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
