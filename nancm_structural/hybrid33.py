"""Stage 5 / v33: native raw-byte law sharing signed recurrent geometry.
Byte and word objectives share parameters; raw byte law is not yet a marginal
of the inherited lower-case word law. No full-scale training."""
import hybrid32 as W, hybrid31 as I, hybrid30 as M, hybrid29 as Q, hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, coordinate_exchange as E, temporal_phasor as T, triangle_field as G, lexical_geometry as J, byte_geometry as K, raw_byte_law as A
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib

def load32(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v31_sha
        assert int(ck["step"])==6144
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v32_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return f






class ByteField(W.LexicalGeometry):
    byte_keys=("byte_code","byte_lower","byte_upper","byte_query","byte_local","byte_bias")
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,available);load32(self,p32)
        self.parent=load32(W.LexicalGeometry(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,available),p32)
        self.inherited_keys=tuple(self.p);self.byte_context=None;self.byte_mode="full"
        rng=np.random.default_rng(3302)
        c=rng.normal(0,.25,(257,K.DIM)).astype(np.float32);c[256]=0
        shapes=((K.DIM,128),(K.DIM,128),(128,K.DIM),(K.DEPTH,K.DIM,K.DIM),(256,))
        for k,a in zip(self.byte_keys,[c]+[np.zeros(s,np.float32) for s in shapes]):
            self.p[k]=a;self.m[k]=np.zeros_like(a);self.v[k]=np.zeros_like(a);self.steps[k]=0

    def _extension(self,x,hidden):
        p={k:torch.tensor(self.p[k],requires_grad=True) for k in (*self.new_keys,*self.coord_keys,*self.temporal_keys,*self.triangle_keys,*self.word_keys[:3],*self.byte_keys)}
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
        xp,xn,yp,yn=(E.matrices(p[k]) for k in self.coord_keys)
        if self.ablate=="feedback":lu=lu*0;lv=lv*0
        ix=torch.as_tensor(x[:,8:40].astype(np.int64)).reshape(-1,8,4)
        s=torch.zeros((len(x),8,128)).scatter_add_(2,ix,torch.full(ix.shape,.25))
        z=h[:,None,:].expand(-1,8,-1)
        rho,theta,gain=T.coefficients(p["time_decay"],p["time_phase"],p["time_amp"])
        events=torch.as_tensor(x[:,40:40+8*T.LENGTH].astype(np.int64)).reshape(-1,T.LENGTH,2,4)
        mem=T.scan(events,rho,theta,gain) if self.stream_override is None else self.stream_override
        c=T.contrast(mem)
        s=s+.5*c*p["time_lower"].tanh()
        z=z+.5*c*p["time_upper"].tanh()
        ids=torch.as_tensor(x[:,40+8*T.LENGTH:].astype(np.int64))
        a,b=J.excite(ids,p["word_code"],p["word_lower"],p["word_upper"])
        if self.lex_mode!="input_off":s=s+a;z=z+b
        if self.byte_context is not None and self.byte_mode not in ("input_off","local_only"):
            ids=torch.as_tensor(self.byte_context,dtype=torch.long)
            ba,bb=K.excite(ids,p["byte_code"],p["byte_lower"],p["byte_upper"])
            s=s+ba;z=z+bb
        lo=s.tanh();hi=z.tanh()
        def transfer(a,b,positive,negative):
            return torch.einsum("cij,bcj->bci",positive,a.relu())-torch.einsum("cij,bcj->bci",negative,b.square())
        edges=None
        def advance(lo,hi,edges):
            points=torch.cat((lo,hi),-1)
            edges=G.edge_update(points,None if self.edge_reset else edges,self.topology[1])
            msg=G.message(edges,p["tri_plus"],p["tri_minus"],self.topology,self.triangle_mode)
            return ((s+transfer(lo,lo,lp,ln)+transfer(hi,hi,lu,lv)+E.exchange(lo,xp,xn)+msg[:,:,:128]).tanh(),
                    (z+transfer(hi,hi,up,un)+transfer(lo,lo,ul,uv)+E.exchange(hi,yp,yn)+msg[:,:,128:]).tanh(),edges)
        for _ in range(4):lo,hi,edges=advance(lo,hi,edges)
        lnext,hnext,enext=advance(lo,hi,edges)
        residual=max(float((lnext-lo).detach().abs().max()),float((hnext-hi).detach().abs().max()),float((enext-edges).detach().abs().max()))
        joint=torch.cat((lo,hi),dim=1).transpose(1,2)
        out=h+.35*(joint*p["field_out"]).sum(-1).tanh()
        if self.ablate=="readout":out=h
        return out,h,p,residual





    def raw_row(self,prefix,backoff):
        words=A.words(prefix)
        return self.features(words)+self.events(words)+self.ids(words),A.context(prefix),backoff.prob(prefix)
    def byte_data(self,files,backoff,count,seed,offset=0):
        rng=np.random.default_rng(seed);xs=[];bs=[];ps=[];ys=[];ids=[]
        for bid,path in files:
            with path.open("rb") as f:f.seek(offset);buf=f.read(16384)
            if len(buf)<256:continue
            positions=np.sort(rng.choice(np.arange(128,len(buf)),min(count,len(buf)-128),replace=False))
            for pos in positions:
                x,b,p=self.raw_row(buf[:pos],backoff)
                xs.append(x);bs.append(b);ps.append(p);ys.append(buf[pos]);ids.append(bid)
        assert ys
        return (np.asarray(xs,np.uint32),np.asarray(bs,np.int64),np.asarray(ps,np.float32),np.asarray(ys,np.int64)),ids
    def byte_forward(self,x,b,prior,y=None,gradient=False):
        assert self.byte_context is None
        self.byte_context=b
        try:
            _,_,v,tape=self.tape(x)
            parent_tape,out,h,p,_=tape
            ids=torch.as_tensor(b,dtype=torch.long)
            hidden=out if self.byte_mode!="local_only" else torch.zeros_like(out)
            scores=K.logits(hidden,ids,p["byte_code"],p["byte_query"],p["byte_local"],p["byte_bias"],torch.as_tensor(prior))
            if not gradient:return scores.softmax(1).detach().numpy()
            loss=torch.nn.functional.cross_entropy(scores,torch.as_tensor(y,dtype=torch.long))
            loss.backward()
            g=D.Dual.back(self,x[:,:8],v,parent_tape,h.grad.numpy())
            g.update({k:a.grad.numpy().copy() for k,a in p.items() if a.grad is not None})
            return float(loss.detach()),g
        finally:self.byte_context=None
    def predict_raw(self,prefix,backoff):
        x,b,p=self.raw_row(prefix,backoff)
        return self.byte_forward(np.asarray([x],np.uint32),np.asarray([b],np.int64),np.asarray([p],np.float32))[0]
    def byte_check(self,data):
        x,b,p,y=(a[:12] for a in data)
        q=self.byte_forward(x,b,p)
        assert np.allclose(q,p,rtol=2e-6,atol=1e-8)
        assert np.array_equal(self.forward(x)[0],self.parent.forward(x)[0])
        rng=np.random.default_rng(3303);saved={k:self.p[k].copy() for k in self.byte_keys[1:]}
        for k in saved:self.p[k][:]=rng.normal(0,.04,self.p[k].shape)
        loss,g=self.byte_forward(x,b,p,y,True);errors={};norms={}
        def f():
            a=self.byte_forward(x,b,p);return float(-np.log(a[np.arange(len(y)),y]).mean())
        for k in self.byte_keys:
            a=self.p[k];ix=np.unravel_index(np.abs(g[k]).argmax(),a.shape);z=float(a[ix]);eps=.005
            a[ix]=z+eps;hi=f();a[ix]=z-eps;lo=f();a[ix]=z
            err=abs((hi-lo)/(2*eps)-float(g[k][ix]));assert err<.002,(k,err);errors[k]=err
            norms[k]=float(np.linalg.norm(g[k]));assert norms[k]>0,k
        for k,a in saved.items():self.p[k][:]=a
        assert np.array_equal(self.forward(x)[0],self.parent.forward(x)[0])
        return {"byte_prior_parity":True,"word_parent_parity":True,"gradient_errors":errors,"gradient_norms":norms}
    def update(self,g,lg,keys):
        self.step+=1
        for k in keys:
            if k not in g:continue
            a=g[k];self.steps[k]+=1;j=self.steps[k]
            self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
            lr=.004 if k in self.byte_keys else .0003
            self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
            if k not in ("read","bias","field_out",*self.fusion_keys,*self.temporal_keys,*self.word_keys,*self.byte_keys):np.clip(self.p[k],0.,3.,out=self.p[k])
            assert np.isfinite(self.p[k]).all()
        self.p["word_code"][0].fill(0);self.p["byte_code"][256].fill(0)
        self.p["fusion_class"][-1]=max(0.,float(self.p["fusion_class"][-1]))
        if lg:
            for c,a in lg.items():
                rows=self.groups[c];self.lt[c]+=1;j=int(self.lt[c])
                self.lm[rows]=.9*self.lm[rows]+.1*a;self.lv[rows]=.999*self.lv[rows]+.001*a*a
                self.lex[rows]-=.001*(self.lm[rows]/(1-.9**j))/(np.sqrt(self.lv[rows]/(1-.999**j))+1e-8)
    def train_byte(self,raw,word,steps,new_only):
        trace=[];start=time.monotonic();wx,wy=word
        for t in range(steps):
            if new_only or t%2==0:
                ix=self.rng.integers(len(raw[-1]),size=16)
                loss,g=self.byte_forward(*(a[ix] for a in raw),gradient=True)
                self.update(g,{},self.byte_keys if new_only else tuple(self.p));kind="byte"
            else:
                ix=self.rng.integers(len(wy),size=16);loss,g,lg=self.direct(wx[ix],wy[ix],True)
                self.update(g,lg,self.inherited_keys);kind="word"
            if t%32 in (0,1) or t==steps-1:trace.append([t+1,kind,loss])
        return {"steps":steps,"new_only":new_only,"trace":trace,"seconds":time.monotonic()-start}
    def byte_metrics(self,data):
        x,b,p,y=data;tot=C.defaultdict(C.Counter)
        modes=("prior","full","input_off","local_only","wrong_word_context")
        start=time.monotonic()
        for i in range(0,len(y),16):
            j=min(i+16,len(y));xx=x[i:j];bb=b[i:j];pp=p[i:j];yy=y[i:j]
            for mode in modes:
                self.byte_mode=mode
                try:
                    q=pp if mode=="prior" else self.byte_forward(np.roll(x,len(x)//2,axis=0)[i:j] if mode=="wrong_word_context" else xx,bb,pp)
                finally:self.byte_mode="full"
                assert np.isfinite(q).all() and q.min()>0 and np.allclose(q.sum(1),1,atol=1e-6)
                order=np.argsort(-q,axis=1,kind="stable")[:,:5];m=tot[mode];m["n"]+=len(yy)
                m["bits"]-=float(np.log2(q[np.arange(len(yy)),yy]).sum());m["top1"]+=int((order[:,0]==yy).sum());m["top5"]+=int((order==yy[:,None]).any(1).sum())
        return {"modes":{k:dict(m,bpb=m["bits"]/m["n"],byte_top1=m["top1"]/m["n"],byte_top5=m["top5"]/m["n"]) for k,m in tot.items()},"seconds":time.monotonic()-start}
    def save(self,path):
        parent=self.v25_sha
        try:self.v25_sha=self.v32_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent

def word_metrics(field,ref,small,large,cal,docs):
    m=C.defaultdict(C.Counter)
    for doc in docs:
        s=field.session();parent=field.parent.session()
        for i,w in enumerate(doc[:88]):
            if i>=16:
                orig,_=V.branches(ref,small,large,s,cal);base=orig[1].law
                for name,l in (("v32",field.parent.law_stream(parent,base)),("v33",field.law_stream(s,base))):
                    law=H.Mixture((l,),(1.,));rank,cert,_=law.rank(budget=2000)
                    z=m[name];z["n"]+=1;z["t1"]+=w==rank[0];z["t5"]+=w in rank;z["cert"]+=cert
                    z["bits"]-=math.log2(max(law.complete(w),1e-300));z["symbols"]+=len(w)+1
            s.observe(w);parent.observe(w)
    return {k:dict(v,word_top1=v["t1"]/v["n"],word_top5=v["t5"]/v["n"],spelling_bps=v["bits"]/v["symbols"]) for k,v in m.items()}

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical"):
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
    cdocs=[B.load(p,200) for _,p in calbooks];cal=U.fit(ref,cdocs)
    small=V.Memory(docs);large=V.Memory(expanded);sem=Z.Semantic(expanded,large)
    data=P.Field(sem).data(expanded,cap=768)
    field=ByteField(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,len(data[1]));field.bind(ref,small,large,cal)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    raw,trainids=field.byte_data(extra,backoff,48,3304)
    vc,validids=field.byte_data(calbooks,backoff,48,3305)
    word,_=field.worddata(expanded,cap=96)
    check=field.byte_check(raw);pre=field.byte_metrics(vc)
    phase1=field.train_byte(raw,word,128,True);mid=field.byte_metrics(vc)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) and field.steps[k]==field.parent.steps[k] for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    phase2=field.train_byte(raw,word,256,False);post=field.byte_metrics(vc)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    trainset={i for i,p in tr+extra};calset={i for i,p in calbooks};assert not (trainset&calset)
    held={i for i,p in picked};assert not held&(trainset|calset)
    dev={}
    for name,files in groups.items():
        chosen=picked if name=="stories" else [(i,p) for i,p in files if split(i) in (7,8) and i not in held][:6]
        assert chosen and not ({i for i,p in chosen}&(trainset|calset))
        held.update(i for i,p in chosen);ds,ids=field.byte_data(chosen,backoff,96,3306,offset=1024)
        dev[name]={"ids":sorted(set(ids)),**field.byte_metrics(ds)}
    # Explicitly exercise arbitrary UTF-8 and punctuation with the real adaptive predictor.
    buf=first(picked[0][1]);prefix=buf[:128];target=buf[128:144]
    predict=lambda h:field.predict_raw(h,backoff)
    coded=A.encode(prefix,target,predict);assert A.decode(prefix,coded,len(target),predict)==target
    assert np.array_equal(predict(prefix),predict((prefix+b"unseen suffix")[:len(prefix)]))
    words=word_metrics(field,ref,small,large,cal,[B.load(p,300) for _,p in picked])
    field.save(a.out.with_name("raw-byte-field.npz"))
    result={"parent_sha256":field.v32_sha,"byte_gradient_check":check,"before":pre,"byte_only":mid,"after":post,"training":[phase1,phase2],"byte_dev":dev,"word_dev":words,"codec":{"bytes":len(target),"bits":coded[1],"roundtrip":True},"baseline_train_ids":sorted(trainset-set(i for i,p in extra)),"byte_train_ids":sorted(set(trainids)),"calibration_ids":sorted(set(validids)),"group_steps":field.steps,"raw_byte_word_marginal_consistency":False}
    a.out.write_text(json.dumps(result,indent=2));print("FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
