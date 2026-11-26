"""Stage 7: competitive anchor-conditioned routing, dense signed readout,
and a bounded counterfactual word likelihood objective."""
import hybrid34 as Z4, hybrid33 as X, hybrid32 as W, hybrid31 as I, hybrid30 as M, hybrid29 as Q, hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, coordinate_exchange as E, temporal_phasor as T, triangle_field as G, lexical_geometry as J, byte_geometry as K, raw_byte_law as A, subword_geometry as Y, selective_resonance as RZ
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib

def load34(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v33_sha
        assert int(ck["step"])==6784
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v34_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    meta=json.loads(path.with_suffix(".token.json").read_text())
    assert meta["split"]==list(Y.SPLIT) and meta["vocabulary"]==len(Y.TOKENS) and meta["parent_sha256"]==f.v33_sha
    f.trust=tuple(meta["trust"]);assert len(f.trust)==2 and all(0<=v<=1 for v in f.trust)
    return f








class Resonance(Z4.SubwordField):
    res_keys=("res_support","res_oppose","res_route","res_plus","res_minus")
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,available);load34(self,p34)
        self.parent=load34(Z4.SubwordField(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,available),p34)
        self.inherited_keys=tuple(self.p);self.res_mode="full"
        for k in self.res_keys:
            a=np.full((8,256),.1 if k in self.res_keys[:2] else 0.,np.float32) if k in self.res_keys[:3] else np.zeros((256,128),np.float32)
            self.p[k]=a;self.m[k]=np.zeros_like(a);self.v[k]=np.zeros_like(a);self.steps[k]=0

    def _extension(self,x,hidden):
        p={k:torch.tensor(self.p[k],requires_grad=True) for k in (*self.new_keys,*self.coord_keys,*self.temporal_keys,*self.triangle_keys,*self.word_keys[:3],*self.byte_keys,*self.sub_keys,*self.res_keys)}
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
        if self.token_context is not None and not self.token_off:
            tc=torch.as_tensor(self.token_context,dtype=torch.long)
            comp=Y.compose(tc,p["byte_code"],p["sub_pair"])
            s=s+.7*(comp@p["sub_lower"]).tanh()
            z=z+.7*(comp@p["sub_upper"]).tanh()
        anchors=torch.cat((s,z),-1).tanh()
        gate=RZ.allocation(anchors,p["res_support"],p["res_oppose"])
        uniform=self.res_mode=="uniform"
        routing=RZ.route(gate,p["res_route"],uniform)
        if self.res_mode=="off":routing=torch.ones_like(routing)
        lo=s.tanh();hi=z.tanh()
        def transfer(a,b,positive,negative):
            return torch.einsum("cij,bcj->bci",positive,a.relu())-torch.einsum("cij,bcj->bci",negative,b.square())
        edges=None
        def advance(lo,hi,edges):
            points=torch.cat((lo,hi),-1)
            edges=G.edge_update(points,None if self.edge_reset else edges,self.topology[1])
            msg=G.message(edges,p["tri_plus"],p["tri_minus"],self.topology,self.triangle_mode)
            return ((s+ routing[:,:,:128]*transfer(lo,lo,lp,ln)+routing[:,:,:128]*transfer(hi,hi,lu,lv)+routing[:,:,:128]*E.exchange(lo,xp,xn)+routing[:,:,:128]*msg[:,:,:128]).tanh(),
                    (z+ routing[:,:,128:]*transfer(hi,hi,up,un)+routing[:,:,128:]*transfer(lo,lo,ul,uv)+routing[:,:,128:]*E.exchange(hi,yp,yn)+routing[:,:,128:]*msg[:,:,128:]).tanh(),edges)
        for _ in range(4):lo,hi,edges=advance(lo,hi,edges)
        lnext,hnext,enext=advance(lo,hi,edges)
        residual=max(float((lnext-lo).detach().abs().max()),float((hnext-hi).detach().abs().max()),float((enext-edges).detach().abs().max()))
        joint=torch.cat((lo,hi),dim=1).transpose(1,2)
        out=h+.35*(joint*p["field_out"]).sum(-1).tanh()
        if self.res_mode!="off":
            out=out+RZ.read(torch.cat((lo,hi),-1),gate,p["res_plus"],p["res_minus"],uniform)
        if self.ablate=="readout":out=h
        return out,h,p,residual







    def update(self,g,lg,keys):
        self.step+=1
        for k in keys:
            if k not in g:continue
            a=g[k];self.steps[k]+=1;j=self.steps[k]
            self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
            lr=.002 if k in self.res_keys else .0001
            self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
            if k not in ("read","bias","field_out",*self.fusion_keys,*self.temporal_keys,*self.word_keys,*self.byte_keys,*self.sub_keys,*self.res_keys):np.clip(self.p[k],0.,3.,out=self.p[k])
            assert np.isfinite(self.p[k]).all()
        self.p["word_code"][0].fill(0);self.p["byte_code"][256].fill(0)
        self.p["fusion_class"][-1]=max(0.,float(self.p["fusion_class"][-1]))
        if lg:
            for c,a in lg.items():
                rows=self.groups[c];self.lt[c]+=1;j=int(self.lt[c])
                self.lm[rows]=.9*self.lm[rows]+.1*a;self.lv[rows]=.999*self.lv[rows]+.001*a*a
                self.lex[rows]-=.001*(self.lm[rows]/(1-.9**j))/(np.sqrt(self.lv[rows]/(1-.999**j))+1e-8)


    def check_res(self,word,token):
        x,y=word;x=x[:16];y=y[:16];inp=np.concatenate((x[:,:40],x[:,41:]),axis=1)
        assert np.array_equal(self.forward(inp)[0],self.parent.forward(inp)[0])
        args=[a[:8] for a in token[:-1]]
        for a,b in zip(self.token_forward(*args),self.parent.token_forward(*args)):assert np.array_equal(a,b)
        rng=np.random.default_rng(3502);saved={k:self.p[k].copy() for k in self.res_keys}
        for k in self.res_keys[2:]:self.p[k][:]=rng.normal(0,.03,self.p[k].shape)
        _,g,_=self.direct(x,y,True);errs={};norms={}
        for k in self.res_keys:
            a=self.p[k];ix=np.unravel_index(np.abs(g[k]).argmax(),a.shape);z=float(a[ix]);eps=.004
            a[ix]=z+eps;hi=self.direct(x,y)[0];a[ix]=z-eps;lo=self.direct(x,y)[0];a[ix]=z
            errs[k]=abs((hi-lo)/(2*eps)-float(g[k][ix]));norms[k]=float(np.linalg.norm(g[k]))
            assert errs[k]<.002 and norms[k]>0,(k,errs[k],norms[k])
        for k,a in saved.items():self.p[k][:]=a
        return {"word_and_token_parent_exact":True,"gradient_errors":errs,"gradient_norms":norms}
    def wrong(self,x,source):
        # Preserve last two words and legacy short context; replace extended
        # history with a different document. Record index/target and retrieval stay fixed.
        z=x.copy();z[:,8:40]=source[:,8:40];z[:,41:41+8*T.LENGTH]=source[:,41:41+8*T.LENGTH]
        z[:,-6:]=source[:,-6:];return z
    def paired(self,x,y,source,gradient=True):
        # Bounded hinge encourages correct history to win by .15 nats.
        # Wrong-context gradients are restricted to the new resonance groups.
        bad=self.wrong(x,source)
        loss,g,lg=self.direct(x,y,True);other,bg,_=self.direct(bad,y,True)
        gap=other-loss;active=gap<.15
        if active:
            for k in self.res_keys:g[k]=1.2*g[k]-.2*bg[k]
        return loss,g,lg,gap
    def train_res(self,word,token,steps,new_only):
        x,y=word;trace=[]
        # Map each row to its actual document; ensure negative belongs to another.
        doc=np.asarray([id(self.records[int(row[40])][0]) for row in x],np.int64)
        for i in range(steps):
            if not new_only and i%4==3:
                ix=self.rng.integers(len(token[-1]),size=16)
                loss,g=self.token_forward(*(a[ix] for a in token),gradient=True);self.update(g,{},tuple(self.p));gap=None
            else:
                ix=self.rng.integers(len(y),size=16);neg=self.rng.integers(len(y),size=16)
                while np.any(doc[ix]==doc[neg]):
                    mask=doc[ix]==doc[neg];neg[mask]=self.rng.integers(len(y),size=int(mask.sum()))
                loss,g,lg,gap=self.paired(x[ix],y[ix],x[neg])
                self.update(g,{} if new_only else lg,self.res_keys if new_only else tuple(self.p))
            if i%32 in (0,3) or i==steps-1:trace.append([i+1,loss,gap])
        return {"steps":steps,"new_only":new_only,"trace":trace}
    def diagnostics(self,word):
        x,y=word;x=x[:256];y=y[:256]
        full=self.wordloss((x,y));ab={}
        for mode in ("uniform","off"):
            self.res_mode=mode
            try:ab[mode]=self.wordloss((x,y))
            finally:self.res_mode="full"
        # A different calibration document for every negative, including wraparound.
        docs=np.asarray([id(self.records[int(row[40])][0]) for row in x],np.int64)
        ix=np.asarray([next(j for j in range(len(x)) if docs[j]!=docs[i]) for i in range(len(x))])
        wrong=self.wordloss((self.wrong(x,x[ix]),y))
        return {"full":full,"ablations":ab,"wrong_extended_history":wrong,"correct_history_advantage_nats":wrong["full_word_nll"]-full["full_word_nll"],"channel_l2":{k:float(np.linalg.norm(self.p[k])) for k in self.res_keys},"new_parameters":sum(self.p[k].size for k in self.res_keys)}
    def save(self,path):
        parent=self.v25_sha
        try:self.v25_sha=self.v34_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent
        path.with_suffix(".token.json").write_text(json.dumps({"split":list(Y.SPLIT),"trust":list(self.trust),"vocabulary":len(Y.TOKENS),"parent_sha256":self.v34_sha},indent=2))

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword"):
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
    field=Resonance(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,len(data[1]));field.bind(ref,small,large,cal)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    train,trainids=field.token_data(extra,backoff,24,3404);valid,validids=field.token_data(calbooks,backoff,24,3405,True)
    word,_=field.worddata(expanded,cap=96)
    wd,_=field.worddata(cdocs,cap=96)
    checks=field.check_res(word,valid);before=field.metrics(valid);pre_word=field.wordloss(wd)
    # Train with nonzero trust, then freeze calibration-only gate again.
    field.trust=tuple(max(.25,v) for v in field.trust)
    p1=field.train_res(word,train,256,True)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) and field.steps[k]==field.parent.steps[k] for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    p2=field.train_res(word,train,256,False)
    uncal=field.metrics(valid);gate=field.calibrate(valid);after=field.metrics(valid)
    diag=field.diagnostics(wd);post_word=field.wordloss(wd)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    forbidden={i for i,p in tr+extra+calbooks};held={i for i,p in picked};assert not held&forbidden
    dev={}
    for name,files in groups.items():
        chosen=picked if name=="stories" else [(i,p) for i,p in files if split(i) in (7,8) and i not in held][:6]
        assert chosen and not ({i for i,p in chosen}&forbidden)
        held.update(i for i,p in chosen);ds,ids=field.token_data(chosen,backoff,48,3406,True,offset=1024)
        dev[name]={"ids":sorted(set(ids)),"metrics":field.metrics(ds),"parent":field.parent.metrics(ds)}
    raw=first(picked[0][1]);prefix=raw[:128];target=raw[128:144]
    pred=lambda h:field.predict_token_byte(h,backoff)
    code=A.encode(prefix,target,pred);assert A.decode(prefix,code,len(target),pred)==target
    words=X.word_metrics(field,ref,small,large,cal,[B.load(p,300) for _,p in picked]);words={"v34":words["v32"],"v35":words["v33"]}
    field.save(a.out.with_name("resonance-field.npz"))
    result={"parent_sha256":field.v34_sha,"checks":checks,"word_cal_before":pre_word,"word_cal_after":post_word,"resonance":diag,"before":before,"uncalibrated":uncal,"calibration":gate,"after":after,"training":[p1,p2],"dev":dev,"word_dev":words,"codec":{"bytes":len(target),"bits":code[1],"roundtrip":True},"token_vocabulary":len(Y.TOKENS),"exact_byte_token_marginal":True,"raw_byte_word_marginal_consistency":False,"train_ids":sorted(set(trainids)),"cal_ids":sorted(set(validids)),"group_steps":field.steps}
    a.out.write_text(json.dumps(result,indent=2));print("FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
