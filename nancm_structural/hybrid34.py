"""Stage 6 / v34: reversible compositional token law with exact byte marginals.
The lower-case word law is still separate. Token boundaries reset per document."""
import hybrid33 as X, hybrid32 as W, hybrid31 as I, hybrid30 as M, hybrid29 as Q, hybrid28 as F, hybrid27 as S, hybrid26 as D, hybrid25 as L, hybrid24 as R, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, coordinate_exchange as E, temporal_phasor as T, triangle_field as G, lexical_geometry as J, byte_geometry as K, raw_byte_law as A, subword_geometry as Y
import numpy as np
import torch
import argparse, collections as C, json, math, pathlib, time, hashlib

def load33(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v32_sha
        assert int(ck["step"])==6528
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v33_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    return f







class SubwordField(X.ByteField):
    sub_keys=("sub_pair","sub_lower","sub_upper","sub_query","sub_cond")
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,available);load33(self,p33)
        self.parent=load33(X.ByteField(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,available),p33)
        self.inherited_keys=tuple(self.p);self.token_context=None;self.token_off=False;self.trust=(.25,.25)
        rng=np.random.default_rng(3402)
        for k,shape in zip(self.sub_keys,((16,16),(16,128),(16,128),(128,16),(16,16))):
            a=np.zeros(shape,np.float32)
            if k=="sub_pair":a[:]=rng.normal(0,.04,shape)
            self.p[k]=a;self.m[k]=np.zeros_like(a);self.v[k]=np.zeros_like(a);self.steps[k]=0

    def _extension(self,x,hidden):
        p={k:torch.tensor(self.p[k],requires_grad=True) for k in (*self.new_keys,*self.coord_keys,*self.temporal_keys,*self.triangle_keys,*self.word_keys[:3],*self.byte_keys,*self.sub_keys)}
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






    def token_forward(self,x,b,t,p,first,p2,target=None,gradient=False):
        assert self.byte_context is None and self.token_context is None
        self.byte_context=b;self.token_context=t
        try:
            _,_,v,tape=self.tape(x);parent_tape,out,h,pp,_=tape
            prior=torch.as_tensor(p);prior2=torch.as_tensor(p2)
            f=torch.as_tensor(first,dtype=torch.long)
            r1=K.logits(out,torch.as_tensor(b,dtype=torch.long),pp["byte_code"],pp["byte_query"],pp["byte_local"],pp["byte_bias"],prior).softmax(-1)
            r2=Y.second_logits(out,f,pp["byte_code"],pp["sub_pair"],pp["sub_query"],pp["sub_cond"],prior2).softmax(-1)
            a,c=self.trust;q1=(1-a)*prior+a*r1;q2=(1-c)*prior2+c*r2
            if not gradient:return q1.detach().numpy(),q2.detach().numpy()
            y=torch.as_tensor(target,dtype=torch.long);idx=torch.arange(len(y))
            assert f.shape[1]==1 and torch.equal(f[:,0],y[:,0])
            one=q1[idx,y[:,0]];two=q2[idx,0,y[:,1].clamp_max(255)]
            mask=y[:,1]!=256
            loss=-(one.log()+torch.where(mask,two.log(),0.)).sum()/(len(y)+mask.sum())
            loss.backward()
            g=D.Dual.back(self,x[:,:8],v,parent_tape,h.grad.numpy())
            g.update({k:a.grad.numpy().copy() for k,a in pp.items() if a.grad is not None})
            return float(loss.detach()),g
        finally:self.byte_context=None;self.token_context=None
    def token_data(self,files,backoff,count,seed,all_branches=False,offset=0):
        rng=np.random.default_rng(seed);rows=[];ids=[]
        for bid,path in files:
            with path.open("rb") as f:f.seek(offset);buf=f.read(8192)
            toks,tail=Y.parse(buf);pos=np.r_[0,np.cumsum([len(t) for t in toks])]
            candidates=np.flatnonzero(pos[:-1]>=128)
            if len(candidates)<1:continue
            for j in sorted(rng.choice(candidates,min(count,len(candidates)),replace=False)):
                prefix=buf[:pos[j]];token=toks[j];x,b,p=self.raw_row(prefix,backoff)
                first=list(Y.SPLIT) if all_branches else [token[0]]
                p2=[backoff.prob(prefix+bytes([f])) for f in first]
                rows.append((x,b,Y.context(toks[max(0,j-8):j]),p,first,p2,[token[0],token[1] if len(token)==2 else 256]))
                ids.append(bid)
        assert rows
        return tuple(np.asarray(a,dtype=dt) for a,dt in zip(zip(*rows),(np.uint32,np.int64,np.int64,np.float32,np.int64,np.float32,np.int64))),ids
    def check_sub(self,ds):
        x,b,t,p,f,p2,y=(a[:12] for a in ds);assert np.array_equal(self.forward(x)[0],self.parent.forward(x)[0])
        assert np.allclose(self.byte_forward(x,b,p),self.parent.byte_forward(x,b,p),atol=0,rtol=0)
        rng=np.random.default_rng(3403);saved={k:self.p[k].copy() for k in self.sub_keys}
        for k in self.sub_keys[1:]:self.p[k][:]=rng.normal(0,.04,self.p[k].shape)
        loss,g=self.token_forward(x,b,t,p,f,p2,y,True);errors={};norms={}
        def fn():
            q1,q2=self.token_forward(x,b,t,p,f,p2);idx=np.arange(len(y));mask=y[:,1]!=256
            return float(-(np.log(q1[idx,y[:,0]]).sum()+np.log(q2[idx[mask],0,y[mask,1]]).sum())/(len(y)+mask.sum()))
        for k in self.sub_keys:
            a=self.p[k];ix=np.unravel_index(np.abs(g[k]).argmax(),a.shape);z=float(a[ix]);eps=.005
            a[ix]=z+eps;hi=fn();a[ix]=z-eps;lo=fn();a[ix]=z
            errors[k]=abs((hi-lo)/(2*eps)-float(g[k][ix]));norms[k]=float(np.linalg.norm(g[k]));assert errors[k]<.002 and norms[k]>0,(k,errors[k],norms[k])
        for k,a in saved.items():self.p[k][:]=a
        return {"word_parent_exact":True,"raw_parent_exact":True,"gradient_errors":errors,"gradient_norms":norms}
    def update(self,g,lg,keys):
        self.step+=1
        for k in keys:
            if k not in g:continue
            a=g[k];self.steps[k]+=1;j=self.steps[k]
            self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
            lr=.0015 if k in self.sub_keys else .00015
            self.p[k]-=lr*(self.m[k]/(1-.9**j))/(np.sqrt(self.v[k]/(1-.999**j))+1e-8)
            if k not in ("read","bias","field_out",*self.fusion_keys,*self.temporal_keys,*self.word_keys,*self.byte_keys,*self.sub_keys):np.clip(self.p[k],0.,3.,out=self.p[k])
            assert np.isfinite(self.p[k]).all()
        self.p["word_code"][0].fill(0);self.p["byte_code"][256].fill(0)
        self.p["fusion_class"][-1]=max(0.,float(self.p["fusion_class"][-1]))
        if lg:
            for c,a in lg.items():
                rows=self.groups[c];self.lt[c]+=1;j=int(self.lt[c])
                self.lm[rows]=.9*self.lm[rows]+.1*a;self.lv[rows]=.999*self.lv[rows]+.001*a*a
                self.lex[rows]-=.001*(self.lm[rows]/(1-.9**j))/(np.sqrt(self.lv[rows]/(1-.999**j))+1e-8)

    def train_sub(self,ds,word,steps,new_only):
        trace=[];wx,wy=word
        for i in range(steps):
            if new_only or i%2==0:
                ix=self.rng.integers(len(ds[-1]),size=16)
                loss,g=self.token_forward(*(a[ix] for a in ds),gradient=True)
                self.update(g,{},self.sub_keys if new_only else tuple(self.p));mode="token"
            else:
                ix=self.rng.integers(len(wy),size=16);loss,g,lg=self.direct(wx[ix],wy[ix],True)
                self.update(g,lg,self.inherited_keys);mode="word"
            if i%32 in (0,1) or i==steps-1:trace.append([i+1,mode,loss])
        return {"steps":steps,"new_only":new_only,"trace":trace}
    def target_mass(self,q1,q2,y):
        ix=np.arange(len(y));a=q1[ix,y[:,0]];b=np.ones(len(y),np.float64);mask=y[:,1]!=256
        branches=np.array([Y.SPLIT.index(int(k)) for k in y[mask,0]],np.int64)
        b[mask]=q2[ix[mask],branches,y[mask,1]]
        return a,b,mask
    def calibrate(self,ds):
        self.trust=(1.,1.);x,b,t,p,f,p2,y=ds
        a=[];c=[]
        for i in range(0,len(y),16):
            q1,q2=self.token_forward(*(v[i:i+16] for v in ds[:-1]));a.append(q1);c.append(q2)
        a=np.concatenate(a);c=np.concatenate(c)
        g1,g2,mask=self.target_mass(a,c,y);b1,b2,_=self.target_mass(p,p2,y)
        grid=(0.,.125,.25,.5,.75,1.)
        first=min(grid,key=lambda z:-np.log((1-z)*b1+z*g1).sum())
        second=min(grid,key=lambda z:-np.log((1-z)*b2+z*g2).sum())
        self.trust=(first,second)
        return {"first":first,"second":second,"grid":list(grid)}
    def metrics(self,ds):
        x,b,t,p,f,p2,y=ds;m=C.defaultdict(C.Counter)
        for i in range(0,len(y),16):
            s=slice(i,i+16);yy=y[s]
            q1,q2=self.token_forward(*(v[s] for v in ds[:-1]))
            self.token_off=True
            try:u1,u2=self.token_forward(*(v[s] for v in ds[:-1]))
            finally:self.token_off=False
            w1,w2=self.token_forward(x[s],b[s],np.roll(t,len(t)//2,axis=0)[s],p[s],f[s],p2[s])
            for mode,a,c in (("count",p[s],p2[s]),("full",q1,q2),("token_input_off",u1,u2),("wrong_token_history",w1,w2)):
                leaf=Y.leaves(a,c);assert np.allclose(leaf.sum(1),1.,atol=2e-6)
                assert np.allclose(leaf[:,248:].reshape(len(yy),8,256).sum(-1),a[:,list(Y.SPLIT)],atol=2e-6)
                idx=np.asarray([Y.INDEX[bytes([j]) if k==256 else bytes([j,k])] for j,k in yy])
                order=np.argsort(-leaf,axis=1,kind="stable")[:,:5]
                z=m[mode];z["n"]+=len(yy);z["top1"]+=int((order[:,0]==idx).sum());z["top5"]+=int((order==idx[:,None]).any(1).sum())
                first,second,mask=self.target_mass(a,c,yy);z["bits"]-=float(np.log2(first).sum()+np.log2(second).sum());z["bytes"]+=len(yy)+int(mask.sum())
                o=np.argsort(-a,axis=1,kind="stable")[:,:5]
                z["byte_t1"]+=int((o[:,0]==yy[:,0]).sum());z["byte_t5"]+=int((o==yy[:,0,None]).any(1).sum())
                ix=np.flatnonzero(mask);branches=np.array([Y.SPLIT.index(int(k)) for k in yy[mask,0]],np.int64)
                oo=np.argsort(-c[ix,branches],axis=1,kind="stable")[:,:5]
                z["byte_t1"]+=int((oo[:,0]==yy[mask,1]).sum());z["byte_t5"]+=int((oo==yy[mask,1,None]).any(1).sum())
        return {k:dict(v,bpb=v["bits"]/v["bytes"],token_top1=v["top1"]/v["n"],token_top5=v["top5"]/v["n"],byte_top1=v["byte_t1"]/v["bytes"],byte_top5=v["byte_t5"]/v["bytes"]) for k,v in m.items()}
    def predict_token_byte(self,prefix,backoff):
        toks,pending=Y.parse(prefix);boundary=prefix[:-1] if pending else prefix
        x,b,p=self.raw_row(boundary,backoff);first=[pending[0]] if pending else list(Y.SPLIT)
        p2=[backoff.prob(boundary+bytes([f])) for f in first]
        q1,q2=self.token_forward(np.asarray([x],np.uint32),np.asarray([b],np.int64),np.asarray([Y.context(toks)],np.int64),np.asarray([p],np.float32),np.asarray([first],np.int64),np.asarray([p2],np.float32))
        return q2[0,0] if pending else q1[0]
    def save(self,path):
        parent=self.v25_sha
        try:self.v25_sha=self.v33_sha;D.Dual.save(self,path)
        finally:self.v25_sha=parent
        path.with_suffix(".token.json").write_text(json.dumps({"split":list(Y.SPLIT),"trust":list(self.trust),"vocabulary":len(Y.TOKENS),"parent_sha256":self.v33_sha},indent=2))

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte"):
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
    field=SubwordField(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,len(data[1]));field.bind(ref,small,large,cal)
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    train,trainids=field.token_data(extra,backoff,24,3404);valid,validids=field.token_data(calbooks,backoff,24,3405,True)
    word,_=field.worddata(expanded,cap=96)
    # Gradient sample must exercise at least one two-byte token.
    both=np.flatnonzero(train[-1][:,1]!=256)[:12];assert len(both)==12
    checks=field.check_sub(tuple(a[both] for a in train));before=field.metrics(valid)
    p1=field.train_sub(train,word,128,True)
    assert all(np.array_equal(field.p[k],field.parent.p[k]) and np.array_equal(field.m[k],field.parent.m[k]) and np.array_equal(field.v[k],field.parent.v[k]) and field.steps[k]==field.parent.steps[k] for k in field.inherited_keys)
    assert all(np.array_equal(getattr(field,k),getattr(field.parent,k)) for k in ("lex","lm","lv","lt"))
    p2=field.train_sub(train,word,128,False)
    uncal=field.metrics(valid);gate=field.calibrate(valid);after=field.metrics(valid)
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    forbidden={i for i,p in tr+extra+calbooks};held={i for i,p in picked};assert not held&forbidden
    dev={}
    for name,files in groups.items():
        chosen=picked if name=="stories" else [(i,p) for i,p in files if split(i) in (7,8) and i not in held][:6]
        assert chosen and not ({i for i,p in chosen}&forbidden)
        held.update(i for i,p in chosen);ds,ids=field.token_data(chosen,backoff,48,3406,True,offset=1024)
        dev[name]={"ids":sorted(set(ids)),"metrics":field.metrics(ds)}
    raw=first(picked[0][1]);prefix=raw[:128];target=raw[128:144]
    pred=lambda h:field.predict_token_byte(h,backoff)
    code=A.encode(prefix,target,pred);assert A.decode(prefix,code,len(target),pred)==target
    words=X.word_metrics(field,ref,small,large,cal,[B.load(p,300) for _,p in picked])
    field.save(a.out.with_name("subword-field.npz"))
    result={"parent_sha256":field.v33_sha,"checks":checks,"before":before,"uncalibrated":uncal,"calibration":gate,"after":after,"training":[p1,p2],"dev":dev,"word_dev":words,"codec":{"bytes":len(target),"bits":code[1],"roundtrip":True},"token_vocabulary":len(Y.TOKENS),"exact_byte_token_marginal":True,"raw_byte_word_marginal_consistency":False,"train_ids":sorted(set(trainids)),"cal_ids":sorted(set(validids)),"group_steps":field.steps}
    a.out.write_text(json.dumps(result,indent=2));print("FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
