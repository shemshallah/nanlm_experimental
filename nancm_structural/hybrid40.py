"""v40: reciprocal memory-to-field excitation and evidence-conditioned triangles.
Frozen inherited checkpoint and v39 transport; fit new feedback with deployed word
likelihood. The raw-event decoder uses the same feedback, and remains normalized."""
import hybrid39 as S39, hybrid38 as P38, hybrid35 as R5, hybrid20 as V, hybrid22 as Z, hybrid23 as P, hybrid18 as U, hybrid15 as H
import base as B, foundation6 as N, temporal_phasor as T, coordinate_exchange as E, triangle_field as G, lexical_geometry as J, byte_geometry as K, subword_geometry as Y, selective_resonance as RZ, raw_byte_law as A, symbiotic_law as SL, feedback_geometry as FG
import argparse, collections as C, hashlib, json, math, pathlib, time
import torch
import numpy as np
from metric_io import write_metrics

KMAX=S39.KMAX
class Reciprocal(S39.Symbiosis):
    def __init__(self,*args,transport):
        super().__init__(*args)
        with np.load(transport,allow_pickle=False) as ck:
            assert str(ck["parent_sha256"].item())==self.v37_sha and int(ck["step"])==256
            self.theta=ck["theta"].copy();self.theta_m=ck["m"].copy();self.theta_v=ck["v"].copy()
        meta=json.loads(transport.with_suffix(".json").read_text())
        assert meta["parent_sha256"]==self.v37_sha and meta["event_cap"]==SL.CAP and meta["token_split"]==sorted(SL.SPLIT)
        self.mix=float(meta["mix"]);self.scale=float(meta["scale"]);self.raw_scale=float(meta["raw_scale"]);self.event_weight=float(meta["event_weight"])
        self.transport_sha=hashlib.sha256(transport.read_bytes()).hexdigest()
        self.transport_meta_sha=hashlib.sha256(transport.with_suffix(".json").read_bytes()).hexdigest()
        rng=np.random.default_rng(4001)
        self.fb={k:rng.normal(0,.04,s).astype(np.float32) for k,s in FG.SHAPES.items()}
        self.fm={k:np.zeros_like(a) for k,a in self.fb.items()};self.fv={k:np.zeros_like(a) for k,a in self.fb.items()}
        self.fb_step=0;self.fb_scale=0.;self.fb_context=None;self.relational_feedback=True
    def _extension(self,x,hidden):
        if self.fb_context is None or self.fb_scale==0:return P38.Portable._extension(self,x,hidden)
        device=self.device
        cpu={k:torch.tensor(self.p[k],requires_grad=False) for k in (*self.new_keys,*self.coord_keys,*self.temporal_keys,*self.triangle_keys,*self.word_keys[:3],*self.byte_keys,*self.sub_keys,*self.res_keys)}
        cpu.update({k:torch.tensor(a,requires_grad=True) for k,a in self.fb.items()})
        p={k:v.to(device) for k,v in cpu.items()}
        host=torch.tensor(hidden,requires_grad=True);h=host.to(device)
        ti,tj,topo=self.indices()
        def norm(a,scale):
            return scale*a/(1+a.abs().sum(-1,keepdim=True))
        def pair(k,scale):
            a=h.new_zeros((8,128,128));v=p[k].tanh()
            a[:,ti,tj]=v;a[:,tj,ti]=v
            return norm(a,scale)
        lp,ln=pair("field_lp",.22),pair("field_ln",.10)
        up,un=pair("field_up",.22),pair("field_un",.10)
        cp=p["field_cp"].tanh();cn=p["field_cn"].tanh()
        lu,lv=norm(cp,.22),norm(cn,.10)
        ul,uv=norm(cp.transpose(1,2),.22),norm(cn.transpose(1,2),.10)
        xp,xn,yp,yn=(E.matrices(p[k]) for k in self.coord_keys)
        if self.ablate=="feedback":lu=lu*0;lv=lv*0
        ix=torch.as_tensor(x[:,8:40].astype(np.int64),device=device).reshape(-1,8,4)
        s=h.new_zeros((len(x),8,128)).scatter_add_(2,ix,h.new_full(ix.shape,.25))
        z=h[:,None,:].expand(-1,8,-1)
        rho,theta,gain=T.coefficients(p["time_decay"],p["time_phase"],p["time_amp"])
        events=torch.as_tensor(x[:,40:40+8*T.LENGTH].astype(np.int64),device=device).reshape(-1,T.LENGTH,2,4)
        mem=T.scan(events,rho,theta,gain) if self.stream_override is None else self.stream_override.to(device)
        c=T.contrast(mem)
        s=s+.5*c*p["time_lower"].tanh()
        z=z+.5*c*p["time_upper"].tanh()
        ids=torch.as_tensor(x[:,40+8*T.LENGTH:].astype(np.int64),device=device)
        a,b=J.excite(ids,p["word_code"],p["word_lower"],p["word_upper"])
        if self.lex_mode!="input_off":s=s+a;z=z+b
        if self.byte_context is not None and self.byte_mode not in ("input_off","local_only"):
            ids=torch.as_tensor(self.byte_context,dtype=torch.long,device=device)
            ba,bb=K.excite(ids,p["byte_code"],p["byte_lower"],p["byte_upper"])
            s=s+ba;z=z+bb
        if self.token_context is not None and not self.token_off:
            tc=torch.as_tensor(self.token_context,dtype=torch.long,device=device)
            comp=Y.compose(tc,p["byte_code"],p["sub_pair"])
            s=s+.7*(comp@p["sub_lower"]).tanh()
            z=z+.7*(comp@p["sub_upper"]).tanh()
        f,n,valid,values=(torch.as_tensor(a,device=device) for a in self.fb_context)
        plo,phi=FG.pulse(f,n,valid,values,p)
        plo=plo*self.fb_scale;phi=phi*self.fb_scale
        signal=torch.cat((plo,phi),-1)
        s=s+plo;z=z+phi
        relation=FG.relation(signal,p["fb_relational"]) if self.relational_feedback else torch.ones_like(signal)
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
            edges=G.edge_update(points,None if self.edge_reset else edges,topo[1])
            msg=G.message(edges,p["tri_plus"],p["tri_minus"],topo,self.triangle_mode)*relation
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
        out=out+FG.read(torch.cat((lo,hi),-1),signal,p["fb_read"])
        return out.to('cpu'),host,cpu,residual
    def query_arrays(self,query):
        f,n,v,words=query;ids=[self.sem.ids.get(w,0) if valid else 0 for w,valid in zip(words,v)]
        return f,n,v,self.p["word_code"][np.asarray(ids,np.int64)]
    def make_row(self,state,ref,small,large,cal):
        assert self.fb_context is None
        ex,f,h,parent=self.branches39(ref,small,large,cal,state);base=self.consensus(ex,f);query=state.attractor.query()
        cand=self.candidates(ex,f,h,query,base);law=self.transported(base,cand)
        words=cand[0];ids=np.asarray([self.sem.ids.get(w,0) for w in words],np.int64)
        x=np.asarray(self.features(state.h)+self.events(state.h)+self.ids(state.h),np.uint32)
        trace=state.trace.detach().numpy()[0].copy()
        return law,words,x,trace,h,self.p["word_code"][ids],self.query_arrays(query)
    def feedback_out(self,x,trace,mem,scale,b=None,t=None):
        assert self.fb_context is None and self.stream_override is None and self.byte_context is None and self.token_context is None
        old=self.fb_scale
        self.fb_scale=scale;self.fb_context=mem;self.stream_override=None if trace is None else torch.as_tensor(trace)
        self.byte_context=b;self.token_context=t
        try:
            _,_,_,tape=self.tape(x)
            return tape[1],tape[3]
        finally:
            self.fb_context=None;self.stream_override=None;self.fb_scale=old;self.byte_context=None;self.token_context=None
    def score_row(self,row,scale,mem=None):
        law,words,x,trace,h,code,query=row
        if scale==0:return law,0.
        if mem is None:mem=query
        if not mem[2].any():return law,0.
        with torch.no_grad():
            out,p=self.feedback_out(x[None],trace[None],tuple(a[None] for a in mem),scale)
            delta=out-torch.as_tensor(h[None])
            lifts=FG.lift(delta,torch.as_tensor(code[None]),p["fb_query"])[0].numpy().astype(np.float64)
        return SL.Transport(law,words,lifts),float(np.linalg.norm(delta.numpy()))
    def collect40(self,docs,ref,small,large,cal,count,seed):
        rng=np.random.default_rng(seed);rows=[]
        for doc in docs:
            if len(doc)<80:continue
            pos=set(rng.choice(np.arange(32,min(len(doc),512)),min(count,min(len(doc),512)-32),replace=False).tolist())
            state=self.session()
            for i,w in enumerate(doc[:max(pos)+1]):
                if i in pos:
                    law,words,x,trace,h,code,mem=self.make_row(state,ref,small,large,cal)
                    pad=KMAX-len(words);assert pad>=0
                    prior=np.asarray([law.complete(k) for k in words],np.float64)
                    assert prior.sum()<=1+1e-8
                    hit=np.asarray([k==w for k in words],np.float64)
                    rows.append((x,trace,h,np.pad(code,((0,pad),(0,0))),np.pad(prior,(0,pad)),np.pad(hit,(0,pad)),max(law.complete(w),1e-300),*mem))
                state.observe(w)
        assert rows
        return tuple(np.asarray(a) for a in zip(*rows))
    def loss40(self,ds,scale=1.,grad=False):
        x,trace,h,code,prior,hit,target,*mem=ds
        if scale==0:return float(-np.log(target).mean())
        with torch.set_grad_enabled(grad):
            out,p=self.feedback_out(x,trace,tuple(mem),scale)
            lifts=FG.lift(out-torch.as_tensor(h),torch.as_tensor(code),p["fb_query"]).double()
            loss=FG.nll(lifts,torch.as_tensor(prior),torch.as_tensor(hit),torch.as_tensor(target))
            if not grad:return float(loss)
            loss.backward()
            g={k:a.grad.detach().numpy().copy() if a.grad is not None else np.zeros_like(self.fb[k]) for k,a in p.items() if k in self.fb}
            assert set(g)==set(self.fb) and all(np.isfinite(a).all() for a in g.values())
            return float(loss.detach()),g
    def check40(self,ds):
        x,trace,h,code,prior,hit,target,*mem=(a[:8] for a in ds)
        empty=list(mem);empty[2]=np.zeros_like(empty[2])
        with torch.no_grad():
            old,_=self.feedback_out(x,trace,tuple(empty),0.)
            new,_=self.feedback_out(x,trace,tuple(empty),1.)
            assert np.array_equal(old.numpy(),new.numpy())
        sample=(x,trace,h,code,prior,hit,target,*mem)
        loss,g=self.loss40(sample,grad=True)
        errors={};norms={}
        for k,a in self.fb.items():
            norm=float(np.linalg.norm(g[k]));norms[k]=norm
            ix=np.unravel_index(np.abs(g[k]).argmax(),a.shape);v=float(a[ix]);eps=.003
            a[ix]=v+eps;hi=self.loss40(sample);a[ix]=v-eps;lo=self.loss40(sample);a[ix]=v
            errors[k]=abs((hi-lo)/(2*eps)-float(g[k][ix]));assert errors[k]<.002,(k,errors[k])
        return {"empty_field_exact":True,"zero_scale_exact":self.loss40(sample,0.)==float(-np.log(target).mean()),"gradient_errors":errors,"gradient_norms":norms}
    def fit40(self,train,valid,steps=256):
        rng=np.random.default_rng(4002);log=[]
        before=self.loss40(valid,0.)
        for _ in range(steps):
            ix=rng.integers(len(train[0]),size=16);loss,g=self.loss40(tuple(a[ix] for a in train),grad=True);self.fb_step+=1;j=self.fb_step
            for k,a in g.items():
                self.fm[k]=.9*self.fm[k]+.1*a;self.fv[k]=.999*self.fv[k]+.001*a*a
                self.fb[k]-=.003*(self.fm[k]/(1-.9**j))/(np.sqrt(self.fv[k]/(1-.999**j))+1e-8)
                assert np.isfinite(self.fb[k]).all()
            if j%32==0 or j==1:log.append([j,loss])
        grid=(0.,.125,.25,.5,1.)
        # Chunked loss avoids retaining the entire corpus solver graph.
        scores={}
        for s in grid:
            scores[str(s)]=float(sum(self.loss40(tuple(a[i:i+16] for a in valid),s)*len(valid[0][i:i+16]) for i in range(0,len(valid[0]),16))/len(valid[0]))
        self.fb_scale=min(grid,key=lambda s:scores[str(s)])
        return {"train_targets":len(train[0]),"selection_targets":len(valid[0]),"parent_nll":before,"selection_nll":scores,"scale":self.fb_scale,"trace":log}
    def word_eval40(self,docs,ref,small,large,cal,long=False,limit=72):
        m=C.defaultdict(C.Counter);start=time.monotonic()
        for j,doc in enumerate(docs):
            state=self.session();wrong=self.bank();other=docs[(j+1)%len(docs)]
            pos=set(range(16,min(len(doc),16+limit))) if not long else set(range(128,min(len(doc),1200),17))
            if not pos:continue
            for i,w in enumerate(doc[:max(pos)+1]):
                if i in pos:
                    row=self.make_row(state,ref,small,large,cal)
                    full,delta=self.score_row(row,self.fb_scale);forced,fd=self.score_row(row,1.)
                    bad,bd=self.score_row(row,self.fb_scale,self.query_arrays(wrong.query(state.attractor.h)))
                    self.relational_feedback=False
                    try:off,od=self.score_row(row,self.fb_scale)
                    finally:self.relational_feedback=True
                    for name,law,d in (("v39",row[0],0.),("v40",full,delta),("forced",forced,fd),("wrong_feedback",bad,bd),("relation_off",off,od)):
                        l=H.Mixture((law,),(1.,));rank,cert,_=l.rank(budget=2000);z=m[name];z["n"]+=1;z["t1"]+=w==rank[0];z["t5"]+=w in rank;z["cert"]+=cert
                        p=law.complete(w);z["bits"]-=math.log2(max(p,1e-300));z["symbols"]+=len(w)+1;z["delta"]+=d
                        if name=="v40":
                            pref="";prod=1.
                            for c in w+B.EOS:
                                prod*=l.next(pref)[c]
                                if c!=B.EOS:pref+=c
                            assert math.isclose(prod,p,rel_tol=1e-6,abs_tol=1e-15)
                state.observe(w)
                if i<len(other):wrong.observe(other[i])
        return {"modes":{k:dict(v,word_top1=v["t1"]/v["n"],word_top5=v["t5"]/v["n"],spelling_bps=v["bits"]/v["symbols"],mean_hidden_delta=v["delta"]/v["n"]) for k,v in m.items()},"seconds":time.monotonic()-start}
    def event39(self,root,ref,small,large,cal,backoff,weight=None,raw_scale=None):
        if self.fb_scale==0:return S39.Symbiosis.event39(self,root,ref,small,large,cal,backoff,weight,raw_scale)
        state=self.session()
        for w in A.words(root):state.observe(w)
        row=self.make_row(state,ref,small,large,cal);word,_=self.score_row(row,self.fb_scale)
        x,b,_=self.raw_row(root,backoff);t,_=Y.parse(root);mem=tuple(a[None] for a in row[-1])
        with torch.no_grad():
            hidden,_=self.feedback_out(np.asarray([x],np.uint32),None,mem,self.fb_scale,np.asarray([b],np.int64),np.asarray([Y.context(t)],np.int64))
        return SL.Event(word,self.root_raw(root,hidden[0].numpy(),backoff,self.raw_scale if raw_scale is None else raw_scale),self.event_weight if weight is None else weight)
    def save40(self,path):
        payload={"fb_step":np.asarray(self.fb_step),"parent_sha256":np.asarray(self.v37_sha),"transport_sha256":np.asarray(self.transport_sha)}
        for k,a in self.fb.items():payload[k]=a;payload["m_"+k]=self.fm[k];payload["v_"+k]=self.fv[k]
        np.savez(path,**payload)
        path.with_suffix(".json").write_text(json.dumps({"fb_scale":self.fb_scale,"parent_sha256":self.v37_sha,"transport_sha256":self.transport_sha,"transport_meta_sha256":self.transport_meta_sha,"feedback_step":self.fb_step,"raw_event_weight":self.event_weight,"raw_scale":self.raw_scale},indent=2))

def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor","unified","transport"):
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
    field=Reciprocal(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,a.attractor,a.unified,len(data[1]),transport=a.transport);field.bind(ref,small,large,cal)

    torch.set_num_threads(max(1,a.threads))
    field.configure(a.profile)


    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    fitbooks=calbooks[:8];selectbooks=calbooks[8:]
    initial={k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()};steps=field.steps.copy()
    train=field.collect40([B.load(p,512) for _,p in fitbooks],ref,small,large,cal,48,4003)
    valid=field.collect40([B.load(p,512) for _,p in selectbooks],ref,small,large,cal,48,4004)
    checks=field.check40(train);fit=field.fit40(train,valid)
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
    mani=S39.manifest({"reference":tr+extra,"cal_fit":fitbooks,"cal_select":selectbooks,"dev_story":picked,**{"dev_fresh_"+k:v for k,v in fresh.items()}})
    a.out.with_name("split_manifest.json").write_text(json.dumps(mani,indent=2))
    story=[B.load(p,1200) for _,p in picked]
    result={"parent_sha256":field.v37_sha,"transport_sha256":field.transport_sha,"checks":checks,"fit":fit,"short":field.word_eval40(story,ref,small,large,cal),"long":field.word_eval40(story,ref,small,large,cal,True),"fresh":{}}
    for name,files in fresh.items():
        rows=field.event_rows(files,3903,12)
        word=field.word_eval40([B.load(p,300) for _,p in files],ref,small,large,cal,limit=32)
        raw=field.raw_eval(rows,ref,small,large,cal,backoff)
        scale=field.fb_scale;field.fb_scale=0.
        try:parent=field.raw_eval(rows,ref,small,large,cal,backoff)
        finally:field.fb_scale=scale
        result["fresh"][name]={"word":word,"raw":raw,"parent_raw":parent}
        write_metrics(a.out,result,"domain:"+name)
    raw=first(picked[0][1]);result["codec"]=field.codec_check(raw[:128],raw[128:144],ref,small,large,cal,backoff)
    assert steps==field.steps and all(initial[k]==hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items())
    result["inherited_unchanged"]=True;result["manifest_sha256"]=hashlib.sha256(a.out.with_name("split_manifest.json").read_bytes()).hexdigest()
    field.save40(a.out.with_name("reciprocal-field.npz"));write_metrics(a.out,result)
if __name__=="__main__":main()
