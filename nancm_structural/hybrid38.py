"""v38: portable FP32 recurrent kernel; host analytic optimizer unchanged.
Run via CPU, T4 or Blackwell profile. No checkpoint training or promotion."""
import hybrid37 as U37, hybrid35 as R5, hybrid26 as D, hybrid23 as P, hybrid22 as Z, hybrid20 as V, hybrid18 as U
import base as B, foundation6 as N, temporal_phasor as T, coordinate_exchange as E, triangle_field as G, lexical_geometry as J, byte_geometry as K, subword_geometry as Y, selective_resonance as RZ, attractor_memory as AM, raw_byte_law as A
import torch
from persistent_triangle import blend as persistent_edge_blend
import numpy as np
from metric_io import write_metrics
import argparse, json, pathlib, time, hashlib, math
PROFILES={"cpu":{"device":"cpu","batches":[1,8,16,32]},
          "t4":{"device":"cuda","batches":[1,8,16,32,64]},
          "blackwell":{"device":"cuda","batches":[1,16,32,64,128]}}
def load37(f,path):
    with np.load(path,allow_pickle=False) as ck:
        assert np.array_equal(ck["words"],np.array(f.sem.words))
        assert np.array_equal(ck["classes"],f.sem.cls)
        assert np.array_equal(ck["upper_pairs"],f.upper_pairs)
        assert np.array_equal(ck["lower_pairs"],f.lower_pairs)
        assert str(ck["parent_sha"].item())==f.v36_sha
        assert 7808<=int(ck['step'])<=8000
        steps=json.loads(str(ck["group_steps_json"].item()))
        assert set(steps)==set(f.p)
        for k in f.p:
            for prefix,group in (("",f.p),("m_",f.m),("v_",f.v)):
                a=ck[prefix+k];assert a.shape==group[k].shape and np.isfinite(a).all();group[k]=a.copy()
        for k,key in (("lex","lexical"),("lm","lex_m"),("lv","lex_v"),("lt","lex_steps"),("proj","projection")):
            a=ck[key];assert a.shape==getattr(f,k).shape and np.isfinite(a).all();setattr(f,k,a.copy())
        f.steps=steps;f.step=int(ck["step"])
        f.rng.bit_generator.state=json.loads(str(ck["rng_json"].item()))
    f.v37_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    meta=json.loads(path.with_suffix(".token.json").read_text())
    assert meta["split"]==list(Y.SPLIT) and meta["vocabulary"]==len(Y.TOKENS) and meta["parent_sha256"]==f.v36_sha
    f.trust=tuple(meta["trust"]);assert len(f.trust)==2 and all(0<=v<=1 for v in f.trust)
    meta=json.loads(path.with_suffix(".memory.json").read_text())
    assert meta["capacity"]==AM.CAP and meta["candidates"]==AM.K and meta["parent_sha256"]==f.v36_sha
    f.memory_scale=float(meta["scale"]);assert 0<=f.memory_scale<=1
    meta=json.loads(path.with_suffix(".unified.json").read_text())
    assert meta["parent_sha256"]==f.v36_sha and meta["accepted_step"]==f.step
    modes={"full":("full","full"),"res_off":("off","full"),"res_and_input_off":("off","input_off")}
    f.word_mode=meta["word_mode"];f.res_mode,f.lex_mode=modes[f.word_mode]
    f.bridge_weight=float(meta["bridge_weight"]);assert 0<=f.bridge_weight<1
    return f











class Portable(U37.Unified):
    def __init__(self,sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,available):
        super().__init__(sem,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,available)
        load37(self,p37);self.profile="legacy";self.device=torch.device("cpu");self.index_cache={}
    def configure(self,profile):
        assert profile in PROFILES or profile=="legacy"
        self.profile=profile
        self.device=torch.device("cpu" if profile=="legacy" else PROFILES[profile]["device"])
        if self.device.type=="cuda":
            if not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
            major,minor=torch.cuda.get_device_capability()
            if profile=="t4" and (major,minor)!=(7,5):raise RuntimeError("T4 profile requires SM 7.5")
            if profile=="blackwell" and major not in (10,12):raise RuntimeError("Blackwell profile requires SM 10.x or 12.x")
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest")
    def indices(self):
        key=str(self.device)
        if key not in self.index_cache:
            self.index_cache[key]=(torch.as_tensor(self.ti,dtype=torch.long,device=self.device),torch.as_tensor(self.tj,dtype=torch.long,device=self.device),tuple(x.to(self.device) for x in self.topology))
        return self.index_cache[key]
    def _extension(self,x,hidden):
        anchor=getattr(self,'edge_anchor',None)
        if self.profile=='legacy':
            if anchor is not None:raise RuntimeError('persistent edge hook requires the portable kernel')
            return R5.Resonance._extension(self,x,hidden)
        device=self.device
        cpu={k:torch.tensor(self.p[k],requires_grad=anchor is None) for k in (*self.new_keys,*self.coord_keys,*self.temporal_keys,*self.triangle_keys,*self.word_keys[:3],*self.byte_keys,*self.sub_keys,*self.res_keys)}
        p={k:v.to(device) for k,v in cpu.items()}
        host=torch.tensor(hidden,requires_grad=anchor is None);h=host.to(device)
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
        persistent=anchor(s,z,topo[1]) if anchor is not None else None
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
            if persistent is None:
                msg=G.message(edges,p["tri_plus"],p["tri_minus"],topo,self.triangle_mode)
            else:
                seed,gate,dp,dm=persistent
                edges=persistent_edge_blend(edges,seed,gate)
                msg=G.message(edges,p["tri_plus"]+dp,p["tri_minus"]+dm,topo,self.triangle_mode)
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
        return out.to('cpu'),host,cpu,residual
    def probe(self,word,token,profile,seconds):
        self.configure("legacy");x,y=word;wx=x[:16];wy=y[:16];td=tuple(a[:16] for a in token)
        q0,h0,_v,_t=self.tape(np.concatenate((wx[:,:40],wx[:,41:]),axis=1))
        del _t,_v
        lw,gw,lg=self.direct(wx,wy,True);lt,gt=self.token_loss(td,True)
        t0=self.token_forward(*td[:-1])
        self.configure(profile)
        q1,h1,_v,_t=self.tape(np.concatenate((wx[:,:40],wx[:,41:]),axis=1))
        del _t,_v
        lwp,gwp,lgp=self.direct(wx,wy,True);ltp,gtp=self.token_loss(td,True)
        t1=self.token_forward(*td[:-1])
        tol=1e-6 if profile=="cpu" else 2e-4
        def check(a,b):
            assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            err=float(np.max(np.abs(a-b))) if a.size else 0.
            assert np.allclose(a,b,rtol=tol,atol=tol),(a.shape,err)
            return err
        assert set(gw)==set(gwp) and set(gt)==set(gtp) and set(lg)==set(lgp)
        errors={"class_probability":check(q0,q1),"hidden":check(h0,h1),
                "word_loss":abs(lw-lwp),"token_loss":abs(lt-ltp),
                "word_gradients":{k:check(gw[k],gwp[k]) for k in gw},
                "token_gradients":{k:check(gt[k],gtp[k]) for k in gt},
                "lex_gradients":{str(k):check(lg[k],lgp[k]) for k in lg},
                "token_probability":[check(a,b) for a,b in zip(t0,t1)]}
        assert errors["word_loss"]<=tol and errors["token_loss"]<=tol
        before={k:(a.copy(),self.m[k].copy(),self.v[k].copy(),self.steps[k]) for k,a in self.p.items()}
        rates=[];deadline=time.monotonic()+max(1,min(seconds,600))
        for batch in PROFILES[profile]["batches"]:
            if time.monotonic()>=deadline:break
            idx=np.arange(batch)%len(x);jx=np.arange(batch)%len(token[-1]);a=x[idx];b=y[idx];td=tuple(v[jx] for v in token)
            if self.device.type=="cuda":torch.cuda.reset_peak_memory_stats()
            start=time.monotonic();n=0
            while n<4 and time.monotonic()<deadline:
                self.direct(a,b,True);self.token_loss(td,True)
                if self.device.type=="cuda":torch.cuda.synchronize()
                n+=1
            elapsed=time.monotonic()-start
            rates.append({"batch":batch,"iterations":n,"seconds":elapsed,"word_plus_token_targets_per_second":2*batch*n/max(elapsed,1e-9),"peak_allocated_bytes":torch.cuda.max_memory_allocated() if self.device.type=="cuda" else None,"peak_reserved_bytes":torch.cuda.max_memory_reserved() if self.device.type=="cuda" else None})
        assert all(np.array_equal(a,self.p[k]) and np.array_equal(m,self.m[k]) and np.array_equal(v,self.v[k]) and j==self.steps[k] for k,(a,m,v,j) in before.items())
        return {"profile":profile,"device":str(self.device),"device_name":torch.cuda.get_device_name() if self.device.type=="cuda" else "CPU","capability":list(torch.cuda.get_device_capability()) if self.device.type=="cuda" else None,"torch":torch.__version__,"tolerance":tol,"errors":errors,"rates":rates,"parameters_and_moments_unchanged":True,"step":self.step,"checkpoint_sha256":self.v37_sha}
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
    field=Portable(sem,a.parent,a.recurrent,a.direct,a.dual,a.symbiotic,a.fusion,a.coordinate,a.temporal,a.relational,a.lexical,a.byte,a.subword,a.resonance,a.attractor,a.unified,len(data[1]));field.bind(ref,small,large,cal)

    torch.set_num_threads(max(1,a.threads))
    def first(p):
        with p.open("rb") as f:return f.read(16384)
    backoff=A.Backoff([first(p) for _,p in tr])
    # Read actual calibration windows only; no optimizer update in smoke.
    token,_=field.token_data(calbooks,backoff,12,3801)
    word,_=field.worddata(cdocs,cap=96)
    result=field.probe(word,token,a.profile,a.smoke_seconds)
    write_metrics(a.out,result)
if __name__=="__main__":main()
