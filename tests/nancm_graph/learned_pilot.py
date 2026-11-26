"""v43 learned signed persistent graph credit, matched exact-only control."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'nancm_structural'))
from pilot import *
from signed_edge_credit import EdgeCredit,likelihood

class Learned(Experiment):
    def features(self,base,ex,field,common,g,h,mw):
        words,x,p=super().features(base,ex,field,common,g,h,mw)
        _,vec,n=g.query(h);bg=g.background/max(1,len(g.doc_ring))
        residual=(vec-bg).astype(np.float32)
        values=np.asarray([g.vec(w) for w in words],np.float32).reshape(-1,128)
        x[:,15:23]=0 # do not let fixed dot-product features substitute for learned geometry
        return words,x,p,residual,values,n
    def dataset(self,examples):
        fs=[];rs=[];vs=[];cs=[];ps=[];hs=[];bs=[]
        for w,ex,f,parent,base,real,bad,*_ in examples:
            words,x,p,residual,values,counts=real;k=len(p)
            fs.append(np.pad(x,((0,MAX-k),(0,0))));rs.append(residual)
            vs.append(np.pad(values,((0,MAX-k),(0,0))));cs.append(counts)
            ps.append(np.pad(p,(0,MAX-k)));hs.append(np.pad(np.asarray([v==w for v in words]),(0,MAX-k)))
            bs.append(max(base.complete(w),1e-300))
        return tuple(torch.as_tensor(np.asarray(a),dtype=torch.float32 if i<4 else torch.float64) for i,a in enumerate((fs,rs,vs,cs,ps,hs,bs)))
    def fit(self,fitdocs,validdocs):
        rows=list(self.examples(fitdocs,long=True));grid=(0.,.25,.5,.75,1.)
        scores={str(s):float(np.mean([-math.log(max((1-s)*ex.complete(w)+s*f.complete(w),1e-300)) for w,ex,f,*_ in rows])) for s in grid}
        self.mix=min(grid,key=lambda s:scores[str(s)]);del rows
        ds=self.dataset(self.examples(fitdocs,long=True));dv=self.dataset(self.examples(validdocs,long=True))
        self.models={};self.optimizers={};report={}
        for name,geom in (("full",True),("exact",False)):
            model=EdgeCredit();opt=torch.optim.Adam(model.parameters(),lr=.015,weight_decay=.005)
            rng=np.random.default_rng(4302);trace=[]
            for i in range(1,385):
                ix=rng.integers(len(ds[-1]),size=16);a=tuple(t[ix] for t in ds)
                opt.zero_grad();loss=likelihood(model(*a[:4],geometry=geom),*a[4:]);loss.backward();opt.step()
                if i==1 or i%64==0:trace.append([i,float(loss.detach())])
            with torch.no_grad():
                z=torch.cat([model(*(a[i:i+16] for a in dv[:4]),geometry=geom) for i in range(0,len(dv[-1]),16)])
                nll={str(s):float(likelihood(z,*dv[4:],scale=s)) for s in (0.,.125,.25,.5,1.)}
            self.scale[name]=min((0.,.125,.25,.5,1.),key=lambda s:nll[str(s)])
            self.models[name]=model;self.optimizers[name]=opt
            report[name]={"selection_nll":nll,"scale":self.scale[name],"trace":trace,"fit_targets":len(ds[-1]),"selection_targets":len(dv[-1]),"candidate_coverage":float(ds[-2].any(1).double().mean()),"parameter_norms":{k:float(v.detach().norm()) for k,v in model.named_parameters()}}
        return {"mix":self.mix,"mix_fit_nll":scores,"controls":report}
    def law(self,base,row,name,forced=False):
        words,x,p,residual,values,counts=row;scale=1. if forced else self.scale[name]
        if not scale:return base
        args=tuple(torch.as_tensor(a[None],dtype=torch.float32) for a in (x,residual,values,counts))
        with torch.no_grad():lift=self.models[name](*args,geometry=name=="full")[0].numpy().astype(np.float64)
        return SL.Transport(base,words,lift*scale)
    def save(self,path,parent):
        torch.save({"parent_sha256":parent,"mix":self.mix,"scale":self.scale,"models":{k:v.state_dict() for k,v in self.models.items()},"optimizers":{k:v.state_dict() for k,v in self.optimizers.items()},"steps":384},path)
def main():
    ap=argparse.ArgumentParser()
    for k in ("data","lib","parent","recurrent","direct","dual","symbiotic","fusion","coordinate","temporal","relational","lexical","byte","subword","resonance","attractor"):
        ap.add_argument("--"+k,type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("learned-edge-metrics.json"));a=ap.parse_args();r=a.data
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
    exp=Learned(field,ref,small,large,cal)
    digest=lambda: {k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
    initial=digest();steps=field.steps.copy()
    fitbooks=extra[::24];selectbooks=calbooks;fit=exp.fit([B.load(p,512) for _,p in fitbooks],[B.load(p,512) for _,p in selectbooks])
    picked=[(i,p) for i,p in groups["stories"] if split(i) in (7,8)][:5];assert len(picked)==5
    used={i for i,p in picked};forbidden={i for i,p in tr+extra+calbooks};assert not used&forbidden
    story=[B.load(p,1200) for _,p in picked];out={"parent_sha256":field.v36_sha,"fit":fit,"short":exp.evaluate(story),"long":exp.evaluate(story,True),"fresh":{}}
    for name,files in groups.items():
        chosen=[]
        for i,p in sorted(files,key=lambda t:B.digest("v4-window:"+t[0])):
            if split(i) not in (7,8) or i in used or i in forbidden:continue
            if len(B.load(p,200))<112:continue
            chosen.append((i,p));used.add(i)
            if len(chosen)==6:break
        assert len(chosen)==6
        out["fresh"][name]={"ids":[i for i,p in chosen],"metrics":exp.evaluate([B.load(p,300) for _,p in chosen],limit=32)}
    assert steps==field.steps and initial==digest()
    out["inherited_unchanged"]=True;out["fit_ids"]=[i for i,p in fitbooks];out["selection_ids"]=[i for i,p in selectbooks];out["dev_ids"]=[i for i,p in picked];out["windows"]=list(WINDOWS)
    a.out.write_text(json.dumps(out,indent=2))
    exp.save(a.out.with_suffix(".pt"),field.v36_sha)
    print("FINAL",json.dumps(out),flush=True)
if __name__=="__main__":main()
