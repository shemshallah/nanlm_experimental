"""v62: exact deployed word-event gradients through v53/v55 into inherited v50 field.
Train short contiguous sequences with teacher initial causal posterior; retain all Adam moments.
"""
import sys,pathlib,copy,bisect
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_switching'))
from pilot60 import *
from credit import laws,sequence
PI=(.5,.4375,.0625);HAZARD=.35

def event_vector(event,outside,surface):
    v=np.zeros(len(outside.words)+1,np.float64);letters=event if FE.IS_LETTER[event[-1]] else event[:-1]
    if not letters:return v,0.,-1
    s=1.;pref=b''
    for b in letters:
        s*=surface.letter(pref,b|32)[int(b<97)];pref+=bytes([b])
    word=letters.lower().decode('ascii');i=bisect.bisect_left(outside.words,word)
    if len(event)==64 and FE.IS_LETTER[event[-1]]:
        end=bisect.bisect_left(outside.words,word+'{');v[i+1:end+1]=1.;v[0]=outside.mass(word);target=-1
    else:
        assert not FE.IS_LETTER[event[-1]]
        s*=surface.delimiter(pref)[event[-1]]
        target=i+1 if i<len(outside.words) and outside.words[i]==word else 0
        v[target]=1. if target else outside.complete(word)
    return v,s,target

def factor(h,index,bank,g,old,os,contrast,cs,cls):
    rr=bank.experts(h);w=weights(old,rr,os);count=index.countlaw(h)
    anchor=w[0]*(1-g)*count
    for a,(law,_) in zip(w[1:],rr):
        for t,p in law.items():anchor[t]+=a*p
    share=w[0]*g
    uniform=np.full(index.V,1./index.V)
    d,f,v=donors(h,uniform,index.prior,cls)
    cr=apply(contrast,d,f,v,cs,MASKS55['full'])
    assert abs(anchor.sum()+share-1)<1e-8
    return anchor,share,d[0],d[1]/uniform,d[2]/uniform,cr

def prepare62(run,index,bank,model,g,old,os,contrast,cs,outside,raw,surface):
    docs={};filt=Switch(PI,HAZARD);last=None;previous=None;checks=0;err=0.;cls=model.cls.cpu().numpy()
    for bid,prefix,event,h in run:
        if bid!=last:
            assert bid not in docs;docs[bid]=[];filt.reset()
        else:assert prefix==previous
        ids=index.encode(h[-512:]);row=(bid,ids,rails(index.code,ids),0)
        x,i,_=feature([row])
        a=factor(ids,index,bank,g,old,os,contrast,cs,cls)
        e,s,t=event_vector(event,outside,surface)
        r=math.prod(raw(prefix+event[:k])[b] for k,b in enumerate(event))
        arr=tuple(torch.as_tensor(np.asarray(v)[None],dtype=torch.float64) for v in a)
        with torch.no_grad():
            lp=model.log_probs(x,i);p,q=laws(lp,*arr)
            if checks<32:
                dense=lp.double().exp().numpy()[0];dense/=dense.sum()
                rr=bank.experts(ids);base=old_mix((1-g)*index.countlaw(ids)+g*dense,[l for l,_ in rr],weights(old,rr,os))
                d,f,v=donors(ids,base,index.prior,cls);qref=fused(base,d,apply(contrast,d,f,v,cs,MASKS55['full']))
                err=max(err,float(np.max(abs(p.numpy()[0]-base))),float(np.max(abs(q.numpy()[0]-qref))))
                for actual,refp in ((p.numpy()[0],base),(q.numpy()[0],qref)):
                    lit=FE.Event(Lift(outside,refp),raw,prefix,surface,0.).literal(event)[1]
                    err=max(err,abs(lit-float(actual@e)*s))
                    assert math.isclose(lit,float(actual@e)*s,rel_tol=1e-7,abs_tol=1e-25)
                checks+=1
            lik=np.array([r,float(p.numpy()[0]@e)*s,float(q.numpy()[0]@e)*s])
        initial=filt.weights();filt.observe(lik)
        docs[bid].append((x[0],i[0],*a,e,s,r,t,initial,len(event)))
        last=bid;previous=prefix+event
    assert checks and err<1e-8
    return docs,{'events':sum(map(len,docs.values())),'bytes':sum(row[-1] for d in docs.values() for row in d),'composition_checks':checks,'composition_max_error':err}

def batch(rows):
    # x,ids, six law factors, event vector, surface, raw, categorical target
    x=torch.stack([r[0] for r in rows]);i=torch.stack([r[1] for r in rows])
    a=tuple(torch.as_tensor(np.asarray(v),dtype=torch.float64) for v in zip(*(r[2:11] for r in rows)))
    target=torch.as_tensor([r[11] for r in rows],dtype=torch.long)
    initial=torch.as_tensor(rows[0][12],dtype=torch.float64)
    return x,i,a,target,initial

def objective(model,rows,raw_credit):
    x,i,a,t,initial=batch(rows);p,q=laws(model.log_probs(x,i),*a[:6]);e,s,r=a[6:]
    if raw_credit:
        lik=torch.stack((r,(p*e).sum(-1)*s,(q*e).sum(-1)*s),-1)
        return sequence(lik,initial,torch.tensor(PI,dtype=torch.float64),HAZARD)/len(rows)
    ok=t>=0
    if not bool(ok.any()):return None
    return -q[ok,t[ok]].log().mean()

def fit62(model,ck,train):
    # Same document segments and original optimizer state for both arms.
    rng=np.random.default_rng(6201);segments=[];docs=list(train.values())
    for _ in range(128):
        d=docs[int(rng.integers(len(docs)))];a=int(rng.integers(max(1,len(d)-7)));seg=d[a:a+8]
        if any(r[11]>=0 for r in seg):segments.append(seg)
    assert len(segments)>=96
    segments=segments[:96];report={};models={};opts={}
    for name,raw_credit in (('word_credit',False),('event_credit',True)):
        m=copy.deepcopy(model);m.requires_grad_(True);opt=torch.optim.AdamW(m.parameters());opt.load_state_dict(copy.deepcopy(ck['optimizers']['coupled']))
        for group in opt.param_groups:group['lr']=.0003
        steps=lambda:[int(s['step']) for s in opt.state.values() if 'step' in s]
        assert min(steps())==max(steps())==768
        trace=[];norms={}
        for n,seg in enumerate(segments,1):
            opt.zero_grad();loss=objective(m,seg,raw_credit);assert loss is not None and torch.isfinite(loss);loss.backward()
            for k,p in m.named_parameters():
                if p.grad is not None:norms[k]=max(norms.get(k,0.),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if n==1 or n%16==0:trace.append([n,float(loss.detach())])
        assert min(steps())==max(steps())==864
        assert len(norms)==len(list(m.parameters())) and min(norms.values())>0
        models[name]=m.eval();opts[name]=opt
        report[name]={'step':864,'trace':trace,'gradient_groups':norms}
    return models,opts,report

def evaluate62(run,args,raw,surface,models):
    index,bank,parent,g,old,os,contrast,cs,outside,_,_=args
    # Independently replay causal posterior with each trained field; same exact windows.
    result={}
    for name,model in [('parent',parent),*models.items()]:
        ar=(index,bank,model,g,old,os,contrast,cs,outside,None,None)
        z=evaluate60(rows58([],6202,0,*ar,raw=run),raw,surface,False,(PI,HAZARD),(PI,1.))
        result[name]=z['modes']['adaptive']
        result[name]['literal_max_error']=z['literal_max_error'];result[name]['token_marginal_max_error']=z['token_marginal_max_error']
    return result

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
    ap.add_argument('--native',action='store_true')
    keys=('lib','parent','recurrent','direct','dual','symbiotic','fusion','coordinate','temporal','relational','lexical','byte','subword','resonance','attractor','unified')
    for k in keys:ap.add_argument('--'+k,type=pathlib.Path)
    a=ap.parse_args();torch.set_num_threads(4)
    groups={k:B.bookfiles(a.data/k) for k in ('stories','sentences','paragraphs')};split=lambda i:int(B.digest('NaNCM-structural-v1:'+i)[:8],16)%10
    books=[(i,p) for i,p in groups['stories'] if split(i)<7];tr=books[:96];calbooks=books[96:108];extra=books[108:396];assert len(extra)==288
    primary,secondary,sel,manifest=cohorts(books,tr,calbooks,extra)
    screen=json.loads(a.screen.read_text());assert screen['report_stage']=='final' and screen['parent_artifacts_unchanged']
    selected={}
    for name,records in screen['clean_screen']['manifest'].items():
        lookup=dict(groups['stories' if name=='calibration' else name]);selected[name]=[]
        for record in records:
            i=record['id'];p=lookup[i];assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'];selected[name].append((i,p))
    lex=set()
    for p in (a.data/'lex').rglob('*.txt'):
        if 'google' in p.name.lower() and not p.name.lower().startswith(('license','readme')):lex.update(B.words(p.read_text(errors='replace')))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr];expanded=docs+[B.load(p,6000) for _,p in extra]
    index=Index(expanded,lex);outside=Outside(index.words[1:],Spelling(expanded))
    files=[a.checkpoint,a.analogy,a.contrast];hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    assert hashes==[screen[k] for k in ('parent_sha256','analogy_sha256','contrast_sha256')]
    ck,ac,cc=[torch.load(p,map_location='cpu',weights_only=True) for p in files]
    assert ck['words']==index.words and ck['split_integrity']==ac['split_integrity']==cc['split_integrity']==manifest
    model=Field(index.code,index.prior)
    for k in ('code','members','cls','slot','logprior','classprior'):assert torch.equal(model.state_dict()[k],ck['models']['coupled'][k])
    model.load_state_dict(ck['models']['coupled']);model.eval();model.requires_grad_(False);g=float(ck['mixture']['coupled'])
    old=OldCredit().double();old.load_state_dict(ac['gains']['full']);old.requires_grad_(False);os=float(ac['scales']['full'])
    contrast=E.Credit().double();contrast.load_state_dict(cc['gains']['full']);contrast.requires_grad_(False);cs=float(cc['scales']['full'])
    bank=Analogy(index.code,index.seq)
    rawdocs=[p.read_bytes()[:16384] for _,p in tr];backoff=RAW.Backoff(rawdocs);surface=FE.Surface(rawdocs)
    field=None;runtime=None
    if a.native:
        import hybrid38 as P38
        import hybrid18 as U, foundation6 as N, hybrid20 as V, hybrid22 as Z, hybrid23 as P
        assert all(getattr(a,k) is not None for k in keys)
        ref=U.Unified(docs,lex,N.Native(a.lib));ref.head.train(ref.head.data([d[:512] for d in docs]))
        cal=U.fit(ref,[B.load(p,200) for _,p in calbooks]);small=V.Memory(docs);large=V.Memory(expanded);sem=Z.Semantic(expanded,large);data=P.Field(sem).data(expanded,cap=768)
        field=P38.Portable(sem,*[getattr(a,k) for k in keys[1:]],len(data[1]));field.bind(ref,small,large,cal);field.configure('cpu');runtime=(ref,small,large,cal)
        initial=digest_field(field);assert field.step==7998
        def raw(h):
            with torch.no_grad():q=field.predict_token_byte(h,backoff).astype(np.float64)
            return q/q.sum()
    else:raw=backoff.prob
    out={'native':a.native,'parent_artifact_hashes':hashes,'vocabulary':len(outside.words),'outside_spelling_mass':outside.z,'domains':{},'event_scope':'contiguous complete-event windows inside v57 screened prefixes; reset posterior per document','native_folded_marginal_claim':False}
    write_metrics(a.out,out,'index')
    assert not a.native,'v62 learns the v50 component, not original v37 weights'
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    assert not {i for i,p in primary+secondary}&{i for group in selected.values() for i,p in group}
    fitrun,fitmanifest=contiguous(primary+secondary,128)
    train,audit=prepare62(fitrun,index,bank,model,g,old,os,contrast,cs,outside,raw,surface)
    out['fit_audit']={'windows':fitmanifest,**audit};write_metrics(a.out,out,'factors')
    models,opts,report=fit62(model,ck,train);del train
    out['fit']=report
    torch.save({'models':{k:m.state_dict() for k,m in models.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'parent_artifact_hashes':hashes,'step':864,'prior':PI,'hazard':HAZARD},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    for name in ('calibration',*groups):
        run,manifest62=contiguous(selected[name][:3],256)
        result=evaluate62(run,args,raw,surface,models)
        assert all(v['bytes']==sum(q['bytes'] for q in manifest62) for v in result.values())
        out['domains'][name]={'windows':manifest62,'metrics':result};write_metrics(a.out,out,'domain:'+name)
    assert all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    out['eligible_for_promotion']=False # small diagnostic; larger matched follow-up required
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
