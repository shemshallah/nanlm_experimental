"""v65: full-vocabulary signed transport trained against actual folded word competitors."""
import sys,pathlib,bisect,copy
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_joint'))
from pilot63 import *
from transition import Transition
from rank_field import Transport,credit,predict

@torch.no_grad()
def prepare65(run,args,raw,surface,core):
    index,bank,geo,g,old,os,contrast,cs,outside,_,_=args
    docs={};last=None;state=prev=None;previous=None;checks=0;err=0.;cert=0
    for source,row in zip(run,rows58([],6501,0,*args,raw=run)):
        bid,prefix,event,l53,l55,_=row
        assert source[:3]==row[:3]
        if bid!=last:state=prev=None;docs[bid]=[]
        else:assert prefix==previous
        ids=index.encode(source[3][-512:]);x,i,_=feature([(bid,ids,rails(index.code,ids),0)])
        geom=torch.cat((x[0],geo.hidden(x,i)),0).float()
        f=torch.as_tensor(features61(prefix,None)[:5],dtype=torch.float64)
        w=predict(core,state,prev,f);wn=w.numpy();share=float(wn[1:].sum())
        parent=FE.Event(fuse(l53,l55,float(wn[2]/share)),raw,prefix,surface,share)
        p=np.stack([np.r_[l.unknown,l.p] for l in (l53,l55)])
        e,s,t=event_vector(event,outside,surface);q=likelihood(row,raw,surface,False)
        z=float(w@torch.as_tensor(q));assert math.isclose(z,parent.mass(event),rel_tol=1e-7,abs_tol=1e-30)
        # Sparse event coefficients, dense parent probabilities kept as float32 for storage.
        ei=np.flatnonzero(e);ev=e[ei];candidates=[];wr=[];wc=[]
        if t>=0:
            target=event[:-1].lower().decode('ascii')
            top,ok,_=parent.rank(k=8,budget=400);cert+=int(ok)
            lexical=np.argsort(-(wn[1]*p[0,1:]+wn[2]*p[1,1:]),kind='stable')[:8]
            words=list(dict.fromkeys([target]+top+[outside.words[j] for j in lexical]))[:13]
            rawlaw=FE.Event(l53,raw,prefix,surface,0.)
            for word in words:
                j=bisect.bisect_left(outside.words,word)
                known=j<len(outside.words) and outside.words[j]==word
                candidates.append(j+1 if known else 0);wr.append(rawlaw.complete(word.encode('ascii')));wc.append(1. if known else outside.complete(word))
                exact=wn[0]*wr[-1]+(wn[1]*p[0,candidates[-1]]+wn[2]*p[1,candidates[-1]])*wc[-1]
                err=max(err,abs(exact-parent.complete(word.encode('ascii'))));checks+=1
        docs[bid].append((geom,p.astype(np.float32),ei,ev,s,q[0],f,torch.tensor(candidates,dtype=torch.long),torch.tensor(wr,dtype=torch.float64),torch.tensor(wc,dtype=torch.float64),len(event)))
        a=w*torch.as_tensor(q);state=a/a.sum();prev=torch.as_tensor(q);last=bid;previous=prefix+event
    assert checks>0 and err<1e-10
    return docs,{'folded_score_checks':checks,'max_folded_score_error':err,'certified_training_rank_searches':cert,'events':sum(map(len,docs.values())),'bytes':sum(row[-1] for d in docs.values() for row in d)}

def segment65(model,core,rows,state=None,prev=None,scale=1.,rank_weight=0.):
    x=torch.stack([r[0] for r in rows]);p=torch.from_numpy(np.stack([r[1] for r in rows])).double();p=p/p.sum(-1,keepdim=True)
    e=torch.zeros((len(rows),p.shape[-1]),dtype=torch.float64)
    for i,r in enumerate(rows):e[i,r[2]]=torch.from_numpy(r[3])
    s=torch.tensor([r[4] for r in rows],dtype=torch.float64);raw=torch.tensor([r[5] for r in rows],dtype=torch.float64);f=torch.stack([r[6] for r in rows])
    return credit(core,model(x,p,scale),e,s,raw,f,[r[7] for r in rows],[r[8] for r in rows],[r[9] for r in rows],state,prev,rank_weight)

@torch.no_grad()
def score65(model,core,docs,scale):
    ev=wd=0.;n=hits=bits=0
    for d in docs.values():
        state=prev=None
        for i in range(0,len(d),16):
            chunk=d[i:i+16];_,state,prev,e,_,w,k,h=segment65(model,core,chunk,state,prev,scale)
            ev+=float(e);wd+=float(w);n+=k;hits+=h;bits+=sum(r[-1] for r in chunk)
    return {'bpb':ev/bits/math.log(2),'candidate_wins':hits,'word_events':n,'folded_nll':wd/max(1,n)}

def fit65(code,core,train,valid):
    torch.manual_seed(6502);seed=Transport(code);rng=np.random.default_rng(6503);keys=list(train);segments=[]
    for _ in range(128):
        key=keys[int(rng.integers(len(keys)))];d=train[key];a=int(rng.integers(max(1,len(d)-11)));segments.append((key,a,min(a+12,len(d))))
    base=score65(seed,core,valid,0.);models={};states={};report={}
    for name,weight in [('event_only',0.),('rank_aware',1.)]:
        model=copy.deepcopy(seed);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);trace=[];norms={}
        for step,(key,a,b) in enumerate(segments,1):
            d=train[key];state=prev=None
            with torch.no_grad():
                for i in range(0,a,16):_,state,prev,*_=segment65(model,core,d[i:min(i+16,a)],state,prev)
            opt.zero_grad();loss,*_=segment65(model,core,d[a:b],state,prev,rank_weight=weight);assert torch.isfinite(loss);loss.backward()
            for k,p in model.named_parameters():
                if p.grad is not None:norms[k]=max(norms.get(k,0.),float(p.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%32==0:trace.append([step,float(loss.detach())])
        assert len(norms)==4 and min(norms.values())>0
        grid={s:score65(model,core,valid,s) for s in (0.,.125,.25,.5,1.)}
        # Calibrate only on screened calibration, requiring no BPB regression.
        eligible=[s for s,v in grid.items() if v['bpb']<=base['bpb']+1e-12]
        assert eligible
        scale=max(eligible,key=lambda s:(grid[s]['candidate_wins'],-grid[s]['folded_nll'],-grid[s]['bpb'],-s))
        models[name]=(model.eval(),scale);states[name]={'model':model.state_dict(),'optimizer':opt.state_dict(),'scale':scale,'updates':128}
        report[name]={'rank_weight':weight,'trace':trace,'gradient_groups':norms,'scale':scale,'selection':{str(s):v for s,v in grid.items()}}
    return models,states,{'parent':base,'arms':report}

@torch.no_grad()
def eval65(run,args,raw,surface,core,systems):
    index,bank,geo,g,old,os,contrast,cs,outside,_,_=args
    state={};prev={};stats=C.defaultdict(C.Counter);last=None;previous=None;err=C.Counter();marg=C.Counter()
    for source,row in zip(run,rows58([],6504,0,*args,raw=run)):
        bid,prefix,event,l53,l55,_=row;assert source[:3]==row[:3]
        if bid!=last:state={};prev={}
        else:assert prefix==previous
        ids=index.encode(source[3][-512:]);x,i,_=feature([(bid,ids,rails(index.code,ids),0)])
        geom=torch.cat((x[0],geo.hidden(x,i)),0).float()
        f=torch.as_tensor(features61(prefix,None)[:5],dtype=torch.float64)
        p=torch.from_numpy(np.stack([np.r_[l.unknown,l.p] for l in (l53,l55)]))
        for name,(model,scale) in systems.items():
            w=predict(core,state.get(name),prev.get(name),f);wn=w.numpy();share=float(wn[1:].sum())
            if scale==0:a,b=l53,l55
            else:
                pp=model(geom,p,scale).numpy();a,b=Lift(outside,pp[0]),Lift(outside,pp[1])
            e=FE.Event(fuse(a,b,float(wn[2]/share)),raw,prefix,surface,share)
            r,s0=FE.Event(a,raw,prefix,surface,0.).literal(event);_,s1=FE.Event(b,raw,prefix,surface,0.).literal(event)
            lik=torch.tensor([r,s0,s1],dtype=torch.float64);z=stats[name];post=w*lik;prob=float(post.sum());state[name]=post/post.sum();prev[name]=lik
            qprefix=b'';prod=1.
            for byte in event:
                q=e.next(qprefix);top=np.argsort(-q,kind='stable')[:5];prod*=q[byte];qprefix+=bytes([byte]);z['bytes']+=1;z['byte_t1']+=int(byte==top[0]);z['byte_t5']+=int(byte in top)
            err[name]=max(err[name],abs(prod-prob));assert e.done(qprefix) and math.isclose(prod,prob,rel_tol=1e-7,abs_tol=1e-30)
            z['bits']-=math.log2(prod);z['events']+=1
            qprefix=b'';i=0
            while i<len(event):
                tokens=e.tokens(qprefix);size=2 if event[i] in b' etaoinr' and not e.done(qprefix+event[i:i+1]) else 1
                target=event[i:i+size];top=heapq.nlargest(5,tokens,key=tokens.get);z['tokens']+=1;z['token_t1']+=int(target==top[0]);z['token_t5']+=int(target in top)
                mass=np.zeros(256)
                for k,v in tokens.items():mass[k[0]]+=v
                marg[name]=max(marg[name],float(abs(mass-e.next(qprefix)).max()));qprefix+=target;i+=size
            if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                top,cert,_=e.rank(budget=1200);target=event[:-1].lower();z['words']+=1;z['certified']+=int(cert);z['word_t1']+=int(cert and target.decode('ascii')==top[0]);z['word_t5']+=int(cert and target.decode('ascii') in top);z['word_bits']-=math.log2(max(e.complete(target),1e-300))
        last=bid;previous=prefix+event
    assert max(marg.values())<1e-8
    return {k:dict(z,bpb=z['bits']/z['bytes'],byte_top1=z['byte_t1']/z['bytes'],byte_top5=z['byte_t5']/z['bytes'],token_top1=z['token_t1']/z['tokens'],token_top5=z['token_t5']/z['tokens'],certified_word_top1=z['word_t1']/z['words'],certified_word_top5=z['word_t5']/z['words'],certification_rate=z['certified']/z['words'],folded_word_nll=z['word_bits']/z['words']*math.log(2),literal_max_error=err[k],token_marginal_max_error=marg[k]) for k,z in stats.items()}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen','transition','transition-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
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
    assert not a.native
    tc=torch.load(a.transition,map_location='cpu',weights_only=True);tm=json.loads(a.transition_metrics.read_text())
    assert tm['report_stage']=='final' and tc['parent_artifact_hashes']==hashes and tc['updates']==96 and tc['scale']==1
    core=Transition(*tc['initial']);core.load_state_dict(tc['transition']);core.requires_grad_(False);core.eval()
    thash=hashlib.sha256(a.transition.read_bytes()).hexdigest()
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    assert not {i for i,p in primary+secondary}&{i for group in selected.values() for i,p in group}
    fitrun,fitmanifest=contiguous(primary+secondary,128)
    train,fit_audit=prepare65(fitrun,args,raw,surface,core)
    calrun,calmanifest=contiguous(selected['calibration'],256)
    valid,cal_audit=prepare65(calrun,args,raw,surface,core)
    out['fit_audit']={'train':fit_audit,'calibration':cal_audit,'train_windows':fitmanifest,'calibration_windows':calmanifest}
    write_metrics(a.out,out,'factors')
    systems,saved,report=fit65(index.code,core,train,valid);del train,valid
    out['fit']=report;out['transition_parent_sha256']=thash
    torch.save({'systems':saved,'parent_artifact_hashes':hashes,'transition_parent_sha256':thash,'updates_added':128},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    systems={'parent_v61':(None,0.),**systems,'rank_forced':(systems['rank_aware'][0],1.)}
    for name in groups:
        run,manifest65=contiguous(selected[name],256);result=eval65(run,args,raw,surface,core,systems)
        assert all(v['bytes']==sum(q['bytes'] for q in manifest65) for v in result.values())
        expected=tm['domains'][name]['metrics']['modes']['learned'];actual=result['parent_v61']
        replay={'bpb_delta':actual['bpb']-expected['bpb'],'ranks_equal':all(actual[k]==expected[k] for k in ('bytes','byte_t1','byte_t5','word_t1','word_t5','certified'))}
        out['domains'][name]={'windows':manifest65,'metrics':result,'parent_replay':replay}
        write_metrics(a.out,out,'domain:'+name)
    assert all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,tc['transition'][k]) for k,v in core.state_dict().items())
    assert hashlib.sha256(a.transition.read_bytes()).hexdigest()==thash
    out['eligible_for_promotion']=False
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
