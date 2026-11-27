"""v61 exact BPTT of context-dependent expert transitions on observed byte events."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_switching'))
from pilot60 import *
from transition import Transition,D

def features61(root,prev):
    tail=root[-64:]
    a=[1.,math.log1p(len(root))/10.,float(root[-1:] == b' '),float(root[-1:] in (b'\n',b'\r')),sum(65<=b<=90 for b in tail)/max(1,len(tail))]
    if prev is None:a += [0.,0.]
    else:
        v=np.log(np.maximum(prev,1e-300));a.extend(np.clip((v[1:]-v[0])/8.,-1.,1.).tolist())
    return np.asarray(a,np.float64)

def sequences(rows,raw,surface,native):
    docs={};last=None;prev=None;previous=None
    for row in rows:
        bid,root,event,*_=row
        if bid!=last:
            assert bid not in docs;docs[bid]=[];prev=None
        else:assert root==previous
        f=features61(root,prev);q=likelihood(row,raw,surface,native)
        docs[bid].append((f,q,len(event)))
        prev=q;previous=root+event;last=bid
    return docs

def padded(docs):
    n=len(docs);t=max(map(len,docs.values()));p=np.ones((n,t,3));f=np.zeros((n,t,D));mask=np.zeros((n,t),bool);size=0
    for i,doc in enumerate(docs.values()):
        for j,(x,q,b) in enumerate(doc):p[i,j]=q;f[i,j]=x;mask[i,j]=True;size+=b
    return (torch.from_numpy(p),torch.from_numpy(f),torch.from_numpy(mask)),size

def grid61(docs):
    scores={}
    for pi in PRIORS:
        for h in HAZARDS:
            loss=0.
            for doc in docs.values():
                m=Switch(pi,h)
                for _,q,_ in doc:loss-=math.log2(m.observe(q))
            scores[pi,h]=loss
    a=min((k for k in scores if k[1]<1),key=scores.get);b=min((k for k in scores if k[1]==1),key=scores.get)
    return a,b,scores

def fit61(train,valid):
    tr,n=padded(train);va,m=padded(valid)
    init,_,_=grid61(train);adaptive,static,scores=grid61(valid)
    torch.manual_seed(6101);model=Transition(*init);opt=torch.optim.AdamW(model.parameters(),lr=.025,weight_decay=.01)
    trace=[]
    def evaluate(scale):
        with torch.no_grad():return float(model(*va,scale)/m/math.log(2))
    before=evaluate(0.)
    for i in range(96):
        opt.zero_grad();loss=model(*tr)/tr[2].sum();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if i==0 or (i+1)%16==0:trace.append([i+1,float(loss.detach()),evaluate(1.)])
    grid={s:evaluate(s) for s in (0.,.125,.25,.5,.75,1.)};scale=min(grid,key=grid.get)
    return model,opt,scale,adaptive,static,{'fit_bytes':n,'fit_events':int(tr[2].sum()),'cal_bytes':m,'initial':init,'fixed_switch':adaptive,'fixed_static':static,'selection_grid':{str(s):v for s,v in grid.items()},'trace':trace,'before':before,'selected_scale':scale,'static_bpb':scores[static]/m,'switch_bpb':scores[adaptive]/m}

def evaluate61(rows,raw,surface,native,model,scale,adaptive,static):
    typ=NativeEvent if native else FE.Event
    learned=model.runtime(scale);base=model.runtime(0.);forced=model.runtime(1.)
    controls={'switch':Switch(*adaptive),'static':Switch(*static)}
    stats=C.defaultdict(C.Counter);last=None;prev=None;previous=None;maxerr=0.;marg=0.;checks=0
    for row in rows:
        bid,root,event,l53,l55,parent=row;a=parent if native else l53
        if bid!=last:
            for m in (learned,base,forced,*controls.values()):m.reset()
            prev=None
        else:assert root==previous
        f=features61(root,prev);q=likelihood(row,raw,surface,native)
        modes={'learned':learned.weights(f),'same_initial_switch':base.weights(f),'forced':forced.weights(f),**{k:m.weights() for k,m in controls.items()}}
        for name,w in modes.items():
            share=float(w[1]+w[2]);e=typ(fuse(a,l55,float(w[2]/share)),raw,root,surface,share);z=stats[name];prefix=b'';prod=1.
            for b in event:
                p=e.next(prefix);rank=np.argsort(-p,kind='stable')[:5];prod*=p[b];prefix+=bytes([b]);z['bytes']+=1;z['byte_t1']+=int(b==rank[0]);z['byte_t5']+=int(b in rank)
            expected=float(w@q);maxerr=max(maxerr,abs(prod-expected));assert e.done(prefix) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            z['bits']-=math.log2(prod);z['events']+=1
            if not native and name in ('learned','switch','static'):
                prefix=b'';i=0
                while i<len(event):
                    tokens=e.tokens(prefix);size=2 if event[i] in b' etaoinr' and not e.done(prefix+event[i:i+1]) else 1
                    target=event[i:i+size];rank=heapq.nlargest(5,tokens,key=tokens.get);z['tokens']+=1;z['token_t1']+=int(target==rank[0]);z['token_t5']+=int(target in rank)
                    mass=np.zeros(256)
                    for k,v in tokens.items():mass[k[0]]+=v
                    marg=max(marg,float(abs(mass-e.next(prefix)).max()));checks+=1;prefix+=target;i+=size
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    rank,cert,_=e.rank(budget=1200);target=event[:-1].lower().decode('ascii')
                    z['words']+=1;z['certified']+=int(cert);z['word_t1']+=int(cert and target==rank[0]);z['word_t5']+=int(cert and target in rank)
        for m in (learned,base,forced):m.observe(f,q)
        for m in controls.values():m.observe(q)
        prev=q;previous=root+event;last=bid
    out={}
    for k,z in stats.items():
        out[k]=dict(z,bpb=z['bits']/z['bytes'],byte_top1=z['byte_t1']/z['bytes'],byte_top5=z['byte_t5']/z['bytes'])
        if z['tokens']:out[k].update(token_top1=z['token_t1']/z['tokens'],token_top5=z['token_t5']/z['tokens'])
        if z['words']:out[k].update(certified_word_top1=z['word_t1']/z['words'],certified_word_top5=z['word_t5']/z['words'],certification_rate=z['certified']/z['words'])
    assert marg<1e-8
    return {'modes':out,'literal_max_error':maxerr,'token_marginal_max_error':marg if checks else None,'marginal_checks':checks}

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
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    minimum=128 if a.native else 256
    fitbooks=(primary+secondary)[:4] if a.native else primary+secondary
    assert not {i for i,p in fitbooks}&{i for group in selected.values() for i,p in group}
    fitrun,fitmanifest=contiguous(fitbooks,minimum)
    train=sequences(rows58([],6101,0,*args,raw=fitrun),raw,surface,a.native)
    run,manifest61=contiguous(selected['calibration'][:3] if a.native else selected['calibration'],minimum)
    valid=sequences(rows58([],6102,0,*args,raw=run),raw,surface,a.native)
    transition,opt,scale,adaptive,static,report=fit61(train,valid)
    out['fit']={'windows':fitmanifest,'calibration_windows':manifest61,**report}
    torch.save({'transition':transition.state_dict(),'optimizer':opt.state_dict(),'scale':scale,'initial':report['initial'],'parent_artifact_hashes':hashes,'updates':96},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    for name in groups:
        run,manifest61=contiguous(selected[name][:3] if a.native else selected[name],minimum)
        result=evaluate61(rows58([],6103,0,*args,raw=run),raw,surface,a.native,transition,scale,adaptive,static)
        assert all(v['bytes']==sum(q['bytes'] for q in manifest61) for v in result['modes'].values())
        out['domains'][name]={'windows':manifest61,'metrics':result};write_metrics(a.out,out,'domain:'+name)
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
