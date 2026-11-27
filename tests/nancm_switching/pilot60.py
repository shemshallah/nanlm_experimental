"""v60 persistent causal posterior over the raw and open word event laws."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_open_bridge'))
from pilot59 import *
from switching import Switch

PRIORS=[(r,(1-r)*(1-g),(1-r)*g) for r in (.25,.5,.75) for g in (.125,.5,.75,.875)]
HAZARDS=(.02,.1,.35,1.)

def likelihood(row,raw,surface,native):
    bid,root,event,l53,l55,parent=row;typ=NativeEvent if native else FE.Event
    a=parent if native else l53
    r,s0=typ(a,raw,root,surface,0.).literal(event)
    _,s1=typ(l55,raw,root,surface,0.).literal(event)
    return np.array([r,s0,s1],np.float64)

def calibrate60(rows,raw,surface,native):
    filters={(p,h):Switch(p,h) for p in PRIORS for h in HAZARDS}
    losses=C.Counter();last=None;n=0;events=0
    for row in rows:
        bid,root,event,*_=row
        if bid!=last:
            for m in filters.values():m.reset()
        else:assert root==previous
        q=likelihood(row,raw,surface,native)
        for k,m in filters.items():losses[k]-=math.log2(m.observe(q))
        previous=root+event;last=bid;n+=len(event);events+=1
    assert n>0
    adaptive=min((k for k in losses if k[1]<1),key=losses.get)
    fixed=min((k for k in losses if k[1]==1),key=losses.get)
    return adaptive,fixed,{'bytes':n,'events':events,'adaptive':adaptive,'static':fixed,'grid':{str(k):v/n for k,v in losses.items()}}

def evaluate60(rows,raw,surface,native,adaptive,fixed):
    typ=NativeEvent if native else FE.Event
    fs={'adaptive':Switch(*adaptive),'same_prior_static':Switch(adaptive[0],1.),'calibrated_static':Switch(*fixed),'v58_fixed':Switch((.5,.125,.375),1.)}
    stats=C.defaultdict(C.Counter);bybook={};last=None;maxerr=0.;maxmarg=0.;checks=0
    for row in rows:
        bid,root,event,l53,l55,parent=row;a=parent if native else l53
        if bid!=last:
            for f in fs.values():f.reset()
        else:assert root==previous
        previous=root+event;last=bid
        lik=likelihood(row,raw,surface,native)
        bm=bybook.setdefault(bid,C.defaultdict(C.Counter))
        for name in ('raw',*fs):
            w=np.array([1.,0.,0.]) if name=='raw' else fs[name].weights()
            share=float(w[1]+w[2]);law=fuse(a,l55,float(w[2]/share)) if share else a
            e=typ(law,raw,root,surface,share);z=stats[name];prefix=b'';prod=1.
            for b in event:
                q=e.next(prefix);rank=np.argsort(-q,kind='stable')[:5];prod*=q[b];prefix+=bytes([b])
                z['bytes']+=1;z['byte_t1']+=int(b==rank[0]);z['byte_t5']+=int(b in rank)
            expected=float(w@lik);maxerr=max(maxerr,abs(prod-expected))
            assert e.done(prefix) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30),(bid,name,prod,expected)
            if name!='raw':
                observed=fs[name].observe(lik);assert math.isclose(observed,prod,rel_tol=1e-7,abs_tol=1e-30)
            z['bits']-=math.log2(prod);z['events']+=1
            for i,weight in enumerate(w):z['prior_'+str(i)]+=float(weight)
            bm[name]['bits']-=math.log2(prod);bm[name]['bytes']+=len(event)
            # Exact event-relative subword and folded ranks for finite-order donors.
            if not native and name in ('adaptive','calibrated_static'):
                prefix=b'';i=0
                while i<len(event):
                    t=e.tokens(prefix);size=2 if event[i] in b' etaoinr' and not e.done(prefix+event[i:i+1]) else 1
                    target=event[i:i+size];rank=heapq.nlargest(5,t,key=t.get);z['tokens']+=1;z['token_t1']+=int(target==rank[0]);z['token_t5']+=int(target in rank)
                    mass=np.zeros(256)
                    for tok,v in t.items():mass[tok[0]]+=v
                    maxmarg=max(maxmarg,float(abs(mass-e.next(prefix)).max()));checks+=1;prefix+=target;i+=size
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    rank,cert,_=e.rank(budget=1200);target=event[:-1].lower().decode('ascii')
                    z['words']+=1;z['certified']+=int(cert);z['word_t1']+=int(cert and target==rank[0]);z['word_t5']+=int(cert and target in rank)
    out={}
    for k,z in stats.items():
        out[k]=dict(z,bpb=z['bits']/z['bytes'],byte_top1=z['byte_t1']/z['bytes'],byte_top5=z['byte_t5']/z['bytes'],mean_prior=[z['prior_'+str(i)]/z['events'] for i in range(3)])
        if z['tokens']:out[k].update(token_top1=z['token_t1']/z['tokens'],token_top5=z['token_t5']/z['tokens'])
        if z['words']:out[k].update(certified_word_top1=z['word_t1']/z['words'],certified_word_top5=z['word_t5']/z['words'],certification_rate=z['certified']/z['words'])
    assert maxmarg<1e-8
    return {'modes':out,'per_book':{bid:{k:dict(v,bpb=v['bits']/v['bytes']) for k,v in book.items()} for bid,book in bybook.items()},'literal_max_error':maxerr,'token_marginal_max_error':maxmarg if checks else None,'marginal_checks':checks}

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
    run,manifest60=contiguous(selected['calibration'][:3] if a.native else selected['calibration'],minimum)
    adaptive,fixed,report=calibrate60(rows58([],6001,0,*args,raw=run),raw,surface,a.native)
    out['calibration']={'windows':manifest60,**report};write_metrics(a.out,out,'calibration')
    for name in groups:
        run,manifest60=contiguous(selected[name][:3] if a.native else selected[name],minimum)
        result=evaluate60(rows58([],6002,0,*args,raw=run),raw,surface,a.native,adaptive,fixed)
        assert all(v['bytes']==sum(q['bytes'] for q in manifest60) for v in result['modes'].values())
        out['domains'][name]={'windows':manifest60,'metrics':result};write_metrics(a.out,out,'domain:'+name)
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
