"""v71 discriminative signed lexical energy; frozen ancestral checkpoint."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tests/nancm_trajectory70'),str(root/'tests/nancm_trajectory'),str(root/'tests/nancm_rank_audit')]
from pilot70 import *
from field71 import Discriminator,evidence,tilt,ratio,factors

@torch.no_grad()
def prepare71(run,args,raw,surface,core,scale,bank):
    session=Session(bank);runtime=core.runtime(scale);last=None;prev=None;data=[]
    for row in rows58([],7101,0,*args,raw=run):
        bid,prefix,event,a,b,_=row
        if bid!=last:session.reset();runtime.reset();prev=None
        d,c,_=session.predict(prefix)
        q=likelihood(row,raw,surface,False);ff=features61(prefix,prev);w=runtime.weights(ff)
        runtime.observe(ff,q);prev=q;last=bid
        if not (len(event)>1 and not FE.IS_LETTER[event[-1]]):continue
        target=event[:-1].lower().decode('ascii')
        ids,p,f,ctx,k=evidence(bank,prefix,a,b,d,c,args[0].prior)
        tid=bank.ids.get(target,0);j=int(np.searchsorted(ids,tid));j=j if j<len(ids) and ids[j]==tid else -1
        wp=np.array([FE.Event(a,raw,prefix,surface,0.).complete(target.encode('ascii')),a.complete(target),b.complete(target)])
        data.append((p,f,ctx,k,j,q,wp,w))
    return data

def pack71(rows):
    n=len(rows);m=max(len(r[0][0]) for r in rows);dim=rows[0][2].shape[-1]
    p=np.zeros((n,2,m));f=np.zeros((n,m,10),np.float32);ctx=np.zeros((n,6,dim),np.float32);k=np.zeros((n,m,dim),np.float32)
    t=np.zeros(n,np.int64);q=np.zeros((n,3));wp=q.copy();w=q.copy()
    for i,(pp,ff,cc,kk,j,qq,ww,wwt) in enumerate(rows):
        l=len(ff);p[i,:,:l]=pp;f[i,:l]=ff;ctx[i]=cc;k[i,:l]=kk;t[i]=j;q[i]=qq;wp[i]=ww;w[i]=wwt
    return tuple(torch.from_numpy(x) for x in (p,f,ctx,k,t,q,wp,w))

def fit71(data,dim):
    torch.manual_seed(7102);net=Discriminator(dim);opt=torch.optim.AdamW(net.parameters(),lr=.0015,weight_decay=.03)
    rng=np.random.default_rng(7103);trace=[]
    # Exact full lexical normalization; candidates outside the selected set
    # retain their original relative probabilities, including unknown spelling.
    for i in range(256):
        rows=[data[j] for j in rng.integers(len(data),size=24)]
        p,f,ctx,k,t,q,wp,w=pack71(rows);opt.zero_grad()
        logit=net(f,ctx,k);r=ratio(p,factors(logit,1.),t)
        z=torch.cat((torch.ones_like(r[:,:1]),r),-1)
        ev=-(w*q*z).sum(-1).clamp_min(1e-300).log().mean()
        wd=-(w*wp*z).sum(-1).clamp_min(1e-300).log().mean()
        loss=ev+.5*wd+.002*logit.square().mean()
        loss.backward();assert all(torch.isfinite(v.grad).all() for v in net.parameters())
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if i==0 or (i+1)%32==0:trace.append([i+1,float(ev.detach()),float(wd.detach())])
    assert all(int(opt.state[p]['step'])==256 for p in net.parameters())
    return net.eval(),opt,trace

@torch.no_grad()
def evaluate71(run,args,raw,surface,core,scale,bank,net,arms,audit=False):
    states={k:core.runtime(scale) for k in arms};prev={k:None for k in arms}
    session=Session(bank);metrics=C.defaultdict(C.Counter);books={};last=None;previous=None
    for row in rows58([],7104,0,*args,raw=run):
        bid,prefix,event,a,b,_=row
        if bid!=last:
            session.reset()
            for k,s in states.items():s.reset();prev[k]=None
        else:assert prefix==previous
        d,c,_=session.predict(prefix)
        ids,p,f,ctx,k=evidence(bank,prefix,a,b,d,c,args[0].prior)
        inputs=tuple(torch.from_numpy(x) for x in (f,ctx,k))
        logit=net(*inputs)
        bad=None
        if any(mode=='wrong' for mode,g in arms.values()):
            ii,pp,ff,cc,kk=evidence(bank,prefix,a,b,d,c,args[0].prior,True);assert np.array_equal(ii,ids)
            bad=net(*[torch.from_numpy(x) for x in (ff,cc,kk)])
        baseq=likelihood(row,raw,surface,False)
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        target=event[:-1].lower().decode('ascii') if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
        for name,(mode,gain) in arms.items():
            if gain:
                ff=factors(bad if mode=='wrong' else logit,gain).numpy()
                aa=tilt(a,ids,ff,Lift);bb=tilt(b,ids,ff,Lift)
                q=np.array([baseq[0],FE.Event(aa,raw,prefix,surface,0.).literal(event)[1],FE.Event(bb,raw,prefix,surface,0.).literal(event)[1]])
                if target is not None:
                    tid=bank.ids.get(target,0);j=int(np.searchsorted(ids,tid));j=j if j<len(ids) and ids[j]==tid else -1
                    rr=ratio(torch.from_numpy(p),torch.from_numpy(ff),torch.tensor(j)).numpy()
                    assert np.allclose(q[1:],baseq[1:]*rr,rtol=1e-9,atol=1e-30)
            else:aa,bb,q=a,b,baseq
            runtime=states[name];features0=features61(prefix,prev[name]);w=runtime.weights(features0);share=float(w[1:].sum())
            law=fuse(aa,bb,float(w[2]/share))
            e=FE.Event(law,raw,prefix,surface,share);z=metrics[name];pfx=b'';prod=1.
            for byte in event:
                v=e.next(pfx);ranked=np.argsort(-v,kind='stable')[:5]
                z['bytes']+=1;z['byte_t1']+=int(byte==ranked[0]);z['byte_t5']+=int(byte in ranked)
                prod*=v[byte];pfx+=bytes([byte])
            expected=float(w@q);assert e.done(pfx) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            assert math.isclose(runtime.observe(features0,q),expected,rel_tol=1e-10,abs_tol=1e-30)
            prev[name]=q;bits=-math.log2(prod);z['bits']+=bits;z['events']+=1
            bm[name]['bits']+=bits;bm[name]['bytes']+=len(event)
            if target is not None:
                z['words']+=1;bm[name]['words']+=1
                z['candidate_hits']+=int(bank.ids.get(target,0) in ids)
                z['word_bits']-=math.log2(max(e.complete(target.encode('ascii')),1e-300))
                if audit:
                    ranked,cert,n=e.rank(budget=1200)
                    z['certified']+=int(cert);z['rank_nodes']+=n
                    h1=int(cert and target==ranked[0]);h5=int(cert and target in ranked)
                    z['word_t1']+=h1;z['word_t5']+=h5;bm[name]['word_t1']+=h1;bm[name]['word_t5']+=h5
            if audit and z['events']==1:
                t=e.tokens(b'');m=np.zeros(256)
                for k0,v in t.items():m[k0[0]]+=v
                assert np.max(np.abs(m-e.next(b'')))<1e-8
                code=RAW.encode(prefix,event,lambda h:e.next(h[len(prefix):]))
                dec=FE.Event(law,raw,prefix,surface,share)
                assert RAW.decode(prefix,code,len(event),lambda h:dec.next(h[len(prefix):]))==event
                z['codec_checks']+=1
        last=bid;previous=prefix+event
    return {'modes':{k:dict(v,bpb=v['bits']/v['bytes'],byte_top1=v['byte_t1']/v['bytes'],byte_top5=v['byte_t5']/v['bytes'],word_top1=v['word_t1']/max(1,v['words']),word_top5=v['word_t5']/max(1,v['words'])) for k,v in metrics.items()},'per_book':{b:{k:dict(v) for k,v in m.items()} for b,m in books.items()}}

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
    assert tm['report_stage']=='final' and tc['parent_artifact_hashes']==hashes and tc['updates']==96
    core=Transition(*tc['initial']);core.load_state_dict(tc['transition']);core.requires_grad_(False);core.eval()
    thash=hashlib.sha256(a.transition.read_bytes()).hexdigest()
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    # Only original TRAIN references, never calibration or screened DEV.
    train_ids={i for i,p in tr+extra};eval_ids={i for group in selected.values() for i,p in group}
    assert not train_ids&eval_ids
    train_sha={hashlib.sha256(p.read_bytes()).hexdigest() for i,p in tr+extra}
    eval_sha={hashlib.sha256(p.read_bytes()).hexdigest() for group in selected.values() for i,p in group}
    assert not train_sha&eval_sha
    trajectory=PathField(index.words,index.code,expanded)
    out['reference']={'books':len(train_ids),'paths':len(trajectory.paths),'dimensions':trajectory.code.shape[1],'train_eval_ids_disjoint':True,'train_eval_sha_disjoint':True}
    write_metrics(a.out,out,'reference')
    fitbooks=(primary+secondary)[:24]
    assert len(fitbooks)>=8 and not {i for i,p in fitbooks}&(train_ids|eval_ids)
    assert not {hashlib.sha256(p.read_bytes()).hexdigest() for i,p in fitbooks}&(train_sha|eval_sha)
    # Use longer fit windows when available; never substitute DEV books.
    run=[];fitwindows=[]
    for book in fitbooks:
        try:r,m=contiguous([book],1024)
        except AssertionError:r,m=contiguous([book],256)
        run.extend(r);fitwindows.extend(m)
    train=prepare71(run,args,raw,surface,core,float(tc['scale']),trajectory)
    net,opt,trace=fit71(train,trajectory.code.shape[1])
    out['fit']={'windows':fitwindows,'updates':256,'examples':len(train),'trace':trace,'parameters':sum(p.numel() for p in net.parameters())}
    torch.save({'discriminator':net.state_dict(),'optimizer':opt.state_dict(),'updates':256,'parent_artifact_hashes':hashes,'transition_sha256':thash},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    run,windows=contiguous(selected['calibration'],256)
    grid={str(x):('full',x) for x in (0.,.0625,.125,.25,.5,.75,1.)}
    valid=evaluate71(run,args,raw,surface,core,float(tc['scale']),trajectory,net,grid)
    chosen=min(grid,key=lambda k:valid['modes'][k]['bpb']);alpha=grid[chosen][1]
    out['calibration']={'windows':windows,'selected_gain':alpha,'metrics':valid};write_metrics(a.out,out,'calibration')
    arms={'parent_v61':('full',0.),'selected':('full',alpha),'forced':('full',1.),'wrong_geometry':('wrong',1.)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate71(run,args,raw,surface,core,float(tc['scale']),trajectory,net,arms,True)
        expected=tm['domains'][name]['metrics']['modes']['learned'];actual=result['modes']['parent_v61']
        assert actual['bytes']==expected['bytes']==sum(q['bytes'] for q in windows)
        ranks=all(actual[k]==expected[k] for k in ('words','word_t1','word_t5','certified','byte_t1','byte_t5'))
        replay={'expected_bpb':expected['bpb'],'bpb_delta':actual['bpb']-expected['bpb'],'ranks_equal':ranks}
        replay['passed']=ranks and abs(replay['bpb_delta'])<1e-8
        if alpha==0.:assert result['modes']['selected']==actual
        out['domains'][name]={'windows':windows,'metrics':result,'parent_replay':replay};write_metrics(a.out,out,'domain:'+name)
    out['historical_replay_verified']=all(v['parent_replay']['passed'] for v in out['domains'].values())
    if not out['historical_replay_verified']:
        write_metrics(a.out,out,'replay_failed');raise AssertionError('historical replay failed')
    assert all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,tc['transition'][k]) for k,v in core.state_dict().items())
    assert hashlib.sha256(a.transition.read_bytes()).hexdigest()==thash
    out['eligible_for_promotion']=False
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
