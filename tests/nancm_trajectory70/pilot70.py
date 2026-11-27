"""v70 exact event-credit BPTT for causal path transport; frozen parent."""
import sys,pathlib,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tests/nancm_trajectory'),str(root/'tests/nancm_rank_audit')]
from pilot69 import *
from field70 import PathField,Session,Allocation,features

def proposals(session,prefix,l53,l55,raw):
    direct,carry,evidence=session.predict(prefix)
    base=fuse(l53,l55,.5)
    f=features(prefix,base,direct,carry,evidence)
    return direct,carry,f,np.array([direct is not None,carry is not None])

def components(row,raw,surface,direct,carry):
    bid,prefix,event,l53,l55,_=row
    q=likelihood(row,raw,surface,False)
    s=[FE.Event(x,raw,prefix,surface,0.).literal(event)[1] if x is not None else 0. for x in (direct,carry)]
    target=event[:-1].lower() if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
    word=np.zeros(5,np.float64)
    if target is not None:
        word[0]=FE.Event(l53,raw,prefix,surface,0.).complete(target)
        for i,x in enumerate((l53,l55,direct,carry),1):
            if x is not None:word[i]=x.complete(target.decode('ascii'))
    return np.r_[q,s],word,target

@torch.no_grad()
def prepare70(run,args,raw,surface,bank):
    session=Session(bank);docs=C.defaultdict(list);last=None
    for row in rows58([],7001,0,*args,raw=run):
        bid,prefix,event,l53,l55,_=row
        if bid!=last:session.reset()
        d,c,f,mask=proposals(session,prefix,l53,l55,raw)
        q,word,target=components(row,raw,surface,d,c)
        docs[bid].append((f,mask,q,word,features61(prefix,None)[:5],target is not None,len(event)))
        last=bid
    return docs

def packed(docs,keys):
    n=len(keys);t=max(len(docs[k]) for k in keys)
    f=np.zeros((n,t,10));a=np.zeros((n,t,2),bool);q=np.ones((n,t,5));word=np.ones((n,t,5))
    f0=np.zeros((n,t,5));wm=np.zeros((n,t),bool);m=np.zeros((n,t),bool);nb=0
    for i,k in enumerate(keys):
        for j,(x,y,z,u,v,w,b) in enumerate(docs[k]):
            f[i,j]=x;a[i,j]=y;q[i,j]=z;word[i,j]=u;f0[i,j]=v;wm[i,j]=w;m[i,j]=True;nb+=b
    return tuple(torch.from_numpy(x) for x in (f,a,q,word,f0,wm,m)),nb

def trajectory_loss(gate,core,scale,batch,word_weight=.25,allocation_scale=1.):
    f,a,q,word,f0,wm,m=batch
    allocations=gate.weights(f,a,allocation_scale)
    state=core.prior.expand(len(f),3);prev=None;total=f.new_zeros(());words=f.new_zeros(())
    for j in range(f.shape[1]):
        if j==0:w=state
        else:
            z=prev.clamp_min(1e-300).log()
            v=((z[:,1:]-z[:,:1])/8).clamp(-1,1)
            w=torch.einsum('bi,bij->bj',state,core.matrix(torch.cat((f0[:,j],v),-1),scale))
        a0,ad,ac=allocations[:,j].unbind(-1)
        def adjusted(x):
            mix=ad*x[:,3]+ac*x[:,4]
            return torch.stack((x[:,0],a0*x[:,1]+mix,a0*x[:,2]+mix),-1)
        lik=adjusted(q[:,j]);joint=w*lik;p=joint.sum(-1).clamp_min(1e-300)
        total=total-torch.where(m[:,j],p.log(),0.).sum()
        pw=(w*adjusted(word[:,j])).sum(-1).clamp_min(1e-300)
        words=words-torch.where(wm[:,j],pw.log(),0.).sum()
        state=torch.where(m[:,j,None],joint/p[:,None],state);prev=lik
    return total+word_weight*words,total,words

def fit70(train,core,scale):
    gate=Allocation();opt=torch.optim.AdamW(gate.parameters(),lr=.035,weight_decay=.01)
    rng=np.random.default_rng(7002);keys=list(train);trace=[]
    for i in range(128):
        ks=[keys[j] for j in rng.choice(len(keys),min(8,len(keys)),replace=False)]
        batch,n=packed(train,ks);opt.zero_grad();loss,ev,wd=trajectory_loss(gate,core,scale,batch)
        (loss/batch[-1].sum()).backward()
        assert torch.isfinite(gate.theta.grad).all()
        torch.nn.utils.clip_grad_norm_(gate.parameters(),1.);opt.step()
        if i==0 or (i+1)%16==0:trace.append([i+1,float(ev.detach()/n/math.log(2)),float(wd.detach()/batch[-2].sum().clamp_min(1))])
    assert int(opt.state[gate.theta]['step'])==128
    return gate.eval(),opt,trace

@torch.no_grad()
def evaluate70(run,args,raw,surface,core,scale,bank,gate,arms,audit=False):
    states={k:core.runtime(scale) for k in arms};prev={k:None for k in arms}
    sessions={k:Session(bank,wrong=(k=='wrong')) for k in ('right','wrong')}
    metrics=C.defaultdict(C.Counter);books={};last=None;previous=None
    for row in rows58([],7003,0,*args,raw=run):
        bid,prefix,event,l53,l55,_=row
        if bid!=last:
            for k,s in states.items():s.reset();prev[k]=None
            for s in sessions.values():s.reset()
        else:assert prefix==previous
        evidence={}
        for mode in sorted(set(mode for mode,g in arms.values())):
            key='wrong' if mode=='wrong' else 'right'
            if key not in evidence:
                d,c,f,mask=proposals(sessions[key],prefix,l53,l55,raw)
                q,word,target=components(row,raw,surface,d,c);evidence[key]=(d,c,f,mask,q,word,target)
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        for name,(mode,gain) in arms.items():
            d,c,f,mask,q,word,target=evidence['wrong' if mode=='wrong' else 'right']
            if mode=='direct':mask=mask.copy();mask[1]=False
            a0,ad,ac=gate.weights(torch.from_numpy(f),torch.from_numpy(mask),gain).numpy()
            runtime=states[name];ff=features61(prefix,prev[name]);w=runtime.weights(ff);share=float(w[1:].sum())
            law=fuse(l53,l55,float(w[2]/share))
            if ad:law=fuse(law,d,float(ad/(a0+ad)))
            if ac:law=fuse(law,c,float(ac))
            qq=q[:3].copy();qq[1:]=a0*qq[1:]+ad*q[3]+ac*q[4]
            e=FE.Event(law,raw,prefix,surface,share);z=metrics[name];p=b'';prod=1.
            for b in event:
                v=e.next(p);ranked=np.argsort(-v,kind='stable')[:5]
                z['bytes']+=1;z['byte_t1']+=int(b==ranked[0]);z['byte_t5']+=int(b in ranked)
                prod*=v[b];p+=bytes([b])
            expected=float(w@qq);assert e.done(p) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            assert math.isclose(runtime.observe(ff,qq),expected,rel_tol=1e-10,abs_tol=1e-30)
            prev[name]=qq;bits=-math.log2(prod);z['bits']+=bits;z['events']+=1
            z['direct_mass']+=float(ad);z['carry_mass']+=float(ac)
            bm[name]['bits']+=bits;bm[name]['bytes']+=len(event)
            if target is not None:
                z['words']+=1;bm[name]['words']+=1
                z['direct_hits']+=int(d is not None and d.complete(target.decode('ascii'))>0)
                z['carry_hits']+=int(c is not None and c.complete(target.decode('ascii'))>0)
                z['carry_available']+=int(c is not None)
                if audit:
                    words,cert,n=e.rank(budget=1200)
                    z['certified']+=int(cert);z['rank_nodes']+=n
                    h1=int(cert and target.decode('ascii')==words[0]);h5=int(cert and target.decode('ascii') in words)
                    z['word_t1']+=h1;z['word_t5']+=h5;bm[name]['word_t1']+=h1;bm[name]['word_t5']+=h5
                    z['word_bits']-=math.log2(max(e.complete(target),1e-300))
            if audit and z['events']==1:
                t=e.tokens(b'');m=np.zeros(256)
                for k,v in t.items():m[k[0]]+=v
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
    run,fitwindows=contiguous(fitbooks,256)
    train=prepare70(run,args,raw,surface,trajectory)
    gate,opt,trace=fit70(train,core,float(tc['scale']))
    out['fit']={'windows':fitwindows,'updates':128,'trace':trace,'theta':gate.theta.detach().tolist()}
    torch.save({'allocation':gate.state_dict(),'optimizer':opt.state_dict(),'updates':128,'parent_artifact_hashes':hashes,'transition_sha256':thash},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    run,windows=contiguous(selected['calibration'],256)
    grid={str(x):('full',x) for x in (0.,.125,.25,.5,.75,1.)}
    valid=evaluate70(run,args,raw,surface,core,float(tc['scale']),trajectory,gate,grid)
    chosen=min(grid,key=lambda k:valid['modes'][k]['bpb']);alpha=grid[chosen][1]
    out['calibration']={'windows':windows,'selected_gain':alpha,'metrics':valid};write_metrics(a.out,out,'calibration')
    arms={'parent_v61':('full',0.),'selected':('full',alpha),'forced':('full',1.),'direct_only':('direct',1.),'wrong_address':('wrong',1.)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate70(run,args,raw,surface,core,float(tc['scale']),trajectory,gate,arms,True)
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
