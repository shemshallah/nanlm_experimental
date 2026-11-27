"""v82: replace the three-byte raw expert by an exact long-context event law."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_address78'))
sys.path.insert(0,str(root/'tests/nancm_hurdle81'))
from pilot78 import *
from field82 import LongByte,LongEvent,MixedEvent,event82

@torch.no_grad()
def parent81(prefix,a,b,index,bank,session,net,ancestry,mem,contexts,gate,competition,oldgain,address):
    a71,b71,a75,b75,a76,b76,laws=context_parent(prefix,a,b,bank,session,net,ancestry,mem,contexts,gate,index)
    pool,p,f,k,c,t=candidate_features(prefix,a75,b75,a76,b76,laws,index)
    energy=competition(*[torch.from_numpy(x) for x in (f,k,c,t)]).double().numpy()
    a77,b77=tilt77(a76,pool,np.exp(oldgain*energy)),tilt77(b76,pool,np.exp(oldgain*energy))
    gp=address.log_probs(address.root(prefix,index.ids,observed_words))[0].double().exp().numpy();gp/=math.fsum(gp)
    donor=Lift(a75.outside,gp)
    return fuse(a77,donor,.25),fuse(b77,donor,.25),pool

@torch.no_grad()
def evaluate82(run,args,raw,surface,core,scale,bank,net,ancestry,contexts,gate,competition,oldgain,address,long,arms,audit=False):
    gen,bridge,memory,coupler=ancestry;session=Session(bank);mem=Runtime(gen,memory,bank.ids)
    states={k:core.runtime(scale) for k in arms};prev={k:None for k in arms};metrics=C.defaultdict(C.Counter)
    last=None;previous=None;books={}
    for row in rows58([],7604,0,*args,raw=run):
        bid,prefix,event,a,b,_=row
        if bid!=last:
            session.reset();mem.reset()
            for k,s in states.items():s.reset();prev[k]=None
        else:assert prefix==previous
        aa,bb,pool=parent81(prefix,a,b,args[0],bank,session,net,ancestry,mem,contexts,gate,competition,oldgain,address)
        q=np.array([likelihood(row,raw,surface,False)[0],component_event(aa,raw,prefix,surface,event),component_event(bb,raw,prefix,surface,event)])
        target=event[:-1].lower().decode('ascii') if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
        raw_old=FE.Event(aa,raw,prefix,surface,0.)
        raw_new=LongEvent(aa,long,prefix,surface,0.)
        new_q=raw_new.literal(event)[0]
        depth,support=long.support(prefix)
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        for name,(mode,gain) in arms.items():
            runtime=states[name];ff=features61(prefix,prev[name]);w=runtime.weights(ff)
            share=float(w[1:].sum());law=fuse(aa,bb,float(w[2]/share))
            e=event82(law,raw,long,prefix,surface,share,gain)
            # Raw state and transition probabilities are independent of
            # the lexical law and arm. Share them across all comparisons.
            if isinstance(e,MixedEvent):
                e.a._states=raw_old._states;e.a._raw=raw_old._raw
                e.b._states=raw_new._states;e.b._raw=raw_new._raw
            else:e._states=raw_old._states;e._raw=raw_old._raw
            adjusted=q.copy();adjusted[0]=(1-gain)*q[0]+gain*new_q
            expected=float(w@adjusted)
            z=metrics[name];pfx=b'';prod=1.
            for byte in event:
                v=e.next(pfx);ranked=np.argsort(-v,kind='stable')[:5]
                z['bytes']+=1;z['byte_t1']+=int(byte==ranked[0]);z['byte_t5']+=int(byte in ranked)
                prod*=v[byte];pfx+=bytes([byte])
            assert e.done(pfx) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            assert math.isclose(runtime.observe(ff,adjusted),expected,rel_tol=1e-10,abs_tol=1e-30)
            prev[name]=adjusted;bits=-math.log2(prod);z['bits']+=bits;z['events']+=1
            z['context_depth_sum']+=depth;z['context_support_sum']+=support;bm[name]['bits']+=bits;bm[name]['bytes']+=len(event)
            if target is not None:
                z['words']+=1;bm[name]['words']+=1;z['word_bits']-=math.log2(max(e.complete(target.encode('ascii')),1e-300))
                if audit:
                    ranked,cert,n=e.rank(budget=1200)
                    z['certified']+=int(cert);z['rank_nodes']+=n
                    h1=int(cert and target==ranked[0]);h5=int(cert and target in ranked)
                    z['word_t1']+=h1;z['word_t5']+=h5;bm[name]['word_t1']+=h1;bm[name]['word_t5']+=h5
                    if name=='parent_v78':
                        for key,ee in [('raw_old',raw_old),('raw_new',raw_new)]:
                            rr,cc,_=ee.rank(budget=1200)
                            z[key+'_certified']+=int(cc);z[key+'_t1']+=int(cc and target==rr[0]);z[key+'_t5']+=int(cc and target in rr)
            if audit and z['events']==1:
                t=e.tokens(b'');m0=np.zeros(256)
                for k,v in t.items():m0[k[0]]+=v
                assert np.max(np.abs(m0-e.next(b'')))<1e-8
                code=RAW.encode(prefix,event,lambda h:e.next(h[len(prefix):]))
                assert RAW.decode(prefix,code,len(event),lambda h:e.next(h[len(prefix):]))==event
                z['codec_checks']+=1
        last=bid;previous=prefix+event
    return {'modes':{k:dict(v,bpb=v['bits']/v['bytes'],byte_top1=v['byte_t1']/v['bytes'],byte_top5=v['byte_t5']/v['bytes'],word_top1=v['word_t1']/max(1,v['words']),word_top5=v['word_t5']/max(1,v['words'])) for k,v in metrics.items()},'per_book':{b:{k:dict(v) for k,v in m.items()} for b,m in books.items()}}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen','transition','transition-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
    ap.add_argument('--v71',type=pathlib.Path,required=True)
    ap.add_argument('--v71-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v72',type=pathlib.Path,required=True)
    ap.add_argument('--v72-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v73',type=pathlib.Path,required=True)
    ap.add_argument('--v73-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v74',type=pathlib.Path,required=True)
    ap.add_argument('--v74-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v75',type=pathlib.Path,required=True)
    ap.add_argument('--v75-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v76',type=pathlib.Path,required=True)
    ap.add_argument('--v76-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v77',type=pathlib.Path,required=True)
    ap.add_argument('--v77-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v78',type=pathlib.Path,required=True)
    ap.add_argument('--v78-metrics',type=pathlib.Path,required=True)
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
    # Restore the actual v71 trained weights and optimizer state; never refit
    # them on DEV. Only the new generative field is optimized here.
    ck71=torch.load(a.v71,map_location='cpu',weights_only=True);m71=json.loads(a.v71_metrics.read_text())
    assert ck71['updates']==256 and ck71['parent_artifact_hashes']==hashes and ck71['transition_sha256']==thash
    assert m71['report_stage']=='final' and m71['calibration']['selected_gain']==.5
    net=Discriminator(trajectory.code.shape[1]);net.load_state_dict(ck71['discriminator']);net.eval();net.requires_grad_(False)
    v71sha=hashlib.sha256(a.v71.read_bytes()).hexdigest()
    ck72=torch.load(a.v72,map_location='cpu',weights_only=True);m72=json.loads(a.v72_metrics.read_text())
    assert ck72['updates']==512 and ck72['parent_artifact_hashes']==hashes and ck72['transition_sha256']==thash
    assert ck72['v71_sha256']==v71sha and m72['report_stage']=='final'
    v72sha=hashlib.sha256(a.v72.read_bytes()).hexdigest()
    ck73=torch.load(a.v73,map_location='cpu',weights_only=True);m73=json.loads(a.v73_metrics.read_text())
    assert ck73['updates']==896 and ck73['parent_artifact_hashes']==hashes and ck73['transition_sha256']==thash
    assert ck73['v72_sha256']==v72sha and ck73['v71_sha256']==v71sha
    assert m73['report_stage']=='replay_failed' and m73['fit']['generator_optimizer_step']==896
    v73sha=hashlib.sha256(a.v73.read_bytes()).hexdigest()
    ck74=torch.load(a.v74,map_location='cpu',weights_only=True);m74=json.loads(a.v74_metrics.read_text())
    assert ck74['updates']==1152 and ck74['parent_artifact_hashes']==hashes and ck74['transition_sha256']==thash
    assert ck74['v73_sha256']==v73sha and ck74['v71_sha256']==v71sha
    assert m74['report_stage']=='final'
    v74sha=hashlib.sha256(a.v74.read_bytes()).hexdigest()
    ck75=torch.load(a.v75,map_location='cpu',weights_only=True);m75=json.loads(a.v75_metrics.read_text())
    assert ck75['updates']==1536 and ck75['parent_artifact_hashes']==hashes and ck75['transition_sha256']==thash
    assert ck75['v74_sha256']==v74sha and ck75['v71_sha256']==v71sha
    assert m75['report_stage']=='final' and m75['calibration']['selected_gain']==.0625
    v75sha=hashlib.sha256(a.v75.read_bytes()).hexdigest()
    ancestry=restore75(index,ck75)
    contexts=ContextField(index.words,index.seq,model.cls.cpu().numpy())
    out['context_index']={'exact_positions':len(contexts.exact.pos),'semantic_positions':len(contexts.semantic.pos),'exact_order':8,'semantic_order':6,'books':len(train_ids),'collision_free':True}
    write_metrics(a.out,out,'context_index')
    ck76=torch.load(a.v76,map_location='cpu',weights_only=True);m76=json.loads(a.v76_metrics.read_text())
    assert ck76['v75_sha256']==v75sha and ck76['updates']==256 and m76['report_stage']=='final'
    assert m76['calibration']['selected_gain']==.5 and ck76['parent_artifact_hashes']==hashes
    v76sha=hashlib.sha256(a.v76.read_bytes()).hexdigest()
    gate=Gate();gate.load_state_dict(ck76['gate']);gate.eval();gate.requires_grad_(False)
    ck77=torch.load(a.v77,map_location='cpu',weights_only=True);m77=json.loads(a.v77_metrics.read_text())
    assert ck77['v76_sha256']==v76sha and ck77['updates']==384 and m77['report_stage']=='final'
    assert ck77['parent_artifact_hashes']==hashes and ck77['transition_sha256']==thash
    oldgain=float(m77['calibration']['selected_gain'])
    v77sha=hashlib.sha256(a.v77.read_bytes()).hexdigest()
    competition=Competition();competition.load_state_dict(ck77['competition']);competition.eval();competition.requires_grad_(False)
    ck78=torch.load(a.v78,map_location='cpu',weights_only=True);m78=json.loads(a.v78_metrics.read_text())
    assert ck78['v77_sha256']==v77sha and ck78['updates']==4096 and m78['report_stage']=='final'
    assert ck78['parent_artifact_hashes']==hashes and ck78['transition_sha256']==thash
    assert m78['calibration']['selected_gain']==.25
    v78sha=hashlib.sha256(a.v78.read_bytes()).hexdigest()
    address=Address(index.code,index.prior,model.members,model.cls,model.slot,ancestry[0].lex.weight.detach())
    address.load_state_dict(ck78['address']);address.eval();address.requires_grad_(False)
    # Keep the exact v78 lexical ancestry; extend only the raw donor.
    longdocs=[p.read_bytes()[:16384] for _,p in tr+extra]
    long=LongByte(longdocs,12)
    out['long_byte']={'books':len(longdocs),'reference_bytes':sum(map(len,longdocs)),'indexed_positions':len(long.idx.pos),'max_order':12,'ids_disjoint':True,'sha_disjoint':True}
    write_metrics(a.out,out,'long_byte_index')
    run,windows=contiguous(selected['calibration'],256)
    grid={str(x):('full',x) for x in (0.,.125,.25,.5,.75,1.)}
    valid=evaluate82(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,address,long,grid,True)
    eligible=[k for k in grid if valid['modes'][k]['bpb']<=valid['modes']['0.0']['bpb']]
    chosen=min(eligible,key=lambda k:(-valid['modes'][k]['word_t1'],-valid['modes'][k]['word_t5'],valid['modes'][k]['word_bits']))
    arm=grid[chosen]
    out['calibration']={'windows':windows,'selected_gain':arm[1],'metrics':valid,'criterion':'certified top1 then top5 subject to no BPB regression'};write_metrics(a.out,out,'calibration')
    arms={'parent_v78':('full',0.),'selected':arm,'forced':('full',1.),'half':('full',.5),'zero_duplicate':('full',0.)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate82(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,address,long,arms,True)
        actual=result['modes']['parent_v78'];expected=m78['domains'][name]['metrics']['modes']['selected']
        assert result['modes']['zero_duplicate']=={k:v for k,v in actual.items() if not k.startswith(('raw_old_','raw_new_'))}
        assert actual['bytes']==expected['bytes']==sum(q['bytes'] for q in windows)
        ranks=all(actual[k]==expected[k] for k in ('words','word_t1','word_t5','certified','byte_t1','byte_t5'))
        delta=actual['bpb']-expected['bpb'];replay={'ranks_equal':ranks,'bpb_delta':delta,'passed':ranks and abs(delta)<1e-8}
        out['domains'][name]={'windows':windows,'metrics':result,'parent_replay':replay,'within_run_zero_identity':True};write_metrics(a.out,out,'domain:'+name)
    # The immutable index is reproducible from this hash-verified manifest.
    torch.save({'v78':ck78,'v78_sha256':v78sha,'long_order':12,'long_gain':arm[1],'reference_manifest':[{'id':i,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for i,p in tr+extra],'byte_cap':16384,'parent_artifact_hashes':hashes,'transition_sha256':thash},a.out.with_suffix('.pt'))
    assert hashlib.sha256(a.v78.read_bytes()).hexdigest()==v78sha
    assert all(torch.equal(v,ck78['address'][k]) for k,v in address.state_dict().items())
    assert hashlib.sha256(a.v77.read_bytes()).hexdigest()==v77sha
    assert all(torch.equal(v,ck77['competition'][k]) for k,v in competition.state_dict().items())
    assert hashlib.sha256(a.v76.read_bytes()).hexdigest()==v76sha
    assert all(torch.equal(v,ck76['gate'][k]) for k,v in gate.state_dict().items())
    assert hashlib.sha256(a.v75.read_bytes()).hexdigest()==v75sha
    for m,k in zip(ancestry,('generator','bridge','memory','coupler')):
        assert all(torch.equal(v,ck75[k][n]) for n,v in m.state_dict().items())
    assert hashlib.sha256(a.v71.read_bytes()).hexdigest()==v71sha
    assert all(torch.equal(v,ck71['discriminator'][k]) for k,v in net.state_dict().items())
    out['historical_replay_verified']=all(v['parent_replay']['passed'] for v in out['domains'].values())
    out['strict_replay_gate_failed']=not out['historical_replay_verified']
    # Preserve diagnostics even when cross-run numerical replay fails.
    # Promotion remains forbidden, regardless of within-run improvement.
    assert all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,tc['transition'][k]) for k,v in core.state_dict().items())
    assert hashlib.sha256(a.transition.read_bytes()).hexdigest()==thash
    out['eligible_for_promotion']=False
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
