"""v73: resume actual generator optimizer, reconcile prospective state causally."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_generator72'))
from pilot72 import *
from field73 import Reconciliation,shifted

def fit73(index,ck):
    net=Generator(index.code,index.prior);net.load_state_dict(ck['generator'])
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01);opt.load_state_dict(ck['optimizer'])
    assert int(opt.state[net.lex.weight]['step'])==512
    torch.manual_seed(7301);bridge=Reconciliation()
    opt.param_groups[0]['lr']=.0006
    opt.add_param_group(dict(params=bridge.parameters(),lr=.0015))
    rng=np.random.default_rng(7302);h,y=batch(index.seq,np.random.default_rng(7303),128)
    def probe():
        with torch.no_grad():
            a=bridge.logits(net,h);b=bridge.logits(net,shifted(h,y[:,0]),h)
            return [float(F.cross_entropy(a,y[:,0])),float(F.cross_entropy(b,y[:,1])),float((a.argmax(-1)==y[:,0]).float().mean())]
    trace=[dict(step=512,reference_probe=probe())]
    for i in range(384):
        x,t=batch(index.seq,rng,32);opt.zero_grad()
        a=bridge.logits(net,x);b=bridge.logits(net,shifted(x,t[:,0]),x)
        _,future=net.logits(x,True)
        # Proper conditional likelihood for each step, plus a marginal forecast.
        # Never equate a marginal forecast with a conditioned prediction.
        loss=F.cross_entropy(a,t[:,0])+.75*F.cross_entropy(b,t[:,1])+.25*F.cross_entropy(future,t[:,1])
        loss.backward();torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(bridge.parameters()),1.);opt.step()
        if (i+1)%64==0:trace.append(dict(step=512+i+1,reference_probe=probe()))
    assert int(opt.state[net.lex.weight]['step'])==896
    assert int(opt.state[bridge.strength]['step'])==384
    assert all(torch.isfinite(v).all() for v in list(net.parameters())+list(bridge.parameters()))
    return net.eval(),bridge.eval(),opt,trace

@torch.no_grad()
def evaluate73(run,args,raw,surface,core,scale,bank,net,parent,generator,bridge,arms,audit=False):
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
        hh=generator.root(prefix,bank.ids)
        gp=bridge.logits(generator,hh).softmax(-1)[0].double().numpy()
        gp/=math.fsum(gp);gen=Lift(a.outside,gp)
        pp=parent(hh)[0].double().numpy();pp/=math.fsum(pp);oldgen=Lift(a.outside,pp)
        wrong=None;unreconciled=None
        if any(mode=='wrong' for mode,g in arms.values()):
            prevh=F.pad(hh[:,:-1],(1,0))
            # Scramble only the previous forecast address, preserve the actual
            # observed word and current prediction state.
            prevh=prevh.flip(-1)
            wp=bridge.logits(generator,hh,prevh).softmax(-1)[0].double().numpy()
            wp/=math.fsum(wp);wrong=Lift(a.outside,wp)
        if any(mode=='off' for mode,g in arms.values()):
            up=generator(hh)[0].double().numpy();up/=math.fsum(up);unreconciled=Lift(a.outside,up)
        ff=factors(logit,.5).numpy()
        a71=tilt(a,ids,ff,Lift);b71=tilt(b,ids,ff,Lift)
        gen_rank=np.argsort(-gp[1:],kind='stable')[:64]+1
        old_rank=np.argsort(-pp[1:],kind='stable')[:64]+1
        baseq=likelihood(row,raw,surface,False)
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        target=event[:-1].lower().decode('ascii') if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
        for name,(mode,gain) in arms.items():
            if mode=='ancestral':
                aa,bb=a,b
            else:
                aa,bb=(a,b) if mode=='generator_only' else (a71,b71)
                if gain:
                    donor=wrong if mode=='wrong' else oldgen if mode=='parent' else unreconciled if mode=='off' else gen
                    aa=fuse(aa,donor,gain);bb=fuse(bb,donor,gain)
            q=np.array([baseq[0],FE.Event(aa,raw,prefix,surface,0.).literal(event)[1],FE.Event(bb,raw,prefix,surface,0.).literal(event)[1]])
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
                tid=bank.ids.get(target,0)
                z['candidate_hits']+=int(tid in ids)
                z['generator_t1']+=int(tid==gen_rank[0])
                z['generator_t5']+=int(tid in gen_rank[:5])
                z['generator_t64']+=int(tid in gen_rank)
                z['new_generator_hits']+=int(tid in gen_rank and tid not in ids)
                z['parent_generator_t64']+=int(tid in old_rank)
                z['parent_generator_t5']+=int(tid in old_rank[:5])
                z['parent_generator_t1']+=int(tid==old_rank[0])
                z['generator_nll']-=math.log(max(gen.complete(target),1e-300))
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
    ap.add_argument('--v71',type=pathlib.Path,required=True)
    ap.add_argument('--v71-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v72',type=pathlib.Path,required=True)
    ap.add_argument('--v72-metrics',type=pathlib.Path,required=True)
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
    parent=Generator(index.code,index.prior);parent.load_state_dict(ck72['generator']);parent.eval();parent.requires_grad_(False)
    v72sha=hashlib.sha256(a.v72.read_bytes()).hexdigest()
    generator,bridge,opt,trace=fit73(index,ck72)
    out['fit']={'updates':384,'generator_optimizer_step':896,'bridge_optimizer_step':384,'batch':32,'reference_tokens':sum(map(len,index.seq)),'trace':trace,'bridge_strength':float(bridge.strength.detach()),'v72_sha256':v72sha}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'generator':generator.state_dict(),'bridge':bridge.state_dict(),'optimizer':opt.state_dict(),'v71':ck71,'v71_sha256':v71sha,'v72_sha256':v72sha,'updates':896,'parent_artifact_hashes':hashes,'transition_sha256':thash},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    run,windows=contiguous(selected['calibration'],256)
    oldgain=float(m72['calibration']['selected_gain'])
    grid={str(x):('full',x) for x in (0.,.03125,.0625,.125,.25,.5,.75)}
    grid['parent_v72']=('parent',oldgain)
    valid=evaluate73(run,args,raw,surface,core,float(tc['scale']),trajectory,net,parent,generator,bridge,grid)
    chosen=min(grid,key=lambda k:valid['modes'][k]['bpb'])
    selected_arm=grid[chosen]
    out['calibration']={'windows':windows,'selected_mode':selected_arm[0],'selected_gain':selected_arm[1],'metrics':valid};write_metrics(a.out,out,'calibration')
    arms={'parent_v61':('ancestral',0.),'parent_v71':('full',0.),'parent_v72':('parent',oldgain),'selected':selected_arm,'forced':('full',.25),'reconciliation_off':('off',.25),'wrong_address':('wrong',.25)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate73(run,args,raw,surface,core,float(tc['scale']),trajectory,net,parent,generator,bridge,arms,True)
        replay={}
        for key,expected in [('parent_v61',tm['domains'][name]['metrics']['modes']['learned']),('parent_v71',m71['domains'][name]['metrics']['modes']['selected']),('parent_v72',m72['domains'][name]['metrics']['modes']['selected'])]:
            actual=result['modes'][key]
            assert actual['bytes']==expected['bytes']==sum(q['bytes'] for q in windows)
            ranks=all(actual[k]==expected[k] for k in ('words','word_t1','word_t5','certified','byte_t1','byte_t5'))
            replay[key]={'expected_bpb':expected['bpb'],'bpb_delta':actual['bpb']-expected['bpb'],'ranks_equal':ranks,'passed':ranks and abs(actual['bpb']-expected['bpb'])<1e-8}
        replay['passed']=all(v['passed'] for v in replay.values())
        out['domains'][name]={'windows':windows,'metrics':result,'parent_replay':replay};write_metrics(a.out,out,'domain:'+name)
    assert hashlib.sha256(a.v72.read_bytes()).hexdigest()==v72sha
    assert all(torch.equal(v,ck72['generator'][k]) for k,v in parent.state_dict().items())
    assert hashlib.sha256(a.v71.read_bytes()).hexdigest()==v71sha
    assert all(torch.equal(v,ck71['discriminator'][k]) for k,v in net.state_dict().items())
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
