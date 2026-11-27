"""v80: resume signed address optimizer with long relational state."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_escape79'))
from pilot79 import *
from field80 import Address80,Positions

def fit80(index,model,ancestry,ck):
    torch.manual_seed(8001)
    net=Address80(index.code,index.prior,model.members,model.cls,model.slot,ancestry[0].lex.weight.detach())
    miss=net.load_state_dict(ck['address'],strict=False)
    assert not miss.unexpected_keys and all(k.startswith('relation.') for k in miss.missing_keys)
    base=[p for k,p in net.named_parameters() if not k.startswith('relation.')]
    opt=torch.optim.AdamW(base,lr=.0015,weight_decay=.01)
    opt.load_state_dict(ck['optimizer'])
    assert int(opt.state[net.delta.weight]['step'])==4096
    opt.param_groups[0]['lr']=.001
    opt.add_param_group(dict(params=net.relation.parameters(),lr=.0015))
    positions=Positions(index.seq,8002,probe_books=8)
    probe=Positions(index.seq[:8],8003);x,y=probe.take(256)
    def measure():
        with torch.no_grad():
            z=net.log_probs(x);r=z.topk(5,dim=-1).indices
            return [float(-z[torch.arange(len(y)),y].mean()),float((r[:,0]==y).float().mean()),float((r==y[:,None]).any(-1).float().mean())]
    trace=[[4096,*measure()]]
    for i in range(8192):
        h,t=positions.take(64);opt.zero_grad()
        loss=-net.target_log(h,t).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if (i+1)%1024==0:
            trace.append([4096+i+1,*measure()]);print('RELATION_PROBE',trace[-1],flush=True)
    assert int(opt.state[net.delta.weight]['step'])==12288
    assert int(opt.state[net.relation.read.weight]['step'])==8192
    return net.eval(),opt,trace,positions.at

@torch.no_grad()
def evaluate80(run,args,raw,surface,core,scale,bank,net,ancestry,contexts,gate,competition,oldgain,address,evidence79,escape,escape_gain,relation,arms,audit=False):
    gen,bridge,memory,coupler=ancestry
    states={k:core.runtime(scale) for k in arms};prev={k:None for k in arms}
    session=Session(bank);mem=Runtime(gen,memory,bank.ids);metrics=C.defaultdict(C.Counter);books={};last=None;previous=None
    for row in rows58([],7604,0,*args,raw=run):
        bid,prefix,event,a,b,_=row
        if bid!=last:
            session.reset();mem.reset()
            for k,s in states.items():s.reset();prev[k]=None
        else:assert prefix==previous
        a71,b71,a75,b75,a76,b76,laws=context_parent(prefix,a,b,bank,session,net,ancestry,mem,contexts,gate,args[0])
        pool,p,f,k,c,t=candidate_features(prefix,a75,b75,a76,b76,laws,args[0])
        energy=competition(*[torch.from_numpy(x) for x in (f,k,c,t)]).double().numpy()
        oldf=np.exp(oldgain*energy)
        a77,b77=tilt77(a76,pool,oldf),tilt77(b76,pool,oldf)
        # Full-vocabulary donor including the exact escape mass.
        h=address.root(prefix,args[0].ids,observed_words)
        gp=address.log_probs(h)[0].double().exp().numpy()
        gp/=math.fsum(gp);donor=Lift(a75.outside,gp)
        newrank=np.argsort(-gp[1:],kind='stable')[:64]+1
        a78,b78=fuse(a77,donor,.25),fuse(b77,donor,.25)
        ls=evidence79.query(prefix);ft=torch.tensor(np.stack([l[1] for l in ls]));cap=torch.tensor([l[2] for l in ls],dtype=torch.float64)
        w0=escape.allocate(ft,cap,escape_gain).numpy()
        a79,b79=compact(a78,ls,w0),compact(b78,ls,w0)
        hh=relation.root(prefix,args[0].ids,observed_words)
        rp=relation.log_probs(hh)[0].double().exp().numpy();rp/=math.fsum(rp)
        rel=Lift(a75.outside,rp);relrank=np.argsort(-rp[1:],kind='stable')[:64]+1
        wrong=None;off=None
        if any(m=='wrong' for m,g in arms.values()):
            wp=relation.log_probs(relation.root(prefix,args[0].ids,observed_words,True))[0].double().exp().numpy();wp/=math.fsum(wp)
            wrong=Lift(a75.outside,wp)
        if any(m=='off' for m,g in arms.values()):
            relation.scale=0.
            op=relation.log_probs(hh)[0].double().exp().numpy();op/=math.fsum(op)
            relation.scale=1.;off=Lift(a75.outside,op)
        rawq=likelihood(row,raw,surface,False)[0]
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        target=event[:-1].lower().decode('ascii') if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
        base_dense=.5*(a75.p+b75.p);top=np.argsort(-base_dense,kind='stable')[:32]
        parent32={a75.words[i] for i in top}
        # Equal-budget rank of context sources; distinguish novel recovery
        # from merely widening the proposal set.
        for name,(mode,gain) in arms.items():
            if mode=='ancestral':aa,bb=a,b
            elif mode=='v71':aa,bb=a71,b71
            elif mode=='v75':aa,bb=a75,b75
            elif mode=='v76':aa,bb=a76,b76
            elif mode=='v77':aa,bb=a77,b77
            elif mode=='v78':aa,bb=a78,b78
            elif not gain:aa,bb=a79,b79
            else:
                dd=wrong if mode=='wrong' else off if mode=='off' else rel
                # Replace a fraction of the existing v78 donor, rather than
                # stacking another independent probability mixture.
                d=fuse(donor,dd,gain)
                aa=compact(fuse(a77,d,.25),ls,w0);bb=compact(fuse(b77,d,.25),ls,w0)
            q=np.array([rawq,component_event(aa,raw,prefix,surface,event),component_event(bb,raw,prefix,surface,event)])
            runtime=states[name];ff0=features61(prefix,prev[name]);w=runtime.weights(ff0);share=float(w[1:].sum())
            law=fuse(aa,bb,float(w[2]/share));e=FE.Event(law,raw,prefix,surface,share);z=metrics[name];pfx=b'';prod=1.
            for byte in event:
                v=e.next(pfx);ranked=np.argsort(-v,kind='stable')[:5]
                z['bytes']+=1;z['byte_t1']+=int(byte==ranked[0]);z['byte_t5']+=int(byte in ranked)
                prod*=v[byte];pfx+=bytes([byte])
            expected=float(w@q);assert e.done(pfx) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            assert math.isclose(runtime.observe(ff0,q),expected,rel_tol=1e-10,abs_tol=1e-30)
            prev[name]=q;bits=-math.log2(prod);z['bits']+=bits;z['events']+=1
            bm[name]['bits']+=bits;bm[name]['bytes']+=len(event)
            if target is not None:
                z['words']+=1;bm[name]['words']+=1
                z['parent32_hits']+=int(target in parent32)
                z['candidate_hits']+=int(target in pool)
                z['pool_size_sum']+=len(pool)
                scores=.5*(p[0]+p[1])*np.exp(energy)
                top=sorted(range(len(pool)),key=lambda i:(-scores[i],pool[i]))[:32]
                z['competition32_hits']+=int(target in [pool[i] for i in top])
                tid=args[0].ids.get(target,0)
                z['address_t1']+=int(tid==newrank[0])
                z['address_t5']+=int(tid in newrank[:5])
                z['address_t32']+=int(tid in newrank[:32])
                z['address_t64']+=int(tid in newrank)
                z['new_address32']+=int(tid in newrank[:32] and target not in pool)
                z['address_nll']-=math.log(max(donor.complete(target),1e-300))
                z['relation_t1']+=int(tid==relrank[0])
                z['relation_t5']+=int(tid in relrank[:5])
                z['relation_t32']+=int(tid in relrank[:32])
                z['relation_t64']+=int(tid in relrank)
                z['new_relation32']+=int(tid in relrank[:32] and target not in pool)
                z['relation_nll']-=math.log(max(rel.complete(target),1e-300))
                z['lexical_mass_sum']+=share
                if audit and name=='parent_v79':
                    # Diagnostic only: isolate word ranking before/after
                    # the inherited raw/lexical Bayesian mixture.
                    for nm,weight in [('raw',0.),('lexical',1.)]:
                        rr,cert0,_=FE.Event(law,raw,prefix,surface,weight).rank(budget=1200)
                        z[nm+'_certified']+=int(cert0)
                        z[nm+'_t1']+=int(cert0 and target==rr[0])
                        z[nm+'_t5']+=int(cert0 and target in rr)

                for ix,(l,_,mass) in enumerate(ls):
                    if l:
                        z['context_available_'+str(ix)]+=1;z['context_hits_'+str(ix)]+=int(l.complete(target)>0)

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
    ap.add_argument('--v79',type=pathlib.Path,required=True)
    ap.add_argument('--v79-metrics',type=pathlib.Path,required=True)
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
    # Extend the same immutable, hash-checked TRAIN books, not DEV or fit books.
    seqs=[np.array(index.encode(B.load(p,20000)),np.int32) for _,p in tr+extra]
    evidence79=Evidence(index,seqs,model.cls.cpu().numpy())
    out['expanded_reference']={'books':len(seqs),'words':sum(map(len,seqs)),'exact_positions':len(evidence79.exact.pos),'semantic_positions':len(evidence79.semantic.pos),'cap_words_per_book':20000,'ids_disjoint':True,'sha_disjoint':True}
    write_metrics(a.out,out,'extended_reference')
    ck79=torch.load(a.v79,map_location='cpu',weights_only=True);m79=json.loads(a.v79_metrics.read_text())
    assert ck79['v78_sha256']==v78sha and ck79['updates']==512 and m79['report_stage']=='final'
    assert ck79['parent_artifact_hashes']==hashes and ck79['transition_sha256']==thash
    escape_gain=float(m79['calibration']['selected_gain'])
    v79sha=hashlib.sha256(a.v79.read_bytes()).hexdigest()
    escape=Escape();escape.load_state_dict(ck79['escape']);escape.eval();escape.requires_grad_(False)
    relation,opt,trace,unique=fit80(index,model,ancestry,ck78)
    out['fit']={'updates':8192,'batch':64,'distinct_positions':unique,'address_optimizer_step':12288,'relation_optimizer_step':8192,'trace':trace,'v79_sha256':v79sha,'v78_sha256':v78sha,'probe_scope':'eight reference books excluded from v80 training, not from inherited training'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'address':relation.state_dict(),'optimizer':opt.state_dict(),'updates':12288,'relation_updates':8192,'v79':ck79,'v79_sha256':v79sha,'parent_artifact_hashes':hashes,'transition_sha256':thash},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    run,windows=contiguous(selected['calibration'],256)
    grid={str(x):('full',x) for x in (0.,.125,.25,.5,.75,1.)}
    valid=evaluate80(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,address,evidence79,escape,escape_gain,relation,grid,True)
    eligible=[k for k in grid if valid['modes'][k]['bpb']<=valid['modes']['0.0']['bpb']]
    chosen=min(eligible,key=lambda k:(-valid['modes'][k]['word_t1'],-valid['modes'][k]['word_t5'],valid['modes'][k]['word_bits']))
    selected_arm=grid[chosen]
    out['calibration']={'windows':windows,'selected_gain':selected_arm[1],'metrics':valid};write_metrics(a.out,out,'calibration')
    arms={'parent_v61':('ancestral',0.),'parent_v71':('v71',0.),'parent_v78':('v78',0.),'parent_v79':('full',0.),'selected':selected_arm,'forced':('full',1.),'relation_off':('off',1.),'wrong_older_history':('wrong',1.),'zero_duplicate':('wrong',0.)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate80(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,address,evidence79,escape,escape_gain,relation,arms,True)
        a0={k:v for k,v in result['modes']['parent_v79'].items() if not k.startswith(('raw_','lexical_t','lexical_certified'))}
        assert result['modes']['zero_duplicate']==a0
        replay={}
        for key,expected in [('parent_v61',tm['domains'][name]['metrics']['modes']['learned']),('parent_v71',m71['domains'][name]['metrics']['modes']['selected']),('parent_v78',m78['domains'][name]['metrics']['modes']['selected']),('parent_v79',m79['domains'][name]['metrics']['modes']['selected'])]:
            actual=result['modes'][key]
            assert actual['bytes']==expected['bytes']==sum(q['bytes'] for q in windows)
            ranks=all(actual[k]==expected[k] for k in ('words','word_t1','word_t5','certified','byte_t1','byte_t5'))
            replay[key]={'expected_bpb':expected['bpb'],'bpb_delta':actual['bpb']-expected['bpb'],'ranks_equal':ranks,'passed':ranks and abs(actual['bpb']-expected['bpb'])<1e-8}
        replay['passed']=all(v['passed'] for v in replay.values())
        out['domains'][name]={'windows':windows,'metrics':result,'parent_replay':replay,'within_run_zero_identity':True};write_metrics(a.out,out,'domain:'+name)
    assert hashlib.sha256(a.v79.read_bytes()).hexdigest()==v79sha
    assert all(torch.equal(v,ck79['escape'][k]) for k,v in escape.state_dict().items())
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
