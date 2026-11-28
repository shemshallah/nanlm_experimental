"""v94: resume v88 optimizer and calibrate its signed field in a trust region."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_signed88'))
from pilot88 import *
from field94 import Trust94,objective94

def fit94(train,probe,ck88):
    field=Field88();field.load_state_dict(ck88['field88'])
    opt=torch.optim.AdamW(field.parameters(),lr=.0015,weight_decay=.03)
    opt.load_state_dict(ck88['optimizer88'])
    assert all(int(s['step'])==512 for s in opt.state.values())
    for group in opt.param_groups:group['lr']=.0003
    model=Trust94(field)
    opt.add_param_group({'params':[model.theta],'lr':.003,'weight_decay':.01})
    rng=np.random.default_rng(9402);ix=np.random.default_rng(9403).choice(len(probe),min(256,len(probe)),replace=False)
    fixed=pack88(probe,ix)
    def measure():
        with torch.no_grad():return [float(v) for v in objective94(model,fixed)]
    trace=[[0,*measure()]]
    for i in range(384):
        opt.zero_grad()
        ev,wd,rank,kl,regress=objective94(model,pack88(train,rng.integers(len(train),size=32)))
        loss=ev+.75*wd+.2*rank+2.*kl+.5*regress
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if (i+1)%64==0:
            trace.append([i+1,*measure()]);print('TRUST94_PROBE',trace[-1],flush=True)
    assert all(int(opt.state[p]['step'])==896 for p in field.parameters())
    assert int(opt.state[model.theta]['step'])==384
    return model.eval().requires_grad_(False),opt,trace

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
    for k in ('v79','v79-metrics','v81','v81-metrics','v86','v86-metrics','v88','v88-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
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
    ck79=torch.load(a.v79,map_location='cpu',weights_only=True);m79=json.loads(a.v79_metrics.read_text())
    ck81=torch.load(a.v81,map_location='cpu',weights_only=True);m81=json.loads(a.v81_metrics.read_text())
    assert ck79['v78_sha256']==ck81['v78_sha256']==v78sha and ck79['updates']==512 and ck81['updates']==256
    assert m79['report_stage']==m81['report_stage']=='final'
    for child in (ck79,ck81):
        assert child['parent_artifact_hashes']==hashes and child['transition_sha256']==thash
    v79sha=hashlib.sha256(a.v79.read_bytes()).hexdigest();v81sha=hashlib.sha256(a.v81.read_bytes()).hexdigest()
    escape=Escape();escape.load_state_dict(ck79['escape']);escape.eval();escape.requires_grad_(False)
    gate81=Gate81();gate81.load_state_dict(ck81['gate81']);gate81.eval();gate81.requires_grad_(False)
    field76.Suffix=FastSuffix;field79.Suffix=FastSuffix
    contexts=ContextField(index.words,index.seq,model.cls.cpu().numpy())
    seqs=[np.array(index.encode(B.load(p,20000)),np.int32) for _,p in tr+extra]
    evidence86=Evidence(index,seqs,model.cls.cpu().numpy())
    frozen=FrozenAddress(address)
    out['integration']={'v79_sha256':v79sha,'v81_sha256':v81sha,'reference_words':sum(map(len,seqs)),'reference_books':len(seqs),'v79_historical_replay':m79['historical_replay_verified'],'v81_historical_replay':m81['historical_replay_verified'],'no_retraining':True}
    write_metrics(a.out,out,'integration')
    ck86=torch.load(a.v86,map_location='cpu',weights_only=True);m86=json.loads(a.v86_metrics.read_text())
    assert m86['report_stage']=='final' and ck86['v78_sha256']==v78sha and ck86['v79_sha256']==v79sha and ck86['v81_sha256']==v81sha
    inherited=tuple(ck86['selected_arm']);v86sha=hashlib.sha256(a.v86.read_bytes()).hexdigest()
    assert inherited==('exact',1.,1.)
    escape.load_state_dict(ck86['v79_escape']);gate81.load_state_dict(ck86['v81_gate'])
    ck88=torch.load(a.v88,map_location='cpu',weights_only=True);m88=json.loads(a.v88_metrics.read_text())
    assert m88['report_stage']=='final' and ck88['v86_sha256']==v86sha and ck88['updates']==512 and ck88['selected_gain']==.125
    v88sha=hashlib.sha256(a.v88.read_bytes()).hexdigest()
    old=Field88();old.load_state_dict(ck88['field88']);old.eval().requires_grad_(False)
    def make(books):
        run=[];windows=[]
        for book in books:
            try:r,m=contiguous([book],1024)
            except AssertionError:r,m=contiguous([book],256)
            run.extend(r);windows.extend(m)
        return prepare88(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,frozen,evidence86,escape,gate81,inherited),windows
    train,fitwindows=make(fitbooks[:-4]);probe,probewindows=make(fitbooks[-4:])
    field,opt,trace=fit94(train,probe,ck88)
    out['fit']={'updates':384,'signed_optimizer_step':896,'gain_optimizer_step':384,'examples':len(train),'probe_examples':len(probe),'windows':fitwindows,'probe_windows':probewindows,'trace':trace,'v88_sha256':v88sha,'params':sum(p.numel() for p in field.parameters()),'probe_previously_seen_by_v88':True}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'v88':ck88,'v88_sha256':v88sha,'trust94':field.state_dict(),'optimizer94':opt.state_dict(),'updates':384},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    run,windows=contiguous(selected['calibration'],256)
    grid={str(x):('full',x) for x in (0.,.25,.5,.75,1.)}
    valid=evaluate88(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,frozen,evidence86,escape,gate81,field,inherited,grid,True)
    prevcal=evaluate88(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,frozen,evidence86,escape,gate81,old,inherited,{'v88':('full',.125)},True)
    eligible=[k for k in grid if valid['modes'][k]['bpb']<=valid['modes']['0.0']['bpb']]
    chosen=min(eligible,key=lambda k:(-valid['modes'][k]['word_t1'],-valid['modes'][k]['word_t5'],valid['modes'][k]['word_bits']))
    arm=grid[chosen]
    out['calibration']={'windows':windows,'selected_gain':arm[1],'metrics':valid,'v88':prevcal};write_metrics(a.out,out,'calibration')
    arms={'parent_v86':('full',0.),'selected':arm,'forced':('full',1.),'wrong_evidence':('wrong',arm[1]),'wrong_full':('wrong',1.),'zero_duplicate':('wrong',0.)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate88(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,frozen,evidence86,escape,gate81,field,inherited,arms,True)
        prior=evaluate88(run,args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,frozen,evidence86,escape,gate81,old,inherited,{'v88':('full',.125)},True)
        assert result['modes']['zero_duplicate']==result['modes']['parent_v86']
        replay={}
        for key,actual,expected in [('v86',result['modes']['parent_v86'],m86['domains'][name]['metrics']['modes']['selected']),('v88',prior['modes']['v88'],m88['domains'][name]['metrics']['modes']['selected'])]:
            assert actual['bytes']==expected['bytes']==sum(q['bytes'] for q in windows)
            ranks=all(actual[k]==expected[k] for k in ('words','word_t1','word_t5','certified','byte_t1','byte_t5'))
            delta=actual['bpb']-expected['bpb'];replay[key]={'ranks_equal':ranks,'bpb_delta':delta,'passed':ranks and abs(delta)<1e-8}
        replay['passed']=all(v['passed'] for v in replay.values())
        result['modes']['v88']=prior['modes']['v88']
        out['domains'][name]={'windows':windows,'metrics':result,'parent_replay':replay,'within_run_zero_identity':True};write_metrics(a.out,out,'domain:'+name)
    torch.save({'v88':ck88,'v88_sha256':v88sha,'trust94':field.state_dict(),'optimizer94':opt.state_dict(),'updates':384,'selected_gain':arm[1]},a.out.with_suffix('.pt'))
    assert hashlib.sha256(a.v88.read_bytes()).hexdigest()==v88sha
    assert hashlib.sha256(a.v86.read_bytes()).hexdigest()==v86sha
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
