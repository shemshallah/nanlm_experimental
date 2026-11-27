"""v69: matched corpus test of persistent trajectory evidence, frozen parent."""
import sys,pathlib,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_rank_audit'))
from pilot67 import *
from trajectory69 import Trajectory

@torch.no_grad()
def evaluate69(run,args,raw,surface,core,scale,trajectory,arms,audit=False):
    states={k:core.runtime(scale) for k in arms};prev={k:None for k in arms}
    metrics=C.defaultdict(C.Counter);books={};last=None;previous=None
    for source,row in zip(run,rows58([],6601,0,*args,raw=run)):
        bid,prefix,event,l53,l55,_=row;assert source[:3]==row[:3]
        if bid!=last:
            for k,s in states.items():s.reset();prev[k]=None
        else:assert prefix==previous
        q=likelihood(row,raw,surface,False);raw0=raw(prefix)
        proposals={mode:trajectory.propose(prefix,l53,l55,raw0,mode) for mode in sorted(set(m for m,g in arms.values() if g))}
        # Same reference hypotheses and evidence budget for all ablations.
        sparse_lik={mode:FE.Event(s,raw,prefix,surface,0.).literal(event)[1] for mode,(s,c,d) in proposals.items() if s is not None}
        target=event[:-1].lower().decode('ascii') if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        for name,(mode,alpha) in arms.items():
            runtime=states[name];f=features61(prefix,prev[name]);w=runtime.weights(f);share=float(w[1:].sum())
            base=fuse(l53,l55,float(w[2]/share))
            sparse,conf,detail=proposals.get(mode,(None,0.,{}))
            gain=alpha*conf if sparse is not None else 0.
            law=fuse(base,sparse,gain) if gain else base
            qq=q.copy()
            if gain:qq[1:]=(1.-gain)*qq[1:]+gain*sparse_lik[mode]
            e=FE.Event(law,raw,prefix,surface,share);z=metrics[name];p=b'';prod=1.
            for b in event:
                d=e.next(p);ranked=np.argsort(-d,kind='stable')[:5]
                z['bytes']+=1;z['byte_t1']+=int(b==ranked[0]);z['byte_t5']+=int(b in ranked)
                prod*=d[b];p+=bytes([b])
            expected=float(w@qq)
            assert e.done(p) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            assert math.isclose(runtime.observe(f,qq),expected,rel_tol=1e-10,abs_tol=1e-30)
            prev[name]=qq;bits=-math.log2(prod)
            z['bits']+=bits;z['events']+=1;z['active']+=int(gain>0);z['gain_sum']+=gain
            bm[name]['bits']+=bits;bm[name]['bytes']+=len(event)
            if target is not None:
                z['words']+=1;bm[name]['words']+=1
                z['lexical_bits']-=math.log2(max(law.complete(target),1e-300))
                if sparse is not None:
                    z['proposal_hits']+=int(sparse.complete(target)>0);z['proposal_positions']+=1
                    z['new_candidate_hits']+=int(sparse.complete(target)>0 and target not in [x.decode('ascii').lower() for x in re.findall(rb'[A-Za-z]+',prefix)])
                if audit:
                    # Identical historical rank algorithm and budgets for every arm.
                    words,cert,n=e.rank(budget=1200)
                    z['certified']+=int(cert);z['rank_nodes']+=n
                    hit1=int(cert and target==words[0]);hit5=int(cert and target in words)
                    z['word_t1']+=hit1;z['word_t5']+=hit5;bm[name]['word_t1']+=hit1;bm[name]['word_t5']+=hit5
                    z['word_bits']-=math.log2(max(e.complete(target.encode('ascii')),1e-300))
            if audit and z['events']==1:
                t=e.tokens(b'');m=np.zeros(256)
                for k,v in t.items():m[k[0]]+=v
                assert np.max(np.abs(m-e.next(b'')))<1e-8
                code=RAW.encode(prefix,event,lambda h:e.next(h[len(prefix):]))
                dec=FE.Event(law,raw,prefix,surface,share)
                assert RAW.decode(prefix,code,len(event),lambda h:dec.next(h[len(prefix):]))==event
                z['codec_checks']+=1
        last=bid;previous=prefix+event
    return {'modes':{k:dict(v,bpb=v['bits']/v['bytes'],byte_top1=v['byte_t1']/v['bytes'],byte_top5=v['byte_t5']/v['bytes'],word_top1=v['word_t1']/max(1,v['words']),word_top5=v['word_t5']/max(1,v['words']),lexical_nll=v['lexical_bits']/max(1,v['words'])*math.log(2)) for k,v in metrics.items()},'per_book':{b:{k:dict(v) for k,v in m.items()} for b,m in books.items()}}

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
    trajectory=Trajectory(index.words,index.code,expanded)
    out['reference']={'books':len(train_ids),'paths':len(trajectory.paths),'dimensions':trajectory.code.shape[1],'train_eval_ids_disjoint':True,'train_eval_sha_disjoint':True}
    write_metrics(a.out,out,'reference')
    run,windows=contiguous(selected['calibration'],256)
    grid={str(x):('full',x) for x in (0.,.0625,.125,.25,.5,.75)}
    valid=evaluate69(run,args,raw,surface,core,float(tc['scale']),trajectory,grid)
    chosen=min(grid,key=lambda k:valid['modes'][k]['bpb']);alpha=grid[chosen][1]
    out['calibration']={'windows':windows,'selected_gain':alpha,'metrics':valid};write_metrics(a.out,out,'calibration')
    arms={'parent_v61':('full',0.),'selected':('full',alpha),'forced':('full',.25),'no_prospective':('signed',.25),'exact_only':('exact',.25),'wrong_address':('wrong',.25)}
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=evaluate69(run,args,raw,surface,core,float(tc['scale']),trajectory,arms,True)
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
