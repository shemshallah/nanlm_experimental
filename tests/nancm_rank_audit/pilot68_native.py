"""v68 native signed transport, calibration-only selection and counterfactuals."""
import sys,pathlib,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_native_stream'))
from pilot64 import *
from tight_native import rank_native
from transport68 import Transport
from word_bound import CompletionBound

def lexical_rank(law,k=5,budget=1500):
    bound=CompletionBound(law);heap=[(-1.,'')];best=[];n=0
    while heap and n<budget:
        if len(best)>=k and -heap[0][0]<best[0][0]:break
        _,p=heapq.heappop(heap);n+=1
        if p:
            item=(law.complete(p),p)
            if len(best)<k:heapq.heappush(best,item)
            elif item>best[0]:heapq.heapreplace(best,item)
        if len(p)<63:
            for c in 'abcdefghijklmnopqrstuvwxyz':
                h=p+c;u=bound(h)
                if u>0 and (len(best)<k or u>=best[0][0]):heapq.heappush(heap,(-u,h))
    cert=len(best)>=k and (not heap or -heap[0][0]<best[0][0])
    return [p for _,p in sorted(best,reverse=True)],cert,n

def run68(rows,raw,surface,adaptive,transport,arms,audit_rank=False):
    switches={name:Switch(*adaptive) for name in arms}
    stats=C.defaultdict(C.Counter);books={};last=None;previous=None
    for row in rows:
        bid,prefix,event,l53,l55,parent=row
        if bid!=last:
            for switch in switches.values():switch.reset()
        else:assert prefix==previous
        rootq=raw(prefix);q=likelihood(row,raw,surface,True)
        proposals={}
        for mode in set(mode for mode,alpha in arms.values() if alpha):
            proposals[mode]=transport.propose(prefix,parent,l55,rootq,mode)
        target=event[:-1].lower().decode('ascii') if len(event)>1 and not FE.IS_LETTER[event[-1]] else None
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        for name,(mode,alpha) in arms.items():
            switch=switches[name];w=switch.weights();share=float(w[1]+w[2])
            base=fuse(parent,l55,float(w[2]/share))
            sparse,conf,details=proposals.get(mode,(None,0.,{}))
            gain=alpha*conf if sparse is not None else 0.
            law=fuse(base,sparse,gain) if gain else base
            qq=q.copy()
            if gain:
                s=NativeEvent(sparse,raw,prefix,surface,0.).literal(event)[1]
                qq[1:]=(1.-gain)*qq[1:]+gain*s
            e=NativeEvent(law,raw,prefix,surface,share)
            z=stats[name];p=b'';prod=1.
            for b in event:
                d=e.next(p);ranked=np.argsort(-d,kind='stable')[:5]
                z['bytes']+=1;z['byte_t1']+=int(b==ranked[0]);z['byte_t5']+=int(b in ranked)
                prod*=d[b];p+=bytes([b])
            expected=float(w@qq)
            assert e.done(p) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            assert math.isclose(switch.observe(qq),expected,rel_tol=1e-10,abs_tol=1e-30)
            bits=-math.log2(prod);z['bits']+=bits;z['events']+=1;z['active']+=int(gain>0);z['gain_sum']+=gain
            bm[name]['bits']+=bits;bm[name]['bytes']+=len(event)
            if target is not None:
                z['words']+=1;bm[name]['words']+=1
                z['lexical_bits']-=math.log2(max(law.complete(target),1e-300))
                if sparse is not None:
                    z['proposal_hits']+=int(sparse.complete(target)>0);z['proposal_positions']+=1
                if audit_rank:
                    words,cert,n=lexical_rank(law);z['lexical_nodes']+=n
                    z['lexical_certified']+=int(cert);z['lexical_t1']+=int(cert and target==words[0]);z['lexical_t5']+=int(cert and target in words)
                    bm[name]['lexical_t1']+=int(cert and target==words[0])
                    bm[name]['lexical_t5']+=int(cert and target in words)
                    # Bounded native ranking on first eight complete word
                    # events per domain, same deterministic sample for all arms.
                    if z['words']<=8:
                        result=rank_native(law,raw,prefix,share,budget=256)
                        ranked=[v[0] for v in result['ranked']];ok=result['certified']
                        z['native_audited']+=1;z['native_certified']+=int(ok);z['native_queries']+=result['queries']
                        z['native_t1']+=int(ok and ranked and target==ranked[0]);z['native_t5']+=int(ok and target in ranked)
            if audit_rank and z['events']==1:
                # Fresh event decoder, independent of observed target cache.
                code=RAW.encode(prefix,event,lambda h:e.next(h[len(prefix):]))
                decoder=NativeEvent(law,PrefixMemo(raw.donor,256),prefix,surface,share)
                assert RAW.decode(prefix,code,len(event),lambda h:decoder.next(h[len(prefix):]))==event
                t=decoder.tokens(b'');m=np.zeros(256)
                for k,v in t.items():m[k[0]]+=v
                assert np.max(np.abs(m-decoder.next(b'')))<1e-8
                z['codec_checks']+=1
        last=bid;previous=prefix+event
    return {'modes':{k:dict(v,bpb=v['bits']/v['bytes'],byte_top1=v['byte_t1']/v['bytes'],byte_top5=v['byte_t5']/v['bytes'],lexical_nll=v['lexical_bits']/max(1,v['words'])*math.log(2)) for k,v in stats.items()},'per_book':{b:{k:dict(v) for k,v in m.items()} for b,m in books.items()}}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen','native-metrics','transition','transition-metrics','native-stream-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
    ap.add_argument('--native',action='store_true')
    keys=('lib','parent','recurrent','direct','dual','symbiotic','fusion','coordinate','temporal','relational','lexical','byte','subword','resonance','attractor','unified')
    for k in keys:ap.add_argument('--'+k,type=pathlib.Path)
    ap.add_argument('--rank-budget',type=int,default=64)
    a=ap.parse_args();torch.set_num_threads(4);assert a.native
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
    previous=json.loads(a.native_metrics.read_text());assert previous['report_stage']=='final' and previous['native'] and previous['recurrent_unchanged']
    assert previous['recurrent_checkpoint_sha256']==field.v37_sha and previous['parent_artifact_hashes']==hashes
    tc=torch.load(a.transition,map_location='cpu',weights_only=True);tm=json.loads(a.transition_metrics.read_text())
    assert tm['report_stage']=='final' and tc['parent_artifact_hashes']==hashes and tc['updates']==96
    transfer=Transition(*tc['initial']);transfer.load_state_dict(tc['transition']);transfer.eval()
    parent64=json.loads(a.native_stream_metrics.read_text())
    assert parent64['report_stage']=='final' and parent64['recurrent_unchanged'] and parent64['historical_replay_verified']
    assert parent64['recurrent_checkpoint_sha256']==field.v37_sha and parent64['parent_artifact_hashes']==hashes
    raw=PrefixMemo(raw,4096)
    out={'native':True,'parent_artifact_hashes':hashes,'recurrent_checkpoint_sha256':field.v37_sha,'scope':'v64 matched windows; calibration-only gain; native ranks audited on first eight words/domain; lexical metrics separate','domains':{}}
    write_metrics(a.out,out,'index')
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    transport=Transport(index.words,index.code)
    adaptive=parent64['calibration']['adaptive']
    run,windows=contiguous(selected['calibration'][:3],128)
    assert not {q[0] for q in run}&{i for name in groups for i,p in selected[name]}
    grid={str(x):('full',x) for x in (0.,.125,.25,.5,.75)}
    valid=run68(rows58([],6401,0,*args,raw=run),raw,surface,adaptive,transport,grid)
    chosen=min(grid,key=lambda k:valid['modes'][k]['bpb']);alpha=grid[chosen][1]
    out['calibration']={'windows':windows,'selected_gain':alpha,'metrics':valid}
    write_metrics(a.out,out,'calibration')
    arms={'parent_v64':('full',0.),'selected':('full',alpha),'forced':('full',.5),'exact_only':('exact',.5),'wrong_address':('wrong',.5)}
    for name in groups:
        run,windows=contiguous(selected[name][:2],128)
        assert windows==parent64['domains'][name]['windows']
        result=run68(rows58([],6402,0,*args,raw=run),raw,surface,adaptive,transport,arms,True)
        actual=result['modes']['parent_v64'];expected=parent64['domains'][name]['metrics']['modes']['native_switch']
        replay={'ranks_equal':all(actual[k]==expected[k] for k in ('bytes','byte_t1','byte_t5')),'bpb_delta':actual['bpb']-expected['bpb']}
        replay['passed']=replay['ranks_equal'] and abs(replay['bpb_delta'])<1e-6
        if alpha==0.:assert result['modes']['selected']==actual
        out['domains'][name]={'windows':windows,'metrics':result,'parent_replay':replay}
        write_metrics(a.out,out,'domain:'+name)
    out['cache']=raw.stats()
    out['historical_replay_verified']=all(v['parent_replay']['passed'] for v in out['domains'].values())
    out['eligible_for_promotion']=False
    if not out['historical_replay_verified']:
        write_metrics(a.out,out,'replay_failed');raise AssertionError('v64 replay failed')
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
