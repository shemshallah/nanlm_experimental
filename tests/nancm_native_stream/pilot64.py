"""Native recurrent replay and contiguous transfer of persistent allocation."""
import sys,pathlib,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tests/nancm_open_bridge'),str(root/'tests/nancm_switching'),str(root/'tests/nancm_transition')]
from pilot59 import *
from pilot60 import calibrate60,likelihood,Switch
from pilot61 import features61
from transition import Transition
from native_prefix import PrefixMemo,folded_rank

def rank_audit(e,root,event,word,raw,share,budget):
    # Audit a single complete event, not a representative word-accuracy sample.
    rank=folded_rank(word,raw,root,share,budget=budget,cap=e.cap)
    target=event[:-1].lower().decode('ascii')
    words=[w for w,_ in rank['ranked']]
    rank.update(target=target,certified_hit1=bool(rank['certified'] and words and words[0]==target),certified_hit5=bool(rank['certified'] and target in words))
    # Native token marginal requires only literal (not case-merged) conditionals.
    q=e.next(b'');t=e.tokens(b'');mass=np.zeros(256)
    for k,v in t.items():mass[k[0]]+=v
    err=float(abs(mass-q).max());assert err<1e-8
    # New memo and event objects for decoder: no encoded-target-dependent cache.
    code=RAW.encode(root,event,lambda h:e.next(h[len(root):]))
    decoder=NativeEvent(word,PrefixMemo(raw.donor,256),root,e.surface,share)
    assert RAW.decode(root,code,len(event),lambda h:decoder.next(h[len(root):]))==event
    return dict(rank=rank,token_marginal_error=err,codec={'bytes':len(event),'bits':code[1],'roundtrip':True})

def evaluate64(rows,raw,surface,adaptive,static,transfer,scale,budget):
    fs={'native_switch':Switch(*adaptive),'native_static':Switch(*static),'v59_static':Switch((.25,.375,.375),1.)}
    learned=transfer.runtime(scale);stats=C.defaultdict(C.Counter);books={};last=None;prev=None;previous=None;err=0.;audit={};parity=0;start=time.monotonic();before=raw.stats()
    for row in rows:
        bid,prefix,event,l53,l55,parent=row
        if bid!=last:
            for f in fs.values():f.reset()
            learned.reset();prev=None
        else:assert prefix==previous
        f=features61(prefix,prev);q=likelihood(row,raw,surface,True)
        modes={'raw':np.array([1.,0.,0.]),**{k:m.weights() for k,m in fs.items()},'transferred_v61':learned.weights(f)}
        bm=books.setdefault(bid,C.defaultdict(C.Counter))
        for name,w in modes.items():
            share=float(w[1]+w[2]);law=fuse(parent,l55,float(w[2]/share)) if share else parent
            e=NativeEvent(law,raw,prefix,surface,share);z=stats[name];p=b'';prod=1.
            for b in event:
                d=e.next(p);rank=np.argsort(-d,kind='stable')[:5];prod*=d[b];p+=bytes([b]);z['bytes']+=1;z['byte_t1']+=int(rank[0]==b);z['byte_t5']+=int(b in rank)
            expected=float(w@q);err=max(err,abs(prod-expected));assert e.done(p) and math.isclose(prod,expected,rel_tol=1e-7,abs_tol=1e-30)
            bits=-math.log2(prod);z['bits']+=bits;z['events']+=1
            bm[name]['bytes']+=len(event);bm[name]['bits']+=bits
            if name=='native_switch' and not audit and len(event)>1 and not FE.IS_LETTER[event[-1]]:
                audit=rank_audit(e,prefix,event,law,raw,share,budget);audit.update(id=bid,offset=len(prefix))
        for m in fs.values():m.observe(q)
        learned.observe(f,q)
        if parity<4:
            expected=raw.donor(prefix)
            assert np.array_equal(raw(prefix),expected)
            parity+=1
        prev=q;previous=prefix+event;last=bid
    after=raw.stats()
    return {'modes':{k:dict(v,bpb=v['bits']/v['bytes'],byte_top1=v['byte_t1']/v['bytes'],byte_top5=v['byte_t5']/v['bytes']) for k,v in stats.items()},'per_book':{b:{k:dict(v,bpb=v['bits']/v['bytes']) for k,v in m.items()} for b,m in books.items()},'event_product_error':err,'audit':audit,'cache_delta':{k:after[k]-before[k] for k in ('hits','misses','evictions')},'uncached_exact_checks':parity,'seconds':time.monotonic()-start}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen','native-metrics','transition','transition-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
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
    raw=PrefixMemo(raw,2048)
    out={'native':True,'parent_artifact_hashes':hashes,'transition_sha256':hashlib.sha256(a.transition.read_bytes()).hexdigest(),'recurrent_checkpoint_sha256':field.v37_sha,'event_scope':'contiguous complete event windows; 2 screened DEV books/domain; no full native folded accuracy claim','domains':{},'replay':{}}
    write_metrics(a.out,out,'index')
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    best=previous['calibration']['selected'];base=previous['calibration']['parent_selected']
    # Replay exactly the historical v59 sampled native windows, without recalibration.
    for name in groups:
        clock=time.monotonic();before=raw.stats()
        z=evaluate(rows58(selected[name],5802,4,*args),raw,surface,best,base,True)
        expected=previous['domains'][name]['modes'];actual=z['modes']
        delta={k:actual[k]['bpb']-expected[k]['bpb'] for k in expected}
        ranks=all(actual[k][m]==expected[k][m] for k in expected for m in ('bytes','byte_t1','byte_t5'))
        out['replay'][name]={'bpb_delta':delta,'ranks_equal':ranks,'seconds':time.monotonic()-clock,'cache_delta':{k:raw.stats()[k]-before[k] for k in ('hits','misses','evictions')}}
        write_metrics(a.out,out,'replay:'+name)
    out['historical_replay_verified']=all(v['ranks_equal'] and max(map(abs,v['bpb_delta'].values()))<1e-6 for v in out['replay'].values())
    # Independent native calibration, same priors/hazards as v60.
    run,windows=contiguous(selected['calibration'][:3],128)
    adaptive,static,report=calibrate60(rows58([],6401,0,*args,raw=run),raw,surface,True)
    out['calibration']={'windows':windows,**report};write_metrics(a.out,out,'calibration')
    for name in groups:
        run,windows=contiguous(selected[name][:2],128)
        result=evaluate64(rows58([],6402,0,*args,raw=run),raw,surface,adaptive,static,transfer,float(tc['scale']),a.rank_budget)
        assert all(v['bytes']==sum(q['bytes'] for q in windows) for v in result['modes'].values())
        out['domains'][name]={'windows':windows,'metrics':result}
        write_metrics(a.out,out,'domain:'+name)
    out['cache']=raw.stats()
    out['eligible_for_promotion']=False # native diagnostic size, not word-accuracy certification
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
