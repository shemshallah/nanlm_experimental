"""v67: same three native rank audit roots as v64, no training."""
import sys,pathlib,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_native_stream'))
from pilot64 import *
from tight_native import rank_native

def audit67(rows,raw,surface,adaptive,expected):
    switch=Switch(*adaptive);last=None;previous=None
    for row in rows:
        bid,prefix,event,l53,l55,parent=row
        if bid!=last:switch.reset()
        else:assert prefix==previous
        w=switch.weights();share=float(w[1]+w[2]);law=fuse(parent,l55,float(w[2]/share))
        q=likelihood(row,raw,surface,True)
        if len(event)>1 and not FE.IS_LETTER[event[-1]]:
            assert bid==expected['id'] and len(prefix)==expected['offset']
            target=event[:-1].lower().decode('ascii');assert target==expected['rank']['target']
            modes={};certified=[]
            for budget in (64,256):
                for name,fn in [('baseline',folded_rank),('tight',rank_native)]:
                    before=raw.stats();clock=time.monotonic()
                    z=fn(law,raw,prefix,share,budget=budget);after=raw.stats()
                    assert z['queries']<=budget
                    z['seconds']=time.monotonic()-clock;z['cache_delta']={k:after[k]-before[k] for k in ('hits','misses','evictions')}
                    if z['certified']:certified.append(z['ranked'])
                    modes[name+'_'+str(budget)]=z
            old=modes['baseline_64'];historical=expected['rank']
            replay=old['certified']==historical['certified'] and old['queries']==historical['queries'] and [x[0] for x in old['ranked']]==[x[0] for x in historical['ranked']]
            replay=replay and max((abs(x[1]-y[1]) for x,y in zip(old['ranked'],historical['ranked'])),default=0.)<1e-6
            for a in certified[1:]:
                assert [x[0] for x in a]==[x[0] for x in certified[0]] and max(abs(x[1]-y[1]) for x,y in zip(a,certified[0]))<1e-12
            return {'id':bid,'offset':len(prefix),'target':target,'historical_rank_replay':bool(replay),'modes':modes}
        switch.observe(q);last=bid;previous=prefix+event
    raise AssertionError('expected rank audit event missing')

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
    out={'native':True,'parent_artifact_hashes':hashes,'recurrent_checkpoint_sha256':field.v37_sha,'scope':'same three historical v64 native word audit events; no representative accuracy claim','domains':{}}
    write_metrics(a.out,out,'index')
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    adaptive=parent64['calibration']['adaptive']
    for name in groups:
        run,windows=contiguous(selected[name][:2],128)
        assert windows==parent64['domains'][name]['windows']
        expected=parent64['domains'][name]['metrics']['audit']
        result=audit67(rows58([],6402,0,*args,raw=run),raw,surface,adaptive,expected)
        out['domains'][name]=result;write_metrics(a.out,out,'domain:'+name)
    out['cache']=raw.stats()
    out['historical_replay_verified']=all(v['historical_rank_replay'] for v in out['domains'].values())
    if not out['historical_replay_verified']:
        write_metrics(a.out,out,'replay_failed');raise AssertionError('native rank replay failed')
    out['eligible_for_promotion']=False
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
