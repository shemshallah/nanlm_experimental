"""v66: matched exact search and error attribution, no training."""
import sys,pathlib,time,bisect
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_joint'))
from pilot63 import *
from transition import Transition
from word_bound import rank

@torch.no_grad()
def audit66(run,args,raw,surface,core,scale):
    index,bank,model,g,old,os,contrast,cs,outside,_,_=args
    runtime=core.runtime(scale);last=None;prev=None;previous=None
    m=C.Counter();by={};errors=[];start=time.monotonic()
    for source,row in zip(run,rows58([],6601,0,*args,raw=run)):
        bid,prefix,event,l53,l55,_=row;assert source[:3]==row[:3]
        if bid!=last:runtime.reset();prev=None
        else:assert prefix==previous
        f=features61(prefix,prev);w=runtime.weights(f);share=float(w[1:].sum())
        law=fuse(l53,l55,float(w[2]/share))
        q=likelihood(row,raw,surface,False)
        e=FE.Event(law,raw,prefix,surface,share)
        prob=float(w@q);assert math.isclose(prob,e.mass(event),rel_tol=1e-7,abs_tol=1e-30)
        m['bytes']+=len(event);m['bits']-=math.log2(prob)
        if len(event)>1 and not FE.IS_LETTER[event[-1]]:
            target=event[:-1].lower().decode('ascii');m['words']+=1
            bm=by.setdefault(bid,C.Counter());bm['words']+=1
            clock=time.monotonic();a=rank(e,budget=1200,tight=False);m['baseline_seconds']+=time.monotonic()-clock
            # Separate event caches for timing and verification.
            ee=FE.Event(law,raw,prefix,surface,share);clock=time.monotonic();b=rank(ee,budget=1200,tight=True);m['tight_seconds']+=time.monotonic()-clock
            m['baseline_nodes']+=a[2];m['tight_nodes']+=b[2]
            for label,res in [('baseline',a),('tight',b)]:
                words,cert,seen=res;m[label+'_certified']+=int(cert);m[label+'_cert128']+=int(cert and seen<=128)
                m[label+'_t1']+=int(cert and words and words[0]==target);m[label+'_t5']+=int(cert and target in words)
            if a[1] and b[1]:
                assert a[0]==b[0],(bid,len(prefix),a,b);m['exact_rank_agreements']+=1
            elif a[1]:m['baseline_only_certified']+=1
            elif b[1]:m['tight_only_certified']+=1
            j=bisect.bisect_left(outside.words,target);known=j<len(outside.words) and outside.words[j]==target
            p=w[1]*l53.p+w[2]*l55.p
            top=np.argsort(-p,kind='stable')[:32]
            m['known']+=int(known);m['oov']+=int(not known)
            lexrank=int((p>p[j]).sum()+1) if known else None
            m['lexical_t1']+=int(known and j==top[0]);m['lexical_t5']+=int(known and j in top[:5]);m['lexical_t32']+=int(known and j in top)
            if b[1]:
                pred=b[0][0];hit=pred==target;bm['t1']+=int(hit);bm['certified']+=1
                if not hit:
                    m['certified_errors']+=1
                    if not known:m['error_oov']+=1
                    elif j not in top:m['error_known_outside_lex32']+=1
                    elif j not in top[:5]:m['error_lex6_32']+=1
                    else:m['error_lex_top5']+=1
                    m['error_lex_top1']+=int(known and j==top[0])
                    rawwin=FE.Event(law,raw,prefix,surface,0.).complete(pred.encode('ascii'))
                    score=b[0] and ee.complete(pred.encode('ascii'));target_score=ee.complete(target.encode('ascii'))
                    m['error_raw_majority_winner']+=int(w[0]*rawwin>score*.5)
                    if len(errors)<24:errors.append({'id':bid,'offset':len(prefix),'target':target,'winner':pred,'target_known':bool(known),'lexical_rank':lexrank,'target_score':target_score,'winner_score':score,'winner_raw_fraction':float(w[0]*rawwin/score)})
        observed=runtime.observe(f,q);assert math.isclose(observed,prob,rel_tol=1e-12)
        prev=q;previous=prefix+event;last=bid
    assert sum(m[k] for k in ('error_oov','error_known_outside_lex32','error_lex6_32','error_lex_top5'))==m['certified_errors']
    return dict(m,bpb=m['bits']/m['bytes'],per_book={k:dict(v) for k,v in by.items()},examples=errors,seconds=time.monotonic()-start)

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
    for name in groups:
        run,windows=contiguous(selected[name],256)
        result=audit66(run,args,raw,surface,core,float(tc['scale']))
        expected=tm['domains'][name]['metrics']['modes']['learned']
        assert result['bytes']==expected['bytes']==sum(q['bytes'] for q in windows)
        assert result['words']==expected['words'] and result['baseline_t1']==expected['word_t1'] and result['baseline_t5']==expected['word_t5'] and result['baseline_certified']==expected['certified']
        assert abs(result['bpb']-expected['bpb'])<1e-8
        out['domains'][name]={'windows':windows,'metrics':result};write_metrics(a.out,out,'domain:'+name)
    assert all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,tc['transition'][k]) for k,v in core.state_dict().items())
    assert hashlib.sha256(a.transition.read_bytes()).hexdigest()==thash
    out['eligible_for_promotion']=False
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
