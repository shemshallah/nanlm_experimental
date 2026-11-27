"""v63 shared geometry-conditioned temporal allocation and exact event/word credit."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tests/nancm_geometric_credit'),str(root/'tests/nancm_transition')]
from pilot62 import *
from pilot61 import features61
from joint import Joint

def segment(geo,gate,rows,f,state=None,prev=None,word_weight=.25):
    x,ids,a,t,_=batch(rows)
    pa,pb=laws(geo.log_probs(x,ids),*a[:6]);e,s,r=a[6:]
    hidden=geo.hidden(x,ids)
    return gate.trajectory(pa,pb,r,e,s,torch.as_tensor(f,dtype=torch.float64),hidden,t,state,prev,word_weight=word_weight)

def fit63(model,ck,parent_gate,tc,train,features):
    rng=np.random.default_rng(6301);keys=list(train);segments=[]
    for _ in range(128):
        key=keys[int(rng.integers(len(keys)))];d=train[key];start=int(rng.integers(max(1,len(d)-15)));segments.append((key,start,min(start+16,len(d))))
    result={};saved={};report={}
    def steps(o):return [int(v['step']) for v in o.state.values() if 'step' in v]
    for name,joint in [('allocation_only',False),('joint',True)]:
        geo=copy.deepcopy(model);geo.requires_grad_(joint);gate=copy.deepcopy(parent_gate)
        go=torch.optim.AdamW(geo.parameters());go.load_state_dict(copy.deepcopy(ck['optimizers']['coupled']))
        to=torch.optim.AdamW(gate.core.parameters());to.load_state_dict(copy.deepcopy(tc['optimizer']))
        so=torch.optim.AdamW([gate.signed],lr=.001,weight_decay=.01)
        for group in go.param_groups:group['lr']=.0003
        for group in to.param_groups:group['lr']=.003
        assert min(steps(go))==max(steps(go))==768 and min(steps(to))==max(steps(to))==96
        norms={};trace=[]
        for n,(key,a,b) in enumerate(segments,1):
            d=train[key];f=features[key];state=prev=None
            # Recompute warm-up with current parameters, detach only at the segment boundary.
            with torch.no_grad():
                for j in range(0,a,16):
                    k=min(j+16,a);_,state,prev,*_=segment(geo,gate,d[j:k],f[j:k],state,prev,0.)
            go.zero_grad();to.zero_grad();so.zero_grad()
            loss,_,_,ev,wd,count=segment(geo,gate,d[a:b],f[a:b],state,prev)
            assert torch.isfinite(loss);loss.backward()
            for prefix,m in [('geometry',geo),('gate',gate)]:
                for k,v in m.named_parameters():
                    if v.grad is not None:norms[prefix+'.'+k]=max(norms.get(prefix+'.'+k,0.),float(v.grad.norm()))
            torch.nn.utils.clip_grad_norm_(list(geo.parameters())+list(gate.parameters()),1.)
            if joint:go.step()
            to.step();so.step()
            if n==1 or n%16==0:trace.append([n,float(loss.detach()),float(ev.detach()),float(wd.detach()),count])
        assert min(steps(go))==max(steps(go))==(896 if joint else 768)
        assert min(steps(to))==max(steps(to))==224 and steps(so)==[128]
        assert norms['gate.signed']>0 and norms['gate.core.theta']>0
        if joint:assert all(norms['geometry.'+k]>0 for k,v in geo.named_parameters())
        else:assert all(torch.equal(v,model.state_dict()[k]) for k,v in geo.state_dict().items())
        result[name]=(geo.eval(),gate.eval())
        saved[name]={'geometry':geo.state_dict(),'transition':gate.state_dict(),'geometry_optimizer':go.state_dict(),'transition_optimizer':to.state_dict(),'signed_optimizer':so.state_dict()}
        report[name]={'geometry_step':896 if joint else 768,'transition_step':224,'signed_step':128,'gradient_groups':norms,'trace':trace}
    return result,saved,report

@torch.no_grad()
def evaluate63(run,args,raw,surface,systems):
    index,bank,_,g,old,os,contrast,cs,outside,_,_=args
    out={}
    for name,(geo,gate) in systems.items():
        ar=(index,bank,geo,g,old,os,contrast,cs,outside,None,None)
        z=C.Counter();last=None;state=prev=None;previous=None;maxerr=0.;marg=0.;checks=0;books={}
        for input_row,row in zip(run,rows58([],6302,0,*ar,raw=run)):
            bid,root,event,l53,l55,_=row;assert (bid,root,event)==input_row[:3]
            if bid!=last:state=prev=None
            else:assert root==previous
            ids=index.encode(input_row[3][-512:]);x,i,_=feature([(bid,ids,rails(index.code,ids),0)])
            hidden=geo.hidden(x,i)[0];f=torch.as_tensor(features61(root,None)[:5],dtype=torch.float64)
            w=gate.predict(f,hidden,state,prev);wn=w.numpy();assert abs(wn.sum()-1)<1e-12 and wn.min()>0
            share=float(wn[1]+wn[2]);e=FE.Event(fuse(l53,l55,float(wn[2]/share)),raw,root,surface,share)
            prefix=b'';prod=1.
            for byte in event:
                q=e.next(prefix);rank=np.argsort(-q,kind='stable')[:5];prod*=q[byte];prefix+=bytes([byte]);z['bytes']+=1;z['byte_t1']+=int(byte==rank[0]);z['byte_t5']+=int(byte in rank)
            q=torch.as_tensor(likelihood(row,raw,surface,False),dtype=torch.float64)
            state,p=gate.advance(w,q);prev=q
            maxerr=max(maxerr,abs(prod-float(p)));assert e.done(prefix) and math.isclose(prod,float(p),rel_tol=1e-7,abs_tol=1e-30)
            bits=-math.log2(prod);z['bits']+=bits;z['events']+=1
            bm=books.setdefault(bid,C.Counter());bm['bits']+=bits;bm['bytes']+=len(event)
            prefix=b'';i=0
            while i<len(event):
                t=e.tokens(prefix);size=2 if event[i] in b' etaoinr' and not e.done(prefix+event[i:i+1]) else 1
                target=event[i:i+size];rank=heapq.nlargest(5,t,key=t.get);z['tokens']+=1;z['token_t1']+=int(target==rank[0]);z['token_t5']+=int(target in rank)
                mass=np.zeros(256)
                for k,v in t.items():mass[k[0]]+=v
                marg=max(marg,float(abs(mass-e.next(prefix)).max()));checks+=1;prefix+=target;i+=size
            if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                rank,cert,_=e.rank(budget=1200);target=event[:-1].lower().decode('ascii');z['words']+=1;z['certified']+=int(cert);z['word_t1']+=int(cert and target==rank[0]);z['word_t5']+=int(cert and target in rank)
            last=bid;previous=root+event
        assert marg<1e-8
        out[name]=dict(z,bpb=z['bits']/z['bytes'],byte_top1=z['byte_t1']/z['bytes'],byte_top5=z['byte_t5']/z['bytes'],token_top1=z['token_t1']/z['tokens'],token_top5=z['token_t5']/z['tokens'],certified_word_top1=z['word_t1']/max(1,z['words']),certified_word_top5=z['word_t5']/max(1,z['words']),certification_rate=z['certified']/max(1,z['words']),literal_max_error=maxerr,token_marginal_max_error=marg,marginal_checks=checks,per_book={bid:dict(v,bpb=v['bits']/v['bytes']) for bid,v in books.items()})
    return out

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
    assert not a.native,'v63 unifies v50/v61 components, not v37 native weights'
    tc=torch.load(a.transition,map_location='cpu',weights_only=True);tm=json.loads(a.transition_metrics.read_text())
    assert tm['report_stage']=='final' and tc['parent_artifact_hashes']==hashes and tc['updates']==96 and tc['scale']==1
    assert tc['initial']==tuple(tm['fit']['initial']) or json.loads(json.dumps(tc['initial']))==tm['fit']['initial']
    parent_gate=Joint(*tc['initial'],d=model.code.shape[1]);parent_gate.core.load_state_dict(tc['transition']);assert torch.count_nonzero(parent_gate.signed)==0
    thash=hashlib.sha256(a.transition.read_bytes()).hexdigest()
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime)
    assert not {i for i,p in primary+secondary}&{i for group in selected.values() for i,p in group}
    fitrun,fitmanifest=contiguous(primary+secondary,128)
    train,audit=prepare62(fitrun,index,bank,model,g,old,os,contrast,cs,outside,raw,surface)
    f=C.defaultdict(list)
    for bid,prefix,event,h in fitrun:f[bid].append(features61(prefix,None)[:5])
    assert all(len(f[k])==len(v) for k,v in train.items())
    out['fit_audit']={'windows':fitmanifest,**audit};write_metrics(a.out,out,'factors')
    systems,saved,report=fit63(model,ck,parent_gate,tc,train,f);del train
    out['fit']=report;out['transition_parent_sha256']=thash
    torch.save({'systems':saved,'parent_artifact_hashes':hashes,'transition_parent_sha256':thash,'updates_added':128,'word_weight':.25},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'fit')
    systems={'parent_v61':(model,parent_gate),**systems}
    for name in ('calibration',*groups):
        run,manifest63=contiguous(selected[name],256);result=evaluate63(run,args,raw,surface,systems)
        assert all(v['bytes']==sum(q['bytes'] for q in manifest63) for v in result.values())
        out['domains'][name]={'windows':manifest63,'metrics':result}
        if name!='calibration':
            expected=tm['domains'][name]['metrics']['modes']['learned'];actual=result['parent_v61']
            out['domains'][name]['parent_replay']={'bpb_delta':actual['bpb']-expected['bpb'],'ranks_equal':all(actual[k]==expected[k] for k in ('bytes','byte_t1','byte_t5','word_t1','word_t5','certified'))}
        write_metrics(a.out,out,'domain:'+name)
    assert all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert all(torch.equal(v,tc['transition'][k]) for k,v in parent_gate.core.state_dict().items()) and torch.count_nonzero(parent_gate.signed)==0
    assert hashlib.sha256(a.transition.read_bytes()).hexdigest()==thash
    out['eligible_for_promotion']=False
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
