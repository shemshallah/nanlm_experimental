"""v58: v55 categorical-to-open law; optional actual v37 recurrent integration.
Separate finite-order folded marginal from history-dependent native literal donor.
"""
import sys,pathlib,re,heapq,json,hashlib,math
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(root/'tests/nancm_contrastive'),str(root/'tests/nancm_folded'),str(root/'nancm_structural')]
from pilot55 import *
from scout import rails
from open_word import Spelling,Outside,Lift,fuse
import folded_event as FE
import raw_byte_law as RAW

class NativeEvent(FE.Event):
    def suffix(self,p):return p
    def states(self,p):raise RuntimeError('native donor is not finite-order; folded marginal not claimed')

def events(files,seed,count):
    rng=np.random.default_rng(seed);out=[]
    for bid,p in files:
        raw=p.read_bytes()[:8192];words=B.load(p,1200);starts=[];start=0
        for i,b in enumerate(raw):
            if not FE.IS_LETTER[b] or i-start==63:
                # Limit to the screened word prefix. Ignore starts inside a capped word.
                if start>=128 and not FE.IS_LETTER[raw[start-1]]:
                    prefix=[x.lower().decode('ascii') for x in re.findall(rb'[A-Za-z]+',raw[:start])]
                    letters=re.findall(rb'[A-Za-z]+',raw[start:i+1]);observed=prefix+[x.lower().decode('ascii') for x in letters]
                    if len(observed)<=len(words) and observed==words[:len(observed)]:starts.append((start,i+1,prefix))
                start=i+1
        assert starts,bid
        for j in sorted(rng.choice(len(starts),min(count,len(starts)),replace=False)):
            a,b,h=starts[j];out.append((bid,raw[:a],raw[a:b],h))
    return out

def digest_field(field):
    arrays={k:hashlib.sha256(v.tobytes()+field.m[k].tobytes()+field.v[k].tobytes()).hexdigest() for k,v in field.p.items()}
    return arrays,field.steps.copy(),field.step,[hashlib.sha256(getattr(field,k).tobytes()).hexdigest() for k in ('lex','lm','lv','lt','proj')],json.dumps(field.rng.bit_generator.state,sort_keys=True)

def rows58(files,seed,count,index,bank,model,g,old,os,contrast,cs,outside,field=None,runtime=None):
    raw=events(files,seed,count);rows=[]
    for bid,prefix,event,h in raw:
        ids=index.encode(h[-512:]);rows.append((bid,ids,rails(index.code,ids),0))
    for (bid,prefix,event,h),row,p53 in zip(raw,rows,iter_parents(index,bank,model,rows,g,old,os)):
        d,f,v=donors(row[1],p53,index.prior,model.cls.cpu().numpy());p55=fused(p53,d,apply(contrast,d,f,v,cs,MASKS55['full']))
        word53=Lift(outside,p53);word55=Lift(outside,p55);parent=None
        if field is not None:
            import hybrid20 as V
            ref,small,large,cal=runtime;state=field.session()
            for w in h[-128:]:state.observe(w)
            with torch.no_grad():
                orig,_=V.branches(ref,small,large,state,cal);base,hidden=field.field_law(state,orig[1].law);parent,_=field.combine(base,hidden,state.attractor.query())
        yield bid,prefix,event,word53,word55,parent

W=(0.,.125,.25,.5,.75)
G=(0.,.125,.25,.5,.75,1.)
def calibration(rows,raw,surface,native):
    typ=NativeEvent if native else FE.Event;scores=C.Counter();n=0
    for bid,root,event,l53,l55,parent in rows:
        e=typ(l55,raw,root,surface,0.);r,s55=e.literal(event)
        s0=typ(parent if native else l53,raw,root,surface,0.).literal(event)[1]
        for w in W:
            for g in G:
                q=(1-w)*r+w*((1-g)*s0+g*s55);scores[w,g]-=math.log2(max(q,1e-300))
        n+=len(event)
    best=min(scores,key=scores.get);base=min((k for k in scores if k[1]==0),key=scores.get)
    return best,base,{'bytes':n,'selected':best,'parent_selected':base,'grid':{str(k):v/n for k,v in scores.items()}}

def evaluate(rows,raw,surface,best,base,native):
    typ=NativeEvent if native else FE.Event;m=C.defaultdict(C.Counter);err=0.;marg=0.;checks=0
    for bid,root,event,l53,l55,parent in rows:
        p=parent if native else l53;w,g=best
        for name,law,weight in [('raw',p,0.),('parent',p,base[0]),('parent_matched_weight',p,w),('v55',l55,w),('hybrid',fuse(p,l55,g),w)]:
            e=typ(law,raw,root,surface,weight);z=m[name];prefix=b'';product=1.
            for b in event:
                q=e.next(prefix);rank=np.argsort(-q,kind='stable')[:5];z['bytes']+=1;z['byte_t1']+=int(b==rank[0]);z['byte_t5']+=int(b in rank);product*=q[b];prefix+=bytes([b])
            err=max(err,abs(product-e.mass(event)));assert e.done(prefix) and math.isclose(product,e.mass(event),rel_tol=1e-7,abs_tol=1e-30)
            z['bits']-=math.log2(max(product,1e-300))
            # Exact event-relative token marginal; avoid exponential native folded enumeration.
            if not native:
                prefix=b'';i=0
                while i<len(event):
                    t=e.tokens(prefix);size=2 if event[i] in b' etaoinr' and not e.done(prefix+event[i:i+1]) else 1
                    target=event[i:i+size];rank=heapq.nlargest(5,t,key=t.get);z['tokens']+=1;z['token_t1']+=int(target==rank[0]);z['token_t5']+=int(target in rank)
                    mass=np.zeros(256)
                    for k,v in t.items():mass[k[0]]+=v
                    marg=max(marg,float(abs(mass-e.next(prefix)).max()));checks+=1;prefix+=target;i+=size
                if len(event)>1 and not FE.IS_LETTER[event[-1]]:
                    target=event[:-1].lower();rank,cert,_=e.rank(budget=1200);z['words']+=1;z['word_t1']+=int(target.decode('ascii')==rank[0]);z['word_t5']+=int(target.decode('ascii') in rank);z['certified']+=int(cert);z['word_bits']-=math.log2(max(e.complete(target),1e-300))
    out={}
    for k,z in m.items():
        out[k]=dict(z,bpb=z['bits']/z['bytes'],byte_top1=z['byte_t1']/z['bytes'],byte_top5=z['byte_t5']/z['bytes'])
        if z['tokens']:out[k].update(token_top1=z['token_t1']/z['tokens'],token_top5=z['token_t5']/z['tokens'])
        if z['words']:out[k].update(word_top1=z['word_t1']/z['words'],word_top5=z['word_t5']/z['words'],folded_word_nll=z['word_bits']/z['words']*math.log(2))
    assert marg<1e-8
    return {'modes':out,'literal_max_error':err,'token_marginal_max_error':marg if checks else None,'marginal_checks':checks}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
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
    out={'native':a.native,'parent_artifact_hashes':hashes,'vocabulary':len(outside.words),'outside_spelling_mass':outside.z,'domains':{},'event_scope':'uniform sampled complete byte events in v57 screened prefix; not continuous stream BPB','native_folded_marginal_claim':False}
    write_metrics(a.out,out,'index')
    args=(index,bank,model,g,old,os,contrast,cs,outside,field,runtime);count=4 if a.native else 12
    best,base,report=calibration(rows58(selected['calibration'],5801,count,*args),raw,surface,a.native);out['calibration']=report
    write_metrics(a.out,out,'calibration')
    for name in groups:
        out['domains'][name]=evaluate(rows58(selected[name],5802,count,*args),raw,surface,best,base,a.native);write_metrics(a.out,out,'domain:'+name)
    assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    if a.native:assert initial==digest_field(field);out['recurrent_checkpoint_sha256']=field.v37_sha;out['recurrent_unchanged']=True
    out['parent_artifacts_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
