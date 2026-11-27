"""v56 mixed concept-address lattice on immutable v53."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_contrastive'))
from pilot55 import *
from lattice import Lattice,Credit,fuse,NAMES
MASKS56={'class_only':(True,True,False,False),'word_only':(False,False,True,True),'full':(True,True,True,True)}

def prepare56(index,bank,model,rows,g,old,old_scale,lattice):
    pp=[];ff=[];vv=[]
    for row,parent in zip(rows,iter_parents(index,bank,model,rows,g,old,old_scale)):
        d,f,v=lattice.donors(row[1],parent);t=row[3];pp.append([parent[t]]+[q[t] for q in d]);ff.append(f);vv.append(v)
    return torch.tensor(pp,dtype=torch.float64),torch.tensor(np.stack(ff),dtype=torch.float64),torch.tensor(np.stack(vv))

def fit56(tr,va):
    models={};opts={};scales={};report={}
    for name,mask in MASKS56.items():
        model=Credit().double();opt=torch.optim.AdamW(model.parameters(),lr=.025,weight_decay=.01);rng=np.random.default_rng(5602);mask=torch.tensor(mask)
        p,f,v=tr;q,z,u=va;trace=[]
        for i in range(384):
            ix=rng.integers(len(p),size=32);opt.zero_grad();loss=-model(p[ix],f[ix],v[ix]&mask).log().mean();loss.backward();opt.step()
            if i==0 or (i+1)%64==0:trace.append([i+1,float(loss.detach())])
        with torch.no_grad():scores={str(s):float(-model(q,z,u&mask,s).log().mean()) for s in GRID54}
        scale=min(GRID54,key=lambda s:scores[str(s)]);models[name]=model;opts[name]=opt;scales[name]=scale
        report[name]={'selection_nll':scores,'scale':scale,'trace':trace,'parameters':model.gain.detach().tolist()}
    return models,opts,scales,report

def evaluate56(index,bank,model,rows,g,models,scales,old,old_scale,lattice):
    m=C.defaultdict(C.Counter);discovery=C.Counter()
    for i,(row,parent) in enumerate(zip(rows,iter_parents(index,bank,model,rows,g,old,old_scale))):
        h=row[1];t=row[3];d,f,v=lattice.donors(h,parent);modes={'parent':parent}
        for name,mask in MASKS56.items():
            w=apply(models[name],d,f,v,scales[name],mask);modes[name]=fuse(parent,d,w)
        # Most recent word fixed; replace earlier context with another book.
        for rotation in (1,2,3):
            # The v55 helper holds the last four fixed, so explicitly choose
            # a different prefix of the other book for the preceding 3.
            bids=sorted(set(r[0] for r in rows));other=bids[(bids.index(row[0])+rotation)%len(bids)]
            candidate=min((r for r in rows if r[0]==other),key=lambda r:abs(len(r[1])-len(h)))[1]
            bad=candidate[:-1]+h[-1:]
            bd,bf,bv=lattice.donors(bad,parent);w=apply(models['full'],bd,bf,bv,scales['full'],MASKS56['full'])
            modes['wrong_'+str(rotation)]=fuse(parent,bd,w)
        # The class-only donor can transfer mass to an unseen lexical item.
        orig=np.argsort(-parent[1:],kind='stable')[:32]+1
        discovery['n']+=1;discovery['outside_parent32']+=int(t>0 and t not in orig)
        for name,p in modes.items():
            rank=np.argsort(-p[1:],kind='stable')[:32]+1;z=m[name];z['n']+=1;z['nll']-=math.log(max(p[t],1e-300));z['t1']+=int(t==rank[0]);z['t5']+=int(t in rank[:5]);z['t32']+=int(t in rank)
            z['new32']+=int(t>0 and t not in orig and t in rank)
    return {'modes':{k:dict(z,nll=z['nll']/z['n'],top1=z['t1']/z['n'],top5=z['t5']/z['n'],top32=z['t32']/z['n']) for k,z in m.items()},'discovery':dict(discovery)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--checkpoint',type=pathlib.Path,required=True);ap.add_argument('--parent-metrics',type=pathlib.Path,required=True);ap.add_argument('--analogy',type=pathlib.Path,required=True);ap.add_argument('--analogy-metrics',type=pathlib.Path,required=True);a=ap.parse_args()
    torch.set_num_threads(4)
    groups={k:B.bookfiles(a.data/k) for k in ('stories','sentences','paragraphs')};split=lambda i:int(B.digest('NaNCM-structural-v1:'+i)[:8],16)%10
    books=[(i,p) for i,p in groups['stories'] if split(i)<7];tr=books[:96];cal=books[96:108];extra=books[108:396]
    assert len(tr)==96 and len(cal)==12 and len(extra)==288
    primary,secondary,sel,manifest=cohorts(books,tr,cal,extra)
    lex=set()
    for p in (a.data/'lex').rglob('*.txt'):
        if 'google' in p.name.lower() and not p.name.lower().startswith(('license','readme')):lex.update(B.words(p.read_text(errors='replace')))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr]+[B.load(p,6000) for _,p in extra]
    index=Index(docs,lex);del docs
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);pm=json.loads(a.parent_metrics.read_text())
    sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();assert sha=='f426598e0439ccefb3281edb951aead3e9ee4414aee211537ea783d44ee52812'
    assert pm['report_stage']=='final' and ck['words']==index.words and ck['split_integrity']==manifest and ck['updates']==768
    model=Field(index.code,index.prior)
    for k in ('code','members','cls','slot','logprior','classprior'):assert torch.equal(model.state_dict()[k],ck['models']['coupled'][k])
    model.load_state_dict(ck['models']['coupled']);model.eval();model.requires_grad_(False);g=float(ck['mixture']['coupled']);assert g==.25
    bank=Analogy(index.code,index.seq)
    out={'parent_sha256':sha,'parent_updates':768,'parent_mixture':g,'reference_rows':len(bank.rows),'reference_words':index.reference_words,'split_integrity':manifest,'domains':{}}
    ac=torch.load(a.analogy,map_location='cpu',weights_only=True);am=json.loads(a.analogy_metrics.read_text())
    assert am['report_stage']=='final' and am['parent_unchanged'] and ac['parent_sha256']==sha and ac['split_integrity']==manifest and ac['updates']==256
    old=OldCredit().double();old.load_state_dict(ac['gains']['full']);old.requires_grad_(False);old.eval();old_scale=float(ac['scales']['full']);assert old_scale==float(am['fit']['arms']['full']['scale'])
    assert np.array_equal(old.gain.detach().numpy(),np.asarray(am['fit']['arms']['full']['parameters']))
    out['analogy_sha256']=hashlib.sha256(a.analogy.read_bytes()).hexdigest();out['analogy_scale']=old_scale
    lattice=Lattice(index.seq,model.cls.cpu().numpy())
    out['lattice_rows']=[len(t[0]) for t in lattice.tables];write_metrics(a.out,out,'index')
    fitrows=examples(index,secondary,5403,96);valid=examples(index,sel,4804,64)
    td=prepare56(index,bank,model,fitrows,g,old,old_scale,lattice);vd=prepare56(index,bank,model,valid,g,old,old_scale,lattice)
    models,opts,scales,report=fit56(td,vd);out['fit']={'targets':len(fitrows),'selection_targets':len(valid),'arms':report}
    write_metrics(a.out,out,'fit')
    torch.save({'parent_sha256':sha,'parent_updates':768,'gains':{k:m.state_dict() for k,m in models.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'scales':scales,'split_integrity':manifest,'updates':384,'analogy_sha256':out['analogy_sha256'],'analogy_scale':old_scale},a.out.with_suffix('.pt'))
    out['selection']=evaluate56(index,bank,model,valid,g,models,scales,old,old_scale,lattice);write_metrics(a.out,out,'calibration')
    selected={};used=set();forbidden={i for i,p in books};hashes={q['sha256'] for v in manifest['manifest'].values() for q in v}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v4-window:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden or len(B.load(p,200))<112:continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest not in hashes
            hashes.add(digest);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        selected[name]=chosen
        out['domains'][name]={'ids':[i for i,p in chosen],'metrics':evaluate56(index,bank,model,examples(index,chosen,4806,96),g,models,scales,old,old_scale,lattice)}
        write_metrics(a.out,out,'domain:'+name)
    out['additional_dev']={}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v56-additional:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden or len(B.load(p,200))<112:continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in hashes:continue
            hashes.add(digest);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        selected['additional_'+name]=chosen
        out['additional_dev'][name]={'ids':[i for i,p in chosen],'metrics':evaluate56(index,bank,model,examples(index,chosen,5606,96),g,models,scales,old,old_scale,lattice)}
        write_metrics(a.out,out,'additional:'+name)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha and all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert hashlib.sha256(a.analogy.read_bytes()).hexdigest()==out['analogy_sha256'] and torch.equal(old.gain,ac['gains']['full']['gain'])
    out['replay']={}
    for name in groups:
        actual=out['domains'][name]['metrics']['modes']['parent'];expected=am['domains'][name]['metrics']['modes']['full']
        out['replay'][name]={'nll_delta':actual['nll']-expected['nll'],'ranks_equal':all(actual[k]==expected[k] for k in ('n','t1','t5','t32'))}
    out['v53_replay_verified']=all(v['ranks_equal'] and abs(v['nll_delta'])<1e-6 for v in out['replay'].values())
    out['parent_unchanged']=True
    out['overlap_audit']=[audit([(i,p,3000) for i,p in tr]+[(i,p,6000) for i,p in extra],{'primary':primary,'secondary':secondary,'selection':sel,**selected},B.load,n) for n in (16,32)]
    out['eligible_for_promotion']=False # diagnostic; strict historical parity and independent validation required
    write_metrics(a.out,out)
if __name__=='__main__':main()
