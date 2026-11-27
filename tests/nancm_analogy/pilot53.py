"""v53 new causal analogical evidence, frozen strongest v50 component."""
import pathlib,sys,hashlib,math,json
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_hierarchical'))
from pilot50 import Index,examples,cohorts,write_metrics,B,C,np,torch,argparse,feature,wrong_ids
from hierarchical_field import Field
from analogy import Analogy,Credit,mix

MASKS={'reference':(True,True,False),'local':(False,False,True),'full':(True,True,True)}
GRID=(0.,.125,.25,.5,.75,1.)

def prepare(index,bank,model,rows,g):
    x,ids,y=feature(rows)
    with torch.no_grad():dense=torch.cat([model.target_logprob(x[i:i+32],ids[i:i+32],y[i:i+32]).double().exp() for i in range(0,len(y),32)]).numpy()
    pp=[];ff=[];vv=[]
    for row,p in zip(rows,dense):
        t=row[3];parent=(1-g)*index.countlaw(row[1])[t]+g*p;rr=bank.experts(row[1])
        pp.append([parent]+[l.get(t,0.) for l,_ in rr]);ff.append(np.stack([f for _,f in rr]));vv.append([bool(l) for l,_ in rr])
    return torch.tensor(pp,dtype=torch.float64),torch.tensor(np.stack(ff),dtype=torch.float64),torch.tensor(vv)

def fit(tr,va):
    models={};opts={};scales={};report={}
    for name,mask in MASKS.items():
        m=Credit().double();opt=torch.optim.AdamW(m.parameters(),lr=.025,weight_decay=.01);rng=np.random.default_rng(5302);mask=torch.tensor(mask)
        p,f,v=tr;q,z,u=va;trace=[]
        for i in range(256):
            ix=rng.integers(len(p),size=32);opt.zero_grad()
            loss=-m(p[ix],f[ix],v[ix]&mask).log().mean();loss.backward();opt.step()
            if i==0 or (i+1)%64==0:trace.append([i+1,float(loss.detach())])
        with torch.no_grad():scores={str(s):float(-m(q,z,u&mask,s).log().mean()) for s in GRID}
        scale=min(GRID,key=lambda s:scores[str(s)]);models[name]=m;opts[name]=opt;scales[name]=scale
        report[name]={'selection_nll':scores,'scale':scale,'trace':trace,'parameters':m.gain.detach().tolist()}
    return models,opts,scales,report

def evaluate(index,bank,model,rows,g,models,scales):
    x,ids,y=feature(rows);wrong=wrong_ids(rows);m=C.defaultdict(C.Counter);oracle=C.Counter()
    for a in range(0,len(rows),12):
        b=min(a+12,len(rows))
        with torch.no_grad():dense=model.log_probs(x[a:b],ids[a:b]).double().exp().numpy();dense/=dense.sum(1,keepdims=True)
        for j,row in enumerate(rows[a:b]):
            i=a+j;hist=row[1];t=row[3];q=index.countlaw(hist);parent=(1-g)*q+g*dense[j];rr=bank.experts(hist)
            badhist=rows[wrong[i]][1][-4:-1]+[hist[-1]]
            bad=(*bank.experts(badhist)[:2],rr[2])
            modes={'count':(q,0.),'parent':(parent,0.)}
            for name,mask in MASKS.items():
                f=torch.tensor(np.stack([f for _,f in rr])[None],dtype=torch.float64);valid=torch.tensor([[bool(l) for l,_ in rr]])&torch.tensor(mask)
                with torch.no_grad():w=models[name].weights(f,valid,scales[name])[0].numpy()
                modes[name]=(mix(parent,[l for l,_ in rr],w),1-w[0])
                if name=='full':
                    f2=torch.tensor(np.stack([f for _,f in bad])[None],dtype=torch.float64);v2=torch.tensor([[bool(l) for l,_ in bad]])
                    with torch.no_grad():w2=models[name].weights(f2,v2,scales[name])[0].numpy()
                    modes['wrong_reference']=(mix(parent,[l for l,_ in bad],w2),1-w2[0])
                    with torch.no_grad():wf=models[name].weights(f,valid,1.)[0].numpy();wb=models[name].weights(f2,v2,1.)[0].numpy()
                    modes['forced']=(mix(parent,[l for l,_ in rr],wf),1-wf[0]);modes['wrong_forced']=(mix(parent,[l for l,_ in bad],wb),1-wb[0])
            old=np.argsort(-parent[1:],kind='stable')[:32]+1
            oracle['n']+=1;oracle['parent32']+=int(t in old)
            for name,(law,_) in zip(('exact_anchor','role_anchor','local'),rr):oracle[name]+=int(t!=0 and t in law)
            oracle['any_new']+=int(t!=0 and t not in old and any(t in l for l,_ in rr))
            oracle['wrong_reference']+=int(t!=0 and any(t in l for l,_ in bad[:2]))
            for name,(p,wt) in modes.items():
                assert abs(p.sum()-1)<1e-7 and np.min(p)>=0
                rank=np.argsort(-p[1:],kind='stable')[:32]+1;z=m[name]
                z['n']+=1;z['nll']-=math.log(max(p[t],1e-300));z['t1']+=int(t==rank[0]);z['t5']+=int(t in rank[:5]);z['t32']+=int(t in rank);z['weight']+=wt
    return {'modes':{k:dict(z,nll=z['nll']/z['n'],top1=z['t1']/z['n'],top5=z['t5']/z['n'],top32=z['t32']/z['n'],mean_analogy_weight=z['weight']/z['n']) for k,z in m.items()},'discovery':dict(oracle)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);ap.add_argument('--checkpoint',type=pathlib.Path,required=True);ap.add_argument('--parent-metrics',type=pathlib.Path,required=True);a=ap.parse_args()
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
    write_metrics(a.out,out,'index')
    fitrows=examples(index,primary,5303,96);valid=examples(index,sel,4804,64)
    td=prepare(index,bank,model,fitrows,g);vd=prepare(index,bank,model,valid,g)
    models,opts,scales,report=fit(td,vd);out['fit']={'targets':len(fitrows),'selection_targets':len(valid),'arms':report}
    write_metrics(a.out,out,'fit')
    torch.save({'parent_sha256':sha,'parent_updates':768,'gains':{k:m.state_dict() for k,m in models.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'scales':scales,'split_integrity':manifest,'updates':256},a.out.with_suffix('.pt'))
    out['unindexed_secondary']=evaluate(index,bank,model,examples(index,secondary,5304,96),g,models,scales);write_metrics(a.out,out,'secondary')
    out['selection']=evaluate(index,bank,model,valid,g,models,scales);write_metrics(a.out,out,'calibration')
    used=set();forbidden={i for i,p in books};hashes={q['sha256'] for v in manifest['manifest'].values() for q in v}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v4-window:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden or len(B.load(p,200))<112:continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest not in hashes
            hashes.add(digest);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        out['domains'][name]={'ids':[i for i,p in chosen],'metrics':evaluate(index,bank,model,examples(index,chosen,4806,96),g,models,scales)}
        write_metrics(a.out,out,'domain:'+name)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha and all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    out['parent_unchanged']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
