"""v51 exact leave-book-out corpus audit and optional v50 optimizer continuation."""
import sys,pathlib,copy,hashlib,math
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_hierarchical'))
from pilot50 import Index,examples,cohorts,write_metrics,B,C,np,torch,argparse,reference_examples,feature,wrong_ids
from hierarchical_field import Field
from crossfit import Crossfit,Exclusion,Gate,loss
from pair_field import PairField

GRID=(0.,.0625,.125,.25,.5,.75,1.)
def reference_data(index,cf,seed,per_book):
    x,ids,y=reference_examples(index,seed,per_book);rng=np.random.default_rng(seed);pp=[];ff=[];included=[];checks=0;err=0.;j=0
    for seq in index.seq:
        if len(seq)<17:continue
        pos=np.sort(rng.choice(np.arange(16,len(seq)),min(per_book,len(seq)-16),replace=False))
        ex=Exclusion(seq,index.V)
        for i in pos:
            t=int(seq[i]);h=seq[i-3:i];assert y[j].item()==t and np.array_equal(ids[j,:3].numpy(),h[::-1]);j+=1
            q,f=cf.query(h,t,ex);p,_=cf.query(h,t)
            pp.append(q);ff.append(f);included.append(p)
            if checks<64:
                dense,feat=cf.query(h,exclude=ex);err=max(err,abs(dense[t]-q));assert np.array_equal(f,feat);checks+=1
    assert j==len(y) and err<1e-12
    p=np.asarray(included);q=np.asarray(pp);features=np.stack(ff)
    return (x,ids,y,torch.tensor(q),torch.tensor(features)),{"targets":j,"indexed_nll":float(-np.log(p).mean()),"leave_book_out_nll":float(-np.log(q).mean()),"excluded_probability_lower_fraction":float((q<p).mean()),"scalar_dense_checks":checks,"max_error":err}

def holdout(index,cf,rows):
    x,ids,y=feature(rows);p=[];f=[];err=0.
    for row in rows:
        q,v=cf.query(row[1]);err=max(err,float(np.max(np.abs(q-index.countlaw(row[1])))));p.append(q[row[3]]);f.append(v)
    assert err<1e-12
    return x,ids,y,torch.tensor(p,dtype=torch.float64),torch.tensor(np.stack(f))

def calibration(model,gate,ds,dynamic):
    x,ids,y,p,f=ds
    with torch.no_grad():
        log=torch.cat([model.target_logprob(x[i:i+32],ids[i:i+32],y[i:i+32]) for i in range(0,len(y),32)])
        scores={}
        for base in GRID:
            for strength in ((0.,.25,.5,1.) if dynamic and 0<base<1 else (0.,)):
                scores[(base,strength)]=float(loss(log,p,gate(f,base,strength)))
    best=min(scores,key=scores.get);return best,{str(k):v for k,v in scores.items()}

def eval_models(index,cf,rows,models,gates,settings):
    x,ids,y,p,f=holdout(index,cf,rows);bad=wrong_ids(rows);ms=C.defaultdict(C.Counter);maxerr=0.
    with torch.no_grad():
        for start in range(0,len(y),12):
            end=min(start+12,len(y));v={}
            for name,model in models.items():
                z=model.log_probs(x[start:end],ids[start:end]).double().exp().numpy();maxerr=max(maxerr,float(abs(z.sum(1)-1).max()));z/=z.sum(1,keepdims=True)
                base,strength=settings[name];g=gates[name](f[start:end],base,strength).numpy()
                v[name]=(z,g)
            z=models['adaptive'].log_probs(x[bad[start:end]],ids[bad[start:end]]).double().exp().numpy();z/=z.sum(1,keepdims=True)
            if 'pair' in models:
                pair=models['pair'];scale=pair.pair_scale
                wrong_pair=pair.log_probs(x[bad[start:end]],ids[bad[start:end]]).double().exp().numpy();wrong_pair/=wrong_pair.sum(1,keepdims=True)
                pair.pair_scale=0.
                try:off=pair.log_probs(x[start:end],ids[start:end]).double().exp().numpy();off/=off.sum(1,keepdims=True)
                finally:pair.pair_scale=scale
            for j,row in enumerate(rows[start:end]):
                q,_=cf.query(row[1]);t=row[3];modes={'count':(q,0.)}
                for name,(dense,g) in v.items():modes[name]=((1-g[j])*q+g[j]*dense[j],float(g[j]))
                g=v['adaptive'][1][j];modes['wrong_field']=((1-g)*q+g*z[j],float(g))
                if 'pair' in v:
                    g=v['pair'][1][j];modes['pair_off']=((1-g)*q+g*off[j],float(g));modes['wrong_pair']=((1-g)*q+g*wrong_pair[j],float(g))
                for name,(p,weight) in modes.items():
                    assert abs(p.sum()-1)<1e-9
                    rank=np.argsort(-p[1:],kind='stable')[:32]+1;m=ms[name];m['n']+=1;m['oov']+=int(t==0);m['nll']-=math.log(max(p[t],1e-300));m['t1']+=int(t==rank[0]);m['t5']+=int(t in rank[:5]);m['t32']+=int(t in rank);m['gate']+=weight
    return {'modes':{k:dict(m,nll=m['nll']/m['n'],top1=m['t1']/m['n'],top5=m['t5']/m['n'],top32=m['t32']/m['n'],mean_gate=m['gate']/m['n']) for k,m in ms.items()},'dense_mass_max_error':maxerr}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('pairwise-metrics.json'));ap.add_argument('--checkpoint',type=pathlib.Path);ap.add_argument('--parent-metrics',type=pathlib.Path);ap.add_argument('--per-book',type=int,default=64);ap.add_argument('--updates',type=int,default=256);a=ap.parse_args();torch.set_num_threads(4)
    groups={k:B.bookfiles(a.data/k) for k in ('stories','sentences','paragraphs')};split=lambda i:int(B.digest('NaNCM-structural-v1:'+i)[:8],16)%10
    books=[(i,p) for i,p in groups['stories'] if split(i)<7];tr=books[:96];cal=books[96:108];extra=books[108:396]
    assert len(tr)==96 and len(cal)==12 and len(extra)==288
    primary,_,sel,manifest=cohorts(books,tr,cal,extra)
    lex=set()
    for p in (a.data/'lex').rglob('*.txt'):
        if 'google' in p.name.lower() and not p.name.lower().startswith(('license','readme')):lex.update(B.words(p.read_text(errors='replace')))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr]+[B.load(p,6000) for _,p in extra]
    index=Index(docs,lex);del docs;cf=Crossfit(index)
    out={'reference_books':384,'reference_words':index.reference_words,'vocabulary':index.V-1,'split_integrity':manifest,'domains':{}}
    ds,audit=reference_data(index,cf,5101,a.per_book);out['reference_audit']=audit
    val=examples(index,sel,4804,64);va=holdout(index,cf,val)
    out['unindexed_calibration_count_nll']=float(-va[3].log().mean());write_metrics(a.out,out,'crossfit_audit')
    if a.checkpoint is None:write_metrics(a.out,out);return
    assert a.parent_metrics is not None
    import json
    pm=json.loads(a.parent_metrics.read_text());ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert pm['report_stage']=='final' and ck['updates']==768 and ck['words']==index.words and ck['split_integrity']==manifest
    assert set(ck['models'])==set(ck['optimizers'])=={'coupled','uncoupled'}
    out['parent_sha256']=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest();out['parent_updates']=ck['updates']
    parent=Field(index.code,index.prior)
    for k in ('code','members','cls','slot','logprior','classprior'):assert torch.equal(parent.state_dict()[k],ck['models']['coupled'][k])
    parent.load_state_dict(ck['models']['coupled']);models={'parent':parent};gates={'parent':Gate()};opts={};gate_opts={};pair_opts={};pair_scales={}
    parent_weight=float(ck['mixture']['coupled']);settings={'parent':(parent_weight,0.)};train_weight=min(.875,max(.125,parent_weight));out['shared_training_gate']=train_weight;out['arms']={}
    for name,dynamic in [('fixed',False),('adaptive',True),('pair',True)]:
        torch.manual_seed(5201)
        if name=='pair':
            model=PairField(index.code,index.prior);missing,unexpected=model.load_state_dict(parent.state_dict(),strict=False)
            assert all(k.startswith('pair.') for k in missing) and not unexpected
            model.pair_scale=0.
            with torch.no_grad():assert torch.equal(model.target_logprob(ds[0][:8],ds[1][:8],ds[2][:8]),parent.target_logprob(ds[0][:8],ds[1][:8],ds[2][:8]))
            model.pair_scale=1.
            inherited=[v for k,v in model.named_parameters() if not k.startswith('pair.')]
            pair_opt=torch.optim.AdamW(model.pair.parameters(),lr=.003,weight_decay=.01);pair_opts[name]=pair_opt
        else:
            model=copy.deepcopy(parent);inherited=list(model.parameters());pair_opt=None
        gate=Gate();opt=torch.optim.AdamW(inherited);opt.load_state_dict(copy.deepcopy(ck['optimizers']['coupled']))
        gopt=torch.optim.AdamW(gate.parameters(),lr=.01,weight_decay=.001)
        def steps():
            return [int(s['step']) for s in opt.state.values() if 'step' in s]
        before_steps=steps();assert before_steps and min(before_steps)==max(before_steps)==768
        rng=np.random.default_rng(5102);trace=[]
        for i in range(a.updates):
            ix=rng.integers(len(ds[2]),size=64);x,ids,y,p,f=(t[ix] for t in ds)
            opt.zero_grad();gopt.zero_grad()
            if pair_opt is not None:pair_opt.zero_grad()
            log=model.target_logprob(x,ids,y)
            l=loss(log,p,gate(f,train_weight,1. if dynamic else 0.));l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if dynamic:gopt.step()
            if pair_opt is not None:pair_opt.step()
            if i==0 or (i+1)%32==0:trace.append([i+1,float(l.detach())])
        after_steps=steps();assert min(after_steps)==max(after_steps)==768+a.updates
        if name=='pair':
            choices=[]
            for scale in (0.,.25,.5,1.):
                model.pair_scale=scale;setting,grid=calibration(model,gate,va,dynamic);choices.append((grid[str(setting)],scale,setting,grid))
            _,scale,settings[name],grid=min(choices,key=lambda item:item[0]);model.pair_scale=scale;pair_scales[name]=scale
            out['pair_scale_selection']=[{'nll':v,'scale':s,'gate':g} for v,s,g,_ in choices]
            assert all(int(st['step'])==a.updates for st in pair_opt.state.values())
        else:settings[name],grid=calibration(model,gate,va,dynamic)
        models[name]=model;gates[name]=gate;opts[name]=opt;gate_opts[name]=gopt
        out['arms'][name]={'selection':grid,'selected':settings[name],'trace':trace,'optimizer_step':after_steps[0]};write_metrics(a.out,out,'fit:'+name)
        torch.save({'parent_sha256':out['parent_sha256'],'words':index.words,'models':{k:v.state_dict() for k,v in models.items()},'gates':{k:v.state_dict() for k,v in gates.items()},'optimizers':{k:v.state_dict() for k,v in opts.items()},'gate_optimizers':{k:v.state_dict() for k,v in gate_opts.items()},'settings':settings,'pair_scales':pair_scales,'pair_optimizers':{k:v.state_dict() for k,v in pair_opts.items()},'updates_added':a.updates,'split_integrity':manifest},a.out.with_suffix('.pt'))
    out['selection']=eval_models(index,cf,val,models,gates,settings);write_metrics(a.out,out,'calibration')
    used=set();forbidden={i for i,p in books};hashes={q['sha256'] for v in manifest['manifest'].values() for q in v}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v4-window:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden or len(B.load(p,200))<112:continue
            sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha not in hashes
            hashes.add(sha);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        out['domains'][name]={'ids':[i for i,p in chosen],'metrics':eval_models(index,cf,examples(index,chosen,4806,96),models,gates,settings)};write_metrics(a.out,out,'domain:'+name)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==out['parent_sha256']
    write_metrics(a.out,out)
if __name__=='__main__':main()
