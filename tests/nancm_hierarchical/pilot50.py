"""v50 component diagnostic: large reference-only supervised field fit.
No indexed continuation lookup is an input during field training.
"""
import pathlib,sys
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_scout'))
from pilot import Index,examples,cohorts,write_metrics,B,C,np,torch,argparse
from hierarchical_field import Field
import hashlib,math,time

def reference_examples(index,seed,per_book=192):
    rng=np.random.default_rng(seed);xs=[];ids=[];ys=[]
    for seq in index.seq:
        if len(seq)<17:continue
        pos=np.sort(rng.choice(np.arange(16,len(seq)),min(per_book,len(seq)-16),replace=False))
        local=seq[pos[:,None]-np.arange(1,5)[None,:]]
        x=np.zeros((len(pos),8,index.code.shape[1]),np.float32);x[:,:4]=index.code[local]
        cum=np.vstack((np.zeros((1,index.code.shape[1]),np.float64),np.cumsum(index.code[seq],axis=0,dtype=np.float64)))
        for j,(a,b) in enumerate(((0,8),(8,32),(32,128),(128,512)),4):
            l=np.maximum(0,pos-b);r=np.maximum(0,pos-a);x[:,j]=(cum[r]-cum[l])/np.maximum(r-l,1)[:,None]
        xs.append(x);ids.append(local);ys.append(seq[pos])
    return tuple(torch.from_numpy(np.concatenate(a)) for a in (xs,ids,ys))

def feature(rows):
    x=torch.tensor(np.stack([r[2] for r in rows]));ids=np.zeros((len(rows),4),np.int64)
    for i,r in enumerate(rows):ids[i,:min(4,len(r[1]))]=r[1][-4:][::-1]
    return x,torch.tensor(ids),torch.tensor([r[3] for r in rows])

def wrong_ids(rows):
    by=C.defaultdict(list)
    for i,row in enumerate(rows):by[row[0]].append(i)
    books=sorted(by);assert len(books)>1
    out=np.empty(len(rows),np.int64)
    for i,b in enumerate(books):
        other=by[books[(i+1)%len(books)]]
        for j,k in enumerate(by[b]):out[k]=other[j%len(other)]
    assert all(rows[i][0]!=rows[j][0] for i,j in enumerate(out))
    return out

GRID=(0.,.0625,.125,.25,.5,.75,1.)
def calib(model,index,rows,coupled):
    x,ids,y=feature(rows);scores=C.Counter();n=0
    with torch.no_grad():
        for i in range(0,len(y),32):
            yy=y[i:i+32];pp=model.target_logprob(x[i:i+32],ids[i:i+32],yy,coupled).double().exp().numpy()
            for r,p in zip(rows[i:i+32],pp):
                q=index.countlaw(r[1])[r[3]]
                for w in GRID:scores[w]-=math.log(max((1-w)*q+w*p,1e-300))
                n+=1
    return min(GRID,key=scores.get),{str(k):v/n for k,v in scores.items()}

def evaluate(models,index,rows,weights):
    x,ids,y=feature(rows);wrong=wrong_ids(rows);m=C.defaultdict(C.Counter);delta=0.;mass=0.
    with torch.no_grad():
        for a in range(0,len(y),12):
            b=min(a+12,len(y));yy=y[a:b]
            qq={}
            for name,model in models.items():
                coupled=name=='coupled';logs=model.log_probs(x[a:b],ids[a:b],coupled)
                sp=model.target_logprob(x[a:b],ids[a:b],yy,coupled)
                delta=max(delta,float((logs[torch.arange(b-a),yy]-sp).abs().max()))
                pp=logs.double().exp().numpy();mass=max(mass,float(np.max(np.abs(pp.sum(1)-1))))
                pp/=pp.sum(1,keepdims=True);qq[name]=pp
            bad=models['coupled'].log_probs(x[wrong[a:b]],ids[wrong[a:b]],True).double().exp().numpy();bad/=bad.sum(1,keepdims=True)
            for j,r in enumerate(rows[a:b]):
                t=r[3];q=index.countlaw(r[1]);full=qq['coupled'][j];off=qq['uncoupled'][j]
                g=weights['coupled'];h=weights['uncoupled']
                modes={'count':q,'coupled':full,'uncoupled':off,'count_coupled':(1-g)*q+g*full,'count_uncoupled':(1-h)*q+h*off,'wrong_field':(1-g)*q+g*bad[j],'forced_half':.5*q+.5*full,'wrong_forced_half':.5*q+.5*bad[j]}
                for name,p in modes.items():
                    z=m[name];z['n']+=1;z['oov']+=int(t==0);z['nll']-=math.log(max(p[t],1e-300))
                    rank=np.argsort(-p[1:],kind='stable')[:32]+1
                    z['t1']+=int(t==rank[0]);z['t5']+=int(t in rank[:5]);z['t32']+=int(t in rank)
    assert delta<2e-5 and mass<2e-5,(delta,mass)
    return {'modes':{k:dict(z,nll=z['nll']/z['n'],top1=z['t1']/z['n'],top5=z['t5']/z['n'],top32=z['t32']/z['n']) for k,z in m.items()},'target_dense_logprob_max_error':delta,'normalization_max_error':mass}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('hierarchical-metrics.json'));ap.add_argument('--updates',type=int,default=768);ap.add_argument('--per-book',type=int,default=192);a=ap.parse_args();r=a.data
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    groups={k:B.bookfiles(r/k) for k in ('stories','sentences','paragraphs')};split=lambda i:int(B.digest('NaNCM-structural-v1:'+i)[:8],16)%10
    books=[(i,p) for i,p in groups['stories'] if split(i)<7];tr=books[:96];cal=books[96:108];extra=books[108:396]
    assert len(tr)==96 and len(cal)==12 and len(extra)==288
    primary,_,sel,manifest=cohorts(books,tr,cal,extra)
    lex=set()
    for p in (r/'lex').rglob('*.txt'):
        if 'google' in p.name.lower() and not p.name.lower().startswith(('license','readme')):lex.update(B.words(p.read_text(errors='replace')))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr]+[B.load(p,6000) for _,p in extra]
    index=Index(docs,lex);del docs
    out={'vocabulary':index.V-1,'reference_words':index.reference_words,'reference_books':384,'split_integrity':manifest,'training_input':'causal prefix only; no indexed continuation count features','domains':{},'arms':{}}
    write_metrics(a.out,out,'index')
    x,ids,y=reference_examples(index,5001,a.per_book);assert (y>0).all()
    out['unique_sampled_reference_positions']=len(y);out['sampled_exposures_per_arm']=a.updates*64
    valid=examples(index,sel,4804,64);vx,vi,vy=feature(valid);models={};opts={};weights={}
    for name,coupled in (('uncoupled',False),('coupled',True)):
        torch.manual_seed(5002);model=Field(index.code,index.prior);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
        rng=np.random.default_rng(5003);trace=[];start=time.monotonic()
        def val():
            with torch.no_grad():return float(-model.target_logprob(vx,vi,vy,coupled).mean())
        before=val()
        for i in range(a.updates):
            ix=rng.integers(len(y),size=64);opt.zero_grad()
            loss=-model.target_logprob(x[ix],ids[ix],y[ix],coupled).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if i==0 or (i+1)%128==0:
                trace.append([i+1,float(loss.detach()),val()]);print('TRAIN',name,trace[-1],flush=True)
        models[name]=model;opts[name]=opt;w,grid=calib(model,index,valid,coupled);weights[name]=w
        out['arms'][name]={'before_selection_nll':before,'after_selection_nll':val(),'trace':trace,'parameters':sum(p.numel() for p in model.parameters()),'classes':len(model.members),'largest_class':int((model.members>=0).sum(1).max()),'mixture_weight':w,'selection_mixture_nll':grid,'seconds':time.monotonic()-start}
        write_metrics(a.out,out,'fit:'+name)
        torch.save({'words':index.words,'models':{k:m.state_dict() for k,m in models.items()},'optimizers':{k:v.state_dict() for k,v in opts.items()},'updates':a.updates,'mixture':weights,'split_integrity':manifest},a.out.with_suffix('.pt'))
    out['unindexed_primary']=evaluate(models,index,examples(index,primary,5004,96),weights)
    out['selection']=evaluate(models,index,valid,weights);write_metrics(a.out,out,'calibration')
    used=set();forbidden={i for i,p in books};hashes={q['sha256'] for v in manifest['manifest'].values() for q in v}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v4-window:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden:continue
            if len(B.load(p,200))<112:continue
            sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha not in hashes
            hashes.add(sha);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        out['domains'][name]={'ids':[i for i,p in chosen],'metrics':evaluate(models,index,examples(index,chosen,4806,96),weights)}
        write_metrics(a.out,out,'domain:'+name)
    write_metrics(a.out,out)
if __name__=='__main__':main()
