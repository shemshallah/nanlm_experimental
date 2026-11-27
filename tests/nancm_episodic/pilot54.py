"""v54: extend verified v53 using three independent local episodic experts."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_analogy'))
from pilot53 import *
from analogy import Credit as OldCredit,mix as old_mix
import episodic as E
MASKS54={'suffix':(True,False,False),'skip':(False,True,False),'topic':(False,False,True),'full':(True,True,True)}
GRID54=(0.,.125,.25,.5,.75,1.)

def tensors(rr):
    return torch.tensor(np.stack([f for _,f in rr])[None],dtype=torch.float64),torch.tensor([[bool(l) for l,_ in rr]])

def weights(m,rr,scale,mask=(True,True,True)):
    f,v=tensors(rr)
    with torch.no_grad():return m.weights(f,v&torch.tensor(mask),scale)[0].numpy()

def prepare54(index,bank,model,rows,g,old,scale):
    x,ids,y=feature(rows)
    with torch.no_grad():dense=torch.cat([model.target_logprob(x[i:i+32],ids[i:i+32],y[i:i+32]).double().exp() for i in range(0,len(y),32)]).numpy()
    pp=[];ff=[];vv=[]
    for row,p in zip(rows,dense):
        t=row[3];h=row[1];base=(1-g)*index.countlaw(h)[t]+g*p;rr=bank.experts(h);w=weights(old,rr,scale)
        parent=w[0]*base+sum(a*l.get(t,0.) for a,(l,_) in zip(w[1:],rr));ep=E.experts(h)
        pp.append([parent]+[l.get(t,0.) for l,_ in ep]);ff.append(np.stack([f for _,f in ep]));vv.append([bool(l) for l,_ in ep])
    return torch.tensor(pp,dtype=torch.float64),torch.tensor(np.stack(ff),dtype=torch.float64),torch.tensor(vv)

def fit54(tr,va):
    models={};opts={};scales={};report={}
    for name,mask in MASKS54.items():
        m=E.Credit().double();opt=torch.optim.AdamW(m.parameters(),lr=.025,weight_decay=.01);rng=np.random.default_rng(5402);mask=torch.tensor(mask)
        p,f,v=tr;q,z,u=va;trace=[]
        for i in range(256):
            ix=rng.integers(len(p),size=32);opt.zero_grad();loss=-m(p[ix],f[ix],v[ix]&mask).log().mean();loss.backward();opt.step()
            if i==0 or (i+1)%64==0:trace.append([i+1,float(loss.detach())])
        with torch.no_grad():scores={str(s):float(-m(q,z,u&mask,s).log().mean()) for s in GRID54}
        scale=min(GRID54,key=lambda s:scores[str(s)]);models[name]=m;opts[name]=opt;scales[name]=scale
        report[name]={'selection_nll':scores,'scale':scale,'trace':trace,'parameters':m.gain.detach().tolist()}
    return models,opts,scales,report

def evaluate54(index,bank,model,rows,g,models,scales,old,old_scale):
    x,ids,y=feature(rows);wrong=wrong_ids(rows);m=C.defaultdict(C.Counter);audit=C.Counter()
    for a in range(0,len(rows),12):
        b=min(a+12,len(rows))
        with torch.no_grad():dense=model.log_probs(x[a:b],ids[a:b]).double().exp().numpy();dense/=dense.sum(1,keepdims=True)
        for j,row in enumerate(rows[a:b]):
            i=a+j;hist=row[1];t=row[3];q=index.countlaw(hist);base=(1-g)*q+g*dense[j]
            rr=bank.experts(hist);parent=old_mix(base,[l for l,_ in rr],weights(old,rr,old_scale))
            ep=E.experts(hist);bad=E.experts(rows[wrong[i]][1][:-4]+hist[-4:])
            modes={'count':(q,0.),'v50':(base,0.),'parent':(parent,0.)}
            for name,mask in MASKS54.items():
                w=weights(models[name],ep,scales[name],mask);modes[name]=(E.mix(parent,[l for l,_ in ep],w),1-w[0])
                if name=='full':
                    # Keep parent and query's last four words fixed. Swap older history.
                    wb=weights(models[name],bad,scales[name]);modes['wrong_history']=(E.mix(parent,[l for l,_ in bad],wb),1-wb[0])
                    wf=weights(models[name],ep,1.);wbf=weights(models[name],bad,1.)
                    modes['forced']=(E.mix(parent,[l for l,_ in ep],wf),1-wf[0]);modes['wrong_forced']=(E.mix(parent,[l for l,_ in bad],wbf),1-wbf[0])
            rank=np.argsort(-parent[1:],kind='stable')[:32]+1;audit['n']+=1;audit['parent32']+=int(t in rank)
            for name,(l,_) in zip(E.NAMES,ep):audit[name]+=int(t>0 and t in l)
            audit['new_target']+=int(t>0 and t not in rank and any(t in l for l,_ in ep))
            for name,(p,wt) in modes.items():
                assert abs(p.sum()-1)<1e-7 and np.min(p)>=0
                rank=np.argsort(-p[1:],kind='stable')[:32]+1;z=m[name];z['n']+=1;z['nll']-=math.log(max(p[t],1e-300));z['t1']+=int(t==rank[0]);z['t5']+=int(t in rank[:5]);z['t32']+=int(t in rank);z['weight']+=wt
    return {'modes':{k:dict(z,nll=z['nll']/z['n'],top1=z['t1']/z['n'],top5=z['t5']/z['n'],top32=z['t32']/z['n'],mean_episodic_weight=z['weight']/z['n']) for k,z in m.items()},'discovery':dict(audit)}

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
    write_metrics(a.out,out,'index')
    fitrows=examples(index,secondary,5403,96);valid=examples(index,sel,4804,64)
    td=prepare54(index,bank,model,fitrows,g,old,old_scale);vd=prepare54(index,bank,model,valid,g,old,old_scale)
    models,opts,scales,report=fit54(td,vd);out['fit']={'targets':len(fitrows),'selection_targets':len(valid),'arms':report}
    write_metrics(a.out,out,'fit')
    torch.save({'parent_sha256':sha,'parent_updates':768,'gains':{k:m.state_dict() for k,m in models.items()},'optimizers':{k:o.state_dict() for k,o in opts.items()},'scales':scales,'split_integrity':manifest,'updates':256,'analogy_sha256':out['analogy_sha256'],'analogy_scale':old_scale},a.out.with_suffix('.pt'))
    out['selection']=evaluate54(index,bank,model,valid,g,models,scales,old,old_scale);write_metrics(a.out,out,'calibration')
    used=set();forbidden={i for i,p in books};hashes={q['sha256'] for v in manifest['manifest'].values() for q in v}
    for name,files in groups.items():
        chosen=[]
        for bid,p in sorted(files,key=lambda t:B.digest('v4-window:'+t[0])):
            if split(bid) not in (7,8) or bid in used or bid in forbidden or len(B.load(p,200))<112:continue
            digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest not in hashes
            hashes.add(digest);used.add(bid);chosen.append((bid,p))
            if len(chosen)==6:break
        assert len(chosen)==6
        out['domains'][name]={'ids':[i for i,p in chosen],'metrics':evaluate54(index,bank,model,examples(index,chosen,4806,96),g,models,scales,old,old_scale)}
        write_metrics(a.out,out,'domain:'+name)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==sha and all(torch.equal(v,ck['models']['coupled'][k]) for k,v in model.state_dict().items())
    assert hashlib.sha256(a.analogy.read_bytes()).hexdigest()==out['analogy_sha256'] and torch.equal(old.gain,ac['gains']['full']['gain'])
    for name in groups:
        actual=out['domains'][name]['metrics']['modes']['parent'];expected=am['domains'][name]['metrics']['modes']['full']
        assert all(actual[k]==expected[k] for k in ('n','t1','t5','t32')) and abs(actual['nll']-expected['nll'])<1e-6,(name,actual,expected)
    out['parent_unchanged']=True;out['v53_replay_verified']=True;write_metrics(a.out,out)
if __name__=='__main__':main()
