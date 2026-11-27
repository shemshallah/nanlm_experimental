"""v57 content-screened development suite and stronger continuation baseline."""
import sys,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_contrastive'))
from pilot55 import *
from discount import Discount
from clean_split import clean
from contrast import donors as contrast_donors

GRID57=(0.,.0625,.125,.25,.5,.75,1.)
PREVIOUS=set('14256 5277 15592 3200 1064 339 1087 497 1015 1122 1226 1113 592 414 862 77 448 13 6974 8696 13441 2532 11315 11182 360 947 752 1039 713 317 233 853 301 673 1095 275'.split())

def unique(files):
    seen=set();out=[]
    for i,p in files:
        if i not in seen:out.append((i,p));seen.add(i)
    return out

def laws(index,bank,model,rows,g,old,old_scale,contrast,cs,discount,wrong=False):
    x,ids,y=feature(rows);bad=wrong_ids(rows) if wrong else None
    cls=model.cls.cpu().numpy()
    for a in range(0,len(rows),12):
        b=min(a+12,len(rows))
        with torch.no_grad():
            dense=model.log_probs(x[a:b],ids[a:b]).double().exp().numpy();dense/=dense.sum(1,keepdims=True)
            if wrong:
                wd=model.log_probs(x[bad[a:b]],ids[bad[a:b]]).double().exp().numpy();wd/=wd.sum(1,keepdims=True)
        for j,row in enumerate(rows[a:b]):
            hist=row[1];count=index.countlaw(hist);base=(1-g)*count+g*dense[j]
            rr=bank.experts(hist);p53=old_mix(base,[l for l,_ in rr],weights(old,rr,old_scale))
            d,f,v=contrast_donors(hist,p53,index.prior,cls)
            p55=fused(p53,d,apply(contrast,d,f,v,cs,MASKS55['full']))
            yield row[3],{'count':count,'v50':base,'v53':p53,'v55':p55,'kn5':discount.query(hist),'kn3_truncated':discount.query(hist,3),'field':dense[j],**({'wrong_field':wd[j]} if wrong else {})}

def calibrate(args,rows):
    scores=C.Counter();n=0
    for target,p in laws(*args[:3],rows,*args[3:]):
        for order in ('kn5','kn3_truncated'):
            for mode in ('field','v55'):
                for g in GRID57:scores[order,mode,g]-=math.log(max((1-g)*p[order][target]+g*p[mode][target],1e-300))
        n+=1
    choices={}
    for mode in ('field','v55'):
        best=min((k for k in scores if k[1]==mode),key=scores.get)
        choices[mode]=(best[0],best[2])
    return choices,{str(k):v/n for k,v in scores.items()}

def evaluate57(args,rows,settings):
    m=C.defaultdict(C.Counter)
    for target,p in laws(*args[:3],rows,*args[3:],wrong=True):
        modes={k:v for k,v in p.items() if k!='wrong_field'}
        for mode,(order,g) in settings.items():
            modes['kn_'+mode]=(1-g)*p[order]+g*p[mode]
            if mode=='field':modes['kn_wrong_field']=(1-g)*p[order]+g*p['wrong_field']
        for name,q in modes.items():
            assert abs(q.sum()-1)<1e-8 and np.min(q)>0
            rank=np.argsort(-q[1:],kind='stable')[:32]+1;z=m[name];z['n']+=1;z['oov']+=int(target==0)
            z['nll']-=math.log(max(q[target],1e-300));z['t1']+=int(target==rank[0]);z['t5']+=int(target in rank[:5]);z['t32']+=int(target in rank)
    return {k:dict(z,nll=z['nll']/z['n'],top1=z['t1']/z['n'],top5=z['t5']/z['n'],top32=z['t32']/z['n']) for k,z in m.items()}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','parent-metrics','analogy','analogy-metrics','contrast','contrast-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
    a=ap.parse_args();torch.set_num_threads(4)
    groups={k:B.bookfiles(a.data/k) for k in ('stories','sentences','paragraphs')};split=lambda i:int(B.digest('NaNCM-structural-v1:'+i)[:8],16)%10
    books=[(i,p) for i,p in groups['stories'] if split(i)<7];tr=books[:96];cal=books[96:108];extra=books[108:396]
    assert len(tr)==96 and len(cal)==12 and len(extra)==288
    primary,secondary,sel,manifest=cohorts(books,tr,cal,extra)
    blocked={i for i,p in tr+extra+primary+secondary}
    calpool=unique(cal+[(i,p) for i,p in books if i not in blocked]);calpool=[(i,p) for i,p in calpool if i not in blocked]
    forbidden={i for i,p in books}|PREVIOUS
    pools=[('calibration',10,calpool)]
    for name,files in groups.items():
        q=unique([(i,p) for i,p in sorted(files,key=lambda t:B.digest('v57-clean:'+t[0])) if split(i) in (7,8) and i not in forbidden])
        pools.append((name,10,q[:48]))
    reference=[(i,p,3000) for i,p in tr]+[(i,p,6000) for i,p in extra]+[(i,p,1200) for i,p in primary+secondary]
    selected,screen=clean(reference,pools,B.load)
    out={'clean_screen':screen,'scope':'exact prefix content separation; not approximate deduplication or untouched TEST','domains':{},'prior_calibration_reused':True}
    write_metrics(a.out,out,'screen')
    lex=set()
    for p in (a.data/'lex').rglob('*.txt'):
        if 'google' in p.name.lower() and not p.name.lower().startswith(('license','readme')):lex.update(B.words(p.read_text(errors='replace')))
    assert len(lex)>1000
    docs=[B.load(p,3000) for _,p in tr]+[B.load(p,6000) for _,p in extra]
    index=Index(docs,lex);del docs
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);pm=json.loads(a.parent_metrics.read_text());sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    assert sha=='f426598e0439ccefb3281edb951aead3e9ee4414aee211537ea783d44ee52812'
    assert pm['report_stage']=='final' and ck['words']==index.words and ck['split_integrity']==manifest and ck['updates']==768
    model=Field(index.code,index.prior)
    for k in ('code','members','cls','slot','logprior','classprior'):assert torch.equal(model.state_dict()[k],ck['models']['coupled'][k])
    model.load_state_dict(ck['models']['coupled']);model.eval();model.requires_grad_(False);g=float(ck['mixture']['coupled'])
    ac=torch.load(a.analogy,map_location='cpu',weights_only=True);am=json.loads(a.analogy_metrics.read_text())
    assert am['report_stage']=='final' and am['parent_unchanged'] and ac['parent_sha256']==sha and ac['split_integrity']==manifest
    old=OldCredit().double();old.load_state_dict(ac['gains']['full']);old.requires_grad_(False);old_scale=float(ac['scales']['full'])
    cc=torch.load(a.contrast,map_location='cpu',weights_only=True);cm=json.loads(a.contrast_metrics.read_text());ash=hashlib.sha256(a.analogy.read_bytes()).hexdigest()
    assert cm['report_stage']=='final' and cm['v53_replay_verified'] and cc['parent_sha256']==sha and cc['analogy_sha256']==ash and cc['split_integrity']==manifest
    contrast=E.Credit().double();contrast.load_state_dict(cc['gains']['full']);contrast.requires_grad_(False);cs=float(cc['scales']['full'])
    csha=hashlib.sha256(a.contrast.read_bytes()).hexdigest()
    bank=Analogy(index.code,index.seq);discount=Discount(index.seq,index.prior,order=5)
    out.update(parent_sha256=sha,analogy_sha256=ash,contrast_sha256=csha,reference_words=index.reference_words,vocabulary=index.V-1,continuation_rows={str(k):len(v[0]) for k,v in discount.tables.items()})
    write_metrics(a.out,out,'index')
    args=(index,bank,model,g,old,old_scale,contrast,cs,discount)
    val=examples(index,selected['calibration'],5701,96);settings,scores=calibrate(args,val)
    out['calibration']={'settings':settings,'scores':scores,'targets':len(val),'metrics':evaluate57(args,val,settings)}
    write_metrics(a.out,out,'calibration')
    for name in groups:
        rows=examples(index,selected[name],5702,96);out['domains'][name]={'ids':[i for i,p in selected[name]],'metrics':evaluate57(args,rows,settings)}
        write_metrics(a.out,out,'domain:'+name)
    for path,digest in [(a.checkpoint,sha),(a.analogy,ash),(a.contrast,csha)]:assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    out['parent_artifacts_unchanged']=True;out['eligible_for_promotion']=False
    write_metrics(a.out,out)
if __name__=='__main__':main()
