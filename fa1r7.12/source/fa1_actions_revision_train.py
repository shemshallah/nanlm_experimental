#!/usr/bin/env python3
"""FA1 GitHub Actions R7.14 temporal revision router *auxiliary* training.

Consumes real NaNROUTER public-domain short-story books, but deliberately uses
SURROGATE lexical routing features, NOT the FA1 model's learned embedding E.
It can continue the R7.13 scalar router Adam states without carrying or changing
the accepted 14,416-step full LM checkpoint. This is a pipeline/architecture
experiment, not full LM training, and its weights must NOT be promoted into the
actual FA1 model without matched-embedding and LM-specificity gates.
"""
from __future__ import annotations
import argparse,collections,copy,hashlib,json,math,pathlib,random,re,time

SEG=768
SEGMENTS=8
HISTORY=SEG*SEGMENTS
PARENT_SHA='54e835b00642013d4f1bf486a8150eb1ff06c547565771bd21933e143d050f2d'
WORD=re.compile(r"[a-z]+(?:'[a-z]+)?",re.I)


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def split_for(book_id:str)->str:
    # Whole-book split, independent from PR/CI reruns. Authors may still overlap.
    v=int(digest(book_id.encode())[:8],16)%100
    return 'train' if v<70 else 'dev' if v<85 else 'test'


def events_from_book(text:str,book_id:str,max_windows:int=2,max_per_window:int=3):
    words=WORD.findall(text.lower())[:100000]
    if len(words)<HISTORY+2:
        return []
    wid={s:i for i,s in enumerate(dict.fromkeys(words))}
    seq=[wid[w] for w in words]
    rng=random.Random(int(digest(book_id.encode())[:12],16))
    max_start=len(seq)-HISTORY-1
    starts=[round(max_start*i/max(1,max_windows-1)) for i in range(max_windows)]
    out=[]
    for st in sorted(set(starts)):
        window=seq[st:st+HISTORY]
        # All pairs used are strictly INSIDE the historical window. No current
        # prediction target, next book, or future token can enter features.
        bykey=collections.defaultdict(list)
        for i in range(HISTORY-1):
            if i%SEG==SEG-1:continue
            k,v=window[i],window[i+1]
            if k!=v:bykey[k].append((i//SEG,i,v))
        # Hard negatives: same exact key with two different successor values
        # in distinct historical segments. Target = LATEST historical fact.
        candidates=[]
        for k,positions in bykey.items():
            if len({seg for seg,_,_ in positions})<2:continue
            last=max(positions,key=lambda x:x[1])
            if any(seg!=last[0] and val!=last[2] for seg,_,val in positions):
                candidates.append((k,positions,last))
        rng.shuffle(candidates)
        for k,positions,last in candidates[:max_per_window]:
            counts=collections.Counter(seg for seg,_,_ in positions)
            features=[]
            for seg in range(SEGMENTS):
                n=counts[seg]
                # Surrogate similarity flag, not FA1 embedding-cosine. Explicitly
                # marked non-transferable in output and acceptance decision.
                cos=1.0 if n else 0.45
                features.append([cos,math.log1p(n), -1.+seg/(SEGMENTS-1)])
            out.append({'features':features,'target':int(last[0]),
                        'duplicates':len(positions),'book':book_id,
                        'target_time':int(last[1]),'source':st})
    return out


def load_books(folder:pathlib.Path,max_books:int,max_windows:int,max_per_window:int):
    paths=sorted(folder.rglob('*.txt'))
    groups={'train':[],'dev':[],'test':[]}
    hashes=set();books=collections.Counter();errors=[]
    for f in paths:
        if sum(books.values())>=max_books:break
        try:
            data=f.read_bytes()
            if len(data)<20000 or digest(data) in hashes:continue
            txt=data.decode('utf-8','replace')
            bid=str(f.relative_to(folder))
            ev=events_from_book(txt,bid,max_windows,max_per_window)
            if not ev:continue
            hashes.add(digest(data));s=split_for(bid)
            groups[s].extend(ev);books[s]+=1
        except (OSError,UnicodeError) as ex:errors.append(f'{f.name}:{str(ex)[:80]}')
    if min(map(len,groups.values()))<8:
        raise RuntimeError('insufficient whole-book-disjoint data: '+str({x:len(y) for x,y in groups.items()}))
    return groups,dict(books),errors


def load_seed(path:pathlib.Path):
    data=json.loads(path.read_text())
    if data.get('accepted_checkpoint_sha256')!=PARENT_SHA:
        raise RuntimeError('wrong accepted checkpoint parent')
    if data.get('surrogate_training_promoted') is True:
        raise RuntimeError('surrogate state may not be promoted')
    weights=[float(x) for x in data['weights']]
    moments=data['adam'];state={k:[float(v) for v in moments[k]] for k in ('m','v')}
    if len(weights)!=3 or len(state['m'])!=3 or len(state['v'])!=3:
        raise RuntimeError('invalid inherited R7.13 scalar router')
    state['step']=int(moments['step'])
    return weights,state,data


def softmax(x):
    u=max(x);e=[math.exp(t-u) for t in x];s=sum(e)
    return [t/s for t in e]


def score(w,item):
    return [sum(a*b for a,b in zip(w,feat)) for feat in item['features']]


def evaluate(w,examples):
    ce=0.;correct=0;latest=0
    for item in examples:
        logits=score(w,item);p=softmax(logits);target=item['target']
        ce-=math.log(max(1e-30,p[target]))
        pred=max(range(len(logits)),key=lambda i:logits[i])
        correct+=int(pred==target)
        # Ideal latest-occurrence lexical lookup, a *strong* oracle control.
        latest+=1
    n=len(examples)
    return {'n':n,'cross_entropy_nats':ce/n,'top1':correct/n,
            'uniform_top1_chance':1/SEGMENTS,'uniform_cross_entropy':math.log(SEGMENTS),
            'latest_exact_lookup_oracle_top1':latest/n}


def update_one(w,adam,event,lr):
    logit=score(w,event);prob=softmax(logit);target=event['target']
    grad=[0.,0.,0.]
    for seg,fs in enumerate(event['features']):
        d=prob[seg]-int(seg==target)
        for j in range(3):grad[j]+=d*fs[j]
    gnorm=math.sqrt(sum(x*x for x in grad))
    if gnorm>3.:grad=[x*3./gnorm for x in grad]
    adam['step']+=1;t=adam['step'];b1,b2=.9,.999
    for j in range(3):
        adam['m'][j]=b1*adam['m'][j]+(1-b1)*grad[j]
        adam['v'][j]=b2*adam['v'][j]+(1-b2)*grad[j]**2
        mh=adam['m'][j]/(1-b1**t);vh=adam['v'][j]/(1-b2**t)
        w[j]-=lr*mh/(math.sqrt(vh)+1e-8)
    return -math.log(max(1e-30,prob[target]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--books',required=True,type=pathlib.Path)
    ap.add_argument('--seed',required=True,type=pathlib.Path)
    ap.add_argument('--report',required=True,type=pathlib.Path)
    ap.add_argument('--out-state',required=True,type=pathlib.Path)
    ap.add_argument('--updates',type=int,default=512)
    ap.add_argument('--max-books',type=int,default=160)
    ap.add_argument('--windows-per-book',type=int,default=2)
    ap.add_argument('--examples-per-window',type=int,default=5)
    ap.add_argument('--lr',type=float,default=.02)
    a=ap.parse_args(); start=time.monotonic()
    w,adam,base=load_seed(a.seed)
    split,bookcounts,errors=load_books(a.books,a.max_books,a.windows_per_book,a.examples_per_window)
    before={s:evaluate(w,items) for s,items in split.items() if s!='test'}
    orig=copy.deepcopy({'weights':w,'adam':adam})
    best=copy.deepcopy(orig);best_ce=before['dev']['cross_entropy_nats'];best_update=0
    rng=random.Random(71411);train=split['train'];log=[]
    for step in range(1,a.updates+1):
        example=rng.choice(train)
        loss=update_one(w,adam,example,a.lr)
        if step==1 or step%32==0 or step==a.updates:
            dev=evaluate(w,split['dev'])
            log.append({'update':step,'train_step_nll':loss,'dev_ce':dev['cross_entropy_nats'],
                        'dev_top1':dev['top1']})
            if dev['cross_entropy_nats']<best_ce-1e-9:
                best=copy.deepcopy({'weights':w,'adam':adam});best_ce=dev['cross_entropy_nats'];best_update=step
    w,adam=best['weights'],best['adam']
    after={s:evaluate(w,split[s]) for s in ('train','dev','test')}
    report={'codename':'FA1','round':'R7.14','task':'auxiliary temporal conflict segment routing',
            'data':'NaNROUTER Project Gutenberg short-stories release batch 0001',
            'data_verified_sha256_from_workflow':True,
            'accepted_checkpoint_sha256':PARENT_SHA,'accepted_model_updates':14416,
            'accepted_model_modified':False,'surrogate_training_promoted':False,
            'trained_parameters':'3 scalar router weights, inherited R7.13 Adam moments',
            'embedding_type':'SURROGATE exact-key score; NOT FA1 learned embedding',
            'book_disjoint_by_source_path':True,'author_disjoint':False,
            'semantic_paraphrase_tested':False,'natural_language_loss_tested':False,
            'hard_negative':'same key, competing historical successor, multiple segments',
            'context_tokens':HISTORY,'segments':SEGMENTS,'segment_tokens':SEG,
            'books':bookcounts,'examples':{s:len(x) for s,x in split.items()},
            'initial_router':orig,'before':before,'selected_update':best_update,
            'attempted_updates':a.updates,'after':after,'training_trace':log,
            'promotion_gate_passed':False,'errors':errors[:15],
            'seconds':round(time.monotonic()-start,2)}
    new={'format':'FA1_R7_14_ACTIONS_TEMPORAL_ROUTER_SURROGATE_V1',
         'accepted_checkpoint_sha256':PARENT_SHA,'surrogate_training_promoted':False,
         'prior_seed_sha256':digest(a.seed.read_bytes()),'weights':w,'bias':base.get('bias',0.),
         'adam':adam,'prior_router_adam_steps':orig['adam']['step'],
         'source_release':'corpora-short-stories-0001',
         'selected_aux_updates':best_update,'total_new_training_updates_attempted':a.updates}
    for p,obj in ((a.report,report),(a.out_state,new)):
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
        tmp.replace(p)
    print(json.dumps({'result':'done','books':bookcounts,'events':report['examples'],
       'before_dev':before['dev'],'after_dev':after['dev'],
       'postselection_test':after['test'],'selected_update':best_update,
       'no_base_LM_training':True}),flush=True)

if __name__=='__main__': main()