"""v96: train a response field and evaluate genuine autoregressive fusion."""
import sys,pathlib,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/nancm_generate95'))
from pilot95 import *
from generate95 import events,draw
from field96 import Chat96,load_data,train96,EOT
import re

@torch.no_grad()
def generate96(base,chat,prompt,limit,seed,temp,gain,reference=None):
    rng=np.random.default_rng(seed);src=prompt.encode();root=b'';p=b'';e=None;t=time.perf_counter()
    for prefix,event in events(src):
        assert prefix==root
        if not re.search(rb'[A-Za-z]+',root):
            root+=event;continue
        e=base.build(root);p=event
        if e.done(p):base.observe(e,p);root+=p;p=b'';e=None
    assert root+p==src
    state,anchor,c=chat.prefill(src)
    out=bytearray();bits=0.;base_bits=0.;chat_bits=0.;stop='byte_limit'
    for i in range(limit):
        if e is None:e=base.build(root)
        bq=e.next(p);q=(1-gain)*bq+gain*c
        assert q.shape==(256,) and abs(q.sum()-1)<1e-8 and q.min()>0
        b=reference[i] if reference is not None else draw(q,rng,temp,1. if temp==0 else .95)
        bits-=math.log2(float(q[b]));base_bits-=math.log2(float(bq[b]));chat_bits-=math.log2(float(c[b]))
        out.append(b)
        if b==EOT:
            stop='learned_eot';break
        p+=bytes([b])
        if e.done(p):base.observe(e,p);root+=p;p=b'';e=None
        state,c=chat.advance(state,anchor,b)
        if reference is None and (out.endswith(b'\nUser:') or out.endswith(b'\nAssistant:')):
            stop='role_boundary';break
    visible=bytes(out[:-1] if stop=='learned_eot' else out)
    text=visible.decode('utf-8',errors='replace');w=re.findall(r'[A-Za-z]+',text.lower());tri=list(zip(w,w[1:],w[2:]))
    return {'prompt':prompt,'continuation':text,'continuation_hex':bytes(out).hex(),
      'bytes':len(out),'words':len(w),'elapsed_seconds':time.perf_counter()-t,
      'temperature':temp,'chat_gain':gain,'seed':seed,'stop_reason':stop,
      'sampled_mean_bits':bits/max(1,len(out)),'base_mean_bits':base_bits/max(1,len(out)),
      'chat_mean_bits':chat_bits/max(1,len(out)),'repeated_trigram_fraction':1-len(set(tri))/len(tri) if tri else 0.,
      'printable_ascii_fraction':sum(32<=b<127 or b in (9,10,13) for b in visible)/max(1,len(visible)),
      'teacher_forced':reference is not None}

def main():
    ap=argparse.ArgumentParser()
    for k in ('data','out','checkpoint','analogy','contrast','screen','transition','transition-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
    ap.add_argument('--v71',type=pathlib.Path,required=True)
    ap.add_argument('--v71-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v72',type=pathlib.Path,required=True)
    ap.add_argument('--v72-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v73',type=pathlib.Path,required=True)
    ap.add_argument('--v73-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v74',type=pathlib.Path,required=True)
    ap.add_argument('--v74-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v75',type=pathlib.Path,required=True)
    ap.add_argument('--v75-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v76',type=pathlib.Path,required=True)
    ap.add_argument('--v76-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v77',type=pathlib.Path,required=True)
    ap.add_argument('--v77-metrics',type=pathlib.Path,required=True)
    ap.add_argument('--v78',type=pathlib.Path,required=True)
    ap.add_argument('--v78-metrics',type=pathlib.Path,required=True)
    for k in ('v79','v79-metrics','v81','v81-metrics','v86','v86-metrics','v88','v88-metrics','v94','v94-metrics'):ap.add_argument('--'+k,type=pathlib.Path,required=True)
    ap.add_argument('--native',action='store_true')
    keys=('lib','parent','recurrent','direct','dual','symbiotic','fusion','coordinate','temporal','relational','lexical','byte','subword','resonance','attractor','unified')
    for k in keys:ap.add_argument('--'+k,type=pathlib.Path)
    ap.add_argument('--prompt',type=str);ap.add_argument('--max-bytes',type=int,default=96)
    ap.add_argument('--chat-data',type=pathlib.Path,required=True);ap.add_argument('--chat-steps',type=int,default=768)
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
    # Only original TRAIN references, never calibration or screened DEV.
    train_ids={i for i,p in tr+extra};eval_ids={i for group in selected.values() for i,p in group}
    assert not train_ids&eval_ids
    train_sha={hashlib.sha256(p.read_bytes()).hexdigest() for i,p in tr+extra}
    eval_sha={hashlib.sha256(p.read_bytes()).hexdigest() for group in selected.values() for i,p in group}
    assert not train_sha&eval_sha
    trajectory=PathField(index.words,index.code,expanded)
    out['reference']={'books':len(train_ids),'paths':len(trajectory.paths),'dimensions':trajectory.code.shape[1],'train_eval_ids_disjoint':True,'train_eval_sha_disjoint':True}
    write_metrics(a.out,out,'reference')
    fitbooks=(primary+secondary)[:24]
    assert len(fitbooks)>=8 and not {i for i,p in fitbooks}&(train_ids|eval_ids)
    assert not {hashlib.sha256(p.read_bytes()).hexdigest() for i,p in fitbooks}&(train_sha|eval_sha)
    # Restore the actual v71 trained weights and optimizer state; never refit
    # them on DEV. Only the new generative field is optimized here.
    ck71=torch.load(a.v71,map_location='cpu',weights_only=True);m71=json.loads(a.v71_metrics.read_text())
    assert ck71['updates']==256 and ck71['parent_artifact_hashes']==hashes and ck71['transition_sha256']==thash
    assert m71['report_stage']=='final' and m71['calibration']['selected_gain']==.5
    net=Discriminator(trajectory.code.shape[1]);net.load_state_dict(ck71['discriminator']);net.eval();net.requires_grad_(False)
    v71sha=hashlib.sha256(a.v71.read_bytes()).hexdigest()
    ck72=torch.load(a.v72,map_location='cpu',weights_only=True);m72=json.loads(a.v72_metrics.read_text())
    assert ck72['updates']==512 and ck72['parent_artifact_hashes']==hashes and ck72['transition_sha256']==thash
    assert ck72['v71_sha256']==v71sha and m72['report_stage']=='final'
    v72sha=hashlib.sha256(a.v72.read_bytes()).hexdigest()
    ck73=torch.load(a.v73,map_location='cpu',weights_only=True);m73=json.loads(a.v73_metrics.read_text())
    assert ck73['updates']==896 and ck73['parent_artifact_hashes']==hashes and ck73['transition_sha256']==thash
    assert ck73['v72_sha256']==v72sha and ck73['v71_sha256']==v71sha
    assert m73['report_stage']=='replay_failed' and m73['fit']['generator_optimizer_step']==896
    v73sha=hashlib.sha256(a.v73.read_bytes()).hexdigest()
    ck74=torch.load(a.v74,map_location='cpu',weights_only=True);m74=json.loads(a.v74_metrics.read_text())
    assert ck74['updates']==1152 and ck74['parent_artifact_hashes']==hashes and ck74['transition_sha256']==thash
    assert ck74['v73_sha256']==v73sha and ck74['v71_sha256']==v71sha
    assert m74['report_stage']=='final'
    v74sha=hashlib.sha256(a.v74.read_bytes()).hexdigest()
    ck75=torch.load(a.v75,map_location='cpu',weights_only=True);m75=json.loads(a.v75_metrics.read_text())
    assert ck75['updates']==1536 and ck75['parent_artifact_hashes']==hashes and ck75['transition_sha256']==thash
    assert ck75['v74_sha256']==v74sha and ck75['v71_sha256']==v71sha
    assert m75['report_stage']=='final' and m75['calibration']['selected_gain']==.0625
    v75sha=hashlib.sha256(a.v75.read_bytes()).hexdigest()
    ancestry=restore75(index,ck75)
    contexts=ContextField(index.words,index.seq,model.cls.cpu().numpy())
    out['context_index']={'exact_positions':len(contexts.exact.pos),'semantic_positions':len(contexts.semantic.pos),'exact_order':8,'semantic_order':6,'books':len(train_ids),'collision_free':True}
    write_metrics(a.out,out,'context_index')
    ck76=torch.load(a.v76,map_location='cpu',weights_only=True);m76=json.loads(a.v76_metrics.read_text())
    assert ck76['v75_sha256']==v75sha and ck76['updates']==256 and m76['report_stage']=='final'
    assert m76['calibration']['selected_gain']==.5 and ck76['parent_artifact_hashes']==hashes
    v76sha=hashlib.sha256(a.v76.read_bytes()).hexdigest()
    gate=Gate();gate.load_state_dict(ck76['gate']);gate.eval();gate.requires_grad_(False)
    ck77=torch.load(a.v77,map_location='cpu',weights_only=True);m77=json.loads(a.v77_metrics.read_text())
    assert ck77['v76_sha256']==v76sha and ck77['updates']==384 and m77['report_stage']=='final'
    assert ck77['parent_artifact_hashes']==hashes and ck77['transition_sha256']==thash
    oldgain=float(m77['calibration']['selected_gain'])
    v77sha=hashlib.sha256(a.v77.read_bytes()).hexdigest()
    competition=Competition();competition.load_state_dict(ck77['competition']);competition.eval();competition.requires_grad_(False)
    ck78=torch.load(a.v78,map_location='cpu',weights_only=True);m78=json.loads(a.v78_metrics.read_text())
    assert ck78['v77_sha256']==v77sha and ck78['updates']==4096 and m78['report_stage']=='final'
    assert ck78['parent_artifact_hashes']==hashes and ck78['transition_sha256']==thash
    assert m78['calibration']['selected_gain']==.25
    v78sha=hashlib.sha256(a.v78.read_bytes()).hexdigest()
    address=Address(index.code,index.prior,model.members,model.cls,model.slot,ancestry[0].lex.weight.detach())
    address.load_state_dict(ck78['address']);address.eval();address.requires_grad_(False)
    ck79=torch.load(a.v79,map_location='cpu',weights_only=True);m79=json.loads(a.v79_metrics.read_text())
    ck81=torch.load(a.v81,map_location='cpu',weights_only=True);m81=json.loads(a.v81_metrics.read_text())
    assert ck79['v78_sha256']==ck81['v78_sha256']==v78sha and ck79['updates']==512 and ck81['updates']==256
    assert m79['report_stage']==m81['report_stage']=='final'
    for child in (ck79,ck81):
        assert child['parent_artifact_hashes']==hashes and child['transition_sha256']==thash
    v79sha=hashlib.sha256(a.v79.read_bytes()).hexdigest();v81sha=hashlib.sha256(a.v81.read_bytes()).hexdigest()
    escape=Escape();escape.load_state_dict(ck79['escape']);escape.eval();escape.requires_grad_(False)
    gate81=Gate81();gate81.load_state_dict(ck81['gate81']);gate81.eval();gate81.requires_grad_(False)
    field76.Suffix=FastSuffix;field79.Suffix=FastSuffix
    contexts=ContextField(index.words,index.seq,model.cls.cpu().numpy())
    seqs=[np.array(index.encode(B.load(p,20000)),np.int32) for _,p in tr+extra]
    evidence86=Evidence(index,seqs,model.cls.cpu().numpy())
    frozen=FrozenAddress(address)
    out['integration']={'v79_sha256':v79sha,'v81_sha256':v81sha,'reference_words':sum(map(len,seqs)),'reference_books':len(seqs),'v79_historical_replay':m79['historical_replay_verified'],'v81_historical_replay':m81['historical_replay_verified'],'no_retraining':True}
    write_metrics(a.out,out,'integration')
    ck86=torch.load(a.v86,map_location='cpu',weights_only=True);m86=json.loads(a.v86_metrics.read_text())
    assert m86['report_stage']=='final' and ck86['v78_sha256']==v78sha and ck86['v79_sha256']==v79sha and ck86['v81_sha256']==v81sha
    inherited=tuple(ck86['selected_arm']);v86sha=hashlib.sha256(a.v86.read_bytes()).hexdigest()
    assert inherited==('exact',1.,1.)
    escape.load_state_dict(ck86['v79_escape']);gate81.load_state_dict(ck86['v81_gate'])
    ck88=torch.load(a.v88,map_location='cpu',weights_only=True);m88=json.loads(a.v88_metrics.read_text())
    assert m88['report_stage']=='final' and ck88['v86_sha256']==v86sha and ck88['updates']==512 and ck88['selected_gain']==.125
    v88sha=hashlib.sha256(a.v88.read_bytes()).hexdigest()
    old=Field88();old.load_state_dict(ck88['field88']);old.eval().requires_grad_(False)
    ck94=torch.load(a.v94,map_location='cpu',weights_only=True);m94=json.loads(a.v94_metrics.read_text())
    assert m94['report_stage']=='final' and ck94['updates']==384 and ck94['v88_sha256']==v88sha
    trust=Trust94(Field88());trust.load_state_dict(ck94['trust94']);trust.eval().requires_grad_(False)
    gain=float(ck94['selected_gain']);assert gain==float(m94['calibration']['selected_gain'])
    train,dev,manifest=load_data(a.chat_data)
    # Fixed benchmark instructions are excluded if an exact normalized
    # instruction match occurs in the external corpus.
    fixed=set()
    for _,prompt in PROMPTS:
        for instruction in re.findall(r'(?:^|\n)User: (.*?)\nAssistant:',prompt):
            fixed.add(hashlib.sha256(' '.join(instruction.lower().split()).encode()).hexdigest())
    before=len(train);train=[r for r in train if r[2] not in fixed];dev=[r for r in dev if r[2] not in fixed]
    manifest['fixed_prompt_matches_removed']=before-len(train)
    manifest['train_records_after_exclusion']=len(train);manifest['dev_records_after_exclusion']=len(dev)
    out['chat_data']=manifest;write_metrics(a.out,out,'chat_data')
    torch.manual_seed(9600);chat=Chat96();opt,metrics=train96(chat,train,dev,a.chat_steps)
    chat.eval().requires_grad_(False);out['chat_fit']=metrics
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'v94':ck94,'v94_sha256':hashlib.sha256(a.v94.read_bytes()).hexdigest(),'chat96':chat.state_dict(),'optimizer96':opt.state_dict(),'chat_data':manifest,'chat_steps':a.chat_steps},a.out.with_suffix('.pt'))
    write_metrics(a.out,out,'chat_fit')
    def factory():
        return Generator95(args,raw,surface,core,float(tc['scale']),trajectory,net,ancestry,contexts,gate,competition,oldgain,frozen,evidence86,escape,gate81,inherited,trust,gain)
    # Held-out prompt/response scoring. No generated answer is selected using
    # these targets. Mixture gain is selected only on this calibration subset.
    rng=np.random.default_rng(9604);order=rng.permutation(len(dev));cal=[dev[i] for i in order[:4]]
    scores={}
    for g in (0.,.5,.75,.9,1.):
        bits=0.;n=0
        for prompt,answer,*_ in cal:
            x=generate96(factory(),chat,prompt.decode(),len(answer),9601,0.,g,answer)
            bits+=x['sampled_mean_bits']*x['bytes'];n+=x['bytes']
        scores[str(g)]=bits/n
    selected=float(min((k for k in scores if float(k)<=.9),key=scores.get))
    out['chat_calibration']={'records':len(cal),'response_bpb':scores,'selected_chat_gain':selected,'unrestricted_best_gain':float(min(scores,key=scores.get))}
    write_metrics(a.out,out,'chat_calibration')
    out['generation']={'results':[],'selected_chat_gain':selected}
    prompts=[('custom',a.prompt)] if a.prompt is not None else PROMPTS+[(f'heldout_{i}',dev[j][0].decode()) for i,j in enumerate(order[4:6])]
    for label,prompt in prompts:
        for temp in (0.,.8):
            x=generate96(factory(),chat,prompt,a.max_bytes,9601,temp,selected)
            x.update(model='v96',prompt_id=label)
            out['generation']['results'].append(x)
            print('GENERATION96 '+json.dumps(x,ensure_ascii=True),flush=True)
            write_metrics(a.out,out,'generation:'+label)
    # A second calibration-independent probe, no oracle answers during
    # free generation; separate teacher-forced reporting.
    held=[]
    for j in order[6:10]:
        prompt,answer,*_=dev[j]
        x=generate96(factory(),chat,prompt.decode(),len(answer),9601,0.,selected,answer)
        held.append({k:x[k] for k in ('bytes','sampled_mean_bits','base_mean_bits','chat_mean_bits')})
    out['chat_holdout']={'records':held,'aggregate_bpb':{k:sum(r[k]*r['bytes'] for r in held)/sum(r['bytes'] for r in held) for k in ('sampled_mean_bits','base_mean_bits','chat_mean_bits')}}
    assert hashlib.sha256(a.v86.read_bytes()).hexdigest()==v86sha
    assert hashlib.sha256(a.v88.read_bytes()).hexdigest()==v88sha
    out['parent_artifacts_unchanged']=True
    torch.save({'v94':ck94,'v94_sha256':hashlib.sha256(a.v94.read_bytes()).hexdigest(),'chat96':chat.state_dict(),'optimizer96':opt.state_dict(),'chat_data':manifest,'chat_steps':a.chat_steps,'selected_chat_gain':selected},a.out.with_suffix('.pt'))
    write_metrics(a.out,out)
if __name__=='__main__':main()
