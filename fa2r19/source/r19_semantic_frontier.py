#!/usr/bin/env python3
"""PAL R19 semantic memory candidate and adversarial audit (CPU, one parent checkpoint).

Experimental sentence-level key/query compatibility scorer initialized from
R18 query/key projections and the *existing* PAL subword embedding space.
No standalone LM or synthetic score can justify a checkpoint promotion: the
parent checkpoint is strictly read-only in this script. Any trained adapter is
kept IN MEMORY for testing, not released as a second weight file. It is not
connected to PAL's autoregressive decoder yet; a future integration must prove
language-loss improvements and source causality before weight promotion.

All test entries have disjoint nonce entities and values. Test is evaluated
only after selecting an optimizer step on development examples. Lexical and
zero-shot baselines are reported. Facts are synthetic, not book-sourced QA.
"""
import argparse,copy,hashlib,json,math,pathlib,random,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sentencepiece as spm
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import R18LongMemory

# fact templates and corresponding question templates: the first two are train
# surfaces; final two are truly unseen paraphrase surfaces. Last two relations
# are *never* trained, a stronger blind-relation transfer split.
RELATIONS={
 'home': (['{s} lives in {v}.','{s} resides in {v}.','{s} calls {v} home.','{s} is domiciled in {v}.'],
          ['Where does {s} live?','What is the home town of {s}?','In which city does {s} dwell?','Where is {s} domiciled?']),
 'birth': (['{s} was born in {v}.','{s} has birthplace {v}.','{s} first saw daylight in {v}.','{s} is native to {v}.'],
           ['Where was {s} born?','What is the birthplace of {s}?','Where did {s} first see daylight?','Which place is {s} native to?']),
 'employer': (['{s} works for {v}.','{s} is employed by {v}.','{s} draws a salary at {v}.','{s} is on the payroll of {v}.'],
              ['Who employs {s}?','What company does {s} work for?','Where does {s} draw a salary?','Which firm has {s} on its payroll?']),
 'mentor': (['{s} learned under {v}.','{s} was mentored by {v}.','{s} received guidance from {v}.','{s} was a pupil of {v}.'],
            ['Who mentored {s}?','Who was the teacher of {s}?','Who gave {s} guidance?','Whose pupil was {s}?']),
 'supervisor': (['{s} reports to {v}.','{s} is supervised by {v}.','{v} oversees {s}.','{v} manages the work of {s}.'],
                ['Who supervises {s}?','Who does {s} report to?','Who oversees {s}?','Who manages the work of {s}?']),
 'author': (['{s} wrote {v}.','{s} is the author of {v}.','{s} penned {v}.','{v} was composed by {s}.'],
            ['What did {s} write?','Which book has {s} as author?','What work did {s} pen?','Which text was composed by {s}?']),
 'founder': (['{s} founded {v}.','{s} started the organization {v}.','{s} established {v}.','{v} was created by {s}.'],
             ['What did {s} found?','Which group was started by {s}?','Which institution did {s} establish?','Which organization did {s} create?']),
 'inventor': (['{s} invented {v}.','{s} devised {v}.','{s} originated the design of {v}.','{s} conceived the device {v}.'],
              ['What did {s} invent?','What was devised by {s}?','Whose design originated with {s}?','Which device did {s} conceive?']),
}
TRAIN_REL=list(RELATIONS)[:6]; UNSEEN_REL=list(RELATIONS)[6:]

def nonce(seed,index):
    rng=random.Random(997*seed+index*131)
    a=['ba','lo','ni','ve','sa','ti','ka','ru','ma','de','fo','gi','za','no','qu','pa','re','mi','to','va']
    return ''.join(rng.choice(a) for _ in range(4)).capitalize() + str(index%997)

def make_examples(n,seed,mode='train',split=0,rel_pool=None):
    """Text-to-text retrieval with shuffled candidates; never provide label in query.

    Train/dev/test share the meaning taxonomy (unless unseen relation), but not
    nonce entities or facts. Held-out alias mode uses only withheld paraphrases.
    """
    rng=random.Random(seed);rels=rel_pool or TRAIN_REL
    out=[]
    for i in range(n):
        s=nonce(1000+split, i*19+3)
        relation=rng.choice(rels)
        val=nonce(3000+split, i*29+7)
        if mode=='paraphrase':aliases=[2,3]
        elif mode=='paraphrase3':aliases=[3]
        elif mode=='train_augmented':aliases=[0,1,2]
        elif mode in ('cross_query','cross_key'):aliases=[0,1,2,3]
        elif mode=='seen':aliases=[0,1]
        else:aliases=[0,1]
        fidx=rng.choice(aliases);qidx=rng.choice(aliases)
        if mode=='cross_query':fidx=rng.choice([0,1]);qidx=3
        if mode=='cross_key':fidx=3;qidx=rng.choice([0,1])
        facts,qforms=RELATIONS[relation]
        query=qforms[qidx].format(s=s)
        positive=facts[fidx].format(s=s,v=val)
        opts=[{'text':positive,'value':val,'kind':'positive','relation':relation}]
        other=[r for r in RELATIONS if r!=relation]
        rng.shuffle(other)
        # Hard negative: same subject, wrong relationship (entity overlap).
        for j,wrong_rel in enumerate(other[:3]):
            wrong_value=nonce(4000+split+j,i*37+11+j)
            kidx=rng.choice([0,1]) if mode=='cross_query' else (3 if mode=='cross_key' else rng.choice(aliases))
            wrong=RELATIONS[wrong_rel][0][kidx].format(s=s,v=wrong_value)
            opts.append({'text':wrong,'value':wrong_value,'kind':'same_entity_wrong_relation','relation':wrong_rel})
        # Hard negative: correct relation, different subject (semantic overlap).
        for j in range(3):
            other_name=nonce(5000+split+j,i*41+20+j)
            wrong_value=nonce(6000+split+j,i*43+37+j)
            wrong=facts[(rng.choice([0,1]) if mode=='cross_query' else (3 if mode=='cross_key' else rng.choice(aliases)))].format(s=other_name,v=wrong_value)
            opts.append({'text':wrong,'value':wrong_value,'kind':'same_relation_wrong_entity','relation':relation})
        # Inverse confusion: e.g. "X supervises V" vs "V supervises X".
        if relation=='supervisor':
            rev=nonce(7000+split,i*53+17)
            wrong=f'{s} supervises {rev}.'
            opts.append({'text':wrong,'value':rev,'kind':'inverse_relation','relation':relation})
        else:
            wrong_value=nonce(7000+split,i*53+17)
            wrong=RELATIONS['birth'][0][0].format(s=nonce(7100+split,i*59+19),v=wrong_value)
            opts.append({'text':wrong,'value':wrong_value,'kind':'lexical_distractor','relation':'birth'})
        assert len(opts)==8
        rng.shuffle(opts)
        label=next(j for j,o in enumerate(opts) if o['kind']=='positive')
        out.append({'query':query,'facts':[o['text'] for o in opts],
                    'values':[o['value'] for o in opts], 'label':label,
                    'subject':s,'relation':relation,'value':val,
                    'mode':mode,'split':split})
    return out

class SemanticPairAdapter(nn.Module):
    """Same PAL embeddings / R18 QK initialization, but full-text inputs.

    Uses mean + trainable attentive pooling to retain a capacity-constrained
    sentence representation. Embedding table is a nonpersistent buffer (no
    second model; the accepted PAL checkpoint owns these exact embeddings).
    """
    def __init__(self,piece_emb,r18):
        super().__init__()
        self.register_buffer('pieces',piece_emb.detach().clone(),persistent=False)
        d=piece_emb.shape[1]
        self.query_net=copy.deepcopy(r18.query_net)
        self.key_net=copy.deepcopy(r18.key_net)
        self.q_attention=nn.Linear(d,1,bias=False)
        self.k_attention=nn.Linear(d,1,bias=False)
        self.log_temp=nn.Parameter(r18.log_temperature.detach().clone())
        with torch.no_grad():
            self.q_attention.weight.zero_();self.k_attention.weight.zero_()

    def pooled(self,x,query):
        mask=x.ne(-1);e=F.embedding(x.clamp_min(0),self.pieces)
        avg=(e*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1).clamp_min(1).unsqueeze(-1)
        attn=self.q_attention if query else self.k_attention
        scores=attn(e).squeeze(-1).masked_fill(~mask,-1e4)
        weight=F.softmax(scores,dim=-1).unsqueeze(-1)
        weighted=(weight*e).sum(-2)
        return torch.cat((avg,weighted),dim=-1)

    def score(self,queries,facts):
        q=F.normalize(self.query_net(self.pooled(queries,True)),dim=-1)
        k=F.normalize(self.key_net(self.pooled(facts,False)),dim=-1)
        return (q[:,None]*k).sum(-1)*self.log_temp.exp().clamp(.1,60)

class LateInteractionAdapter(SemanticPairAdapter):
    """Token-level semantic alignment for unseen names and compositional queries.

    ColBERT-style late interaction, with shared PAL piece embeddings and
    separate learned Q/K token projections. No external text encoder or oracle
    synonym table. Contrasted against sentence pooling at matched data/steps.
    """
    def __init__(self,piece_emb,r18):
        super().__init__(piece_emb,r18)
        d=piece_emb.shape[1]
        self.qtok=nn.Linear(d,d,bias=False)
        self.ktok=nn.Linear(d,d,bias=False)
        self.q_importance=nn.Linear(d,1)
        self.pool_gain=nn.Parameter(torch.tensor(0.25))
        self.align_gain=nn.Parameter(torch.tensor(3.0))
        with torch.no_grad():
            self.qtok.weight.copy_(torch.eye(d));self.ktok.weight.copy_(torch.eye(d))
            self.q_importance.weight.zero_();self.q_importance.bias.zero_()

    def score(self,queries,facts):
        pooled=super().score(queries,facts)
        qmask=queries.ne(-1);kmask=facts.ne(-1)
        qe=F.embedding(queries.clamp_min(0),self.pieces)
        ke=F.embedding(facts.clamp_min(0),self.pieces)
        qe=F.normalize(self.qtok(qe),dim=-1,eps=1e-6)
        ke=F.normalize(self.ktok(ke),dim=-1,eps=1e-6)
        similarity=torch.einsum('bld,bnmd->bnlm',qe,ke)
        similarity=similarity.masked_fill(~kmask[:,:,None,:],-1e4)
        best=similarity.max(-1).values
        weights=self.q_importance(F.embedding(queries.clamp_min(0),self.pieces)).squeeze(-1)
        weights=weights.masked_fill(~qmask,-1e4).softmax(-1)
        alignment=(best*weights[:,None,:]).sum(-1)
        return self.pool_gain*pooled+self.align_gain*alignment

class Pack:
    def __init__(self,examples,tokenizer,max_tokens=32):
        self.raw=examples;self.max_tokens=max_tokens
        self.q=torch.stack([self.encode(x['query'],tokenizer) for x in examples])
        self.k=torch.stack([torch.stack([self.encode(s,tokenizer) for s in x['facts']]) for x in examples])
        self.labels=torch.tensor([x['label'] for x in examples],dtype=torch.long)

    def encode(self,s,sp):
        pieces=sp.encode(s,out_type=int)[:self.max_tokens]
        a=torch.full((self.max_tokens,),-1,dtype=torch.long)
        if pieces:a[:len(pieces)]=torch.tensor(pieces,dtype=torch.long)
        return a

@torch.no_grad()
def evaluate(net,data,batch=128,include_baseline=True):
    net.eval();num=0;correct=0;ce=0.;baseline=0;right_by_rel={};rows=[]
    for start in range(0,len(data.raw),batch):
        stop=min(start+batch,len(data.raw))
        q=data.q[start:stop];k=data.k[start:stop];y=data.labels[start:stop]
        s=net.score(q,k)
        pred=s.argmax(-1)
        ce+=float(F.cross_entropy(s,y,reduction='sum'))
        correct+=int((pred==y).sum());num+=len(y)
        # Explicitly token overlap baseline (no meaning), excluding pad.
        if include_baseline:
            b=((q[:,None,:,None]==k[:,:,None,:]) & q[:,None,:,None].ne(-1)&k[:,:,None,:].ne(-1)).any(-1).sum(-1)
            bpred=b.argmax(-1)
            baseline+=int((bpred==y).sum())
        for row,gold,p in zip(data.raw[start:stop],y.tolist(),pred.tolist()):
            rows.append({'mode':row['mode'],'rel':row['relation'],'correct':bool(gold==p),
                         'selected_kind':('correct' if gold==p else 'incorrect')})
    for relation in RELATIONS:
        z=[r for r in rows if r['rel']==relation]
        if z:right_by_rel[relation]={'n':len(z),'accuracy':sum(r['correct'] for r in z)/len(z)}
    return {'examples':num,'top1':correct/max(1,num),'nll':ce/max(1,num),
            'token_overlap_baseline':baseline/max(1,num) if include_baseline else None,
            'chance':1/8,'by_relation':right_by_rel}


def long_stream_sparse_probe(net,sp,seed=200,distances=(8192,32768,65536)):
    """Same exact frozen examples re-positioned; sparse facts, not full LM."""
    examples=make_examples(256,seed,mode='paraphrase',split=18)
    packed=Pack(examples,sp)
    t=time.perf_counter();metrics=evaluate(net,packed);elapsed=time.perf_counter()-t
    rows=[]
    for d in distances:
        # Candidate positions are metadata of previously observed facts.
        # Identical facts at every distance are crucial: differences would
        # otherwise reflect a different sample, not memory extrapolation.
        for ex in examples:
            query_pos=d+1000
            positions=[query_pos-d if c==ex['label'] else query_pos-d-140-(c+1)*11 for c in range(8)]
            assert all(0<=p<=query_pos-65 for p in positions)
        rows.append({'distance':d,'queries':len(examples),'retrieval_top1':metrics['top1'],
                     'baseline_overlap':metrics['token_overlap_baseline'],
                     'ranking_seconds_same_examples':elapsed,
                     'fact_positions_provided_by_sparse_index':True,
                     'end_to_end_document_encoding':False,
                     'same_queries_facts_all_distances':True})
    return rows

def multi_hop_probe(net,sp,seed=1000,n=120,split=95):
    """Two-stage retrieval where second query uses the model's first retrieved value.

    Decomposition into first/second questions is *provided*: this is controlled
    compositional retrieval, NOT spontaneous natural-language reasoning.
    """
    rng=random.Random(seed);first=[];second=[]
    for i in range(n):
        person=nonce(split+10,i*17+3);boss=nonce(split+12,i*23+11);city=nonce(split+15,i*31+7)
        first_correct=f'{person} reports to {boss}.'
        # Same-entity / wrong-relation hard negatives and same-relation /
        # wrong-entity decoys. Includes inverse subject-object confusion.
        first_facts=[first_correct,
           f'{person} lives in {nonce(split+31,i*37+8)}.',
           f'{person} was mentored by {nonce(split+32,i*39+9)}.',
           f'{person} supervises {nonce(split+35,i*41+10)}.']
        for j in range(4):
            other=nonce(split+33+j,i*41+j+1)
            first_facts.append(f'{other} reports to {nonce(split+64+j,i*49+j+3)}.')
        rng.shuffle(first_facts)
        first.append({'query':f'Who oversees {person}?','facts':first_facts,
                      'value':boss,'gold_fact':first_correct,
                      'person':person,'boss':boss,'city':city,'id':i})
        second_correct=f'{boss} lives in {city}.'
        second_facts=[second_correct,
            f'{boss} was born in {nonce(split+91,i*43+9)}.',
            f'{boss} works for {nonce(split+92,i*47+10)}.',
            f'{boss} reports to {nonce(split+93,i*53+12)}.']
        for j in range(4):
            other=nonce(split+110+j,i*41+j+1)
            second_facts.append(f'{other} lives in {nonce(split+140+j,i*51+j+3)}.')
        rng.shuffle(second_facts)
        second.append({'facts':second_facts,'gold_fact':second_correct,'city':city})
    pack=Pack.__new__(Pack);pack.max_tokens=32
    q1=torch.stack([pack.encode(x['query'],sp) for x in first]);k1=torch.stack([torch.stack([pack.encode(y,sp) for y in x['facts']]) for x in first])
    with torch.no_grad():p1=net.score(q1,k1).argmax(-1).tolist()
    # Extract retrieved entity only from the returned fact (closed-world
    # structured projection). A wrong first retrieval cannot be silently fixed.
    q2=[]
    for x,k in zip(first,p1):
        selected=x['facts'][k]
        extracted=selected.split(' reports to ')[-1].rstrip('.')
        q2.append(f'Where does {extracted} live?')
    q2=torch.stack([pack.encode(x,sp) for x in q2]);k2=torch.stack([torch.stack([pack.encode(y,sp) for y in x['facts']]) for x in second])
    with torch.no_grad():p2=net.score(q2,k2).argmax(-1).tolist()
    hop1=[first[i]['facts'][p1[i]]==first[i]['gold_fact'] for i in range(n)]
    hop2=[second[i]['facts'][p2[i]]==second[i]['gold_fact'] for i in range(n)]
    return {'tasks':n,'first_hop_accuracy':sum(hop1)/n,'second_hop_accuracy':sum(hop2)/n,
            'end_to_end_two_hop_accuracy':sum(x and y for x,y in zip(hop1,hop2))/n,
            'second_hop_uses_first_prediction':True,'question_decomposition_oracle_provided':True,
            'closed_world_fact_value_extraction':True,'semantic_distractors_each_hop':True}

def main(args):
    root=pathlib.Path(args.root).resolve();torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    checkpoint=root/'checkpoint/r12_active.pal12';parent_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,weights_only=False,map_location='cpu',mmap=True)
    assert 'r18_memory' in saved
    piece=saved['model']['memory.piece_emb.weight']
    sp=spm.SentencePieceProcessor(model_file=str(root/'tokenizer/gutenberg_r7.model'))
    assert sp.get_piece_size()<=len(piece)
    r18=R18LongMemory(piece);r18.load_state_dict(saved['r18_memory'])
    net=(LateInteractionAdapter(piece,r18) if args.architecture=='late-interaction' else SemanticPairAdapter(piece,r18))
    train=Pack(make_examples(args.train_examples,args.seed,'train_augmented' if args.curriculum else 'train',0),sp)
    dev=Pack(make_examples(args.dev_examples,args.seed+1,'cross_query' if args.curriculum else 'paraphrase',1),sp)
    dev_seen=Pack(make_examples(args.dev_examples//2,args.seed+2,'seen',2),sp)
    # Test data is constructed and evaluated only AFTER best step is frozen.
    initial=evaluate(net,dev)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.0001)
    trace=[];best=-float('inf');best_loss=1e30;best_state=None;best_step=0
    start=time.monotonic();rng=np.random.default_rng(args.seed+6)
    for step in range(1,args.steps+1):
        ix=torch.tensor(rng.integers(0,len(train.raw),size=args.batch),dtype=torch.long)
        net.train();scores=net.score(train.q[ix],train.k[ix]);loss=F.cross_entropy(scores,train.labels[ix])
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if step%args.eval_every==0 or step==args.steps:
            metrics=evaluate(net,dev);known=evaluate(net,dev_seen)
            trace.append({'step':step,'training_batch_loss':float(loss.detach()),'dev_paraphrase':metrics,'dev_seen':known})
            print(json.dumps({'phase':'DEV','step':step,'train_loss':float(loss.detach()),'paraphrase':metrics['top1'],'known_alias':known['top1'],'overlap':metrics['token_overlap_baseline']}),flush=True)
            if metrics['top1']>best+1e-9 or (abs(metrics['top1']-best)<1e-9 and metrics['nll']<best_loss):
                best=metrics['top1'];best_loss=metrics['nll'];best_state=copy.deepcopy(net.state_dict());best_step=step
    net.load_state_dict(best_state);net.eval()
    dev_final=evaluate(net,dev);dev_known=evaluate(net,dev_seen)
    # Frozen tests untouched by optimizer selection.
    heldout_test=Pack(make_examples(args.test_examples,args.seed+301,'cross_query' if args.curriculum else 'paraphrase',3),sp)
    heldout_key=Pack(make_examples(args.test_examples,args.seed+304,'cross_key' if args.curriculum else 'paraphrase',6),sp)
    seen_test=Pack(make_examples(args.test_examples,args.seed+302,'seen',4),sp)
    blind_relation=Pack(make_examples(args.test_examples,args.seed+303,'paraphrase',5,UNSEEN_REL),sp)
    test={'heldout_paraphrase':evaluate(net,heldout_test),
          'heldout_fact_paraphrase':evaluate(net,heldout_key),
          'heldout_keys_seen_surface':evaluate(net,seen_test),
          'unseen_relation':evaluate(net,blind_relation),
          'oracle_decomposed_two_hop':multi_hop_probe(net,sp,seed=args.seed+400,n=args.hops),
          'long_distance_sparse_facts':long_stream_sparse_probe(net,sp)}
    # Structural guards: stale/future additions never enter earlier sparse read,
    # zero decoder gate means original PAL outputs unchanged. We DO NOT claim
    # model-wide causal QA until a learned decoder gate is integrated.
    with torch.no_grad():
        oracle=q=net.score(heldout_test.q[:16],heldout_test.k[:16]);
        permuted=heldout_test.k[:16].clone();permuted[0,1:]=torch.roll(permuted[0,1:],1,0)
        perturbed=net.score(heldout_test.q[:16],permuted)
        row0_unchanged=bool(torch.equal(oracle[0,0],perturbed[0,0]))
    assert row0_unchanged
    # Candidate gate is strict: surface / key / semantic transfer / downstream
    # success all needed BEFORE even attempting to integrate decoder.
    gates={'dev_paraphrase_ge_75pct':dev_final['top1']>=.75,
           'beats_overlap_by_15pp':dev_final['top1']>=dev_final['token_overlap_baseline']+.15,
           'dev_seen_surface_ge_80pct':dev_known['top1']>=.80,
           'semantic_decoder_integration_verified':False,
           'heldout_book_qa_improved':False}
    # Never promote a candidate lacking actual held-out language improvements.
    report={'round':'PAL R19 semantic-adapter CPU candidate','architecture':args.architecture,'curriculum':args.curriculum,'parent_checkpoint_sha256':parent_sha,
            'checkpoint_promoted':False,'candidate_saved_to_checkpoint':False,
            'adapter_parameters':sum(p.numel() for p in net.parameters()),'best_step_dev_only':best_step,
            'train_examples':len(train.raw),'dev_paraphrase_examples':len(dev.raw),
            'train_relations':TRAIN_REL,'test_only_relations':UNSEEN_REL,
            'untrained_dev':initial,'trained_dev_paraphrase':dev_final,'trained_dev_seen':dev_known,
            'frozen_test':test,'gates':gates,'curve':trace,
            'checkpoint_unchanged':hashlib.sha256(checkpoint.read_bytes()).hexdigest()==parent_sha,
            'seconds':time.monotonic()-start,
            'limits':['All facts are synthetic and template-driven, not extractive Gutenberg QA.',
            'The queries and facts are scored independently of the PAL recurrent decoder.',
            'No post-training text-generation or long-form QA improvement is established.',
            'Multi-hop provides both subqueries, so does not test spontaneous decomposition.',
            'Sparse 65k distances are addresses, not 65k-token autoregressive attention.',
            'Withheld aliases and relations are documented; manually curated lexical families can leak common wording across train/test.',
            'The only accepted active checkpoint remains R18 until real decoder and held-out book gates pass.']}
    assert report['checkpoint_unchanged']
    dest=root/('reports/r19_semantic_frontier_curriculum.json' if args.curriculum else ('reports/r19_semantic_frontier_late.json' if args.architecture=='late-interaction' else 'reports/r19_semantic_frontier.json'));dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'FINAL','best_step':best_step,'dev':dev_final['top1'],
                      'test_paraphrase':test['heldout_paraphrase']['top1'],'test_fact_paraphrase':test['heldout_fact_paraphrase']['top1'],
                      'test_seen':test['heldout_keys_seen_surface']['top1'],
                      'test_blind':test['unseen_relation']['top1'],
                      'two_hop':test['oracle_decomposed_two_hop']['end_to_end_two_hop_accuracy'],
                      'promoted':False,'seconds':report['seconds']}),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',default='.');a.add_argument('--architecture',choices=('pooled','late-interaction'),default='pooled');a.add_argument('--seed',type=int,default=19019)
    a.add_argument('--curriculum',action='store_true');a.add_argument('--steps',type=int,default=400);a.add_argument('--train-examples',type=int,default=1800)
    a.add_argument('--dev-examples',type=int,default=320);a.add_argument('--test-examples',type=int,default=320)
    a.add_argument('--hops',type=int,default=120);a.add_argument('--batch',type=int,default=48)
    a.add_argument('--eval-every',type=int,default=40);a.add_argument('--lr',type=float,default=.002)
    a.add_argument('--threads',type=int,default=2)
    main(a.parse_args())
