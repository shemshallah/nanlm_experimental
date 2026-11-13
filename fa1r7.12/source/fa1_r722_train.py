#!/usr/bin/env python3
"""R7.22 CPU semantic-gate pilot: only one R7 model, one fifth AdamW optimizer.

Historical TRAIN/DEV only, no historical TEST selection. Correct retrieval vs
frequency-only old-past and independent TRAIN donor controls. Report outcomes
and write a new unified candidate ONLY when all dev gates pass. No promotion.
"""
from __future__ import annotations
import argparse,copy,json,pathlib,random,time
import numpy as np,torch
import torch.nn.functional as F
from r7_core import Corpus
from fa1_r721_joint_train import gather_joint
from fa1_r718_far_residual import pick_positions
from fa1_r722_unified import load_unified,memory_trainable,save_unified
from fa1_r722_semantic import CausalSemanticIndex,HORIZON,EXCLUSION
from r710_checkpoint_integrity import deep_equal,sha256

def gather_sem(root,base,split,positions,seed,train_donor=False,corpus_dir=None):
    c=Corpus(root,split,corpus_dir=corpus_dir)
    src=Corpus(root,'train',corpus_dir=corpus_dir) if train_donor else c
    rng=np.random.default_rng(seed)
    pos=np.asarray(positions,dtype=np.int64)
    feats=[];prob=[];count=[];freq=[]
    donor_indices=(pick_positions(src,len(pos),seed+1) if corpus_dir else
                   rng.integers(HORIZON+1,len(src.seq),len(pos),dtype=np.int64)) if train_donor else pos
    ix=None;lastdoc=None
    for source_pos,actual_pos in zip(donor_indices,pos):
        source_pos=int(source_pos);actual_pos=int(actual_pos)
        doc=0 if src.doc_start is None else int(src.doc_start[source_pos])
        if ix is None or lastdoc!=doc or source_pos<ix.position or source_pos-ix.position>1024:
            ix=CausalSemanticIndex(src.seq,base.E,source_pos,document_start=doc)
        else:ix.advance(source_pos)
        lastdoc=doc
        target=int(c.seq[actual_pos])
        context=np.asarray(c.seq[actual_pos-4:actual_pos],dtype=np.int64)
        record=ix.query(target=target,context=context)
        feats.append(record['feature']);prob.append(record['target_probability']);count.append(record['n'])
        lo=max(doc,source_pos-HORIZON)
        hi=source_pos-EXCLUSION-1
        # A valid causal old-past unigram baseline; no query conditioning.
        old=src.seq[lo:hi]
        freq.append(float(np.mean(old==target)) if len(old) else 0.)
    return {'semantic_features':torch.tensor(np.asarray(feats),dtype=torch.float32),
            'semantic_prob':torch.tensor(prob,dtype=torch.float32),
            'semantic_count':torch.tensor(count,dtype=torch.float32),
            'frequency_prob':torch.tensor(freq,dtype=torch.float32),
            'nonempty':int(np.count_nonzero(count)),
            'target_supported':int(np.count_nonzero(np.asarray(prob)>0))}

@torch.no_grad()
def evaluation(gate,prior,valid,donor):
    correct,mass=gate(prior,valid['semantic_features'],valid['semantic_prob'],valid['semantic_count'])
    freq,_=gate(prior,valid['semantic_features'],valid['frequency_prob'],valid['semantic_count'])
    wrong,_=gate(prior,donor['semantic_features'],donor['semantic_prob'],donor['semantic_count'])
    return {'base_nll':float(prior.mean()),'correct_nll':float(correct.mean()),
            'gain':float((prior-correct).mean()),
            'correct_vs_old_freq_advantage':float((freq-correct).mean()),
            'correct_vs_train_donor_advantage':float((wrong-correct).mean()),
            'mean_gate_mass':float(mass.mean()),'nonempty':valid['nonempty'],
            'target_supported':valid['target_supported']}

def main(args):
    t=time.monotonic();torch.set_num_threads(2);torch.manual_seed(args.seed)
    np.random.seed(args.seed);random.seed(args.seed)
    base_digest=sha256(args.candidate)
    m,opts,base,comp,blob=load_unified(args.candidate,args.root)
    memory_trainable(m,semantic_only=True)
    inherited=copy.deepcopy([o.state_dict() for o in opts]);core=copy.deepcopy(m.state_dict())
    tr=gather_joint(args.root,m,base,comp,'train',args.train,args.seed+10,args.batch,args.corpus_dir)
    dv=gather_joint(args.root,m,base,comp,'dev',args.dev,args.seed+20,args.batch,args.corpus_dir)
    prior_train=m.joint_nll(tr,semantic_enabled=False)[0].detach()
    prior_dev=m.joint_nll(dv,semantic_enabled=False)[0].detach()
    trsem=gather_sem(args.root,base,'train',tr['positions'],args.seed+30,corpus_dir=args.corpus_dir)
    dvsem=gather_sem(args.root,base,'dev',dv['positions'],args.seed+40,corpus_dir=args.corpus_dir)
    donor=gather_sem(args.root,base,'dev',dv['positions'],args.seed+50,train_donor=True,corpus_dir=args.corpus_dir)
    m.semantic.eval()
    initial=evaluation(m.semantic,prior_dev,dvsem,donor)
    rng=np.random.default_rng(args.seed+60)
    opt=opts[4];opt.param_groups[-1]['lr']=args.lr
    best=None;weights=None;hist=[]
    for step in range(1,args.steps+1):
        ids=torch.from_numpy(rng.choice(args.train,min(args.train,args.batch_train),replace=False))
        m.semantic.train()
        correct, _=m.semantic(prior_train[ids],trsem['semantic_features'][ids],
                              trsem['semantic_prob'][ids],trsem['semantic_count'][ids])
        freq,_=m.semantic(prior_train[ids],trsem['semantic_features'][ids],
                          trsem['frequency_prob'][ids],trsem['semantic_count'][ids])
        adv=(freq-correct).mean()
        loss=correct.mean()+args.contrast*F.softplus((args.margin-adv)*8)/8
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.semantic.parameters(),1.)
        opt.step()
        if step==1 or step%args.every==0 or step==args.steps:
            m.semantic.eval();ev=evaluation(m.semantic,prior_dev,dvsem,donor)
            hist.append({'step':step,'dev':ev})
            acceptable=(ev['gain']>args.min_gain and
                ev['correct_vs_old_freq_advantage']>args.min_adv and
                ev['correct_vs_train_donor_advantage']>args.min_adv)
            if acceptable and (best is None or ev['correct_nll']<best['correct_nll']):
                best=dict(ev,step=step)
                weights=(copy.deepcopy(m.state_dict()),[copy.deepcopy(o.state_dict()) for o in opts])
    if weights is None:
        # Reject: revert *all* updates including fifth optimizer's new slots.
        m.load_state_dict(core)
        for opt,state in zip(opts,inherited):opt.load_state_dict(state)
    else:
        m.load_state_dict(weights[0]);
        for opt,state in zip(opts,weights[1]):opt.load_state_dict(state)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in core.items() if not k.startswith('semantic.'))
    assert all(deep_equal(o.state_dict(),s) for o,s in zip(opts[:4],inherited[:4]))
    assert all(deep_equal(opts[4].state_dict()['state'][k],v) for k,v in inherited[4]['state'].items())
    save=None
    if best is not None:
        save=save_unified(args.output_checkpoint,m,opts,args.candidate,
             train_updates=best['step'],selection={'status':('external book-disjoint dev selected; unpromoted' if args.corpus_dir else 'historical dev selected; unpromoted'),
                                                    'metrics':best,'separate_corpus_dir_supplied':bool(args.corpus_dir),'new_books_tested':False})
        recovered,_,_,_,_=load_unified(args.output_checkpoint,args.root)
        assert all(torch.equal(v,recovered.state_dict()[k]) for k,v in m.state_dict().items())
    assert sha256(args.candidate)==base_digest
    result={'round':'FA1 R7.22','status':'selected-unpromoted' if best else 'rejected-identity-kept',
        'accepted_parent_step':14416,'train_examples':args.train,'dev_examples':args.dev,
        'steps_executed':args.steps,'selected_steps':best['step'] if best else 0,
        'initial_dev':initial,'selected_dev':best,'history':hist,
        'candidate_saved':save,'fifth_optimizer_continuity':True,
        'correct_history_frequency_control':True,'independent_train_donor_control':True,
        'checkpoint_seed_unchanged':True,'full_new_book_evaluation':False,'separate_corpus_dir_supplied':bool(args.corpus_dir),
        'trained_routes':'32K-128K semantic gate only; 128K-1M gate unchanged',
        'historical_splits_only':not bool(args.corpus_dir),'promoted':False,
        'elapsed_s':round(time.monotonic()-t,2)}
    pathlib.Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/fa1_full')
    p.add_argument('--corpus-dir',default=None,help='Independent document-disjoint NaNROUTER corpus from prepare_document_corpus.py')
    p.add_argument('--candidate',default='/mnt/data/fa1r722/checkpoint/r7_unified_candidate.pt')
    p.add_argument('--train',type=int,default=128);p.add_argument('--dev',type=int,default=96)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--batch-train',type=int,default=32)
    p.add_argument('--steps',type=int,default=64);p.add_argument('--every',type=int,default=8)
    p.add_argument('--lr',type=float,default=.005);p.add_argument('--seed',type=int,default=72201)
    p.add_argument('--contrast',type=float,default=.2);p.add_argument('--margin',type=float,default=.01)
    p.add_argument('--min-gain',type=float,default=.0001);p.add_argument('--min-adv',type=float,default=.0001)
    p.add_argument('--output',default='/mnt/data/fa1r722/reports/R722_SEMANTIC_GATE_PILOT.json')
    p.add_argument('--output-checkpoint',default='/mnt/data/fa1r722/checkpoint/r7_unified_trained_candidate.pt')
    main(p.parse_args())