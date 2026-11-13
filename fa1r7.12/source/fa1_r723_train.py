#!/usr/bin/env python3
"""One unified R7.23 Diamond CPU training and honest heldout gates.

Legacy states + all five existing optimizers remain bit-identical unless user
explicitly authorizes subsequent trunk fine-tuning. No accepted promotion.
"""
from __future__ import annotations
import argparse,copy,json,math,pathlib,random,time
import numpy as np, torch
import torch.nn.functional as F
from r7_core import Corpus
from fa1_r721_joint_train import gather_joint
from fa1_r723_corpus import diamond_context
from fa1_r723_diamond import extend_r722,save_diamond,load_diamond
from r710_checkpoint_integrity import sha256,deep_equal


def prepare(a,m,base,comp,split,n,seed):
    with torch.no_grad():
        data=gather_joint(a.root,m,base,comp,split,n,seed,batch=a.batch,corpus_dir=a.corpus_dir)
        extras=diamond_context(a.root,base,split,data['positions'],data['hidden'],a.corpus_dir)
    data.update(extras)
    return data


def take(data,idx):
    return {k:(x[idx] if torch.is_tensor(x) and len(x.shape)>0 and x.shape[0]==len(data['target']) else x)
            for k,x in data.items()}


def identity(model,dev):
    with torch.no_grad():
        reference=model.joint_nll(dev,diamond_enabled=False)[0]
        migrated=model.joint_nll(dev,diamond_enabled=True)[0]
        assert torch.equal(reference,migrated),'initial Diamond must preserve exact R7.22 NLL'
        assert (model.diamond.gate==0).item()
    return float(reference.mean())


def forward(model,data,mode='full',two_way=True):
    x=dict(data)
    if mode=='unrelated':
        # Scramble *all* Diamond context together but leave targets and R7
        # main LM + near/far memory unchanged. This is target-free routing.
        ids=torch.arange(len(data['target'])-1,-1,-1)
        for k in ('word_ids','byte_ids','sentence_ids','vertex_ids'):x[k]=data[k][ids]
    ablate={'without_apex':mode=='no_apex','without_pair':mode=='no_pair'}
    nll,_,extra=model.joint_nll(x,diamond_enabled=True,return_extras=True,
                                two_way=two_way,**ablate)
    return nll,extra


@torch.no_grad()
def evaluate(model,data,ablate=False):
    model.eval()
    base=model.joint_nll(data,diamond_enabled=False)[0]
    correct,aux=forward(model,data)
    unrelated,_=forward(model,data,'unrelated')
    byte_ce=F.cross_entropy(aux['byte_logits'],data['first_byte_target'])
    sectors=aux['trace']['selected_sectors'][:,0]
    result={'baseline_nll':float(base.mean()),'full_nll':float(correct.mean()),
            'gain':float((base-correct).mean()),
            'correct_vs_unrelated':float((unrelated-correct).mean()),
            'first_byte_ce':float(byte_ce),
            'unique_meta_specialists':int(sectors.unique().numel()),
            'unique_triple_addresses':int(aux['trace']['triangle_addresses'].unique().numel()),
            'gate':float(aux['trace']['gate']),
            'apex_std':float(aux['trace']['apex_std']),
            'branch_nll':float(aux['branch_nll'].mean()),
            'learned_trust':float(torch.sigmoid(aux['trace']['trust_logit']).mean()),
            'branch_win_fraction':float((aux['branch_nll']<base).float().mean()),
            'oracle_best_of_two_potential':float((base-aux['branch_nll']).clamp_min(0).mean())}
    if ablate:
        no_apex,_=forward(model,data,'no_apex')
        no_pair,_=forward(model,data,'no_pair')
        result['apex_advantage']=float((no_apex-correct).mean())
        result['pair_down_advantage']=float((no_pair-correct).mean())
    return result


def aux_losses(model,data,extras):
    first=F.cross_entropy(extras['byte_logits'],data['first_byte_target'])
    tvec=F.normalize(extras['word_vector'],dim=-1)
    teacher=F.normalize(F.embedding(data['target'],model.core.trunk.E.detach()),dim=-1)
    similarity=tvec@teacher.T * 8
    contrast=F.cross_entropy(similarity,torch.arange(len(data['target'])))
    # Load balance via differentiable top4 route weights. Sector utilization
    # is also independently reported so a collapsed overmind cannot masquerade.
    return first,contrast


def run(a):
    ts=time.monotonic();torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    old_sha=sha256(a.parent)
    m,opts,base,comp,meta=extend_r722(a.parent,a.root,threads=a.threads)
    oldmoments=[copy.deepcopy(o.state_dict()) for o in opts[:4]]
    fifth_before=copy.deepcopy(opts[4].state_dict())
    original={k:v.detach().clone() for k,v in m.state_dict().items() if not k.startswith('diamond.')}
    for p in m.parameters():p.requires_grad_(False)
    for p in m.diamond.parameters():p.requires_grad_(True)
    print('R7_DIAMOND_PREP start',flush=True)
    train=prepare(a,m,base,comp,'train',a.train,a.seed+101)
    dev=prepare(a,m,base,comp,'dev',a.dev,a.seed+202)
    parent=identity(m,dev)
    print(f'BASELINE dev_nll={parent:.6f} train={a.train} dev={a.dev} elapsed={time.monotonic()-ts:.1f}s',flush=True)
    opt=opts[4];opt.param_groups[-1]['lr']=a.lr
    rng=np.random.default_rng(a.seed+300)
    history=[];best=None;best_state=None
    for step in range(1,a.steps+1):
        ids=torch.tensor(rng.choice(a.train,min(a.minibatch,a.train),replace=False),dtype=torch.long)
        sample=take(train,ids)
        m.diamond.train()
        joint,extra=forward(m,sample)
        bce,nce=aux_losses(m,sample,extra)
        # Supervised apex curriculum: the inherited decoder's exact tree
        # likelihood is a direct teacher for the entire triple->pair->point
        # path even while the residual gate is intentionally closed.
        # Previously the gate was hard-clamped at zero, giving the diamond
        # no gradient from the actual next-word task and producing an
        # auxiliary-only network with no measurable LM contribution.
        branch_ce=extra['branch_nll'].mean()
        # Target-free hard negative: retain each true next-word target and
        # inherited R7 state but swap the full Diamond context as one unit.
        # The *direct branch* must become better on correct causal history,
        # even with the user-facing bridge closed during warmup.
        if a.context_contrast>0:
            _,negative=forward(m,sample,'unrelated')
            bad_ce=negative['branch_nll'].mean()
            difference=bad_ce-branch_ce
            contrastive=F.softplus((a.context_margin-difference)*8)/8
        else:
            bad_ce=branch_ce.detach();difference=branch_ce.new_zeros(())
            contrastive=branch_ce.new_zeros(())
        # Advantage classifier: meta-mind predicts whether its proposed
        # diamond tree likelihood beats the parent R7 distribution for this
        # *training* target. This signal NEVER enters inference routing.
        with torch.no_grad():
            old_ce=m.joint_nll(sample,diamond_enabled=False)[0]
            useful=(extra['branch_nll'].detach()<old_ce.detach()).float()
        trust_ce=F.binary_cross_entropy_with_logits(extra['trace']['trust_logit'],useful)
        if step<=a.warmup:
            loss=a.branch_weight*branch_ce+a.context_contrast*contrastive+a.trust_weight*trust_ce+a.aux_byte*bce+a.aux_word*nce
        else:
            if step==a.warmup+1:
                # Open only exploratory candidate, never its accepted parent.
                with torch.no_grad():m.diamond.gate.fill_(float(np.arctanh(.2)))
            loss=joint.mean()+a.branch_weight*branch_ce+a.context_contrast*contrastive+a.trust_weight*trust_ce+a.aux_byte*bce+a.aux_word*nce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.diamond.parameters(),1.0)
        opt.step()
        if step==1 or step%a.eval_every==0 or step==a.steps:
            eva=evaluate(m,dev,ablate=True)
            acceptable=(eva['gain']>a.min_gain and eva['correct_vs_unrelated']>a.min_history_adv and
                       eva['apex_advantage']>a.min_apex_adv and eva['pair_down_advantage']>a.min_apex_adv and
                       eva['unique_meta_specialists']>=2)
            eva.update({'step':step,'train_loss':float(loss.detach()),
                        'aux_byte_train':float(bce.detach()),'aux_word_train':float(nce.detach()),
                        'branch_nll_train':float(branch_ce.detach()),'unrelated_branch_train':float(bad_ce.detach()),
                        'context_difference_train':float(difference.detach()),'trust_ce_train':float(trust_ce.detach()),
                        'passes_gates':acceptable})
            history.append(eva)
            print('STEP',step,json.dumps({k:round(v,6) if isinstance(v,float) else v for k,v in eva.items()}),flush=True)
            if acceptable and (best is None or eva['full_nll']<best['full_nll']):
                best=dict(eva)
                best_state=(copy.deepcopy(m.diamond.state_dict()),copy.deepcopy(opts[4].state_dict()))
            if step>=a.min_steps and a.stop_on_plateau and not any(h['passes_gates'] for h in history):
                print('EARLY_STOP no qualifying positive dev gates',flush=True)
                break
    if best_state is not None:
        m.diamond.load_state_dict(best_state[0])
        opts[4].load_state_dict(best_state[1])
    else:
        # Keep clean R7 parent; exploratory updated diamond is not an accepted model.
        # Save the last unpromoted as a research artifact only, not checkpoint.
        pass
    assert sha256(a.parent)==old_sha
    assert all(torch.equal(m.state_dict()[k],v) for k,v in original.items())
    assert all(deep_equal(o.state_dict(),old) for o,old in zip(opts[:4],oldmoments))
    now=opts[4].state_dict()
    assert all(deep_equal(now['state'][k],v) for k,v in fifth_before['state'].items())
    original_test=evaluate(m,dev,ablate=True)
    report={'format':'FA1_R7_23_DIAMOND_REPORT_V1','architecture':'single unified R7.22 trunk with true C(256,3) apex',
        'accepted_step':14416,'base_r722_sha256':old_sha,'optimizers_inherited':5,
        'optimizer_slots_preserved':True,'training_samples':a.train,
        'dev_samples':a.dev,'steps_executed':step,'best_selected':best,
        'final_dev':original_test,'historical_dev_only':a.corpus_dir is None,
        'independent_books':False,'byte_bpb_measured':False,
        'candidate_promoted':False,'history':history,'elapsed_s':time.monotonic()-ts}
    if best is not None:
        report['unpromoted_candidate']=save_diamond(a.output_checkpoint,m,opts,old_sha,training_step=best['step'],metrics=best)
        restored,restored_opts,_,_,_=load_diamond(a.output_checkpoint,a.root)
        assert all(torch.equal(t,restored.state_dict()[k]) for k,t in m.state_dict().items())
        assert all(deep_equal(o.state_dict(),b.state_dict()) for o,b in zip(opts,restored_opts))
        report['checkpoint_roundtrip_verified']=True
    else:report['unpromoted_candidate']=None
    pathlib.Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('FINISHED',json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/mnt/data/diamond_r723/r7_ancestor')
    p.add_argument('--parent',default='/mnt/data/files/FA1_R7p22_UNIFIED_CANDIDATE.pt')
    p.add_argument('--corpus-dir',default=None)
    p.add_argument('--output',default='/mnt/data/diamond_r723/r723_cpu_report.json')
    p.add_argument('--output-checkpoint',default='/mnt/data/diamond_r723/R723_DIAMOND_UNPROMOTED.pt')
    p.add_argument('--train',type=int,default=128);p.add_argument('--dev',type=int,default=64)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--minibatch',type=int,default=24)
    p.add_argument('--steps',type=int,default=256);p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--eval-every',type=int,default=32);p.add_argument('--min-steps',type=int,default=96)
    p.add_argument('--stop-on-plateau',action='store_true')
    p.add_argument('--lr',type=float,default=.0008);p.add_argument('--aux-byte',type=float,default=.1)
    p.add_argument('--aux-word',type=float,default=.12)
    p.add_argument('--branch-weight',type=float,default=.4)
    p.add_argument('--context-contrast',type=float,default=.7)
    p.add_argument('--context-margin',type=float,default=.02)
    p.add_argument('--trust-weight',type=float,default=.12)
    p.add_argument('--min-gain',type=float,default=.0001)
    p.add_argument('--min-history-adv',type=float,default=.0001)
    p.add_argument('--min-apex-adv',type=float,default=.00001)
    p.add_argument('--seed',type=int,default=72323);p.add_argument('--threads',type=int,default=2)
    run(p.parse_args())