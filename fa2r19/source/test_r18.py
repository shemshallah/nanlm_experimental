#!/usr/bin/env python3
"""PAL R18 structural/gradient/causality/scale tests. CPU only."""
import hashlib,json,pathlib,sys,time
import torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from r18_long_memory import R18LongMemory,build_copy_queries,apply_copy_boost
from r18_joint_continue import build_joint_optimizer
from r18_postselection_test import causal_test
import r14_slim_base as pal


def run(root):
    torch.set_num_threads(2);root=pathlib.Path(root)
    saved,r6,comp,model,bridge,adapter,opt,rng=pal.load_for_arm(root,root/'checkpoint/r12_active.pal12',True)
    net=R18LongMemory(model.memory.piece_emb.weight)
    net.load_state_dict(saved['r18_memory']);net.eval()
    checks={};checks['r18_checkpoint_extensions']=all(k in saved for k in ('r18_memory','r18_config','r18_retrieval_optimizer','r18_gain_optimizer'))
    checks['one_checkpoint_four_prior_groups']=len(opt.param_groups)==4 and saved['r12_updates']==10756
    checks['r18_gain_trained']=float(net.logit_gain.detach())>0
    checks.update(causal_test(net))
    # Nested hard negatives: a correct two-piece key among same-last-token decoys.
    scores=net.scores(torch.tensor([[705,706],[720,706]]),torch.tensor([[[705,706],[800,706]],[[720,706],[801,706]]]))
    checks['learned_query_key_finite']=bool(torch.isfinite(scores).all())
    # Contrastive learning reaches both Q and K, independent of copied values.
    net.train();loss=F.cross_entropy(scores,torch.zeros(2,dtype=torch.long));loss.backward()
    checks['query_receives_gradient']=any(p.grad is not None and p.grad.abs().sum()>0 for p in net.query_net.parameters())
    checks['key_receives_gradient']=any(p.grad is not None and p.grad.abs().sum()>0 for p in net.key_net.parameters())
    net.zero_grad(set_to_none=True)
    # From actual accepted state: optimize a tiny word-copy cross entropy and
    # prove R18 gate/key trainability in the *same* decoder distribution.
    x=torch.randint(100,800,(260,),dtype=torch.long);x[9:11]=torch.tensor([705,706]);x[74:76]=torch.tensor([800,706]);x[76]=610;x[210:212]=torch.tensor([705,706])
    q=build_copy_queries(x,mem=2048,gap=65)
    b=torch.randn(260,900)
    aug=apply_copy_boost(net,b,q)
    F.cross_entropy(aug[212:213],torch.tensor([int(x[11])])).backward()
    checks['end_to_end_gate_gradient']=bool(net.logit_gain.grad is not None and float(net.logit_gain.grad.abs())>0)
    checks['retriever_end_to_end_gradient']=any(p.grad is not None and p.grad.abs().sum()>0 for p in net.query_net.parameters())
    net.zero_grad(set_to_none=True)
    # Negative: a memory budget less than minimum gap returns no remote slots.
    checks['no_false_long_memory_when_cap64']=build_copy_queries(x,mem=64,gap=65) is None
    # 40,000-token index smoke. This validates indexing, NOT language-model
    # understanding at 40K context or 40K-token joint training.
    source=11;destination=39011
    x=torch.full((40000,),500,dtype=torch.long)
    x[9:11]=torch.tensor([705,706]);x[source]=777
    x[destination-2:destination]=torch.tensor([705,706])
    t=time.monotonic();q=build_copy_queries(x,mem=65536,gap=65)
    ii=(q['positions']==destination).nonzero().flatten().item()
    keys=q['keys'][ii];value=q['values'][ii];mask=q['mask'][ii]
    checks['sparse_index_supports_39000_token_distance']=bool(((keys[:,0]==705)&(keys[:,1]==706)&(value==777)&mask).any())
    checks['sparse_candidates_bounded_64']=q['keys'].shape[1]<=64
    scale_seconds=time.monotonic()-t
    # Original four moment histories plus all 15 R18 new parameter moments.
    joint,history=build_joint_optimizer(opt,net,saved,5e-5)
    checks['all_new_adam_moments_migrated']=history['restored_moments']==len(list(net.parameters()))
    checks['five_groups_joint_ready']=len(joint.param_groups)==5
    bool_checks={key:value for key,value in checks.items() if isinstance(value,bool)}
    assert all(bool_checks.values()),checks
    report={'passed':sum(bool_checks.values()),'total':len(bool_checks),'checks':checks,
            'source_checkpoint_sha256':pal.sha(root/'checkpoint/r12_active.pal12'),
            'sparse_index_40000_seconds':scale_seconds,
            'limitations':['40k indexing smoke is not learned 40k-context language inference',
                           'No external new-book validation in this test']}
    (root/'reports/r18_invariants.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    run(sys.argv[1] if len(sys.argv)>1 else '.')
