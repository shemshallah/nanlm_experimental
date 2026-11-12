#!/usr/bin/env python3
"""R7.11 hard negative, clustered document / order controls, overlay restore audit."""
import json, pathlib, time, argparse
import torch, numpy as np
from torch.nn import functional as F
from r711_selective_writer import restore,build_examples,score_batches,evaluate,HORIZON
from r711_delta import restore_overlay
from r710_checkpoint_integrity import verify,deep_equal,sha256
from r7_core import Corpus
from r74_stream import OLDER,OUTER_CONTEXT

def paired_data(model,E,data,batch=16):
    def get(**kw):return score_batches(model.r74.outer,E,data,batch=batch,**kw)[0].detach().numpy()
    corr=get();wrong=get(permuted=True);rev=get(reversed=True);zero=get(zero=True)
    return {'n':len(corr),'correct_ce':float(corr.mean()),'wrong_ce':float(wrong.mean()),
      'reversed_ce':float(rev.mean()),'zero_ce':float(zero.mean()),
      'unrelated_advantage':float((wrong-corr).mean()),
      'reversed_advantage':float((rev-corr).mean()),
      'top1':float(score_batches(model.r74.outer,E,data,batch=batch)[1].mean())},(corr,wrong,rev)

def doc_ci(root,split,data,arrays,seed):
    c=Corpus(root,split);ids=c.doc_start[data['positions']] if c.doc_start is not None else np.zeros(len(data['positions']),dtype=np.int64)
    unique=np.unique(ids)
    if c.doc_start is None:return {'document_metadata_available':False,'distinct_documents':None,'ci':None,'reason':'Legacy corpus has no per-position document-start metadata; book-cluster confidence unavailable.'}
    if len(unique)<2:return {'document_metadata_available':True,'distinct_documents':len(unique),'ci':None}
    group=[np.flatnonzero(ids==i) for i in unique];rng=np.random.default_rng(seed)
    delta=arrays[1]-arrays[0];drev=arrays[2]-arrays[0]
    # Equal-weight individual queries; resample whole documents with replacement.
    boot=[];boot_order=[]
    for _ in range(1500):
        chosen=rng.integers(0,len(group),size=len(group))
        idx=np.concatenate([group[j] for j in chosen]);boot.append(float(delta[idx].mean()));boot_order.append(float(drev[idx].mean()))
    return {'distinct_documents':len(unique),'queries_per_document':{str(int(i)):int((ids==i).sum()) for i in unique},
       'ci95_unrelated_advantage':list(map(float,np.quantile(boot,[.025,.975]))),
       'ci95_reversed_advantage':list(map(float,np.quantile(boot_order,[.025,.975]))),
       'note':'Document bootstrap for historical repeatedly used splits; not sealed independent generalization.'}


def main(a):
    root=a.root;torch.set_num_threads(2);t=time.monotonic()
    parent_data,pinfo=verify(root/'checkpoint/pal_active.pt',root,expected_step=14416)
    parent,opts0,_,_,prior=restore(root,root/'checkpoint/pal_active.pt')
    model,opts,_,_,patch=restore_overlay(root,a.overlay)
    E=model.trunk.E.detach()
    assert torch.equal(parent.trunk.E,model.trunk.E)
    assert float(model.r74.outer.dir_gate.detach())==0
    # Compare complete inherited matrix-state, not merely layer parameter values.
    corpus=Corpus(root,'dev');h=torch.tensor(corpus.get(np.array([13000,35000,52000,75000]),OUTER_CONTEXT)[0][:,:OLDER].copy()).long()
    with torch.no_grad():
        old_state=parent.initial(h);new_state=model.initial(h)
    assert torch.equal(old_state,new_state), 'zero-gate migration altered physical memory'
    # No legacy optimizer moments/weights changed.
    assert all(torch.equal(model.state_dict()[k],v) for k,v in parent_data['model'].items())
    for i,name in enumerate(('legacy_accum','adapter_adamw','mirror_adamw','outer_adamw','retrieval_adamw')):
        if name=='outer_adamw':
            old=opts0[i].state_dict();new=opts[i].state_dict()
            assert deep_equal(old['param_groups'],new['param_groups'])
            assert all(k in new['state'] and deep_equal(v,new['state'][k]) for k,v in old['state'].items())
        else:assert deep_equal(opts0[i].state_dict(),opts[i].state_dict())
    checks={'accepted_checkpoint_unchanged':sha256(root/'checkpoint/pal_active.pt')==pinfo['sha256'],
      'bitwise_identical_inherited_matrix':True,'bitwise_identical_inherited_parameters':True,
      'five_inherited_optimizer_histories_unchanged':True,'auxiliary_updates_restored':patch['directed_aux_updates']}
    # Additional sampled history queries that were not used for selection.
    results={}
    for split,seed in [('dev',1702),('test',1703)]:
        data=build_examples(root,split,a.count,seed,E)
        inherited,inarr=paired_data(parent,E,data,a.batch)
        trained,outarr=paired_data(model,E,data,a.batch)
        results[split]={'before_untrained_migration':inherited,'after_aux_training':trained,
          'ce_improvement':inherited['correct_ce']-trained['correct_ce'],
          'document_cluster':doc_ci(root,split,data,outarr,seed+99)}
        print(json.dumps({'split':split,'query_seed':seed,'count':a.count,
            'before_ce':inherited['correct_ce'],'after_ce':trained['correct_ce'],
            'after_memory_advantage':trained['unrelated_advantage'],
            'document_clusters':results[split]['document_cluster']['distinct_documents']}),flush=True)
    # A two-sample gradient check isolates the new learned pair path.
    data=build_examples(root,'train',4,711,E)
    st,km,vm=model.r74.outer.pair_state(data['history'],E)
    scores=model.r74.outer.pair_score(st,data['query'],data['answers'],E,km,vm)
    loss=F.cross_entropy(scores,torch.zeros(4,dtype=torch.long))
    grads=torch.autograd.grad(loss,(model.r74.outer.dir_key.weight,model.r74.outer.dir_value.weight,model.r74.outer.dir_select.weight))
    assert all(g.isfinite().all() and g.abs().sum()>0 for g in grads)
    checks['directed_writer_key_value_selector_gradients_nonzero']=True
    result={'checks':checks,'independent_book_validation':False,'alternate_query_seed':True,
        'selected_hyperparameters_unchanged':True,'historical_split_results':results,'elapsed_seconds':round(time.monotonic()-t,2)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','elapsed_seconds':result['elapsed_seconds'],'checks':checks}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--overlay',type=pathlib.Path,default=pathlib.Path('adaptation/r711_experimental_delta.pt'))
    p.add_argument('--count',type=int,default=512);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/R7p11_validation.json'))
    main(p.parse_args())
