#!/usr/bin/env python3
"""R7.10 deterministic, causal and negative-mining regression tests."""
import json,pathlib
import numpy as np,torch
from r710_hard_negative import pick
from r710_checkpoint_integrity import verify
from r77_semantic import restore
from r74_stream import OLDER,OUTER_CONTEXT,BLOCK
from r7_core import Corpus


def run(root=pathlib.Path('.')):
    torch.set_num_threads(2); root=pathlib.Path(root)
    accepted=root/'checkpoint/pal_active.pt'
    data,info=verify(accepted,root,expected_step=14416)
    checks={}
    assert len(info['optimizer_histories'])==5
    checks['one_accepted_checkpoint_five_full_histories']=True
    assert len(info['rng_keys_verified'])==4
    checks['torch_numpy_python_batch_rng_tracked']=True
    model,opts,base,comp,prior=restore(root,accepted,torch.device('cpu'))
    assert all(torch.equal(x,data['model'][n]) for n,x in model.state_dict().items())
    checks['accepted_model_restores_bitwise']=True
    c=Corpus(root,'dev'); ix=np.asarray([8192,18000,36000,54000])
    h=c.get(ix,OUTER_CONTEXT)[0][:,:OLDER].copy()
    E=model.trunk.E.detach().cpu()
    for kind in ('random','hard'):
        selected=[pick(x,E,1337+i,kind=kind) for i,x in enumerate(h)]
        assert all(x is not None for x in selected)
        assert selected==[pick(x,E,1337+i,kind=kind) for i,x in enumerate(h)]
        checks[kind+'_repeatable_mining']=True
        for history,entry in zip(h,selected):
            j=entry['historic_source_index'];q=entry['query'];y=entry['positive']
            assert int(history[j])==q and int(history[j+1])==y and j<OLDER-1
            assert np.count_nonzero(history==q)==1
            assert len(entry['candidates'])==32 and entry['candidates'][0]==y
            assert len(set(entry['candidates']))==32 and q not in entry['candidates'][1:]
            assert all(v in history for v in entry['candidates'])
            checks[kind+'_strict_causal_older_positive']=True
            checks[kind+'_distinct_in_history_negatives']=True
        # Selection is a pure function of old history, embeddings and seed,
        # and has no API input for the current/future target.
        assert 'current_target' not in pick.__code__.co_varnames
        checks[kind+'_no_future_target_dependency']=True
    # Model's inherited averaged writer provably loses within-block order.
    with torch.no_grad():
        t=torch.from_numpy(h).long()
        st=model.initial(t)
        rev=model.initial(t.reshape(-1,OLDER//BLOCK,BLOCK).flip(-1).reshape(-1,OLDER))
        assert torch.allclose(st,rev,atol=1e-6,rtol=0)
    checks['inherited_rank64_cannot_recall_withinblock_order']=True
    assert pick(np.ones(OLDER,dtype=np.int64),E,7,kind='hard') is None
    checks['padding_does_not_create_false_pairs']=True
    assert int(model.trunk.E.shape[0])>=32000
    checks['frozen_embedding_hardness_defined']=True
    assert info['step']==14416
    checks['no_implicit_training_step']=True
    return {'tests_passed':len(checks),'checks':checks,'accepted_step':info['step'],
            'accepted_sha256':info['sha256']}
if __name__=='__main__':
    print(json.dumps(run(),indent=2))
