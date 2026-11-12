#!/usr/bin/env python3
"""R7.12 independent-query resampling and causal wrong-memory control.
Historical held-out splits are not new books, so do not report book CIs.
"""
import argparse,pathlib,json,torch,numpy as np
from r711_delta import restore_overlay
from r712_lm_bridge import CalibrationGate,collect,summary,score
from r710_checkpoint_integrity import sha256
from r75_retrieval import OPT_NAMES

def restore(root,overlay):
    root=pathlib.Path(root);overlay=pathlib.Path(overlay)
    expected=overlay.with_suffix(overlay.suffix+'.sha256')
    if not expected.exists() or expected.read_text().split()[0]!=sha256(overlay):raise ValueError('corrupt R7.12 overlay')
    obj=torch.load(overlay,map_location='cpu',weights_only=False)
    if obj.get('format')!='PAL_R7_12_CAUSAL_LM_GATE_OVERLAY_V1':raise ValueError('unknown overlay format')
    if obj.get('parent_sha256')!=sha256(root/'checkpoint/pal_active.pt'):raise ValueError('different accepted checkpoint')
    if obj.get('r711_delta_sha256')!=sha256(root/'adaptation/r711_experimental_delta.pt'):raise ValueError('different directed writer')
    model,opts,base,comp,_=restore_overlay(root,root/'adaptation/r711_experimental_delta.pt')
    gate=CalibrationGate();gate.load_state_dict(obj['gate_state'],strict=True)
    opts[4].add_param_group({'params':list(gate.parameters()),'lr':.008,'weight_decay':0.0})
    now=opts[4].state_dict()
    if now['param_groups'][-1]['params']!=obj['new_fifth_group']['params']:raise ValueError('optimizer id mismatch')
    now['param_groups'][-1]=obj['new_fifth_group']
    now['state'].update(obj['new_fifth_states'])
    opts[4].load_state_dict(now)
    return model,opts,base,comp,gate,obj

def paired_ci(base,correct,wrong,gate,seed=1):
    a=base['baseline'].numpy();corr=score(gate,correct).detach().numpy();mis=score(gate,wrong).detach().numpy()
    assert np.array_equal(a,correct['baseline'].numpy()) and np.array_equal(a,wrong['baseline'].numpy())
    differences=a-corr;adv=mis-corr;rng=np.random.default_rng(seed)
    # Query bootstrap IS NOT a book bootstrap and may exaggerate precision.
    ix=rng.integers(0,len(a),size=(2000,len(a)))
    return {'positive_gain_ci95_query_bootstrap':np.quantile(differences[ix].mean(1),[.025,.975]).tolist(),
       'correct_vs_wrong_ci95_query_bootstrap':np.quantile(adv[ix].mean(1),[.025,.975]).tolist(),
       'correct_vs_wrong_mean':float(adv.mean()),'correct_nll':float(corr.mean()),'wrong_nll':float(mis.mean()),
       'legacy_nll':float(a.mean()),'n':len(a)}

def main(args):
    torch.set_num_threads(2)
    root=args.root;model,opts,base,comp,gate,obj=restore(root,root/'adaptation/r712_gate_delta.pt')
    saved=sha256(root/'checkpoint/pal_active.pt');reports={}
    for split,seed in (('dev',817),('test',818)):
        data=collect(root,model,base,comp,split,args.count,seed,args.batch,False)
        wrong=collect(root,model,base,comp,split,args.count,seed,args.batch,True)
        result=paired_ci(data,data,wrong,gate,seed)
        reports[split]={'correct':summary(gate,data),'wrong_memory':summary(gate,wrong),
             'paired_statistics':result,'book_cluster_ci_available':False}
        print(json.dumps({'phase':'EVAL','split':split,'correct_gain':reports[split]['correct']['gain'],
             'wrong_gain':reports[split]['wrong_memory']['gain'],'paired_advantage':result['correct_vs_wrong_mean']}),flush=True)
    assert saved==sha256(root/'checkpoint/pal_active.pt')
    report={'r712_validation':'reseeding and source-mismatch controls',
       'accepted_sha256':saved,'five_optimizers_intact':len(opts)==5,
       'historical_data_not_independent_books':True,
       'pairwise_query_bootstrap_not_book_level':True,'results':reports}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':'DONE','output':str(args.output),'parent_unchanged':True}),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('.'))
    p.add_argument('--count',type=int,default=512);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('reports/R7p12_validation512.json'))
    main(p.parse_args())
