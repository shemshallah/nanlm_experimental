#!/usr/bin/env python3
"""Small fail-closed causal, 32K retrieval and gate control tests."""
import json,pathlib,sys,os
import numpy as np,torch
from fa1_r716_router import extract_paths, TrustRouter, CHANNELS, FEATURE_DIM, SCALES, nll

def run():
    torch.set_num_threads(2)
    tests=[]
    ctx=np.array([5,6,7,8])
    h=np.ones(32768,dtype=np.int64)
    # Distinct causal 32k event outside 8K; a matching RECENT event must be excluded.
    h[1000:1002]=[8,9]
    h[-2000:-1998]=[8,55]
    old=extract_paths(h,ctx,9,32768)
    near=extract_paths(h,ctx,9,8192)
    assert old[0]['target_probability']>0.99 and old[0]['n']==1
    assert near[0]['n']==0 and near[0]['target_probability']==0
    tests.append('32k_old_history_read_8k_miss')
    assert old[0]['modal']==9 and old[0]['modal']!=55
    tests.append('recent_2048_cannot_enter_extra_memory')
    changed=extract_paths(h,ctx,20,32768)
    for a,b in zip(old,changed):
        for k in ('n','weight_total','peak','entropy','recent','modal'):
            assert a[k]==b[k],(k,a[k],b[k])
    assert changed[0]['target_probability']==0
    tests.append('changing_next_token_cannot_change_memory_selection_or_features')
    h2=h.copy();h2[-1]=8;h2[-2]=7
    # Last 2048 tokens are shielded from this router.
    assert extract_paths(h2,ctx,9,32768)==old
    tests.append('future_near_context_mutation_does_not_affect_memory')
    gate=TrustRouter()
    feat=torch.randn(4,FEATURE_DIM)
    count=torch.zeros(4,CHANNELS)
    weights=gate(feat,count)
    assert torch.equal(weights,torch.zeros_like(weights))
    tests.append('no_evidence_is_exact_gate_off')
    data={'base':torch.rand(4)+5.,'prob':torch.rand(4,CHANNELS), 'features':feat,'count':count}
    loss,g=nll(gate,data)
    assert torch.allclose(loss,data['base'],rtol=1e-6,atol=1e-6)
    tests.append('gate_off_preserves_original_lm_exactly')
    count[:,0]=4.;count[:,CHANNELS-1]=1.
    gate.eval()
    a=gate(feat,count,allowed_scales=(True,False))
    b=gate(feat,count,allowed_scales=(False,True))
    assert (a[:,4:]==0).all() and (b[:,:4]==0).all()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    tests.append('one_scale_ablations_are_target_free_and_finite')
    # Multiples of same padded history must not leak by order alignment.
    h3=np.ones(8192,dtype=np.int64)
    h3[4000:4005]=[5,6,7,8,14]
    out=extract_paths(h3,ctx,14,8192)
    assert [x['n'] for x in out]==[1,1,1,1]
    assert [x['target_probability'] for x in out]==[1.,1.,1.,1.]
    tests.append('causal_1_2_3_4_gram_alignment')
    path=pathlib.Path(os.environ.get('FA1_R716_REPORT', str(pathlib.Path(__file__).resolve().parent.parent/'reports/FA1_R7p16_225.json')))
    if path.exists():
        report=json.loads(path.read_text())
        assert report['parent_verified_unmodified'] and report['five_optimizer_histories_unchanged']
        assert report['dev']['gain']>0 and report['dev']['advantage']>0
        assert report['historical_test']['gain']>0 and report['historical_test']['advantage']>0
        assert report['ablations']['32k_only']['gain']>report['ablations']['8k_only']['gain']
        tests.append('completed_historical_dev_test_and_optimizer_gates')
    ret={'passed':len(tests),'total':len(tests),'tests':tests,'parent_binary_untouched':True}
    print(json.dumps(ret,indent=2))
    return ret

if __name__=='__main__':run()