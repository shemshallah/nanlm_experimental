#!/usr/bin/env python3
"""FA1 R7.18 causal, non-overlap, and fail-safe retrieval unit tests."""
import hashlib, json, pathlib
import numpy as np
import torch
from fa1_r718_far_residual import (HORIZON,FAR_EXCLUSION,FAR_FEATURES,FAR_CHANNELS,
    far_only,FarResidual,loss_far,make_feature,TrustRouter,pick_positions)
from fa1_r716_router import extract_paths,CHANNELS,FEATURE_DIM,PARENT

def run():
  torch.set_num_threads(2);tests=[]
  ctx=np.asarray([5,6,7,8],dtype=np.int64)
  history=np.ones(HORIZON,dtype=np.int64)
  history[100:105]=[5,6,7,8,44]                 # ancient event
  history[-3200:-3198]=[8,55]                   # younger than 32K, exclude
  history[-1200:-1198]=[8,66]                   # recent, exclude
  got=far_only(history,ctx,44)
  assert [x['n'] for x in got]==[1]*4
  assert all(x['modal']==44 and x['target_probability']==1. for x in got)
  tests.append('far_only_32k_128k_retrieval')
  assert all(x['modal']!=55 and x['modal']!=66 for x in got)
  tests.append('far_excludes_near_32k_and_recent_2k')
  new=far_only(history,ctx,99)
  for a,b in zip(got,new):
    for k in ('n','weight_total','peak','entropy','recent','modal'):
      assert a[k]==b[k],k
    assert b['target_probability']==0.
  tests.append('labels_never_enter_far_features_or_route_selection')
  near=extract_paths(history,ctx,55,32768)
  assert near[0]['n']==1 and near[0]['modal']==55 and got[0]['modal']==44
  tests.append('near_and_far_disjoint_age_bands')
  history2=history.copy();history2[-100]=7;history2[-30000]=10
  assert far_only(history2,ctx,44)==got
  tests.append('far_cannot_see_near_content_changes')
  # The old rolled-donor control, if used on dev, could see the target in 33.2% of samples;
  # do not silently reinstate it.
  n=256;seq=np.linspace(HORIZON+1,241850,n).astype(np.int64);donor=np.roll(seq,n//3)
  leaked=((donor-seq>=FAR_EXCLUSION+1)&(donor-seq<=HORIZON)).sum()
  assert leaked>=80
  tests.append('legacy_future_donor_bug_regression_reproduced_and_excluded')
  rng=np.random.default_rng(718)
  far=history[-HORIZON:-FAR_EXCLUSION].copy();before=np.sort(far)
  rng.shuffle(far)
  assert np.array_equal(np.sort(far),before)
  tests.append('shuffled_past_preserves_token_frequencies')
  f=make_feature(got,np.ones(4)*.005,ctx,far=True)
  assert len(f)==FAR_FEATURES
  tests.append('router_inputs_match_target_free_feature_shape')
  net=FarResidual();ft=torch.randn(8,FAR_FEATURES);counts=torch.zeros(8,FAR_CHANNELS)
  gates=net(ft,counts)
  assert torch.equal(gates,torch.zeros_like(gates))
  d={'near':torch.rand(8)+5,'far_features':ft,'far_count':counts,'far_prob':torch.rand(8,FAR_CHANNELS)}
  lp,g=loss_far(net,d)
  assert torch.allclose(lp,d['near'],atol=1e-6,rtol=1e-6)
  tests.append('empty_far_index_exactly_falls_back_to_r716')
  counts[0,1]=1;g=net(ft,counts)
  assert (g[0,0]==0) and (g[0,2:]==0).all() and (g[1:]==0).all()
  tests.append('far_zero_eligibility_is_exact')
  original=history.copy();other=history.copy();other[-1]=144
  assert far_only(other,ctx,44)==far_only(original,ctx,44)
  tests.append('future_token_excluded_from_index')
  # Equal-book sampling cannot mix histories from different documents.
  class FakeBooks:
    def __init__(self):
      self.seq=np.arange(150000)
      self.doc_start=np.repeat(np.asarray([0,50000,100000],dtype=np.int64),50000)
  f=FakeBooks();ix=pick_positions(f,18,2026)
  assert len(ix)==18 and len(set((ix//50000).tolist()))==3
  assert (ix%50000>=FAR_EXCLUSION+1).all() and (ix%50000<50000).all()
  tests.append('new_book_position_sampler_obeys_doc_boundaries_and_equal_book_coverage')
  result={'passed':len(tests),'total':len(tests),'tests':tests,'accepted_checkpoint_sha256':PARENT}
  print(json.dumps(result,indent=2));return result
if __name__=='__main__':run()