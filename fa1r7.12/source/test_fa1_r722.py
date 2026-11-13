#!/usr/bin/env python3
"""R7.22 standalone deterministic CPU tests; no checkpoint or corpus required."""
import json
import numpy as np
import torch
from fa1_r722_semantic import CausalSemanticIndex,SemanticGate,STRIDE,HORIZON,EXCLUSION

def tests():
    checks={}
    rng=np.random.default_rng(722)
    vocab=768;N=175000
    embeddings=rng.normal(0,1,(vocab,96)).astype(np.float32)
    # Synthetic embedding analogs test the mechanism, not natural-language meaning.
    embeddings[11]=embeddings[10]+rng.normal(0,.005,96)
    seq=rng.integers(2,vocab,size=N,dtype=np.int32)
    # Stable 4-word contexts well inside the far memory band.
    query=135000
    seq[query-4:query]=[5,6,7,10]
    older=query-45000
    older=(older//STRIDE)*STRIDE
    seq[older-3:older+1]=[5,6,7,11]
    seq[older+1]=22
    idx=CausalSemanticIndex(seq,embeddings,query,0)
    a=idx.query(target=22)
    checks['candidate_in_old_far_band']=bool(a['n'] and np.all(a['positions']+1<query-EXCLUSION) and
                                 np.all(a['positions']>=query-HORIZON))
    checks['paraphrase_like_embedding_has_support']=bool(older in a['positions'])
    feat=a['feature'].copy()
    b=idx.query(target=23)
    checks['target_free_routing']=bool(np.array_equal(feat,b['feature']) and a['modal']==b['modal'] and
                                            np.array_equal(a['positions'],b['positions']))
    seq2=seq.copy();seq2[query:]=rng.integers(2,vocab,size=N-query)
    x=CausalSemanticIndex(seq2,embeddings,query,0).query(target=22)
    checks['future_token_mutation_cannot_change_retrieval']=bool(np.array_equal(x['positions'],a['positions'])
                               and np.array_equal(x['feature'],a['feature']))
    # Retroactively edit a target token not read by this query; features remain
    # target-free, but likelihood changes when target changes.
    idx.advance(query+1)
    c=idx.query(target=22)
    rebuild=CausalSemanticIndex(seq,embeddings,query+1,0).query(target=22)
    checks['incremental_same_as_full_build']=bool(np.array_equal(c['positions'],rebuild['positions'])
        and np.allclose(c['feature'],rebuild['feature'],rtol=1e-6,atol=1e-6)
        and abs(c['target_probability']-rebuild['target_probability'])<1e-8)
    # Boundary across otherwise matching histories must never be indexed.
    idx2=CausalSemanticIndex(seq,embeddings,query,query-40000)
    z=idx2.query(target=22)
    checks['document_boundary_excludes_earlier_history']=bool(older not in z.get('positions',()))
    # Hard stop at far band exclusion, even when the surface pattern matches.
    too_recent=((query-10000)//STRIDE)*STRIDE
    seq3=seq.copy();seq3[too_recent-3:too_recent+1]=[5,6,7,10];seq3[too_recent+1]=22
    result=CausalSemanticIndex(seq3,embeddings,query,0).query(target=22)
    checks['near_band_excluded']=bool(too_recent not in result.get('positions',()))
    try:CausalSemanticIndex(seq,embeddings,query).advance(query-1)
    except ValueError:checks['backward_seeks_fail_closed']=True
    else:checks['backward_seeks_fail_closed']=False
    # A second, disjoint long tier reaches up to 1,048,576 past tokens.
    longN=1180000; longpos=1120000;longpast=((longpos-900000)//32)*32
    longseq=rng.integers(2,vocab,size=longN,dtype=np.int32)
    longseq[longpos-4:longpos]=[5,6,7,10]
    longseq[longpast-3:longpast+1]=[5,6,7,11];longseq[longpast+1]=22
    longidx=CausalSemanticIndex(longseq,embeddings,longpos,0,stride=32,
             horizon=1048576,exclusion=131072)
    longhit=longidx.query(target=22)
    checks['one_megatoken_far_band_reachable']=bool(longpast in longhit.get('positions',()) and
               np.all(longhit['positions']<=longpos-131072-2))
    longseq2=longseq.copy();longseq2[longpos:]=2
    longctrl=CausalSemanticIndex(longseq2,embeddings,longpos,0,stride=32,
             horizon=1048576,exclusion=131072).query(target=22)
    checks['one_megatoken_future_safe']=bool(np.array_equal(longctrl['positions'],longhit['positions'])
              and np.array_equal(longctrl['feature'],longhit['feature']))
    longidx.advance(longpos+1)
    longnext=longidx.query(target=22)
    longreset=CausalSemanticIndex(longseq,embeddings,longpos+1,0,stride=32,
             horizon=1048576,exclusion=131072).query(target=22)
    checks['one_megatoken_incremental_parity']=bool(np.array_equal(longnext['positions'],longreset['positions'])
          and np.allclose(longnext['feature'],longreset['feature'],atol=1e-6))
    torch.manual_seed(722)
    gate=SemanticGate();gate.eval()
    l=torch.tensor([5.1,4.2,6.7])
    f=torch.randn(3,16)
    p=torch.rand(3)
    n=torch.tensor([1.,0.,9.])
    out,w=gate(l,f,p,n)
    checks['identity_parity_exact']=bool(torch.equal(out,l) and torch.count_nonzero(w)==0)
    gate.train();loss,weight=gate(l,f,p,n)
    loss.mean().backward()
    checks['joint_gate_gradient']=bool(gate.open_fraction.grad is not None and gate.open_fraction.grad.abs().item()>0)
    gate.eval();gate.open_fraction.data.fill_(1.)
    new,w=gate(l,f,p,n)
    checks['bounded_nonnegative_gate']=bool(torch.all(w>=0) and torch.all(w<=.125) and w[1]==0)
    checks['mixture_matches_closed_form']=bool(torch.allclose(new,-torch.log((1-w)*torch.exp(-l)+w*p),atol=1e-6))
    passed=sum(checks.values())
    assert passed==len(checks),checks
    return {'round':'FA1 R7.22','tests_passed':passed,'tests_total':len(checks),'checks':checks,
            'synthetic_only':True,'natural_language_semantic_gain_proven':False,'promoted':False}
if __name__=='__main__':print(json.dumps(tests(),indent=2))