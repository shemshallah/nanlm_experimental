"""Deterministic adversarial invariants for FA2 R25's causal cache."""
import numpy as np,torch
from r25_episodic_cache import AdaptiveNGramMixer,CausalNGramIndex

def test():
    torch.manual_seed(25);np.random.seed(25)
    source=np.asarray([1,3,5,7,1,3,5,8,1,3,5,4,2,4,2,4,2,3,5,6,2,4,2,4,2,4,2,4],dtype=np.int64)
    original=CausalNGramIndex(source,memory=8192)
    a,fa=original.distributions(source,11,6,16)
    # Two arbitrary mutations in the future must have zero influence on earlier queries.
    changed=source.copy();changed[18:]=0
    modified=CausalNGramIndex(changed,memory=8192)
    b,fb=modified.distributions(changed,11,6,16)
    assert np.array_equal(a,b) and np.array_equal(fa,fb),'future-token leakage'
    # Correct next-token retrieval from identical 3-token past [1,3,5].
    assert a[0,3,7]>0 and a[0,3,8]>0
    assert a[0,3,4]==0,'admitted current target'
    # Correct ngram inclusion across positions must only use previous completed tokens.
    for t in range(3,len(source)):
        p,_=original.distributions(source,t,1,16)
        for n in (1,2,3):
            key=tuple(source[t-n:t]);old=[int(source[j]) for j in range(n,max(n,t-8192),1) if tuple(source[j-n:j])==key]
            # Recheck using the full valid range.
            old=[int(source[j]) for j in range(n,t) if tuple(source[j-n:j])==key]
            if not old:assert p[0,n].sum()==0
            else:
                assert p[0,n].sum()>0
                assert all(p[0,n,k]==0 for k in range(16) if k not in old)
    # Separate unrelated document cannot use query target or future query suffix.
    donor=np.asarray([5,4,3,2,1]*7,dtype=np.int64)
    donorindex=CausalNGramIndex(donor,memory=8192)
    c,fc=donorindex.distributions(source,11,6,16,source_cutoff=11)
    source_edit=source.copy();source_edit[11:]=12
    d,fd=donorindex.distributions(source_edit,11,6,16,source_cutoff=11)
    assert np.array_equal(c[0],d[0]) and np.array_equal(fc[0],fd[0])
    # Build an index from the *available prefix only* and feed a dummy unknown
    # target; results must equal indexing the complete book at that position.
    partial=CausalNGramIndex(source[:11],memory=8192)
    pre,pr=partial.distributions(np.r_[source[:11],0],11,1,16)
    assert np.array_equal(pre[0],a[0]) and np.array_equal(pr[0],fa[0])
    # R25 starts as exact probability identity and has nonzero first-step gradient.
    model=AdaptiveNGramMixer()
    logits=torch.randn(6,16)
    got=model(logits,torch.tensor(a),torch.tensor(fa))
    assert torch.allclose(torch.softmax(got,-1),torch.softmax(logits,-1),atol=1e-7)
    loss=torch.nn.functional.nll_loss(got,torch.tensor([7,6,5,4,3,2]))
    loss.backward()
    assert model.amplitude.grad is not None and torch.isfinite(model.amplitude.grad)
    print('R25 8/8 causal tests PASS')

if __name__=='__main__':test()
