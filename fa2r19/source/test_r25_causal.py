"""Deterministic adversarial checks for the R25 causal episodic cache."""
import numpy as np
import torch
from r25_episodic_cache import AdaptiveNGramMixer, CausalNGramIndex

def main():
    torch.manual_seed(25)
    source=np.asarray([1,3,5,7,1,3,5,8,1,3,5,4,2,4,2,4,2,3,5,6,2,4,2,4,2,4,2,4],dtype=np.int64)
    ix=CausalNGramIndex(source, memory=8192)
    c,feat=ix.distributions(source,11,6,16)
    changed=source.copy();changed[18:]=0
    alt,a_feat=CausalNGramIndex(changed,memory=8192).distributions(changed,11,6,16)
    assert np.array_equal(c,alt) and np.array_equal(feat,a_feat), "future leakage"
    assert c[0,3,7]>0 and c[0,3,8]>0 and c[0,3,4]==0, "target admitted"
    for t in range(3,len(source)):
        probs,_=ix.distributions(source,t,1,16)
        for n in (1,2,3):
            key=tuple(source[t-n:t])
            old=[int(source[j]) for j in range(n,t) if tuple(source[j-n:j])==key]
            if not old:assert probs[0,n].sum()==0
            else:
                assert probs[0,n].sum()>0
                assert all(probs[0,n,k]==0 for k in range(16) if k not in old)
    donor=CausalNGramIndex(np.asarray([5,4,3,2,1]*7,dtype=np.int64),memory=8192)
    d,df=donor.distributions(source,11,6,16,source_cutoff=11)
    suffix=source.copy();suffix[11:]=12
    e,ef=donor.distributions(suffix,11,6,16,source_cutoff=11)
    assert np.array_equal(d[0],e[0]) and np.array_equal(df[0],ef[0])
    # Prefix-only construction: no future tokens even allocated in source.
    prefix_ix=CausalNGramIndex(source[:11],memory=8192)
    prefix,_=prefix_ix.distributions(np.r_[source[:11],0],11,1,16)
    assert np.array_equal(prefix[0],c[0])
    mix=AdaptiveNGramMixer()
    logits=torch.randn(6,16)
    out=mix(logits,torch.from_numpy(c),torch.from_numpy(feat))
    assert torch.allclose(out.softmax(-1),logits.softmax(-1),atol=1e-7)
    torch.nn.functional.nll_loss(out,torch.tensor([7,6,5,4,3,2])).backward()
    assert mix.amplitude.grad is not None and torch.isfinite(mix.amplitude.grad)
    print("R25 causal / identity checks PASS")

if __name__=="__main__":
    main()
