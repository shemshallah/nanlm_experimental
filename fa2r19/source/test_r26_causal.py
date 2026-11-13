"""R26 tests: local and far memory, target isolation, identity, gradients."""
import numpy as np,torch
from r26_span_memory import CausalHighOrderIndex,AdaptiveSpanMixer,ORDERS,CHANNELS

def test():
    torch.manual_seed(26)
    x=np.array(([1,2,3,4,5,6,7,8]*8)+([9,3,2,4,5,6,7,8]*8),dtype=np.int64)
    vocab=12; i=CausalHighOrderIndex(x)
    for q in range(8,len(x)-3):
        b=np.ones((1,vocab),dtype=np.float32)/vocab
        p,feats=i.features(x,q,1,b,x[q:q+1])
        for k,(n,h) in enumerate(CHANNELS):
            js=[j for j in range(n,q) if tuple(x[j-n:j])==tuple(x[q-n:q]) and j>=q-h]
            if not js:assert p[0,k]==0 and feats[0,k,0]==0
            else:
                assert p[0,k]>=0 and feats[0,k,0]>0
                assert p[0,k]==0 if x[q] not in x[js] else p[0,k]>0
    q=31;prob=np.ones((4,vocab),dtype=np.float32)/vocab
    p,feats=i.features(x,q,4,prob,x[q:q+4])
    truncated=CausalHighOrderIndex(x[:q]);pp,ff=truncated.features(x,q,1,prob[:1],x[q:q+1])
    assert np.array_equal(pp[0],p[0]) and np.array_equal(ff[0],feats[0])
    tampered=x.copy();tampered[q+3:]=11
    ix2=CausalHighOrderIndex(tampered)
    pp2,ff2=ix2.features(tampered,q,3,prob[:3],tampered[q:q+3])
    assert np.array_equal(pp2,p[:3]) and np.array_equal(ff2,feats[:3]),'future changes affected past'
    # Capped indexing must be bitwise identical to full-document indexing
    # for every query wholly inside the retained causal prefix.
    cap=CausalHighOrderIndex(x[:50])
    safe_q=38
    fullp,fullf=i.features(x,safe_q,4,prob,x[safe_q:safe_q+4])
    capp,capf=cap.features(x,safe_q,4,prob,x[safe_q:safe_q+4])
    assert np.array_equal(fullp,capp) and np.array_equal(fullf,capf)
    # Changing the label of the current prediction cannot change gate data.
    altered=x.copy();altered[safe_q]=11
    _,altered_feats=i.features(altered,safe_q,1,prob[:1],altered[safe_q:safe_q+1])
    assert np.array_equal(altered_feats[0],fullf[0])
    donor=CausalHighOrderIndex(x[::-1].copy())
    d,fd=donor.features(x,q,4,prob,x[q:q+4],source_cutoff=q)
    assert d.shape==(4,6) and fd.shape==(4,6,4)
    out=donor.full_distributions(x,q,4,vocab,source_cutoff=q)
    assert np.allclose(out.sum(-1)[fd[:,:,0]>0],1,atol=1e-6)
    m=AdaptiveSpanMixer();base=torch.log_softmax(torch.randn(4,vocab),dim=-1)
    targs=torch.from_numpy(x[q:q+4].copy()).long();tc=torch.from_numpy(p);tf=torch.from_numpy(feats)
    z=m.target_logprob(base.gather(1,targs[:,None]).squeeze(-1),tc,tf)
    assert torch.allclose(z,base.gather(1,targs[:,None]).squeeze(-1),atol=1e-7)
    result=m(base,torch.from_numpy(i.full_distributions(x,q,4,vocab)),tf)
    assert torch.allclose(result.exp(),base.exp(),atol=1e-7)
    (-z.mean()).backward();assert torch.isfinite(m.amplitude.grad)
    with torch.no_grad():m.amplitude.copy_(torch.tensor(.75))
    re=m(base,torch.from_numpy(i.full_distributions(x,q,4,vocab)),tf)
    point=m.target_logprob(base.gather(1,targs[:,None]).squeeze(-1),tc,tf)
    assert torch.allclose(re.gather(1,targs[:,None]).squeeze(-1),point,atol=1e-6)
    print('R26 8/8 causality, probability and identity checks PASS')
if __name__=='__main__':test()
