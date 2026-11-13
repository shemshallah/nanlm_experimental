"""FA2 R27 trust: exact parent parity, causal gating and gradient tests."""
import numpy as np
import torch

from r26_span_memory import AdaptiveSpanMixer, CausalHighOrderIndex
from r27_trust import CounterfactualTrustGate, near_only_target, near_only_full


def run():
    torch.manual_seed(27)
    vocab=24
    tokens=np.tile(np.array([1,2,3,4,5,6,7,8],dtype=np.int32),12)
    idx=CausalHighOrderIndex(tokens)
    q=64
    T=8
    base=torch.log_softmax(torch.randn(T,vocab),dim=-1)
    target=torch.as_tensor(tokens[q:q+T].copy(),dtype=torch.long)
    cache,features=idx.features(tokens,q,T,base.exp().numpy(),target.numpy())
    full_candidates=idx.full_distributions(tokens,q,T,vocab)
    feats=torch.from_numpy(features)
    c=torch.from_numpy(cache)
    cs=torch.from_numpy(full_candidates)
    r26=AdaptiveSpanMixer()
    with torch.no_grad():
        r26.amplitude.fill_(.90)
    base_y=base.gather(1,target[:,None]).squeeze(-1)
    r26_full=r26(base,cs,feats)
    r26_near=near_only_full(r26,base,cs,feats)
    r26_y=r26.target_logprob(base_y,c,feats)
    r26_near_y=near_only_target(r26,base_y,c,feats)
    assert torch.allclose(r26_full.gather(1,target[:,None]).squeeze(-1),r26_y,atol=1e-6)
    assert torch.allclose(r26_near.gather(1,target[:,None]).squeeze(-1),r26_near_y,atol=1e-6)
    gate=CounterfactualTrustGate()
    assert torch.equal(gate(r26_full,r26_near,feats,position=132000),r26_full),"R26 full logits parity lost"
    assert torch.equal(gate(r26_y,r26_near_y,feats,position=132000),r26_y),"R26 target parity lost"
    # Modified target label alone cannot change gate inputs or fallback.
    alternative=target.clone();alternative[0]=(alternative[0]+3)%vocab
    alt_cache,alt_features=idx.features(tokens,q,T,base.exp().numpy(),alternative.numpy())
    assert np.array_equal(alt_features,features)
    assert np.allclose(gate.fallback_fraction(feats).detach().numpy(),
                       gate.fallback_fraction(torch.from_numpy(alt_features)).detach().numpy())
    # No far-channel candidate => exact R26 identity even with positive fallback.
    without_long=feats.clone();without_long[:,3:]=0.
    with torch.no_grad():
        gate.reversion.fill_(.85)
        gate.gate[-1].bias.fill_(4.)
    out=gate(r26_full,r26_near,without_long,position=132000)
    assert torch.equal(out,r26_full)
    # All full-vocab rows are valid probability distributions (true mixtures).
    with torch.no_grad():
        gate.gate[-1].bias.zero_()
    blend=gate(r26_full,r26_near,feats,position=132000)
    assert torch.allclose(blend.exp().sum(-1),torch.ones(T),atol=5e-6)
    assert torch.isfinite(blend).all()
    # Both model arms share EXACT same R26 parameters (nothing trained in R26).
    old={k:v.detach().clone() for k,v in r26.state_dict().items()}
    gate.zero_grad(set_to_none=True)
    with torch.no_grad():
        gate.reversion.zero_()
    # Force distinct counterfactual outcomes: the repeated toy text can otherwise
    # make near and far proposals exactly equal, yielding a legitimate zero gradient.
    # Positions strictly under 128K cannot be modified, even when enabled.
    assert torch.equal(gate(r26_y,r26_near_y + .25,feats,position=33000),r26_y)
    loss=-gate(r26_y,r26_near_y + .25,feats,position=132000).mean()
    loss.backward()
    assert gate.reversion.grad is not None
    assert torch.isfinite(gate.reversion.grad)
    assert torch.abs(gate.reversion.grad)>0,"no learning signal at identity init"
    assert all(torch.equal(v,r26.state_dict()[k]) for k,v in old.items())
    # Proper causal prefix index: changing future never changes a past gate.
    changed=tokens.copy();changed[q+T:]=18
    fresh=CausalHighOrderIndex(changed)
    _,oldf=idx.features(tokens,q,T,base.exp().numpy(),target.numpy())
    _,newf=fresh.features(changed,q,T,base.exp().numpy(),target.numpy())
    assert np.array_equal(oldf,newf)
    print("R27 PASS: full/target parity, near-only, target isolation, "
          "probability normalization, gradients, original weights, future isolation")


if __name__=="__main__":
    run()
