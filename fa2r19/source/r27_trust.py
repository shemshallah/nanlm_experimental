"""FA2 R27: distance-aware counterfactual trust in the SAME R26 PAL model.

The original R26 full distribution and its R25+near-R26 counterfactual are
mixed by a new, zero-initialized gate. No original model parameter is modified.
All gate features are derived from causal prefix retrieval; never from targets.
"""
import torch
from torch import nn
from torch.nn import functional as F


def near_only_target(r26, base_y, target_cache_prob, features):
    """Recompute R26 with ONLY its 8K channels; keep R26 weights untouched."""
    if target_cache_prob.ndim != 2 or target_cache_prob.shape[1] != 6:
        raise ValueError("R26 target probabilities must be [T,6]")
    if features.shape != (len(target_cache_prob), 6, 4):
        raise ValueError("R26 features must be [T,6,4]")
    ps = torch.cat((target_cache_prob[:, :3],
                    torch.zeros_like(target_cache_prob[:, 3:])), dim=1)
    fs = torch.cat((features[:, :3], torch.zeros_like(features[:, 3:])), dim=1)
    return r26.target_logprob(base_y, ps, fs)


def near_only_full(r26, base_logprobs, candidates, features):
    """Full-vocabulary counterfactual for inference, not just labeled scoring."""
    if candidates.ndim != 3 or candidates.shape[1] != 6:
        raise ValueError("R26 candidates must be [T,6,V]")
    if features.shape != (len(candidates), 6, 4):
        raise ValueError("R26 features must be [T,6,4]")
    cs = torch.cat((candidates[:, :3], torch.zeros_like(candidates[:, 3:])), dim=1)
    fs = torch.cat((features[:, :3], torch.zeros_like(features[:, 3:])), dim=1)
    return r26(base_logprobs, cs, fs)


class CounterfactualTrustGate(nn.Module):
    """A causal gate that discounts the long R26 channels when untrustworthy.

    Both proposed distributions come from the SAME frozen unified model:
    full R26 and a counterfactual R26 with 128K channels masked out.
    At construction, the output is bitwise-identical to full R26. Learn a
    bounded near-only fallback; no separate language model is introduced.
    """
    def __init__(self, width=48):
        super().__init__()
        self.reversion = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, width),
                                  nn.SiLU(), nn.Linear(width, 1))
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.zero_()

    def fallback_fraction(self, features, position=0):
        if features.ndim != 3 or features.shape[1:] != (6, 4):
            raise ValueError("R26 features must be [T,6,4]")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("non-finite trust features")
        evidence = features.reshape(len(features), 24)
        near = (features[:, :3, 0] > 0)
        long = (features[:, 3:, 0] > 0)
        # R26 age is log(1+nearest_age)/log(1+horizon).
        # Compute observed age ONLY from prefix statistics, not the label.
        far_age = torch.expm1(features[:, 3:, 2] * torch.log1p(
            features.new_tensor(131072.0)))
        remote = long & (far_age > 8192.0)
        age_weight = long.float()
        summary = torch.stack((
            near.float().mean(-1), long.float().mean(-1),
            remote.float().mean(-1),
            features[:, :3, 1].amax(-1),
            features[:, 3:, 1].amax(-1),
            features[:, 3:, 2].amax(-1),
            (features[:, 3:, 3] * age_weight).sum(-1) /
                age_weight.sum(-1).clamp_min(1.0),
            (features[:, 3:, 0] * age_weight).sum(-1) /
                age_weight.sum(-1).clamp_min(1.0)
        ), dim=-1)
        score = self.gate(torch.cat((evidence, summary), dim=-1)).squeeze(-1)
        # Exact zero at init; useful gradients at zero. 0.95 safety bound.
        # Forward is exactly zero at initialization. The straight-through
        # term guarantees a nonzero derivative across PyTorch clamp-at-zero
        # conventions, while retaining a nonnegative forward mixture.
        raw = torch.tanh(self.reversion)
        positive = raw.clamp_min(0.0)
        budget = .95 * (positive + (raw - raw.detach()) *
                        (raw <= 0).to(raw.dtype))
        # Absolute query position is available before the target is read.
        # Preserve ALL inherited R26 near/33K/65K predictions exactly.
        if isinstance(position, torch.Tensor):
            pos = position.to(device=features.device, dtype=features.dtype)
        else:
            pos = features.new_tensor(position)
        if pos.ndim == 0:
            pos = pos + torch.arange(len(features), device=features.device)
        elif tuple(pos.shape) != (len(features),):
            raise ValueError("positions must be scalar start or [T]")
        mature = (pos >= 131072).to(features.dtype)
        return budget * torch.sigmoid(score) * long.any(-1).float() * mature

    def forward(self, full_logprobs, near_logprobs, features, position=0):
        if full_logprobs.shape != near_logprobs.shape:
            raise ValueError("full and near probabilities have different shapes")
        if full_logprobs.ndim not in (1, 2) or len(full_logprobs) != len(features):
            raise ValueError("expected [T] target or [T,V] full log probabilities")
        alpha = self.fallback_fraction(features, position)
        if full_logprobs.ndim == 2:
            alpha = alpha[:, None]
        # float64 log1p(expm1()) avoids float32 cancellation and preserves
        # exact full R26 output at alpha=0, without blocking gate gradients.
        full = full_logprobs.double()
        near = near_logprobs.double()
        result = full + torch.log1p(alpha.double() * torch.expm1(near-full))
        return result.to(full_logprobs.dtype)
