"""FA2 R27: distance-aware counterfactual trust in the same R26 PAL model."""
import torch
from torch import nn

def near_only_target(r26, base_y, target_cache_prob, features):
    if target_cache_prob.ndim!=2 or target_cache_prob.shape[1]!=6:raise ValueError('shape')
    ps=torch.cat((target_cache_prob[:,:3],torch.zeros_like(target_cache_prob[:,3:])),1)
    fs=torch.cat((features[:,:3],torch.zeros_like(features[:,3:])),1)
    return r26.target_logprob(base_y,ps,fs)

def near_only_full(r26,base_logprobs,candidates,features):
    cs=torch.cat((candidates[:,:3],torch.zeros_like(candidates[:,3:])),1)
    fs=torch.cat((features[:,:3],torch.zeros_like(features[:,3:])),1)
    return r26(base_logprobs,cs,fs)

class CounterfactualTrustGate(nn.Module):
    def __init__(self,width=48):
        super().__init__()
        self.reversion=nn.Parameter(torch.tensor(0.0))
        self.gate=nn.Sequential(nn.LayerNorm(32),nn.Linear(32,width),nn.SiLU(),nn.Linear(width,1))
        with torch.no_grad():
            self.gate[-1].weight.zero_();self.gate[-1].bias.zero_()
    def fallback_fraction(self,features,position=0):
        if features.ndim!=3 or features.shape[1:]!=(6,4):raise ValueError('shape')
        if not bool(torch.isfinite(features).all()):raise ValueError('nonfinite features')
        evidence=features.reshape(len(features),24)
        near=features[:,:3,0]>0
        long=features[:,3:,0]>0
        far_age=torch.expm1(features[:,3:,2]*torch.log1p(features.new_tensor(131072.0)))
        remote=long&(far_age>8192.0)
        aw=long.float()
        summary=torch.stack((near.float().mean(-1),long.float().mean(-1),remote.float().mean(-1),
            features[:,:3,1].amax(-1),features[:,3:,1].amax(-1),features[:,3:,2].amax(-1),
            (features[:,3:,3]*aw).sum(-1)/aw.sum(-1).clamp_min(1.),
            (features[:,3:,0]*aw).sum(-1)/aw.sum(-1).clamp_min(1.)),dim=-1)
        score=self.gate(torch.cat((evidence,summary),dim=-1)).squeeze(-1)
        raw=torch.tanh(self.reversion)
        positive=raw.clamp_min(0.)
        budget=.95*(positive+(raw-raw.detach())*(raw<=0).to(raw.dtype))
        pos=position.to(device=features.device,dtype=features.dtype) if isinstance(position,torch.Tensor) else features.new_tensor(position)
        if pos.ndim==0:pos=pos+torch.arange(len(features),device=features.device)
        elif pos.shape!=(len(features),):raise ValueError('position shape')
        return budget*torch.sigmoid(score)*long.any(-1).float()*(pos>=131072).float()
    def forward(self,full_logprobs,near_logprobs,features,position=0):
        if full_logprobs.shape!=near_logprobs.shape:raise ValueError('shape')
        alpha=self.fallback_fraction(features,position)
        if full_logprobs.ndim==2:alpha=alpha[:,None]
        full=full_logprobs.double();near=near_logprobs.double()
        return (full+torch.log1p(alpha.double()*torch.expm1(near-full))).to(full_logprobs.dtype)