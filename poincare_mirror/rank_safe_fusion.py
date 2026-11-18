"""R67 causal, mass-preserving calibration of the existing FA2 memory rail."""
import torch
from torch.nn import functional as F


def protect_parent_space_rank(parent,candidate):
    """Maximal convex joint-memory blend retaining a prior space/EOS winner.

    The guard reads only the inherited logits and candidate logits for this
    prediction. No targets, future bytes, separate model, or optimizer needed.
    """
    if parent.shape!=candidate.shape or parent.shape[-1]!=257:
        raise ValueError("Expected matched 257-way parent/candidate logits")
    p=F.softmax(parent.float(),-1)
    q=F.softmax(candidate.float(),-1)
    winner=parent.argmax(-1)
    protected=winner.eq(32)|winner.eq(256)
    winning_p=p.gather(-1,winner.unsqueeze(-1))
    delta=q-p
    relative_change=delta-delta.gather(-1,winner.unsqueeze(-1))
    allowed=(winning_p-p-1e-5*winning_p)/relative_change.clamp_min(1e-30)
    other=torch.arange(257,device=parent.device).ne(winner.unsqueeze(-1))
    allowed=torch.where((relative_change>0)&other,allowed,
                        torch.ones_like(allowed))
    alpha=allowed.clamp(0.,1.).min(-1).values
    alpha=torch.where(protected,alpha,torch.ones_like(alpha))
    return (p+alpha.unsqueeze(-1)*delta).clamp_min(1e-30).log()


def protect_parent_winner_rank(parent, candidate):
    """R70 opt-in, maximal memory blend retaining *every* parent top-1 class.

    No labels, DEV statistics, future tokens or second model are consulted.
    The probability simplex and parent argmax are both preserved; future
    memory can still redistribute nonwinning mass and reduce cross entropy.
    This is intentionally an optional candidate, not R69 default behavior.
    """
    if parent.shape != candidate.shape or parent.shape[-1] != 257:
        raise ValueError('Expected matched 257-class future logits')
    p = F.softmax(parent.float(), -1)
    q = F.softmax(candidate.float(), -1)
    winner = parent.argmax(-1)
    p_winner = p.gather(-1, winner.unsqueeze(-1))
    delta = q - p
    rival_gain = delta - delta.gather(-1, winner.unsqueeze(-1))
    cap = (p_winner-p-1e-5*p_winner)/rival_gain.clamp_min(1e-30)
    other = torch.arange(257, device=p.device).ne(winner.unsqueeze(-1))
    cap = torch.where((rival_gain>0) & other, cap,
                      torch.ones_like(cap))
    alpha = cap.clamp(0., 1.).min(-1).values
    result = (p + alpha.unsqueeze(-1)*delta).clamp_min(1e-30)
    return result.log()
