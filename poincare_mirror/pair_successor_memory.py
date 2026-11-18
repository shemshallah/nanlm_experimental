"""R54: strictly causal, exact-pair successor-byte memory for the unified FA2 LM.

At position t an address match at u<t contributes only the already observed
input byte at u+1<=t. No target label or future token enters the memory.
"""
import math

import torch
from rank_safe_fusion import protect_parent_space_rank, protect_parent_winner_rank
from torch import nn


def retrieve_pair_successors(pair_ids, observed, horizon=32, vocab=257,
                             query_map=None, value_control="exact",
                             key_mode="pair", query_byte_map=None,
                             engine="vectorized"):
    """Return B,T,V next-byte histograms, B,T edge coverage, B,T hit positions.

    The wrong-value control preserves matched locations and coverage. R59
    compound modes require exact pair AND current observed byte (optionally
    its predecessor) to match earlier observed keys. Every successor is x[u+1]
    with u<t, hence x[u+1] is already available at prediction position t.
    """
    if pair_ids.ndim != 3 or observed.ndim != 2 or pair_ids.shape[:2] != observed.shape:
        raise ValueError("Need matching B,T,E pair IDs and B,T observed bytes")
    if pair_ids.dtype != torch.long or observed.dtype != torch.long:
        raise ValueError("Pair addresses and observed bytes must use torch.long")
    if pair_ids.device != observed.device:
        raise ValueError("Addresses and bytes must use the same device")
    if engine not in ("vectorized","indexed"):
        raise ValueError("Unknown causal retrieval engine")
    if engine == "indexed" and pair_ids.device.type != "cpu":
        raise ValueError("Indexed causal retrieval is CPU-only")
    if (horizon < 1 or vocab < 2 or value_control not in ("exact", "rotated")
            or key_mode not in ("pair", "pair_byte", "pair_bigram",
                                "head_byte", "hierarchical")):
        raise ValueError("Invalid memory horizon, vocabulary or value control")
    if query_map is not None and (query_map.shape != (32640,) or
                                  query_map.dtype != torch.long):
        raise ValueError("Query map must be 32640 integer entries")
    if query_byte_map is not None and (query_byte_map.shape != (vocab,) or
                                       query_byte_map.dtype != torch.long):
        raise ValueError("Query byte map must have one integer per vocabulary entry")
    if torch.any((pair_ids < 0) | (pair_ids >= 32640)) or torch.any(
            (observed < 0) | (observed >= vocab)):
        raise ValueError("Addresses or observed bytes out of range")
    batch, length, edges = pair_ids.shape
    query_map = query_map.to(pair_ids.device) if query_map is not None else None
    query_byte_map = (query_byte_map.to(observed.device)
                      if query_byte_map is not None else None)
    if query_byte_map is not None and torch.any(
            (query_byte_map < 0) | (query_byte_map >= vocab)):
        raise ValueError("Query byte remapping exceeds vocabulary")
    if engine == "indexed":
        from indexed_pair_successor import retrieve_indexed_pair_successors
        return retrieve_indexed_pair_successors(
            pair_ids,observed,horizon,vocab,query_map,value_control,
            key_mode,query_byte_map)
    previous = torch.cat((observed[:, :1], observed[:, :-1]), dim=1)
    # R60: O(32640) canonical-index lookup; evaluated only for coarse keys.
    # The exact canonical encoding is a*(511-a)//2 + b-a-1 for a<b.
    endpoints = (torch.triu_indices(256,256,1,device=pair_ids.device)
                 if key_mode in ("head_byte","hierarchical") else None)
    histograms = [torch.zeros(batch, vocab, device=pair_ids.device)]
    coverage = [torch.zeros(batch, device=pair_ids.device)]
    positions = [torch.zeros(batch, device=pair_ids.device)]
    for t in range(1, length):
        first = max(0, t-horizon)
        keys = pair_ids[:,first:t,:]
        query = pair_ids[:,t,:]
        if query_map is not None:
            query = query_map[query]  # only queries remapped, never stored keys
        match = query[:,:,None,None].eq(keys[:,None,:,:])  # B,E,H,E
        if key_mode != "pair":
            query_byte = observed[:, t]
            if query_byte_map is not None:
                query_byte = query_byte_map[query_byte]
            # Compound keys are formed from OBSERVED input x[t], never a label.
            match = match & query_byte[:,None,None,None].eq(
                observed[:,first:t][:,None,:,None])
            if key_mode in ("head_byte", "hierarchical"):
                # Shared endpoint allows a causal geometric backoff even when
                # exact pair IDs do not repeat within a short model window.
                q_a,q_b = endpoints[0][query],endpoints[1][query]  # B,E
                k_a,k_b = endpoints[0][keys],endpoints[1][keys]  # B,H,E
                overlap = (q_a[:,:,None,None].eq(k_a[:,None,:,:]) |
                           q_a[:,:,None,None].eq(k_b[:,None,:,:]) |
                           q_b[:,:,None,None].eq(k_a[:,None,:,:]) |
                           q_b[:,:,None,None].eq(k_b[:,None,:,:]))
                head_match = overlap & query_byte[:,None,None,None].eq(
                    observed[:,first:t][:,None,:,None])
                if key_mode == "hierarchical":
                    # Exact compound matches win per query edge. Only absent
                    # exact matches fall back to shared-head lexical matches.
                    has_exact = match.any(dim=(2,3),keepdim=True)
                    match = torch.where(has_exact,match,head_match)
                    confidence = torch.where(has_exact.squeeze(-1).squeeze(-1),
                                             1.0,0.25)
                else:
                    match = head_match
                    confidence = torch.full((batch,edges),0.25,device=observed.device)
            if key_mode == "pair_bigram":
                # No invented predecessor at u=0: that historical key is invalid.
                valid = torch.arange(first,t,device=observed.device).ge(1)
                match = match & valid[None,None,:,None]
                match = match & observed[:,t-1][:,None,None,None].eq(
                    previous[:,first:t][:,None,:,None])
        hit_counts = match.sum((2,3))                       # B,E
        active = hit_counts > 0
        by_time = match.sum(-1).float() / hit_counts.clamp_min(1)[:,:,None]
        historical_weight = by_time.sum(1) / active.sum(1).clamp_min(1)[:,None]
        # For u in [first,t), the successor x[u+1] is present among inputs
        # x[0:t+1]. The target x[t+1] is never read, even when u=t-1.
        successors = observed[:,first+1:t+1]
        if value_control == "rotated":
            successors = successors.roll(1,dims=1)  # same observed multiset
        hist = torch.zeros(batch,vocab,device=pair_ids.device)
        hist.scatter_add_(1,successors,historical_weight)
        histograms.append(hist)
        if key_mode in ("head_byte","hierarchical"):
            # Head-only backoff contributes at quarter-confidence; exact
            # matches retain R59 unit confidence. This is not a probability.
            coverage.append((active.float()*confidence).mean(1))
        else:
            coverage.append(active.float().mean(1))
        positions.append(match.any(dim=(1,3)).float().sum(1))
    return (torch.stack(histograms,1),
            torch.stack(coverage,1),
            torch.stack(positions,1))


class PairSuccessorTrust(nn.Module):
    """Causal successor recall with optional mass-preserving content expert.

    R58 adds a second four-parameter trust rail. R59 adds a third optional
    four-parameter rail using compound (geometric-pair, observed-byte) keys.
    Both are zero-initialized expansions of the one inherited FA2 model.
    The rails adjust the *conditional*
    distribution among 255 non-space, non-EOS bytes while retaining exactly
    the original group log-sum-exp, space logit and EOS logit. The inherited
    general trust rail and its Adam moments remain unchanged.
    """
    def __init__(self, horizon=32, smoothing=0.10, conditional_content=False,
                 joint_content=False, future_content=False):
        super().__init__()
        if horizon < 1 or not 0.0 < smoothing < 1.0:
            raise ValueError("Invalid local memory settings")
        if joint_content and not conditional_content:
            raise ValueError("R59 joint rail requires R58 content rail")
        self.weight = nn.Parameter(torch.zeros(4))
        self.register_parameter('content_weight',
                                nn.Parameter(torch.zeros(4)) if conditional_content else None)
        self.register_parameter('joint_weight',
                                nn.Parameter(torch.zeros(4)) if joint_content else None)
        self.register_parameter('future_weight',
                                nn.Parameter(torch.zeros(2,4)) if future_content else None)
        # R74 optional counterfactual readout; accepted R69 is unchanged.
        self.register_parameter('counterfactual_weight',None)
        self.horizon = horizon
        self.smoothing = smoothing
        self.query_map = None
        self.value_control = "exact"
        self.conditional_query_map = None
        self.conditional_value_control = 'exact'
        self.joint_query_map = None
        self.joint_value_control = 'exact'
        self.joint_query_byte_map = None
        self.joint_key_mode = 'pair_byte'
        self.space_rank_guard = False
        self.read_engine = "vectorized"

    def enable_conditional_content(self):
        """Add one zero-init tensor to the SAME module; older outputs are exact."""
        if self.content_weight is not None:
            raise ValueError('Conditional content expert already enabled')
        self.content_weight = nn.Parameter(torch.zeros_like(self.weight.detach()))
        return self.content_weight

    def enable_joint_content(self):
        """R59: add exactly one zero tensor; retain all inherited Adam slots."""
        if self.content_weight is None:
            raise ValueError("R59 requires R58 content capacity")
        if self.joint_weight is not None:
            raise ValueError("R59 joint rail already enabled")
        self.joint_weight = nn.Parameter(torch.zeros_like(self.weight.detach()))
        return self.joint_weight

    def enable_future_content(self):
        """R69: two zero-initialized temporal rails in this SAME module."""
        if self.joint_weight is None or self.future_weight is not None:
            raise ValueError('Future recall requires the original compound rail')
        self.future_weight=nn.Parameter(torch.zeros(2,4,device=self.joint_weight.device,
                                                    dtype=self.joint_weight.dtype))
        return self.future_weight

    def enable_counterfactual_content(self):
        if self.joint_weight is None or self.counterfactual_weight is not None:
            raise ValueError('R74 needs the inherited joint rail')
        self.counterfactual_weight=nn.Parameter(torch.zeros(
            4,device=self.joint_weight.device,dtype=self.joint_weight.dtype))
        return self.counterfactual_weight

    def counterfactual_forward(self,base,exact,negative):
        """Learn a contrast between matching causal factual and corrupt reads.

        The caller must independently construct both histories from the
        observed prefix, never the next-byte target or another book.
        """
        if self.counterfactual_weight is None:
            raise ValueError('R74 counterfactual readout is disabled')
        if base.ndim!=3 or base.shape[-1]!=257 or len(exact)!=3 or len(negative)!=3:
            raise ValueError('R74 requires aligned B,T,257 logits and triples')
        eh,ec,ep=exact
        nh,nc,np=negative
        for h,c,p in (exact,negative):
            if (h.shape!=base.shape or c.shape!=base.shape[:2] or
                    p.shape!=base.shape[:2] or not torch.isfinite(h).all() or
                    not torch.isfinite(c).all() or not torch.isfinite(p).all() or
                    torch.any(h<0) or torch.any(h.sum(-1)>1.00001) or
                    torch.any(c<0) or torch.any(c>1) or torch.any(p<0)):
                raise ValueError('Invalid R74 causal evidence')
        if (not torch.allclose(ec,nc,atol=1e-6,rtol=0) or
                not torch.allclose(ep,np,atol=1e-6,rtol=0)):
            raise ValueError('R74 requires matching control locations')
        with torch.no_grad():
            smooth_e=(eh.float()*(1-self.smoothing)+self.smoothing/257).clamp_min(1e-12)
            smooth_n=(nh.float()*(1-self.smoothing)+self.smoothing/257).clamp_min(1e-12)
            log_ratio=(smooth_e.log()-smooth_n.log()).clamp(-3.,3.)
            tv=(eh.float()-nh.float()).abs().sum(-1)*.5
            baseline=base.detach().float().softmax(-1)
            advantage=((eh.float()-nh.float())*baseline).sum(-1)
            entropy=-(eh.float()*eh.float().clamp_min(1e-12).log()).sum(-1)/math.log(257)
            features=torch.stack((ec.float(),tv,advantage,1-entropy),-1)
        strength=(torch.tanh((features.to(self.counterfactual_weight.dtype)*
                             self.counterfactual_weight).sum(-1))*
                  ec.to(self.counterfactual_weight.dtype)*tv.to(
                      self.counterfactual_weight.dtype))
        mask=torch.ones(257,device=base.device,dtype=torch.bool)
        mask[32]=False
        mask[256]=False
        before=base[...,mask]
        proposed=before+(strength[...,None]*log_ratio[...,mask]).to(base.dtype)
        offset=torch.logsumexp(proposed.float(),-1,keepdim=True)-torch.logsumexp(
            before.float(),-1,keepdim=True)
        out=base.clone()
        out[...,mask]=proposed-offset.to(base.dtype)
        return out

    def future_forward(self,future_logits,history,*,protect_winner=False,
                       horizon_bridge=0.0):
        """Causal +2/+3 conditional byte correction with protected mass.

        history=(hist2,cov2,hits2,hist3,cov3,hits3); each histogram only
        contains source episodes whose lagged target was already observed.
        A zero future_weight reproduces inherited future logits exactly.
        """
        if self.future_weight is None:
            raise ValueError('R69 future memory is not enabled')
        if len(history)!=6 or len(future_logits)!=2:
            raise ValueError('Require both future logits and two causal histories')
        if not isinstance(horizon_bridge,(float,int)) or not 0.0<=horizon_bridge<=1.0:
            raise ValueError('R72 horizon bridge must be between zero and one')
        corrected=[]
        for index,base in enumerate(future_logits):
            hist,cov,hits=history[3*index:3*index+3]
            if (hist.shape!=base.shape or cov.shape!=base.shape[:2] or
                    hits.shape!=base.shape[:2] or base.shape[-1]!=257):
                raise ValueError('Misaligned causal future history')
            if (not torch.isfinite(hist).all() or not torch.isfinite(cov).all() or
                    not torch.isfinite(hits).all() or torch.any(hist<0) or
                    torch.any(hist.sum(-1)>1.00001) or torch.any(cov<0) or
                    torch.any(cov>1) or torch.any(hits<0)):
                raise ValueError('Invalid causal future memory distribution')
            if index==1 and horizon_bridge:
                # R72: causal +2-to-+3 fallback only when the +3 geometric
                # successor history is sparse. These are HISTORICAL observed
                # successors, never either current future target label.
                prior_hist,prior_cov,prior_hits=history[:3]
                if (prior_hist.shape!=hist.shape or prior_cov.shape!=cov.shape or
                        prior_hits.shape!=hits.shape or
                        not torch.isfinite(prior_hist).all() or
                        not torch.isfinite(prior_cov).all() or
                        not torch.isfinite(prior_hits).all() or
                        torch.any(prior_hist<0) or
                        torch.any(prior_hist.sum(-1)>1.00001) or
                        torch.any(prior_cov<0) or torch.any(prior_cov>1) or
                        torch.any(prior_hits<0)):
                    raise ValueError('Invalid R72 causal bridge evidence')
                # The bridge is small when +3 has its own reliable evidence.
                fallback=horizon_bridge*prior_cov*(1-cov)
                hist=(1-fallback[...,None])*hist+fallback[...,None]*prior_hist
                cov=cov+fallback*(1-cov)
                hits=hits+fallback*prior_hits
            with torch.no_grad():
                smooth=(hist*(1-self.smoothing)+self.smoothing/257).clamp_min(1e-12)
                evidence=smooth.log()+math.log(257)
                entropy=-(hist*hist.clamp_min(1e-12).log()).sum(-1)/math.log(257)
                agreement=(hist*base.detach().float().softmax(-1)).sum(-1)
                features=torch.stack((cov,torch.log1p(hits)/math.log1p(self.horizon),
                                      1-entropy,agreement),-1)
            strength=torch.tanh((features.to(self.future_weight.dtype)*
                                 self.future_weight[index]).sum(-1))*cov
            mask=torch.ones(257,device=base.device,dtype=torch.bool)
            mask[32]=False
            mask[256]=False
            before=base[...,mask]
            proposed=before+(strength[...,None]*evidence[...,mask]).to(base.dtype)
            offset=(torch.logsumexp(proposed.float(),-1,keepdim=True)-
                    torch.logsumexp(before.float(),-1,keepdim=True))
            output=base.clone()
            output[...,mask]=proposed-offset.to(base.dtype)
            corrected.append(protect_parent_winner_rank(base,output)
                             if protect_winner else output)
        return tuple(corrected)

    def _read_features(self, pairs, observed, vocab, query_map, value_control,
                       baseline, key_mode="pair", query_byte_map=None):
        with torch.no_grad():
            copy, coverage, positions = retrieve_pair_successors(
                pairs, observed, self.horizon, vocab, query_map, value_control,
                key_mode=key_mode, query_byte_map=query_byte_map,
                engine=self.read_engine)
            log_evidence = (copy * (1-self.smoothing) + self.smoothing/vocab).log()
            log_evidence = log_evidence + math.log(vocab)
            entropy = -(copy*copy.clamp_min(1e-12).log()).sum(-1)/math.log(vocab)
            agreement = (copy*baseline.detach().float().softmax(-1)).sum(-1)
            features = torch.stack((coverage,
                torch.log1p(positions)/math.log1p(self.horizon),
                1-entropy, agreement),-1)
        return log_evidence, coverage, positions, features

    def forward(self, logits, pairs, observed, joint_memory_override=None):
        if logits.ndim != 3 or logits.shape[:2] != pairs.shape[:2]:
            raise ValueError("Logits and geometric address windows must align")
        vocab = logits.size(-1)
        log_evidence, coverage, positions, features = self._read_features(
            pairs, observed, vocab, self.query_map, self.value_control, logits)
        trust = torch.tanh((features.to(self.weight.dtype)*self.weight).sum(-1))
        strength = trust * coverage
        corrected = logits + (strength.unsqueeze(-1)*log_evidence).to(logits.dtype)
        stats = {"coverage":coverage.detach(),
                 "matched_positions":positions.detach(),
                 "trust":strength.detach()}
        if self.content_weight is not None:
            if vocab != 257:
                raise ValueError('R58 requires the inherited 257-byte/EOS vocabulary')
            if self.conditional_query_map is None and self.conditional_value_control == 'exact':
                content_log, content_cov, content_features = log_evidence,coverage,features
            else:
                content_log, content_cov, _, content_features = self._read_features(
                    pairs,observed,vocab,self.conditional_query_map,
                    self.conditional_value_control,corrected)
            content_strength = torch.tanh((content_features.to(self.content_weight.dtype)*
                                            self.content_weight).sum(-1))*content_cov
            content_mask = torch.ones(vocab, device=logits.device, dtype=torch.bool)
            content_mask[32] = False  # literal space, protected
            content_mask[256] = False  # EOS, protected
            # Preserve the exact parent space/EOS logits and the log total
            # probability mass over the other 255 byte values. The expert
            # changes only P(byte | non-space, non-EOS).
            before = corrected[...,content_mask]
            proposed = before + (content_strength.unsqueeze(-1)*
                                  content_log[...,content_mask]).to(logits.dtype)
            normalization = (torch.logsumexp(proposed.float(),dim=-1,keepdim=True)
                             -torch.logsumexp(before.float(),dim=-1,keepdim=True))
            corrected = corrected.clone()
            corrected[...,content_mask] = proposed - normalization.to(logits.dtype)
            stats['content_trust'] = content_strength.detach()
            stats['content_coverage'] = content_cov.detach()
        if joint_memory_override is not None and self.joint_weight is None:
            raise ValueError("External book history needs R59/R60 joint rail")
        if self.joint_weight is not None:
            if self.content_weight is None or vocab != 257:
                raise ValueError("R59 needs R58 content and byte/EOS vocabulary")
            parent_joint = corrected
            if joint_memory_override is None:
                joint_log,joint_cov,joint_positions,joint_features = self._read_features(
                    pairs,observed,vocab,self.joint_query_map,self.joint_value_control,
                    corrected,key_mode=self.joint_key_mode,
                    query_byte_map=self.joint_query_byte_map)
            else:
                # R62 explicitly injected causal per-book prefix memory.
                # A source stream guarantees no future target is included;
                # the neural rail validates shapes, finiteness and mass.
                if len(joint_memory_override)!=3:
                    raise ValueError("External joint read needs histogram, coverage, positions")
                read,cov,pos=joint_memory_override
                if (read.shape!=logits.shape or cov.shape!=logits.shape[:2] or
                        pos.shape!=logits.shape[:2]):
                    raise ValueError("External joint history shape mismatch")
                with torch.no_grad():
                    read=read.to(logits.device,dtype=torch.float32)
                    joint_cov=cov.to(logits.device,dtype=torch.float32)
                    joint_positions=pos.to(logits.device,dtype=torch.float32)
                    if (not torch.isfinite(read).all() or
                            not torch.isfinite(joint_cov).all() or
                            not torch.isfinite(joint_positions).all() or
                            torch.any(read<0) or
                            torch.any(joint_cov<0) or torch.any(joint_cov>1) or
                            torch.any(joint_positions<0) or
                            torch.any(read.sum(-1)>1.00001)):
                        raise ValueError("Invalid external history probabilities/counts")
                    joint_log=(read*(1-self.smoothing)+self.smoothing/vocab).log()
                    joint_log=joint_log+math.log(vocab)
                    entropy=-(read*read.clamp_min(1e-12).log()).sum(-1)/math.log(vocab)
                    agreement=(read*corrected.detach().float().softmax(-1)).sum(-1)
                    joint_features=torch.stack((
                        joint_cov,
                        torch.log1p(joint_positions)/math.log1p(self.horizon),
                        1-entropy,agreement),-1)
            joint_strength = torch.tanh((joint_features.to(self.joint_weight.dtype)*
                                         self.joint_weight).sum(-1))*joint_cov
            # Same probability-mass-preserving decomposition as R58, now
            # specialized on the joint lexical/geometric address.
            mask = torch.ones(vocab,device=logits.device,dtype=torch.bool)
            mask[32] = False
            mask[256] = False
            before = corrected[...,mask]
            proposed = before + (joint_strength.unsqueeze(-1)*
                                  joint_log[...,mask]).to(logits.dtype)
            offset = (torch.logsumexp(proposed.float(),dim=-1,keepdim=True)
                      -torch.logsumexp(before.float(),dim=-1,keepdim=True))
            corrected = corrected.clone()
            corrected[...,mask] = proposed-offset.to(logits.dtype)
            if self.space_rank_guard:
                corrected = protect_parent_space_rank(parent_joint,corrected)
            stats["joint_trust"] = joint_strength.detach()
            stats["joint_coverage"] = joint_cov.detach()
            stats["joint_matched_positions"] = joint_positions.detach()
        return corrected,stats
