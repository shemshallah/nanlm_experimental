"""Verified recursive self-construction for the unified Terminal-RSI training core.

The RSI actor, not the epoch loop, proposes and verifies structural changes:
  - which specialist route receives more attention,
  - how strongly the recurrent refinement feeds back, or
  - whether new additive capacity is admitted through the zero-gated bridge.

All proposals are scored on a separate INTERNAL causal TRAIN probe. Rejected
mutations restore their exact tensor. The verified reward trains the original
v103 actor and critic through their inherited AdamW optimizer histories.
The public book-disjoint DEV and sealed TEST are never used by this controller.

This is self-modification of active differentiable topology and training
policy, NOT arbitrary Python source rewriting or independently verified AGI.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch.nn import functional as F

SCHEMA = "terminal-rsi-self-builder-v2"
LEGACY_SCHEMA = "terminal-rsi-self-builder-v1"


class RSISelfBuilder:
    def __init__(self, model, optimizer, history: torch.Tensor,
                 targets: torch.Tensor, auxiliary: torch.Tensor, *,
                 interval: int = 50, tolerance: float = 1e-6,
                 audit_history: torch.Tensor | None = None,
                 audit_targets: torch.Tensor | None = None,
                 audit_auxiliary: torch.Tensor | None = None,
                 min_gain_bpb: float = 0.002):
        if interval < 1 or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("RSI cycle interval must be positive, tolerance finite and nonnegative")
        if not math.isfinite(min_gain_bpb) or min_gain_bpb <= 0:
            raise ValueError("RSI minimum construction gain must be finite and positive")
        if audit_history is None or audit_targets is None or audit_auxiliary is None:
            raise ValueError("Separate internal TRAIN verification/audit probes are mandatory")
        for h, t, a in ((history, targets, auxiliary),
                        (audit_history, audit_targets, audit_auxiliary)):
            if h.ndim != 2 or t.ndim != 1 or a.ndim != 2:
                raise ValueError("RSI probes must have [batch,context], [batch] and [batch,6] shapes")
            if len(h) == 0 or len(h) != len(t) or a.shape != (len(t), 6):
                raise ValueError("Non-empty causal RSI probes and six inherited features are required")
        if history.shape[1] != audit_history.shape[1]:
            raise ValueError("RSI verification and audit must use identical causal context length")
        self.model = model
        self.optimizer = optimizer
        self.history = history
        self.targets = targets
        self.auxiliary = auxiliary
        self.audit_history = audit_history
        self.audit_targets = audit_targets
        self.audit_auxiliary = audit_auxiliary
        self.min_gain_bpb = min_gain_bpb
        self.interval = interval
        self.tolerance = tolerance
        self.total_updates = 0
        self.proposals = 0
        self.accepted = 0
        self.rejected = 0
        self.events: list[dict] = []

    def state_dict(self) -> dict:
        return {
            "schema": SCHEMA, "interval": self.interval,
            "tolerance": self.tolerance, "min_gain_bpb": self.min_gain_bpb,
            "verification_count": len(self.targets), "audit_count": len(self.audit_targets),
            "total_updates": self.total_updates,
            "proposals": self.proposals, "accepted": self.accepted,
            "rejected": self.rejected, "events": list(self.events),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
            raise ValueError("Wrong RSI self-construction checkpoint schema")
        if state.get("interval") != self.interval or state.get("tolerance") != self.tolerance:
            raise ValueError("Cannot change RSI construction policy across checkpoint resume")
        if state["schema"] == SCHEMA and (
            state.get("min_gain_bpb") != self.min_gain_bpb
            or state.get("verification_count") != len(self.targets)
            or state.get("audit_count") != len(self.audit_targets)
        ):
            raise ValueError("Cannot change RSI verifier policy across checkpoint resume")
        if any(not isinstance(state.get(k), int) or state[k] < 0
               for k in ("total_updates", "proposals", "accepted", "rejected")):
            raise ValueError("Malformed RSI self-construction counters")
        if state["accepted"] + state["rejected"] != state["proposals"]:
            raise ValueError("RSI mutation ledger has inconsistent acceptance totals")
        events = state.get("events")
        if not isinstance(events, list) or len(events) != state["proposals"]:
            raise ValueError("RSI mutation ledger cannot be replayed")
        self.total_updates = state["total_updates"]
        self.proposals = state["proposals"]
        self.accepted = state["accepted"]
        self.rejected = state["rejected"]
        self.events = list(events)

    @torch.no_grad()
    def probe_bpb(self, split: str = "verification") -> float:
        """Score all eight target bits on one causal INTERNAL TRAIN partition."""
        if split == "verification":
            history, targets, auxiliary = self.history, self.targets, self.auxiliary
        elif split == "audit":
            history, targets, auxiliary = (
                self.audit_history, self.audit_targets, self.audit_auxiliary)
        else:
            raise ValueError("RSI internal probe must be verification or audit")
        was_training = self.model.training
        self.model.eval()
        try:
            logits, bits = self.model(history, targets, auxiliary)
            loss = F.binary_cross_entropy_with_logits(logits.float(), bits.float())
            bpb = float(8 * loss.item() / math.log(2))
            if not math.isfinite(bpb):
                raise ValueError("Nonfinite RSI internal verifier score")
            return bpb
        finally:
            self.model.train(was_training)

    def propose_and_verify(self) -> dict:
        """Commit mutations only when TWO disjoint causal TRAIN probes improve."""
        prior = self.model.original.feedback_state.detach().clone()
        actor_logits = self.model.original.actor_logits + self.model.original.actor_feedback(prior)
        distribution = torch.distributions.Categorical(logits=actor_logits)
        action = int(distribution.sample().item())
        if action == 0:
            idx = 1 + (self.proposals % 3)
            tensor = self.model.upgrades.router.bias
            label = f"specialist_route_{idx}"
        elif self.proposals % 2 == 0:
            idx = None
            tensor = self.model.residual_gate
            label = "additive_capacity_gate"
        else:
            idx = None
            tensor = self.model.upgrades.refine_scale
            label = "recurrent_refinement_depth"
        delta = (0.08 if self.proposals % 2 == 0 else -0.08)
        original = tensor.detach().clone()
        before = self.probe_bpb()
        audit_before = self.probe_bpb("audit")
        with torch.no_grad():
            if idx is None:
                tensor.add_(delta)
            else:
                tensor[idx].add_(delta)
        try:
            after_candidate = self.probe_bpb()
            audit_candidate = self.probe_bpb("audit")
        except Exception:
            with torch.no_grad():
                tensor.copy_(original)
            raise
        gain = before - after_candidate
        audit_gain = audit_before - audit_candidate
        mean_gain = 0.5 * (gain + audit_gain)
        # Never accept a tiny apparent gain in one tiny sample if the other
        # internal TRAIN audit regresses. DEV/TEST is excluded from both.
        accepted = (gain > self.tolerance and audit_gain > self.tolerance
                    and mean_gain >= self.min_gain_bpb)
        if not accepted:
            with torch.no_grad():
                tensor.copy_(original)
        # CONTINUOUS, correctly signed measured BPB reward: failed trials are
        # penalized according to actual damage, not by arbitrary fixed -0.25.
        # A tiny change cannot produce an outsized critic/actor reward.
        reward_value = float(math.tanh(8 * mean_gain)) if accepted else float(min(0.0, math.tanh(8 * mean_gain)))
        reward = torch.tensor(reward_value, device=prior.device)
        critic = self.model.original.reward_critic(prior).squeeze()
        advantage = (reward - critic).detach()
        actor_loss = -distribution.log_prob(torch.tensor(action, device=prior.device)) * advantage
        critic_loss = 0.5 * (critic - reward).square()
        self.optimizer.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        self.optimizer.step()
        self.model.original.observe_terminal(accepted)
        self.proposals += 1
        self.accepted += int(accepted)
        self.rejected += int(not accepted)
        event = {
            "proposal": self.proposals, "update": self.total_updates,
            "action": action, "modification": label, "delta": delta,
            "verified": accepted, "internal_bpb_before": before,
            "internal_bpb_candidate": after_candidate,
            "internal_audit_bpb_before": audit_before,
            "internal_audit_bpb_candidate": audit_candidate,
            "verified_bpb_gain": gain, "audit_bpb_gain": audit_gain,
            "mean_bpb_gain": mean_gain,
            "minimum_required_bpb_gain": self.min_gain_bpb,
            "reward": reward_value, "reward_definition": "accepted:tanh(8*mean_gain), rejected:min(0,tanh(8*mean_gain))",
            "rsi_actor_critic_updated": True,
            "restored_rejected_tensor_exactly": not accepted,
        }
        self.events.append(event)
        return event

    def train_step(self, step_fn: Callable, history: torch.Tensor,
                   targets: torch.Tensor, auxiliary: torch.Tensor):
        """RSI is the OUTER training driver: byte step, observation, then build."""
        report = step_fn(self.model, self.optimizer, history, targets, auxiliary)
        self.total_updates += 1
        if self.total_updates % self.interval == 0:
            mutation = self.propose_and_verify()
            print("RSI SELF-BUILD", mutation, flush=True)
        return report
