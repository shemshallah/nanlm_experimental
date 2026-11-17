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

SCHEMA = "terminal-rsi-self-builder-v1"


class RSISelfBuilder:
    def __init__(self, model, optimizer, history: torch.Tensor,
                 targets: torch.Tensor, auxiliary: torch.Tensor, *,
                 interval: int = 50, tolerance: float = 1e-6):
        if interval < 1 or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("RSI cycle interval must be positive, tolerance finite and nonnegative")
        if history.ndim != 2 or targets.ndim != 1 or auxiliary.ndim != 2:
            raise ValueError("RSI probes must have [batch,context], [batch] and [batch,6] shapes")
        if len(history) == 0 or len(history) != len(targets) or auxiliary.shape != (len(targets), 6):
            raise ValueError("Non-empty causal RSI probes and six inherited features are required")
        self.model = model
        self.optimizer = optimizer
        self.history = history
        self.targets = targets
        self.auxiliary = auxiliary
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
            "tolerance": self.tolerance, "total_updates": self.total_updates,
            "proposals": self.proposals, "accepted": self.accepted,
            "rejected": self.rejected, "events": list(self.events),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema") != SCHEMA:
            raise ValueError("Wrong RSI self-construction checkpoint schema")
        if state.get("interval") != self.interval or state.get("tolerance") != self.tolerance:
            raise ValueError("Cannot change RSI construction policy across checkpoint resume")
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
    def probe_bpb(self) -> float:
        was_training = self.model.training
        self.model.eval()
        try:
            logits, bits = self.model(self.history, self.targets, self.auxiliary)
            loss = F.binary_cross_entropy_with_logits(logits.float(), bits.float())
            bpb = float(8 * loss.item() / math.log(2))
            if not math.isfinite(bpb):
                raise ValueError("Nonfinite RSI internal verifier score")
            return bpb
        finally:
            self.model.train(was_training)

    def propose_and_verify(self) -> dict:
        """One actual tensor mutation, independent causal check, and RSI update."""
        prior = self.model.original.feedback_state.detach().clone()
        actor_logits = self.model.original.actor_logits + self.model.original.actor_feedback(prior)
        distribution = torch.distributions.Categorical(logits=actor_logits)
        action = int(distribution.sample().item())
        # The original actor has exactly two choices. Route choice explores
        # the 3 existing learned specialists; refinement explores the feedback
        # loop and entry gate. No tensor shape/Adam state is discarded.
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
        with torch.no_grad():
            if idx is None:
                tensor.add_(delta)
            else:
                tensor[idx].add_(delta)
        try:
            after_candidate = self.probe_bpb()
        except Exception:
            with torch.no_grad():
                tensor.copy_(original)
            raise
        accepted = after_candidate < before - self.tolerance
        if not accepted:
            with torch.no_grad():
                tensor.copy_(original)
        # Reward reflects the verified prediction effect, NOT the proposed
        # action or actor's own confidence. A failed trial is not promoted.
        reward_value = (float(math.tanh((before - after_candidate) * 8))
                        if accepted else -0.25)
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
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        event = {
            "proposal": self.proposals, "update": self.total_updates,
            "action": action, "modification": label, "delta": delta,
            "verified": accepted, "internal_bpb_before": before,
            "internal_bpb_candidate": after_candidate,
            "reward": reward_value, "rsi_actor_critic_updated": True,
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
