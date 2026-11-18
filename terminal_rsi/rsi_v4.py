"""RSI-v4: counterfactual, cost-aware self-modification for Terminal-RSI.

This module replaces the v3 fixed ``+/-0.08`` gate bandit with:

* a telemetry-conditioned training state,
* typed high-leverage interventions (LR, clipping, weight decay, gates,
  router temperature, batch size),
* an ensemble critic and a factorized proposal policy with a replay buffer,
* multi-shard paired branch evaluation with common random numbers,
* difference-in-differences credit assignment over a real training horizon,
* cost-aware acceptance and persistent meta-learning data.

The evaluator is deliberately conservative: DEV/TEST are never used.  Only
disjoint internal TRAIN shards are scored.  A branch is accepted only when it
beats the no-op branch by both an economic minimum and a paired standard error.

The controller is designed to be driven by ``train_verbatim.py`` through four
callbacks: ``capture``, ``restore``, ``train`` and ``evaluate``.  This keeps the
controller independent of the v103 model internals and makes exact branch
rollback possible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields
import copy
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .legacy_v103_reference import PARITY_CONTEXT

# ---------------------------------------------------------------------------
# Typed action space
# ---------------------------------------------------------------------------

LR_MULTS = (0.8, 0.9, 1.0, 1.1, 1.25)
UPGRADE_LR_MULTS = (0.5, 1.0, 2.0)
CLIPS = (0.5, 1.0, 2.0, 4.0)
WEIGHT_DECAYS = (0.0, 0.01, 0.1)
GATE_DELTAS = (-0.08, -0.02, 0.0, 0.02, 0.08)
REFINE_DELTAS = (-0.08, -0.02, 0.0, 0.02, 0.08)
ROUTER_TEMPS = (0.8, 1.0, 1.25, 1.5)
BATCH_SIZES = (2, 4, 8)
HIERARCHY_DELTAS = (-0.02, 0.0, 0.02)

PRIMITIVES: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("lr_mult", LR_MULTS),
    ("upgrade_lr_mult", UPGRADE_LR_MULTS),
    ("grad_clip", CLIPS),
    ("weight_decay", WEIGHT_DECAYS),
    ("residual_gate_delta", GATE_DELTAS),
    ("refine_scale_delta", REFINE_DELTAS),
    ("router_temperature", ROUTER_TEMPS),
    ("batch_size", BATCH_SIZES),
    ("hierarchy_bottom_up_delta", HIERARCHY_DELTAS),
    ("hierarchy_top_down_delta", HIERARCHY_DELTAS),
    ("hierarchy_lateral_delta", HIERARCHY_DELTAS),
)
PRIMITIVE_NAMES = tuple(name for name, _ in PRIMITIVES)
NEUTRAL_INDEX = {
    "lr_mult": LR_MULTS.index(1.0),
    "upgrade_lr_mult": UPGRADE_LR_MULTS.index(1.0),
    "grad_clip": CLIPS.index(1.0),
    "weight_decay": WEIGHT_DECAYS.index(0.01),
    "residual_gate_delta": GATE_DELTAS.index(0.0),
    "refine_scale_delta": REFINE_DELTAS.index(0.0),
    "router_temperature": ROUTER_TEMPS.index(1.0),
    "batch_size": BATCH_SIZES.index(4),
    "hierarchy_bottom_up_delta": HIERARCHY_DELTAS.index(0.0),
    "hierarchy_top_down_delta": HIERARCHY_DELTAS.index(0.0),
    "hierarchy_lateral_delta": HIERARCHY_DELTAS.index(0.0),
}
NEUTRAL_CHOICES = tuple(NEUTRAL_INDEX[name] for name in PRIMITIVE_NAMES)
ACTION_DIM = sum(len(values) for _, values in PRIMITIVES)


@dataclass(frozen=True)
class Action:
    """One factorized intervention over the typed primitive space."""

    choices: tuple[int, ...]

    def __post_init__(self):
        if len(self.choices) != len(PRIMITIVES):
            raise ValueError("Action has the wrong number of primitives")
        for value, (name, options) in zip(self.choices, PRIMITIVES):
            if not 0 <= int(value) < len(options):
                raise ValueError(f"Action {name}={value} is out of range")

    @classmethod
    def neutral(cls) -> "Action":
        return cls(NEUTRAL_CHOICES)

    @classmethod
    def from_tuple(cls, values: Sequence[int]) -> "Action":
        return cls(tuple(int(v) for v in values))

    def vector(self) -> torch.Tensor:
        row = torch.zeros(ACTION_DIM, dtype=torch.float32)
        offset = 0
        for value, (_, options) in zip(self.choices, PRIMITIVES):
            row[offset + value] = 1.0
            offset += len(options)
        return row

    def values(self) -> dict[str, Any]:
        return {
            name: options[value]
            for value, (name, options) in zip(self.choices, PRIMITIVES)
        }

    def label(self) -> str:
        v = self.values()
        return (
            f"lr={v['lr_mult']:.2f} up_lr={v['upgrade_lr_mult']:.2f} "
            f"clip={v['grad_clip']:.2f} wd={v['weight_decay']:.3f} "
            f"gate={v['residual_gate_delta']:+.2f} "
            f"refine={v['refine_scale_delta']:+.2f} "
            f"temp={v['router_temperature']:.2f} batch={v['batch_size']} "
            f"bu={v['hierarchy_bottom_up_delta']:+.2f} "
            f"td={v['hierarchy_top_down_delta']:+.2f} "
            f"lat={v['hierarchy_lateral_delta']:+.2f}"
        )

    def is_noop(self) -> bool:
        return self.choices == NEUTRAL_CHOICES

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.choices)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def apply_action(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    base_lrs: Sequence[float],
    action: Action,
) -> None:
    """Apply an intervention in place.  Exact rollback is the caller's job."""
    values = action.values()
    lr_mult = float(values["lr_mult"])
    upgrade_mult = float(values["upgrade_lr_mult"])
    for index, group in enumerate(optimizer.param_groups):
        mult = lr_mult * (upgrade_mult if index == 2 else 1.0)
        group["lr"] = float(base_lrs[index]) * mult
    if len(optimizer.param_groups) > 2:
        optimizer.param_groups[2]["weight_decay"] = float(values["weight_decay"])

    upgrades = getattr(model, "upgrades", None)
    if upgrades is None:
        raise ValueError("RSI-v4 requires a model with an 'upgrades' module")
    with torch.no_grad():
        model.residual_gate.add_(float(values["residual_gate_delta"])).clamp_(-2.0, 2.0)
        upgrades.refine_scale.add_(float(values["refine_scale_delta"])).clamp_(-6.0, 6.0)
    upgrades.router_temperature = float(values["router_temperature"])
    hierarchy = getattr(upgrades, "hierarchy", None)
    if hierarchy is not None:
        with torch.no_grad():
            hierarchy.bottom_up_gates.add_(
                float(values["hierarchy_bottom_up_delta"])).clamp_(-1.0, 1.0)
            hierarchy.top_down_gates.add_(
                float(values["hierarchy_top_down_delta"])).clamp_(-1.0, 1.0)
            hierarchy.lateral_gates.add_(
                float(values["hierarchy_lateral_delta"])).clamp_(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

STATE_DIM = 20


@dataclass
class TelemetryState:
    bpb: float = 0.0
    bit_accuracy: float = 0.0
    grad_norm: float = 0.0
    loss_ema: float = 0.0
    loss_slope: float = 0.0
    lr_mult: float = 1.0
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    residual_gate: float = 0.0
    refine_scale: float = -3.0
    router_temperature: float = 1.0
    batch_size: int = 4
    updates_per_second: float = 0.0
    proposals: int = 0
    accepted: int = 0
    rejected: int = 0
    step_fraction: float = 0.0
    mean_reward: float = 0.0
    last_gain: float = 0.0
    verified_gain_sum: float = 0.0

    def vector(self) -> torch.Tensor:
        values = [
            self.bpb / 10.0,
            self.bit_accuracy,
            math.log1p(max(self.grad_norm, 0.0)) / 5.0,
            self.loss_ema / 10.0,
            math.tanh(10.0 * self.loss_slope),
            _clamp(math.log(max(self.lr_mult, 1e-6)), -1.5, 1.5) / 1.5,
            self.grad_clip / 4.0,
            self.weight_decay * 10.0,
            math.tanh(self.residual_gate),
            math.tanh(self.refine_scale / 3.0),
            (self.router_temperature - 1.0),
            self.batch_size / 8.0,
            math.log1p(max(self.updates_per_second, 0.0)) / 5.0,
            min(1.0, self.proposals / 100.0),
            self.accepted / max(1, self.proposals),
            self.rejected / max(1, self.proposals),
            _clamp(self.step_fraction, 0.0, 1.0),
            math.tanh(self.mean_reward),
            math.tanh(10.0 * self.last_gain),
            math.tanh(10.0 * self.verified_gain_sum),
        ]
        return torch.tensor(values, dtype=torch.float32)


class TelemetryTracker:
    def __init__(self) -> None:
        self.state = TelemetryState()

    def observe(
        self,
        report: dict,
        *,
        step: int,
        target_updates: int,
        updates_per_second: float,
    ) -> TelemetryState:
        bpb = float(report.get("bpb", 0.0))
        if not math.isfinite(bpb) or bpb < 0:
            bpb = 0.0
        previous = self.state.loss_ema
        ema = 0.98 * previous + 0.02 * bpb if previous else bpb
        self.state.bpb = bpb
        self.state.bit_accuracy = float(report.get("bit_accuracy", 0.0))
        self.state.grad_norm = float(report.get("grad_norm", self.state.grad_norm))
        self.state.loss_ema = ema
        self.state.loss_slope = ema - previous
        self.state.updates_per_second = float(updates_per_second)
        self.state.step_fraction = step / max(1, target_updates)
        return self.state

    def state_dict(self) -> dict:
        return asdict(self.state)

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        for key, value in state.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)


# ---------------------------------------------------------------------------
# Probe shards
# ---------------------------------------------------------------------------


def build_contexts(raw_bytes: torch.Tensor, target_positions: torch.Tensor,
                   history: int) -> torch.Tensor:
    """Causal contexts for arbitrary target byte positions.

    This is intentionally a local copy of the v103 helper so that rsi_v4 has no
    import cycle with train_verbatim.py.  The semantics are identical.
    """
    positions = target_positions.to("cpu", torch.long).reshape(-1)
    if history < 32 or positions.numel() == 0:
        raise ValueError("Need at least the original 32 observed byte positions")
    if int(positions.min()) < 32 or int(positions.max()) >= len(raw_bytes):
        raise ValueError("Invalid target position for causal context")
    deltas = torch.arange(history, 0, -1, dtype=torch.long)
    indices = positions[:, None] - deltas[None, :]
    causal = indices >= 0
    past = raw_bytes[indices.clamp_min(0)].long()
    past = torch.where(causal, past, torch.zeros_like(past))
    target = raw_bytes[positions].long()
    return torch.cat((past, target[:, None]), dim=-1)


class ProbeShards:
    """A fixed, disjoint, stratified internal TRAIN validation set."""

    def __init__(self, windows: list[torch.Tensor], auxiliary: list[torch.Tensor],
                 positions: torch.Tensor, history: int):
        if len(windows) != len(auxiliary) or not windows:
            raise ValueError("Probe shards require matching windows and auxiliary")
        self.windows = windows
        self.auxiliary = auxiliary
        self.positions = positions
        self.history = history

    @classmethod
    def build(cls, raw_tensor: torch.Tensor, aux_data: torch.Tensor, *,
              n_shards: int, positions_per_shard: int, history: int,
              seed: int) -> "ProbeShards":
        if n_shards < 1 or positions_per_shard < 1:
            raise ValueError("Probe shard counts must be positive")
        total = n_shards * positions_per_shard
        low, high = PARITY_CONTEXT, len(raw_tensor) - 1
        if high <= low or total > high - low:
            raise ValueError("Training stream is too short for the requested probes")
        generator = torch.Generator().manual_seed(int(seed))
        edges = torch.linspace(low, high, total + 1).long()
        chosen = []
        for index in range(total):
            lo = int(edges[index].item())
            hi = int(edges[index + 1].item())
            if hi <= lo:
                hi = min(high + 1, lo + 1)
            value = lo + int(torch.randint(0, max(1, hi - lo), (1,), generator=generator).item())
            chosen.append(min(value, high))
        positions = torch.tensor(chosen, dtype=torch.long)
        windows = []
        auxiliary = []
        for shard in range(n_shards):
            shard_positions = positions[shard * positions_per_shard:(shard + 1) * positions_per_shard]
            windows.append(build_contexts(raw_tensor, shard_positions, history))
            auxiliary.append(
                aux_data[(shard_positions - PARITY_CONTEXT).to(aux_data.device)]
                .to(dtype=torch.int32))
        return cls(windows, auxiliary, positions, history)

    def evaluate(self, model: nn.Module, chunk_size: int = 8) -> list[float]:
        if chunk_size < 1:
            raise ValueError("Probe evaluation chunk size must be positive")
        was_training = model.training
        device = next(model.parameters()).device
        model.eval()
        results: list[float] = []
        try:
            with torch.no_grad():
                for window, aux in zip(self.windows, self.auxiliary):
                    window = window.to(device)
                    aux = aux.to(device)
                    total_loss = 0.0
                    total_bytes = 0
                    for start in range(0, len(window), chunk_size):
                        part = window[start:start + chunk_size]
                        part_aux = aux[start:start + chunk_size]
                        logits, bits = model(part[:, :-1], part[:, -1], part_aux)
                        loss = F.binary_cross_entropy_with_logits(
                            logits.float(), bits.float(), reduction="sum")
                        total_loss += float(loss.item())
                        total_bytes += len(part)
                    results.append(float(
                        total_loss / max(total_bytes, 1) / math.log(2)))
        finally:
            model.train(was_training)
        return results


def probe_noise_audit(
    model: nn.Module,
    raw_tensor: torch.Tensor,
    aux_data: torch.Tensor,
    *,
    history: int,
    n_shards: int = 4,
    positions_per_shard: int = 256,
    repeats: int = 20,
    seed: int = 2026,
    eval_chunk: int = 8,
) -> dict:
    """A/A audit: how much does the measured BPB move across probe sets?"""
    values: list[float] = []
    shard_values: list[list[float]] = []
    for index in range(repeats):
        probe = ProbeShards.build(
            raw_tensor, aux_data, n_shards=n_shards,
            positions_per_shard=positions_per_shard, history=history,
            seed=seed + index,
        )
        scores = probe.evaluate(model, chunk_size=eval_chunk)
        shard_values.append(scores)
        values.extend(scores)
    diffs = []
    for scores in shard_values:
        if len(scores) >= 2:
            half = len(scores) // 2
            diffs.append(statistics.mean(scores[:half]) - statistics.mean(scores[half:]))
    return {
        "repeats": repeats,
        "n_shards": n_shards,
        "positions_per_shard": positions_per_shard,
        "mean_bpb": statistics.mean(values),
        "sd_across_shards": statistics.stdev(values) if len(values) > 1 else 0.0,
        "sd_paired_half_difference": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
        "min_shard_bpb": min(values),
        "max_shard_bpb": max(values),
    }


# ---------------------------------------------------------------------------
# Critic, policy, replay
# ---------------------------------------------------------------------------


class EnsembleCritic(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 64, members: int = 3):
        super().__init__()
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )
            for _ in range(members)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state.float(), action.float()], dim=-1)
        return torch.stack([member(x) for member in self.members], dim=0)

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(state, action)
        means = out[..., 0]
        logvars = out[..., 1].clamp(-8.0, 4.0)
        mean = means.mean(dim=0)
        variance = logvars.exp().mean(dim=0) + means.var(dim=0, unbiased=False)
        return mean, variance

    def nll_loss(self, state: torch.Tensor, action: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        out = self.forward(state, action)
        means = out[..., 0]
        logvars = out[..., 1].clamp(-8.0, 4.0)
        target = target.float().unsqueeze(0)
        nll = 0.5 * (logvars + (target - means).pow(2) * torch.exp(-logvars))
        return nll.mean()


class WorldModel(nn.Module):
    """Predicts short-horizon training telemetry for imagined branch ranking."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return self.body(torch.cat([state.float(), action.float()], dim=-1))

    def loss(self, state: torch.Tensor, action: torch.Tensor,
             next_bpb: torch.Tensor, next_grad_norm: torch.Tensor,
             next_bit_accuracy: torch.Tensor) -> torch.Tensor:
        prediction = self.forward(state, action)
        target = torch.stack([
            next_bpb / 10.0,
            torch.log1p(next_grad_norm.clamp_min(0.0)) / 5.0,
            next_bit_accuracy,
        ], dim=-1)
        return F.smooth_l1_loss(prediction, target)


class ProposalPolicy(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden, len(options)) for _, options in PRIMITIVES
        ])
        with torch.no_grad():
            for head, ((name, options)) in zip(self.heads, PRIMITIVES):
                head.weight.zero_()
                head.bias.zero_()
                head.bias[NEUTRAL_INDEX[name]] = 1.0

    def logits(self, state: torch.Tensor) -> list[torch.Tensor]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        hidden = self.body(state.float())
        return [head(hidden) for head in self.heads]

    def sample(self, state: torch.Tensor, *, temperature: float = 1.0,
               epsilon: float = 0.0) -> tuple[Action, torch.Tensor, torch.Tensor]:
        logits = self.logits(state)
        choices: list[int] = []
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for row, (_, options) in zip(logits, PRIMITIVES):
            distribution = Categorical(logits=row[0] / max(temperature, 1e-6))
            if epsilon > 0 and random.random() < epsilon:
                index = random.randrange(len(options))
            else:
                index = int(distribution.sample().item())
            choices.append(index)
            log_probs.append(distribution.log_prob(
                torch.tensor(index, device=row.device)))
            entropies.append(distribution.entropy())
        log_prob = torch.stack(log_probs).sum()
        entropy = torch.stack(entropies).sum()
        return Action(tuple(choices)), log_prob, entropy

    def log_prob(self, state: torch.Tensor, actions: Sequence[Action]) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        logits = self.logits(state)
        total = torch.zeros(state.shape[0], dtype=torch.float32, device=state.device)
        for head_logits, name in zip(logits, PRIMITIVE_NAMES):
            position = PRIMITIVE_NAMES.index(name)
            indices = torch.tensor(
                [action.choices[position] for action in actions],
                dtype=torch.long, device=head_logits.device,
            )
            total = total + F.log_softmax(head_logits, dim=-1).gather(
                1, indices.unsqueeze(-1)).squeeze(-1)
        return total

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(next(self.parameters()).device)
        total = torch.zeros(state.shape[0], dtype=torch.float32, device=state.device)
        for row in self.logits(state):
            probs = F.softmax(row, dim=-1)
            total = total - (probs * probs.clamp_min(1e-9).log()).sum(-1)
        return total


@dataclass
class Transition:
    state: list[float]
    action: list[int]
    utility: float
    bpb: float
    noop_bpb: float
    cost: float
    step: int
    next_bpb: float = 0.0
    next_grad_norm: float = 0.0
    next_bit_accuracy: float = 0.0


class ReplayBuffer:
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.transitions: list[Transition] = []

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)
        if len(self.transitions) > self.capacity:
            del self.transitions[:len(self.transitions) - self.capacity]

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.transitions:
            return []
        return random.sample(self.transitions, min(batch_size, len(self.transitions)))

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "transitions": [asdict(item) for item in self.transitions],
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self.capacity = int(state.get("capacity", self.capacity))
        allowed = {item.name for item in fields(Transition)}
        self.transitions = [
            Transition(**{key: value for key, value in item.items()
                          if key in allowed})
            for item in state.get("transitions", [])
        ]

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2))

    def load_json(self, path: Path) -> None:
        self.load_state_dict(json.loads(path.read_text()))

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Branch snapshots and controller
# ---------------------------------------------------------------------------


@dataclass
class BranchSnapshot:
    model: Any
    optimizer: Any
    sampler: dict
    torch_rng: Any
    python_rng: Any
    cuda_rng: Any
    step: int
    update_count: int
    training: bool
    action: tuple[int, ...]
    batch_size: int
    grad_clip: float


class FastStateSnapshot:
    """Preallocated in-place snapshot of model and optimizer tensors.

    This exists specifically for CPU v4 branch search.  ``copy.deepcopy`` of the
    full model/optimizer state allocates hundreds of MB on every branch and was
    the dominant v4 overhead.  Here buffers are allocated once per cycle and
    reused, so branch snapshots are pure memory copies.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        state = model.state_dict()
        self.model_tensors = list(state.values())
        self.model_buffers = [torch.empty_like(tensor) for tensor in self.model_tensors]
        self.optimizer_tensors: list[tuple[dict, str]] = []
        self.optimizer_buffers: list[torch.Tensor] = []
        self.optimizer_scalars: list[tuple[dict, str]] = []
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    self.optimizer_tensors.append((item, key))
                    self.optimizer_buffers.append(torch.empty_like(value))
                else:
                    self.optimizer_scalars.append((item, key))
        self.group_values: list[dict] | None = None
        self.scalar_values: list[Any] | None = None

    def capture(self) -> "FastStateSnapshot":
        with torch.no_grad():
            for buffer, tensor in zip(self.model_buffers, self.model_tensors):
                buffer.copy_(tensor)
            for buffer, (item, key) in zip(self.optimizer_buffers,
                                           self.optimizer_tensors):
                buffer.copy_(item[key])
        self.group_values = copy.deepcopy([
            {key: value for key, value in group.items() if key != "params"}
            for group in self.optimizer.param_groups
        ])
        self.scalar_values = [
            copy.deepcopy(item[key]) for item, key in self.optimizer_scalars
        ]
        return self

    def restore(self) -> None:
        with torch.no_grad():
            for tensor, buffer in zip(self.model_tensors, self.model_buffers):
                tensor.copy_(buffer)
            for (item, key), buffer in zip(self.optimizer_tensors,
                                           self.optimizer_buffers):
                item[key].copy_(buffer)
        if self.scalar_values is not None:
            for (item, key), value in zip(self.optimizer_scalars,
                                          self.scalar_values):
                item[key] = value
        if self.group_values is not None:
            for group, values in zip(self.optimizer.param_groups,
                                     self.group_values):
                group.update(values)



@dataclass
class BranchResult:
    action: Action
    shard_bpbs: list[float]
    mean_bpb: float
    reports: list[dict]
    snapshot: BranchSnapshot
    utility: float = 0.0
    standard_error: float = 0.0


@dataclass
class RSIv4Config:
    cycle_updates: int = 200
    horizon: int = 10
    candidates: int = 2
    n_shards: int = 4
    positions_per_shard: int = 256
    eval_chunk: int = 8
    min_gain: float = 0.01
    z_score: float = 2.0
    batch_size: int = 4
    fixed_batch: bool = False
    branch_batch_size: int = 64
    grad_clip: float = 1.0
    seed: int = 2026
    critic_steps: int = 32
    policy_steps: int = 16
    replay_capacity: int = 4096
    critic_lr: float = 3e-4
    policy_lr: float = 3e-4
    entropy_coef: float = 0.01
    target_updates: int = 1000


class RSIv4Controller:
    """Counterfactual branching controller.

    The training loop supplies four callbacks.  ``capture`` returns a
    ``BranchSnapshot``; ``restore`` installs one; ``train`` runs real updates
    and returns v103-style reports; ``evaluate`` returns one BPB per internal
    TRAIN shard.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        probe: ProbeShards,
        config: RSIv4Config,
        *,
        device: torch.device | str = "cpu",
        meta_path: Path | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.probe = probe
        self.config = config
        self.device = torch.device(device)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_action = Action.neutral()
        self.batch_size = int(config.batch_size)
        self.fixed_batch = bool(config.fixed_batch)
        self._active_branch_batch: int | None = None
        self.grad_clip = float(config.grad_clip)
        self.policy = ProposalPolicy(STATE_DIM).to(self.device)
        self.critic = EnsembleCritic(STATE_DIM, ACTION_DIM).to(self.device)
        self.world_model = WorldModel(STATE_DIM, ACTION_DIM).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.policy_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr)
        self.world_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.telemetry = TelemetryTracker()
        self.proposals = 0
        self.accepted = 0
        self.rejected = 0
        self.cycles = 0
        self.branch_updates = 0
        self.events: list[dict] = []
        self._state_snapshots: list[FastStateSnapshot] | None = None
        self._snapshot_index = 0
        self.meta_path = Path(meta_path) if meta_path else None
        if self.meta_path and self.meta_path.is_file():
            try:
                if self.meta_path.suffix == ".pt":
                    state = torch.load(
                        self.meta_path, map_location="cpu", weights_only=True)
                    self.load_state_dict(state)
                else:
                    self.replay.load_json(self.meta_path)
            except (OSError, ValueError, TypeError, RuntimeError):
                pass
        # Neutral action is a no-op relative to the normal optimizer defaults.
        # Runtime hyperparameters are applied here; gate deltas are deliberately
        # not re-applied because a resumed/meta model may already contain them.
        self._apply_runtime(self.current_action)

    # -- action application -------------------------------------------------
    def _apply_runtime(self, action: Action) -> None:
        """Restore LR/weight-decay/temperature fields without touching gates."""
        values = action.values()
        lr_mult = float(values["lr_mult"])
        upgrade_mult = float(values["upgrade_lr_mult"])
        for index, group in enumerate(self.optimizer.param_groups):
            mult = lr_mult * (upgrade_mult if index == 2 else 1.0)
            group["lr"] = float(self.base_lrs[index]) * mult
        if len(self.optimizer.param_groups) > 2:
            self.optimizer.param_groups[2]["weight_decay"] = float(values["weight_decay"])
        self.model.upgrades.router_temperature = float(values["router_temperature"])
        self.current_action = action
        if not self.fixed_batch:
            self.batch_size = int(values["batch_size"])
        self.grad_clip = float(values["grad_clip"])

    def apply(self, action: Action) -> None:
        apply_action(self.model, self.optimizer, self.base_lrs, action)
        values = action.values()
        self.current_action = action
        if not self.fixed_batch:
            self.batch_size = int(values["batch_size"])
        self.grad_clip = float(values["grad_clip"])

    def action_label(self) -> str:
        """Human-readable action; with a fixed batch, do not imply it changed."""
        label = self.current_action.label()
        if not self.fixed_batch:
            return label
        parts = label.split()
        return " ".join(
            f"batch=fixed({self.batch_size})" if item.startswith("batch=")
            else item for item in parts)

    def restore_action_state(self, choices: Sequence[int], batch_size: int,
                             grad_clip: float) -> None:
        self.current_action = Action.from_tuple(choices)
        if not self.fixed_batch:
            self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)

    # -- fast CPU branch snapshots -----------------------------------------
    @property
    def active_branch_batch(self) -> int | None:
        return self._active_branch_batch

    def capture_state(self) -> FastStateSnapshot:
        if self._state_snapshots is None:
            count = max(3, self.config.candidates + 2)
            self._state_snapshots = [
                FastStateSnapshot(self.model, self.optimizer) for _ in range(count)
            ]
        if self._snapshot_index >= len(self._state_snapshots):
            self._state_snapshots.append(
                FastStateSnapshot(self.model, self.optimizer))
        snapshot = self._state_snapshots[self._snapshot_index]
        self._snapshot_index += 1
        return snapshot.capture()

    def restore_state(self, snapshot: FastStateSnapshot) -> None:
        snapshot.restore()

    # -- telemetry ----------------------------------------------------------
    def state_vector(self, step: int) -> torch.Tensor:
        state = self.telemetry.state
        state.proposals = self.proposals
        state.accepted = self.accepted
        state.rejected = self.rejected
        state.lr_mult = float(self.current_action.values()["lr_mult"])
        state.grad_clip = self.grad_clip
        state.weight_decay = float(self.current_action.values()["weight_decay"])
        state.residual_gate = float(self.model.residual_gate.detach().float().item())
        state.refine_scale = float(self.model.upgrades.refine_scale.detach().float().item())
        state.router_temperature = float(getattr(self.model.upgrades, "router_temperature", 1.0))
        state.batch_size = self.batch_size
        state.step_fraction = step / max(1, self.config.target_updates)
        if self.events:
            state.mean_reward = statistics.mean(
                float(event.get("best_utility", 0.0)) for event in self.events[-20:])
            state.last_gain = float(self.events[-1].get("best_utility", 0.0))
            state.verified_gain_sum = sum(
                float(event.get("best_utility", 0.0))
                for event in self.events[-50:] if event.get("accepted"))
        return state.vector()

    def observe(self, report: dict, *, step: int, updates_per_second: float) -> None:
        self.telemetry.observe(
            report, step=step, target_updates=self.config.target_updates,
            updates_per_second=updates_per_second)

    # -- branching ----------------------------------------------------------
    def should_cycle(self, step: int, remaining: int) -> bool:
        if self.config.cycle_updates <= 0 or self.config.horizon <= 0:
            return False
        if remaining < (self.config.candidates + 1) * self.config.horizon:
            return False
        return step > 0 and step % self.config.cycle_updates == 0

    def run_cycle(
        self,
        *,
        capture: Callable[[], BranchSnapshot],
        restore: Callable[[BranchSnapshot], None],
        train: Callable[[int], list[dict]],
        evaluate: Callable[[], list[float]],
        step: int,
        branch_runner: Callable[[Any, BranchSnapshot, list, int], list] | None = None,
    ) -> BranchResult:
        self._snapshot_index = 0
        cycle_start = time.monotonic()
        base = capture()
        base_seconds = time.monotonic() - cycle_start
        state = self.state_vector(step)
        pool: list[Action] = []
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        target_pool = max(1, 2 * (1 + self.config.candidates))
        while len(pool) < target_pool - 1 and attempts < 64:
            attempts += 1
            with torch.no_grad():
                action, _, _ = self.policy.sample(state, epsilon=0.10)
            if action.is_noop() or action.as_tuple() in seen:
                continue
            seen.add(action.as_tuple())
            pool.append(action)
        proposal_scores: dict[tuple[int, ...], dict] = {}
        if len(self.replay) >= 8 and pool:
            pool_states = state.to(self.device).repeat(len(pool), 1)
            pool_actions = torch.stack([action.vector() for action in pool]).to(self.device)
            with torch.no_grad():
                mean, variance = self.critic.predict(pool_states, pool_actions)
                world = self.world_model(pool_states, pool_actions)
                # Utility is higher-is-better; predicted next BPB is lower-is-better.
                rank_score = (mean + 0.25 * torch.sqrt(variance.clamp_min(0.0))
                              - 0.05 * world[:, 0] * 10.0)
            order = torch.argsort(rank_score, descending=True).tolist()
            selected = [pool[index] for index in order[:self.config.candidates]]
            for index, action in enumerate(pool):
                proposal_scores[action.as_tuple()] = {
                    "critic_mean": float(mean[index]),
                    "critic_std": float(torch.sqrt(variance[index].clamp_min(0.0))),
                    "predicted_bpb": float(world[index, 0] * 10.0),
                    "rank_score": float(rank_score[index]),
                }
        else:
            selected = pool[:self.config.candidates]
        # The counterfactual baseline must be the CURRENT policy, not neutral.
        # Using neutral here silently reset every accepted LR/gate/hierarchy
        # intervention on every tick, so RSI could never accumulate progress.
        candidates = [self.current_action] + selected
        propose_seconds = time.monotonic() - cycle_start - base_seconds

        if branch_runner is not None:
            results = branch_runner(self, base, candidates, step)
            restore_seconds = train_seconds = evaluate_seconds = capture_seconds = 0.0
        else:
            results: list[BranchResult] = []
            restore_seconds = 0.0
            train_seconds = 0.0
            evaluate_seconds = 0.0
            capture_seconds = 0.0
            for action in candidates:
                branch_start = time.monotonic()
                restore(base)
                restore_seconds += time.monotonic() - branch_start
                self.apply(action)
                train_start = time.monotonic()
                self._active_branch_batch = int(self.config.branch_batch_size)
                try:
                    reports = train(self.config.horizon)
                finally:
                    self._active_branch_batch = None
                train_seconds += time.monotonic() - train_start
                eval_start = time.monotonic()
                shard_bpbs = evaluate()
                evaluate_seconds += time.monotonic() - eval_start
                capture_start = time.monotonic()
                snapshot = capture()
                capture_seconds += time.monotonic() - capture_start
                mean_bpb = statistics.mean(shard_bpbs) if shard_bpbs else float("inf")
                results.append(BranchResult(
                    action=action, shard_bpbs=shard_bpbs, mean_bpb=mean_bpb,
                    reports=reports, snapshot=snapshot))

        noop = results[0]
        best: BranchResult | None = None
        best_diff = -float("inf")
        best_se = float("inf")
        for candidate in results[1:]:
            if len(candidate.shard_bpbs) != len(noop.shard_bpbs):
                continue
            diffs = [before - after for before, after in
                     zip(noop.shard_bpbs, candidate.shard_bpbs)]
            if not all(math.isfinite(value) for value in diffs):
                continue
            mean_diff = statistics.mean(diffs)
            se = (statistics.stdev(diffs) / math.sqrt(len(diffs))
                  if len(diffs) > 1 else float("inf"))
            candidate.utility = mean_diff
            candidate.standard_error = se
            if mean_diff > best_diff:
                best = candidate
                best_diff = mean_diff
                best_se = se

        accepted = False
        if best is not None and math.isfinite(best_diff):
            threshold = max(self.config.min_gain, self.config.z_score * best_se)
            accepted = best_diff > threshold
        chosen = best if accepted else noop
        select_start = time.monotonic()
        restore(chosen.snapshot)
        select_seconds = time.monotonic() - select_start

        self.cycles += 1
        self.branch_updates += len(results) * self.config.horizon
        self.proposals += max(0, len(results) - 1)
        self.accepted += int(accepted)
        self.rejected += max(0, len(results) - 1 - int(accepted))
        event = {
            "cycle": self.cycles,
            "step": step,
            "noop_bpb": noop.mean_bpb,
            "candidates": [
                {
                    "action": result.action.as_tuple(),
                    "label": result.action.label(),
                    "bpb": result.mean_bpb,
                    "utility": result.utility,
                    "standard_error": result.standard_error,
                    "critic_mean": proposal_scores.get(
                        result.action.as_tuple(), {}).get("critic_mean"),
                    "critic_std": proposal_scores.get(
                        result.action.as_tuple(), {}).get("critic_std"),
                    "predicted_bpb": proposal_scores.get(
                        result.action.as_tuple(), {}).get("predicted_bpb"),
                    "rank_score": proposal_scores.get(
                        result.action.as_tuple(), {}).get("rank_score"),
                }
                for result in results
            ],
            "best_utility": best_diff if best is not None else 0.0,
            "best_standard_error": best_se if best is not None else 0.0,
            "accepted": accepted,
            "chosen": chosen.action.as_tuple(),
            "branch_updates": len(results) * self.config.horizon,
            "ledger": {"proposals": self.proposals, "accepted": self.accepted,
                       "rejected": self.rejected},
        }
        self.events.append(event)

        # Every branch outcome is a training-progress transition.  The no-op
        # branch is the counterfactual; its utility is exactly zero.
        for result in results:
            utility = 0.0 if result is noop else result.utility
            next_bpb = (statistics.mean(float(r.get("bpb", 0.0)) for r in result.reports)
                        if result.reports else 0.0)
            next_grad = (statistics.mean(float(r.get("grad_norm", 0.0)) for r in result.reports)
                         if result.reports else 0.0)
            next_acc = (statistics.mean(float(r.get("bit_accuracy", 0.0)) for r in result.reports)
                        if result.reports else 0.0)
            self.replay.add(Transition(
                state=state.tolist(),
                action=list(result.action.as_tuple()),
                utility=float(utility),
                bpb=float(result.mean_bpb),
                noop_bpb=float(noop.mean_bpb),
                cost=float(self.config.horizon),
                step=int(step),
                next_bpb=next_bpb,
                next_grad_norm=next_grad,
                next_bit_accuracy=next_acc,
            ))
        self._learn()
        learn_seconds = max(0.0, time.monotonic() - cycle_start - (
            base_seconds + propose_seconds + restore_seconds + train_seconds
            + evaluate_seconds + capture_seconds + select_seconds))
        total_seconds = time.monotonic() - cycle_start

        print(
            f"RSI-V4 TIMING cycle={self.cycles} base={base_seconds:.3f}s "
            f"propose={propose_seconds:.3f}s restore={restore_seconds:.3f}s "
            f"train={train_seconds:.3f}s eval={evaluate_seconds:.3f}s "
            f"capture={capture_seconds:.3f}s select={select_seconds:.3f}s "
            f"learn={learn_seconds:.3f}s total={total_seconds:.3f}s",
            flush=True,
        )
        print(
            f"RSI-V4 CYCLE {self.cycles} step={step} noop_bpb={noop.mean_bpb:.6f} "
            f"best={'none' if best is None else best.action.label()} "
            f"diff={best_diff:+.6f} se={best_se:.6f} "
            f"accepted={accepted} ledger={self.accepted}/{self.proposals}",
            flush=True,
        )
        for result in results[1:]:
            print(
                f"RSI-V4 BRANCH {result.action.label()} bpb={result.mean_bpb:.6f} "
                f"diff={result.utility:+.6f} se={result.standard_error:.6f}",
                flush=True,
            )
        print(
            f"RSI-V4 CHOSEN {chosen.action.label()} "
            f"accepted={accepted} "
            f"cumulative_gain={sum(float(e.get('best_utility', 0.0)) for e in self.events if e.get('accepted')):+.6f}",
            flush=True,
        )
        return chosen

    # -- learning -----------------------------------------------------------
    def _batch_tensors(self, batch: list[Transition]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.tensor([item.state for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.stack([
            Action.from_tuple(item.action).vector() for item in batch
        ]).to(self.device)
        targets = torch.tensor([item.utility for item in batch], dtype=torch.float32, device=self.device)
        return states, actions, targets

    def _learn(self) -> None:
        if len(self.replay) < 4:
            return
        self.critic.train()
        for _ in range(self.config.critic_steps):
            batch = self.replay.sample(min(32, len(self.replay)))
            states, actions, targets = self._batch_tensors(batch)
            loss = self.critic.nll_loss(states, actions, targets)
            if not torch.isfinite(loss):
                break
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
        self.critic.eval()

        self.world_model.train()
        for _ in range(self.config.critic_steps):
            batch = self.replay.sample(min(32, len(self.replay)))
            states, actions, _ = self._batch_tensors(batch)
            next_bpb = torch.tensor(
                [item.next_bpb for item in batch], dtype=torch.float32,
                device=self.device)
            next_grad = torch.tensor(
                [item.next_grad_norm for item in batch], dtype=torch.float32,
                device=self.device)
            next_acc = torch.tensor(
                [item.next_bit_accuracy for item in batch], dtype=torch.float32,
                device=self.device)
            world_loss = self.world_model.loss(
                states, actions, next_bpb, next_grad, next_acc)
            if not torch.isfinite(world_loss):
                break
            self.world_optimizer.zero_grad(set_to_none=True)
            world_loss.backward()
            nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
            self.world_optimizer.step()
        self.world_model.eval()

        self.policy.train()
        for _ in range(self.config.policy_steps):
            batch = self.replay.sample(min(32, len(self.replay)))
            states, actions, targets = self._batch_tensors(batch)
            action_objects = [Action.from_tuple(item.action) for item in batch]
            log_prob = self.policy.log_prob(states, action_objects)
            with torch.no_grad():
                baseline, _ = self.critic.predict(states, actions)
            advantage = targets - baseline
            if advantage.numel() > 1 and float(advantage.std()) > 1e-8:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            entropy = self.policy.entropy(states)
            loss = -(log_prob * advantage).mean() - self.config.entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                break
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
        self.policy.eval()

    # -- persistence --------------------------------------------------------
    def _config_state(self) -> dict:
        result = {}
        for key, value in asdict(self.config).items():
            if isinstance(value, (int, float, bool, str)) or value is None:
                result[key] = value
        return result

    def state_dict(self) -> dict:
        return {
            "schema": "terminal-rsi-v4-controller-v1",
            "config": self._config_state(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "world_model": self.world_model.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "telemetry": self.telemetry.state_dict(),
            "base_lrs": list(self.base_lrs),
            "current_action": list(self.current_action.as_tuple()),
            "batch_size": self.batch_size,
            "grad_clip": self.grad_clip,
            "proposals": self.proposals,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "cycles": self.cycles,
            "branch_updates": self.branch_updates,
            "events": list(self.events),
        }

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        if state.get("schema") != "terminal-rsi-v4-controller-v1":
            raise ValueError("Wrong RSI-v4 controller checkpoint schema")
        self.policy.load_state_dict(state["policy"])
        self.critic.load_state_dict(state["critic"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state.get("world_model") is not None:
            self.world_model.load_state_dict(state["world_model"])
        if state.get("world_optimizer") is not None:
            self.world_optimizer.load_state_dict(state["world_optimizer"])
        self.replay.load_state_dict(state.get("replay"))
        self.telemetry.load_state_dict(state.get("telemetry"))
        self.base_lrs = [float(value) for value in state.get("base_lrs", self.base_lrs)]
        self._apply_runtime(Action.from_tuple(
            state.get("current_action", NEUTRAL_CHOICES)))
        self.proposals = int(state.get("proposals", self.proposals))
        self.accepted = int(state.get("accepted", self.accepted))
        self.rejected = int(state.get("rejected", self.rejected))
        self.cycles = int(state.get("cycles", self.cycles))
        self.branch_updates = int(state.get("branch_updates", self.branch_updates))
        self.events = list(state.get("events", self.events))

    def save_meta(self, path: Path) -> None:
        payload = self.state_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".pt":
            torch.save(payload, path)
            return
        path.write_text(json.dumps({
            "schema": payload["schema"],
            "config": payload["config"],
            "replay": payload["replay"],
            "events": payload["events"],
            "proposals": self.proposals,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "cycles": self.cycles,
            "branch_updates": self.branch_updates,
        }, indent=2))
