"""Contracts for the RSI-v4 counterfactual self-modification controller."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import torch
from torch import nn

from terminal_rsi.rsi_v4 import (
    ACTION_DIM, NEUTRAL_CHOICES, PRIMITIVES, STATE_DIM, Action, EnsembleCritic,
    FastStateSnapshot, ProbeShards, ProposalPolicy, ReplayBuffer, RSIv4Config,
    RSIv4Controller, Transition, WorldModel, apply_action, probe_noise_audit,
)


class TinyUpgrades(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(4, 4)
        self.refine_scale = nn.Parameter(torch.tensor(-3.0))
        self.router_temperature = 1.0


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.upgrades = TinyUpgrades()
        self.residual_gate = nn.Parameter(torch.tensor(0.0))
        self.linear = nn.Linear(4, 4)

    def forward(self, previous, target, aux=None):
        # Not a language model; only enough structure for probe tests.
        batch = len(target)
        logits = torch.zeros(batch, 8)
        bits = torch.zeros(batch, 8)
        return logits, bits


def tiny_optimizer(model):
    return torch.optim.AdamW([
        {"params": list(model.linear.parameters()), "lr": 2e-3},
        {"params": list(model.upgrades.router.parameters()), "lr": 5e-4},
        {"params": [model.residual_gate, model.upgrades.refine_scale], "lr": 1e-3},
    ])


def test_action_vector_and_neutral():
    neutral = Action.neutral()
    assert neutral.is_noop()
    assert neutral.as_tuple() == NEUTRAL_CHOICES
    vector = neutral.vector()
    assert vector.shape == (ACTION_DIM,)
    assert int(vector.sum().item()) == len(PRIMITIVES)
    assert Action.from_tuple(neutral.as_tuple()) == neutral


def test_apply_action_and_limits():
    model = TinyModel()
    optimizer = tiny_optimizer(model)
    base_lrs = [2e-3, 5e-4, 1e-3]
    action = Action.from_tuple((
        0,  # lr_mult 0.8
        2,  # upgrade_lr_mult 2.0
        2,  # grad_clip 2.0
        2,  # weight_decay 0.1
        4,  # residual_gate_delta +0.08
        0,  # refine_scale_delta -0.08
        2,  # router_temperature 1.25
        2,  # batch_size 8
    ))
    apply_action(model, optimizer, base_lrs, action)
    assert abs(optimizer.param_groups[0]["lr"] - 1.6e-3) < 1e-12
    assert abs(optimizer.param_groups[1]["lr"] - 4.0e-4) < 1e-12
    assert abs(optimizer.param_groups[2]["lr"] - 1.6e-3) < 1e-12
    assert abs(optimizer.param_groups[2]["weight_decay"] - 0.1) < 1e-12
    assert abs(float(model.residual_gate) - 0.08) < 1e-6
    assert abs(float(model.upgrades.refine_scale) + 3.08) < 1e-6
    assert abs(model.upgrades.router_temperature - 1.25) < 1e-6


def test_policy_and_critic_shapes():
    policy = ProposalPolicy(STATE_DIM)
    critic = EnsembleCritic(STATE_DIM, ACTION_DIM)
    state = torch.randn(1, STATE_DIM)
    action, log_prob, entropy = policy.sample(state)
    assert isinstance(action, Action)
    assert torch.isfinite(log_prob)
    assert torch.isfinite(entropy)
    batch_state = torch.randn(3, STATE_DIM)
    actions = [Action.neutral(), action, Action.from_tuple((1, 1, 1, 1, 1, 1, 1, 1))]
    log_probs = policy.log_prob(batch_state, actions)
    assert log_probs.shape == (3,)
    assert torch.isfinite(log_probs).all()
    action_vectors = torch.stack([a.vector() for a in actions])
    mean, variance = critic.predict(batch_state, action_vectors)
    assert mean.shape == (3,)
    assert variance.shape == (3,)
    targets = torch.tensor([0.0, 0.1, -0.1])
    loss = critic.nll_loss(batch_state, action_vectors, targets)
    assert torch.isfinite(loss)
    loss.backward()

    world = WorldModel(STATE_DIM, ACTION_DIM)
    predicted = world(batch_state, action_vectors)
    assert predicted.shape == (3, 3)
    next_bpb = torch.tensor([1.0, 2.0, 3.0])
    next_grad = torch.tensor([0.5, 1.0, 2.0])
    next_acc = torch.tensor([0.2, 0.4, 0.6])
    world_loss = world.loss(batch_state, action_vectors, next_bpb, next_grad, next_acc)
    assert torch.isfinite(world_loss)
    world_loss.backward()


def test_replay_roundtrip(tmp_path: Path):
    replay = ReplayBuffer(4)
    for index in range(6):
        replay.add(Transition(
            state=[float(index)] * STATE_DIM,
            action=list(NEUTRAL_CHOICES),
            utility=float(index) / 10,
            bpb=1.0,
            noop_bpb=1.1,
            cost=10.0,
            step=index,
        ))
    assert len(replay) == 4
    path = tmp_path / "replay.json"
    replay.save_json(path)
    restored = ReplayBuffer()
    restored.load_json(path)
    assert len(restored) == 4
    assert restored.transitions[-1].step == 5


def test_probe_shards_and_audit():
    raw = torch.randint(0, 256, (4096,), dtype=torch.uint8)
    aux = torch.zeros(len(raw) - 32, 6, dtype=torch.int32)
    probe = ProbeShards.build(
        raw, aux, n_shards=2, positions_per_shard=8, history=32, seed=7)
    assert len(probe.windows) == 2
    assert len(probe.positions) == 16
    assert len(set(probe.positions.tolist())) == 16
    model = TinyModel()
    scores = probe.evaluate(model)
    assert len(scores) == 2
    assert all(math.isfinite(score) for score in scores)
    audit = probe_noise_audit(
        model, raw, aux, history=32, n_shards=2,
        positions_per_shard=8, repeats=2, seed=11)
    assert audit["repeats"] == 2
    assert math.isfinite(audit["sd_across_shards"])


class FakeControllerHarness:
    def __init__(self, candidate_bpb, noop_bpb):
        self.candidate_bpb = candidate_bpb
        self.noop_bpb = noop_bpb

    def snapshot(self, controller, step):
        from terminal_rsi.rsi_v4 import BranchSnapshot
        return BranchSnapshot(
            model={}, optimizer={}, sampler={}, torch_rng=None,
            python_rng=None, cuda_rng=[], step=step, update_count=0,
            training=True, action=controller.current_action.as_tuple(),
            batch_size=controller.batch_size, grad_clip=controller.grad_clip)


def test_controller_accepts_statistically_better_branch():
    torch.manual_seed(1)
    random.seed(1)
    model = TinyModel()
    optimizer = tiny_optimizer(model)
    config = RSIv4Config(
        cycle_updates=10, horizon=2, candidates=1, n_shards=4,
        min_gain=0.001, z_score=1.0, critic_steps=2, policy_steps=2,
        replay_capacity=64, target_updates=100)
    probe = ProbeShards.build(
        torch.randint(0, 256, (4096,), dtype=torch.uint8),
        torch.zeros(4096 - 32, 6, dtype=torch.int32),
        n_shards=4, positions_per_shard=4, history=32, seed=3)
    controller = RSIv4Controller(model, optimizer, probe, config)
    chosen_action = Action.from_tuple((1, 1, 1, 1, 4, 4, 1, 1))
    # Force a deterministic non-neutral proposal.
    controller.policy.sample = lambda state, **kwargs: (
        chosen_action, torch.tensor(0.0), torch.tensor(0.0))

    def capture():
        return FakeControllerHarness(None, None).snapshot(controller, 10)

    def restore(snapshot):
        controller.restore_action_state(
            snapshot.action, snapshot.batch_size, snapshot.grad_clip)

    def train(steps):
        return [{"bpb": 1.0, "bit_accuracy": 0.5, "grad_norm": 1.0}] * steps

    def evaluate():
        return [1.0] * 4 if not controller.current_action.is_noop() else [2.0] * 4

    result = controller.run_cycle(
        capture=capture, restore=restore, train=train,
        evaluate=evaluate, step=10)
    assert result.action.is_noop() is False
    assert controller.accepted == 1
    assert controller.proposals >= 1
    assert controller.replay.transitions


def test_controller_state_roundtrip(tmp_path: Path):
    model = TinyModel()
    optimizer = tiny_optimizer(model)
    config = RSIv4Config(
        cycle_updates=10, horizon=2, candidates=1, n_shards=2,
        positions_per_shard=4, critic_steps=1, policy_steps=1,
        replay_capacity=16, target_updates=100)
    probe = ProbeShards.build(
        torch.randint(0, 256, (4096,), dtype=torch.uint8),
        torch.zeros(4096 - 32, 6, dtype=torch.int32),
        n_shards=2, positions_per_shard=4, history=32, seed=3)
    controller = RSIv4Controller(model, optimizer, probe, config)
    controller.proposals = 4
    controller.accepted = 1
    controller.rejected = 3
    controller.cycles = 2
    controller.replay.add(Transition(
        state=[0.0] * STATE_DIM, action=list(NEUTRAL_CHOICES),
        utility=0.01, bpb=1.0, noop_bpb=1.01, cost=2.0, step=10))
    state = controller.state_dict()
    path = tmp_path / "v4_state.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    fresh = RSIv4Controller(model, optimizer, probe, config)
    fresh.load_state_dict(loaded)
    assert fresh.proposals == 4
    assert fresh.accepted == 1
    assert fresh.rejected == 3
    assert fresh.cycles == 2
    assert len(fresh.replay) == 1
    for left, right in zip(controller.policy.parameters(), fresh.policy.parameters()):
        assert torch.equal(left, right)


def test_meta_pt_roundtrip(tmp_path: Path):
    model = TinyModel()
    optimizer = tiny_optimizer(model)
    config = RSIv4Config(
        cycle_updates=10, horizon=2, candidates=1, n_shards=2,
        positions_per_shard=4, critic_steps=1, policy_steps=1,
        replay_capacity=16, target_updates=100)
    probe = ProbeShards.build(
        torch.randint(0, 256, (4096,), dtype=torch.uint8),
        torch.zeros(4096 - 32, 6, dtype=torch.int32),
        n_shards=2, positions_per_shard=4, history=32, seed=3)
    controller = RSIv4Controller(model, optimizer, probe, config)
    controller.proposals = 7
    controller.accepted = 3
    controller.branch_updates = 42
    path = tmp_path / "meta.pt"
    controller.save_meta(path)
    fresh = RSIv4Controller(
        TinyModel(), tiny_optimizer(model), probe, config, meta_path=path)
    assert fresh.proposals == 7
    assert fresh.accepted == 3
    assert fresh.branch_updates == 42


def test_branch_snapshot_restores_exact_model_and_optimizer_state():
    import copy
    model = TinyModel()
    optimizer = tiny_optimizer(model)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        loss = (model.linear.weight ** 2).sum() + model.residual_gate ** 2
        loss.backward()
        optimizer.step()

    model_snapshot = {key: value.detach().clone()
                      for key, value in model.state_dict().items()}
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    one_step()
    assert not torch.equal(model_snapshot["linear.weight"], model.linear.weight)
    model.load_state_dict(model_snapshot)
    optimizer.load_state_dict(optimizer_snapshot)
    for key, value in model.state_dict().items():
        assert torch.equal(value, model_snapshot[key])
    assert len(optimizer.state) == 0


def test_controller_uses_critic_and_world_model_ranking():
    import itertools
    torch.manual_seed(4)
    random.seed(4)
    model = TinyModel()
    optimizer = tiny_optimizer(model)
    config = RSIv4Config(
        cycle_updates=10, horizon=2, candidates=1, n_shards=2,
        min_gain=0.001, z_score=1.0, critic_steps=1, policy_steps=1,
        replay_capacity=64, target_updates=100)
    probe = ProbeShards.build(
        torch.randint(0, 256, (4096,), dtype=torch.uint8),
        torch.zeros(4096 - 32, 6, dtype=torch.int32),
        n_shards=2, positions_per_shard=4, history=32, seed=3)
    controller = RSIv4Controller(model, optimizer, probe, config)
    for index in range(10):
        controller.replay.add(Transition(
            state=[0.01 * index] * STATE_DIM,
            action=list(NEUTRAL_CHOICES),
            utility=0.001 * index,
            bpb=5.0,
            noop_bpb=5.0,
            cost=2.0,
            step=index,
            next_bpb=4.9,
            next_grad_norm=1.0,
            next_bit_accuracy=0.5,
        ))
    candidates = [
        Action.from_tuple((0, 0, 0, 0, 0, 0, 0, 0)),
        Action.from_tuple((1, 1, 1, 1, 1, 1, 1, 1)),
        Action.from_tuple((2, 2, 2, 2, 2, 2, 2, 2)),
    ]
    iterator = itertools.cycle(candidates)
    controller.policy.sample = lambda state, **kwargs: (
        next(iterator), torch.tensor(0.0), torch.tensor(0.0))

    def capture():
        return FakeControllerHarness(None, None).snapshot(controller, 10)

    def train(steps):
        return [{"bpb": 1.0, "bit_accuracy": 0.5, "grad_norm": 1.0}] * steps

    def evaluate():
        return [1.0] * 2 if not controller.current_action.is_noop() else [2.0] * 2

    controller.run_cycle(
        capture=capture, restore=lambda snapshot: None,
        train=train, evaluate=evaluate, step=10)
    assert controller.events
    scored = [item for item in controller.events[-1]["candidates"]
              if item.get("critic_mean") is not None]
    assert scored
    assert controller.replay.transitions[-1].next_bpb > 0


def test_fast_state_snapshot_restores_exact_tensors():
    model = TinyModel()
    optimizer = tiny_optimizer(model)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = (model.linear.weight ** 2).sum() + model.residual_gate ** 2
        loss.backward()
        optimizer.step()

    step()  # create Adam state
    snapshot = FastStateSnapshot(model, optimizer).capture()
    before_model = {key: value.detach().clone()
                    for key, value in model.state_dict().items()}
    before_state = {
        index: {key: (value.detach().clone() if torch.is_tensor(value) else value)
                for key, value in state.items()}
        for index, state in optimizer.state.items()
    }
    before_groups = [dict(group) for group in optimizer.param_groups]
    for _ in range(3):
        step()
    optimizer.param_groups[0]["lr"] = 123.0
    assert not torch.equal(before_model["linear.weight"], model.linear.weight)
    snapshot.restore()
    for key, value in model.state_dict().items():
        assert torch.equal(value, before_model[key])
    for index, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(value, before_state[index][key])
            else:
                assert value == before_state[index][key]
    for group, before in zip(optimizer.param_groups, before_groups):
        assert group["lr"] == before["lr"]
        assert group["weight_decay"] == before["weight_decay"]


def test_controller_rejects_worse_branch():
    torch.manual_seed(2)
    random.seed(2)
    model = TinyModel()
    optimizer = tiny_optimizer(model)
    config = RSIv4Config(
        cycle_updates=10, horizon=2, candidates=1, n_shards=4,
        min_gain=0.001, z_score=1.0, critic_steps=2, policy_steps=2,
        replay_capacity=64, target_updates=100)
    probe = ProbeShards.build(
        torch.randint(0, 256, (4096,), dtype=torch.uint8),
        torch.zeros(4096 - 32, 6, dtype=torch.int32),
        n_shards=4, positions_per_shard=4, history=32, seed=3)
    controller = RSIv4Controller(model, optimizer, probe, config)
    chosen_action = Action.from_tuple((0, 0, 0, 0, 0, 0, 0, 0))
    controller.policy.sample = lambda state, **kwargs: (
        chosen_action, torch.tensor(0.0), torch.tensor(0.0))

    def capture():
        return FakeControllerHarness(None, None).snapshot(controller, 10)

    def train(steps):
        return [{"bpb": 1.0, "bit_accuracy": 0.5, "grad_norm": 1.0}] * steps

    def evaluate():
        return [2.0] * 4 if not controller.current_action.is_noop() else [1.0] * 4

    result = controller.run_cycle(
        capture=capture, restore=lambda snapshot: None,
        train=train, evaluate=evaluate, step=10)
    assert result.action.is_noop()
    assert controller.accepted == 0
    assert controller.rejected >= 1
