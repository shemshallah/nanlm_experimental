#!/usr/bin/env python3
"""Independent R30 topology, causality, inherited-identity and gradient tests."""
import math
import torch
from torch import nn

from r30_diamond_overmind import (
    HEADS, PAIR_COUNT, TRIPLE_COUNT, SparseDiamondOvermind,
    attach_r30, pair_address, triple_address,
)
from r23_stream_carry import ContinuousPALState


def check_lattice():
    assert HEADS == 256 and PAIR_COUNT == 32640 and TRIPLE_COUNT == 2763520
    pairs = torch.triu_indices(HEADS, HEADS, offset=1)
    ids = pair_address(*pairs)
    assert ids.numel() == PAIR_COUNT
    assert torch.equal(ids.sort().values, torch.arange(PAIR_COUNT))
    assert torch.equal(ids, pair_address(pairs[1], pairs[0]))
    tri_i = torch.tensor([0, 0, 0, 1, 253], dtype=torch.long)
    tri_j = torch.tensor([1, 1, 2, 2, 254], dtype=torch.long)
    tri_k = torch.tensor([2, 3, 3, 3, 255], dtype=torch.long)
    ids = triple_address(tri_i, tri_j, tri_k)
    assert torch.equal(ids, torch.tensor([0, 1, 254, 32385, TRIPLE_COUNT - 1]))
    assert torch.equal(ids, triple_address(tri_k, tri_i, tri_j))
    try:
        pair_address(torch.tensor([4]), torch.tensor([4]))
        raise AssertionError("duplicate pair was accepted")
    except ValueError:
        pass


def check_overmind():
    torch.manual_seed(30)
    module = SparseDiamondOvermind(48, width=32, active_heads=8,
                                   active_pairs=12, active_triples=10,
                                   active_apex=6, active_abstract=5,
                                   triple_buckets=4096)
    states = torch.randn(2, 4, 48)
    pristine = states.clone()
    delta, detail, memory = module(states, return_detail=True, return_apex=True)
    assert torch.equal(states, pristine), "Overmind mutated inherited PAL states"
    assert torch.equal(delta, torch.zeros_like(delta)), "not exact zero-output migration"
    assert detail["base_heads"].shape == (2, 4, 8)
    assert detail["pair_addresses"].shape == (2, 4, 12)
    assert detail["triangle_addresses"].shape == (2, 4, 10)
    assert detail["apex_heads"].shape == (2, 4, 6)
    assert memory.shape == (2, HEADS, 32)
    assert int(detail["pair_addresses"].max()) < PAIR_COUNT
    assert int(detail["triangle_addresses"].max()) < TRIPLE_COUNT
    assert bool((detail["active_pair_edges"] <= math.comb(8, 2)).all())

    # No future token may influence logits or concept routing at an earlier time.
    module.output_gain.data.fill_(.1)
    initial, details_a = module(states, return_detail=True)
    future_changed = states.clone()
    future_changed[:, -1] += 123
    future, details_b = module(future_changed, return_detail=True)
    assert torch.equal(initial[:, :-1], future[:, :-1])
    for key in details_a:
        assert torch.equal(details_a[key][:, :-1], details_b[key][:, :-1]), key

    # Streaming all four tokens and two consecutive chunks should agree.
    all_delta, all_apex = module(states, return_apex=True)
    left, history = module(states[:, :2], return_apex=True)
    right, history = module(states[:, 2:], prior_apex=history, return_apex=True)
    assert torch.allclose(torch.cat((left, right), 1), all_delta, atol=1e-6, rtol=1e-6)
    assert torch.allclose(history, all_apex, atol=1e-6, rtol=1e-6)

    stream = ContinuousPALState("book-A")
    stream.r30_apex = history
    stream.detach()
    assert stream.r30_apex.grad_fn is None
    stream.reset("book-B")
    assert stream.r30_apex is None

    loss = initial.square().mean()
    loss.backward()
    for key in ("head_router.weight", "pair_addresses.weight", "triple_hash.weight",
                "apex_heads", "apex_router.weight", "triple_feedback.weight",
                "down_plus.weight", "apex_memory_update.weight",
                "temporal_feedback.weight", "output_gain"):
        param = dict(module.named_parameters())[key]
        assert param.grad is not None, f"missing backward path: {key}"
        assert bool(torch.isfinite(param.grad).all()), f"nonfinite gradient: {key}"
        assert float(param.grad.abs().sum()) > 0, f"dead gradient: {key}"


def check_optimizer_inheritance():
    torch.manual_seed(31)

    class Original(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.memory = nn.Module()
            self.model.memory.subword_out = nn.Linear(40, 300)
            self.legacy = nn.Linear(40, 40)

    old = Original()
    optimizer = torch.optim.Adam(old.parameters(), lr=.001)
    old.legacy(torch.randn(3, 40)).square().mean().backward()
    optimizer.step()
    prior_params = {name: value.detach().clone() for name, value in old.named_parameters()}
    prior_optim = optimizer.state_dict()
    attach_r30(old, optimizer, width=24, active_heads=6,
               active_pairs=8, active_triples=7, active_apex=4)
    assert len(optimizer.param_groups) == len(prior_optim["param_groups"]) + 1
    assert prior_optim["state"].keys() == optimizer.state_dict()["state"].keys()
    for name, previous in prior_params.items():
        assert torch.equal(dict(old.named_parameters())[name], previous), name
    for p_id, old_state in prior_optim["state"].items():
        for name, prev in old_state.items():
            now = optimizer.state_dict()["state"][p_id][name]
            assert torch.equal(prev, now) if torch.is_tensor(prev) else prev == now


if __name__ == "__main__":
    torch.set_num_threads(2)
    check_lattice()
    check_overmind()
    check_optimizer_inheritance()
    print("R30 DIAMOND OVERMIND PASS: exact 256/32640/2763520/256 topology; "
          "zero-init; gradient; causal future isolation; Adam inheritance")
