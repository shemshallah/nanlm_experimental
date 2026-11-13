"""FA2 R30: sparse, bidirectional 256 -> C(256,2) -> C(256,3) -> 256 Overmind.

Logical node counts are exact: 256 base heads, 32,640 unique pairs,
2,763,520 unique triangles, and 256 independently trainable apex heads.
Only concepts induced by the top-k active heads are materialized per token.
The zero-initialized output gain guarantees exact inherited FA2 predictions.

This module is an *experimental residual*, not an accepted language-model gain.
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F


HEADS = 256
PAIR_COUNT = HEADS * (HEADS - 1) // 2
TRIPLE_COUNT = HEADS * (HEADS - 1) * (HEADS - 2) // 6


def pair_address(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
    """Canonical lexicographic address of an unordered pair of 256 heads."""
    a, b = torch.minimum(i, j), torch.maximum(i, j)
    if torch.any(a == b) or torch.any(a < 0) or torch.any(b >= HEADS):
        raise ValueError("pair addresses require two different heads in [0,256)")
    return a * (2 * HEADS - a - 1) // 2 + b - a - 1


def triple_address(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Unique lexicographic rank in the 2,763,520-element triple lattice."""
    a, _ = torch.sort(torch.stack((i, j, k), dim=-1), dim=-1)
    i, j, k = a.unbind(-1)
    if torch.any(i == j) or torch.any(j == k) or torch.any(i < 0) or torch.any(k >= HEADS):
        raise ValueError("triple addresses require three distinct heads in [0,256)")
    c3 = lambda n: n * (n - 1) * (n - 2) // 6
    c2 = lambda n: n * (n - 1) // 2
    return TRIPLE_COUNT - c3(HEADS - i) + c2(HEADS - i - 1) - c2(HEADS - j) + k - j - 1


class SparseDiamondOvermind(nn.Module):
    """Trainable, causal per-token diamond mega-layer with downward feedback.

    Upward: base heads -> sparse abstract prototypes -> pair concepts ->
    triangle concepts -> apex heads. Downward: apex attention -> triangles ->
    paired S+/S- edges -> base heads -> original PAL hidden state and decoder.

    Each apex head can also keep a document-bound recurrent state, updated
    causally across tokens and returned to the caller for cross-chunk carry.
    """

    def __init__(
        self,
        dim: int,
        *,
        width: int = 64,
        active_heads: int = 12,
        active_pairs: int = 24,
        active_triples: int = 24,
        active_apex: int = 8,
        abstract_count: int = 64,
        active_abstract: int = 8,
        pair_rank: int = 8,
        triple_buckets: int = 16384,
        feedback_steps: int = 2,
    ):
        super().__init__()
        if dim < 1 or width < 4 or not 3 <= active_heads <= HEADS:
            raise ValueError("invalid diamond hidden size or active head count")
        num_pairs = math.comb(active_heads, 2)
        num_triples = math.comb(active_heads, 3)
        if not (1 <= active_pairs <= num_pairs and 1 <= active_triples <= num_triples):
            raise ValueError("invalid active pair or triangle budget")
        if not (1 <= active_apex <= HEADS and 1 <= active_abstract <= abstract_count):
            raise ValueError("invalid apex or abstract budget")
        if not 1 <= feedback_steps <= 4 or pair_rank < 1 or triple_buckets < 1:
            raise ValueError("invalid feedback, pair rank, or triple hash budget")

        self.dim, self.width = dim, width
        self.active_heads, self.active_pairs = active_heads, active_pairs
        self.active_triples, self.active_apex = active_triples, active_apex
        self.active_abstract, self.feedback_steps = active_abstract, feedback_steps
        self.triple_buckets = triple_buckets

        pair_slots = list(itertools.combinations(range(active_heads), 2))
        tri_slots = list(itertools.combinations(range(active_heads), 3))
        pair_pos = {pair: index for index, pair in enumerate(pair_slots)}
        tri_edges = [(pair_pos[(a, b)], pair_pos[(a, c)], pair_pos[(b, c)])
                     for a, b, c in tri_slots]
        self.register_buffer("pair_slots", torch.tensor(pair_slots, dtype=torch.long), persistent=False)
        self.register_buffer("tri_slots", torch.tensor(tri_slots, dtype=torch.long), persistent=False)
        self.register_buffer("tri_edges", torch.tensor(tri_edges, dtype=torch.long), persistent=False)

        self.input_norm = nn.LayerNorm(dim)
        self.project_in = nn.Linear(dim, width)
        self.head_router = nn.Linear(width, HEADS)
        self.base_heads = nn.Parameter(torch.randn(HEADS, width) / math.sqrt(width))
        self.phases = nn.Parameter(2 * math.pi * torch.arange(HEADS) / HEADS)

        self.abstract_keys = nn.Parameter(torch.randn(abstract_count, width) / math.sqrt(width))
        self.abstract_query = nn.Linear(width, width)
        self.pair_addresses = nn.Embedding(PAIR_COUNT, pair_rank)
        self.pair_decode = nn.Linear(pair_rank, width, bias=False)
        self.pair_build = nn.Linear(2 * width + 2, width)
        self.pair_score = nn.Linear(width, 1)

        self.triple_hash = nn.Embedding(triple_buckets, pair_rank)
        self.triple_decode = nn.Linear(pair_rank, width, bias=False)
        self.triple_build = nn.Linear(2 * width + 2, width)
        self.triple_score = nn.Linear(width, 1)

        self.apex_heads = nn.Parameter(torch.randn(HEADS, width) / math.sqrt(width))
        self.apex_router = nn.Linear(2 * width, HEADS)
        self.apex_update = nn.Linear(2 * width, width)
        self.apex_memory_update = nn.Linear(2 * width, width)
        self.apex_memory_gate = nn.Parameter(torch.tensor(-2.))
        self.temporal_feedback = nn.Linear(width, width, bias=False)
        self.triple_feedback = nn.Linear(width, width, bias=False)
        self.apex_feedback = nn.Linear(width, width, bias=False)
        self.down_plus = nn.Linear(width, width, bias=False)
        self.down_minus = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(2 * width, dim, bias=False)
        self.output_gain = nn.Parameter(torch.zeros(()))

    def forward(self, states: torch.Tensor, *, prior_apex: torch.Tensor | None = None,
                return_detail: bool = False, return_apex: bool = False):
        if states.ndim != 3 or states.shape[-1] != self.dim or states.shape[1] == 0:
            raise ValueError("expected PAL states [batch, time, hidden_dim]")
        batch, steps, _ = states.shape
        if prior_apex is not None and prior_apex.shape != (batch, HEADS, self.width):
            raise ValueError("apex carry must be [batch, 256, overmind_width]")
        x = self.project_in(self.input_norm(states)).reshape(-1, self.width)
        n, w = x.shape

        # Bottom heads and sparse abstract concepts are content routed.
        head_logits = self.head_router(x)
        head_selected = head_logits.topk(self.active_heads, dim=-1)
        head_ids = head_selected.indices.sort(dim=-1).values
        head_scores = head_logits.gather(-1, head_ids)
        head_probs = F.softmax(head_scores, dim=-1)
        head_values = self.base_heads[head_ids] + x[:, None, :] * head_probs[..., None]

        abstract_logits = self.abstract_query(x) @ self.abstract_keys.T / math.sqrt(w)
        abstract_top = abstract_logits.topk(self.active_abstract, dim=-1)
        abstract = (F.softmax(abstract_top.values, -1)[..., None]
                    * self.abstract_keys[abstract_top.indices]).sum(dim=1)

        # C(256,2), with a unique learned low-rank code per physical pair ID.
        a, b = self.pair_slots.unbind(-1)
        ids_a, ids_b = head_ids[:, a], head_ids[:, b]
        pair_ids = pair_address(ids_a, ids_b)
        ha, hb = head_values[:, a], head_values[:, b]
        angle = self.phases[ids_b] - self.phases[ids_a]
        plus, minus = (ha + hb) * .5, (ha - hb) * .5
        pair = torch.tanh(self.pair_build(torch.cat((plus, minus,
                             angle.cos()[..., None], angle.sin()[..., None]), -1))
                          + self.pair_decode(self.pair_addresses(pair_ids)))
        pair_logits = self.pair_score(pair).squeeze(-1) + head_scores[:, a] + head_scores[:, b]
        chosen_pairs = pair_logits.topk(self.active_pairs, dim=-1).indices
        pair_mask = torch.zeros_like(pair_logits).scatter(1, chosen_pairs, 1.)

        # C(256,3): exact tuple ID; stored feature is factored + hash-residual.
        c0, c1, c2 = self.tri_slots.unbind(-1)
        triples = triple_address(head_ids[:, c0], head_ids[:, c1], head_ids[:, c2])
        tri_edges = self.tri_edges
        pa, pb, pc = (pair[:, tri_edges[:, z]] for z in range(3))
        phase_a, phase_b, phase_c = (self.phases[head_ids[:, z]] for z in (c0, c1, c2))
        # Signed oriented triangle area in the complex unit circle.
        area = ((phase_b - phase_a).sin() + (phase_c - phase_b).sin()
                - (phase_c - phase_a).sin()) * .5
        curvature = ((phase_b - phase_a).cos() + (phase_c - phase_b).cos()
                     + (phase_c - phase_a).cos()) / 3
        tri = torch.tanh(self.triple_build(torch.cat(((pa + pb + pc) / 3,
                         (pa + pc - 2 * pb) / 3, area[..., None], curvature[..., None]), -1))
                         + self.triple_decode(self.triple_hash(triples % self.triple_buckets)))
        support = pair_mask[:, tri_edges].sum(dim=-1)
        tri_logits = (self.triple_score(tri).squeeze(-1)
                      + (head_scores[:, c0] + head_scores[:, c1] + head_scores[:, c2]) / 3
                      + .25 * support)
        chosen_tri = tri_logits.topk(self.active_triples, dim=-1)
        tri = tri.gather(1, chosen_tri.indices[..., None].expand(-1, -1, w))
        tri_ids = triples.gather(1, chosen_tri.indices)
        chosen_edges = tri_edges[chosen_tri.indices]

        # Each apex head is independently trainable, and attends to selected
        # triadic concepts; two alternating passes implement the feedback loop.
        tri_weight = F.softmax(chosen_tri.values, -1)
        seed = (tri * tri_weight[..., None]).sum(dim=1)
        apex_logits = self.apex_router(torch.cat((x + abstract, seed), -1))
        apex_choice = apex_logits.topk(self.active_apex, dim=-1)
        apex_probs = F.softmax(apex_choice.values, dim=-1)
        apex = (self.apex_heads[apex_choice.indices] * apex_probs[..., None]
                + seed[:, None, :])
        tri_state = tri
        for _ in range(self.feedback_steps):
            upward = torch.softmax(apex @ tri_state.transpose(1, 2) / math.sqrt(w)
                                     + chosen_tri.values[:, None, :], dim=-1)
            apex_context = upward @ tri_state
            apex = torch.tanh(self.apex_update(torch.cat((apex, apex_context), dim=-1)))
            downward = torch.softmax(upward.transpose(1, 2), dim=-1)
            tri_state = tri_state + .25 * self.triple_feedback(downward @ apex)
            apex = apex + .25 * self.apex_feedback(apex_context)

        # Document-bound recurrent Overmind: each selected apex head retains
        # its own previous concept state. No future token can affect an earlier
        # token's carry. Inactive apex heads retain their previous values.
        recurrent = (states.new_zeros((batch, HEADS, w)) if prior_apex is None
                     else prior_apex)
        memory_mix = torch.sigmoid(self.apex_memory_gate)
        apex_trajectory = apex.reshape(batch, steps, self.active_apex, w)
        apex_indices = apex_choice.indices.reshape(batch, steps, self.active_apex)
        apex_weights = apex_probs.reshape(batch, steps, self.active_apex)
        summaries = []
        for token in range(steps):
            chosen = apex_indices[:, token]
            old = recurrent.gather(1, chosen[..., None].expand(-1, -1, w))
            proposed = torch.tanh(self.apex_memory_update(torch.cat(
                (apex_trajectory[:, token], old), dim=-1)))
            updated = (1 - memory_mix) * old + memory_mix * proposed
            recurrent = recurrent.scatter(1, chosen[..., None].expand(-1, -1, w), updated)
            summaries.append((updated * apex_weights[:, token, :, None]).sum(dim=1))
        history = torch.stack(summaries, dim=1).reshape(n, w)
        tri_state = tri_state + .25 * self.temporal_feedback(history)[:, None, :]

        # Triple-to-pair incidence, then orientation-aware S+/S- messages.
        pair_fb = torch.zeros(n, len(self.pair_slots), w, device=x.device, dtype=x.dtype)
        pair_count = torch.zeros(n, len(self.pair_slots), 1, device=x.device, dtype=x.dtype)
        edges = chosen_edges.reshape(n, -1)
        pair_fb.scatter_add_(1, edges[..., None].expand(-1, -1, w),
                             tri_state[:, :, None, :].expand(-1, -1, 3, -1).reshape(n, -1, w))
        pair_count.scatter_add_(1, edges[..., None],
                                torch.ones(n, edges.shape[1], 1, device=x.device, dtype=x.dtype))
        pair_act = torch.maximum(pair_mask, pair_count.squeeze(-1).gt(0).to(x.dtype))
        pair_state = pair + pair_fb / pair_count.clamp_min(1)
        pair_strength = F.softmax(pair_logits.masked_fill(pair_act == 0, -1e4), dim=-1)
        pair_up = self.down_plus(pair_state) * pair_strength[..., None]
        pair_down = self.down_minus(pair_state) * pair_strength[..., None]

        heads_back = torch.zeros(n, self.active_heads, w, device=x.device, dtype=x.dtype)
        heads_back.index_add_(1, a, pair_up + pair_down)
        heads_back.index_add_(1, b, pair_up - pair_down)
        # Feed the apex's abstracted triangle attention back into PAL's
        # original hidden state. Output starts exactly zero for safe migration.
        head_context = (heads_back * head_probs[..., None]).sum(dim=1)
        feedback = self.readout(torch.cat((head_context + abstract, seed), dim=-1))
        delta = torch.tanh(self.output_gain) * feedback.reshape(batch, steps, self.dim)
        if not return_detail:
            return (delta, recurrent) if return_apex else delta
        details = {
            "base_heads": head_ids.reshape(batch, steps, -1).detach(),
            "abstract_nodes": abstract_top.indices.reshape(batch, steps, -1).detach(),
            "pair_addresses": pair_ids.gather(1, chosen_pairs).reshape(batch, steps, -1).detach(),
            "triangle_addresses": tri_ids.reshape(batch, steps, -1).detach(),
            "apex_heads": apex_choice.indices.reshape(batch, steps, -1).detach(),
            "active_pair_edges": pair_act.sum(dim=-1).reshape(batch, steps).detach(),
        }
        return (delta, details, recurrent) if return_apex else (delta, details)


def attach_r30(model: nn.Module, optimizer=None, *, learning_rate: float = 0.0003, **kwargs):
    """Attach the residual to the SAME UnifiedFA2, preserving existing states."""
    if hasattr(model, "r30"):
        raise ValueError("R30 Overmind already attached")
    dim = model.model.memory.subword_out.in_features
    residual = SparseDiamondOvermind(dim, **kwargs)
    model.add_module("r30", residual)
    if optimizer is not None:
        optimizer.add_param_group({"params": list(residual.parameters()),
                                   "lr": learning_rate, "weight_decay": 1e-5})
    return residual
