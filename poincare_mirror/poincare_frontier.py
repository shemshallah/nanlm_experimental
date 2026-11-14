#!/usr/bin/env python3
"""Poincare Mirror LM -- a causal, paired-MoE research scaffold.

Goals:
  * 256 signed experts = 128 structurally mirrored (+/-) pairs.
  * Per-token causal context (RoPE attention), with no future-token injection.
  * Experts produce hidden features, NOT full-vocabulary logits.
  * Optional Poincare-ball pair routing, safe inside the unit ball.
  * One continuous, resumable optimizer/model checkpoint (latest.pt).
  * Strictly distinct train/validation input files and token counts.
  * Optional same-head prediction at horizons +2/+3 (training labels only).

This is a *prototype*, NOT a distributed frontier training kernel. Its Python
expert dispatch will need replacing with grouped GEMM / expert parallelism.

Requires: PyTorch, NumPy; optionally SentencePiece for a fixed tokenizer.
Examples:
  python poincare_frontier.py smoke
  python poincare_frontier.py prepare --train-text train.txt --valid-text valid.txt --out-dir data
  python poincare_frontier.py prepare-books --source-dir books/ --out-dir data
  python poincare_frontier.py train --data-dir data --run-dir run --steps 1000
  python poincare_frontier.py generate --run-dir run --prompt 'Once upon a time'
"""
import argparse
import copy
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 257
    dim: int = 256
    layers: int = 4
    heads: int = 4
    pairs: int = 128
    topk_pairs: int = 2
    expert_hidden: int = 128
    context: int = 256
    moe_every: int = 2
    hyperbolic_strength: float = 0.2
    balance_weight: float = 0.01
    future2_weight: float = 0.10
    future3_weight: float = 0.05
    future_mode: str = 'offset'
    future_block_index: int = -1  # fixed transition anchor when growing model depth
    norm_divisor: int = 0  # old width when growing; preserves RMSNorm exactly
    dispatch_mode: str = 'grouped'  # prototype loop or chunked batched gather
    gather_chunk: int = 32
    symbiotic_rank: int = 0  # opt-in sparse geometric Diamond
    # Non-divisible widening preserves the original RoPE head geometry.
    # For example, 192D / 16 heads (12D each) -> 256D / 20 mixed heads:
    # the 16 old 12D heads plus four new 16D heads.
    legacy_attention_dim: int = 0
    legacy_attention_heads: int = 0
    hierarchy_rank: int = 0  # opt-in byte/pair/word/sentence causal feedback
    attention_window: int = 0  # 0 = inherited dense attention; nonzero bounds long-context work
    geometry_memory_chunk: int = 0  # 0 = inherited geometry; nonzero enables causal apex-memory loops
    micro_meta: bool = False  # 256 point microtransformers -> sparse triangle cloud -> 256 apex MoE
    simplex_transfer: bool = False  # sparse signed boundary transport, simplex memory, apex tetra bridges
    triangle_curvature: bool = False  # opt-in hyperbolic triangle phase/rotation memory
    curvature_aux_weight: float = 0.02  # directly supervise geometric writes on next byte

    def check(self):
        if self.legacy_attention_dim:
            old_dim, old_heads = self.legacy_attention_dim, self.legacy_attention_heads
            added_dim, added_heads = self.dim - old_dim, self.heads - old_heads
            assert 0 < old_dim < self.dim and 0 < old_heads < self.heads
            assert old_dim % old_heads == 0 and (old_dim // old_heads) % 2 == 0
            assert added_dim % added_heads == 0 and (added_dim // added_heads) % 2 == 0
        else:
            assert self.legacy_attention_heads == 0
            assert self.dim % self.heads == 0
            assert (self.dim // self.heads) % 2 == 0
        assert 1 <= self.topk_pairs <= self.pairs
        assert self.moe_every >= 1 and self.layers >= 1
        assert self.context >= 4
        assert self.future_mode in ('offset', 'rollout')
        assert self.future_block_index == -1 or 0 <= self.future_block_index < self.layers
        assert self.norm_divisor == 0 or 0 < self.norm_divisor <= self.dim
        assert self.dispatch_mode in ('grouped', 'gather') and self.gather_chunk > 0
        assert self.symbiotic_rank == 0 or 8 <= self.symbiotic_rank <= 64
        assert self.hierarchy_rank == 0 or 8 <= self.hierarchy_rank <= 128
        assert self.attention_window == 0 or self.attention_window >= 4
        assert self.geometry_memory_chunk == 0 or self.geometry_memory_chunk >= 4
        assert not self.geometry_memory_chunk or self.symbiotic_rank
        assert not self.micro_meta or self.symbiotic_rank
        assert not self.micro_meta or self.symbiotic_rank % 2 == 0
        assert not self.simplex_transfer or self.micro_meta
        assert not self.triangle_curvature or self.simplex_transfer
        assert 0 <= self.curvature_aux_weight <= 1


class RMSNorm(nn.Module):
    def __init__(self, dim, divisor=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.divisor = int(divisor or dim)

    def forward(self, x):
        z = x.float()
        if self.divisor == z.shape[-1]:
            variance = z.square().mean(dim=-1, keepdim=True)
        else:
            # When widening, the original coordinate normalization remains
            # invariant for [old_state, zeros] instead of silently rescaling.
            variance = z.square().sum(dim=-1, keepdim=True) / self.divisor
        z = z * torch.rsqrt(variance + 1e-6)
        return z.to(x.dtype) * self.scale


@lru_cache(maxsize=32)
def _rope_trig(length, dim, device, dtype):
    """Immutable position factors shared by every block; never part of state_dict."""
    inv = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    theta = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv)
    return theta.cos().to(dtype)[None, None], theta.sin().to(dtype)[None, None]


def apply_rope(x):
    """x: [batch, heads, time, head_dim], with even head dimension."""
    dim = x.shape[-1]
    co, si = _rope_trig(x.shape[-2], dim, x.device, x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * co - odd * si, even * si + odd * co), -1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, dim, heads, legacy_dim=0, legacy_heads=0, window=0):
        super().__init__()
        self.heads = heads
        self.legacy_dim = legacy_dim
        self.legacy_heads = legacy_heads
        self.window = window
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, d).unbind(2)

        def attend(q0, k0, v0, heads):
            width = q0.shape[-1]
            q0 = apply_rope(q0.reshape(b, t, heads, width // heads).transpose(1, 2))
            k0 = apply_rope(k0.reshape(b, t, heads, width // heads).transpose(1, 2))
            v0 = v0.reshape(b, t, heads, width // heads).transpose(1, 2)
            if not self.window or t <= self.window:
                # The inherited short-context operation is bit-for-bit intact.
                a = F.scaled_dot_product_attention(q0, k0, v0, is_causal=True, dropout_p=0.)
            else:
                # Query tiling bounds peak score memory and avoids materializing
                # a quadratic [time,time] mask at 32K+ bytes. RoPE coordinates
                # remain absolute; both the window and causality are explicit.
                pieces = []
                chunk = min(128, self.window)
                for start in range(0, t, chunk):
                    end = min(t, start + chunk)
                    left = max(0, start - self.window + 1)
                    qi = torch.arange(start, end, device=x.device)
                    ki = torch.arange(left, end, device=x.device)
                    allowed = (ki[None, :] <= qi[:, None]) & (
                        ki[None, :] > qi[:, None] - self.window)
                    pieces.append(F.scaled_dot_product_attention(
                        q0[:, :, start:end], k0[:, :, left:end], v0[:, :, left:end],
                        attn_mask=allowed, is_causal=False, dropout_p=0.))
                a = torch.cat(pieces, dim=-2)
            return a.transpose(1, 2).contiguous().view(b, t, width)

        if self.legacy_dim:
            cut = self.legacy_dim
            old = attend(q[..., :cut], k[..., :cut], v[..., :cut], self.legacy_heads)
            new = attend(q[..., cut:], k[..., cut:], v[..., cut:],
                         self.heads - self.legacy_heads)
            a = torch.cat((old, new), dim=-1)
        else:
            a = attend(q, k, v, self.heads)
        return self.out(a)


class CausalTokenHierarchy(nn.Module):
    """Four causal token scales with two shared cross-scale attention loops.

    The byte head and loss remain authoritative. The pair level represents a
    causal two-byte proxy, *not* a trained BPE vocabulary. Word and sentence
    levels use online prefix means; an unfinished segment never reads later
    bytes. Four scales attend to each other only at the same time position,
    then send one learned residual back to the original byte stream.
    """
    def __init__(self, dim, rank):
        super().__init__()
        self.rank = rank
        self.input = nn.Linear(dim, rank, bias=False)
        self.scale_qkv = nn.Linear(rank, rank * 3, bias=False)
        self.scale_logits = nn.Parameter(torch.zeros(4))
        self.to_hidden = nn.Linear(rank, dim, bias=False)

    @staticmethod
    def prefix_mean(z, starts):
        b, t, r = z.shape
        places = torch.arange(t, device=z.device).expand(b, -1)
        segment_start = torch.cummax(torch.where(starts, places, 0), dim=1).values
        accumulated = F.pad(z.cumsum(1), (0, 0, 1, 0))
        before = accumulated.gather(1, segment_start.unsqueeze(-1).expand(-1, -1, r))
        return (accumulated[:, 1:] - before) / (places - segment_start + 1).unsqueeze(-1)

    def forward(self, hidden, byte_ids):
        b, t, _ = hidden.shape
        z = self.input(hidden)
        previous = F.pad(byte_ids[:, :-1], (1, 0), value=256)
        word_break = ((previous == 32) | (previous == 9) | (previous == 10) |
                      (previous == 13) | (previous == 256))
        sentence_break = ((previous == 46) | (previous == 33) |
                          (previous == 63) | (previous == 10) | (previous == 256))
        pair = torch.cat((z[:, :1], (z[:, 1:] + z[:, :-1]) * 0.5), dim=1)
        levels = torch.stack((z, pair, self.prefix_mean(z, word_break),
                              self.prefix_mean(z, sentence_break)), dim=2)
        # Self-attention is over *scales at time t*, not future time indices.
        # Base causal attention across time remains in every transformer block.
        for _ in range(2):
            q, k, v = self.scale_qkv(levels).reshape(b, t, 4, 3, self.rank).unbind(3)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.rank)
            levels = F.layer_norm(levels + scores.softmax(-1) @ v, (self.rank,))
        pooled = (levels * self.scale_logits.softmax(0).view(1, 1, 4, 1)).sum(2)
        return self.to_hidden(pooled)


class SwiGLU(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.up = nn.Linear(dim, inner * 2, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


class CausalGeometricMemory(nn.Module):
    """Two shared attention loops from token apex states to completed chunks.

    Memory keys/values come from the same geometric trajectory, not a second
    predictor. A position can read only chunks that ended *before* its current
    chunk. This creates content-addressed long-range retrieval with no labels,
    future tokens, or partially completed chunks entering a memory value.
    The zero-initialized residual gate preserves inherited logits on migration.
    """
    def __init__(self, rank, chunk):
        super().__init__()
        self.rank, self.chunk = rank, chunk
        self.query = nn.Linear(rank, rank, bias=False)
        self.key = nn.Linear(rank, rank, bias=False)
        self.value = nn.Linear(rank, rank, bias=False)
        self.feedback = nn.Linear(rank, rank, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, states):
        b, t, r = states.shape
        count = t // self.chunk
        if not count:
            return states
        # All bank entries have exactly chunk tokens. Trailing incomplete
        # chunks are absent; per-query masking excludes the *current* chunk.
        completed = states[:, :count*self.chunk].reshape(
            b, count, self.chunk, r).mean(2)
        keys, values = self.key(completed), self.value(completed)
        current_chunk = torch.arange(t, device=states.device) // self.chunk
        allowed = (torch.arange(count, device=states.device)[None, :] <
                   current_chunk[:, None])
        memory = states
        for _ in range(2):
            scores = self.query(memory) @ keys.transpose(1, 2) / math.sqrt(r)
            # Finite masking makes rows with no eligible chunks safe. Those
            # rows are explicitly zeroed after softmax.
            scores = scores.masked_fill(~allowed[None], torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1) * allowed.any(-1)[None, :, None]
            message = self.feedback(weights @ values)
            refined = F.layer_norm(memory + message, (r,))
            memory = torch.where(allowed.any(-1)[None, :, None], refined, memory)
        return states + torch.tanh(self.gate) * (memory - states)


class SparseSimplexTransfer(nn.Module):
    """Typed concept tokens and sparse up/down transport on oriented simplices.

    Addresses are exact combinadic integers. Only 8/28/56/70 active
    vertices/edges/faces/tetrahedra are materialized per token. Pair concepts
    have exact trainable storage; tetrahedra use a hashed code plus their exact
    174,792,640-way Fourier address, never a dense 174M-row weight tensor.
    Boundary matrices preserve orientation (d2@d1=d3@d2=0). Unsigned
    incidence also carries content, since a boundary of a boundary cancels.
    """
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.pair_shared = nn.Embedding(32640, rank)
        self.pair_anti = nn.Embedding(32640, rank)
        self.tetra_table = nn.Embedding(8192, rank)
        self.register_buffer('tetra_freq',
            2 * math.pi * torch.pow(2., torch.arange(rank // 2)), persistent=False)
        self.edge_to_face = nn.Linear(rank, rank, bias=False)
        self.face_to_tetra = nn.Linear(rank, rank, bias=False)
        self.tetra_to_face = nn.Linear(rank, rank, bias=False)
        self.face_to_edge = nn.Linear(rank, rank, bias=False)
        self.edge_to_point = nn.Linear(rank, rank, bias=False)
        self.apex_bridge = nn.Linear(rank, rank, bias=False)
        self.store_write = nn.Linear(rank, rank, bias=False)
        self.store_read = nn.Linear(rank, rank, bias=False)
        # Both the upward router and downward decoder see exactly the old
        # activations at gate=0, even though prior to_hidden is already trained.
        self.gate = nn.Parameter(torch.zeros(()))

        pairs = list(itertools.combinations(range(8), 2))
        faces = list(itertools.combinations(range(8), 3))
        tetrahedra = list(itertools.combinations(range(8), 4))
        self.register_buffer('edge_vertices', torch.tensor(pairs), persistent=False)
        self.register_buffer('tetra_vertices', torch.tensor(tetrahedra), persistent=False)
        self.register_buffer('face_vertices', torch.tensor(faces), persistent=False)
        pair_index = {item: n for n, item in enumerate(pairs)}
        face_index = {item: n for n, item in enumerate(faces)}
        d1 = torch.zeros(28, 8)
        d2 = torch.zeros(56, 28)
        d3 = torch.zeros(70, 56)
        for n, (a, b) in enumerate(pairs):
            d1[n, a], d1[n, b] = -1., 1.
        for n, (a, b, c) in enumerate(faces):
            for edge, sign in [((b,c), 1.), ((a,c), -1.), ((a,b), 1.)]:
                d2[n, pair_index[edge]] = sign
        for n, (a, b, c, d) in enumerate(tetrahedra):
            for face, sign in [((b,c,d),1.), ((a,c,d),-1.),
                               ((a,b,d),1.), ((a,b,c),-1.)]:
                d3[n, face_index[face]] = sign
        self.register_buffer('d1', d1, persistent=False)
        self.register_buffer('d2', d2, persistent=False)
        self.register_buffer('d3', d3, persistent=False)

    @staticmethod
    def tetra_address(a, b, c, d):
        """Exact 0..174792639 colex address for 0<=a<b<c<d<256."""
        return d*(d-1)*(d-2)*(d-3)//24 + c*(c-1)*(c-2)//6 + b*(b-1)//2 + a

    def upward(self, point, pair, triangle, tetra, addresses, sign):
        """Compose typed edge->face->tetra tokens, with explicit signed incidence."""
        n, _, rank = point.shape
        a, b = self.edge_vertices.unbind(-1)
        pa, pb = addresses[:, a], addresses[:, b]
        pair_id = pa*(511-pa)//2 + pb-pa-1
        edge = F.layer_norm(pair + self.pair_shared(pair_id) +
                            (2*sign.reshape(n,1,1)-1)*self.pair_anti(pair_id), (rank,))
        up_face = torch.einsum('fe,ner->nfr', self.d2.abs()+.5*self.d2, edge) / 3.
        face = F.layer_norm(triangle + self.edge_to_face(up_face), (rank,))
        up_tetra = torch.einsum('tf,nfr->ntr', self.d3.abs()+.5*self.d3, face) / 4.
        a, b, c, d = self.tetra_vertices.unbind(-1)
        tetra_id = self.tetra_address(addresses[:,a], addresses[:,b],
                                      addresses[:,c], addresses[:,d])
        phase = tetra_id.float().unsqueeze(-1) * (self.tetra_freq/174792640.)
        exact_code = torch.cat((phase.sin(), phase.cos()), -1)
        tet = F.layer_norm(tetra + self.face_to_tetra(up_tetra) +
                           .1*(self.tetra_table(tetra_id % 8192)+exact_code), (rank,))
        up = F.layer_norm(face.mean(1)+tet.mean(1), (rank,))
        return up, (edge,face,tet)

    def causal_store(self, upwards, batch, length):
        """O(context) within-window associative history, strictly prefix-only."""
        write = self.store_write(upwards.reshape(batch, length, self.rank))
        n = torch.arange(1, length+1, device=write.device, dtype=write.dtype)
        return self.store_read(write.cumsum(1)/n.view(1,length,1))

    def downward(self, point, parts, apex):
        """Meta->cross-level tetra bridge->faces->edges->intake points."""
        edge, face, tet = parts
        n, _, rank = point.shape
        meta = apex.reshape(n, rank)
        # The 256 meta expert layer acts as the fourth vertex of every
        # selected lower-level triangle: 56 virtual bridge tetrahedra/token.
        i, j, k = self.face_vertices.unbind(-1)
        a, b, c = point[:,i], point[:,j], point[:,k]
        u, v, w = b-a, c-a, meta[:,None,:]-a
        det = (torch.cross(u[...,:3], v[...,:3], dim=-1)*w[...,:3]).sum(-1)
        norm = (u.square().sum(-1)*v.square().sum(-1)*w.square().sum(-1)).clamp_min(1e-18).sqrt()
        bridge = (a+b+c+meta[:,None,:])/4 + torch.tanh(det/norm).unsqueeze(-1)*w/8

        tet = F.layer_norm(tet+meta[:,None,:],(rank,))
        to_face = torch.einsum('tf,ntr->nfr',self.d3.abs()+.5*self.d3,tet)/5.
        face = F.layer_norm(face + self.tetra_to_face(to_face) + self.apex_bridge(bridge), (rank,))
        to_edge = torch.einsum('fe,nfr->ner',self.d2.abs()+.5*self.d2,face)/6.
        edge = F.layer_norm(edge + self.face_to_edge(to_edge), (rank,))
        to_point = torch.einsum('ep,ner->npr',self.d1.abs()+.5*self.d1,edge)/7.
        point = F.layer_norm(point + self.edge_to_point(to_point), (rank,))
        return F.layer_norm(point.mean(1)+edge.mean(1)+face.mean(1)+tet.mean(1), (rank,))


class TriangleCurvatureMemory(nn.Module):
    """Curvature-aware complex/rotary memory over the 56 active triangle faces.

    Vertices are learned points in a radius-.75 Poincare disk (K=-1).
    Angles are between geodesic tangent directions from Mobius subtraction.
    Their Gauss--Bonnet defect is the hyperbolic triangle area, not the
    arbitrary sum of angles around a non-manifold vertex in the 8-clique.
    A signed defect rotates complex coordinate pairs before a causal write;
    the current phase conjugates the strictly-prior memory on readout.
    """
    def __init__(self, rank, vocab_size):
        super().__init__()
        if rank % 2:
            raise ValueError('Rotary triangle memory needs an even rank')
        self.rank = rank
        self.coordinates = nn.Linear(rank, 2, bias=False)
        self.angles = nn.Linear(10, rank, bias=False)
        self.write = nn.Linear(rank, rank, bias=False)
        self.read = nn.Linear(rank, rank, bias=False)
        self.aux_head = nn.Linear(rank, vocab_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.latest_logits = None

    @staticmethod
    def rotate(state, phase):
        """Exact orthogonal 2D rotations in each even/odd feature pair."""
        co, si = phase.cos().unsqueeze(-1), phase.sin().unsqueeze(-1)
        x, y = state[..., ::2], state[..., 1::2]
        return torch.stack((co*x-si*y, si*x+co*y), dim=-1).flatten(-2)

    @staticmethod
    def mobius_displacement(x, y):
        """(-x) Mobius-add y: geodesic tangent direction at x in disk."""
        xx = x.square().sum(-1, keepdim=True)
        yy = y.square().sum(-1, keepdim=True)
        dot = (x*y).sum(-1, keepdim=True)
        return ((1-2*dot+yy)*(-x)+(1-xx)*y) / (1-2*dot+xx*yy).clamp_min(1e-7)

    @staticmethod
    def corner(x, y, z):
        a = TriangleCurvatureMemory.mobius_displacement(x, y)
        b = TriangleCurvatureMemory.mobius_displacement(x, z)
        cross = a[...,0]*b[...,1] - a[...,1]*b[...,0]
        dot = (a*b).sum(-1)
        return torch.atan2(cross.abs().clamp_min(1e-9), dot)

    @classmethod
    def triangle_features(cls, disk, faces):
        """Angle sin/cos, unsigned curvature defect, and oriented holonomy."""
        i,j,k = faces.unbind(-1)
        a,b,c = disk[:,i], disk[:,j], disk[:,k]
        alpha = cls.corner(a,b,c)
        beta = cls.corner(b,c,a)
        gamma = cls.corner(c,a,b)
        # K=-1 implies area = pi - (alpha+beta+gamma) by Gauss--Bonnet.
        defect = (math.pi-alpha-beta-gamma).clamp(0., math.pi)
        ab,ac = b-a,c-a
        orientation = ab[...,0]*ac[...,1]-ab[...,1]*ac[...,0]
        sign = torch.tanh(20.*orientation)  # smooth, differentiable orientation
        phase = sign*defect
        features = torch.stack((alpha.sin(),alpha.cos(),beta.sin(),beta.cos(),
                                gamma.sin(),gamma.cos(),phase.sin(),phase.cos(),
                                defect/math.pi,sign),dim=-1)
        return features, phase, defect

    def forward(self, point, triangle, face_indices, weights, batch, length):
        raw = self.coordinates(point.float())
        radius = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
        disk = .75*torch.tanh(radius)*raw/radius.clamp_min(1e-8)
        features, phase, _ = self.triangle_features(disk, face_indices)
        i,j,k = face_indices.unbind(-1)
        face_weight = weights[:,i]*weights[:,j]*weights[:,k]
        face_weight = face_weight/face_weight.sum(1,keepdim=True).clamp_min(1e-12)
        rotated = self.rotate(triangle.float(), phase)
        face = F.layer_norm(triangle.float()+.5*rotated+self.angles(features), (self.rank,))
        current = (face*face_weight.unsqueeze(-1)).sum(1)
        current_phase = (phase*face_weight).sum(1)
        write = self.rotate(self.write(current),current_phase)
        write = write.reshape(batch,length,self.rank)
        current_phase = current_phase.reshape(batch,length)
        # Read *prior* positions only. Every document/window is a new batch row.
        previous = write.cumsum(1)-write
        divisor = torch.arange(length,device=write.device,dtype=write.dtype).clamp_min(1)
        previous = previous/divisor.view(1,length,1)
        read = self.rotate(F.layer_norm(self.read(previous),(self.rank,)),-current_phase)
        # Directly supervise the *retrieved* historical signal; otherwise an
        # auxiliary head can solve its task from the current write and never
        # make the memory read predictively useful.
        self.latest_logits = self.aux_head(read)
        return read


class SymbioticGeometry(nn.Module):
    """Causal, sparsely addressed point/pair/triangle/tetra/apex feedback.

    Conceptual addresses C(256,2/3/4) are virtual: eight selected points
    activate 28 edges, 56 triangles and 70 tetrahedra per token. The sign
    comes from the SAME mirrored experts, not a second model or optimizer.
    Prefix state resets at each document-safe input window.
    """
    POINTS = 256
    ACTIVE = 8

    def __init__(self, dim, rank, memory_chunk=0, micro_meta=False,
                 simplex_transfer=False, triangle_curvature=False, vocab_size=257):
        super().__init__()
        self.rank = rank
        self.micro_meta = micro_meta
        # Inference-only ablation switches, never stored as learned parameters.
        self.point_micro_enabled = True
        self.triangle_abstract_enabled = True
        self.meta_moe_enabled = True
        self.simplex_enabled = True
        self.simplex = SparseSimplexTransfer(rank) if simplex_transfer else None
        self.curvature_enabled = True
        self.curvature = TriangleCurvatureMemory(rank,vocab_size) if triangle_curvature else None
        self.point_query = nn.Linear(dim, rank, bias=False)
        self.expert_query = nn.Linear(dim, rank, bias=False)
        self.point_keys = nn.Parameter(torch.randn(256, rank) * .02)
        self.point_values = nn.Parameter(torch.randn(256, rank) * .02)
        self.apex_keys = nn.Parameter(torch.randn(256, rank) * .02)
        self.apex_values = nn.Parameter(torch.randn(256, rank) * .02)
        self.level_logits = nn.Parameter(torch.zeros(4))
        self.feedback_query = nn.Linear(rank, rank, bias=False)
        self.to_hidden = nn.Linear(rank, dim, bias=False)
        self.memory = CausalGeometricMemory(rank, memory_chunk) if memory_chunk else None
        if micro_meta:
            # The 256 point addresses each own an FF adapter, while sharing
            # attention projections. Only eight addressed points are executed.
            micro_inner, meta_inner = max(4, rank // 4), max(8, rank // 2)
            self.micro_qkv = nn.Linear(rank, 3 * rank, bias=False)
            self.micro_up = nn.Parameter(torch.randn(256, rank, micro_inner) * .02)
            self.micro_down = nn.Parameter(torch.randn(256, micro_inner, rank) * .02)
            # Exact triangle IDs span C(256,3)=2,763,520 without an enormous
            # parameter table; a hash embedding plus continuous exact-address
            # Fourier features provides differentiable sparse representations.
            self.triangle_table = nn.Embedding(8192, rank)
            self.register_buffer('triangle_freq',
                                 (2 * math.pi * torch.pow(2., torch.arange(rank // 2))),
                                 persistent=False)
            self.meta_query = nn.Linear(rank, rank, bias=False)
            self.meta_up = nn.Parameter(torch.randn(256, rank, meta_inner) * .02)
            self.meta_down = nn.Parameter(torch.randn(256, meta_inner, rank) * .02)
        for order in (2, 3, 4):
            self.register_buffer('choose'+str(order),
                                 torch.combinations(torch.arange(8), r=order), persistent=False)

    @staticmethod
    def pair_address(a, b):
        """Unique canonical index in 0..32639 when 0 <= a < b < 256."""
        if torch.any(a < 0) or torch.any(a >= b) or torch.any(b >= 256):
            raise ValueError('Require 0 <= a < b < 256')
        return a * (511 - a) // 2 + b - a - 1

    @staticmethod
    def triangle_address(a, b, c):
        """Colexicographic unique rank for a < b < c in 0..255."""
        return c * (c - 1) * (c - 2) // 6 + b * (b - 1) // 2 + a

    def run_microtransformers(self, point, selected):
        """Two same-token attention/FF loops on eight active point nodes."""
        for _ in range(2):
            q, k, v = self.micro_qkv(point).chunk(3, dim=-1)
            message = (q @ k.transpose(-1, -2) / math.sqrt(self.rank)).softmax(-1) @ v
            hidden = F.silu(torch.einsum('nkr,nkrh->nkh', point, self.micro_up[selected]))
            ff = torch.einsum('nkh,nkhr->nkr', hidden, self.micro_down[selected])
            point = F.layer_norm(point + message + ff, (self.rank,))
        return point

    @staticmethod
    def pool(values, weights):
        return (values * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-12)

    def forward(self, tokens, expert, signed_probability):
        batch, length, dim = tokens.shape
        rank = self.rank
        query = self.point_query(tokens.reshape(-1, dim)) + self.expert_query(expert.reshape(-1, dim))
        scores = query @ self.point_keys.T / math.sqrt(rank)
        score, selected = scores.topk(self.ACTIVE, dim=-1)
        selected, order = selected.sort(dim=-1)  # canonical signed orientation
        point_ids = selected
        weights = score.softmax(-1).gather(1, order)
        point = self.point_values[selected]
        if self.micro_meta and self.point_micro_enabled:
            point = self.run_microtransformers(point, selected)
        i, j = self.choose2.unbind(-1)
        a, b = point[:, i], point[:, j]
        symmetric, antisymmetric = (a+b)*.5, (b-a)*.5
        pair = symmetric + (2*signed_probability.reshape(-1,1,1)-1)*antisymmetric
        pair_mean = self.pool(pair, weights[:,i]*weights[:,j])

        i, j, k = self.choose3.unbind(-1)
        a, b, c = point[:,i], point[:,j], point[:,k]
        u, v = b-a, c-a
        area = u[...,0]*v[...,1] - u[...,1]*v[...,0]
        norm = (u.square().sum(-1)*v.square().sum(-1)).clamp_min(1e-12).sqrt()
        triangle = (a+b+c)/3 + torch.tanh(area/norm).unsqueeze(-1)*(v-u)/6
        if self.micro_meta and self.triangle_abstract_enabled:
            addresses = self.triangle_address(selected[:,i],selected[:,j],selected[:,k])
            phase = addresses.float().unsqueeze(-1) * (self.triangle_freq / 2763520.)
            exact_features = torch.cat((phase.sin(),phase.cos()),dim=-1)
            triangle = triangle + .1 * (self.triangle_table(addresses % 8192) + exact_features)
        tri_mean = self.pool(triangle, weights[:,i]*weights[:,j]*weights[:,k])
        if self.curvature is not None:
            self.curvature.latest_logits = None
            if self.curvature_enabled:
                curvature_read = self.curvature(point,triangle,self.choose3,weights,batch,length)
                # A zero-valued cold gate guarantees migration parity. The
                # larger slope lets an independently supervised curvature
                # memory become measurable before millions of updates.
                curvature_gate = torch.tanh(8.*self.curvature.gate)

        i, j, k, l = self.choose4.unbind(-1)
        a, b, c, d = point[:,i], point[:,j], point[:,k], point[:,l]
        u, v, w = b-a, c-a, d-a
        vol = (torch.cross(u[...,:3], v[...,:3], dim=-1)*w[...,:3]).sum(-1)
        norm = (u.square().sum(-1)*v.square().sum(-1)*w.square().sum(-1)).clamp_min(1e-18).sqrt()
        tetra = (a+b+c+d)/4 + torch.tanh(vol/norm).unsqueeze(-1)*w/8
        tetra_mean = self.pool(tetra, weights[:,i]*weights[:,j]*weights[:,k]*weights[:,l])

        if self.simplex is not None and self.simplex_enabled:
            simplex_up, simplex_parts = self.simplex.upward(
                point, pair, triangle, tetra, point_ids, signed_probability)
            simplex_store = self.simplex.causal_store(simplex_up, batch, length)
            simplex_gate = torch.tanh(self.simplex.gate)

        level = self.level_logits.softmax(0)
        local = (level[0]*self.pool(point, weights) + level[1]*pair_mean +
                 level[2]*tri_mean + level[3]*tetra_mean).reshape(batch,length,rank)
        if self.simplex is not None and self.simplex_enabled:
            local = local + simplex_gate*(simplex_up.reshape(batch,length,rank)+simplex_store)
        if self.curvature is not None and self.curvature_enabled:
            local = local + curvature_gate*curvature_read
        # Each position receives only its own or preceding concept states.
        positions = torch.arange(1,length+1,device=local.device,dtype=local.dtype)
        prefix = local.cumsum(1)/positions.view(1,length,1)
        state = F.layer_norm(local+prefix,(rank,))
        for _ in range(2):
            query = self.feedback_query(state)
            if self.micro_meta and self.meta_moe_enabled:
                # Meta routing depends explicitly on abstract triangle content.
                query = query + self.meta_query(tri_mean.reshape(batch,length,rank))
            logits = query @ self.apex_keys.T/math.sqrt(rank)
            score, selected = logits.topk(self.ACTIVE,dim=-1)
            values = self.apex_values[selected]
            if self.micro_meta and self.meta_moe_enabled:
                meta_input = (state + tri_mean.reshape(batch,length,rank)).unsqueeze(-2)
                meta_input = meta_input.expand(-1,-1,self.ACTIVE,-1)
                meta_hidden = F.silu(torch.einsum('btkr,btkrh->btkh',
                                                  meta_input, self.meta_up[selected]))
                values = values + torch.einsum('btkh,btkhr->btkr',
                                                meta_hidden, self.meta_down[selected])
            apex = (values*score.softmax(-1).unsqueeze(-1)).sum(-2)
            state = F.layer_norm(state+apex+prefix,(rank,))

        if self.memory is not None:
            state = self.memory(state)

        # Downward apex -> triangle -> mirror pair -> point closure.
        down = state.reshape(-1,rank)
        def read(values):
            score = (values*down.unsqueeze(1)).sum(-1)/math.sqrt(rank)
            return (values*score.softmax(-1).unsqueeze(-1)).sum(1)
        feedback = F.layer_norm(down+read(triangle)+read(pair)+read(point),(rank,))
        if self.simplex is not None and self.simplex_enabled:
            back = self.simplex.downward(point, simplex_parts, down)
            feedback = feedback + simplex_gate*(back+simplex_store.reshape(-1,rank))
        if self.curvature is not None and self.curvature_enabled:
            feedback = feedback + curvature_gate*curvature_read.reshape(-1,rank)
        return self.to_hidden(feedback).reshape(batch,length,dim)


class MirrorPairMoE(nn.Module):
    """An exact S+/S- mirror parameterization with sparse pair dispatch.

    For each pair i: W+_i = S_i + A_i, W-_i = S_i - A_i.
    Two mirrored experts share S_i and learn antisymmetric A_i.
    Sign mixing is learned, so the anti-symmetric information is usable.
    """
    def __init__(self, cfg):
        super().__init__()
        self.pairs, self.topk = cfg.pairs, cfg.topk_pairs
        self.strength = cfg.hyperbolic_strength
        self.dispatch_mode = cfg.dispatch_mode
        self.gather_chunk = cfg.gather_chunk
        d, f, p = cfg.dim, cfg.expert_hidden, cfg.pairs
        self.router = nn.Linear(d, p, bias=False)
        # Persistent, non-trainable capacity gate. Newly grown pairs begin
        # inactive, preserving old top-k choices, then activate gradually.
        # A buffer introduces no extra optimizer parameter on v25 resume.
        self.register_buffer('pair_bias', torch.zeros(p))
        self.sign_router = nn.Linear(d, p)
        self.geo_query = nn.Linear(d, d, bias=False)
        self.centers = nn.Parameter(torch.randn(p, d) * 0.025)
        self.in_shared = nn.Parameter(torch.randn(p, d, f) * (0.45 / math.sqrt(d)))
        self.in_anti = nn.Parameter(torch.randn(p, d, f) * (0.15 / math.sqrt(d)))
        self.out_shared = nn.Parameter(torch.randn(p, f, d) * (0.45 / math.sqrt(f)))
        self.out_anti = nn.Parameter(torch.randn(p, f, d) * (0.15 / math.sqrt(f)))
        self.normalizer = RMSNorm(d, cfg.norm_divisor or d)
        self.symbiotic = (SymbioticGeometry(d, cfg.symbiotic_rank, cfg.geometry_memory_chunk,
                                          cfg.micro_meta, cfg.simplex_transfer,
                                          cfg.triangle_curvature, cfg.vocab_size)
                         if cfg.symbiotic_rank else None)

    @staticmethod
    def to_ball(z):
        # tanh(componentwise) is NOT a valid Poincare ball projection in dimension > 1.
        norm = torch.linalg.vector_norm(z.float(), dim=-1, keepdim=True)
        return (1. - 1e-4) * torch.tanh(norm) * z.float() / norm.clamp_min(1e-8)

    @staticmethod
    def hyperbolic_distance(query, centers):
        # Poincare-ball distance for curvature -1. Keep geometry float32.
        q2 = query.square().sum(dim=-1, keepdim=True)
        c2 = centers.square().sum(dim=-1).unsqueeze(0)
        # Gram identity avoids materializing [tokens, pairs, hidden_dim].
        # It gives the same squared Euclidean distances up to FP roundoff.
        diff2 = (q2 + c2 - 2 * (query @ centers.T)).clamp_min(0)
        cosh = 1 + 2 * diff2 / ((1 - q2).clamp_min(1e-6) * (1 - c2).clamp_min(1e-6))
        return torch.acosh(cosh.clamp_min(1 + 1e-6))

    def forward(self, x):
        b, t, d = x.shape
        flat = self.normalizer(x).reshape(-1, d)
        scores = self.router(flat).float() + self.pair_bias.float()
        if self.strength != 0:
            q = self.to_ball(self.geo_query(flat))
            c = self.to_ball(self.centers)
            scores = scores - self.strength * self.hyperbolic_distance(q, c)
        top_scores, top_idx = scores.topk(self.topk, dim=-1)
        routing_weights = top_scores.softmax(dim=-1)
        sign_probs = torch.sigmoid(self.sign_router(flat))
        out = torch.zeros_like(flat)

        # Both paths compute only selected expert outputs. Chunked batched
        # gather avoids GPU synchronization from unique().tolist() and avoids
        # materializing all-expert vocabulary logits. It still needs a fused
        # production kernel before frontier-scale GPU training.
        if self.dispatch_mode == 'grouped':
            for pair in top_idx.unique().tolist():
                token_idx, slot_idx = (top_idx == pair).nonzero(as_tuple=True)
                xs = flat.index_select(0, token_idx)
                si, ai = self.in_shared[pair], self.in_anti[pair]
                so, ao = self.out_shared[pair], self.out_anti[pair]
                plus = F.silu(xs @ (si + ai)) @ (so + ao)
                minus = F.silu(xs @ (si - ai)) @ (so - ao)
                sign = sign_probs[token_idx, pair].unsqueeze(-1)
                weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
                out.index_add_(0, token_idx, weights * (sign * plus + (1 - sign) * minus))
        else:
            pieces = []
            for start in range(0, flat.shape[0], self.gather_chunk):
                stop = start + self.gather_chunk
                xx = flat[start:stop]
                selected = top_idx[start:stop]
                n = xx.shape[0]
                x_selected = xx[:, None].expand(-1, self.topk, -1).reshape(-1, d)
                pair_flat = selected.reshape(-1)
                win_s, win_a = self.in_shared[pair_flat], self.in_anti[pair_flat]
                wout_s, wout_a = self.out_shared[pair_flat], self.out_anti[pair_flat]
                plus = torch.bmm(F.silu(torch.bmm(x_selected[:, None], win_s + win_a)),
                                 wout_s + wout_a).squeeze(1)
                minus = torch.bmm(F.silu(torch.bmm(x_selected[:, None], win_s - win_a)),
                                  wout_s - wout_a).squeeze(1)
                sign = sign_probs[start:stop].gather(-1, selected).reshape(-1, 1)
                mixed = (sign * plus + (1 - sign) * minus).reshape(n, self.topk, d)
                pieces.append((mixed * routing_weights[start:stop, :, None]).sum(dim=1))
            out = torch.cat(pieces, dim=0)

        # Standard Switch-style auxiliary term (reported separately).
        # Unlike a post-hoc pair count, this is based on the actual routed tokens.
        fraction = F.one_hot(top_idx, self.pairs).float().mean(dim=(0, 1)).detach()
        mean_prob = scores.softmax(-1).mean(0)
        aux = self.pairs * (fraction * mean_prob).sum()
        out = out.reshape(b, t, d)
        if self.symbiotic is not None:
            signed = (sign_probs.gather(1,top_idx)*routing_weights).sum(-1).reshape(b,t)
            out = out + self.symbiotic(x,out,signed)
        return out, aux


class Block(nn.Module):
    def __init__(self, cfg, use_moe):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, cfg.norm_divisor or cfg.dim)
        self.att = CausalAttention(cfg.dim, cfg.heads,
                                   cfg.legacy_attention_dim, cfg.legacy_attention_heads,
                                   cfg.attention_window)
        self.n2 = RMSNorm(cfg.dim, cfg.norm_divisor or cfg.dim)
        self.ff = MirrorPairMoE(cfg) if use_moe else SwiGLU(cfg.dim, 4 * cfg.dim)
        self.use_moe = use_moe

    def forward(self, x):
        x = x + self.att(self.n1(x))
        if self.use_moe:
            y, aux = self.ff(self.n2(x))
            return x + y, aux
        return x + self.ff(self.n2(x)), x.new_zeros(())


class MirrorLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, (i + 1) % cfg.moe_every == 0) for i in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_divisor or cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # a single shared vocabulary head
        self.hierarchy = CausalTokenHierarchy(cfg.dim, cfg.hierarchy_rank) if cfg.hierarchy_rank else None
        # SAME causal rail/head for future +2 / +3, without observing future tokens.
        self.future_offsets = nn.Parameter(torch.zeros(2, cfg.dim))
        self.apply(self._init)
        # Exact old-logit preservation when opting into geometry on a checkpoint.
        for block in self.blocks:
            if block.use_moe and block.ff.symbiotic is not None:
                nn.init.zeros_(block.ff.symbiotic.to_hidden.weight)
        if self.hierarchy is not None:
            nn.init.zeros_(self.hierarchy.to_hidden.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, targets=None):
        x = self.embed(ids)
        aux = x.new_zeros(())
        for block in self.blocks:
            x, a = block(x)
            aux = aux + a
        if self.hierarchy is not None:
            x = x + self.hierarchy(x, ids)
        h = self.norm(x)
        logits = self.head(h)
        metrics = {'balancing': aux}
        if targets is not None:
            metrics['next_ce'] = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            metrics['loss'] = metrics['next_ce'] + self.cfg.balance_weight * aux
            if self.cfg.triangle_curvature and self.cfg.curvature_aux_weight:
                aux_logits = [block.ff.symbiotic.curvature.latest_logits
                              for block in self.blocks if block.use_moe and
                              block.ff.symbiotic is not None and
                              block.ff.symbiotic.curvature is not None and
                              block.ff.symbiotic.curvature.latest_logits is not None]
                if aux_logits:
                    metrics['curvature_aux_ce'] = torch.stack([
                        F.cross_entropy(p.flatten(0,1),targets.flatten())
                        for p in aux_logits]).mean()
                    metrics['loss'] = metrics['loss'] + (
                        self.cfg.curvature_aux_weight*metrics['curvature_aux_ce'])
        return logits, h, metrics

    def future_logits(self, h, disable_temporal=False):
        """Predict t+2 and t+3 from the SAME causal NOW representation.

        offset: original, constant horizon vectors (v24 compatibility).
        rollout: reuse the last backbone FF/MoE transformation twice as a
        single shared learned state transition. Adds NO new weight branch and
        never reads future target tokens. This is a research ablation rather
        than a proven long-range forecaster.
        """
        if disable_temporal:
            return self.head(h), self.head(h)
        if self.cfg.future_mode == 'offset':
            return (self.head(h + self.future_offsets[0]),
                    self.head(h + self.future_offsets[1]))
        block = self.blocks[self.cfg.future_block_index]
        state = h
        outputs = []
        for _ in range(2):
            update = block.ff(block.n2(state))
            if block.use_moe:
                update = update[0]
            state = self.norm(state + update)
            outputs.append(self.head(state))
        return outputs[0], outputs[1]

    def loss_with_future(self, ids):
        # ids has shape [B, context+3]. No label bytes enter the model input.
        x = ids[:, :-3]
        logits, h, m = self(x, ids[:, 1:-2])
        loss = m['loss']
        if self.cfg.future2_weight or self.cfg.future3_weight:
            p2, p3 = self.future_logits(h)
            if self.cfg.future2_weight:
                m['future2_ce'] = F.cross_entropy(p2.flatten(0, 1), ids[:, 2:-1].flatten())
                loss = loss + self.cfg.future2_weight * m['future2_ce']
            if self.cfg.future3_weight:
                m['future3_ce'] = F.cross_entropy(p3.flatten(0, 1), ids[:, 3:].flatten())
                loss = loss + self.cfg.future3_weight * m['future3_ce']
        m['loss'] = loss
        return m


class TokenCodec:
    def __init__(self, model_file=None):
        self.model_file = model_file
        self.sp = None
        if model_file:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
            self.vocab_size = self.sp.vocab_size()
            self.eos = self.sp.eos_id()
        else:
            self.vocab_size = 257
            self.eos = 256

    def encode(self, text):
        return self.sp.encode(text, out_type=int) if self.sp else list(text.encode('utf-8'))

    def decode(self, ids):
        if self.sp:
            return self.sp.decode([int(x) for x in ids])
        return bytes(int(i) for i in ids if 0 <= i < 256).decode('utf-8', errors='replace')


def token_dtype(vocab_size):
    # Fits 257-byte+EOS and SentencePiece vocabularies up to 65,535 ids.
    return '<u2' if vocab_size <= 65536 else '<u4'


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(args):
    if Path(args.train_text).resolve() == Path(args.valid_text).resolve():
        raise ValueError('Train and validation must be different files')
    train_hash, valid_hash = file_hash(args.train_text), file_hash(args.valid_text)
    if train_hash == valid_hash:
        raise ValueError('Train and validation are byte-identical')
    codec = TokenCodec(args.tokenizer_model)
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(codec.vocab_size)
    def one(source, target):
        count = 0
        doc_lengths = []
        with open(source, encoding='utf-8') as inp, open(dest / (target + '.bin'), 'wb') as out:
            for line in inp:
                ids = codec.encode(line)
                if codec.eos >= 0:
                    ids.append(codec.eos)
                arr = np.asarray(ids, dtype=dtype)
                arr.tofile(out)
                count += len(ids)
                if ids:
                    doc_lengths.append(len(ids))
        return count, doc_lengths
    ntrain, train_lengths = one(args.train_text, 'train')
    nvalid, valid_lengths = one(args.valid_text, 'valid')
    meta = dict(vocab_size=codec.vocab_size, tokenizer_model=str(Path(args.tokenizer_model).resolve()) if args.tokenizer_model else None,
                tokenizer_sha256=file_hash(args.tokenizer_model) if args.tokenizer_model else None,
                train_sha256=train_hash, valid_sha256=valid_hash, train_tokens=ntrain, valid_tokens=nvalid,
                unit='sentencepiece_tokens' if args.tokenizer_model else 'UTF8_bytes_plus_eos',
                token_dtype=dtype, train_doc_lengths=train_lengths, valid_doc_lengths=valid_lengths,
                eos_id=codec.eos,
                bin_train_sha256=file_hash(dest / 'train.bin'),
                bin_valid_sha256=file_hash(dest / 'valid.bin'))
    (dest / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))



def prepare_books(args):
    """Split a folder of book .txt files BEFORE tokenization.

    File content hashes deduplicate exact copies (even renamed files).
    Validation uses entire unseen books, not random windows of train books.
    For large corpora this is a streaming *per-book* importer; each book is
    loaded individually. EOS is written once per book, NOT once per line.
    """
    root = Path(args.source_dir).resolve()
    if not root.is_dir():
        raise ValueError(f'Corpus directory not found: {root}')
    test_fraction = getattr(args, 'test_fraction', 0.0)
    if not (0.0 < args.valid_fraction < 0.5):
        raise ValueError('--valid-fraction must be strictly between 0 and 0.5')
    if test_fraction < 0.0 or test_fraction >= 0.5 or args.valid_fraction + test_fraction >= 1.0:
        raise ValueError('Invalid --test-fraction or combined held-out fraction')
    exclude_ids_path = getattr(args, 'exclude_gutenberg_ids', None)
    excluded_ids = set()
    if exclude_ids_path:
        ids = json.loads(Path(exclude_ids_path).read_text())
        if not isinstance(ids, list) or not all(isinstance(x, int) and x > 0 for x in ids):
            raise ValueError('--exclude-gutenberg-ids must be a JSON list of positive integer IDs')
        excluded_ids = set(ids)
    candidates = sorted(p for p in root.rglob('*.txt') if p.is_file())
    records = []
    seen = {}
    duplicate_files = []
    excluded_files = []
    for path in candidates:
        size = path.stat().st_size
        if size < args.min_bytes or size > args.max_doc_bytes:
            continue
        digest = file_hash(path)
        rel = str(path.relative_to(root))
        name_match = re.match(r'^(\d+)(?:--|\.txt$)', path.name)
        if name_match and int(name_match.group(1)) in excluded_ids:
            excluded_files.append(rel)
            continue
        if digest in seen:
            duplicate_files.append({'duplicate': rel, 'original': seen[digest]})
            continue
        seen[digest] = rel
        records.append({'path': rel, 'sha256': digest, 'bytes': size})
    if len(records) < (3 if test_fraction else 2):
        raise ValueError('Too few different readable .txt books after filters')
    # Hash ordering independent of file enumeration and random library versions.
    records.sort(key=lambda r: hashlib.sha256(
        (str(args.seed) + ':' + r['sha256']).encode()).hexdigest())
    nvalid = max(1, round(len(records) * args.valid_fraction))
    ntest = max(1, round(len(records) * test_fraction)) if test_fraction else 0
    if nvalid + ntest >= len(records):
        raise ValueError('Held-out splits leave no training books')
    test = records[:ntest]
    valid = records[ntest:ntest + nvalid]
    train = records[ntest + nvalid:]
    codec = TokenCodec(args.tokenizer_model)
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(codec.vocab_size)
    def write_split(name, files):
        n = 0
        lengths = []
        with (dest / (name + '.bin')).open('wb') as output:
            for item in files:
                source = root / item['path']
                # Full-book encode preserves cross-line context, unlike line-by-line EOS.
                content = source.read_text(encoding='utf-8', errors='replace')
                ids = codec.encode(content)
                if codec.eos >= 0:
                    ids.append(codec.eos)
                np.asarray(ids, dtype=dtype).tofile(output)
                n += len(ids)
                lengths.append(len(ids))
        return n, lengths
    ntrain, train_lengths = write_split('train', train)
    nvalid_tokens, valid_lengths = write_split('valid', valid)
    ntest_tokens, test_lengths = write_split('test', test) if test else (0, [])
    manifest = {'split_unit': 'whole_book', 'seed': args.seed,
                'train_books': train, 'valid_books': valid,
                'test_books': test,
                'exact_duplicate_files_ignored': duplicate_files,
                'excluded_prior_gutenberg_ids': sorted(excluded_ids),
                'excluded_files': excluded_files}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    (dest / 'split_manifest.json').write_bytes(manifest_bytes)
    meta = dict(vocab_size=codec.vocab_size,
                tokenizer_model=str(Path(args.tokenizer_model).resolve()) if args.tokenizer_model else None,
                tokenizer_sha256=file_hash(args.tokenizer_model) if args.tokenizer_model else None,
                train_sha256=file_hash(dest / 'train.bin'),
                valid_sha256=file_hash(dest / 'valid.bin'),
                train_tokens=ntrain, valid_tokens=nvalid_tokens,
                train_books=len(train), valid_books=len(valid),
                unique_books=len(records), duplicates_ignored=len(duplicate_files),
                split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                unit='sentencepiece_tokens' if args.tokenizer_model else 'UTF8_bytes_plus_eos',
                split_unit='whole_book', token_dtype=dtype,
                train_doc_lengths=train_lengths, valid_doc_lengths=valid_lengths, eos_id=codec.eos)
    if test:
        meta.update(test_sha256=file_hash(dest / 'test.bin'),
                    test_tokens=ntest_tokens, test_books=len(test),
                    test_doc_lengths=test_lengths)
    (dest / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2))

class DocumentWindowSampler:
    """Uniform over *valid positions* inside documents, never across EOS.

    The offsets come from the import manifest, not inferred from arbitrary
    concatenation. For old v25 archives, EOS markers can reconstruct them.
    """
    def __init__(self, mem, context, lengths=None):
        self.mem = mem
        self.length = context + 3
        if lengths is None:
            lengths = [len(mem)]
        lengths = np.asarray(lengths, dtype=np.int64)
        if lengths.ndim != 1 or (lengths <= 0).any() or sum(lengths) != len(mem):
            raise ValueError('Document boundaries do not match token file')
        starts = np.cumsum(np.r_[0, lengths[:-1]])
        eligible = np.maximum(lengths - self.length + 1, 0)
        keep = eligible > 0
        self.doc_starts = starts[keep]
        self.doc_lengths = lengths[keep]
        self.end_cdf = np.cumsum(eligible[keep])
        self.available_positions = int(self.end_cdf[-1]) if len(self.end_cdf) else 0
        self.total_documents = len(lengths)
        self.eligible_documents = int(keep.sum())
        if self.available_positions == 0:
            raise ValueError(f'No document long enough for {self.length} token windows')

    def windows(self, n, rng, device):
        draws = rng.integers(0, self.available_positions, size=n)
        di = np.searchsorted(self.end_cdf, draws, side='right')
        base = np.r_[0, self.end_cdf[:-1]][di]
        offsets = self.doc_starts[di] + draws - base
        ary = np.stack([self.mem[int(x):int(x)+self.length] for x in offsets]).astype(np.int64)
        return torch.from_numpy(ary).to(device)

    def exhaustive(self, batch, device, stride):
        """Non-overlapping windows by default; weighted by actual token count."""
        if stride <= 0:
            raise ValueError('stride must be positive')
        pending = []
        for start, doc_len in zip(self.doc_starts, self.doc_lengths):
            for offset in range(int(start), int(start + doc_len - self.length + 1), stride):
                pending.append(self.mem[offset:offset+self.length])
                if len(pending) == batch:
                    yield torch.from_numpy(np.stack(pending).astype(np.int64)).to(device)
                    pending.clear()
        if pending:
            yield torch.from_numpy(np.stack(pending).astype(np.int64)).to(device)


def legacy_document_lengths(mem, meta, split):
    """Reconstruct book boundaries from v25 EOS without changing its data hash."""
    lengths = meta.get(split + '_doc_lengths')
    if lengths is not None:
        return lengths
    if meta.get('split_unit') != 'whole_book':
        return [len(mem)]
    eos = meta.get('eos_id')
    if eos is None:
        eos = TokenCodec(meta.get('tokenizer_model')).eos
    # For old imports, EOS is appended once per book.
    if eos is None or eos < 0:
        raise ValueError('Cannot reconstruct book boundaries without EOS')
    ends = np.flatnonzero(np.asarray(mem) == eos) + 1
    if len(ends) != meta[split + '_books'] or not len(ends) or ends[-1] != len(mem):
        raise ValueError('Legacy EOS boundary reconstruction failed; refusing cross-book sampling')
    return np.diff(np.r_[0, ends]).tolist()


def sample_batch(mem, batch, context, rng, device):
    length = context + 3
    if len(mem) <= length:
        raise ValueError(f'Need > {length} tokens, got {len(mem)}')
    idx = rng.integers(0, len(mem) - length + 1, size=batch)
    arr = np.stack([mem[int(i):int(i) + length] for i in idx]).astype(np.int64)
    return torch.from_numpy(arr).to(device)


def load_meta(directory):
    return json.loads((Path(directory) / 'meta.json').read_text())


def checkpoint_path(run_dir):
    return Path(run_dir) / 'latest.pt'


def save_checkpoint(path, model, optimizer, step, seen_tokens, meta, rng, cfg,
                    growth=None, growth_history=None):
    """Atomic single-checkpoint replacement; never fork optimizer histories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                   seen_tokens=seen_tokens, config=asdict(cfg), data=meta,
                   rng_numpy=rng.bit_generator.state,
                   rng_torch=torch.get_rng_state(), rng_python=random.getstate(),
                   growth=growth, growth_history=growth_history or [])
    if torch.cuda.is_available():
        payload['rng_cuda'] = torch.cuda.get_rng_state_all()
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_v25_compatible(model, state):
    """Allow only the v26 pair_bias buffer absent in prior v25 checkpoints."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not x.endswith('.pair_bias') for x in missing):
        raise ValueError(f'Unsupported checkpoint schema: missing={missing}, unexpected={unexpected}')


def restore_checkpoint(path, model, optimizer=None, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    load_v25_compatible(model, checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def morph_tensor(name, target, source, old_cfg, cfg, preserve_new_pairs=False):
    """Copy a parameter/moment into the equivalent old-coordinate subspace.

    QKV and SwiGLU split layouts are NOT prefix arrays after width changes;
    they must be copied segment by segment. All other tensors are embedded
    in their leading-coordinate subspaces.
    """
    if target.shape == source.shape:
        target.copy_(source)
        return
    if target.ndim != source.ndim or any(a > b for a, b in zip(source.shape, target.shape)):
        raise ValueError('Cannot morph tensor ' + name)
    d0, d1 = old_cfg.dim, cfg.dim
    p0, p1 = old_cfg.pairs, cfg.pairs
    is_old_moe = '.ff.' in name and name.startswith('blocks.') and \
        int(name.split('.')[1]) < old_cfg.layers and \
        (int(name.split('.')[1]) + 1) % old_cfg.moe_every == 0
    is_pair_array = is_old_moe and any(name.endswith(sfx) for sfx in (
        'router.weight', 'sign_router.weight', 'sign_router.bias',
        'centers', 'in_shared', 'in_anti', 'out_shared', 'out_anti', 'pair_bias'))
    if preserve_new_pairs and is_pair_array and p1 > p0:
        target[:p0].zero_()  # keep new pair rows randomly initialized and masked
    else:
        target.zero_()
    if name.endswith('.att.qkv.weight'):
        for h in range(3):
            target[h*d1:h*d1+d0, :d0].copy_(source[h*d0:(h+1)*d0, :d0])
    elif name.endswith('.ff.up.weight'):
        oldinner, newinner = 4*d0, 4*d1
        target[:oldinner, :d0].copy_(source[:oldinner, :d0])
        target[newinner:newinner+oldinner, :d0].copy_(source[oldinner:2*oldinner, :d0])
    else:
        slices = tuple(slice(0, n) for n in source.shape)
        target[slices].copy_(source)
    if preserve_new_pairs and name.endswith('.scale') and d1 > d0:
        target[d0:].fill_(1.)


def widen_config(cfg, target_dim):
    """Grow hidden width without changing any inherited RoPE frequencies.

    Integral multiples retain uniform head geometry. Other increases create
    new heterogeneous heads and keep all trained heads exactly intact. Later
    increases extend the newer head group without changing its geometry.
    """
    if target_dim < cfg.dim:
        raise ValueError('Width cannot shrink')
    if target_dim == cfg.dim:
        return
    initial_dim, initial_heads = cfg.dim, cfg.heads
    if cfg.legacy_attention_dim:
        newer_dim = cfg.dim - cfg.legacy_attention_dim
        newer_heads = cfg.heads - cfg.legacy_attention_heads
        head_dim = newer_dim // newer_heads
        if (target_dim - cfg.dim) % head_dim:
            raise ValueError('Mixed-head growth must add whole inherited new-group heads')
        cfg.heads += (target_dim - cfg.dim) // head_dim
    elif target_dim % initial_dim == 0:
        cfg.heads *= target_dim // initial_dim
    else:
        extra = target_dim - initial_dim
        new_head_dim = next((h for h in range(min(extra, 16), 1, -1)
                             if h % 2 == 0 and extra % h == 0), None)
        if new_head_dim is None:
            raise ValueError('Additional width needs an even-sized attention head group')
        cfg.legacy_attention_dim = initial_dim
        cfg.legacy_attention_heads = initial_heads
        cfg.heads += extra // new_head_dim
    cfg.norm_divisor = cfg.norm_divisor or initial_dim
    cfg.dim = target_dim
    cfg.check()


def expand_architecture(model, optimizer, saved, previous_cfg, bias=-12.0, widen_noise=1e-4):
    """Function-preserving pair AND/OR depth growth with Adam-state migration.

    Old parameters are copied by name, never by guessed optimizer slot number.
    Added blocks begin as exact identity residuals. Newly added expert pairs
    are initially inactive, so top-k selection remains unchanged.
    """
    old_pairs, old_layers = previous_cfg.pairs, previous_cfg.layers
    new_pairs, new_layers = model.cfg.pairs, model.cfg.layers
    if new_pairs < old_pairs or new_layers < old_layers or model.cfg.dim < previous_cfg.dim:
        raise ValueError('Only non-shrinking expansions are supported')
    if (new_pairs == old_pairs and new_layers == old_layers
            and model.cfg.dim == previous_cfg.dim
            and model.cfg.expert_hidden == previous_cfg.expert_hidden
            and model.cfg.symbiotic_rank == previous_cfg.symbiotic_rank
            and model.cfg.hierarchy_rank == previous_cfg.hierarchy_rank
            and model.cfg.geometry_memory_chunk == previous_cfg.geometry_memory_chunk
            and model.cfg.micro_meta == previous_cfg.micro_meta
            and model.cfg.simplex_transfer == previous_cfg.simplex_transfer
            and model.cfg.triangle_curvature == previous_cfg.triangle_curvature):
        raise ValueError('No capacity expansion requested')
    if previous_cfg.symbiotic_rank and model.cfg.symbiotic_rank != previous_cfg.symbiotic_rank:
        raise ValueError('Cannot reshape an already trained Diamond')
    if (previous_cfg.geometry_memory_chunk and
            model.cfg.geometry_memory_chunk != previous_cfg.geometry_memory_chunk):
        raise ValueError('Cannot reshape an already trained geometric memory')
    if previous_cfg.hierarchy_rank and model.cfg.hierarchy_rank != previous_cfg.hierarchy_rank:
        raise ValueError('Cannot reshape a trained token hierarchy')
    if previous_cfg.micro_meta and not model.cfg.micro_meta:
        raise ValueError('Cannot discard trained microtransformer or meta-MoE weights')
    if previous_cfg.simplex_transfer and not model.cfg.simplex_transfer:
        raise ValueError('Cannot discard trained simplex transfer and concept memory')
    if previous_cfg.triangle_curvature and not model.cfg.triangle_curvature:
        raise ValueError('Cannot discard trained triangle curvature memory')
    if model.cfg.moe_every != previous_cfg.moe_every:
        raise ValueError('Cannot change MoE placement during expansion')
    if model.cfg.expert_hidden < previous_cfg.expert_hidden:
        raise ValueError('Cannot shrink expert hidden capacity')
    if model.cfg.dim != previous_cfg.dim:
        continuing_mixed = (previous_cfg.legacy_attention_dim and
                            model.cfg.legacy_attention_dim == previous_cfg.legacy_attention_dim and
                            model.cfg.legacy_attention_heads == previous_cfg.legacy_attention_heads and
                            (model.cfg.dim - model.cfg.legacy_attention_dim) *
                            (previous_cfg.heads - previous_cfg.legacy_attention_heads) ==
                            (previous_cfg.dim - previous_cfg.legacy_attention_dim) *
                            (model.cfg.heads - model.cfg.legacy_attention_heads))
        uniform = (model.cfg.legacy_attention_dim == 0 and
                   model.cfg.dim % previous_cfg.dim == 0 and
                   model.cfg.heads * previous_cfg.dim == previous_cfg.heads * model.cfg.dim)
        mixed = (model.cfg.legacy_attention_dim == previous_cfg.dim and
                 model.cfg.legacy_attention_heads == previous_cfg.heads)
        if not (uniform or mixed or continuing_mixed):
            raise ValueError('Width growth must preserve inherited attention heads')
        norm0 = previous_cfg.norm_divisor or previous_cfg.dim
        if model.cfg.norm_divisor != norm0:
            raise ValueError('Must preserve original RMSNorm divisor on width growth')
    old_model = MirrorLM(previous_cfg)
    old_param_names = list(dict(old_model.named_parameters()))
    new_param_names = list(dict(model.named_parameters()))
    old_lookup = set(old_param_names)
    new_state = model.state_dict()
    with torch.no_grad():
        for index in range(old_layers, new_layers):
            block = model.blocks[index]
            # Zero both residual branches, rather than removing or freezing them.
            block.att.out.weight.zero_()
            if block.use_moe:
                block.ff.out_shared.zero_()
                block.ff.out_anti.zero_()
            else:
                block.ff.down.weight.zero_()
        new_state = model.state_dict()
        for name, target in new_state.items():
            if name not in saved['model']:
                if name.endswith('.pair_bias') or any(name.startswith(f'blocks.{i}.')
                                                    for i in range(old_layers, new_layers)) or (
                    not previous_cfg.symbiotic_rank and '.ff.symbiotic.' in name) or (
                    not previous_cfg.micro_meta and '.ff.symbiotic.' in name and any(
                        part in name for part in ('.micro_', '.meta_', '.triangle_table.'))) or (
                    not previous_cfg.simplex_transfer and '.ff.symbiotic.simplex.' in name) or (
                    not previous_cfg.triangle_curvature and '.ff.symbiotic.curvature.' in name) or (
                    not previous_cfg.geometry_memory_chunk and '.ff.symbiotic.memory.' in name) or (
                    not previous_cfg.hierarchy_rank and name.startswith('hierarchy.')):
                    continue
                raise ValueError('Unknown new state entry: ' + name)
            source = saved['model'][name]
            morph_tensor(name, target, source, previous_cfg, model.cfg, preserve_new_pairs=True)
        model.load_state_dict(new_state)
        if model.cfg.dim > previous_cfg.dim:
            # Tiny symmetry-breaking seed: makes new coordinates trainable.
            # Setting widen_noise=0 leaves logits exactly unchanged but
            # zero new coordinates can remain inactive indefinitely.
            with torch.no_grad():
                model.embed.weight[:, previous_cfg.dim:].normal_(0., widen_noise)
        if model.cfg.expert_hidden > previous_cfg.expert_hidden:
            # Seed *incoming* new expert features, while keeping their
            # outgoing projections exactly zero at the expansion boundary.
            with torch.no_grad():
                for b in model.blocks[:old_layers]:
                    if b.use_moe:
                        b.ff.in_shared[:old_pairs, :previous_cfg.dim,
                                       previous_cfg.expert_hidden:].normal_(0., widen_noise)
        if new_pairs > old_pairs:
            for b in model.blocks[:old_layers]:
                if b.use_moe:
                    b.ff.pair_bias[old_pairs:].fill_(bias)
            for b in model.blocks[old_layers:]:
                if b.use_moe:
                    b.ff.pair_bias[old_pairs:].fill_(bias)

    # Rebuild optimizer ID map by unique parameter name, preserving all
    # old moments, scalar steps and hyperparameters. New blocks have empty
    # Adam states until they receive their first real gradient.
    old_opt = saved['optimizer']
    if len(old_opt['param_groups']) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError('Growth currently supports a single AdamW parameter group')
    old_ids = old_opt['param_groups'][0]['params']
    if len(old_param_names) != len(old_ids):
        raise ValueError('Old optimizer param IDs do not match named parameters')
    old_id_by_name = dict(zip(old_param_names, old_ids))
    new_opt = optimizer.state_dict()
    new_ids = new_opt['param_groups'][0]['params']
    if len(new_ids) != len(new_param_names):
        raise ValueError('New optimizer params mismatch')
    for setting, val in old_opt['param_groups'][0].items():
        if setting != 'params':
            new_opt['param_groups'][0][setting] = copy.deepcopy(val)
    new_opt['state'] = {}
    new_named = dict(model.named_parameters())
    old_named = dict(old_model.named_parameters())
    for name, nid in zip(new_param_names, new_ids):
        if name not in old_lookup:
            continue
        oid = old_id_by_name[name]
        if oid not in old_opt['state']:
            continue
        state = copy.deepcopy(old_opt['state'][oid])
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            newshape = new_named[name].shape
            if value.shape == newshape:
                continue
            morphed = value.new_zeros(newshape)
            morph_tensor(name, morphed, value, previous_cfg, model.cfg)
            state[key] = morphed
        new_opt['state'][nid] = state
    optimizer.load_state_dict(new_opt)


def expand_pair_capacity(model, optimizer, saved, previous_cfg, bias=-12.0):
    """Backward-compatible alias used by early v26 capacity tests."""
    if model.cfg.layers != previous_cfg.layers:
        raise ValueError('Use expand_architecture for depth growth')
    return expand_architecture(model, optimizer, saved, previous_cfg, bias)


def update_growth_bias(model, growth, step):
    if growth is None:
        return
    elapsed = max(0, step - growth['start_step'])
    ramp = max(1, growth['warmup_steps'])
    bias = growth['initial_bias'] * max(0., 1. - elapsed / ramp)
    with torch.no_grad():
        for block in model.blocks:
            if block.use_moe:
                block.ff.pair_bias[growth['from_pairs']:growth['to_pairs']].fill_(bias)


def configure(args, vocab_size):
    return Config(vocab_size=vocab_size, dim=args.dim, layers=args.layers,
                  heads=args.heads, pairs=args.pairs, topk_pairs=args.topk,
                  expert_hidden=args.expert_hidden, context=args.context,
                  moe_every=args.moe_every, hyperbolic_strength=args.geo,
                  balance_weight=args.balance, symbiotic_rank=args.symbiotic_rank or 0,
                  hierarchy_rank=getattr(args, 'hierarchy_rank', None) or 0,
                  attention_window=getattr(args, 'attention_window', None) or 0,
                  geometry_memory_chunk=getattr(args, 'geometry_memory_chunk', None) or 0,
                  micro_meta=getattr(args,'micro_meta',False),
                  simplex_transfer=getattr(args,'simplex_transfer',False),
                  triangle_curvature=getattr(args,'triangle_curvature',False),
                  future2_weight=0.1 if args.future2 is None else args.future2,
                  future3_weight=0.05 if args.future3 is None else args.future3, future_mode=args.future_mode or 'offset')


def parse_context_mix(spec, maximum):
    """Parse ``context:weight`` entries without changing model or tokenizer."""
    entries = []
    for entry in spec.split(','):
        try:
            length, weight = entry.strip().split(':')
            length, weight = int(length), float(weight)
        except (ValueError, TypeError) as exc:
            raise ValueError('Expected --context-mix 256:3,512:2,...') from exc
        if not 4 <= length <= maximum or not math.isfinite(weight) or weight <= 0:
            raise ValueError('Context lengths must be within model maximum; weights finite and > 0')
        if length in [row[0] for row in entries]:
            raise ValueError('Duplicate context in --context-mix')
        entries.append((length, weight))
    if not entries:
        raise ValueError('Empty context mixture')
    total = sum(weight for _, weight in entries)
    return ([length for length, _ in entries],
            np.array([weight / total for _, weight in entries], dtype=np.float64))


def train(args):
    meta = load_meta(args.data_dir)
    if 'split_manifest_sha256' in meta:
        expected = meta['split_manifest_sha256']
        actual = file_hash(Path(args.data_dir) / 'split_manifest.json')
        if expected != actual:
            raise ValueError('Book split manifest changed; refusing to train')
    trainmem = np.memmap(Path(args.data_dir) / 'train.bin', dtype=meta.get('token_dtype', '<u4'), mode='r')
    valmem = np.memmap(Path(args.data_dir) / 'valid.bin', dtype=meta.get('token_dtype', '<u4'), mode='r')
    for split in ['train', 'valid']:
        expected = meta.get('bin_' + split + '_sha256')
        if expected is None and meta.get('split_unit') == 'whole_book':
            expected = meta[split + '_sha256']
        if expected and file_hash(Path(args.data_dir) / (split + '.bin')) != expected:
            raise ValueError(f'{split} data changed; refusing to train')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cpu' and getattr(args, 'cpu_threads', 0):
        torch.set_num_threads(args.cpu_threads)
    path = checkpoint_path(args.run_dir)
    saved = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
    changing_data = bool(saved and saved['data'] != meta)
    if changing_data:
        if not args.adopt_data:
            raise ValueError('Data changed: add --adopt-data for an explicit, recorded corpus transition')
        old_meta = saved['data']
        for field in ('vocab_size', 'tokenizer_sha256', 'unit'):
            if old_meta.get(field) != meta.get(field):
                raise ValueError(f'Cannot retain weights across a tokenizer change: {field}')
        # token_dtype describes on-disk packing, not the vocabulary mapping.
        # The R27 checkpoint predates this manifest key, and Gutenberg uses
        # compact uint16 for the same 257 byte+EOS symbols.
        if meta.get('split_unit') != 'whole_book':
            raise ValueError('Adopted corpora must have independently split whole books')
    if saved:
        cfg = Config(**saved['config'])
        if args.future_mode is not None and args.future_mode != cfg.future_mode:
            print(f'SWITCHED temporal objective: {cfg.future_mode} -> {args.future_mode}; same model and optimizer weights')
            cfg.future_mode = args.future_mode
        if args.future2 is not None:
            cfg.future2_weight = args.future2
        if args.future3 is not None:
            cfg.future3_weight = args.future3
    else:
        cfg = configure(args, meta['vocab_size'])
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    old_cfg = copy.deepcopy(cfg)
    enable_symbiotic = bool(saved and args.symbiotic_rank is not None and args.symbiotic_rank != cfg.symbiotic_rank)
    if enable_symbiotic:
        if cfg.symbiotic_rank or args.symbiotic_rank == 0:
            raise ValueError('Diamond can only be enabled once; existing learned geometry cannot be reset')
        cfg.symbiotic_rank = args.symbiotic_rank
    requested_hierarchy_rank = getattr(args, 'hierarchy_rank', None)
    enable_hierarchy = bool(saved and requested_hierarchy_rank is not None and
                            requested_hierarchy_rank != cfg.hierarchy_rank)
    if enable_hierarchy:
        if cfg.hierarchy_rank or requested_hierarchy_rank == 0:
            raise ValueError('Token hierarchy can only be enabled once')
        cfg.hierarchy_rank = requested_hierarchy_rank
    new_memory_chunk = getattr(args, 'geometry_memory_chunk', None)
    enable_geometry_memory = bool(saved and new_memory_chunk is not None and
                                  new_memory_chunk != cfg.geometry_memory_chunk)
    if enable_geometry_memory:
        if cfg.geometry_memory_chunk or new_memory_chunk == 0:
            raise ValueError('Already trained geometric memory cannot be resized or disabled')
        cfg.geometry_memory_chunk = new_memory_chunk
    enable_micro_meta = bool(saved and getattr(args, 'micro_meta', False) and not cfg.micro_meta)
    if enable_micro_meta:
        cfg.micro_meta = True
    enable_simplex = bool(saved and getattr(args, 'simplex_transfer', False) and not cfg.simplex_transfer)
    if enable_simplex:
        cfg.simplex_transfer = True
    enable_curvature = bool(saved and getattr(args,'triangle_curvature',False) and
                            not cfg.triangle_curvature)
    if enable_curvature:
        cfg.triangle_curvature = True
    history = list(saved.get('growth_history', [])) if saved else []
    new_attention_window = getattr(args, 'attention_window', None)
    if new_attention_window is not None and new_attention_window != cfg.attention_window:
        if saved and old_cfg.attention_window:
            raise ValueError('Cannot change a trained attention window')
        if saved and new_attention_window and new_attention_window < old_cfg.context:
            raise ValueError('Attention window must cover inherited context for logit-preserving migration')
        history.append({'kind':'local_causal_attention','step':saved['step'] if saved else 0,
                        'from':cfg.attention_window,'to':new_attention_window})
        cfg.attention_window = new_attention_window
    growth = saved.get('growth') if saved else None
    if enable_symbiotic:
        history.append({'kind':'symbiotic_diamond','step':saved['step'],
                        'rank':cfg.symbiotic_rank,'points':256,'active_points':8,
                        'active_pairs':28,'active_triangles':56,'active_tetrahedra':70,
                        'apex_concepts':256,'feedback_passes':2,
                        'output_initialized_to_zero':True})
    if enable_hierarchy:
        history.append({'kind':'causal_token_hierarchy','step':saved['step'],
                        'rank':cfg.hierarchy_rank,'scales':['byte','byte_pair_proxy','word','sentence'],
                        'cross_scale_attention_passes':2,'output_initialized_to_zero':True})
    if enable_geometry_memory:
        history.append({'kind':'causal_geometric_memory','step':saved['step'],
                        'chunk_size':cfg.geometry_memory_chunk,
                        'apex_memory_attention_passes':2,'gate_initialized_to_zero':True})
    if enable_micro_meta:
        history.append({'kind':'sparse_point_microtransformers_triangle_meta_moe',
                        'step':saved['step'],'point_nodes':256,'active_points':8,
                        'implicit_triangle_addresses':2763520,'active_triangles':56,
                        'meta_experts':256,'active_meta_experts':8})
    if enable_simplex:
        history.append({'kind':'causal_sparse_simplex_concept_transfer',
                        'step':saved['step'],'pair_tokens':32640,
                        'triangle_tokens':2763520,'tetra_tokens':174792640,
                        'active_per_prediction':[8,28,56,70],
                        'meta_anchored_tetra_per_token':56,
                        'direct_pair_storage':True,'hashed_tetra_storage':True,
                        'causal_prefix_state':True,'gate_init':0.})
    if enable_curvature:
        history.append({'kind':'triangle_hyperbolic_curvature_phase_memory',
                        'step':saved['step'],'curvature':-1,'active_triangles':56,
                        'memory':'phase_aligned_strictly_prior_causal',
                        'auxiliary_next_byte_weight':cfg.curvature_aux_weight,
                        'gate_init':0.,'preserves_logits_at_migration':True})
    if args.dispatch_mode is not None and args.dispatch_mode != cfg.dispatch_mode:
        history.append({'kind':'dispatch', 'step':saved['step'] if saved else 0,
                        'from':cfg.dispatch_mode, 'to':args.dispatch_mode})
        cfg.dispatch_mode = args.dispatch_mode
    if args.gather_chunk is not None:
        cfg.gather_chunk = args.gather_chunk
    if getattr(args, 'cpu_pair_open_steps', None) is not None:
        if not saved or not growth or not (growth['from_pairs'] < growth['to_pairs']):
            raise ValueError('--cpu-pair-open-steps needs an in-progress pair-growth checkpoint')
        if args.cpu_pair_open_steps < 1:
            raise ValueError('--cpu-pair-open-steps must be positive')
        elapsed = max(0, saved['step'] - growth['start_step'])
        oldbias = growth['initial_bias'] * max(0., 1. - elapsed / max(1, growth['warmup_steps']))
        if oldbias >= -1e-8:
            raise ValueError('Pair gate already open; cannot reschedule')
        old_growth = growth.copy()
        growth = dict(growth, start_step=saved['step'],
                      warmup_steps=args.cpu_pair_open_steps, initial_bias=oldbias)
        history.append({'kind':'cpu_pair_gate_schedule_continuation', 'step':saved['step'],
                        'previous':old_growth,'new':growth.copy(),
                        'bias_continuity':oldbias})
        print(f'RESCHEDULED new mirror-pair gate: bias continuous at {oldbias:.5f}, '
              f'new release across {args.cpu_pair_open_steps} updates; model/optimizer unchanged')
    if changing_data:
        print('EXPLICIT CORPUS TRANSITION: retaining the SAME model and optimizer, new train/validation tokens')
        history.append({'kind': 'data_transition', 'step': saved['step'],
                        'from_train_sha256': saved['data'].get('train_sha256'),
                        'to_train_sha256': meta.get('train_sha256'),
                        'from_valid_sha256': saved['data'].get('valid_sha256'),
                        'to_valid_sha256': meta.get('valid_sha256'),
                        'warning': 'A newly held-out book may overlap with earlier training; audit provenance'})
    growing = bool(saved and args.grow_pairs and args.grow_pairs > cfg.pairs)
    deepen = bool(saved and args.add_layers > 0)
    widening = bool(saved and args.widen_dim and args.widen_dim > cfg.dim)
    expert_growing = bool(saved and args.grow_expert_hidden and args.grow_expert_hidden > cfg.expert_hidden)
    if args.widen_dim is not None and (not saved or args.widen_dim < cfg.dim):
        raise ValueError('--widen-dim requires a checkpoint and nonshrinking width')
    if args.grow_expert_hidden is not None and (not saved or args.grow_expert_hidden < cfg.expert_hidden):
        raise ValueError('--grow-expert-hidden requires a checkpoint and nonshrinking inner size')
    if args.add_layers < 0:
        raise ValueError('--add-layers must be nonnegative')
    if args.add_layers > 0 and not saved:
        raise ValueError('--add-layers needs a prior checkpoint; initialize with --layers')
    if args.set_context is not None:
        if not saved:
            raise ValueError('--set-context only applies to an existing checkpoint; initialize with --context')
        if args.set_context < cfg.context:
            raise ValueError('--set-context cannot shrink the trained context in this workflow')
        if args.set_context != cfg.context:
            print(f'EXPANDED context {cfg.context} -> {args.set_context} tokens; exact same learned weights')
            history.append({'kind': 'context', 'from': cfg.context,
                            'to': args.set_context, 'step': saved['step']})
            cfg.context = args.set_context
    if args.grow_pairs is not None and not saved:
        raise ValueError('--grow-pairs requires an existing checkpoint; initialize with --pairs instead')
    if args.grow_pairs is not None and saved and args.grow_pairs < cfg.pairs:
        raise ValueError('Cannot shrink trained mirror-pair capacity')
    if widening:
        before_dim = cfg.dim
        widen_config(cfg, args.widen_dim)
        history.append({'kind': 'width', 'from': before_dim,
                        'to': args.widen_dim, 'step': saved['step'],
                        'noise_std': args.widen_noise,
                        'attention_layout': 'mixed' if cfg.legacy_attention_dim else 'uniform'})
    if expert_growing:
        history.append({'kind': 'expert_hidden', 'from': cfg.expert_hidden,
                        'to': args.grow_expert_hidden, 'step': saved['step']})
        cfg.expert_hidden = args.grow_expert_hidden
    if deepen:
        cfg.layers += args.add_layers
        # Anchor the temporal transition to the OLD final block. New blocks
        # initially have identity residuals and must not change +2/+3 logits.
        if cfg.future_block_index == -1:
            cfg.future_block_index = old_cfg.layers - 1
        history.append({'kind': 'depth', 'from': old_cfg.layers,
                        'to': cfg.layers, 'step': saved['step']})
    if growing:
        if growth and saved['step'] < growth['start_step'] + growth['warmup_steps']:
            raise ValueError('Complete previous capacity warmup before another growth')
        cfg.pairs = args.grow_pairs
        growth = {'from_pairs': old_cfg.pairs, 'to_pairs': cfg.pairs,
                  'start_step': saved['step'], 'warmup_steps': args.grow_warmup,
                  'initial_bias': -12.0}
        history.append(growth.copy())
    model = MirrorLM(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr or 3e-4, weight_decay=0.1,
                              foreach=False if device.type == 'cpu' else None)
    rng = np.random.default_rng(args.seed)
    if saved:
        if (growing or deepen or widening or expert_growing or enable_symbiotic or
                enable_hierarchy or enable_geometry_memory or enable_micro_meta or
                enable_simplex or enable_curvature):
            expand_architecture(model, optim, saved, old_cfg, widen_noise=args.widen_noise)
            ckpt = saved
            print(f'EXPANDED capacity: width {old_cfg.dim}->{cfg.dim}, depth {old_cfg.layers}->{cfg.layers}, '
                  f'pairs {old_cfg.pairs}->{cfg.pairs}, expert_hidden {old_cfg.expert_hidden}->{cfg.expert_hidden}; '
                  f'symbiotic_rank {old_cfg.symbiotic_rank}->{cfg.symbiotic_rank}; '
                  f'hierarchy_rank {old_cfg.hierarchy_rank}->{cfg.hierarchy_rank}; '
                  'old weights and Adam moments retained')
        else:
            ckpt = restore_checkpoint(path, model, optim)
        rng.bit_generator.state = ckpt['rng_numpy']
        torch.set_rng_state(ckpt['rng_torch'])
        random.setstate(ckpt['rng_python'])
        if 'rng_cuda' in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
        if args.lr is not None:
            for pg in optim.param_groups: pg['lr'] = args.lr
        start = ckpt['step']
        seen = ckpt['seen_tokens']
        print(f'RESUMED one canonical weight set at step={start} tokens={seen}')
    else:
        start, seen = 0, 0
        print('Initialized unified model (no v23 checkpoint was supplied).')
    if args.train_context is not None and not 4 <= args.train_context <= cfg.context:
        raise ValueError('--train-context must be within 4..checkpoint max context')
    mix = parse_context_mix(args.context_mix, cfg.context) if args.context_mix else None
    if mix and args.train_context is not None:
        raise ValueError('Use --context-mix OR --train-context, not both')
    active_context = max(mix[0]) if mix else (args.train_context or cfg.context)
    eval_context = args.eval_context or active_context
    if not 4 <= eval_context <= cfg.context:
        raise ValueError('--eval-context must be within 4..checkpoint maximum context')
    if args.grad_accum < 1:
        raise ValueError('--grad-accum must be positive')
    train_doc_lengths = legacy_document_lengths(trainmem, meta, 'train')
    train_samplers = {length: DocumentWindowSampler(trainmem, length, train_doc_lengths)
                      for length in (mix[0] if mix else [active_context])}
    train_sampler = train_samplers[active_context]
    valid_sampler = DocumentWindowSampler(valmem, eval_context,
        legacy_document_lengths(valmem, meta, 'valid'))
    if mix:
        history.append({'kind': 'cpu_context_mixture', 'step': start,
                        'lengths': mix[0], 'weights': mix[1].tolist(),
                        'evaluation_context': eval_context})
    print(f'Window-safe sampling: training {train_sampler.eligible_documents}/'
          f'{train_sampler.total_documents} docs, validation {valid_sampler.eligible_documents}/'
          f'{valid_sampler.total_documents} docs')
    nparam = sum(p.numel() for p in model.parameters())
    print(f'parameters={nparam:,} pairs={cfg.pairs} signed_experts={2*cfg.pairs} device={device} '
          f'max_context={cfg.context} train_context={active_context} eval_context={eval_context} '
          f'context_mix={args.context_mix or "none"} grad_accum={args.grad_accum}')
    fixed_rng = np.random.default_rng(2026)
    val_batches = [valid_sampler.windows(args.batch, fixed_rng, device)
                   for _ in range(args.eval_batches)]
    if args.steps < 0:
        raise ValueError('--steps cannot be negative')
    if args.steps == 0:
        # Record a whole-book corpus transition and an honest BEFORE-training
        # baseline at precisely the old optimization step, without updating
        # a single learned weight. This enables before/after CE comparison.
        model.eval()
        with torch.no_grad():
            initial_losses = [model.loss_with_future(v) for v in val_batches]
        ce = torch.stack([m['next_ce'] for m in initial_losses]).mean().item()
        ce2 = torch.stack([m.get('future2_ce',m['next_ce']) for m in initial_losses]).mean().item()
        ce3 = torch.stack([m.get('future3_ce',m['next_ce']) for m in initial_losses]).mean().item()
        report = {'step':start,'tokens_seen':seen,'zero_training_updates':True,
                  'active_train_context':active_context,'validation_ce':ce,
                  'validation_h2_ce':ce2,'validation_h3_ce':ce3,
                  'data_adopted':changing_data}
        print(json.dumps(report))
        save_checkpoint(path, model, optim, start, seen, meta, rng, cfg, growth, history)
        return
    model.train()
    for step in range(start + 1, start + args.steps + 1):
        update_growth_bias(model, growth, step)
        optim.zero_grad(set_to_none=True)
        train_ce_sum = 0.0
        tokens_this_step = 0
        for _micro in range(args.grad_accum):
            selected_context = int(rng.choice(mix[0], p=mix[1])) if mix else active_context
            batch = train_samplers[selected_context].windows(args.batch, rng, device)
            tokens_this_step += args.batch * selected_context
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bool(args.bf16 and device.type == 'cuda')):
                losses = model.loss_with_future(batch)
                micro_loss = losses['loss'] / args.grad_accum
            micro_loss.backward()
            train_ce_sum += losses['next_ce'].detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        seen += tokens_this_step
        if step == start + 1 or step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_losses = [model.loss_with_future(v) for v in val_batches]
                val_ce = torch.stack([v['next_ce'] for v in val_losses]).mean().item()
                val_h2 = torch.stack([v.get('future2_ce', v['next_ce']) for v in val_losses]).mean().item()
                val_h3 = torch.stack([v.get('future3_ce', v['next_ce']) for v in val_losses]).mean().item()
            print(json.dumps({'step': step, 'tokens_seen': seen,
                'train_ce': round(train_ce_sum / args.grad_accum, 5),
                'validation_ce': round(val_ce, 5),
                'validation_h2_ce': round(val_h2, 5), 'validation_h3_ce': round(val_h3, 5),
                'validation_ppl': round(math.exp(min(val_ce, 30)), 3)}))
            model.train()
        if step % args.save_every == 0 or step == start + args.steps:
            save_checkpoint(path, model, optim, step, seen, meta, rng, cfg, growth, history)
    print(f'Canonical checkpoint: {path}')


@torch.no_grad()
def generate(args):
    path = checkpoint_path(args.run_dir)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = Config(**saved['config'])
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = MirrorLM(cfg).to(device)
    load_v25_compatible(model, saved['model'])
    model.eval()
    codec = TokenCodec(saved['data']['tokenizer_model'])
    if saved['data']['tokenizer_model'] and file_hash(saved['data']['tokenizer_model']) != saved['data']['tokenizer_sha256']:
        raise ValueError('Tokenizer changed since training')
    ids = codec.encode(args.prompt)
    if not ids:
        raise ValueError('Prompt must not be empty')
    torch.manual_seed(args.seed)
    for _ in range(args.max_new_tokens):
        inp = torch.tensor([ids[-cfg.context:]], dtype=torch.long, device=device)
        logits, _, _ = model(inp)
        scores = logits[0, -1].float() / max(args.temperature, 1e-4)
        if args.topk_sample > 0:
            vals, top_ids = scores.topk(min(args.topk_sample, scores.numel()))
            p = vals.softmax(-1)
            idx = int(top_ids[torch.multinomial(p, 1)].item())
        else:
            idx = int(scores.argmax().item())
        ids.append(idx)
    print(codec.decode(ids))



@torch.no_grad()
def evaluate(args):
    """Test ablations on the SAME weights; never fork or retrain checkpoints."""
    path = checkpoint_path(args.run_dir)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = Config(**saved['config'])
    checkpoint_context = cfg.context
    if args.context_override:
        cfg.context = args.context_override
        cfg.check()
    split = getattr(args, 'split', 'valid')
    if split == 'test' and 'test_doc_lengths' not in saved['data']:
        raise ValueError('This checkpoint does not include a separately held-out test split')
    split_path = Path(args.data_dir) / (split + '.bin')
    valmem = np.memmap(split_path, dtype=saved['data'].get('token_dtype', '<u4'), mode='r')
    expected_val = saved['data'].get('bin_' + split + '_sha256') or (
        saved['data'].get(split + '_sha256') if saved['data'].get('split_unit') == 'whole_book' else None)
    if expected_val and file_hash(split_path) != expected_val:
        raise ValueError(f'Held-out {split} tokens changed')
    if load_meta(args.data_dir) != saved['data']:
        raise ValueError('Evaluation data/tokenizer differs from checkpoint')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = MirrorLM(cfg).to(device)
    load_v25_compatible(model, saved['model'])
    model.eval()
    rng = np.random.default_rng(args.seed)
    sampler = DocumentWindowSampler(valmem, cfg.context,
        legacy_document_lengths(valmem, saved['data'], split))
    if args.exhaustive:
        batches = list(sampler.exhaustive(args.batch, device, args.eval_stride or cfg.context))
        if args.max_eval_batches:
            batches = batches[:args.max_eval_batches]
    else:
        batches = [sampler.windows(args.batch, rng, device) for _ in range(args.eval_batches)]
    if not batches:
        raise ValueError('No evaluation windows')
    moes = [b.ff for b in model.blocks if b.use_moe]

    def measure(disable_temporal=False):
        sums = np.zeros(9, dtype=np.float64)
        total = 0
        for ids in batches:
            x = ids[:, :-3]
            logits, h, losses = model(x, ids[:, 1:-2])
            p2, p3 = model.future_logits(h, disable_temporal=disable_temporal)
            targets = [ids[:, 1:-2], ids[:, 2:-1], ids[:, 3:]]
            predictions = [logits, p2, p3]
            positions = int(targets[0].numel())
            total += positions
            for j, (logit, target) in enumerate(zip(predictions, targets)):
                sums[j] += F.cross_entropy(logit.flatten(0, 1), target.flatten()).item() * positions
                sums[j+3] += (logit.argmax(-1) == target).float().mean().item() * positions
                top5 = logit.topk(min(5, cfg.vocab_size), dim=-1).indices
                sums[j+6] += (top5 == target.unsqueeze(-1)).any(-1).float().mean().item() * positions
        means = sums / total
        return {'ce_1': round(means[0], 6), 'ce_2': round(means[1], 6),
                'ce_3': round(means[2], 6),
                'acc_1': round(means[3], 6), 'acc_2': round(means[4], 6),
                'acc_3': round(means[5], 6),
                'top5_1': round(means[6], 6), 'top5_2': round(means[7], 6),
                'top5_3': round(means[8], 6),
                'ppl_1': round(math.exp(min(means[0], 30.)), 4),
                'ppl_2': round(math.exp(min(means[1], 30.)), 4),
                'ppl_3': round(math.exp(min(means[2], 30.)), 4),
                'bits_per_token_1': round(means[0] / math.log(2), 6),
                'bits_per_token_2': round(means[1] / math.log(2), 6),
                'bits_per_token_3': round(means[2] / math.log(2), 6)}

    results = {'base': measure()}
    if args.unigram_baseline:
        trainmem = np.memmap(Path(args.data_dir) / 'train.bin',
                             dtype=saved['data'].get('token_dtype', '<u4'), mode='r')
        counts = np.ones(cfg.vocab_size, dtype=np.float64)  # add-one smoothed
        for offset in range(0, len(trainmem), 1_000_000):
            block = trainmem[offset:offset + 1_000_000]
            counts += np.bincount(block.astype(np.int64), minlength=cfg.vocab_size)
        log_p = np.log(counts / counts.sum())
        targets = np.concatenate([v[:, 1:-2].cpu().numpy().ravel() for v in batches])
        base_ce = float(-log_p[targets].mean())
        results['unigram_prior'] = {'ce_1': round(base_ce, 6),
                                     'ppl_1': round(math.exp(base_ce), 4)}
    if args.ablations:
        simplex_geometries = [m.symbiotic for m in moes
                              if m.symbiotic is not None and m.symbiotic.simplex is not None]
        if simplex_geometries:
            for geometry in simplex_geometries:
                geometry.simplex_enabled = False
            results['no_simplex_transfers'] = measure()
            for geometry in simplex_geometries:
                geometry.simplex_enabled = True
        curvature_geometries = [m.symbiotic for m in moes
                                if m.symbiotic is not None and m.symbiotic.curvature is not None]
        if curvature_geometries:
            for geometry in curvature_geometries:
                geometry.curvature_enabled = False
            results['no_triangle_curvature_memory'] = measure()
            for geometry in curvature_geometries:
                geometry.curvature_enabled = True
        micro_meta = [m.symbiotic for m in moes
                      if m.symbiotic is not None and m.symbiotic.micro_meta]
        if micro_meta:
            for attr, result_name in (
                    ('point_micro_enabled','no_point_microtransformers'),
                    ('triangle_abstract_enabled','no_triangle_abstract_address'),
                    ('meta_moe_enabled','no_triangle_fed_meta_moe')):
                for geometry in micro_meta:
                    setattr(geometry, attr, False)
                results[result_name] = measure()
                for geometry in micro_meta:
                    setattr(geometry, attr, True)
        if model.hierarchy is not None:
            original_hierarchy = model.hierarchy
            model.hierarchy = None
            results['no_token_hierarchy'] = measure()
            model.hierarchy = original_hierarchy
        geometric_memories = [m.symbiotic.memory for m in moes
                              if m.symbiotic is not None and m.symbiotic.memory is not None]
        if geometric_memories:
            saved_gates = [memory.gate.detach().clone() for memory in geometric_memories]
            for memory in geometric_memories:
                memory.gate.zero_()
            results['no_geometric_temporal_memory'] = measure()
            for memory, value in zip(geometric_memories, saved_gates):
                memory.gate.copy_(value)
        original_geometry = [m.symbiotic for m in moes]
        if any(g is not None for g in original_geometry):
            for m in moes:
                m.symbiotic = None
            results['no_symbiotic_geometry'] = measure()
            for m, g in zip(moes, original_geometry):
                m.symbiotic = g
        prev = [m.strength for m in moes]
        for m in moes:
            m.strength = 0.
        results['no_hyperbolic_routing'] = measure()
        for m, strength in zip(moes, prev):
            m.strength = strength

        anti_backup = [(m.in_anti.clone(), m.out_anti.clone()) for m in moes]
        for m in moes:
            m.in_anti.zero_()
            m.out_anti.zero_()
        results['no_antisymmetric_weights'] = measure()
        for m, (ia, oa) in zip(moes, anti_backup):
            m.in_anti.copy_(ia)
            m.out_anti.copy_(oa)

        results['no_temporal_mechanism'] = measure(disable_temporal=True)
    print(json.dumps({'checkpoint_step': saved['step'], 'split': split,
                      'data_split_sha256': expected_val or file_hash(split_path),
                      'evaluation_seed':args.seed,'exhaustive':bool(args.exhaustive),
                      'eval_stride':args.eval_stride or cfg.context,
                      'checkpoint_context': checkpoint_context,
                      'evaluation_context': cfg.context, 'future_mode': cfg.future_mode, 'evaluated_tokens':
                      sum(v.shape[0] * cfg.context for v in batches),
                      'ablations_use_same_weights': True, 'results': results}, indent=2))


def count_parameters_exact(cfg):
    """Parameter count of MirrorLM without constructing its tensors.

    Counts tied vocabulary embedding once; does not count nontrainable
    pair_bias buffers or optimizer slots.
    """
    cfg.check()
    d, p, f = cfg.dim, cfg.pairs, cfg.expert_hidden
    total = cfg.vocab_size * d + 3*d  # embedding, final norm, two future offsets
    for i in range(cfg.layers):
        if (i + 1) % cfg.moe_every == 0:
            total += 5*d*d + 4*p*d*f + 3*p*d + p + 3*d
            if cfg.symbiotic_rank:
                r = cfg.symbiotic_rank
                total += 3*d*r + 1024*r + r*r + 4
                if cfg.geometry_memory_chunk:
                    total += 4*r*r + 1
                if cfg.micro_meta:
                    total += 4*r*r + 8192*r + 512*r*(max(4,r//4)+max(8,r//2))
                if cfg.simplex_transfer:
                    total += (65280+8192)*r + 8*r*r + 1
                if cfg.triangle_curvature:
                    total += 2*r*r + (12+cfg.vocab_size)*r + 1
        else:
            total += 16*d*d + 2*d
    if cfg.hierarchy_rank:
        r = cfg.hierarchy_rank
        total += 2*d*r + 3*r*r + 4
    return total


def size_report(args):
    cfg = Config(vocab_size=args.vocab_size, dim=args.dim, layers=args.layers,
                 heads=args.heads, pairs=args.pairs, topk_pairs=args.topk,
                 expert_hidden=args.expert_hidden, context=args.context,
                 moe_every=args.moe_every)
    total = count_parameters_exact(cfg)
    print(json.dumps({'estimated_exact_parameters': total,
                      'signed_expert_transformations': 2*cfg.pairs,
                      'selected_mirror_pairs_per_token': cfg.topk_pairs,
                      'context': cfg.context,
                      'static_weights_gib_fp32': round(total*4/2**30, 3),
                      'weights_gradients_adam_moments_gib_fp32': round(total*16/2**30, 3),
                      'warning': 'No attention activations, kernel workspaces, KV caches, '
                      'optimizer overhead, distributed buffers, or training throughput included.'}, indent=2))



def smoke(_args):
    torch.manual_seed(7)
    cfg = Config(dim=32, heads=4, layers=2, pairs=4, topk_pairs=2,
                 expert_hidden=48, context=8, vocab_size=64, moe_every=2)
    model = MirrorLM(cfg)
    batch = torch.randint(0, cfg.vocab_size, (2, cfg.context + 3))
    m = model.loss_with_future(batch)
    assert all(torch.isfinite(v).all() for v in m.values())
    m['loss'].backward()
    routed = model.blocks[1].ff
    assert routed.in_shared.grad is not None and routed.in_anti.grad is not None
    assert routed.centers.grad is not None
    # The representation really resides inside the high-dimensional ball.
    z = torch.randn(30, cfg.dim) * 10
    assert torch.linalg.vector_norm(routed.to_ball(z), dim=-1).max().item() < 1.
    # Changing future inputs cannot affect the previous position logits (causality).
    a = batch[:, :-3].clone()
    b = a.clone()
    b[:, 5:] = torch.randint(0, cfg.vocab_size, b[:, 5:].shape)
    model.eval()
    with torch.no_grad():
        la = model(a)[0][:, :5]
        lb = model(b)[0][:, :5]
    assert torch.allclose(la, lb, atol=2e-5), 'causality violation'
    cfg.future_mode = 'rollout'
    m_roll = model.loss_with_future(batch)
    assert all(torch.isfinite(v).all() for v in m_roll.values())
    with torch.no_grad():
        _, ht, _ = model(a)
        r2, r3 = model.future_logits(ht)
        assert r2.shape == r3.shape == model(a)[0].shape
    print('PASS: mirrored gradients, geometry, causality, finite rollout, same vocabulary head')
    print('This is a computational smoke test, not an accuracy or scaling result.')



def grow_checkpoint(args):
    """Capacity-migrate ONE canonical checkpoint before data adoption/training.

    Saves atomically only after measuring the old-context output drift. The
    training step, AdamW history, token count, dataset fingerprint and RNG
    histories are preserved. --widen-noise 0 ensures exact zero-extension;
    nonzero noise makes new hidden channels trainable at small output drift.
    """
    path = checkpoint_path(args.run_dir)
    if not path.exists():
        raise FileNotFoundError('No previous checkpoint: ' + str(path))
    torch.set_num_threads(max(1, args.threads))
    saved = torch.load(path, map_location='cpu', weights_only=False)
    original = Config(**saved['config'])
    cfg = copy.deepcopy(original)
    if args.widen_dim is not None:
        widen_config(cfg, args.widen_dim)
    if args.grow_pairs is not None:
        if args.grow_pairs < cfg.pairs:
            raise ValueError('Cannot shrink paired experts')
        cfg.pairs = args.grow_pairs
    if args.grow_expert_hidden is not None:
        if args.grow_expert_hidden < cfg.expert_hidden:
            raise ValueError('Cannot shrink expert hidden')
        cfg.expert_hidden = args.grow_expert_hidden
    if args.add_layers < 0:
        raise ValueError('Cannot remove trained residual layers')
    if args.add_layers:
        cfg.future_block_index = original.layers - 1 if cfg.future_block_index == -1 else cfg.future_block_index
        cfg.layers += args.add_layers
    if args.set_context is not None:
        if args.set_context < cfg.context:
            raise ValueError('Cannot shrink trained context')
        cfg.context = args.set_context
    if getattr(args, 'attention_window', None) is not None:
        if original.attention_window and args.attention_window != original.attention_window:
            raise ValueError('Cannot change a trained attention window')
        if args.attention_window and args.attention_window < original.context:
            raise ValueError('Attention window must preserve inherited context')
        cfg.attention_window = args.attention_window
    if getattr(args, 'geometry_memory_chunk', None) is not None:
        if original.geometry_memory_chunk and args.geometry_memory_chunk != original.geometry_memory_chunk:
            raise ValueError('Cannot resize already trained geometric memory')
        cfg.geometry_memory_chunk = args.geometry_memory_chunk
    if getattr(args, 'micro_meta', False):
        cfg.micro_meta = True
    if getattr(args, 'simplex_transfer', False):
        cfg.simplex_transfer = True
    if getattr(args, 'triangle_curvature', False):
        cfg.triangle_curvature = True
    if args.dispatch_mode is not None:
        cfg.dispatch_mode = args.dispatch_mode
    if args.gather_chunk is not None:
        cfg.gather_chunk = args.gather_chunk
    requested_symbiotic_rank = getattr(args, 'symbiotic_rank', None)
    if requested_symbiotic_rank is not None:
        if original.symbiotic_rank and requested_symbiotic_rank != original.symbiotic_rank:
            raise ValueError('Cannot resize trained geometric concepts')
        cfg.symbiotic_rank = requested_symbiotic_rank
    requested_hierarchy_rank = getattr(args, 'hierarchy_rank', None)
    if requested_hierarchy_rank is not None:
        if original.hierarchy_rank and requested_hierarchy_rank != original.hierarchy_rank:
            raise ValueError('Cannot resize a trained token hierarchy')
        cfg.hierarchy_rank = requested_hierarchy_rank
    cfg.check()
    expanded = any((cfg.dim != original.dim, cfg.layers != original.layers,
                    cfg.pairs != original.pairs, cfg.expert_hidden != original.expert_hidden,
                    cfg.symbiotic_rank != original.symbiotic_rank,
                    cfg.hierarchy_rank != original.hierarchy_rank,
                    cfg.geometry_memory_chunk != original.geometry_memory_chunk,
                    cfg.micro_meta != original.micro_meta,
                    cfg.simplex_transfer != original.simplex_transfer,
                    cfg.triangle_curvature != original.triangle_curvature))
    changed = (expanded or cfg.context != original.context or
               cfg.dispatch_mode != original.dispatch_mode or
               cfg.attention_window != original.attention_window)
    if not changed:
        raise ValueError('No capacity/context/dispatch change requested')
    if expanded and saved.get('growth'):
        prior = saved['growth']
        if cfg.pairs > original.pairs and saved['step'] < prior['start_step'] + prior['warmup_steps']:
            raise ValueError('A previous pair warmup is unfinished; finish its training before growing more pairs')
    old_model = MirrorLM(original)
    load_v25_compatible(old_model, saved['model'])
    old_model.eval()
    new_model = MirrorLM(cfg)
    optim = torch.optim.AdamW(new_model.parameters(), lr=3e-4, weight_decay=0.1)
    if expanded:
        expand_architecture(new_model, optim, saved, original,
                            bias=args.initial_pair_bias, widen_noise=args.widen_noise)
    else:
        load_v25_compatible(new_model, saved['model'])
        optim.load_state_dict(saved['optimizer'])
    new_model.eval()
    # Check exact same input sequence: extending max context must not be
    # mistaken for equivalence between sequences with different histories.
    generator = torch.Generator().manual_seed(2731)
    seq = torch.randint(0, original.vocab_size, (1, min(48, original.context)), generator=generator)
    with torch.no_grad():
        old_logits, old_h, _ = old_model(seq)
        new_logits, new_h, _ = new_model(seq)
        old_future = old_model.future_logits(old_h)
        new_future = new_model.future_logits(new_h)
    differences = [float((a-b).abs().max()) for a,b in
                   [(old_logits,new_logits), *zip(old_future,new_future)]]
    if any(not math.isfinite(d) or d > args.max_logit_drift for d in differences):
        raise AssertionError(f'Capacity growth changed learned logits: max abs drifts={differences}')
    # Require moment preservation for every previously learned coordinate;
    # optimizer state step, epoch and RNG histories survive untouched.
    old_params = list(dict(old_model.named_parameters()))
    new_params = list(dict(new_model.named_parameters()))
    old_ids = saved['optimizer']['param_groups'][0]['params']
    new_ids = optim.state_dict()['param_groups'][0]['params']
    new_state = optim.state_dict()['state']
    new_ids_by_name = dict(zip(new_params, new_ids))
    old_ids_by_name = dict(zip(old_params, old_ids))
    moment_fields = 0
    for name in old_params:
        old_st = saved['optimizer']['state'].get(old_ids_by_name[name], {})
        new_st = new_state.get(new_ids_by_name[name], {})
        for key in ('exp_avg', 'exp_avg_sq'):
            if key not in old_st: continue
            old_m = old_st[key]
            new_m = new_st[key]
            check_m = torch.empty_like(new_m)
            morph_tensor(name, check_m, old_m, original, cfg)
            if not torch.equal(new_m, check_m):
                raise AssertionError('Adam moment corruption: ' + name + '.' + key)
            moment_fields += 1
        if 'step' in old_st and not torch.equal(new_st['step'], old_st['step']):
            raise AssertionError('Adam optimizer step changed: ' + name)
    history = list(saved.get('growth_history', []))
    history.append({'kind':'unified_capacity_migration', 'step':saved['step'],
       'old':asdict(original), 'new':asdict(cfg),
       'test_max_abs_logit_drifts': differences, 'widen_noise':args.widen_noise,
       'adam_moment_fields_verified':moment_fields})
    growth = saved.get('growth')
    if cfg.pairs > original.pairs:
        growth = {'from_pairs':original.pairs, 'to_pairs':cfg.pairs,
             'start_step':saved['step'], 'warmup_steps':args.grow_warmup,
             'initial_bias':args.initial_pair_bias}
    rng = np.random.default_rng()
    rng.bit_generator.state = saved['rng_numpy']
    # Save without clobbering checkpoint RNG streams (expansion itself consumes
    # temporary RNG draws), enabling exact original-data continuation.
    save_checkpoint(path, new_model, optim, saved['step'], saved['seen_tokens'],
                    saved['data'], rng, cfg, growth, history)
    new_saved = torch.load(path, map_location='cpu', weights_only=False)
    new_saved['rng_torch'] = saved['rng_torch']
    new_saved['rng_python'] = saved['rng_python']
    if 'rng_cuda' in saved:
        new_saved['rng_cuda'] = saved['rng_cuda']
    # Replacing RNG values does not touch model/optimizer: atomic final save.
    import tempfile
    with tempfile.NamedTemporaryFile(prefix='r28_', suffix='.pt', dir=path.parent, delete=False) as fh:
        temporary = Path(fh.name)
    try:
        torch.save(new_saved, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    report = {'checkpoint':str(path), 'step':saved['step'],
              'tokens_seen':saved['seen_tokens'], 'prior_parameters':count_parameters_exact(original),
              'expanded_parameters':count_parameters_exact(cfg),
              'old_context':original.context,'new_context':cfg.context,
              'old_mirror_pairs':original.pairs,'new_mirror_pairs':cfg.pairs,
              'new_signed_experts':2*cfg.pairs,'new_hidden_dim':cfg.dim,
              'new_layers':cfg.layers, 'new_expert_hidden':cfg.expert_hidden,
              'new_symbiotic_rank':cfg.symbiotic_rank,
              'new_hierarchy_rank':cfg.hierarchy_rank,
              'new_attention_window':cfg.attention_window,
              'new_geometry_memory_chunk':cfg.geometry_memory_chunk,
              'micro_meta':cfg.micro_meta,
              'simplex_transfer':cfg.simplex_transfer,
              'triangle_curvature':cfg.triangle_curvature,
              'new_attention_heads':cfg.heads,
              'legacy_attention_heads':cfg.legacy_attention_heads,
              'max_abs_logit_drift_h1_h2_h3':differences,
              'adam_moment_fields_verified':moment_fields,
              'data_unchanged': new_saved['data'] == saved['data'],
              'growth_requires_training':bool(cfg.pairs > original.pairs)}
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='cmd', required=True)
    commands.add_parser('smoke')
    size = commands.add_parser('size', help='estimate exact parameter count without model allocation')
    size.add_argument('--vocab-size', type=int, default=32000)
    size.add_argument('--dim', type=int, default=256)
    size.add_argument('--layers', type=int, default=8)
    size.add_argument('--heads', type=int, default=8)
    size.add_argument('--pairs', type=int, default=32)
    size.add_argument('--topk', type=int, default=2)
    size.add_argument('--expert-hidden', type=int, default=256)
    size.add_argument('--context', type=int, default=2048)
    size.add_argument('--moe-every', type=int, default=2)
    gr = commands.add_parser('grow', help='atomically grow one existing checkpoint without synthetic training')
    gr.add_argument('--run-dir', required=True)
    gr.add_argument('--widen-dim', type=int,
                    help='includes function-preserving 192D to 256D mixed-head migration')
    gr.add_argument('--add-layers', type=int, default=0)
    gr.add_argument('--grow-pairs',type=int)
    gr.add_argument('--grow-expert-hidden',type=int)
    gr.add_argument('--symbiotic-rank',type=int,help='enable 256-point Diamond without changing inherited logits or Adam states')
    gr.add_argument('--hierarchy-rank',type=int,
                    help='enable causal byte/pair/word/sentence attention feedback without changing prior logits')
    gr.add_argument('--set-context',type=int)
    gr.add_argument('--attention-window',type=int,
                    help='bounded causal attention; >= inherited context retains old logits')
    gr.add_argument('--geometry-memory-chunk',type=int,
                    help='two causal geometric apex-memory loops over completed chunks')
    gr.add_argument('--micro-meta', action='store_true',
                    help='256 point microtransformers feeding 256 apex MoE experts via sparse C(256,3)')
    gr.add_argument('--simplex-transfer', action='store_true',
                    help='oriented pair/face/tetra concept tokens, causal storage and meta tetra feedback')
    gr.add_argument('--triangle-curvature', action='store_true',
                    help='hyperbolic triangle angle/rotation memory with causal phase-aligned reads')
    gr.add_argument('--dispatch-mode',choices=('grouped','gather'))
    gr.add_argument('--gather-chunk',type=int)
    gr.add_argument('--widen-noise',type=float,default=1e-4)
    gr.add_argument('--initial-pair-bias',type=float,default=-12.)
    gr.add_argument('--grow-warmup',type=int,default=4000)
    gr.add_argument('--max-logit-drift',type=float,default=5e-4)
    gr.add_argument('--threads',type=int,default=2)
    gr.add_argument('--report',default=None)
    prep = commands.add_parser('prepare')
    prep.add_argument('--train-text', required=True)
    prep.add_argument('--valid-text', required=True)
    prep.add_argument('--out-dir', required=True)
    prep.add_argument('--tokenizer-model', default=None, help='fixed SentencePiece .model; else byte tokenizer')
    books = commands.add_parser('prepare-books', help='deterministic, deduplicated, book-level split')
    books.add_argument('--source-dir', required=True)
    books.add_argument('--out-dir', required=True)
    books.add_argument('--tokenizer-model', default=None)
    books.add_argument('--valid-fraction', type=float, default=0.10)
    books.add_argument('--test-fraction', type=float, default=0.0,
                       help='reserve untouched whole books for a separate test split')
    books.add_argument('--exclude-gutenberg-ids', default=None,
                       help='JSON list of old-corpus Gutenberg IDs to exclude before splitting')
    books.add_argument('--min-bytes', type=int, default=128)
    books.add_argument('--max-doc-bytes', type=int, default=64 * 1024 * 1024)
    books.add_argument('--seed', type=int, default=2026)
    tr = commands.add_parser('train')
    tr.add_argument('--data-dir', required=True)
    tr.add_argument('--adopt-data', action='store_true', help='explicitly transition to a new whole-book corpus with the same tokenizer and weights')
    tr.add_argument('--run-dir', required=True)
    tr.add_argument('--steps', type=int, default=1000, help='additional steps beyond latest checkpoint')
    tr.add_argument('--batch', type=int, default=4, help='microbatch size')
    tr.add_argument('--grad-accum', type=int, default=1, help='optimizer step accumulates this many microbatches')
    tr.add_argument('--train-context', type=int, default=None, help='active training window <= checkpoint maximum; for context curriculum')
    tr.add_argument('--context-mix', default=None,
                    help='stochastic CPU curriculum, e.g. 256:3,512:3,1024:2,2048:1,4096:1; same weights/optimizer')
    tr.add_argument('--eval-context', type=int, default=None,
                    help='fixed validation window size independent of training context mixture')
    tr.add_argument('--bf16', action='store_true', help='use CUDA bfloat16 autocast when supported')
    tr.add_argument('--context', type=int, default=256)
    tr.add_argument('--dim', type=int, default=256)
    tr.add_argument('--layers', type=int, default=4)
    tr.add_argument('--heads', type=int, default=4)
    tr.add_argument('--pairs', type=int, default=128)
    tr.add_argument('--topk', type=int, default=2)
    tr.add_argument('--expert-hidden', type=int, default=128)
    tr.add_argument('--moe-every', type=int, default=2)
    tr.add_argument('--geo', type=float, default=0.2)
    tr.add_argument('--balance', type=float, default=0.01)
    tr.add_argument('--future2', type=float, default=None, help='aux loss weight, optional override on resume')
    tr.add_argument('--future3', type=float, default=None, help='aux loss weight, optional override on resume')
    tr.add_argument('--future-mode', choices=('offset', 'rollout'), default=None,
                    help='shared-state rollout or legacy offsets; existing weights are preserved on switch')
    tr.add_argument('--lr', type=float, default=None, help='initial or explicitly overridden optimizer LR')
    tr.add_argument('--grow-pairs', type=int, default=None, help='increase pair capacity within the same checkpoint')
    tr.add_argument('--set-context', type=int, default=None, help='expand causal context with the same learned weights')
    tr.add_argument('--attention-window', type=int, default=None,
                    help='limit attention to a sliding window (>= inherited context on migration)')
    tr.add_argument('--geometry-memory-chunk', type=int, default=None,
                    help='two loops of causal apex self-attention over completed memory blocks')
    tr.add_argument('--micro-meta', action='store_true',
                    help='activate 256 sparse point microtransformers and triangle-fed apex MoE')
    tr.add_argument('--simplex-transfer', action='store_true',
                    help='activate sparse oriented simplex concept transport and causal storage')
    tr.add_argument('--triangle-curvature', action='store_true',
                    help='activate supervised curvature-phase memory; requires simplex transfer')
    tr.add_argument('--add-layers', type=int, default=0, help='append identity-initialized residual blocks')
    tr.add_argument('--widen-dim', type=int, default=None,
                    help='expand width without changing learned head geometry, including 192D to 256D')
    tr.add_argument('--grow-expert-hidden', type=int, default=None, help='expand every pair expert inner width')
    tr.add_argument('--symbiotic-rank', type=int, default=None,
                    help='enable shared 256-point/triangle/tetra/apex feedback with zero-output migration')
    tr.add_argument('--hierarchy-rank',type=int,default=None,
                    help='causal byte/pair/word/sentence cross-scale attention; zero-output migration')
    tr.add_argument('--widen-noise', type=float, default=1e-4, help='small new-channel seed; set 0 for exact no-op')
    tr.add_argument('--dispatch-mode', choices=('grouped','gather'), default=None)
    tr.add_argument('--gather-chunk', type=int, default=None)
    tr.add_argument('--grow-warmup', type=int, default=1000, help='gradually unmask new pairs over steps')
    tr.add_argument('--cpu-pair-open-steps', type=int, default=None,
                    help='explicitly shorten an in-progress pair gate on ONE checkpoint; rebase bias continuously')
    tr.add_argument('--seed', type=int, default=1337)
    tr.add_argument('--eval-every', type=int, default=100)
    tr.add_argument('--eval-batches', type=int, default=4)
    tr.add_argument('--save-every', type=int, default=250)
    tr.add_argument('--cpu', action='store_true')
    tr.add_argument('--cpu-threads', type=int, default=2,
                    help='CPU intra-op threads; benchmarked 2 here (default 2)')
    ev = commands.add_parser('evaluate')
    ev.add_argument('--data-dir', required=True)
    ev.add_argument('--run-dir', required=True)
    ev.add_argument('--split', choices=('valid','test'), default='valid',
                    help='evaluate untouched book-level test split explicitly; never used for training or checkpoint selection')
    ev.add_argument('--batch', type=int, default=4)
    ev.add_argument('--eval-batches', type=int, default=8)
    ev.add_argument('--seed', type=int, default=2026)
    ev.add_argument('--ablations', action='store_true')
    ev.add_argument('--exhaustive', action='store_true', help='deterministic within-document sequential validation')
    ev.add_argument('--context-override', type=int, default=0, help='test longer causal context without changing weights')
    ev.add_argument('--eval-stride', type=int, default=0, help='stride for exhaustive windows, default=context')
    ev.add_argument('--max-eval-batches', type=int, default=0, help='cap exhaustive validation work')
    ev.add_argument('--unigram-baseline', action='store_true')
    ev.add_argument('--cpu', action='store_true')
    ge = commands.add_parser('generate')
    ge.add_argument('--run-dir', required=True)
    ge.add_argument('--prompt', required=True)
    ge.add_argument('--max-new-tokens', type=int, default=100)
    ge.add_argument('--temperature', type=float, default=0.8)
    ge.add_argument('--topk-sample', type=int, default=30)
    ge.add_argument('--seed', type=int, default=1337)
    ge.add_argument('--cpu', action='store_true')
    a = parser.parse_args()
    return {'smoke': smoke, 'grow': grow_checkpoint, 'size': size_report, 'prepare': prepare, 'prepare-books': prepare_books, 'train': train, 'evaluate': evaluate, 'generate': generate}[a.cmd](a)


if __name__ == '__main__':
    main()

