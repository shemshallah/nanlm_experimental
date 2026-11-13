#!/usr/bin/env python3
"""R7.31 layered, causal, sparse-attention decoder for the SAME FA1 Diamond.

This is an in-place addition to R7.30. All previously trained Parameter objects
and their single unified optimizer states are preserved. Attention runs over
only active vertices/edges/triangles, never dense C(256,3) memory.

The byte head models complete UTF-8 *word spellings* with an EOW symbol. Its
conditional bits/byte is NOT an uninterrupted raw-text bits/byte metric.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fa1_r730_factorized_geometry import FactorizedContextualGeometry

ROUND = 'FA1_R7_31_LAYERED_DIAMOND_V1'
EOW = 256
BYTE_CLASSES = 257
WIDTH = 32


class AttentionLoop(nn.Module):
    """Pre-norm residual attention, sparse in number of active concepts."""

    def __init__(self, width=WIDTH):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                nn.Linear(2 * width, width))

    def forward(self, x):
        q = self.norm(x)
        update, _ = self.attn(q, q, q, need_weights=False)
        x = x + update
        return x + self.ff(self.ff_norm(x))


class LayeredDiamondDecoder(nn.Module):
    """Byte, learned n-gram/subword, R7 token, word/sentence + geometry loops.

    'Subword' is a learned convolutional byte n-gram representation, NOT a
    separately fitted BPE tokenizer. Inherited R7 token IDs remain unchanged.
    """

    def __init__(self, width=WIDTH):
        super().__init__()
        self.width = width
        self.byte_embed = nn.Embedding(BYTE_CLASSES, width)
        self.byte_span3 = nn.Conv1d(width, width, 3, padding=1)
        self.byte_span5 = nn.Conv1d(width, width, 5, padding=2)
        self.byte_loop = AttentionLoop(width)
        self.subword_loop = AttentionLoop(width)
        self.token_proj = nn.Linear(96, width)
        self.token_loop = AttentionLoop(width)
        self.word_loop = AttentionLoop(width)
        self.sentence_proj = nn.Linear(96, width)
        self.sentence_loop = AttentionLoop(width)
        self.context_mix = nn.Linear(5 * width, width, bias=False)
        self.context_gain = nn.Parameter(torch.zeros(()))
        self.point_loop = AttentionLoop(width)
        self.pair_loop = AttentionLoop(width)
        self.triangle_loop = AttentionLoop(width)
        self.point_gain = nn.Parameter(torch.zeros(()))
        self.pair_gain = nn.Parameter(torch.zeros(()))
        self.triangle_gain = nn.Parameter(torch.zeros(()))
        self.word_readout = nn.Sequential(nn.Linear(3 * width, 2 * width),
                                           nn.GELU(), nn.Linear(2 * width, 96))
        self.word_gain = nn.Parameter(torch.zeros(()))
        self.byte_condition = nn.Linear(3 * width, width)
        self.byte_geometry_gain = nn.Parameter(torch.zeros(()))
        self.byte_position = nn.Embedding(65, width)
        self.byte_ar_loops = nn.ModuleList([
            nn.TransformerEncoderLayer(width, 4, dim_feedforward=2 * width,
                                       dropout=0., batch_first=True, norm_first=True)
            for _ in range(2)])
        self.byte_head = nn.Linear(width, BYTE_CLASSES)

    def context_update(self, byte_ids, word_ids, sentence_ids, trunk):
        """Strictly old context; no next-token labels or teacher-forced bytes."""
        b = self.byte_embed(byte_ids.long())
        byte = self.byte_loop(b.reshape(-1, 8, 16, self.width).mean(2))[:, -1]
        signal = b.transpose(1, 2)
        sub = (self.byte_span3(signal) + self.byte_span5(signal)).transpose(1, 2)
        sub = self.subword_loop(F.avg_pool1d(sub.transpose(1, 2), 4).transpose(1, 2))[:, -1]
        token = self.token_proj(F.embedding(word_ids.long(), trunk.E.detach()))
        token = self.token_loop(token)
        word = self.word_loop(F.avg_pool1d(token.transpose(1, 2), 4).transpose(1, 2))[:, -1]
        sent = self.sentence_proj(F.embedding(sentence_ids.long(), trunk.E.detach()))
        sent = self.sentence_loop(sent)[:, -1]
        features = torch.cat((byte, sub, token[:, -1], word, sent), dim=-1)
        return torch.tanh(self.context_gain) * self.context_mix(features)

    def concept_update(self, level, x):
        if level not in ('point', 'pair', 'triangle'):
            raise ValueError('unknown geometric layer: ' + str(level))
        loop = getattr(self, level + '_loop')
        gain = getattr(self, level + '_gain')
        # Subtract the original state so gain==0 is *exactly* an identity.
        return x + torch.tanh(gain) * (loop(x) - x)

    @staticmethod
    def route(point, apex):
        # Existing pair->point descent and apex must both be live.
        return (torch.tanh(point.norm(dim=-1, keepdim=True)) *
                torch.tanh(apex.norm(dim=-1, keepdim=True)))

    def descend(self, point, apex, query):
        fused = torch.cat((point * torch.tanh(apex), apex * torch.tanh(point), query), -1)
        route = self.route(point, apex)
        # R7.32 optionally learns *when* geometric evidence deserves trust.
        # These scalars equal one at attachment, exactly preserving every
        # inherited R7.31 logit and optimizer moment.
        word_trust = byte_trust = 1.
        if hasattr(self, 'word_geometry_router'):
            word_trust = 2. * torch.sigmoid(self.word_geometry_router(fused))
            byte_trust = 2. * torch.sigmoid(self.byte_geometry_router(fused))
        word = torch.tanh(self.word_gain) * route * word_trust * self.word_readout(fused)
        # Keep the geometric byte contribution separately gated; the byte
        # prefix transformer may learn useful syntax even when it is zero.
        baseline = self.byte_condition(torch.cat((torch.zeros_like(point),
                                                   torch.zeros_like(apex), query), -1))
        geometric = self.byte_condition(fused) - baseline
        byte_condition = baseline + torch.tanh(self.byte_geometry_gain) * route * byte_trust * geometric
        return word, byte_condition

    def autoregressive(self, byte_targets, condition):
        """Teacher-forced 257-class next-byte/EOW logits; causal per position.

        byte_targets is [B,T] containing target UTF-8 bytes, one EOW, and
        -100 padding. Target at t only enters the input at t+1. Forward word
        routing never sees this tensor.
        """
        if byte_targets.ndim != 2 or byte_targets.size(0) != condition.size(0):
            raise ValueError('expected [batch, byte positions] labels')
        batch, length = byte_targets.shape
        if length > 65:
            raise ValueError('maximum conditional word spelling is 64 bytes')
        if bool(((byte_targets != -100) &
                 ((byte_targets < 0) | (byte_targets > EOW))).any()):
            raise ValueError('invalid byte or EOW label')
        start = torch.full((batch, 1), EOW, device=byte_targets.device, dtype=torch.long)
        previous = torch.cat((start, byte_targets[:, :-1].clamp_min(0)), dim=1)
        positions = torch.arange(length, device=byte_targets.device)
        x = self.byte_embed(previous) + self.byte_position(positions)[None]
        x = x + condition[:, None, :]
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        for layer in self.byte_ar_loops:
            x = layer(x, src_mask=causal)
        return self.byte_head(x)


def attach_layered(model, optimizer, learning_rate=2.e-4):
    """R7.30 -> R7.31 migration; adds exactly one optimizer group."""
    if type(model.diamond.geometry) is not FactorizedContextualGeometry:
        raise TypeError('layered decoder requires the real R7.30 geometry')
    if hasattr(model.diamond, 'hierarchy'):
        raise ValueError('layered decoder is already attached')
    previous = {id(p) for p in model.parameters()}
    model.diamond.hierarchy = LayeredDiamondDecoder(model.diamond.d)
    added = [(n, p) for n, p in model.named_parameters() if id(p) not in previous]
    if not added or not all(n.startswith('diamond.hierarchy.') for n, _ in added):
        raise AssertionError('unexpected R7.31 parameter registration')
    if any(id(p) in previous for _, p in added):
        raise AssertionError('inherited R7 Parameter replaced')
    optimizer._name_by_id.update({id(p): n for n, p in added})
    optimizer._group_names.append({'algorithm': 'adamw', 'names': [n for n, _ in added]})
    optimizer._adamw.add_param_group({'params': [p for _, p in added],
                                     'lr': learning_rate, 'weight_decay': 0.})
    optimizer._source_group_counts[-1] += 1
    optimizer.param_groups = optimizer._legacy_groups + optimizer._adamw.param_groups
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {
            id(p) for p in model.parameters()}:
        raise AssertionError('not all R7 parameters are in the one optimizer')
    return model, optimizer


def load_layered(path, root, threads=2):
    """Rebuild one network and one optimizer without resetting old states."""
    from fa1_r727_checkpoint import load_unified
    from fa1_r729_contextual_geometry import ContextualGeometry, attach_contextual
    from fa1_r730_factorized_geometry import attach_factorized
    model, optimizer, base, comp, meta = load_unified(path, root, threads)
    if meta.get('architecture_round') == ROUND:
        if not hasattr(model.diamond, 'hierarchy'):
            raise AssertionError('R7.31 saved decoder was not restored')
        return model, optimizer, base, comp, meta
    if type(model.diamond.geometry) is not FactorizedContextualGeometry:
        if type(model.diamond.geometry) is not ContextualGeometry:
            attach_contextual(model, optimizer)
        attach_factorized(model, optimizer)
    attach_layered(model, optimizer)
    return model, optimizer, base, comp, meta
