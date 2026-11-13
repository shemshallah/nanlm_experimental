#!/usr/bin/env python3
"""FA1 R7.25 causal controls for the existing (not a replacement) Diamond.

Swap only target-free Diamond context with a donor from a different book.
The inherited R7 hidden state, memory mixture, target, and book label remain
unchanged. Sampling with replacement handles unbalanced book sizes without
ever permitting a same-book donor or a future target to enter the context.
"""
from __future__ import annotations

import torch

DIAMOND_KEYS = ('word_ids', 'byte_ids', 'sentence_ids', 'vertex_ids')


def different_book_donors(book_ids, seed=725):
    books = torch.as_tensor(book_ids).detach().cpu().reshape(-1)
    if books.numel() < 2 or torch.unique(books).numel() < 2:
        raise ValueError('R7.25 requires at least two distinct books for hard negatives')
    rng = torch.Generator(device='cpu').manual_seed(int(seed))
    indices = torch.arange(books.numel(), dtype=torch.long)
    donors = torch.empty_like(indices)
    for book in torch.unique(books):
        original = indices[books == book]
        eligible = indices[books != book]
        pick = torch.randint(eligible.numel(), (original.numel(),), generator=rng)
        donors[original] = eligible[pick]
    assert torch.all(books[donors] != books)
    return donors


def unrelated_book_context(data, seed=725):
    """Return a copied input dict and donor indices; never change gold labels.

    All four local Diamond histories and the 92 near/far and 12 modal channels
    must come from the SAME foreign-book example. The inherited mixture still
    reads the original near/far features from the unchanged input dict.
    """
    donors = different_book_donors(data['book_ids'], seed)
    if len(data['target']) != len(donors):
        raise ValueError('book ID and target lengths differ')
    control = dict(data)
    for key in DIAMOND_KEYS:
        control[key] = data[key][donors.to(data[key].device)]
    memory = data.get('diamond_memory_features')
    if memory is None:
        memory = torch.cat((data['near_features'], data['far_features']), dim=-1)
    control['diamond_memory_features'] = memory[donors.to(memory.device)]
    for target_key, source_key in (
        ('diamond_modal_ids', 'memory_modal_ids'),
        ('diamond_modal_mask', 'memory_modal_mask'),
    ):
        source = data[target_key] if target_key in data else data[source_key]
        control[target_key] = source[donors.to(source.device)]
    return control, donors
