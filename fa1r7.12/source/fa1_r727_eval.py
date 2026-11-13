#!/usr/bin/env python3
"""FA1 R7.27 CPU audit on complete, non-overlapping NaNROUTER books.

The R7 model predicts vocabulary words and has a separately trained head for
the FIRST UTF-8 byte of the next word. It does not assign a likelihood to the
complete byte stream. Therefore first-byte accuracy is measurable, but genuine
raw-byte BPB is deliberately unavailable; do not relabel word CE as BPB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from r7_core import Corpus
from fa1_r723_train import prepare, forward
from fa1_r725_controls import unrelated_book_context
from fa1_r727_optimizer import load_r727_geometric
from fa1_r727_checkpoint import load_unified, digest


def validate_book_corpus(directory, split):
    root = Path(directory)
    inventory = {}
    all_hashes = set()
    for name in ('train', 'dev', 'test'):
        meta = json.loads((root / (name + '_documents.json')).read_text())
        docs = meta['documents']
        if not docs or meta.get('source') != 'individual Gutenberg text files':
            raise ValueError('whole-book metadata is missing or malformed')
        digests = [x['sha256'] for x in docs]
        if len(set(digests)) != len(digests) or all_hashes.intersection(digests):
            raise ValueError('train/dev/test share a book SHA-256')
        all_hashes.update(digests)
        if sum(int(x['words']) for x in docs) != meta['total_word_targets']:
            raise ValueError('incorrect document token counts')
        offsets = np.memmap(root / (name + '_doc_start.u32'), dtype='<u4')
        if len(offsets) != meta['total_word_targets']:
            raise ValueError('document-start index has wrong length')
        for doc in docs:
            start = int(doc['first_record'])
            if start >= len(offsets) or int(offsets[start]) != start:
                raise ValueError('document starts are not aligned')
        inventory[name] = {'books': len(docs), 'word_targets': int(meta['total_word_targets'])}
    if inventory[split]['books'] < 2:
        raise ValueError('need two or more independent books for cross-book controls')
    return inventory


@torch.no_grad()
def normalized_trunk_word_topk(model, base, hidden, targets):
    """Exactly normalized 32K-word tree head, without target-aware memory mix.

    The inherited conditional tree defines a probability for every leaf.
    Propagating log probabilities from the root permits correct top-k ranking;
    computing a top-k of target-specific NLLs would not be an equivalent test.
    """
    vocab = len(base.vocab)
    children = np.asarray(base.children, dtype=np.int64)
    if children.shape != (vocab - 1, 2):
        raise ValueError('unexpected R7 Huffman tree shape')
    layers = [[2 * vocab - 2]]
    visited = {layers[0][0]}
    while any(node >= vocab for node in layers[-1]):
        layer = []
        for node in layers[-1]:
            if node < vocab:
                continue
            for child in children[node - vocab]:
                child = int(child)
                if child in visited or child >= node or child < 0:
                    raise ValueError('cyclic or duplicated vocabulary tree node')
                visited.add(child)
                layer.append(child)
        layers.append(layer)
    if len(visited) != 2 * vocab - 1:
        raise ValueError('tree does not cover the full vocabulary')
    trunk = model.core.trunk
    logits = hidden.float() @ trunk.W.T + trunk.B
    log_all = torch.full((len(targets), 2 * vocab - 1), -torch.inf, dtype=logits.dtype)
    log_all[:, 2 * vocab - 2] = 0.
    for layer in layers[:-1]:
        inner = np.asarray([node for node in layer if node >= vocab], dtype=np.int64)
        if not len(inner):
            continue
        columns = torch.from_numpy(inner - vocab)
        outputs = children[inner - vocab]
        values = logits[:, columns]
        old = log_all[:, torch.from_numpy(inner)]
        log_all[:, torch.from_numpy(outputs[:, 0])] = old + F.logsigmoid(-values)
        log_all[:, torch.from_numpy(outputs[:, 1])] = old + F.logsigmoid(values)
    words = log_all[:, :vocab]
    totals = torch.logsumexp(words, dim=-1)
    if not torch.allclose(totals, torch.zeros_like(totals), atol=3.e-5):
        raise AssertionError('trunk word distribution not normalized')
    ranking = words.topk(5, dim=-1).indices
    word_ce = -words.gather(1, targets.long()[:, None]).squeeze(1)
    expected = trunk.nll(hidden, targets)
    if not torch.allclose(word_ce, expected, atol=2.e-4):
        raise AssertionError('tree probability disagrees with inherited trunk loss')
    return {'trunk_vocabulary_size': vocab,
            'trunk_word_loss_nats_per_word': float(word_ce.mean()),
            'trunk_next_word_top1': float((ranking[:, 0] == targets).float().mean()),
            'trunk_next_word_top5': float((ranking == targets[:, None]).any(-1).float().mean()),
            'trunk_normalization_verified': True,
            'full_model_word_top1': None,
            'full_model_word_top5': None,
            'full_model_word_accuracy_reason':
                'full near/far/Diamond mixture exposes target-conditioned NLL, not all-word normalized ranking'}


@torch.no_grad()
def evaluate(model, base, data, seed):
    model.eval()
    nll, extras = forward(model, data)
    byte_logits = extras['byte_logits']
    if byte_logits.shape != (len(nll), 256):
        raise ValueError('expected normalized 256-class first-byte head')
    log_p = F.log_softmax(byte_logits.float(), dim=-1)
    if not torch.allclose(log_p.exp().sum(dim=-1), torch.ones_like(nll), atol=1.e-6):
        raise ValueError('first-byte probabilities are not normalized')
    target = data['first_byte_target']
    sorted_bytes = log_p.topk(5, dim=-1).indices
    first_byte_nll = F.nll_loss(log_p, target, reduction='none')
    result = {
        'vocabulary_word_loss_nats_per_word': float(nll.mean()),
        'first_byte_nll_nats_per_next_word': float(first_byte_nll.mean()),
        'first_byte_cross_entropy_bits_per_next_word': float(first_byte_nll.mean() / np.log(2)),
        'first_byte_top1': float((sorted_bytes[:, 0] == target).float().mean()),
        'first_byte_top5': float((sorted_bytes == target[:, None]).any(-1).float().mean()),
        'first_byte_softmax_normalization_verified': True,
        'raw_byte_bits_per_byte': None,
        'raw_byte_top1': None,
        'raw_byte_top5': None,
        'raw_byte_metric_reason': 'auxiliary head predicts only first byte of each next vocabulary word',
        'n_word_targets': int(len(nll)),
    }
    result.update(normalized_trunk_word_topk(model, base, data['hidden'], data['target']))
    if 'book_ids' in data and torch.unique(data['book_ids']).numel() >= 2:
        wrong, donors = unrelated_book_context(data, seed)
        foreign, _ = forward(model, wrong)
        result['diamond_correct_history_advantage_nats_per_word'] = float((foreign - nll).mean())
        result['different_book_donors'] = int(len(donors))
        result['sampled_books'] = int(torch.unique(data['book_ids']).numel())
    return result


def run(a):
    if a.split != 'dev':
        raise ValueError('R7.27 gate selection is DEV-only: sealed test books remain untouched')
    inventory = validate_book_corpus(a.corpus_dir, a.split)
    torch.set_num_threads(a.threads)
    source_sha = digest(a.checkpoint)
    if a.unified_checkpoint:
        model, optimizer, base, comp, metadata = load_unified(a.checkpoint, a.root, a.threads)
    else:
        model, optimizer, base, comp, metadata = load_r727_geometric(a.checkpoint, a.root, a.threads)
    data = prepare(a, model, base, comp, a.split, a.targets, a.seed)
    if 'book_ids' not in data or len(torch.unique(data['book_ids'])) < 2:
        raise ValueError('sampling did not cover multiple independent DEV books')
    result = {'format': 'FA1_R7_27_BOOK_DISJOINT_CPU_METRICS_V1',
              'checkpoint_sha256': source_sha, 'checkpoint_format': metadata['format'],
              'promotion': False, 'model_parameter_tensors': len(list(model.parameters())),
              'optimizer_object_count': 1, 'optimizer_parameter_groups': len(optimizer.param_groups),
              'split': 'dev', 'seed': a.seed, 'corpus_inventory': inventory,
              'new_batch_book_sha_disjoint': True,
              'disjoint_from_original_R7_ancestor_training_books_verified': False,
              'not_a_sealed_independent_ancestor_generalization_test': True,
              'oov_word_targets': int((data['target'] == 0).sum()),
              'metrics': evaluate(model, base, data, a.seed + 11)}
    if a.output:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='R7.13 ancestor with corpus vocab/tree')
    p.add_argument('--candidate', dest='checkpoint', required=True)
    p.add_argument('--corpus-dir', required=True, help='document-disjoint prepared NaNROUTER data')
    p.add_argument('--split', choices=['dev'], default='dev')
    p.add_argument('--targets', type=int, default=96)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=727)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--unified-checkpoint', action='store_true')
    p.add_argument('--output', default='')
    run(p.parse_args())
