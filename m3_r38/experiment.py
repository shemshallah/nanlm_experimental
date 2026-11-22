"""Book-disjoint causal training/evaluation for a frozen-parent R38 adapter.

The caller MUST provide books from NaNROUTER Release with a prelocked split.
The factory is a callable constructing a *fresh identical frozen parent* for
one book; never cache document-local state across books.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from .episodic import M3R38Predictor, corrected_logp, nll_grad, GROUPS

LOG2 = float(np.log(2.0))


def fit(books, parent_factory, optimizer, *, warmup=64, batch=64):
    pending = np.zeros(GROUPS)
    count = total = 0
    for data in books:
        model = M3R38Predictor(parent_factory(), optimizer.theta)
        for i, byte in enumerate(data):
            if i >= warmup:
                base, feat = model.predict_parts()
                lp = corrected_logp(base, feat, optimizer.theta)
                pending += nll_grad(lp, feat, byte)
                count += 1
                total += 1
                if count == batch:
                    optimizer.update(pending/count)
                    model.theta[:] = optimizer.theta
                    pending[:] = 0.0
                    count = 0
            model.observe(int(byte))
    if count:
        optimizer.update(pending/count)
    return {'fit_scored_positions': total, 'updates': optimizer.steps,
            'theta': optimizer.theta.tolist()}


def evaluate(books, parent_factory, theta, *, warmup=64):
    """Paired exact metrics, per-book CE, ranking changes, no updates."""
    summaries = []
    for idx, data in enumerate(books):
        model = M3R38Predictor(parent_factory(), theta)
        totals = {arm: Counter() for arm in ('parent', 'r38')}
        flips = Counter()
        positions = 0
        for i, target in enumerate(data):
            if i >= warmup:
                base, feat = model.predict_parts()
                cand = corrected_logp(base, feat, model.theta)
                for arm, lp in (('parent', base), ('r38', cand)):
                    m = totals[arm]
                    bits = -float(lp[target])/LOG2
                    m['loss'] += bits
                    if target != 32:
                        m['nonspace_loss'] += bits
                    m['top1'] += int(int(np.argmax(lp)) == target)
                    m['top5'] += int(target in np.argpartition(lp, -5)[-5:])
                flips['top1'] += int(np.argmax(base) != np.argmax(cand))
                flips['top5'] += int(set(np.argpartition(base,-5)[-5:]) !=
                                     set(np.argpartition(cand,-5)[-5:]))
                positions += 1
            model.observe(int(target))
        if positions == 0:
            raise ValueError('book shorter than warmup')
        summaries.append({'index': idx, 'sha256': hashlib.sha256(data).hexdigest(),
                          'positions': positions, 'nonspace': sum(b != 32 for b in data[warmup:]),
                          'parent': dict(totals['parent']), 'r38': dict(totals['r38']),
                          'flips': dict(flips)})
    n = sum(x['positions'] for x in summaries)
    ns = sum(x['nonspace'] for x in summaries)
    out = {'books':len(summaries), 'positions':n, 'per_book':summaries}
    for arm in ('parent', 'r38'):
        agg = {k: sum(x[arm][k] for x in summaries) for k in ('loss','nonspace_loss','top1','top5')}
        out[arm] = {'bpb': agg['loss']/n, 'nonspace_bpb': agg['nonspace_loss']/max(ns,1),
                    'top1': agg['top1']/n, 'top5': agg['top5']/n}
    out['r38']['books_improved'] = sum(x['r38']['loss'] < x['parent']['loss'] for x in summaries)
    out['r38']['gain_bpb'] = out['parent']['bpb']-out['r38']['bpb']
    out['r38']['top1_changed'] = sum(x['flips']['top1'] for x in summaries)
    out['r38']['top5_changed'] = sum(x['flips']['top5'] for x in summaries)
    return out


def strict_gate(report, *, min_books_ratio=0.75):
    old, new = report['parent'], report['r38']
    return {'lower_bpb': new['bpb'] < old['bpb'],
            'lower_nonspace_bpb': new['nonspace_bpb'] <= old['nonspace_bpb'],
            'top1_not_worse': new['top1'] >= old['top1'],
            'top5_not_worse': new['top5'] >= old['top5'],
            'book_majority': new['books_improved'] >= int(np.ceil(report['books']*min_books_ratio)),
            'actual_rerank': new['top1_changed'] > 0 or new['top5_changed'] > 0}