"""Book-disjoint causal training/evaluation for a frozen-parent R38 adapter.

The caller MUST provide books from NaNROUTER Release with a prelocked split.
The factory is a callable constructing a *fresh identical frozen parent* for
one book; never cache document-local state across books. Independent TEST is
not used for tuning.
"""
from __future__ import annotations
import hashlib
from collections import Counter
import numpy as np
from .episodic import CausalSuccessors, M3R38Predictor, corrected_logp, nll_grad, GROUPS

LOG2 = float(np.log(2.0))
ARMS = ('parent','r38','wrong_successors','unrelated_history')


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


def evaluate(books, parent_factory, theta, *, warmup=64, bootstrap_seed=38):
    """Paired causal metrics, per-book loss, wrong-values and unrelated-history nulls."""
    if len(books) < 2:
        raise ValueError('null controls require at least two distinct books')
    if len({hashlib.sha256(x).hexdigest() for x in books}) != len(books):
        raise ValueError('evaluation books contain duplicate bytes')
    summaries = []
    for idx, data in enumerate(books):
        other = books[(idx+1) % len(books)]
        model = M3R38Predictor(parent_factory(), theta)
        unrelated = CausalSuccessors()
        totals = {arm: Counter() for arm in ARMS}
        flips = Counter()
        positions = 0
        for i, target in enumerate(data):
            if i >= warmup:
                base, feat = model.predict_parts()
                predictions = {
                    'parent': base,
                    'r38': corrected_logp(base, feat, model.theta),
                    'wrong_successors': corrected_logp(
                        base, model.memory.features(base, wrong_values=True), model.theta),
                    'unrelated_history': corrected_logp(
                        base, unrelated.features(base), model.theta)}
                for arm, lp in predictions.items():
                    m = totals[arm]
                    bits = -float(lp[target])/LOG2
                    m['loss'] += bits
                    if target != 32:
                        m['nonspace_loss'] += bits
                    m['top1'] += int(int(np.argmax(lp)) == target)
                    m['top5'] += int(target in np.argpartition(lp, -5)[-5:])
                cand = predictions['r38']
                flips['top1'] += int(np.argmax(base) != np.argmax(cand))
                flips['top5'] += int(set(np.argpartition(base,-5)[-5:]) !=
                                     set(np.argpartition(cand,-5)[-5:]))
                positions += 1
            # Update current and null histories AFTER scoring the current target.
            model.observe(int(target))
            if i < len(other):
                unrelated.observe(int(other[i]))
        if positions == 0:
            raise ValueError('book shorter than warmup')
        summaries.append({'index': idx, 'sha256': hashlib.sha256(data).hexdigest(),
                          'unrelated_sha256': hashlib.sha256(other).hexdigest(),
                          'positions': positions,
                          'nonspace': sum(b != 32 for b in data[warmup:]),
                          **{arm: dict(totals[arm]) for arm in ARMS},
                          'flips': dict(flips)})
    n = sum(x['positions'] for x in summaries)
    ns = sum(x['nonspace'] for x in summaries)
    out = {'books':len(summaries), 'positions':n, 'per_book':summaries}
    for arm in ARMS:
        agg = {k: sum(x[arm][k] for x in summaries)
               for k in ('loss','nonspace_loss','top1','top5')}
        out[arm] = {'bpb': agg['loss']/n,
                    'nonspace_bpb': agg['nonspace_loss']/max(ns,1),
                    'top1': agg['top1']/n, 'top5': agg['top5']/n}
    new = out['r38']
    new['books_improved'] = sum(x['r38']['loss'] < x['parent']['loss'] for x in summaries)
    new['gain_bpb'] = out['parent']['bpb']-new['bpb']
    new['top1_changed'] = sum(x['flips']['top1'] for x in summaries)
    new['top5_changed'] = sum(x['flips']['top5'] for x in summaries)
    # Paired book-level bootstrap. Descriptive on previously opened DEV only.
    rng = np.random.default_rng(bootstrap_seed)
    num = len(summaries)
    gains = np.array([x['parent']['loss']-x['r38']['loss'] for x in summaries])
    denom = np.array([x['positions'] for x in summaries])
    draws = rng.integers(0,num,size=(4000,num))
    resampled = gains[draws].sum(axis=1) / denom[draws].sum(axis=1)
    new['whole_book_bootstrap_95pct_bits'] = np.quantile(
        resampled,[.025,.975]).tolist()
    new['bootstrap_is_descriptive'] = True
    return out


def strict_gate(report, *, min_books_ratio=0.75):
    old, new = report['parent'], report['r38']
    return {'lower_bpb': new['bpb'] < old['bpb'],
            'lower_nonspace_bpb': new['nonspace_bpb'] <= old['nonspace_bpb'],
            'top1_not_worse': new['top1'] >= old['top1'],
            'top5_not_worse': new['top5'] >= old['top5'],
            'book_majority': new['books_improved'] >= int(np.ceil(report['books']*min_books_ratio)),
            'positive_book_bootstrap': new['whole_book_bootstrap_95pct_bits'][0] > 0,
            'beats_wrong_successors': new['bpb'] < report['wrong_successors']['bpb'],
            'beats_unrelated_history': new['bpb'] < report['unrelated_history']['bpb'],
            'actual_rerank': new['top1_changed'] > 0 or new['top5_changed'] > 0}