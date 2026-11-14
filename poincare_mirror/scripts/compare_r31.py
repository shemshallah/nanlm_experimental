#!/usr/bin/env python3
"""Same-position development gate. Test books are never consulted here."""
import argparse
import json
import math
from pathlib import Path


def compare(before, after, min_gain):
    for field in ('split', 'evaluation_context', 'evaluated_tokens'):
        if before[field] != after[field]:
            raise ValueError(f'Non-comparable evaluation: {field}')
    # New corpus reports carry exact input identities; refuse silently
    # comparing a different book set or nonmatching sampling schedule.
    for field in ('data_split_sha256','evaluation_seed','exhaustive','eval_stride'):
        if field in before or field in after:
            # `exhaustive=False` is a valid explicit sampling protocol.
            if field not in before or field not in after or before[field] != after[field]:
                raise ValueError(f'Non-comparable evaluation: {field}')
    if before['split'] != 'valid':
        raise ValueError('Acceptance decisions must only use development books')
    if before['evaluated_tokens'] < 1024:
        raise ValueError('Too few held-out positions for acceptance')
    b = before['results']['base']; a = after['results']['base']
    metrics = {}
    for name in ('ce_1', 'ce_2', 'ce_3', 'acc_1'):
        if not math.isfinite(float(b[name])) or not math.isfinite(float(a[name])):
            raise ValueError(f'Nonfinite metric: {name}')
        metrics[name] = {'before':float(b[name]), 'after':float(a[name]),
                         'improvement':float(b[name]) - float(a[name])}
    accepted = (metrics['ce_1']['improvement'] > min_gain
                and metrics['ce_2']['improvement'] > min_gain
                and metrics['ce_3']['improvement'] > -0.005
                and metrics['acc_1']['improvement'] < 0.01)
    return {'promotable_on_development': accepted,
            'checkpoint_before':before['checkpoint_step'],
            'checkpoint_after':after['checkpoint_step'],
            'evaluated_tokens':before['evaluated_tokens'],
            'context':before['evaluation_context'],
            'minimum_ce_improvement':min_gain, 'metrics':metrics,
            'warning':'Development-only eligibility, not frontier quality or a final acceptance decision'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum-improvement',type=float,default=0.0001)
    args=parser.parse_args()
    report=compare(json.loads(args.before.read_text()),
                   json.loads(args.after.read_text()),args.minimum_improvement)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
