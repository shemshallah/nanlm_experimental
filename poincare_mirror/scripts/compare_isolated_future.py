#!/usr/bin/env python3
"""Strict matched-position gate for a forecast-only continuation.

Inputs are JSON reports from scripts/eval_external_bytes.py. Use book-disjoint
development data for checkpoint selection; a reserved external test is for
one final, explicitly labeled evaluation, never repeated candidate selection.
"""
import argparse
import json
import math
from pathlib import Path


MATCH_FIELDS = ('test_sha256', 'target_positions_sha256', 'test_positions',
                'context_bytes', 'seed')
METRICS = ('ce_nats_per_byte', 'bits_per_byte', 'top1', 'top5')


def compare(before, after, min_ce_gain=0.001, max_now_drift=0.00002):
    if min_ce_gain < 0 or max_now_drift < 0:
        raise ValueError('Gate tolerances must be nonnegative')
    for field in MATCH_FIELDS:
        if field not in before or field not in after or before[field] != after[field]:
            raise ValueError(f'Unmatched evaluation identity: {field}')
    if after['checkpoint_step'] <= before['checkpoint_step']:
        raise ValueError('Candidate did not advance the checkpoint step')
    result = {'baseline_step':before['checkpoint_step'],
              'candidate_step':after['checkpoint_step'],
              'matched':{field:before[field] for field in MATCH_FIELDS},
              'horizons':{}, 'eligible':False}
    passes = []
    for horizon in ('+1','+2','+3'):
        original=before['results'][horizon]
        candidate=after['results'][horizon]
        for metric in METRICS:
            if not math.isfinite(original[metric]) or not math.isfinite(candidate[metric]):
                raise ValueError(f'Nonfinite {horizon} {metric}')
        gain = original['ce_nats_per_byte'] - candidate['ce_nats_per_byte']
        accuracy_ok = all(candidate[key] >= original[key] - 1e-9
                          for key in ('top1','top5'))
        accepted = (gain >= -max_now_drift if horizon == '+1'
                    else gain >= min_ce_gain) and accuracy_ok
        result['horizons'][horizon] = {
            'ce_nats_gain':round(gain,6),
            'bpb_gain':round(original['bits_per_byte']-candidate['bits_per_byte'],6),
            'top1_gain':round(candidate['top1']-original['top1'],6),
            'top5_gain':round(candidate['top5']-original['top5'],6),
            'passes':accepted}
        passes.append(accepted)
    result['eligible']=all(passes)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    p.add_argument('--min-future-ce-gain',type=float,default=.001)
    p.add_argument('--max-now-ce-drift',type=float,default=.00002)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=compare(json.loads(args.before.read_text()),
                   json.loads(args.after.read_text()),
                   args.min_future_ce_gain,args.max_now_ce_drift)
    output=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output)
    print(output,end='')
    raise SystemExit(0 if result['eligible'] else 2)


if __name__=='__main__':
    main()
