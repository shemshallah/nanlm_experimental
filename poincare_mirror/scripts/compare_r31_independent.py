#!/usr/bin/env python3
"""Fail-closed independent-book + aligned long-context gate for one checkpoint.

Test books are evaluated only AFTER model selection on development books;
this report does not promote or change any checkpoint by itself.
"""
import argparse
import json
import math
from pathlib import Path


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite '+label)
    return result


def compare(development, test_before, test_after, long_before, long_after,
            max_aux_regression=.005, max_long_regression_bpb=.005):
    if development.get('promotable_on_development') is not True:
        raise ValueError('Development selection gate has not passed')
    for label, before, after in [('test',test_before,test_after),
                                 ('long_context',long_before,long_after)]:
        if before.get('split') != ('test' if label=='test' else 'valid') or after.get('split') != before['split']:
            raise ValueError('Wrong held-out split for '+label)
        for field in ('data_split_sha256',):
            if not before.get(field) or before[field] != after.get(field):
                raise ValueError('Nonidentical books in '+label+': '+field)
        if len(before['data_split_sha256']) != 64:
            raise ValueError('Missing full split SHA-256 in '+label)
    for field in ('evaluation_context','evaluated_tokens','evaluation_seed','exhaustive','eval_stride'):
        if test_before.get(field) != test_after.get(field):
            raise ValueError('Test positions not comparable: '+field)
    if not test_before.get('exhaustive') or test_before['evaluated_tokens'] < 2048:
        raise ValueError('Independent test needs at least 2048 deterministic tokens')
    if test_before['checkpoint_step'] != development['checkpoint_before'] or test_after['checkpoint_step'] != development['checkpoint_after']:
        raise ValueError('Test did not compare the development-selected checkpoints')
    for field in ('seed','windows','short','long','same_target_positions'):
        if long_before.get(field) != long_after.get(field):
            raise ValueError('Misaligned context comparison: '+field)
    if (long_before['same_target_positions'] < 1024 or
            long_before['same_target_positions'] != long_before['short']*long_before['windows'] or
            long_before['long'] <= long_before['short']):
        raise ValueError('Insufficient aligned long-context evidence')
    if long_before['step'] != development['checkpoint_before'] or long_after['step'] != development['checkpoint_after']:
        raise ValueError('Long-context checkpoints do not match development gate')
    test_metrics = {}
    for field in ('ce_1','ce_2','ce_3','acc_1'):
        b = finite(test_before['results']['base'][field],'test before '+field)
        a = finite(test_after['results']['base'][field],'test after '+field)
        test_metrics[field] = {'before':b,'after':a,'improvement':b-a}
    short,long = str(long_before['short']),str(long_before['long'])
    long_metrics = {}
    for context in (short,long):
        b = finite(long_before['context_comparison'][context]['bpb'],'context before '+context)
        a = finite(long_after['context_comparison'][context]['bpb'],'context after '+context)
        long_metrics[context] = {'before_bpb':b,'after_bpb':a,'improvement_bpb':b-a}
    before_penalty = long_metrics[long]['before_bpb']-long_metrics[short]['before_bpb']
    after_penalty = long_metrics[long]['after_bpb']-long_metrics[short]['after_bpb']
    test_pass = (test_metrics['ce_1']['improvement'] > 0 and
                 test_metrics['ce_2']['improvement'] >= -max_aux_regression and
                 test_metrics['ce_3']['improvement'] >= -max_aux_regression and
                 test_metrics['acc_1']['improvement'] <= .01)
    long_pass = (long_metrics[long]['improvement_bpb'] >= -max_long_regression_bpb and
                 after_penalty <= before_penalty+max_long_regression_bpb)
    return {'development_pass':True,'independent_test_pass':test_pass,
            'long_context_nonregression_pass':long_pass,
            'eligible_for_manual_promotion':bool(test_pass and long_pass),
            'frontier_context_advantage_demonstrated':bool(after_penalty <= 0),
            'checkpoint_before':development['checkpoint_before'],
            'checkpoint_after':development['checkpoint_after'],
            'test_positions':test_before['evaluated_tokens'],
            'aligned_positions':long_before['same_target_positions'],
            'test_metrics':test_metrics,'context_metrics':long_metrics,
            'before_long_minus_short_bpb':round(before_penalty,6),
            'after_long_minus_short_bpb':round(after_penalty,6),
            'warning':'Manual review required. No automatic checkpoint promotion or frontier claim.'}


def main():
    ap=argparse.ArgumentParser()
    for name in ('development','test-before','test-after','long-before','long-after','output'):
        ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args()
    report=compare(*(json.loads(getattr(args,name).read_text()) for name in
                    ('development','test_before','test_after','long_before','long_after')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
