"""Five-seed synthetic stress check; never reports Terminal-Bench capability."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

from .standalone import run


def compare(seeds=range(2026, 2031), sft_updates=200, rpr_updates=100):
    results = []
    for seed in seeds:
        report = run(seed=seed, sft_updates=sft_updates, rpr_updates=rpr_updates)
        before, after = report['stages'][0]['heldout'], report['stages'][-1]['heldout']
        results.append(dict(seed=seed, before=before, after=after,
                            exact_success_change=round(after['top1'] - before['top1'], 6),
                            bits_per_action_change=round(after['bits_per_action'] - before['bits_per_action'], 6)))
    if not results:
        raise ValueError('Need one or more deterministic seeds')
    return dict(schema='terminal-rsi-standalone-synthetic-multiseed-v1',
                results=results, seeds=len(results),
                heldout_task_count_per_seed=results[0]['before']['tasks'],
                mean_success_before=round(statistics.mean(r['before']['top1'] for r in results), 6),
                mean_success_after=round(statistics.mean(r['after']['top1'] for r in results), 6),
                mean_bits_per_action_before=round(statistics.mean(r['before']['bits_per_action'] for r in results), 6),
                mean_bits_per_action_after=round(statistics.mean(r['after']['bits_per_action'] for r in results), 6),
                benchmark_eligible=False, candidate_promoted=False,
                conclusion_scope='Synthetic family-shared task generalization only; no independent terminal benchmark',
                warning='Neither stage gains nor seed averages isolate recursive learning from extra training updates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-seed', type=int, default=2026)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--sft-updates', type=int, default=200)
    parser.add_argument('--rpr-updates', type=int, default=100)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 20:
        parser.error('Seed count must be 1..20')
    report = compare(range(args.start_seed, args.start_seed+args.seeds),
                     args.sft_updates, args.rpr_updates)
    data = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data)
    print(data)


if __name__ == '__main__':
    main()
