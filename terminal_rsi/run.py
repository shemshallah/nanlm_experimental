"""Run a deterministic, isolated Terminal-RSI mechanics experiment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .standalone import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--sft-updates', type=int, default=200)
    parser.add_argument('--rpr-updates', type=int, default=100)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--candidate', type=Path,
                        help='Optional unpromoted actor+Adam+critic JSON; never commits/publishes it')
    args = parser.parse_args()
    result = run(seed=args.seed, sft_updates=args.sft_updates,
                 rpr_updates=args.rpr_updates, include_checkpoint=bool(args.candidate))
    candidate = result.pop('candidate_checkpoint', None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    if args.candidate:
        args.candidate.parent.mkdir(parents=True, exist_ok=True)
        args.candidate.write_text(json.dumps(candidate, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
