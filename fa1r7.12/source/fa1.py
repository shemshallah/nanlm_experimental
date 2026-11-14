#!/usr/bin/env python3
"""One FA1 continuation entry point: inherited R7 model, current full stack.

This is not a new architecture, checkpoint format, or optimizer. The original
training/evaluation implementation remains in fa1_r728_geometric_memory; the
canonical CLI selects all in-place extensions instead of silently defaulting
to the older pre-layered geometry mode. No checkpoint is promoted here.
"""

from fa1_r728_geometric_memory import parse_args, run


def canonical_args():
    args = parse_args()
    # The loaders attach missing modules onto the same inherited model with
    # zero/identity initialization, preserving original weights and moments.
    args.contextual = True
    args.factorized = True
    args.layered = True
    args.calibrated = True
    args.matched_far_history = True
    args.apex_feedback = True
    args.history_value_writer = True
    return args


if __name__ == '__main__':
    run(canonical_args())
