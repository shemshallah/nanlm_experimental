#!/usr/bin/env python3
"""FA1 master default: promoted R7.64 with original unified optimizer history."""
from __future__ import annotations

from pathlib import Path

from fa1_r727_checkpoint import load_unified
from fa1_r764_multiresolution import ROUND

DEFAULT_CHECKPOINT_NAME = 'FA1_R764_PROMOTED_DEV_GATE.pt'
DEFAULT_ARCHITECTURE_ROUND = ROUND


def load_default(root, checkpoint=None, threads=2):
    """Load the ONE promoted FA1 model; explicit older paths remain usable."""
    target = (Path(checkpoint) if checkpoint is not None else
              Path(root) / 'checkpoint' / DEFAULT_CHECKPOINT_NAME)
    model, optimizer, base, comp, metadata = load_unified(
        target, root, threads)
    if (metadata.get('architecture_round') != ROUND or
            metadata.get('promoted') is not True or
            len(optimizer.param_groups) != 37):
        raise ValueError('FA1 default requires promoted R7.64 / 37 groups')
    return model, optimizer, base, comp, metadata
