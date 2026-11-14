#!/usr/bin/env python3
"""Fail closed before R7 continuation can overwrite its checkpoint ancestor."""
from __future__ import annotations

from pathlib import Path


def validate_artifact_paths(checkpoint, output, candidate_checkpoint=''):
    """Reject aliased checkpoint/report/candidate destinations up front.

    Includes symlinks and existing hard links: a distinct filename is not
    sufficient protection for the accepted model or optimizer histories.
    """
    paths = {'source checkpoint': Path(checkpoint), 'report': Path(output)}
    if candidate_checkpoint:
        paths['candidate checkpoint'] = Path(candidate_checkpoint)
    entries = list(paths.items())
    for i, (left_name, left) in enumerate(entries):
        for right_name, right in entries[i+1:]:
            same = left.resolve() == right.resolve()
            if left.exists() and right.exists() and left.samefile(right):
                same = True
            if same:
                raise ValueError('R7 protected paths alias: ' +
                                 left_name + ' and ' + right_name)
