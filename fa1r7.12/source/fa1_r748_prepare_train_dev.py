#!/usr/bin/env python3
"""Reconstruct verified TRAIN and DEV from the archived R7.47 SHA split.

Does not read, tokenize, copy, or construct the sealed TEST corpus. The raw
NaNROUTER release is checked against each archived document's SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_document_corpus import RGX, longctx


def run(a):
    archive = Path(a.split_metadata)
    source = Path(a.raw_text)
    target = Path(a.output)
    if target.exists() and list(target.iterdir()):
        raise FileExistsError('never overwrite prepared split data')
    target.mkdir(parents=True, exist_ok=True)
    vocabulary = json.loads((Path(a.root) / 'corpus/vocab.json').read_text())
    lookup = {word: i for i, word in enumerate(vocabulary)}
    inventory = {p.name: p for p in source.glob('*.txt')}
    status = {}
    for split in ('train', 'dev'):
        metadata = json.loads((archive / f'{split}_documents.json').read_text())
        expected = metadata['documents']
        counters = {'books': 0, 'words': 0, 'oov': 0}
        with (target / f'{split}_temporal.i32').open('wb') as rec, \
             (target / f'{split}_longctx2048.i8').open('wb') as ctx, \
             (target / f'{split}_char_offsets.u32').open('wb') as offsets, \
             (target / f'{split}_char_blob.bin').open('wb') as charblob, \
             (target / f'{split}_doc_start.u32').open('wb') as docstart:
            charblob.write(b'\x00')
            spelling_offset = 1
            for doc in expected:
                original_basename = doc['relative_file'].split('__', 1)[-1]
                if original_basename not in inventory:
                    raise FileNotFoundError(original_basename)
                raw = inventory[original_basename].read_bytes()
                if hashlib.sha256(raw).hexdigest() != doc['sha256']:
                    raise ValueError(f'{split} raw-book SHA mismatch: {original_basename}')
                words = RGX.findall(raw.decode('utf8', errors='replace').lower())
                if len(words) != doc['words'] or doc['first_record'] != counters['words']:
                    raise ValueError(f'{split} token length or ordering mismatch')
                ids = np.asarray([lookup.get(word, 0) for word in words], dtype='<i4')
                rows = np.ones((len(ids), 7), dtype='<i4')
                rows[:, 3] = ids
                for lag in (1, 2, 3):
                    rows[lag:, 3 - lag] = ids[:-lag]
                    rows[:-lag, 3 + lag] = ids[lag:]
                rows.tofile(rec)
                longctx(ids).tofile(ctx)
                spelling = np.zeros(len(ids), dtype='<u4')
                for i in np.flatnonzero(ids == 0):
                    encoded = words[int(i)].encode('utf8') + b'\x00'
                    spelling[i] = spelling_offset
                    charblob.write(encoded)
                    spelling_offset += len(encoded)
                    counters['oov'] += 1
                spelling.tofile(offsets)
                np.full(len(ids), counters['words'], dtype='<u4').tofile(docstart)
                counters['words'] += len(ids)
                counters['books'] += 1
        if (counters['books'] != len(expected) or
                counters['words'] != metadata['total_word_targets'] or
                counters['oov'] != metadata['unknown_targets']):
            raise AssertionError('reconstructed document inventory differs from archive')
        shutil.copyfile(archive / f'{split}_documents.json',
                        target / f'{split}_documents.json')
        status[split] = counters
    print(json.dumps({'status': status, 'independent_test_opened': False}, indent=2))
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--split-metadata', required=True)
    parser.add_argument('--raw-text', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
