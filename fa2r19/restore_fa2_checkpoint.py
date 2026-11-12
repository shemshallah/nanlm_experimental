#!/usr/bin/env python3
"""Reassemble FA2.1 accepted checkpoint from GitHub committed binary parts.

Only uses Python stdlib. Verifies every part, compressed stream, and restored
checkpoint digest before atomically installing weights. No model reinitialization.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore(root, overwrite=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'MODEL_PARTS.json').read_text())
    target = root / manifest['source_checkpoint']
    if target.exists():
        if sha256(target) == manifest['source_checkpoint_sha256']:
            print('Accepted checkpoint already present and valid:', target)
            return target
        if not overwrite:
            raise FileExistsError(f'{target} exists but has a different hash; pass --overwrite to replace')
    target.parent.mkdir(parents=True, exist_ok=True)
    decompressor = zlib.decompressobj()
    packed_hash = hashlib.sha256()
    packed_bytes = 0
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.fa2checkpoint-', delete=False) as temp:
        tmp = Path(temp.name)
        try:
            for entry in manifest['parts']:
                part = root / entry['path']
                if not part.is_file():
                    raise FileNotFoundError(part)
                if part.stat().st_size != entry['length'] or sha256(part) != entry['sha256']:
                    raise ValueError(f'Corrupt or wrong checkpoint part: {part}')
                with part.open('rb') as f:
                    for buf in iter(lambda: f.read(1024 * 1024), b''):
                        packed_hash.update(buf)
                        packed_bytes += len(buf)
                        temp.write(decompressor.decompress(buf))
            temp.write(decompressor.flush())
            temp.flush()
            os.fsync(temp.fileno())
            if not decompressor.eof or decompressor.unused_data:
                raise ValueError('Compressed stream is incomplete or has extra bytes')
            if packed_bytes != manifest['compressed_bytes'] or packed_hash.hexdigest() != manifest['compressed_sha256']:
                raise ValueError('Compressed stream failed SHA-256 verification')
            if tmp.stat().st_size != manifest['source_checkpoint_bytes'] or sha256(tmp) != manifest['source_checkpoint_sha256']:
                raise ValueError('Restored checkpoint failed SHA-256 verification')
            os.replace(tmp, target)
            print('Restored FA2.1 unified checkpoint:', target)
            print('SHA-256:', sha256(target))
            return target
        finally:
            if tmp.exists():
                tmp.unlink()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    restore(args.root, overwrite=args.overwrite)