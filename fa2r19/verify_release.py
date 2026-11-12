#!/usr/bin/env python3
"""Validate a FA2.1 release ZIP without deserializing its PyTorch checkpoint.

Known-good full R19 archive is required for publication; a new accepted
checkpoint must deliberately update the pinned digests and evaluation gate.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ARCHIVE_SHA256 = '0761209f51c7eef54e6e0cee6606d01ede4b2231b6dd1222e481e7dc7b20eef9'
CHECKPOINT_SHA256 = 'dd040106568786e7fbad2a880f4010635b626b51c3697e2ab08712905a2de91f'
CHECKPOINT_PATH = 'checkpoint/r12_active.pal12'
MAX_EXTRACTED_BYTES = 512 * 1024 * 1024


def sha256_stream(handle):
    digest = hashlib.sha256()
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def inspect_release(path, expected_archive=ARCHIVE_SHA256, expected_checkpoint=CHECKPOINT_SHA256):
    with path.open('rb') as stream:
        got_archive = sha256_stream(stream)
    if got_archive != expected_archive:
        raise ValueError(f'archive SHA-256 mismatch: {got_archive}')
    with zipfile.ZipFile(path) as archive:
        names = set()
        total = 0
        for item in archive.infolist():
            name = item.filename
            part = PurePosixPath(name)
            if (not name or name.startswith('/') or '\\' in name or ':' in part.parts[0]
                    or '..' in part.parts or name in names):
                raise ValueError(f'unsafe or duplicate entry: {name}')
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'archive symbolic link not allowed: {name}')
            names.add(name)
            total += item.file_size
        if total > MAX_EXTRACTED_BYTES:
            raise ValueError(f'archive expands beyond limit: {total}')
        ckpts = [n for n in names if n.startswith('checkpoint/') and n.endswith('.pal12')]
        if ckpts != [CHECKPOINT_PATH]:
            raise ValueError(f'expected one accepted checkpoint, found: {ckpts}')
        required = ('requirements.txt','source/r18_verify_lineage.py',
                    'source/test_r19.py','source/r19_semantic_frontier.py',
                    'tokenizer/gutenberg_r7.model')
        missing = [n for n in required if n not in names]
        if missing:
            raise ValueError(f'required release files missing: {missing}')
        with archive.open(CHECKPOINT_PATH) as checkpoint:
            got_checkpoint = sha256_stream(checkpoint)
        if got_checkpoint != expected_checkpoint:
            raise ValueError(f'checkpoint SHA-256 mismatch: {got_checkpoint}')
        return dict(valid=True, archive_sha256=got_archive,
                    checkpoint_sha256=got_checkpoint, entries=len(names),
                    uncompressed_bytes=total, checkpoint_path=CHECKPOINT_PATH)


def extract_release(archive_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError('refusing to extract into a nonempty directory')
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(dest)


def self_test():
    with tempfile.TemporaryDirectory() as t:
        zpath = Path(t) / 'test.zip'
        with zipfile.ZipFile(zpath,'w') as z:
            z.writestr(CHECKPOINT_PATH,b'accepted-one')
            for name in ('requirements.txt','source/r18_verify_lineage.py',
                         'source/test_r19.py','source/r19_semantic_frontier.py',
                         'tokenizer/gutenberg_r7.model'):
                z.writestr(name,b'x')
        with zpath.open('rb') as f:
            h = sha256_stream(f)
        ch = hashlib.sha256(b'accepted-one').hexdigest()
        assert inspect_release(zpath,h,ch)['valid']
        with zipfile.ZipFile(zpath,'a') as z:
            z.writestr('../escape.txt','unsafe')
        with zpath.open('rb') as f:
            h = sha256_stream(f)
        try:
            inspect_release(zpath,h,ch)
        except ValueError as e:
            assert 'unsafe' in str(e)
        else:
            raise AssertionError('path traversal was admitted')
    return {'self_test':'passed','checkpoint_deserialized':False}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('archive',nargs='?',type=Path)
    ap.add_argument('--extract',type=Path)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--report',type=Path)
    opts=ap.parse_args()
    if opts.self_test:
        outcome=self_test()
    else:
        if not opts.archive:
            ap.error('archive is required unless --self-test')
        outcome=inspect_release(opts.archive)
        if opts.extract:
            extract_release(opts.archive,opts.extract)
            outcome['extracted_to']=str(opts.extract)
    txt=json.dumps(outcome,sort_keys=True,indent=2)+'\n'
    if opts.report:
        opts.report.parent.mkdir(parents=True,exist_ok=True)
        opts.report.write_text(txt)
    print(txt,end='')

if __name__=='__main__':
    main()