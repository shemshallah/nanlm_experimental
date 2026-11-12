#!/usr/bin/env python3
"""Fetch pinned NaNROUTER GitHub Release Gutenberg batch, verify SHA256, extract safely.

Use on a networked machine. Current sandbox has no outbound release-asset access.
Default points to the independently inspected NaNROUTER short-story batch #1.
"""
import argparse,hashlib,pathlib,tarfile,urllib.request,shutil,subprocess,sys,os
URL='https://github.com/shemshallah/NaNROUTER/releases/download/corpora-short-stories-0001/short_stories-0001.tar.zst'
SHA='bceb6ef83ffa89ccb257bf55d9a5ad0e56a70d2b7bfd54599bacd26dd5beb2a5'

def download(url,out,expected):
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PAL-R7-audited-corpus-importer'}),timeout=120) as src,out.open('wb') as f:
        for chunk in iter(lambda:src.read(1024*1024),b''):
            size+=len(chunk);f.write(chunk);h.update(chunk)
    got=h.hexdigest()
    if got.lower()!=expected.lower():
        out.unlink(missing_ok=True);raise ValueError(f'download hash mismatch expected {expected} got {got}')
    print(f'RELEASE_VERIFIED bytes={size} sha256={got}',flush=True)

def extract_safely(path,out,max_uncompressed=1_500_000_000):
    out.mkdir(parents=True,exist_ok=True)
    # streaming zstd: prefer system zstd, then optional Python zstandard
    process=None;stream=None
    if shutil.which('zstd'):
        process=subprocess.Popen(['zstd','-dc',str(path)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stream=process.stdout
    else:
        try:import zstandard
        except ImportError as e:raise RuntimeError('Install zstd CLI or pip install zstandard') from e
        fh=path.open('rb');stream=zstandard.ZstdDecompressor().stream_reader(fh)
    count=0;total=0
    try:
        with tarfile.open(fileobj=stream,mode='r|') as tf:
            for member in tf:
                if not member.isfile():
                    if member.isdir():continue
                    raise ValueError('unsafe nonregular archive entry '+member.name)
                pure=pathlib.PurePosixPath(member.name)
                if pure.is_absolute() or '..' in pure.parts:raise ValueError('archive traversal rejected: '+member.name)
                if member.size<0 or total+member.size>max_uncompressed:raise ValueError('archive uncompressed-size budget exceeded')
                total+=member.size
                # Keep files inside one dedicated extraction root. Do not follow symlinks.
                dest=out.joinpath(*pure.parts)
                dest.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(member) as src,dest.open('wb') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)
                count+=1
    finally:
        try:stream.close()
        except Exception:pass
        if process is not None:
            ret=process.wait()
            if ret:raise RuntimeError(f'zstd exited nonzero {ret}')
    print(f'EXTRACTED regular_files={count} bytes={total} into={out}',flush=True)
    return count

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default=URL);parser.add_argument('--sha256',default=SHA)
    parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path('nanrouter_release_short_stories_0001'))
    parser.add_argument('--skip-download',action='store_true')
    args=parser.parse_args();archive=args.output/'short_stories-0001.tar.zst';archive.parent.mkdir(parents=True,exist_ok=True)
    if not args.skip_download:download(args.url,archive,args.sha256)
    else:
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=args.sha256:raise ValueError('local archive hash mismatch')
    extract_safely(archive,args.output/'books')
