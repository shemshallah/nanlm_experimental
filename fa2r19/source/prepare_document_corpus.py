#!/usr/bin/env python3
"""Prepare new, *whole-document* word/character/long-context records for PAL R7.

Input: independent Gutenberg .txt files, NOT the old aggregated R6 train.txt.
Outputs are add-on data only; never rewrite the learned R6 vocabulary/tree/weights.
Split is by complete file (SHA-based, deduplicated), before tokenization.
"""
import argparse,hashlib,json,re,struct,pathlib,collections,sys
import numpy as np
from infer_unified_now import RGX

MAX_CONTEXT=2048

def mix(x):
    x=np.asarray(x,dtype=np.uint32);x=x^(x>>16);x=(x*np.uint32(0x7feb352d));x=x^(x>>15);x=(x*np.uint32(0x846ca68b));return x^(x>>16)

def longctx(seq):
    n=len(seq);seq=np.asarray(seq,dtype=np.uint32)
    out=np.zeros((n,72),dtype=np.int8)
    for band in range(9):
        first=4<<band;last=(8<<band)-1
        if n<first:continue
        h=mix(seq^np.uint32((0x9e3779b9*(band+1))&0xffffffff))
        buck=h&7;sign=np.where((h&8)!=0,-1,1).astype(np.int32)
        ix=np.arange(n,dtype=np.int64)
        hi=ix-first+1;lo=np.maximum(ix-last,0)
        factor=16/np.sqrt(last-first+1)
        available=hi>0
        for j in range(8):
            v=np.where(buck==j,sign,0)
            prefix=np.r_[0,np.cumsum(v,dtype=np.int64)]
            counts=prefix[np.maximum(hi,0)]-prefix[lo]
            v=counts*factor
            features=(np.sign(v)*np.floor(np.abs(v)+.5)).clip(-127,127).astype(np.int8)
            features[~available]=0
            out[:,band*8+j]=features
    return out

def prepare(root,indir,outdir,limit=None):
    root=pathlib.Path(root);indir=pathlib.Path(indir);outdir=pathlib.Path(outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    vocab=json.loads((root/'corpus/vocab.json').read_text());lookup={w:i for i,w in enumerate(vocab)}
    files=sorted(p for p in indir.rglob('*.txt') if p.is_file())
    if limit is not None:files=files[:limit]
    if not files:raise FileNotFoundError('no .txt documents found under '+str(indir))
    seen=set();unique=[]
    for path in files:
        raw=path.read_bytes();hashval=hashlib.sha256(raw).hexdigest()
        if hashval in seen:continue
        seen.add(hashval);unique.append((path,raw,hashval))
    # Stable given this input inventory. Split whole books by sorted content hash,
    # so new books cannot leak passages between training and evaluation.
    unique.sort(key=lambda item:item[2]);nfiles=len(unique)
    if nfiles<3:raise ValueError('at least three distinct books required for train/dev/test')
    ntest=max(1,round(nfiles*.05));ndev=max(1,round(nfiles*.05))
    assigned={'train':unique[:nfiles-ntest-ndev],
              'dev':unique[nfiles-ntest-ndev:nfiles-ntest],
              'test':unique[nfiles-ntest:]}
    for part,items in assigned.items():
        # Multiprocessing-safe fixed binary layout, with explicit document start offsets.
        with (outdir/f'{part}_temporal.i32').open('wb') as rec, (outdir/f'{part}_longctx2048.i8').open('wb') as ctx, (outdir/f'{part}_char_offsets.u32').open('wb') as offs, (outdir/f'{part}_char_blob.bin').open('wb') as spell, (outdir/f'{part}_doc_start.u32').open('wb') as starts, (outdir/f'{part}.txt').open('wb') as textfile:
            spell.write(b'\x00');spelloff=1;wordcount=0;oov=0
            docmeta=[]
            for path,raw,digest in items:
                original=raw.decode('utf8',errors='replace')
                words=RGX.findall(original.lower());n=len(words)
                if not n:continue
                ids=np.array([lookup.get(w,0) for w in words],dtype='<i4')
                # The first and last three words have BOS padding, but intervening
                # source linebreaks do NOT erase history within a document.
                e=np.ones((n,7),dtype='<i4');e[:,3]=ids
                for lag in (1,2,3):
                    e[lag:,3-lag]=ids[:-lag]
                    e[:-lag,3+lag]=ids[lag:]
                e.tofile(rec);longctx(ids).tofile(ctx)
                arr=np.zeros(n,dtype='<u4')
                for j,w in enumerate(words):
                    if ids[j]==0:
                        b=w.encode('utf8')+b'\x00'
                        if spelloff+len(b)>=2**32:raise ValueError('character blob exceeds uint32')
                        arr[j]=spelloff;spell.write(b);spelloff+=len(b);oov+=1
                arr.tofile(offs)
                np.full(n,wordcount,dtype='<u4').tofile(starts)
                textfile.write(raw.rstrip(b'\r\n')+b'\n\n')
                docmeta.append({'relative_file':str(path.relative_to(indir)),'sha256':digest,'words':n,'first_record':wordcount})
                wordcount+=n
        (outdir/f'{part}_documents.json').write_text(json.dumps({'documents':docmeta,'source':'individual Gutenberg text files','split_policy':'whole-file content hash sort, 90/5/5 approximate','total_word_targets':wordcount,'unknown_targets':oov},indent=2))
    print(json.dumps({part:{'documents':len(items),'files':sum(1 for _ in items)} for part,items in assigned.items()},indent=2))
    return assigned

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();prepare(a.root,a.input,a.out,a.limit)