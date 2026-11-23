#!/usr/bin/env python3
"""NaNCM v175: book-disjoint causal windows from one NaNROUTER Release.

NO labels are read from historically exposed DEV/TEST. The complete books are
streamed, fingerprinted, group-split before examples, and reduced to an exact
12k-vocabulary causal word dataset. No corpus data are committed to Git.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, tarfile
from collections import Counter
from pathlib import Path
import numpy as np

RE_WORD = re.compile(rb'[A-Za-z]+')
RE_ID = re.compile(r'(\d{4,7})--')
D = 48

def sha256(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def tar_members(path:Path):
    if path.suffix=='.zst':
        import zstandard
        with path.open('rb') as inp:
            with zstandard.ZstdDecompressor().stream_reader(inp) as raw:
                with tarfile.open(fileobj=raw,mode='r|') as tar:
                    for item in tar:
                        if item.isfile() and item.size>200 and item.size<12_000_000 and item.name.lower().endswith('.txt'):
                            f=tar.extractfile(item)
                            if f is not None:yield item.name,f.read()
    else:
        with tarfile.open(path,'r:*') as tar:
            for item in tar:
                if item.isfile() and 200<item.size<12_000_000 and item.name.lower().endswith('.txt'):
                    f=tar.extractfile(item)
                    if f is not None:yield item.name,f.read()

def summarize_archive(path:Path,vocab:dict,legacy_ids:set,max_books:int,per_book:int):
    book_rows=[];seen_hash=set();seen_id=set();counts=Counter();datasets={name:[[],[],[],[]] for name in ('train','val','test')}
    for name,raw in tar_members(path):
        counts['tar_text_files']+=1
        base=Path(name).name
        m=RE_ID.search(base)
        gid=m.group(1) if m else 'sha-'+sha256(raw)[:16]
        if gid in legacy_ids:
            counts['inherited_book_id_excluded']+=1;continue
        content_sha=sha256(raw)
        if content_sha in seen_hash or gid in seen_id:
            counts['duplicate_content_or_id_excluded']+=1;continue
        seen_hash.add(content_sha);seen_id.add(gid)
        if len(book_rows)>=max_books:break
        words=RE_WORD.findall(raw.lower());n=len(words)
        if n<1500:
            counts['too_short_excluded']+=1;continue
        # Deterministic group split; no context from any book appears across splits.
        h=int.from_bytes(hashlib.sha256(('v175-book-split-'+gid).encode()).digest()[:8],'little')
        split='train' if h%10<8 else 'val' if h%10==8 else 'test'
        idx=len(book_rows)
        sampled=0;sampled_oov=0
        # Spaced positions spread across the WHOLE book, never a single clip.
        starts=np.linspace(max(48,int(n*.02)),min(n-2,int(n*.98)),max(per_book*4,per_book+1),dtype=np.int64)
        used=set()
        X,Y,B,P=datasets[split]
        for pos0 in starts:
            pos=int(pos0)
            if pos in used:continue
            used.add(pos)
            target=words[pos].decode('ascii')
            y=vocab.get(target,-1)
            if y<0:
                sampled_oov+=1;continue
            hist=words[max(0,pos-D):pos]
            x=np.full((D,),-1,dtype=np.int16)
            for j,w in enumerate(hist[-D:]):x[D-len(hist)+j]=vocab.get(w.decode('ascii'),-1)
            if not np.any(x>=0):continue
            X.append(x);Y.append(y);B.append(idx);P.append(pos)
            sampled+=1
            if sampled==per_book:break
        if sampled<min(20,per_book//3):
            if sampled:
                del X[-sampled:];del Y[-sampled:];del B[-sampled:];del P[-sampled:]
            seen_hash.remove(content_sha);seen_id.remove(gid)
            counts['sparse_vocabulary_excluded']+=1;continue
        book_rows.append({'id':gid,'path':name,'sha256':content_sha,'split':split,'bytes':len(raw),
                          'tokens':n,'samples':sampled,'skipped_oov_targets':sampled_oov})
        counts['book_'+split]+=1;counts['samples_'+split]+=sampled
    return book_rows,datasets,counts

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',required=True,type=Path);p.add_argument('--vocab',type=Path,default=Path(__file__).with_name('vocab175.txt'))
    p.add_argument('--excluded',type=Path,default=Path(__file__).with_name('inherited_gutenberg_ids175.txt'))
    p.add_argument('--out',required=True,type=Path);p.add_argument('--max-books',type=int,default=500)
    p.add_argument('--per-book',type=int,default=240);p.add_argument('--release-tag',default='corpora-short-stories-0003')
    p.add_argument('--sha256',default=None)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    names=args.vocab.read_text().splitlines();assert len(names)==len(set(names))==12000
    vocab={name:i for i,name in enumerate(names)}
    inherited=set(args.excluded.read_text().splitlines())
    archive_sha=hashlib.file_digest(args.archive.open('rb'),'sha256').hexdigest()
    if args.sha256 is not None and archive_sha!=args.sha256:raise RuntimeError('Release archive sha256 mismatch')
    book_rows,dat,counts=summarize_archive(args.archive,vocab,inherited,args.max_books,args.per_book)
    # Require independent book groups and no inherited IDs before writing an artifact.
    seen={k:{b['id'] for b in book_rows if b['split']==k} for k in dat}
    assert all(not (seen[a]&seen[b]) for a in dat for b in dat if a!=b)
    assert not (set().union(*seen.values())&inherited)
    if min(map(len,seen.values()))<3:raise RuntimeError('Insufficient distinct books for book-disjoint validation/test')
    reports={}
    for split,(X,Y,B,P) in dat.items():
        a=np.stack(X).astype(np.int16);y=np.asarray(Y,dtype=np.int16)
        book=np.asarray(B,dtype=np.int16);idx=np.asarray(P,dtype=np.int32)
        np.savez_compressed(args.out/(split+'.npz'),X=a,Y=y,book=book,word_pos=idx)
        reports[split]={'books':len(seen[split]),'samples':len(y),'npz_sha256':sha256((args.out/(split+'.npz')).read_bytes()),
                        'known_history_pct':float((a>=0).mean()*100)}
    manifest={'version':'v175','source':'NaNROUTER GitHub Releases','release_tag':args.release_tag,
              'archive':args.archive.name,'archive_sha256':archive_sha,'vocab_sha256':sha256(args.vocab.read_bytes()),
              'excluded_ids_sha256':sha256(args.excluded.read_bytes()),'context_words':D,
              'sampling':'deterministic whole-book spaced positions, causal preceding 48 words, target in inherited 12k vocabulary',
              'book_split':'SHA256(v175-book-split-<Gutenberg ID>) modulo 10: train 0..7, val 8, test 9',
              'old_inherited_book_ids_excluded':len(inherited),'corpus_file_counts':dict(counts),'data':reports,
              'books':book_rows,'disjoint_checked':True,
              'caveat':'Book-disjoint for v175; inherited older checkpoints may contain other pretraining data. No pristine full-lineage guarantee.'}
    (args.out/'manifest175.json').write_text(json.dumps(manifest,indent=2))
    with (args.out/'SHA256SUMS').open('w') as sums:
        for p in sorted(args.out.glob('*')):
            if p.name!='SHA256SUMS':sums.write(f'{sha256(p.read_bytes())}  {p.name}\n')
    print('V175 CORPUS',json.dumps({'archive_sha256':archive_sha,'counts':counts,'sets':reports},default=int),flush=True)

if __name__=='__main__':main()
