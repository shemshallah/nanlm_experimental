#!/usr/bin/env python3
"""Reserve a new, hash-disjoint NaNROUTER book set for R7.29.

This reserves previously unused source files. Their SHA hashes, not path
names, exclude books consumed by the entire R7.28 train/dev/test experiment.
The resulting symlink folder is only local input to prepare_document_corpus.
No original corpus files are overwritten or removed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def select(source, previous, destination, limit):
    old=set()
    for split in ('train','dev','test'):
        docs=json.loads((previous/(split+'_documents.json')).read_text())['documents']
        old.update(d['sha256'] for d in docs)
    files=[]
    for p in source.rglob('*.txt'):
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest not in old:
            files.append((digest,p))
    files.sort(key=lambda x:x[0])
    unique={}
    for sha,path in files:
        unique.setdefault(sha,path)
    chosen=list(unique.items())[:limit]
    if len(chosen)<limit or len(chosen)<40:
        raise ValueError('not enough new unused books for independent evaluation')
    destination.mkdir(parents=True,exist_ok=True)
    for index,(sha,path) in enumerate(chosen):
        dst=destination/(f'{index:04d}_{sha[:12]}.txt')
        if dst.is_symlink():
            if dst.resolve()!=path.resolve():
                raise ValueError('existing reservation points to a different book')
        elif dst.exists():
            raise FileExistsError('refusing to replace reserved path '+str(dst))
        else:
            dst.symlink_to(path.resolve())
    return {'new_books':len(chosen),'previous_book_hashes':len(old),
            'new_source_sha256':[sha for sha,_ in chosen],
            'all_previous_sha_disjoint':all(sha not in old for sha,_ in chosen)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--limit',type=int,default=300)
    a=p.parse_args()
    manifest=select(a.source,a.previous,a.destination,a.limit)
    a.manifest.parent.mkdir(parents=True,exist_ok=True)
    if a.manifest.exists():
        if json.loads(a.manifest.read_text())!=manifest:
            raise FileExistsError('reservation manifest changed')
    else:
        a.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='new_source_sha256'}),flush=True)
