#!/usr/bin/env python3
"""Recreate verified R7.57 TRAIN/DEV arrays from NaNROUTER batch4; TEST sealed."""
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path
from prepare_document_corpus import prepare

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def restore(archive,root,metadata,destination):
    archive,root,metadata,destination=map(Path,(archive,root,metadata,destination))
    if destination.exists():raise FileExistsError(destination)
    expected={p:json.loads((metadata/(p+'_documents.json')).read_text())
              for p in ('train','dev')}
    if [len(expected[p]['documents']) for p in ('train','dev')]!=[270,15]:
        raise ValueError('not the original R7.57 book-disjoint split')
    members=subprocess.check_output(['tar','--zstd','-tf',str(archive.resolve())],text=True).splitlines()
    lookup={Path(m).name:m for m in members if m.startswith('./') and
            Path(m).name==m[2:] and m.endswith('.txt') and '..' not in Path(m).parts}
    selected=[]
    for part in ('train','dev'):
        for row in expected[part]['documents']:
            prefix,name=Path(row['relative_file']).name.split('__',1)
            if prefix!=row['sha256'][:16] or name not in lookup:
                raise ValueError('source book not found: '+name)
            selected.append((row,lookup[name]))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fa1_r758_',dir=destination.parent) as tmp:
        work=Path(tmp)
        files=work/'members.txt'
        files.write_text(''.join(member+'\n' for _,member in selected))
        raw=work/'raw';raw.mkdir()
        subprocess.run(['tar','--zstd','-xf',str(archive.resolve()),
                        '-C',str(raw),'-T',str(files)],check=True)
        inputs=work/'inputs'
        for row,member in selected:
            path=raw/member[2:]
            if not path.is_file() or sha(path)!=row['sha256']:
                raise AssertionError('book checksum mismatch: '+member)
            out=inputs/row['relative_file']
            out.parent.mkdir(parents=True,exist_ok=True)
            path.rename(out)
        print('TRAIN=270 DEV=15 verified; independent TEST unopened',flush=True)
        processed=work/'processed'
        prepare(root,inputs,processed,presplit=True,splits=('train','dev'))
        for part in ('train','dev'):
            rebuilt=json.loads((processed/(part+'_documents.json')).read_text())
            if rebuilt!=expected[part]:
                raise AssertionError('processed identity differs from '+part)
        processed.rename(destination)
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','root','metadata','destination'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    print(restore(args.archive,args.root,args.metadata,args.destination))
