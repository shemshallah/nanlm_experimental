#!/usr/bin/env python3
"""Verified, leakage-guarded importer for NaNROUTER Actions raw-prose books.

Does not randomize or merge the frozen train/dev/test Gutenberg-ID splits.
The optional test.bin is NEVER consumed by the model's training code.
"""
import argparse, hashlib, json, sys, zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import TokenCodec, token_dtype, file_hash


def import_archive(archive, outdir, require_counts=False):
    archive, outdir = Path(archive), Path(outdir)
    with zipfile.ZipFile(archive) as z:
        names=set(z.namelist())
        manifest=json.loads(z.read('MANIFEST.json'))
        lines=z.read('SHA256SUMS').decode('ascii').splitlines()
        checks={}
        for line in lines:
            sha,name=line.split(None,1)
            name=name.strip()
            if name in checks: raise ValueError('Duplicate checksum entry: '+name)
            checks[name]=sha.lower()
        actual={n for n in names if n.endswith('.txt') and n.startswith(('train/','dev/','test/'))}
        if actual!=set(checks):
            raise ValueError(f'Archive/checksum listing differs: missing={set(checks)-actual} extra={actual-set(checks)}')
        documents=manifest['documents']
        mapping={f"{r['split']}/{r['gutenberg_id']}.txt":r for r in documents}
        if actual!=set(mapping): raise ValueError('Manifest book listing differs from archive')
        if require_counts and (len(actual)!=38 or any(len([p for p in actual if p.startswith(x+'/')])!=n for x,n in [('train',12),('dev',12),('test',14)])):
            raise ValueError('Unexpected frozen R11 sample size')
        contents={}
        seen_ids=set()
        seen_sha={}
        for name in sorted(actual):
            data=z.read(name)
            digest=hashlib.sha256(data).hexdigest()
            if digest!=checks[name] or digest!=mapping[name]['sha256']:
                raise ValueError('Failed SHA256 validation: '+name)
            if len(data)!=mapping[name]['bytes']:
                raise ValueError('Byte count changed: '+name)
            book=int(mapping[name]['gutenberg_id'])
            if book in seen_ids: raise ValueError('Gutenberg ID appears twice: '+str(book))
            if digest in seen_sha: raise ValueError(f'Identical text leaked across splits: {seen_sha[digest]} and {name}')
            seen_ids.add(book); seen_sha[digest]=name
            contents[name]=data
        codec=TokenCodec()
        dtype=token_dtype(codec.vocab_size)
        outdir.mkdir(parents=True,exist_ok=True)
        spec=[('train','train'),('dev','valid'),('test','test')]
        stats={}
        for split,output in spec:
            files=sorted(n for n in actual if n.startswith(split+'/'))
            lengths=[]
            with (outdir/(output+'.bin')).open('wb') as dst:
                for name in files:
                    # One end-of-book marker; never one EOS per line.
                    data=contents[name]
                    text=data.decode('utf-8',errors='replace')
                    ids=codec.encode(text)
                    ids.append(codec.eos)
                    np.asarray(ids,dtype=dtype).tofile(dst)
                    lengths.append(len(ids))
            stats[output]={'tokens':sum(lengths),'doc_lengths':lengths,'books':len(files),
                           'gutenberg_ids':[mapping[n]['gutenberg_id'] for n in files]}
        split_manifest={
            'source_repository':'shemshallah/NaNROUTER',
            'source_artifact':'pal-r11-nanrouter-raw-prose-books',
            'source_artifact_id':13913881475,
            'source_workflow_run':43151215151,
            'source_archive_sha256':file_hash(archive),
            'source_manifest_sha256':hashlib.sha256(z.read('MANIFEST.json')).hexdigest(),
            'upstream_split_policy':manifest['split_policy'],
            'counts':{k:stats[k]['books'] for k in stats},
            'books':[{**mapping[n], 'archive_name':n} for n in sorted(actual)],
            'verification':'All source files verified against archive SHA256SUMS, manifest SHA256, byte lengths, unique Gutenberg IDs, and unique text hashes; test never mixed with training.'
        }
        blob=(json.dumps(split_manifest,indent=2,sort_keys=True)+'\n').encode()
        (outdir/'split_manifest.json').write_bytes(blob)
        meta={
            'vocab_size':257,'tokenizer_model':None,'tokenizer_sha256':None,
            'unit':'UTF8_bytes_plus_eos','token_dtype':dtype,'eos_id':256,
            'split_unit':'whole_book','split_manifest_sha256':hashlib.sha256(blob).hexdigest(),
            'train_sha256':file_hash(outdir/'train.bin'),'valid_sha256':file_hash(outdir/'valid.bin'),
            'test_sha256':file_hash(outdir/'test.bin'),
            'train_tokens':stats['train']['tokens'],'valid_tokens':stats['valid']['tokens'],
            'test_tokens':stats['test']['tokens'],
            'train_doc_lengths':stats['train']['doc_lengths'],
            'valid_doc_lengths':stats['valid']['doc_lengths'],
            'test_doc_lengths':stats['test']['doc_lengths'],
            'train_books':stats['train']['books'],'valid_books':stats['valid']['books'],
            'test_books':stats['test']['books'],'unique_books':len(actual),
            'duplicates_ignored':0,
            'source_artifact':'pal-r11-nanrouter-raw-prose-books',
            'source_archive_sha256':file_hash(archive),
        }
        (outdir/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps({'archive':str(archive),'output':str(outdir),
                          'split_books':{k:stats[k]['books'] for k in stats},
                          'split_tokens':{k:stats[k]['tokens'] for k in stats},
                          'all_checksums_verified':True,'test_isolated':True},indent=2))
        return meta

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--require-frozen-counts',action='store_true')
    a=p.parse_args()
    import_archive(a.archive,a.out_dir,a.require_frozen_counts)
