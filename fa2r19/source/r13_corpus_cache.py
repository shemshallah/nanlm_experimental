#!/usr/bin/env python3
"""Persistent, source-verified PAL R11/R12 preprocessing cache for large corpora.

Checks the manifest, all source book byte hashes, pretrained tokenizer and
vocabulary fingerprints. Cached arrays are *data*, never executable pickles.
"""
import hashlib,json,pathlib,time
from dataclasses import asdict
import numpy as np
from r11_raw import Book,read_books as uncached_read_books

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def read_books_cached(root,compositor,vocab,max_books=0,cache_dir=None):
 root=pathlib.Path(root);cache=pathlib.Path(cache_dir or (root/'.pal_preprocessed'))
 manifest_path=root/'MANIFEST.json';manifest=json.loads(manifest_path.read_text())
 tokenizer=getattr(compositor,'sp',None)
 # Avoid accepting a cache associated with a different SentencePiece model.
 tokens=[tokenizer.id_to_piece(i) for i in range(tokenizer.get_piece_size())]
 key_data={'manifest':sha(manifest_path),'vocab':hashlib.sha256('\n'.join(vocab).encode()).hexdigest(),
           'sentencepiece':hashlib.sha256('\n'.join(tokens).encode()).hexdigest(),
           'max_books':max_books,'cache_version':2}
 key=hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()[:24]
 dest=cache/key;index=dest/'index.json'
 def has_valid_cache():
  if not index.exists():return False
  meta=json.loads(index.read_text())
  if meta['key_data']!=key_data:return False
  for row in meta['documents']:
   src=root/row['split']/(row['gutenberg_id']+'.txt')
   if not src.exists() or sha(src)!=row['sha256']:return False
   if not (dest/(row['split']+'_'+row['gutenberg_id']+'.npz')).exists():return False
  return True
 if has_valid_cache():
  books={'train':[],'dev':[],'test':[]}
  meta=json.loads(index.read_text())
  for row in meta['documents']:
   f=dest/(row['split']+'_'+row['gutenberg_id']+'.npz')
   with np.load(f,allow_pickle=False) as z:
    fields={k:z[k].copy() for k in ('tokens','token_tag','chunks','byte_starts','examples','context')}
   books[row['split']].append(Book(row['gutenberg_id'],row['split'],**fields,
                              raw_sha=row['sha256'],raw_len=row['bytes']))
  print(json.dumps({'phase':'R13_CACHE_HIT','books':len(meta['documents']),'cache_key':key}),flush=True)
  return books,manifest
 books,_=uncached_read_books(root,compositor,vocab,max_books)
 dest.mkdir(parents=True,exist_ok=True);rows=[]
 for split,docs in books.items():
  for book in docs:
   np.savez(dest/(split+'_'+book.book_id+'.npz'),tokens=book.tokens,token_tag=book.token_tag,
            chunks=book.chunks,byte_starts=book.byte_starts,examples=book.examples,context=book.context)
   rows.append({'gutenberg_id':book.book_id,'split':split,'sha256':book.raw_sha,'bytes':book.raw_len})
 index.write_text(json.dumps({'key_data':key_data,'documents':rows},indent=2)+'\n')
 print(json.dumps({'phase':'R13_CACHE_BUILD','books':len(rows),'cache_key':key}),flush=True)
 return books,manifest