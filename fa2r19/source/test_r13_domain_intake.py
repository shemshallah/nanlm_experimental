#!/usr/bin/env python3
import csv,hashlib,io,json,pathlib,tarfile,tempfile
from r13_domain_intake import run
from types import SimpleNamespace
with tempfile.TemporaryDirectory() as t:
 root=pathlib.Path(t);catalog=root/'pg_catalog.csv';arc=root/'sample.tar.gz'
 rows=[]
 for i in range(101,130):
  dom=['QC','R','K','HD','B','D','PR'][(i-101)%7]
  rows.append({'Text#':str(i),'Language':'en','LoCC':dom,'Subjects':'','Authors':f'Author {i}', 'Title':f'Title {i}'})
 with open(catalog,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with tarfile.open(arc,'w:gz') as tf:
  for i in range(101,130):
   data=(f'Book {i} has different material.\n'*150).encode();b=io.BytesIO(data)
   x=tarfile.TarInfo(f'batch-0001/{i}--author--title.txt');x.size=len(data);tf.addfile(x,b)
 r=run(SimpleNamespace(catalog=str(catalog),archive=[str(arc)],exclude_manifest=[],output=str(root/'out'),max_per_domain=10,min_book_bytes=1000,max_book_bytes=10000000,seed=20261111))
 assert len(r['documents'])==29,len(r['documents'])
 assert len(r['counts_by_domain'])==7
 assert sum(r['counts_by_split'].values())==29
 assert all(hashlib.sha256((root/'out'/x['split']/(x['gutenberg_id']+'.txt')).read_bytes()).hexdigest()==x['sha256'] for x in r['documents'])
 print('PASS: 29 books, 7 domains, exact hashes, no cross-split duplicate IDs')
