#!/usr/bin/env python3
"""Deterministic native NaNCM169 byte sidecar from TRAIN-only fitted statistics.
Does not read DEV/TEST data or modify any inherited optimizer/model weights.
"""
import json,struct,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
ng=json.loads((R/'checkpoints/v169_byte_ngram.json').read_text())
bg=json.loads((R/'checkpoints/v169_byte_bridge.json').read_text())
assert ng['format']=='N169BYTE3' and bg['format']=='N169BYTE1'
with (R/'checkpoints/v164_native_distribution.tsv').open('rb') as f:header=f.readline().decode().strip().split('\t')
assert header[0]=='N164DIST1' and len(header)==4
fps=[int(x) for x in header[1:]]

def data_family(data):
 result=[]
 for keyhex,counts in data.items():
  kbytes=bytes.fromhex(keyhex)
  assert len(kbytes)<=3
  integer=int.from_bytes(kbytes,'little')
  values=sorted((int(b),int(c)) for b,c in counts.items() if int(c)>0)
  assert 1<=len(values)<=256 and all(0<=b<=255 and 0<c<2**32 for b,c in values)
  result.append((len(kbytes),integer,values))
 result.sort(key=lambda x:(x[0],x[1]));return result

rows_ng=[row for m in ng['orders'] for row in data_family(m)]
rows_alpha=data_family({k:{'0':v[0],'1':v[1]} for k,v in bg['alpha_counts'].items()})
rows_other=data_family(bg['other_counts'])
output=R/'checkpoints/v169_native_byte_lattice.bin'
with output.open('wb') as f:
 f.write(b'N169BYT1')
 f.write(struct.pack('<3Q',*fps))
 f.write(struct.pack('<3I',len(rows_ng),len(rows_alpha),len(rows_other)))
 f.write(struct.pack('<4I',*(bg['upper']+bg['lower'])))
 for rows in (rows_ng,rows_alpha,rows_other):
  for k,key,values in rows:
   f.write(struct.pack('<BIH',k,key,len(values)))
   for b,n in values:f.write(struct.pack('<BI',b,n))
res={'format':'N169BYT1','source':'100 NaNROUTER TRAIN books (300 windows)',
 'model_fingerprints':fps,'byte_ngram_sha':hashlib.sha256((R/'checkpoints/v169_byte_ngram.json').read_bytes()).hexdigest(),
 'byte_bridge_sha':hashlib.sha256((R/'checkpoints/v169_byte_bridge.json').read_bytes()).hexdigest(),
 'rows':[len(rows_ng),len(rows_alpha),len(rows_other)],'binary_size':output.stat().st_size,
 'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
(R/'results/v169_native_byte_pack.json').write_text(json.dumps(res,indent=2))
print('FINAL',json.dumps(res))