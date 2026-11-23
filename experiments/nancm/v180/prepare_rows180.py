#!/usr/bin/env python3
"""Export labeled, disjoint evaluation rows. Never use val or test labels for fitting."""
import numpy as np,struct,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/v180';OUT.mkdir(exist_ok=True)
for split in ('train','val'):
 d=np.load(ROOT/'results/v175_release3'/f'{split}.npz');X=d['X'];Y=d['Y'];B=d['book'];P=d['word_pos']
 path=OUT/f'{split}_rows.bin'
 with open(path,'wb') as f:
  f.write(b'NC180ROW');f.write(struct.pack('<I',len(Y)))
  for x,y,b,p in zip(X,Y,B,P):
   f.write(struct.pack('<ii',int(b),int(p)));f.write(np.r_[x,y].astype('<i2').tobytes())
 print(split,len(Y),path,hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
