#!/usr/bin/env python3
"""Pack 1600 TRAIN-only BPE merges' first-subword predictive counts for native v176.
Never uses validation or TEST target labels to construct the native model.
"""
from pathlib import Path
import sys,struct,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from train_long175 import load_models,load_split,verify_corpus
from eval_bpe_crosslevel176 import BPEFirst

def main():
 m,o,names,fp,old=load_models();root=ROOT/'results/v175_release3';verify_corpus(root,names)
 X,Y,B=load_split('train',root)
 ck=json.loads((ROOT/'checkpoints/v176_v133_style_bpe.json').read_text())
 b=BPEFirst(X,Y,names,ck['merges'])
 assert b.V==1856 and len(b.token)==12000
 cross=ROOT/'checkpoints/v176_native_crosslevel.bin';cross_sha=hashlib.sha256(cross.read_bytes()).digest()
 data=bytearray(b'N176BPE1'+struct.pack('<3Q',*map(int,fp))+cross_sha)
 data.extend(struct.pack('<I I f f I I',12000,b.V,24.,.35,len(b.c1),len(b.c2)))
 data.extend(b.token.astype('<u2').tobytes())
 data.extend(b.global_count.astype('<u4').tobytes())
 def rows(items,arity):
  for key,counts in sorted(items):
   ks=(key,) if arity==1 else key
   data.extend(struct.pack('<'+'i'*arity,*ks))
   data.extend(struct.pack('<H',len(counts)))
   for tok,n in sorted(counts.items()):
    assert 0<=tok<b.V and 0<n<=2**32-1
    data.extend(struct.pack('<HI',int(tok),int(n)))
 rows(b.c1.items(),1);rows(b.c2.items(),2)
 path=ROOT/'checkpoints/v176_native_bpefirst.bin';path.write_bytes(data)
 report={'status':'PASS','path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest(),
       'crosslevel_sidecar_sha256':cross_sha.hex(),'bytes':len(data),'bpe_merge_count':len(ck['merges']),
       'BPE_first_vocab':b.V,'distinct_last_word_contexts':len(b.c1),'distinct_last2_contexts':len(b.c2),
       'source':'95,040 TRAIN-only words from NaNROUTER release0003','beta_selected_val':.35,'prior':24.,
       'parent_fingerprints':list(map(int,fp))}
 (ROOT/'results/v176_bpe_native_pack.json').write_text(json.dumps(report,indent=2))
 print('V176 BPE PACK',json.dumps(report),flush=True)
if __name__=='__main__':main()