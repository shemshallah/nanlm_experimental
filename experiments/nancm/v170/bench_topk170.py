#!/usr/bin/env python3
"""Matched v169/v170 next-byte rank diagnostic from the complete recovery ZIP."""
import ctypes,math,json,sys,os
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
L=ctypes.CDLL(str(R/'bin/libwordfield170.so'))
for v in ('169','170'):
 f=getattr(L,f'wf{v}_byte_open');f.argtypes=[ctypes.c_char_p]*4;f.restype=ctypes.c_void_p
 f=getattr(L,f'wf{v}_byte_next');f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)];f.restype=ctypes.c_int
 f=getattr(L,f'wf{v}_byte_close');f.argtypes=[ctypes.c_void_p];f.restype=None
L.wf170_byte_reset.argtypes=[ctypes.c_void_p];L.wf170_byte_reset.restype=ctypes.c_int
pack=str(R/'checkpoints/v169_native_byte_lattice.bin').encode()
hs={v:getattr(L,f'wf{v}_byte_open')(*MODEL,pack) for v in ('169','170')}
assert all(hs.values())
def dist(v,past):
 a=(F*256)()
 assert getattr(L,f'wf{v}_byte_next')(hs[v],ctypes.c_char_p(past),len(past),a)==0
 return [float(x) for x in a]
result={}
for split in os.environ.get('V170_SPLITS','dev,test').split(','):
 bybook={v:[] for v in hs}
 for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
  if bi>=24:break
  assert L.wf170_byte_reset(hs['170'])==0
  start=max(2,skip)
  while start<len(raw) and (65<=raw[start-1]<=90 or 97<=raw[start-1]<=122):start+=1
  d={v:{'n':0,'loss':0.,'top1':0,'top5':0,'top16':0} for v in hs}
  for i in range(start,len(raw)):
   for v in hs:
    p=dist(v,raw[:i]);rank=sorted(range(256),key=lambda j:(-p[j],j)).index(raw[i]);x=d[v]
    x['n']+=1;x['loss']+=-math.log2(p[raw[i]]);x['top1']+=rank<1;x['top5']+=rank<5;x['top16']+=rank<16
  for v in hs:
   d[v]['book']=bi;d[v]['bpb']=d[v]['loss']/d[v]['n'];bybook[v].append(d[v])
  print('BOOK',split,bi,{v:(d[v]['top1'],d[v]['top5']) for v in hs},flush=True)
 result[split]={v:{k:sum(row[k] for row in bybook[v]) for k in ('n','loss','top1','top5','top16')} for v in hs}
 for v in hs:
  result[split][v]['bpb']=result[split][v]['loss']/result[split][v]['n']
  result[split][v]['bybook']=bybook[v]
 result[split]['books_better_bpb']=sum(a['bpb']<b['bpb'] for a,b in zip(bybook['170'],bybook['169']))
 print('FINAL',split,{v:{k:x for k,x in result[split][v].items() if k!='bybook'} for v in hs},flush=True)
for v in hs:getattr(L,f'wf{v}_byte_close')(hs[v])
(R/'results/v170_topk_native_comparison.json').write_text(json.dumps(result,indent=2))
