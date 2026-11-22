#!/usr/bin/env python3
"""Paired original v164 vs *trained* signed cloud v165 on held-out books.
DEV chooses a cloud strength; reused TEST is diagnostic only, never tuned.
No target/future word is sent to begin() or the beam decoder.
"""
import os,sys,json,time,ctypes,math,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'parent'
sys.path.insert(0,str(P/'scripts'));sys.path.insert(0,str(R/'scripts'))
from test_native_speller156 import load,MODEL,F,getwords,voc
from beam_words156 import decode
from train_gate163 import setup,winread

def bits(lib,h,word):
 b=0.
 for i in range(len(word)+1):
  out=(F*27)();e=F(1.);pre=word[:i].encode()
  assert lib.wf148_next(h,pre,i,out,ctypes.byref(e))==0
  q=float(out[26 if i==len(word) else ord(word[i])-97])+float(e.value)/27.
  b-=math.log2(max(1e-25,q))
 return b

def run(split='dev',strengths=(0.,.1,.25,.5,1.),books=24,per_book=20,beam=False):
 setup(core=False,domain=True)
 os.environ.update(V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',
  V160_ASSOC_BUDGET='.4',V160_ASSOC_INDEX=str(R/'checkpoints/v161_unified.tsv'),
  V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),
  V165_CLOUD_INDEX=str(R/'checkpoints/v165_signed_cloud.bin'))
 engines={}
 for strength in strengths:
  os.environ['V165_SIGNED_ENERGY']=str(strength)
  l=load(R/'bin/libwordfield165.so');h=l.wf148_open(*MODEL)
  assert h,strength
  engines[strength]=(l,h)
 rows=[];begin=time.monotonic()
 for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
  if bi>=books:break
  words=getwords(raw,skip)
  if per_book and len(words)>per_book:
   words=[words[min(len(words)-1,int((i+.5)*len(words)/per_book))] for i in range(per_book)]
  for pos,word in words:
   row={'book_index':bi,'word':word,'oov':word not in voc,'models':{}}
   for strength,(lib,h) in engines.items():
    assert lib.wf148_begin(h,ctypes.c_char_p(raw[:pos]),pos,F(.25))>=0
    v={'bits':bits(lib,h,word)}
    if beam:v['words']=decode(lib,h)[0]
    row['models'][str(strength)]=v
   rows.append(row)
  print('BOOK',split,bi+1,'/',books,'total',len(rows),flush=True)
 for lib,h in engines.values():lib.wf148_close(h)
 result={'split':split,'n':len(rows),'books':books,'words_per_book':per_book,
  'sample':'deterministically spaced word positions in distinct books',
  'strengths':strengths,'beam':96 if beam else None,'metrics':{},'seconds':round(time.monotonic()-begin,2),
  'warning':'Spelling/EOS bits per symbol, NOT full 256-byte BPB; TEST reused in prior research'}
 for strength in strengths:
  name=str(strength)
  b=sum(r['models'][name]['bits'] for r in rows)
  m={'bits_per_spelling_symbol':round(b/sum(len(r['word'])+1 for r in rows),6)}
  if beam:
   for k in (1,5,32):
    h=sum(r['word'] in r['models'][name]['words'][:k] for r in rows)
    m[f'top{k}']={'hits':h,'pct':round(h/max(1,len(rows))*100,4)}
   m['oov_top32']=sum(r['oov'] and r['word'] in r['models'][name]['words'][:32] for r in rows)
  result['metrics'][name]=m
 fn=R/'results'/f'v165_{split}_b{books}_w{per_book}_{"beam" if beam else "loss"}.json';fn.write_text(json.dumps(result,indent=2))
 fn.with_name(fn.stem+'_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
 print('FINAL',json.dumps(result,indent=2),flush=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('split',choices=['dev','test'],default='dev');p.add_argument('--strengths',nargs='+',type=float,default=[0.,.1,.25,.5,1.]);p.add_argument('--books',type=int,default=24);p.add_argument('--per-book',type=int,default=20);p.add_argument('--beam',action='store_true')
 a=p.parse_args();run(a.split,tuple(a.strengths),a.books,a.per_book,a.beam)
