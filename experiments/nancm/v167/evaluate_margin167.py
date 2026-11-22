#!/usr/bin/env python3
"""Matched strict 96-beam and native symbol loss across checkpoint/strength arms.
DEV for selection; repeated TEST only for frozen comparison afterward.
"""
import argparse,ctypes,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords
from beam_words156 import decode

def samples(split):
 rows=[]
 for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
  if bi>=24:break
  words=getwords(raw,skip)
  if not words:continue
  chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))]
  for pos,truth in chosen:rows.append((bi,raw[:pos],truth))
 assert len(rows)==480
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--limit',type=int,default=480);p.add_argument('--fast',action='store_true');a=p.parse_args()
 setup(core=False,domain=True,gate=None)
 os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'))
 arms=[('v166_promoted',R/'checkpoints/v166_lexical_birth.bin',.05),('v166_high',R/'checkpoints/v166_lexical_birth.bin',.5),('v167_margin_005',R/'checkpoints/v167_margin_candidate.bin',.05),('v167_margin_025',R/'checkpoints/v167_margin_candidate.bin',.25),('v167_margin_050',R/'checkpoints/v167_margin_candidate.bin',.5)]
 if a.fast:arms=arms[:1]+arms[2::2]
 rows=samples(a.split)[:a.limit]
 result={'split':a.split,'book_count':len(set(x[0] for x in rows)),'n':len(rows),'beam':96,'length_max':20,'arms':{},'reused_test':a.split=='test','not_full_byte_BPB':True}
 for name,model,strength in arms:
  t0=time.monotonic()
  os.environ['V166_LEXICAL_BIRTH']=str(strength);os.environ['V166_BIRTH_MODEL']=str(model)
  lib=load(R/'bin/libwordfield166.so');h=lib.wf148_open(*MODEL);assert h,name
  hits={1:0,5:0,32:0};loss=0;symbols=0;by_book={};active=0
  lib.wf166_candidate_count.argtypes=[ctypes.c_void_p];lib.wf166_candidate_count.restype=ctypes.c_int
  for bi,past,truth in rows:
   assert lib.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
   active+=int(lib.wf166_candidate_count(h)>0)
   for i in range(len(truth)+1):
    o=(F*27)();e=F();pre=truth[:i].encode();rc=lib.wf148_next(h,pre,len(pre),o,ctypes.byref(e));assert rc==0,(name,rc,pre)
    p=float(o[26] if i==len(truth) else o[ord(truth[i])-97])+e.value/27
    loss-=math.log2(max(1e-25,p));symbols+=1
   ranked,_=decode(lib,h,beam=96,steps=20,topk=32)
   for k in hits:hits[k]+=int(truth in ranked[:k])
   z=by_book.setdefault(bi,{'n':0,'hits1':0});z['n']+=1;z['hits1']+=int(truth==ranked[0] if ranked else False)
  lib.wf148_close(h)
  result['arms'][name]={'strength':strength,'model_sha256':__import__('hashlib').sha256(model.read_bytes()).hexdigest(),
      'top1':hits[1],'top5':hits[5],'top32':hits[32],
      'bits_per_spelling_symbol':loss/symbols,'positions_active':active,
      'per_book_top1':by_book,'seconds':round(time.monotonic()-t0,2)}
  print('ARM',name,json.dumps({k:v for k,v in result['arms'][name].items() if k!='per_book_top1'}),flush=True)
 (R/f'results/v167_{a.split}_margin_sweep.json').write_text(json.dumps(result,indent=2))
 print('FINAL',json.dumps({k:v for k,v in result.items() if k!='arms'}),flush=True)
if __name__=='__main__':main()