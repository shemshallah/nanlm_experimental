#!/usr/bin/env python3
"""Matched independent-book native 96-beam sweep of ANN strengths; DEV only."""
import ctypes,json,math,os,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords
from beam_words156 import decode
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4')
l=load(R/'bin/libwordfield168_ann.so')
l.wf168_ann_alpha.argtypes=[ctypes.c_void_p];l.wf168_ann_alpha.restype=ctypes.c_double
split=os.getenv('V168_SPLIT','dev').lower()
assert split in ('dev','test')
rows=[]
for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
 if bi>=24:break
 words=getwords(raw,skip)
 sample=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
 if os.getenv('V168_ALL_WORDS')=='1':sample=words
 for p,w in sample:rows.append((bi,raw[:p],w))
assert len(rows)>0
for beta in [float(x) for x in os.getenv("V168_BETAS","0,.02,.05,.10,.20,.35").split(",")]:
 os.environ['V168_ANN_STRENGTH']=str(beta)
 h=l.wf148_open(*MODEL);assert h,beta
 tic=time.monotonic();loss=0;symbols=0;hits={1:0,5:0,32:0};active=0;bybook={};diag=[]
 for bi,past,truth in rows:
  assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  active+=int(l.wf168_ann_alpha(h)>0)
  for i in range(len(truth)+1):
   pre=truth[:i].encode();o=(F*27)();e=F();assert l.wf148_next(h,pre,len(pre),o,ctypes.byref(e))==0
   p=o[26] if i==len(truth) else o[ord(truth[i])-97]
   loss-=math.log2(max(1e-25,float(p)+e.value/27));symbols+=1
  rank,_=decode(l,h,beam=96,steps=20,topk=32)
  for k in hits:hits[k]+=int(truth in rank[:k])
  bybook.setdefault(bi,[0,0,0]);bybook[bi][0]+=int(truth in rank[:1]);bybook[bi][1]+=int(truth in rank[:5]);bybook[bi][2]+=int(truth in rank[:32])
  diag.append({'book':bi,'truth':truth,'top':rank[:5]})
 l.wf148_close(h)
 result={'model':'NaNCM v168 signed ANN native','split':split.upper(),'beta':beta,'n':len(rows),'top1':hits[1],'top5':hits[5],'top32':hits[32],'bits_per_spelling_symbol':loss/symbols,'active':active,'bybook':bybook,'seconds':round(time.monotonic()-tic,2)}
 stem=f'v168_native_ann_{split}_{"all" if os.getenv("V168_ALL_WORDS")=="1" else "sample"}_{str(beta).replace(".","_")}'
 (R/f'results/{stem}.json').write_text(json.dumps(result,indent=2))
 (R/f'results/{stem}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in diag))
 print('ARM',json.dumps({k:v for k,v in result.items() if k!='bybook'}),flush=True)