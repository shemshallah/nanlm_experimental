#!/usr/bin/env python3
"""Evaluate an explicitly normalized 256-byte bridge for the existing
27-way native word predictor. This is a newly trained bridge, NOT a claim
that historical whole-word symbol BPB was full-byte BPB. Train data: 100
TRAIN books only. No DEV/TEST bytes enter training.
"""
import argparse,ctypes,json,math,os,sys,collections,hashlib,time,re
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
ALPHA=set(range(65,91))|set(range(97,123));OTHER=[c for c in range(256) if c not in ALPHA]
def isalpha(b):return (65<=b<=90) or (97<=b<=122)

def train_bridge():
 # Two-byte history for category/conditional non-alpha distribution; smoothed
 # with a fixed global prior. Lowercase vs uppercase observed at word start.
 alpha=collections.defaultdict(lambda:[0,0]);other=collections.defaultdict(collections.Counter)
 other_global=collections.Counter();upper=[0,0];lower=[0,0];source=hashlib.sha256();n=0;books=[]
 cohort=json.loads((R/'external_v163/COHORTS.json').read_text())
 for part in range(3):
  windows=list(winread(R/f'external_v163/v163_calibration_{part}.bin'))
  for (raw,skip),metadata in zip(windows,cohort['calibration_windows'][part]):
   books.append(metadata['name']);source.update(raw);source.update(metadata['name'].encode())
   for i in range(max(2,skip),len(raw)):
    b=raw[i];h=raw[i-2:i];prev=raw[i-1]
    if not isalpha(prev):
     for k in (0,1,2):alpha[h[-k:] if k else b''][int(isalpha(b))]+=1
    if not isalpha(b):
     other_global[b]+=1
     for k in (0,1):other[h[-k:] if k else b''][b]+=1
    else:
     v=upper if (not isalpha(prev)) else lower
     v[int(65<=b<=90)]+=1
    n+=1
 # Unique book counts across the 3 sampled windows, calibration split fixed.
 return {'format':'N169BYTE1','training_source':'NaNROUTER 100 TRAIN books / external_v163 calibration windows',
         'windows':len(books),'unique_books':len(set(books)),'source_sha256':source.hexdigest(),'bytes':n,
         'alpha_counts':{x.hex():v for x,v in alpha.items()},
         'other_counts':{x.hex():dict(v) for x,v in other.items()},
         'other_global':dict(other_global),'upper':upper,'lower':lower,
         'not_full_neural_byte_model':True}

class Bridge:
 def __init__(self,pack):
  self.a={bytes.fromhex(k):v for k,v in pack['alpha_counts'].items()}
  self.o={bytes.fromhex(k):{int(b):n for b,n in v.items()} for k,v in pack['other_counts'].items()}
  self.og={int(b):n for b,n in pack['other_global'].items()};self.other_total=sum(self.og.values());
  self.up=pack['upper'];self.lo=pack['lower']
  # small all-204-support Dirichlet prior, no impossible bytes.
  self.global_other={i:(self.og.get(i,0)+.1)/(self.other_total+len(OTHER)*.1) for i in OTHER}
 def alpha_start(self,past):
  h=past[-2:]
  base=self.a.get(b'',[1,1]);pa=(base[1]+1)/(sum(base)+2.)
  for k,prior in ((1,40.),(2,80.)):
   if len(h)>=k:
    x=self.a.get(h[-k:])
    if x:pa=(x[1]+prior*pa)/(sum(x)+prior)
  return min(.999999,max(.000001,pa))
 def nonalpha(self,past):
  h=past[-1:];counts=self.o.get(h,{}) if h else {}
  den=sum(counts.values())+25
  return {i:(counts.get(i,0)+25*self.global_other[i])/den for i in OTHER}
 def uppercase(self,inside):
  cnt=self.lo if inside else self.up
  return (cnt[1]+.5)/(sum(cnt)+1.)
 def dist(self,past,prefix,symbol_probs):
  nonalpha=self.nonalpha(past);up=self.uppercase(bool(prefix)); p=[0.]*256
  letters=symbol_probs[:26];total=sum(letters)
  if prefix:
   p_non=symbol_probs[26]
   scale=1.
  else:
   p_non=1-self.alpha_start(past)
   scale=(1-p_non)/max(1e-30,total)
  for i,x in enumerate(letters):
   p[97+i]=x*scale*(1-up)
   p[65+i]=x*scale*up
  for b,x in nonalpha.items():p[b]=x*p_non
  return p

def eval_byte(lib,bridge,max_books=24,split="dev"):
 setup(core=False,domain=True,gate=None)
 os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(R/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
 L=load(R/'bin'/lib);h=L.wf148_open(*MODEL);assert h
 s=0.;nn=0;letter_n=0;non_n=0;case_n=0;max_norm=0.;bybook=[];perbyte=[];t=time.monotonic()
 for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
  if bi>=max_books:break
  # Skip any incomplete word at the left edge of the scored window.
  start=max(2,skip)
  while start<len(raw) and isalpha(raw[start-1]):start+=1
  prefix='';needs_begin=True;word_start=start;bookloss=0.;bookn=0
  for i in range(start,len(raw)):
   if needs_begin:
    assert L.wf148_begin(h,ctypes.c_char_p(raw[:i]),i,F(.25))>=0,(lib,bi,i)
    needs_begin=False
   arr=(F*27)();e=F();b=prefix.encode()
   assert L.wf148_next(h,b,len(b),arr,ctypes.byref(e))==0,(lib,bi,i,prefix)
   symbols=[float(a)+e.value/27 for a in arr]
   total=sum(symbols)
   assert abs(total-1)<3e-4,total
   p=bridge.dist(raw[:i],prefix,symbols);norm=sum(p);max_norm=max(max_norm,abs(norm-1));
   assert abs(norm-1)<5e-4,(lib,bi,i,norm)
   actual=raw[i];ll=-math.log2(max(1e-30,p[actual]));s+=ll;nn+=1;bookloss+=ll;bookn+=1;perbyte.append((bi,i,actual,bool(prefix),p[actual]))
   if isalpha(actual):
    letter_n+=1;case_n+=int(65<=actual<=90)
    prefix+=chr(actual).lower()
    if len(prefix)>63:prefix='';needs_begin=True
   else:
    non_n+=1;prefix='';needs_begin=True
  bybook.append({'book':bi,'bytes':bookn,'bpb':bookloss/bookn})
  print('BOOK',lib,bi,'bytes',bookn,'bpb',round(bookloss/bookn,6),'seconds',round(time.monotonic()-t,1),flush=True)
 L.wf148_close(h)
 (R/f'results/v169_byte_{split}_{lib}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in perbyte))
 return {'library':lib,'books':len(bybook),'bytes':nn,'full_256_way_bpb':s/nn,'max_abs_normalization_error':max_norm,'letter_bytes':letter_n,'nonletter_bytes':non_n,'capital_letters':case_n,'bybook':bybook,'note':'new 256-way probabilistic bridge, trained only on TRAIN documents; not historical model byte BPB'}

def main():
 a=argparse.ArgumentParser();a.add_argument('--max-books',type=int,default=24);a.add_argument('--split',choices=['dev','test'],default='dev');a.add_argument('--libs',default='libwordfield167.so,libwordfield168_ann.so,libwordfield169.so');p=a.parse_args()
 path=R/'checkpoints/v169_byte_bridge.json'
 if not path.exists():
  pack=train_bridge();path.write_text(json.dumps(pack,separators=(',',':')))
 else:pack=json.loads(path.read_text())
 bridge=Bridge(pack)
 results={'training':{k:v for k,v in pack.items() if k not in ('alpha_counts','other_counts','other_global')},'dev':[],'caveat':'Full 256-way conditional distribution of explicitly constructed auxiliary bridge; cannot compare directly to historical 27-way spelling/EOS numbers.'}
 for name in p.libs.split(','):
  results['dev'].append(eval_byte(name,bridge,p.max_books,p.split))
 out=R/f'results/v169_byte_bridge_{p.split}.json';out.write_text(json.dumps(results,indent=2));print('FINAL',json.dumps({i['library']:i['full_256_way_bpb'] for i in results['dev']}),flush=True)
if __name__=='__main__':main()