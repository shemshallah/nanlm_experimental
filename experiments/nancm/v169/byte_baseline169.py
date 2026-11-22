#!/usr/bin/env python3
"""Native-independent smoothed 256-byte n-gram baseline on identical corpus,
heldout windows and byte positions as byte_bridge169.py. No DEV training.
"""
import argparse,json,sys,math,collections,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import winread

def is_alpha(b):return 65<=b<=90 or 97<=b<=122

def train():
 x=[collections.defaultdict(collections.Counter) for _ in range(4)]
 for part in range(3):
  for raw,skip in winread(R/f'external_v163/v163_calibration_{part}.bin'):
   for i in range(max(skip,3),len(raw)):
    for k in range(4):x[k][raw[i-k:i] if k else b''][raw[i]]+=1
 return x

def probs(x,past):
 counts=x[0][b''];total=sum(counts.values());prior={i:(counts.get(i,0)+.1)/(total+25.6) for i in range(256)}
 for k in range(1,4):
  if len(past)<k:break
  row=x[k].get(past[-k:])
  if row:
   t=sum(row.values());strength=(10.,30.,80.)[k-1]
   prior={i:(row.get(i,0)+strength*prior[i])/(t+strength) for i in range(256)}
 return prior

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--split",choices=["dev","test"],default="dev");args=ap.parse_args()
 t0=time.monotonic();x=train();logloss=0.;total=0;bybook=[];perbyte=[]
 for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{args.split}160.bin')):
  if bi>=24:break
  start=max(skip,2)
  while start<len(raw) and is_alpha(raw[start-1]):start+=1
  bookloss=0;n=0
  for i in range(start,len(raw)):
   pred=probs(x,raw[:i]);loss=-math.log2(max(1e-30,pred[raw[i]]));bookloss+=loss;n+=1;perbyte.append((bi,i,raw[i],is_alpha(raw[i-1]),pred[raw[i]]))
  logloss+=bookloss;total+=n;bybook.append({'book':bi,'bytes':n,'bpb':bookloss/n})
 (R/f'results/v169_byte_ngram_{args.split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in perbyte))
 out={'model':'causal_smoothed_byte_trigram','train_source':'same 100 TRAIN books as v169 byte bridge','heldout_dev_books':24,'scored_bytes':total,'full_256_way_bpb':logloss/total,'bybook':bybook,'seconds':round(time.monotonic()-t0,3)}
 (R/f'results/v169_byte_ngram_baseline_{args.split}.json').write_text(json.dumps(out,indent=2));print('FINAL',json.dumps({k:v for k,v in out.items() if k!='bybook'}),flush=True)
if __name__=='__main__':main()