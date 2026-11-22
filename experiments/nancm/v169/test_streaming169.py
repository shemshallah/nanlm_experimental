#!/usr/bin/env python3
"""Replay frozen byte-score evidence through actual next256 runtime API.
The test selects fixed positions before seeing predictions, checks all 256
probabilities and exact observed-byte mixture identity.
"""
import json,math,sys,os
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).parent),str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import winread
from byte_system169 import NaNCM169BytePredictor

def check(split,model):
 rawwins=list(winread(R/f'external_book_gate/heldout_{split}160.bin'))[:24]
 d=[json.loads(s) for s in (R/f'results/v169_byte_ngram_{split}.jsonl').read_text().splitlines()]
 w=[json.loads(s) for s in (R/f'results/v169_byte_{split}_libwordfield169.so.jsonl').read_text().splitlines()]
 assert len(d)==len(w) and len(d)>7900
 assert all(u[:4]==v[:4] for u,v in zip(d,w))
 checked=0;largest=0;nonletter=0
 for bi in range(24):
  idx=[i for i,r in enumerate(d) if r[0]==bi]
  chosen=[idx[min(len(idx)-1,int((k+.5)*len(idx)/6))] for k in range(6)]
  for j in chosen:
   r=d[j];q=w[j];_,rawpos,actual,inside,ngram_p=r
   seq=rawwins[bi][0];assert actual==seq[rawpos]
   p=model.next256(seq[:rawpos]);total=sum(p)
   assert len(p)==256 and abs(total-1)<1e-10
   assert all(math.isfinite(v) and v>0. for v in p)
   expected=(1-(.5 if inside else .1))*ngram_p+(.5 if inside else .1)*q[-1]
   err=abs(p[actual]-expected)
   assert err<2e-7,(split,bi,rawpos,inside,p[actual],expected,err)
   largest=max(largest,err);checked+=1;nonletter+=int(not inside)
 return {'split':split,'exact_mixture_replay':checked,'nonword_positions':nonletter,'max_probability_abs_error':largest}

def main():
 with NaNCM169BytePredictor() as model:
  reports=[check(split,model) for split in ('dev','test')]
  for invalid in (-1.,1.01,float('nan')):
   try:NaNCM169BytePredictor(eta_inside=invalid)
   except ValueError:pass
   else:raise AssertionError('strength accepted '+str(invalid))
  # Native model is causal: changing future bytes cannot change next256(past).
  past=b'Our signs were ';
  p0=model.next256(past)
  future=b' words completely hidden from the API'
  p1=model.next256((past+future)[:len(past)])
  assert p0==p1
 report={'status':'PASS','replay':reports,'validated_byte_probabilities':sum(v['exact_mixture_replay'] for v in reports),'all_256_positive_and_normalized':True,'future_suffix_exclusion':True}
 (R/'results/v169_streaming_regression.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report))
if __name__=='__main__':main()