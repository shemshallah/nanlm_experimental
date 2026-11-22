#!/usr/bin/env python3
"""Extract pretarget gate features and TARGET likelihoods only from TRAIN books."""
import ctypes,hashlib,json,os,re,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
cohort=json.loads((R/'external_v163/COHORTS.json').read_text())
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5')
os.environ.pop('V168_GATE_FILE',None)
l=load(R/'bin/libwordfield168.so')
for name in ('wf168_neural_word_prob','wf168_parent_word_prob'):
 f=getattr(l,name);f.argtypes=[ctypes.c_void_p,ctypes.c_char_p];f.restype=ctypes.c_double
l.wf168_features.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_int];l.wf168_features.restype=ctypes.c_int
l.wf167_effective_alpha.argtypes=[ctypes.c_void_p];l.wf167_effective_alpha.restype=ctypes.c_double
h=l.wf148_open(*MODEL);assert h
rows=[];t0=time.monotonic()
for part in range(3):
 wins=list(winread(R/f'external_v163/v163_calibration_{part}.bin'))
 names=cohort['calibration_windows'][part]
 for bi,((raw,skip),info) in enumerate(zip(wins,names)):
  book=info['name'];assert book in set(cohort['calibration_books'])
  words=[(m.start(),m.group().lower().decode('ascii')) for m in re.finditer(rb'[A-Za-z]+',raw)]
  targets=[(pos,w) for j,(pos,w) in enumerate(words) if j>=3 and pos>=skip and len(w)<=20]
  # Up to 24 evenly distributed raw next-word events per window, not v164 index rows.
  if len(targets)>24:targets=[targets[min(len(targets)-1,int((j+.5)*len(targets)/24))] for j in range(24)]
  for pos,w in targets:
   past=raw[:pos];assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
   feats=(ctypes.c_double*7)();assert l.wf168_features(h,feats,7)==7
   alpha=l.wf167_effective_alpha(h)
   p=l.wf168_parent_word_prob(h,w.encode());q=l.wf168_neural_word_prob(h,w.encode())
   assert 0<=p<=1.0001 and 0<=q<=1.0001,(book,w,p,q)
   rows.append({'book':book,'word':w,'f':list(feats),'p':p,'q':q,'v167_alpha':alpha})
 if True:print('PART',part,'rows',len(rows),'seconds',round(time.monotonic()-t0,2),flush=True)
l.wf148_close(h)
assert len(rows)>4000
out=R/'results/v168_gate_calibration_rows.jsonl';out.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
print('FINAL',json.dumps({'rows':len(rows),'books':len(set(r['book'] for r in rows)),'active':sum(r['v167_alpha']>0 for r in rows),'in_vocabulary':sum(r['q']>0 for r in rows),'seconds':round(time.monotonic()-t0,2),'output':str(out)}),flush=True)