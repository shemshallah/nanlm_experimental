#!/usr/bin/env python3
"""Native v170 prequential regression against archived v169 expert outputs.
Requires full cumulative v169/v170 recovery files, not source-only checkout.
"""
import ctypes,hashlib,json,math,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield170.so'))
for fn in ('wf170_byte_open','wf169_byte_open'):
 f=getattr(lib,fn);f.argtypes=[ctypes.c_char_p]*4;f.restype=ctypes.c_void_p
for fn in ('wf170_byte_next','wf169_byte_next'):
 f=getattr(lib,fn);f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)];f.restype=ctypes.c_int
for fn in ('wf170_byte_close','wf169_byte_close'):
 f=getattr(lib,fn);f.argtypes=[ctypes.c_void_p];f.restype=None
lib.wf170_byte_reset.argtypes=[ctypes.c_void_p];lib.wf170_byte_reset.restype=ctypes.c_int
lib.wf170_byte_set_adaptive.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.wf170_byte_set_adaptive.restype=ctypes.c_int
lib.wf170_byte_last_eta.argtypes=[ctypes.c_void_p];lib.wf170_byte_last_eta.restype=ctypes.c_double
args=(*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode())
modern=lib.wf170_byte_open(*args);legacy=lib.wf169_byte_open(*args)
assert modern and legacy

def predict(handle,api,past):
 o=(F*256)()
 assert getattr(lib,api)(handle,ctypes.c_char_p(past),len(past),o)==0
 p=[float(x) for x in o]
 assert all(x>0 and math.isfinite(x) for x in p) and abs(sum(p)-1)<1e-5
 return p

def archives(split):
 x=[json.loads(s) for s in (ROOT/f'results/v169_byte_{split}_libwordfield169.so.jsonl').read_text().splitlines()]
 y=[json.loads(s) for s in (ROOT/f'results/v169_byte_ngram_{split}.jsonl').read_text().splitlines()]
 assert len(x)==len(y) and all(a[:4]==b[:4] for a,b in zip(x,y))
 return [(a[0],a[1],a[2],a[3],a[4],b[4]) for a,b in zip(x,y)]

def eta(inside,evidence):
 prior=.5 if inside else .1
 logit=math.log(prior/(1-prior))+.2*evidence
 return max(.01,min(.95,1/(1+math.exp(-max(-80.,min(80.,logit))))))

assert lib.wf170_byte_set_adaptive(modern,0)==0
rollback=0
for split in ('dev','test'):
 rawwins=list(winread(ROOT/f'external_book_gate/heldout_{split}160.bin'))[:24]
 rows=archives(split)
 for j in range(0,len(rows),max(1,len(rows)//144)):
  bi,pos,*_=rows[j];past=rawwins[bi][0][:pos]
  a=predict(modern,'wf170_byte_next',past)
  b=predict(legacy,'wf169_byte_next',past)
  assert a==b
  rollback+=1
assert rollback>=288
assert lib.wf170_byte_set_adaptive(modern,1)==0

results={}
for split in ('dev','test'):
 rows=archives(split)
 windows=list(winread(ROOT/f'external_book_gate/heldout_{split}160.bin'))[:24]
 last_book=None;prev=None;ev=[0.,0.];total=0.;base=0.;n=0;maxerr=0.;bybook=[]
 for bi,pos,actual,inside,pword,pgram in rows:
  if bi!=last_book:
   if last_book is not None:bybook.append({'book':last_book,'n':n-book_start_n,'bpb':(total-book_start_loss)/(n-book_start_n)})
   assert lib.wf170_byte_reset(modern)==0
   ev=[0.,0.];prev=None;last_book=bi;book_start_n=n;book_start_loss=total
  if prev:
   prev_pos,prev_inside,prev_word,prev_gram=prev
   assert pos==prev_pos+1
   idx=int(prev_inside)
   ev[idx]=.9*ev[idx]+math.log(prev_word/prev_gram)
  mix=eta(inside,ev[int(inside)])
  probs=predict(modern,'wf170_byte_next',windows[bi][0][:pos])
  assert abs(lib.wf170_byte_last_eta(modern)-mix)<1e-11
  oracle=(1-mix)*pgram+mix*pword
  maxerr=max(maxerr,abs(probs[actual]-oracle))
  assert abs(probs[actual]-oracle)<2e-6
  total+=-math.log2(probs[actual])
  base+=-math.log2((.5 if inside else .1)*pword+(.5 if inside else .9)*pgram)
  n+=1;prev=(pos,inside,pword,pgram)
 bybook.append({'book':last_book,'n':n-book_start_n,'bpb':(total-book_start_loss)/(n-book_start_n)})
 results[split]={'bytes':n,'books':len(bybook),'native_bpb':total/n,'v169_bpb':base/n,'delta_bpb':(total-base)/n,'max_reference_error':maxerr,'bybook':bybook}
 print('NATIVE',split,{k:v for k,v in results[split].items() if k!='bybook'},flush=True)
 assert results[split]['delta_bpb']<0.
# Non-training on identical prefixes and correct reset at discontinuities.
p=b'The courtyard was '
assert lib.wf170_byte_reset(modern)==0
a=predict(modern,'wf170_byte_next',p)
b=predict(modern,'wf170_byte_next',p)
assert a==b and abs(lib.wf170_byte_last_eta(modern)-.1)<1e-14
predict(modern,'wf170_byte_next',p+b'a')
predict(modern,'wf170_byte_next',b'An unrelated book ')
assert abs(lib.wf170_byte_last_eta(modern)-.1)<1e-14
assert lib.wf170_byte_set_adaptive(modern,-1)==-1
lib.wf170_byte_close(modern);lib.wf169_byte_close(legacy)
report={'status':'PASS','fallback_exact_vectors':rollback,'frozen_gate':{'decay':.9,'gain':.2,'clamp':[.01,.95]},'dev':results['dev'],'reused_test':results['test'],'not_certified':'DEV-selected and reused-TEST books, no pristine generalization or strict word top1 claim'}
(ROOT/'results/v170_native_streaming_results.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('dev','reused_test')}),flush=True)
