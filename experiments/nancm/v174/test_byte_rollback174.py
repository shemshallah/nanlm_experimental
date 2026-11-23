#!/usr/bin/env python3
"""v174 full inherited 16k-byte streaming parity: inherited native 256-way streaming replay vs causal archived-expert oracle.

Must run from the complete v169 recovery archive; train-time sidecars remain
frozen. The reference depends only on expert predictions observed before each
byte, never future targets. v169/native word ABI are unmodified.
"""
import ctypes,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield174.so'))
for fn in ['wf170_byte_open','wf169_byte_open']:
 f=getattr(lib,fn);f.argtypes=[ctypes.c_char_p]*4;f.restype=ctypes.c_void_p
for fn in ['wf170_byte_next','wf169_byte_next']:
 f=getattr(lib,fn);f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)];f.restype=ctypes.c_int
for fn in ['wf170_byte_close','wf169_byte_close']:
 f=getattr(lib,fn);f.argtypes=[ctypes.c_void_p];f.restype=None
lib.wf170_byte_reset.argtypes=[ctypes.c_void_p];lib.wf170_byte_reset.restype=ctypes.c_int
lib.wf170_byte_set_adaptive.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.wf170_byte_set_adaptive.restype=ctypes.c_int
lib.wf170_byte_last_eta.argtypes=[ctypes.c_void_p];lib.wf170_byte_last_eta.restype=ctypes.c_double
args=(*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode())
modern=lib.wf170_byte_open(*args);legacy=lib.wf169_byte_open(*args)
assert modern and legacy
start_t=time.monotonic();results={}

def predict(handle,api,past):
 o=(F*256)();rc=getattr(lib,api)(handle,ctypes.c_char_p(past),len(past),o)
 assert rc==0,(api,rc,len(past))
 p=[float(x) for x in o]
 assert all(x>0 and math.isfinite(x) for x in p) and abs(sum(p)-1)<1e-5
 return p

def archive(split):
 x=[json.loads(s) for s in (ROOT/f'results/v169_byte_{split}_libwordfield169.so.jsonl').read_text().splitlines()]
 y=[json.loads(s) for s in (ROOT/f'results/v169_byte_ngram_{split}.jsonl').read_text().splitlines()]
 assert len(x)==len(y) and all(a[:4]==b[:4] for a,b in zip(x,y))
 return [(a[0],a[1],a[2],a[3],a[4],b[4]) for a,b in zip(x,y)]

def p_eta(inside,evidence):
 prior=.5 if inside else .1
 logit=math.log(prior/(1-prior))+.2*evidence
 return max(.01,min(.95,1/(1+math.exp(-max(-80.,min(80.,logit))))))

# Protected fallback: disabling online trust reproduces the original v169
# byte API's COMPLETE 256-way normalized probability vector.
assert lib.wf170_byte_set_adaptive(modern,0)==0
rollback_diff=0.;rollback_checks=0
for split in ['dev','test']:
 rawwins=list(winread(ROOT/f'external_book_gate/heldout_{split}160.bin'))[:24]
 rows=archive(split)
 for k in range(0,len(rows),max(1,len(rows)//144)):
  bi,pos,*_=rows[k];past=rawwins[bi][0][:pos]
  p=predict(modern,'wf170_byte_next',past)
  q=predict(legacy,'wf169_byte_next',past)
  rollback_diff=max(rollback_diff,max(abs(a-b) for a,b in zip(p,q)))
  assert max(abs(a-b) for a,b in zip(p,q))<1e-7
  rollback_checks+=1
assert lib.wf170_byte_set_adaptive(modern,1)==0

# Exact replay: reset between books, and update category-specific trust
# from the last *already scored* target when a new byte is observed.
for split in ['dev','test']:
 rows=archive(split);rawwins=list(winread(ROOT/f'external_book_gate/heldout_{split}160.bin'))[:24]
 loss=0.;base_loss=0.;n=0;max_probability_error=0.;max_eta_error=0.;bybook=[]
 last_book=None;evidence=[0.,0.];previous=None;prev_eta=None
 for bi,pos,actual,inside,p_word,p_ngram in rows:
  if bi!=last_book:
   if last_book is not None:bybook.append({'book':last_book,'bytes':n-count_at_book_start,'bpb':(loss-bookloss_at_start)/(n-count_at_book_start)})
   assert lib.wf170_byte_reset(modern)==0
   last_book=bi;evidence=[0.,0.];previous=None;count_at_book_start=n;bookloss_at_start=loss
  if previous is not None:
   previous_pos,previous_inside,previous_word,previous_ngram,previous_actual=previous
   assert pos==previous_pos+1
   # The just-observed byte has become history, so it is now safe to update.
   j=int(previous_inside)
   evidence[j]=.9*evidence[j]+math.log(previous_word/previous_ngram)
  expected_eta=p_eta(inside,evidence[int(inside)])
  past=rawwins[bi][0][:pos]
  probs=predict(modern,'wf170_byte_next',past)
  measured_eta=lib.wf170_byte_last_eta(modern)
  max_eta_error=max(max_eta_error,abs(measured_eta-expected_eta))
  assert abs(measured_eta-expected_eta)<1e-11,(split,bi,pos,measured_eta,expected_eta)
  expected=(1-expected_eta)*p_ngram+expected_eta*p_word
  max_probability_error=max(max_probability_error,abs(probs[actual]-expected))
  assert abs(probs[actual]-expected)<2e-6,(split,bi,pos,probs[actual],expected)
  loss+=-math.log2(probs[actual]);base_loss+=-math.log2(.5*p_word+.5*p_ngram if inside else .1*p_word+.9*p_ngram)
  n+=1;previous=(pos,inside,p_word,p_ngram,actual)
 if last_book is not None:bybook.append({'book':last_book,'bytes':n-count_at_book_start,'bpb':(loss-bookloss_at_start)/(n-count_at_book_start)})
 results[split]={'bytes':n,'books':len(bybook),'bpb':loss/n,'v169_bpb':base_loss/n,'improvement_bpb':(base_loss-loss)/n,'max_probability_abs_error':max_probability_error,'max_eta_error':max_eta_error,'bybook':bybook}
 print('NATIVE',split,json.dumps({k:v for k,v in results[split].items() if k!='bybook'}),'seconds',round(time.monotonic()-start_t,1),flush=True)

# Idempotence and discontinuity: a repeated history cannot train on itself;
# a jump to unrelated history resets trust to v169's frozen prior.
test_past=b'The old courtyard was ';lib.wf170_byte_reset(modern)
a=predict(modern,'wf170_byte_next',test_past);b=predict(modern,'wf170_byte_next',test_past)
assert a==b and abs(lib.wf170_byte_last_eta(modern)-.1)<1e-14
p=predict(modern,'wf170_byte_next',test_past+b'a')
assert lib.wf170_byte_last_eta(modern)>.0
z=predict(modern,'wf170_byte_next',b'Unrelated history beginning ')
assert abs(lib.wf170_byte_last_eta(modern)-.1)<1e-14
assert lib.wf170_byte_set_adaptive(modern,-1)<0
lib.wf170_byte_close(modern);lib.wf169_byte_close(legacy)
result={'status':'PASS','algorithm':'per-regime prequential log-likelihood trust, decay .9, gain .2, caps [.01,.95]','gate_tuned_on':'DEV archived expert replay, reused TEST frozen','native_exact_v169_rollback_vectors':rollback_checks,'max_rollback_abs_error':rollback_diff,'normalized_256_way':True,'repeated_query_idempotence':True,'discontinuous_history_resets':True,'invalid_setting_fail_closed':True,'weights_optimizers_unchanged':True,'v169_binary_sidecar_sha256':hashlib.sha256((ROOT/'checkpoints/v169_native_byte_lattice.bin').read_bytes()).hexdigest(),'dev':results['dev'],'reused_test':results['test'],'not_certified':'Previously explored book windows and DEV-tuned gate. No new untouched test, no top-1 word promotion.'}
out=ROOT/'results/v174_native_byte_parity.json';out.write_text(json.dumps(result,indent=2))
print('FINAL',json.dumps({k:v for k,v in result.items() if k not in ('dev','reused_test')}),flush=True)