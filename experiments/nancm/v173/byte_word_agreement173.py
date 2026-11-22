#!/usr/bin/env python3
"""Cross-field inference: strictly causal native BYTE head informs word start.
TRAIN-book validation selects gamma; DEV evaluated afterward, never used for selection.
The native byte head is called only with the preceding bytes, never target suffix.
"""
import ctypes,os,sys,time,math,json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS='32',V171_ANN_CANDIDATES='64',V172_ANN_RAILS='1',V172_TEACHER_SHARE='0')
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield172.so'))
lib.wf169_byte_open.argtypes=[ctypes.c_char_p]*4;lib.wf169_byte_open.restype=ctypes.c_void_p
lib.wf169_byte_close.argtypes=[ctypes.c_void_p]
lib.wf169_byte_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)];lib.wf169_byte_next.restype=ctypes.c_int
handle=lib.wf169_byte_open(*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode());assert handle
rtrain=[json.loads(s) for s in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
meta=[json.loads(s) for s in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(rtrain)]
assert all(a['book']==b['book'] and a['truth']==b['truth'] for a,b in zip(rtrain,meta))
rdev=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))

def calc(rows,histories,label):
 allout=[];start=time.perf_counter()
 for i,(row,past) in enumerate(zip(rows,histories)):
  o=(F*256)()
  assert lib.wf169_byte_next(handle,ctypes.c_char_p(past),len(past),o)==0,(label,i)
  probs=[float(o[97+j])+float(o[65+j]) for j in range(26)]
  assert all(x>0 and math.isfinite(x) for x in probs)
  allout.append(probs)
  if (i+1)%200==0:print('BYTES',label,i+1,len(rows),'s',round(time.perf_counter()-start,1),flush=True)
 return allout
Btr=calc(rtrain,[m['history'].encode('latin1') for m in meta],'train')
Bdv=calc(rdev,[wins[int(r['book'])][0][:r['pos']] for r in rdev],'dev')
lib.wf169_byte_close(handle)
books=sorted({r['book'] for r in rtrain},key=lambda b:hashlib.sha256(('v173-byte-calib-'+b).encode()).digest());cal=set(books[-20:])
fold=[i for i,r in enumerate(rtrain) if r['book'] in cal]

def predict(r,b,lam):
 return max(r['candidates'],key=lambda c:math.log(max(1e-16,c['p']))+lam*math.log(max(1e-9,b[ord(c['word'][0])-97])))['word']

def count(rows,byte,ids,lam):return sum(predict(rows[i],byte[i],lam)==rows[i]['truth'] for i in ids)
val=[(count(rtrain,Btr,fold,x),x) for x in [i/20 for i in range(41)]]
selected=max(val,key=lambda a:(a[0],-a[1]));full=list(range(len(rdev)))
report={'method':'causal byte-first-character agreement','scope':'archived portfolio candidate lists, not native 96-beam','train_rows':len(rtrain),'train_val_book_count':20,'train_val_positions':len(fold),'train_val_baseline':count(rtrain,Btr,fold,0.),'train_val_sweep':val,'selected_lam':selected[1],'selected_train_val_top1':selected[0],'dev_rows':len(rdev),'dev_base':count(rdev,Bdv,full,0.),'dev_selected':count(rdev,Bdv,full,selected[1]),'dev_first_letter_top1':sum(max(range(26),key=lambda j:Bdv[i][j])==ord(r['truth'][0])-97 for i,r in enumerate(rdev)),'CAVEAT':'DEV used as report not hyperparam selection, but historically exposed; all metrics rank archived v169 candidate subset'}
(ROOT/'results/v173_byte_word_agreement.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps({k:v for k,v in report.items() if k!='train_val_sweep'}),flush=True)