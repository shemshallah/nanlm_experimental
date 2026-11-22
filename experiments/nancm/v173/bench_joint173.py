#!/usr/bin/env python3
"""Locked gamma from TRAIN-book agreement; native 96-beam full-word ranking.
No future bytes passed to the decoder. DEV historically exposed; not a fresh test.
"""
import ctypes,os,sys,json,collections,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import MODEL,F
from beam_words156 import decode
ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['train-val','dev'],required=True);ap.add_argument('--gamma',type=float,default=.1);args=ap.parse_args()
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS='32',V171_ANN_CANDIDATES='64',V172_ANN_RAILS='1',V172_TEACHER_SHARE='0')
from ctypes import c_void_p,c_char_p,c_size_t,c_double,POINTER
lib=ctypes.CDLL(str(ROOT/'bin/libwordfield173.so'))
lib.wf173_joint_open.argtypes=[c_char_p]*4+[c_double];lib.wf173_joint_open.restype=c_void_p
lib.wf173_joint_close.argtypes=[c_void_p]
lib.wf173_joint_begin.argtypes=[c_void_p,c_void_p,c_size_t,F];lib.wf173_joint_begin.restype=ctypes.c_int
lib.wf173_joint_next.argtypes=[c_void_p,c_char_p,c_size_t,POINTER(F),POINTER(F)];lib.wf173_joint_next.restype=ctypes.c_int
h=lib.wf173_joint_open(*MODEL,str(ROOT/'checkpoints/v169_native_byte_lattice.bin').encode(),args.gamma);assert h
class API:
 wf148_next=lib.wf173_joint_next
if args.cohort=='train-val':
 rows=[json.loads(z) for z in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
 positions=[json.loads(z) for z in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(rows)]
 books=sorted({r['book'] for r in rows},key=lambda b:hashlib.sha256(('v173-byte-calib-'+b).encode()).digest());cal=set(books[-20:])
 items=[(p['history'].encode('latin1'),r) for p,r in zip(positions,rows) if r['book'] in cal]
else:
 rows=[json.loads(z) for z in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
 wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
 items=[(wins[int(r['book'])][0][:r['pos']],r) for r in rows]
metric=collections.Counter();bybook=collections.defaultdict(collections.Counter);start=time.perf_counter()
try:
 for i,(past,row) in enumerate(items):
  assert lib.wf173_joint_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  words,calls=decode(API,h,beam=96,steps=20,topk=32)
  for c in (metric,bybook[row['book']]):
   c['n']+=1;c['top1']+=int(row['truth'] in words[:1]);c['top5']+=int(row['truth'] in words[:5]);c['top32']+=int(row['truth'] in words[:32]);c['parent_top1']+=int(row['truth']==row['baseline']);c['native_calls']+=calls
  if (i+1)%50==0:print('BEAM',args.cohort,i+1,'sec',round(time.perf_counter()-start,1),'top1',metric['top1'],flush=True)
finally:lib.wf173_joint_close(h)
report={'cohort':args.cohort,'gamma':args.gamma,'method':'native normalized first-letter product, frozen train selected gamma','metric':dict(metric),'per_book':{str(k):dict(v) for k,v in bybook.items()},'time_s':round(time.perf_counter()-start,2),'full_96beam':True,'not_fresh_test':True}
path=ROOT/f'results/v173_joint_beam_{args.cohort}.json';path.write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps({k:v for k,v in report.items() if k!='per_book'}),flush=True)