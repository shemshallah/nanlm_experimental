#!/usr/bin/env python3
"""Strict whole-word 96-beam paired benchmark with frozen v172 teacher.
TRAIN-book validation selects share; DEV is only then evaluated at that share.
"""
import os,sys,argparse,json,hashlib,collections,ctypes,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F
from beam_words156 import decode
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train-val','dev'],default='train-val');ap.add_argument('--share',type=float,required=True);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--neighbors',type=int,default=32);ap.add_argument('--rails',type=int,default=1);args=ap.parse_args()
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05',V171_ANN_NEIGHBORS=str(args.neighbors),V171_ANN_CANDIDATES='1024' if args.neighbors>32 else '64',V172_ANN_RAILS=str(args.rails),V172_TEACHER_SHARE=str(args.share),V172_TEACHER_MODEL=str(ROOT/'checkpoints/v172_native_signed_teacher.bin'))
l=load(ROOT/'bin/libwordfield172.so');h=l.wf148_open(*MODEL);assert h
if args.split=='train-val':
 orig=[json.loads(s) for s in (ROOT/'results/v169_portfolio_train.jsonl').read_text().splitlines()]
 pos=[json.loads(s) for s in (ROOT/'results/v172_train_context_positions.jsonl').read_text().splitlines()][:len(orig)]
 cal=set(sorted({r['book'] for r in orig},key=lambda b:hashlib.sha256(('v172-calib-'+b).encode()).digest())[-20:])
 items=[(m['history'].encode('latin1'),r) for r,m in zip(orig,pos) if r['book'] in cal]
else:
 rows=[json.loads(s) for s in (ROOT/'results/v169_portfolio_dev.jsonl').read_text().splitlines()]
 wins=list(winread(ROOT/'external_book_gate/heldout_dev160.bin'))
 items=[(wins[int(r['book'])][0][:r['pos']],r) for r in rows]
if args.limit:items=items[:args.limit]
print('START',args.split,'share',args.share,'rails',args.rails,'neighbors',args.neighbors,'n',len(items),flush=True)
t=time.perf_counter();met=collections.Counter();book=collections.defaultdict(collections.Counter)
try:
 for past,r in items:
  assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  words,calls=decode(l,h,beam=96,steps=20,topk=32)
  for c in (met,book[r['book']]):
   c['n']+=1;c['top1']+=r['truth'] in words[:1];c['top5']+=r['truth'] in words[:5];c['top32']+=r['truth'] in words[:32]
finally:l.wf148_close(h)
report={'split':args.split,'share':args.share,'rails':args.rails,'neighbors':args.neighbors,'metrics':dict(met),'time_s':round(time.perf_counter()-t,2),'bookwise':{k:dict(v) for k,v in book.items()},'status':'NOT_PROMOTED_RESEARCH_DIAGNOSTIC'}
file=ROOT/f'results/v172_teacher_beam_{args.split}_{args.share:.3f}_{args.rails}rails_{args.neighbors}nbr.json'
file.write_text(json.dumps(report,indent=2));print('FINAL',json.dumps({k:v for k,v in report.items() if k!='bookwise'}),flush=True)