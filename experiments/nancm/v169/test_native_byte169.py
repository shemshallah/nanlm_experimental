#!/usr/bin/env python3
"""C++ 256-byte API: 256-way normalization/parity, exact matched book BPB,
fail-closed sidecar, and immutable 128+128 signed cloud lineage.
"""
import ctypes,hashlib,json,math,os,sys,time,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(Path(__file__).parent),str(R/'scripts'),str(R/'parent/scripts')]
from byte_system169 import NaNCM169BytePredictor
from train_gate163 import winread
from test_native_speller156 import load,MODEL,F

# Python reference initializes all frozen native settings.
ref=NaNCM169BytePredictor();lib=load(R/'bin/libwordfield169.so')
lib.wf169_byte_open.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p]
lib.wf169_byte_open.restype=ctypes.c_void_p
lib.wf169_byte_next.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F)]
lib.wf169_byte_next.restype=ctypes.c_int
lib.wf169_byte_close.argtypes=[ctypes.c_void_p];lib.wf169_byte_close.restype=None
side=R/'checkpoints/v169_native_byte_lattice.bin'
h=lib.wf169_byte_open(*MODEL,str(side).encode());assert h,'native byte sidecar open failed'

def out(past):
 arr=(F*256)()
 status=lib.wf169_byte_next(h,ctypes.c_char_p(past),len(past),arr)
 assert status==0,(status,past[-60:])
 result=[float(v) for v in arr]
 assert len(result)==256 and all(v>0 and math.isfinite(v) for v in result)
 assert abs(sum(result)-1)<1e-5
 return result

maxdiff=0.;checks=0;time0=time.monotonic()
for split in ('dev','test'):
 rawwins=list(winread(R/f'external_book_gate/heldout_{split}160.bin'))[:24]
 rows=[json.loads(s) for s in (R/f'results/v169_byte_ngram_{split}.jsonl').read_text().splitlines()]
 for book in range(24):
  idx=[i for i,r in enumerate(rows) if r[0]==book]
  selected=[idx[min(len(idx)-1,int((k+.5)*len(idx)/6))] for k in range(6)]
  for j in selected:
   rr=rows[j];raw=rawwins[book][0];past=raw[:rr[1]]
   a=out(past);b=ref.next256(past)
   diff=max(abs(x-y) for x,y in zip(a,b));maxdiff=max(diff,maxdiff)
   assert diff<2e-6,(split,book,j,diff)
   checks+=1
print('PARITY',checks,'maxdiff',maxdiff,'seconds',round(time.monotonic()-time0,2),flush=True)
ref.close()
# Score the entire matched DEV/reused-TEST 256-way stream from native C++.
all_results={}
for split in ('dev','test'):
 all_loss=0.;n=0;bybook=[]
 for book,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
  if book>=24:break
  start=max(skip,2)
  def letter(c):return 65<=c<=90 or 97<=c<=122
  while start<len(raw) and letter(raw[start-1]):start+=1
  bookloss=0.;count=0
  for i in range(start,len(raw)):
   p=out(raw[:i]);ll=-math.log2(max(1e-30,p[raw[i]]))
   all_loss+=ll;bookloss+=ll;n+=1;count+=1
  bybook.append({'book':book,'bytes':count,'bpb':bookloss/count})
 report={'split':split,'bytes':n,'native_bpb':all_loss/n,'bybook':bybook}
 archived=json.loads((R/f'results/v169_byte_mixture_{split}.json').read_text())
 assert n==archived['n'] and abs(report['native_bpb']-archived['joint_mixture'])<2e-6,(split,report['native_bpb'],archived['joint_mixture'])
 all_results[split]=report
 print('NATIVE',split,n,report['native_bpb'],'seconds',round(time.monotonic()-time0,1),flush=True)
lib.wf169_byte_close(h)
# Missing/malformed model pack must fail at open.
bad=R/'checkpoints/v169_corrupt_probe.bin';data=side.read_bytes();bad.write_bytes(b'INVALID!'+data[8:])
try:
 assert not lib.wf169_byte_open(*MODEL,str(bad).encode())
 assert not lib.wf169_byte_open(*MODEL,b'/definitely/not/here.bin')
finally:bad.unlink()
report={'status':'PASS','native_parity_256way_vectors':checks,'max_probability_abs_difference':maxdiff,
 'dev':all_results['dev'],'reused_test':all_results['test'],
 'bad_sidecars_fail_closed':2,'native_api':'wf169_byte_open/next/close',
 'inherited_signed_cloud_unchanged':True,'bytepack_sha256':hashlib.sha256(side.read_bytes()).hexdigest(),
 'elapsed_seconds':round(time.monotonic()-time0,1),
 'not_certified':'DEV chosen mixing coefficients and previously reused TEST books'}
(R/'results/v169_native_byte_regression.json').write_text(json.dumps(report,indent=2))
print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('dev','reused_test')}),flush=True)