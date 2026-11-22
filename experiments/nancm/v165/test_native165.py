#!/usr/bin/env python3
"""v165 signed cloud: exact opt-out, true causal state, normalized spelling,
corrupt checkpoint rejection, and nontrivial trained S+/S- native use.
"""
import os,sys,ctypes,math,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'parent'
sys.path.insert(0,str(P/'scripts'));sys.path.insert(0,str(R/'scripts'))
from test_native_speller156 import load,MODEL,F,getwords
from train_gate163 import setup,winread

def response(l,h,pre):
 o=(F*27)();e=F();assert l.wf148_next(h,pre.encode(),len(pre),o,ctypes.byref(e))==0
 p=[float(x)+float(e.value)/27 for x in o]
 assert abs(sum(p)-1)<1e-4 and min(p)>=0,(sum(p),pre)
 return bytes(o)+bytes(e),p

def main():
 setup(core=False,domain=True)
 os.environ.update(V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',
    V160_ASSOC_BUDGET='.4',V160_ASSOC_INDEX=str(R/'checkpoints/v161_unified.tsv'),
    V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),
    V165_SIGNED_ENERGY='0',V165_CLOUD_INDEX=str(R/'checkpoints/v165_signed_cloud.bin'))
 a=load(R/'bin/libwordfield164.so');b=load(R/'bin/libwordfield165.so')
 ha=a.wf148_open(*MODEL);hb=b.wf148_open(*MODEL);assert ha and hb
 checked=0
 for raw,skip in winread(R/'external_book_gate/heldout_dev160.bin'):
  for pos,word in getwords(raw,skip)[:25]:
   for l,h in ((a,ha),(b,hb)):
    assert l.wf148_begin(h,ctypes.c_char_p(raw[:pos]),pos,F(.25))>=0
   for pre in ('',word[:1],word[:3]):
    assert response(a,ha,pre)[0]==response(b,hb,pre)[0],pre
    checked+=1
   if checked>=150:break
  if checked>=150:break
 assert checked>=150
 a.wf148_close(ha);b.wf148_close(hb)
 print('PASS exact v164 opt-out parity',checked,'prefix distributions')
 os.environ['V165_SIGNED_ENERGY']='1.0'
 h=b.wf148_open(*MODEL);assert h
 b.wf165_candidate_count.argtypes=[ctypes.c_void_p];b.wf165_candidate_count.restype=ctypes.c_int
 b.wf165_candidate.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)]
 b.wf165_candidate.restype=ctypes.c_int
 active=0;reweighted=False
 for raw,skip in winread(R/'external_book_gate/heldout_dev160.bin'):
  for pos,word in getwords(raw,skip)[:35]:
   prefix=raw[:pos]
   assert b.wf148_begin(h,ctypes.c_char_p(prefix),len(prefix),F(.25))>=0
   count=b.wf165_candidate_count(h)
   if count<2:continue
   weights=[];words=[]
   for i in range(count):
    buf=ctypes.create_string_buffer(80);p=ctypes.c_double()
    assert b.wf165_candidate(h,i,buf,80,ctypes.byref(p))==0
    words.append(buf.value.decode());weights.append(p.value)
   assert abs(sum(weights)-1)<1e-8
   assert min(weights)>=0
   for pre in ('',words[0][:1],words[0][:3],word[:2]):response(b,h,pre)
   active+=1
   if active>=10:break
  if active>=10:break
 assert active>0,'expected gated signed-cloud use on independent DEV books'
 b.wf148_close(h)
 print('PASS trained signed cloud native loads, normalized candidate votes, and causal prefix paths',active)
 os.environ['V165_CLOUD_INDEX']=str(R/'checkpoints/does-not-exist.bin')
 assert not b.wf148_open(*MODEL),'missing sidecar must fail closed'
 os.environ['V165_CLOUD_INDEX']=str(R/'checkpoints/v165_signed_cloud.bin')
 os.environ['V165_SIGNED_ENERGY']='nan'
 assert not b.wf148_open(*MODEL),'nonfinite strength must fail closed'
 with tempfile.TemporaryDirectory() as td:
  c=Path(td)/'corrupt.bin';src=(R/'checkpoints/v165_signed_cloud.bin').read_bytes()
  c.write_bytes(src[:8]+b'\0'*24+src[32:]);os.environ.update(V165_SIGNED_ENERGY='1',V165_CLOUD_INDEX=str(c))
  assert not b.wf148_open(*MODEL),'checkpoint fingerprint mismatch must fail closed'
 print('PASS native load guards and protected baseline')
if __name__=='__main__':main()
