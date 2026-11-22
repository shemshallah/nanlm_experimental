#!/usr/bin/env python3
"""v164 must leave all historical documentation probability vectors bit-identical."""
import os,sys,ctypes,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'parent';sys.path.insert(0,str(P/'scripts'));sys.path.insert(0,str(R/'scripts'))
from test_native_speller156 import load,wins,getwords,MODEL,F
from train_gate163 import setup
setup(core=False,domain=True,gate=None)
os.environ.update(V160_ASSOC_BUDGET='.4',V160_ASSOC_INDEX=str(R/'checkpoints/v161_unified.tsv'),V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5')
parent=load(R/'bin/libwordfield158.so');cand=load(R/'bin/libwordfield164.so')
cand.wf164_candidate_count.argtypes=[ctypes.c_void_p];cand.wf164_candidate_count.restype=ctypes.c_int
cand.wf164_domain_score.argtypes=[ctypes.c_void_p];cand.wf164_domain_score.restype=ctypes.c_double
hs=[l.wf148_open(*MODEL) for l in [parent,cand]];assert all(hs)
checked=0;maxscore=-1e9;active=0
for fname in ['win.bin','win_test.bin']:
 for raw,skip in wins(fname):
  for pos,word in getwords(raw,skip):
   past=raw[:pos]
   for l,h in zip([parent,cand],hs):assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
   active+=cand.wf164_candidate_count(hs[1])>0
   maxscore=max(maxscore,cand.wf164_domain_score(hs[1]))
   for prefix in [' ',word[:1],word[:4]]:
    if prefix==' ':prefix=''
    v=[]
    for l,h in zip([parent,cand],hs):
     p=(F*27)();e=F(0.)
     assert l.wf148_next(h,prefix.encode(),len(prefix),p,ctypes.byref(e))==0
     v.append((bytes(p),bytes(e)))
    assert v[0]==v[1],(fname,word,prefix)
    checked+=1
for l,h in zip([parent,cand],hs):l.wf148_close(h)
assert active==0,(active,maxscore)
print(f'PASS v164 full historical documentation fallback {checked} exact prefix vectors; max book-domain score {maxscore:.6f}')
INHERITED_SHA256={
 'checkpoints/best157.pt':'1542abb136fd24341ca92f6984145e7b05b3e3c30d460f11297038506e23ef3f',
 'checkpoints/cloud_joint157_best.pt':'f4bd3c3e4f486dabb0f039e6028cacc33173acf590d92d85ecf9a096967aeab1',
 'checkpoints/v157_joint_native.bin':'530ae274c8ca8bc515de1adbafdc30d13918fd56e3ba632d7e80726a97cba22c',
 'parent/checkpoints/native_cloud155/v155_joint_trained.cloud.opt':'0e2ef782499e1d1fb0561cf28837f2196aa50f0330a20454d42db7ad4011acc0',
 'parent/checkpoints/native_cloud155/v155_joint_trained.cloud':'78c2f81b7ba13bcbd4c4accf2746628514d9e513cab700cc8f4df041e31f8efc',
 'parent/checkpoints/native_cloud155/v155_vocab_head.bin':'b6658a6f32f1003dc0d120dc079485da3dce407a9697bf0a2225c1ef9c05479b'
}
for filename,digest in INHERITED_SHA256.items():
 assert hashlib.sha256((R/filename).read_bytes()).hexdigest()==digest,filename
print('PASS six inherited model/cloud/optimizer artifacts unchanged')
