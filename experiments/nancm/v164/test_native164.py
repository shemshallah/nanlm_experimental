#!/usr/bin/env python3
"""v164 exact rollback, loaded probabilistic multiword path, and causal invariance.
Must also pass with official NaNROUTER index when available.
"""
import os,ctypes,sys,tempfile,math,random
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'parent';sys.path.insert(0,str(P/'scripts'))
from test_native_speller156 import load,MODEL,F,wins,getwords

def env():
 os.environ.update(V155_HEAD_STRENGTH='.05',V155_PROTECT_TOP1='1',V155_VOCAB_HEAD=str(P/'checkpoints/native_cloud155/v155_vocab_head.bin'),V156_OPEN_SHARE='.35',V156_SPELLER=str(P/'checkpoints/speller156.bin'),V157_JOINT_HEAD=str(R/'checkpoints/v157_joint_native.bin'),V157_WORD_SHARE='.04',V157_CHAR_SHARE='.10',V158_POINTER_THRESHOLD='.8',V158_POINTER_BUDGET='.35',V158_TOPIC_BUDGET='.015',V159_BYTE_BUDGET='0',V159_CHAR_BUDGET='0',V160_ASSOC_BUDGET='0',V164_DISTRIBUTION_BUDGET='0',V164_DOMAIN_GATE='0')

def output(lib,h,prefix):
 p=(F*27)();e=F(1.)
 assert lib.wf148_next(h,prefix.encode(),len(prefix),p,ctypes.byref(e))==0
 v=[float(p[i])+float(e.value)/27 for i in range(27)]
 assert min(v)>=0 and abs(sum(v)-1)<1e-4,(sum(v),prefix)
 return bytes(p)+bytes(e),v

def main():
 env();base=load(R/'bin/libwordfield162.so');cand=load(R/'bin/libwordfield164.so')
 handles=[l.wf148_open(*MODEL) for l in [base,cand]];assert all(handles)
 n=0
 for name in ('win.bin','win_test.bin'):
  for raw,skip in wins(name):
   for pos,w in getwords(raw,skip):
    for l,h in zip([base,cand],handles):assert l.wf148_begin(h,ctypes.c_char_p(raw[:pos]),pos,F(.25))>=0
    for prefix in ('',w[:1],w[:3]):
     assert output(base,handles[0],prefix)[0]==output(cand,handles[1],prefix)[0]
     n+=1
    if n>=360:break
   if n>=360:break
  if n>=360:break
 for l,h in zip([base,cand],handles):l.wf148_close(h)
 print('PASS exact v162 rollback',n)
 with tempfile.TemporaryDirectory() as td:
  idx=Path(td)/'index.tsv';idx.write_text('N164DIST1\t10657386406110969289\t7997860904003825501\t16896651846733574110\n1\tthe\tforest\t8\t25\n1\tthe\twindow\t5\t25\n1\tthe\tgarden\t3\t25\n2\twas the\tbrave\t3\t4\n2\twas the\twonderful\t1\t4\n3\the was the\tbridge\t2\t2\n')
  os.environ['V164_DISTRIBUTION_BUDGET']='0.7';os.environ['V164_DISTRIBUTION_INDEX']=str(idx)
  lib=load(R/'bin/libwordfield164.so');h=lib.wf148_open(*MODEL);assert h
  lib.wf164_candidate_count.argtypes=[ctypes.c_void_p];lib.wf164_candidate_count.restype=ctypes.c_int
  prefix=b'The man and woman were running through the town when he was the '
  assert lib.wf148_begin(h,ctypes.c_char_p(prefix),len(prefix),F(.25))>=0
  assert lib.wf164_candidate_count(h)>=2
  for w in ('','b','br','bridge','f','forest','window','zzzzz'):
   output(lib,h,w)
  # Verify exact whole-word mixture identity, not just locally normalized chars.
  lib.wf164_candidate_count.argtypes=[ctypes.c_void_p];lib.wf164_candidate_count.restype=ctypes.c_int
  lib.wf164_candidates.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)];lib.wf164_candidates.restype=ctypes.c_int
  q={};alpha=0.
  for k in range(lib.wf164_candidate_count(h)):
   outword=ctypes.create_string_buffer(96);weight=ctypes.c_double();a=ctypes.c_double()
   assert lib.wf164_candidates(h,k,outword,96,ctypes.byref(weight),ctypes.byref(a))==0
   q[outword.value.decode()]=weight.value;alpha=a.value
  bas=base.wf148_open(*MODEL);assert bas
  assert base.wf148_begin(bas,ctypes.c_char_p(prefix),len(prefix),F(.25))>=0
  def word_prob(l,handle,word):
   prob=1.
   for j in range(len(word)+1):
    _,v=output(l,handle,word[:j]);prob*=v[26 if j==len(word) else ord(word[j])-97]
   return prob
  for word in ['bridge','brave','forest','window','badger']:
   old=word_prob(base,bas,word);new=word_prob(lib,h,word)
   expected=(1-alpha)*old+alpha*q.get(word,0.)
   assert abs(new-expected)<1e-5,(word,old,new,expected,alpha)
  base.wf148_close(bas)
  print('PASS exact normalized complete-word mixture identity',len(q),'candidates, alpha',round(alpha,4))
  # Changing only unknown future bytes (not passed to begin) cannot change output.
  before=output(lib,h,'b')[0]
  assert lib.wf148_begin(h,ctypes.c_char_p(prefix),len(prefix),F(.25))>=0
  assert output(lib,h,'b')[0]==before
  lib.wf148_close(h)
  # Missing and malformed external index must fail closed.
  os.environ['V164_DISTRIBUTION_INDEX']=str(Path(td)/'missing');assert not lib.wf148_open(*MODEL)
  os.environ['V164_DISTRIBUTION_INDEX']=str(idx);os.environ['V164_DISTRIBUTION_BUDGET']='nan';assert not lib.wf148_open(*MODEL)
  corrupt=Path(td)/'wrong_fingerprint.tsv'
  corrupt.write_text(idx.read_text().replace('10657386406110969289','1',1))
  os.environ['V164_DISTRIBUTION_INDEX']=str(corrupt);os.environ['V164_DISTRIBUTION_BUDGET']='0.7';assert not lib.wf148_open(*MODEL)
  os.environ['V164_DISTRIBUTION_BUDGET']='0.7';os.environ['V160_ASSOC_BUDGET']='.1';os.environ['V160_ASSOC_INDEX']='/missing';assert not lib.wf148_open(*MODEL)
 print('PASS loaded distribution, normalized EOS paths, deterministic causal responses, bad configuration fails closed')
if __name__=='__main__':main()
