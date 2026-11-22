#!/usr/bin/env python3
"""Native dev-only 96-beam next-word benchmark; no target text supplied at begin."""
import ctypes,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(R/'scripts'),str(R/'parent/scripts')]
from train_gate163 import setup,winread
from test_native_speller156 import load,MODEL,F,getwords
from beam_words156 import decode
setup(core=False,domain=True,gate=None)
os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5')
l=load(R/'bin/libwordfield168.so')
l.wf167_effective_alpha.argtypes=[ctypes.c_void_p];l.wf167_effective_alpha.restype=ctypes.c_double
rows=[]
for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
 if bi>=24:break
 words=getwords(raw,skip)
 sample=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
 for p,w in sample:rows.append((bi,raw[:p],w))
assert len(rows)==480
arms=[
 ('v167_protected',None,'1','.05','.5','.5','1.4','v166_lexical_birth.bin'),
 ('v168_book_trained_gate','v168_gate_candidate.txt','1','.05','.5','.5','1.4','v166_lexical_birth.bin'),
 ('v168_constant_half',None,'0','.5','.5','.5','1.4','v166_lexical_birth.bin'),
 ('v168_raw_checkpoint',None,'1','.05','.5','.5','1.4','v168_raw_candidate.bin')]
for name,gate,mode,alpha,high,threshold,domain,model in arms:
 os.environ.update(V166_BIRTH_MODEL=str(R/'checkpoints'/model),V166_LEXICAL_BIRTH=alpha,V167_GATE_MODE=mode,V167_HIGH_ALPHA=high,V167_TRUST_THRESHOLD=threshold,V167_DOMAIN_MIN=domain)
 if gate:os.environ['V168_GATE_FILE']=str(R/'checkpoints'/gate)
 else:os.environ.pop('V168_GATE_FILE',None)
 h=l.wf148_open(*MODEL);assert h,name
 t0=time.monotonic();loss=0;symbols=0;hits={1:0,5:0,32:0};active=0;alphas=[];bybook={};diag=[]
 for bi,past,truth in rows:
  assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
  a=l.wf167_effective_alpha(h);alphas.append(a);active+=int(a>0)
  for i in range(len(truth)+1):
   pre=truth[:i].encode();o=(F*27)();e=F();assert l.wf148_next(h,pre,len(pre),o,ctypes.byref(e))==0
   p=o[26] if i==len(truth) else o[ord(truth[i])-97]
   loss-=math.log2(max(1e-25,float(p)+e.value/27));symbols+=1
  pred,_=decode(l,h,beam=96,steps=20,topk=32)
  for k in hits:hits[k]+=int(truth in pred[:k])
  bybook.setdefault(bi,[0,0,0]);bybook[bi][0]+=int(truth in pred[:1]);bybook[bi][1]+=int(truth in pred[:5]);bybook[bi][2]+=int(truth in pred[:32])
  diag.append({'book':bi,'truth':truth,'top1':pred[0] if pred else '', 'alpha':a,'top5':pred[:5]})
 l.wf148_close(h)
 out={'arm':name,'n':len(rows),'top1':hits[1],'top5':hits[5],'top32':hits[32],'bits_per_spelling_symbol':loss/symbols,'active':active,'mean_alpha':sum(alphas)/len(alphas),'bybook':bybook,'seconds':round(time.monotonic()-t0,2)}
 (R/f'results/v168_dev_{name}.json').write_text(json.dumps(out,indent=2));(R/f'results/v168_dev_{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in diag))
 print('ARM',json.dumps({k:v for k,v in out.items() if k!='bybook'}),flush=True)