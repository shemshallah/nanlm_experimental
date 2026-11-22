#!/usr/bin/env python3
"""Frozen native 96-beam and teacher-forced spelling/EOS validation.
DEV selects V164_BUDGET. TEST evaluated once after freeze; never use test to tune.
"""
import os,sys,json,time,ctypes,math,argparse
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'parent';sys.path.insert(0,str(P/'scripts'));sys.path.insert(0,str(R/'scripts'))
from test_native_speller156 import load,MODEL,F,getwords,voc,wins
from beam_words156 import decode
from train_gate163 import setup,winread

def given_word_bits(l,h,word):
 bits=0.
 for i in range(len(word)+1):
  out=(F*27)();esc=F(1.);pre=word[:i].encode()
  assert l.wf148_next(h,pre,i,out,ctypes.byref(esc))==0
  j=26 if i==len(word) else ord(word[i])-97
  q=float(out[j])+float(esc.value)/27
  bits-=math.log2(max(q,1.e-25))
 return bits

def run(split,budgets=(.35,.6),per_window=20,combo=False,window_start=0,window_stop=24):
 setup(core=False,domain=True,gate=None)
 os.environ.update(V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'))
 opts=[('v158',R/'bin/libwordfield158.so',0.,0.),('v162',R/'bin/libwordfield162.so',.4,0.)]
 opts.extend((f'v164_{"combo" if combo else "solo"}_b{b}',R/'bin/libwordfield164.so',.4 if combo else 0.,b) for b in budgets)
 states={}
 for name,path,b162,b164 in opts:
  os.environ.update(V160_ASSOC_BUDGET=str(b162),V160_ASSOC_INDEX=str(R/'checkpoints/v161_unified.tsv'),V164_DISTRIBUTION_BUDGET=str(b164))
  l=load(path);h=l.wf148_open(*MODEL)
  assert h,(name,'fail open')
  states[name]=(l,h)
 n=0;nOOV=0;data=[];start=time.monotonic()
 path=R/'external_book_gate'/f'heldout_{split}160.bin'
 for window_i,(raw,skip) in enumerate(winread(path)):
  if window_i<window_start:continue
  if window_i>=window_stop:break
  words=getwords(raw,skip)
  if per_window: words=[words[min(len(words)-1,int((j+.5)*len(words)/min(len(words),per_window)))] for j in range(min(len(words),per_window))] if words else []
  for pos,word in words:
   preds={};bits={};nOOV+=int(word not in voc)
   for name,(l,h) in states.items():
    assert l.wf148_begin(h,ctypes.c_char_p(raw[:pos]),pos,F(.25))>=0
    bits[name]=given_word_bits(l,h,word)
    preds[name]=decode(l,h)[0]
   data.append({'word':word,'oov':word not in voc,'predictions':preds,'word_bits':bits})
 n=len(data)
 for l,h in states.values():l.wf148_close(h)
 result={'split':split,'n':n,'oov':nOOV,'books':window_stop-window_start,'window_range':[window_start,window_stop],'per_window_max':per_window,'native_beam':96,
         'trained_index_split':'500 TRAIN books; 51 DEV, 64 TEST isolated by book','model_config':{name:{'singleword_budget':b162,'distribution_budget':b164} for name,_,b162,b164 in opts},
         'combo':combo,'models':{},'paired':{},'elapsed_seconds':round(time.monotonic()-start,2)}
 for name,_,_,_ in opts:
  summary={'bits_per_spelling_symbol':round(sum(d['word_bits'][name] for d in data)/sum(len(d['word'])+1 for d in data),5)}
  for k in [1,5,32]:
   hits=sum(d['word'] in d['predictions'][name][:k] for d in data)
   summary[f'top{k}']={'hits':hits,'pct':round(100*hits/n,3)}
  summary['oov_top32_correct']=sum(d['oov'] and d['word'] in d['predictions'][name][:32] for d in data)
  result['models'][name]=summary
  if name!='v158':
   result['paired'][name]={f'top{k}':{
     'fixes':sum(d['word'] not in d['predictions']['v158'][:k] and d['word'] in d['predictions'][name][:k] for d in data),
     'regressions':sum(d['word'] in d['predictions']['v158'][:k] and d['word'] not in d['predictions'][name][:k] for d in data)} for k in (1,5,32)}
 out=R/'results'/f'v164_{split}_{"combo" if combo else "book"}_{window_start}_{window_stop}_native.json';out.write_text(json.dumps(result,indent=2));out.with_name(f'v164_{split}_{"combo" if combo else "book"}_{window_start}_{window_stop}_rows.jsonl').write_text('\n'.join(json.dumps(d) for d in data)+'\n')
 print(json.dumps(result,indent=2),flush=True);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('split',choices=['dev','test']);a.add_argument('--budgets',nargs='+',type=float,default=[.35,.6]);a.add_argument('--per-window',type=int,default=20);a.add_argument('--combo',action='store_true');a.add_argument('--window-start',type=int,default=0);a.add_argument('--window-stop',type=int,default=24);p=a.parse_args();run(p.split,tuple(p.budgets),p.per_window,p.combo,p.window_start,p.window_stop)
