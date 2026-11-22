#!/usr/bin/env python3
"""NaNCM active native next-word CLI, with matched predecessor comparisons.
This is the actual C++ predictor via ctypes, not a Python surrogate head.
"""
import argparse,ctypes,json,math,os,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F
from beam_words156 import decode

def configure(mode):
    setup(core=False,domain=True,gate=None)
    os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),
        V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'),
        V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5')
    return {'v167':R/'bin/libwordfield167.so','v166':R/'bin/libwordfield166.so','v164':R/'bin/libwordfield164.so','v158':R/'bin/libwordfield158.so'}[mode]

def prob(l,h,word):
    value=1.
    for i in range(len(word)+1):
        pre=word[:i].encode('ascii');o=(F*27)();e=F()
        assert l.wf148_next(h,pre,len(pre),o,ctypes.byref(e))==0
        j=26 if i==len(word) else ord(word[i])-97
        value*=max(0.,o[j]+e.value/27)
    return value

def run(prompt,mode='v167',steps=1,beam=96,topk=5):
    path=configure(mode)
    if not path.exists():raise RuntimeError(f'Native {mode} binary missing: build from cumulative recovery archive using make {mode}')
    lib=load(path);h=lib.wf148_open(*MODEL)
    if not h:raise RuntimeError('Native model failed to open; check the configured model/fingerprint/checkpoints')
    if mode=='v167':
        for name in ('wf167_effective_alpha','wf167_route_score','wf167_parent_ratio','wf167_neural_top_q'):
            f=getattr(lib,name);f.argtypes=[ctypes.c_void_p];f.restype=ctypes.c_double
    history=prompt.encode('utf-8')
    if history and ((65<=history[-1]<=90) or (97<=history[-1]<=122)):
        history+=b' '
    generated=[]
    try:
        for _ in range(steps):
            rc=lib.wf148_begin(h,ctypes.c_char_p(history),len(history),F(.25))
            if rc<0:raise RuntimeError(f'Native model begin failed: {rc}')
            ranked,calls=decode(lib,h,beam=beam,steps=20,topk=max(topk,5))
            if not ranked:break
            row={'top': [{'word':w,'prob':round(prob(lib,h,w),8)} for w in ranked[:topk]],'native_prefix_calls':calls}
            if mode=='v167':
                row.update({'effective_alpha':lib.wf167_effective_alpha(h),'domain_score':lib.wf167_route_score(h),
                    'relative_agreement':lib.wf167_parent_ratio(h),'neural_max_p':lib.wf167_neural_top_q(h)})
            generated.append(row)
            history+=(ranked[0]+' ').encode('ascii')
    finally:lib.wf148_close(h)
    return {'mode':mode,'prompt':prompt,'steps':len(generated),'generated_text':history.decode('utf-8','replace'),'predictions':generated,
        'warning':'Experimental native word predictor. Top probabilities are word-level; not full-byte language model output.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prompt',required=True);p.add_argument('--mode',choices=['v167','v166','v164','v158'],default='v167');p.add_argument('--steps',type=int,default=1);p.add_argument('--topk',type=int,default=5);p.add_argument('--beam',type=int,default=96);p.add_argument('--output',type=Path)
    a=p.parse_args();assert 1<=a.steps<=30 and 1<=a.topk<=32 and 1<=a.beam<=128
    res=run(a.prompt,a.mode,a.steps,a.beam,a.topk)
    text=json.dumps(res,indent=2);print(text)
    if a.output:a.output.write_text(text+'\n')