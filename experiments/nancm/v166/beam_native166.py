#!/usr/bin/env python3
"""Strict native, no-target-prefix 96-beam word decoding for experimental v166.
DEV tuning only, frozen TEST audit; not full-byte BPB.
"""
import argparse,ctypes,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords,voc
from beam_words156 import decode

def run(split='dev',strength=.5,limit=0,beam=96):
    setup(core=False,domain=True,gate=None)
    os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V166_LEXICAL_BIRTH=str(strength),V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'))
    lib=load(R/'bin/libwordfield166.so');h=lib.wf148_open(*MODEL);assert h
    n=0;hits={1:0,5:0,32:0};oov=0;calls=0;start=time.time();rows=[]
    for book,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
        if book>=24:break
        words=getwords(raw,skip)
        if not words:continue
        words=[words[min(len(words)-1,int((j+.5)*len(words)/min(len(words),20)))] for j in range(min(len(words),20))]
        for pos,truth in words:
            past=raw[:pos];assert lib.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
            ranked,ncalls=decode(lib,h,beam=beam,steps=20,topk=32)
            n+=1;calls+=ncalls;oov+=int(truth not in voc)
            for k in hits:hits[k]+=int(truth in ranked[:k])
            rows.append({'word':truth,'book':book,'top32':ranked,'out_of_vocab':truth not in voc})
            if n%20==0:
                print('PROGRESS',n,'top1',hits[1],'top5',hits[5],'top32',hits[32],'seconds',round(time.time()-start,2),flush=True)
            if limit and n>=limit:break
        if limit and n>=limit:break
    lib.wf148_close(h)
    result={'split':split,'strength':strength,'n':n,'book_count':len(set(x['book'] for x in rows)),'beam':beam,'max_length':20,'top1_hits':hits[1],'top5_hits':hits[5],'top32_hits':hits[32],
     'top1_pct':100*hits[1]/n,'top5_pct':100*hits[5]/n,'top32_pct':100*hits[32]/n,'oov':oov,'native_prefix_calls':calls,'elapsed':round(time.time()-start,2),
     'warning':'Previously reused book DEV/TEST windows; not pristine and not whole-byte BPB.'}
    path=R/f'results/v166_{split}_beam{n}.json';path.write_text(json.dumps(result,indent=2));path.with_suffix('.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='dev',choices=['dev','test']);p.add_argument('--strength',type=float,default=.5);p.add_argument('--limit',type=int,default=0);p.add_argument('--beam',type=int,default=96);a=p.parse_args();run(a.split,a.strength,a.limit,a.beam)