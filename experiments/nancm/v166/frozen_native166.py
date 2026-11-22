#!/usr/bin/env python3
"""Native frozen TEST scored from previous DEV-selected arms, no tuning."""
import argparse,ctypes,json,math,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords


def run(split='test'):
    cfg=json.loads((Path(__file__).with_name('FROZEN_DEV_SELECTION.json')).read_text())
    arms={'v164':0.0,'v166_top1_guarded':cfg['arms']['top1_guarded']['V166_LEXICAL_BIRTH'],'v166_loss_first':cfg['arms']['loss_first']['V166_LEXICAL_BIRTH']}
    setup(core=False,domain=True)
    os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(R/'checkpoints/v166_lexical_birth.bin'))
    all_rows=[];t0=time.time()
    for bi,(raw,skip) in enumerate(winread(R/f'external_book_gate/heldout_{split}160.bin')):
        if bi>=24:break
        words=getwords(raw,skip)
        if not words:continue
        chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(len(words),20)))] for j in range(min(20,len(words)))]
        for pos,truth in chosen:all_rows.append((bi,raw[:pos],truth))
    assert len(all_rows)==480
    report={'split':split,'frozen_arms':arms,'samples':len(all_rows),'books':24,'results':{},'not_full_byte_bpb':True}
    for name,strength in arms.items():
        os.environ['V166_LEXICAL_BIRTH']=str(strength)
        lib=load(R/'bin/libwordfield166.so');h=lib.wf148_open(*MODEL);assert h
        loss=0;active=0
        lib.wf166_candidate_count.argtypes=[ctypes.c_void_p];lib.wf166_candidate_count.restype=ctypes.c_int
        for bi,past,truth in all_rows:
            assert lib.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
            active+=int(lib.wf166_candidate_count(h)>0)
            for i in range(len(truth)+1):
                a=(F*27)();e=F(1.);pre=truth[:i].encode()
                assert lib.wf148_next(h,pre,len(pre),a,ctypes.byref(e))==0
                j=26 if i==len(truth) else ord(truth[i])-97
                loss-=math.log2(max(1.e-25,a[j]+e.value/27))
        lib.wf148_close(h)
        total=sum(len(w)+1 for _,_,w in all_rows)
        report['results'][name]={'bits_per_spelling_symbol':loss/total,'active_positions':active}
        print('ARM',name,report['results'][name],flush=True)
    report['relative_delta']={k:100*(v['bits_per_spelling_symbol']/report['results']['v164']['bits_per_spelling_symbol']-1) for k,v in report['results'].items() if k!='v164'}
    report['elapsed_s']=round(time.time()-t0,2)
    p=R/f'results/v166_{split}_FROZEN.json';p.write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':run()