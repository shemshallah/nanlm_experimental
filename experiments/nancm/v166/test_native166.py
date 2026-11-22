#!/usr/bin/env python3
"""Native v166 parity, fail-closed, normalized composition and DEV evaluation.
No TEST examples are used for model/alpha choice. Strict causal past-only begin.
"""
import argparse, ctypes, hashlib, json, math, os, random, sys, time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(R/'scripts'))
from train_gate163 import setup,winread
sys.path.insert(0,str(R/'parent/scripts'))
from test_native_speller156 import load,MODEL,F,getwords

BASE=R/'bin/libwordfield164.so';NEW=R/'bin/libwordfield166.so'
def open_env(strength):
    setup(core=False,domain=True,gate=None)
    os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V164_DISTRIBUTION_INDEX=str(R/'checkpoints/v164_native_distribution.tsv'),V166_LEXICAL_BIRTH=str(strength))
    if strength>0:os.environ['V166_BIRTH_MODEL']=str(R/'checkpoints/v166_lexical_birth.bin')
    else:os.environ.pop('V166_BIRTH_MODEL',None)
    l=load(NEW);h=l.wf148_open(*MODEL);assert h,('fail open',strength)
    l.wf166_candidate_count.argtypes=[ctypes.c_void_p];l.wf166_candidate_count.restype=ctypes.c_int
    l.wf166_candidate.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_double)]
    l.wf166_candidate.restype=ctypes.c_int
    return l,h

def arr(l,h,pre):
    a=(F*27)();e=F(1.);
    rc=l.wf148_next(h,pre,len(pre),a,ctypes.byref(e));assert rc==0,(rc,pre)
    p=[float(x)+float(e.value)/27 for x in a]
    assert all(x>=0 and math.isfinite(x) for x in p) and abs(sum(p)-1.)<2e-4,(sum(p),pre)
    return p

def bits(l,h,word):
    ret=0
    for i in range(len(word)+1):
        p=arr(l,h,word[:i].encode())
        j=26 if i==len(word) else ord(word[i])-97
        ret-=math.log2(max(1.e-25,p[j]))
    return ret

def samples():
    for bi,(raw,skip) in enumerate(winread(R/'external_book_gate/heldout_dev160.bin')):
        if bi>=24:break
        words=getwords(raw,skip)
        chosen=[words[min(len(words)-1,int((j+.5)*len(words)/min(20,len(words))))] for j in range(min(20,len(words)))] if words else []
        for pos,word in chosen:yield bi,raw[:pos],word

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='Small deterministic smoke, full frozen sweep remains default')
    args=ap.parse_args()
    start=time.time();sample=list(samples());assert len(sample)==480
    if args.quick: sample=sample[::20]
    lib,h=open_env(0)
    blib=load(BASE);bh=blib.wf148_open(*MODEL);assert bh
    # Exact v164 bitwise parity at 150 frozen native histories/prefixes.
    comparisons=0
    for bi,past,truth in sample[:150]:
        assert lib.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
        assert blib.wf148_begin(bh,ctypes.c_char_p(past),len(past),F(.25))>=0
        for i in range(min(5,len(truth))+1):
            pre=truth[:i].encode();a=(F*27)();b=(F*27)();ea=F();eb=F()
            assert lib.wf148_next(h,pre,len(pre),a,ctypes.byref(ea))==0
            assert blib.wf148_next(bh,pre,len(pre),b,ctypes.byref(eb))==0
            assert bytes(a)==bytes(b) and bytes(ea)==bytes(eb),'rollback diverged'
            comparisons+=1
    lib.wf148_close(h);blib.wf148_close(bh)
    # Malformed strength or sidecar must fail closed; never load degraded model.
    os.environ['V166_LEXICAL_BIRTH']='nan';assert not lib.wf148_open(*MODEL)
    os.environ['V166_LEXICAL_BIRTH']='.05';os.environ['V166_BIRTH_MODEL']='/nonexistent/v166.bin';assert not lib.wf148_open(*MODEL)
    report={'samples':len(sample),'books':24,'rollback_vectors_exact':comparisons,'active_domain_positions':None,'strengths':{},'fail_closed':True,'status':'experimental','quick':args.quick}
    for strength in ((0.0,0.05,0.5) if args.quick else (0.0,0.02,0.05,0.10,0.20,0.35,0.50)):
        l,h=open_env(strength);loss=0;gen_active=0;candidate_hit={5:0,32:0,64:0,96:0};birth_outside_v164=0
        for bi,past,word in sample:
            assert l.wf148_begin(h,ctypes.c_char_p(past),len(past),F(.25))>=0
            loss+=bits(l,h,word)
            count=l.wf166_candidate_count(h)
            gen_active+=int(count>0)
            if count:
                ranked=[]
                for ix in range(count):
                    buf=ctypes.create_string_buffer(64);p=ctypes.c_double()
                    assert l.wf166_candidate(h,ix,buf,64,ctypes.byref(p))==0
                    assert 0<p.value<=1
                    ranked.append(buf.value.decode())
                for k in candidate_hit:candidate_hit[k]+=int(word in ranked[:k])
        l.wf148_close(h)
        total=sum(len(w)+1 for _,_,w in sample)
        report['strengths'][str(strength)]={'bits_per_spelling_symbol':loss/total,'proposal_active':gen_active,'candidate_hits':candidate_hit,'n':len(sample)}
        print('STRENGTH',strength,report['strengths'][str(strength)],flush=True)
    report.update({'elapsed_s':round(time.time()-start,2),'warning':'spelling/EOS bits/symbol, NOT full 256-byte BPB; repeated DEV sample only'})
    (R/('results/v166_native_smoke.json' if args.quick else 'results/v166_native_dev.json')).write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()