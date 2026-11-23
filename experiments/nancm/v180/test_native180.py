#!/usr/bin/env python3
"""Real inherited signed-cloud native and normalized train-only word memory parity.
VAL previously explored; test split untouched. Exact 256-way byte oracle unchanged.
"""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'results/v180'
sys.path.insert(0,str(ROOT/'experiments/nancm/v175'))
from train_long175 import load_models
F=ctypes.c_float

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=500);ap.add_argument('--full',action='store_true');a=ap.parse_args();st=time.monotonic()
 _,_,names,_,_=load_models()
 d=np.load(R/'val_neural_13200.npz');v=np.load(ROOT/'results/v175_release3/val.npz');X=v['X'][d['rows']];Y=d['target'];N=len(Y) if a.full else a.n
 lib=ctypes.CDLL(str(ROOT/'bin/libwordfield180.so'))
 lib.wf175_now_open.argtypes=[ctypes.c_char_p];lib.wf175_now_open.restype=ctypes.c_void_p
 lib.wf175_now_close.argtypes=[ctypes.c_void_p]
 lib.wf176_cross_open.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf176_cross_open.restype=ctypes.c_void_p
 lib.wf176_cross_add_bpe.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf176_cross_add_bpe.restype=ctypes.c_int
 lib.wf176_cross_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.c_size_t];lib.wf176_cross_predict.restype=ctypes.c_int
 lib.wf176_cross_close.argtypes=[ctypes.c_void_p]
 lib.wf180_memory_open.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf180_memory_open.restype=ctypes.c_void_p
 lib.wf180_memory_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.c_size_t];lib.wf180_memory_predict.restype=ctypes.c_int
 lib.wf180_memory_close.argtypes=[ctypes.c_void_p]
 lib.wf180_memory_stats.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)];lib.wf180_memory_stats.restype=ctypes.c_int
 h=lib.wf175_now_open(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode());assert h
 cross=lib.wf176_cross_open(h,str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode());assert cross
 assert lib.wf176_cross_add_bpe(cross,str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())==0
 assert not lib.wf180_memory_open(cross,b'/not/a/file')
 model=lib.wf180_memory_open(cross,str(R/'train_observed_stream.bin').encode());assert model
 stat=(ctypes.c_uint32*4)();assert lib.wf180_memory_stats(model,stat)==0
 p=np.empty(12000,dtype=np.float32);q=np.empty(12000,dtype=np.float32);ptr=p.ctypes.data_as(ctypes.POINTER(F));pq=q.ctypes.data_as(ctypes.POINTER(F))
 rng=np.random.default_rng(180);idx=np.arange(len(Y)) if a.full else rng.choice(len(Y),size=min(N,len(Y)),replace=False)
 maxerr=0.;parentdiff=0.;exact_top1=0;parent_t1=0;loss=0.;parent_loss=0.
 try:
  for z,j in enumerate(idx):
   raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in X[j])+' ').encode()
   c=ctypes.c_char_p(raw)
   assert lib.wf176_cross_predict(cross,c,len(raw),pq,12000)==0
   assert lib.wf180_memory_predict(model,c,len(raw),ptr,12000)==0
   ng=d['ngram'][j];expected=q.astype('float64');scale=1.
   for k in range(4):
    total=int(ng[k,4]);
    if total==0:continue
    for t in (0,1):
     word=int(ng[k,t]);ifcount=int(ng[k,2+t]);
     if word<12000:scale-=.1*ifcount/total
   expected*=scale
   for k in range(4):
    total=int(ng[k,4]);
    if total==0:continue
    for t in (0,1):
     word=int(ng[k,t]);ifcount=int(ng[k,2+t]);
     if word<12000:expected[word]+=.1*ifcount/total
   err=float(np.max(np.abs(expected-p)));maxerr=max(maxerr,err)
   if err>2e-6:raise AssertionError(('full vector parity',j,err))
   parentdiff=max(parentdiff,float(abs(float(d['target_prob'][j])-float(q[int(Y[j])]))))
   assert abs(float(p.sum())-1.)<2e-5 and (p>=0).all() and np.isfinite(p).all()
   ix=int(np.argmax(p));exact_top1+=int(ix==int(Y[j]));parent_t1+=int(int(np.argmax(q))==int(Y[j]))
   loss-=np.log(max(1e-30,float(p[int(Y[j])])));parent_loss-=np.log(max(1e-30,float(q[int(Y[j])])));
   if (z+1)%2500==0:print('NATIVE',z+1,'s',round(time.monotonic()-st,1),flush=True)
 finally:
  lib.wf180_memory_close(model);lib.wf176_cross_close(cross);lib.wf175_now_close(h)
 result=dict(status='PASS',samples=len(idx),source='reused independent-book NaNROUTER Release3 VAL (no TEST)',
  max_complete_vector_parity_abs_err=maxerr,max_parent_target_saved_abs_err=parentdiff,
  trained_ngram_context_sizes=list(stat),native_word_top1=exact_top1,native_parent_top1=parent_t1,
  native_word_nll=loss/len(idx),native_parent_nll=parent_loss/len(idx),
  byte_model='inherited unchanged',model_weights='inherited unchanged',elapsed_seconds=time.monotonic()-st)
 print('NATIVE WORD FUSION',json.dumps(result,indent=2),flush=True)
 (R/('native_full_val180.json' if a.full else 'native_sample_val180.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
