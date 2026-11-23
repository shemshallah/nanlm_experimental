#!/usr/bin/env python3
"""v180 frozen inherited 128+128 signed-cloud distribution, no retraining/TEST labels.
Gather neural top-64 and P(word) for separate causal TRAIN-derived ngram candidates.
"""
import argparse,ctypes,json,sys,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/nancm/v175'))
from train_long175 import load_models
F=ctypes.c_float

def cand_read(path):
 d=Path(path).read_bytes();assert d[:8]==b'NC180CAN'
 n=int.from_bytes(d[8:12],'little');dt=np.dtype([('book','<i4'),('pos','<i4'),('target','<i4'),('cand','<u4',(4,6))])
 arr=np.frombuffer(d,dtype=dt,offset=12);assert len(arr)==n
 return arr

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','val'],default='train')
 ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
 torch=None
 from train_long175 import load_models
 _,_,names,_,_=load_models()
 arr=cand_read(ROOT/f'results/v180/{"train_calib" if args.split=="train" else "val"}_candidates.bin')
 d=np.load(ROOT/f'results/v175_release3/{args.split}.npz');X=d['X'];Y=d['Y'];B=d['book'];P=d['word_pos'];pos={(int(b),int(p)):i for i,(b,p) in enumerate(zip(B,P))}
 rows=np.array([pos[(int(r['book']),int(r['pos']))] for r in arr]);assert (Y[rows]==arr['target']).all()
 if args.limit:
  rng=np.random.default_rng(180 if args.split=='train' else 181)
  books=np.unique(B[rows]);sample=[];each=max(1,args.limit//len(books))
  for book in books:
   ix=np.flatnonzero(B[rows]==book);sample.extend(rng.choice(ix,size=min(len(ix),each),replace=False).tolist())
  selected=np.array(sample[:args.limit],dtype=int);arr=arr[selected];rows=rows[selected]
 lib=ctypes.CDLL(str(ROOT/'bin/libwordfield176.so'))
 lib.wf175_now_open.argtypes=[ctypes.c_char_p];lib.wf175_now_open.restype=ctypes.c_void_p
 lib.wf175_now_close.argtypes=[ctypes.c_void_p]
 lib.wf176_cross_open.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf176_cross_open.restype=ctypes.c_void_p
 lib.wf176_cross_add_bpe.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf176_cross_add_bpe.restype=ctypes.c_int
 lib.wf176_cross_close.argtypes=[ctypes.c_void_p]
 lib.wf176_cross_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(F),ctypes.c_size_t];lib.wf176_cross_predict.restype=ctypes.c_int
 h=lib.wf175_now_open(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode());assert h
 c=lib.wf176_cross_open(h,str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode());assert c
 assert lib.wf176_cross_add_bpe(c,str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())==0
 probs=np.empty(12000,dtype='float32');pt=probs.ctypes.data_as(ctypes.POINTER(F))
 k=64;n=len(rows)
 ids=np.zeros((n,k),dtype='<i2');vals=np.zeros((n,k),dtype='<f4');nm=np.zeros((n,4,2),dtype='<f4');targetp=np.zeros(n,dtype='<f4')
 begin=time.monotonic()
 try:
  for z,j in enumerate(rows):
   raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in X[j])+' ').encode('ascii')
   rc=lib.wf176_cross_predict(c,ctypes.c_char_p(raw),len(raw),pt,12000)
   if rc!=0:raise RuntimeError((z,j,rc))
   ranked=np.argpartition(probs,-k)[-k:];ranked=ranked[np.argsort(-probs[ranked],kind='stable')]
   ids[z]=ranked;vals[z]=probs[ranked];targetp[z]=probs[int(Y[j])]
   for t in range(4):
    for s in range(2):
     idx=int(arr['cand'][z,t,s]);nm[z,t,s]=probs[idx] if idx<12000 else 0.
   if (z+1)%1000==0:print('PRED',args.split,z+1,n,'seconds',round(time.monotonic()-begin,2),flush=True)
 finally:
  lib.wf176_cross_close(c);lib.wf175_now_close(h)
 output=ROOT/f'results/v180/{args.split}_neural_{n}.npz'
 np.savez_compressed(output,rows=rows,book=arr['book'],pos=arr['pos'],target=arr['target'],ngram=arr['cand'],neural_top=ids,neural_probs=vals,ngram_probs=nm,target_prob=targetp)
 print('SAVED',output,'seconds',round(time.monotonic()-begin,2),'SHA',hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
 for sz in [1,5,10,20,32,64]:
  covered=np.any(ids[:,:sz]==arr['target'][:,None],axis=1)
  ng=np.any(arr['cand'][:,:,:2]==arr['target'][:,None,None],axis=(1,2))
  print('ORACLE',sz,'neural',round(float(covered.mean()),5),'union_ng',round(float(np.mean(covered|ng)),5),flush=True)

if __name__=='__main__':main()
