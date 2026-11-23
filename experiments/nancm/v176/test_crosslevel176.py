#!/usr/bin/env python3
"""Native trained v176 12k-word simplex parity and inherited rollback."""
import ctypes,json,hashlib,sys,argparse
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v175')]
from train_long175 import load_models,load_split,verify_corpus
from train_counterfactual176 import CounterfactualGate,features,mix_logprobs
from eval_crosslevel176 import CausalLetter
from eval_bpe_crosslevel176 import BPEFirst

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=300);p.add_argument('--bpe',action='store_true');args=p.parse_args();torch.set_num_threads(2)
 m,opt,names,fp,old=load_models()
 ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',weights_only=False,map_location='cpu');m.load_state_dict(ck['model_state_dict']);m.eval()
 gk=torch.load(ROOT/'checkpoints/v176_counterfactual_word_gate.pt',weights_only=False,map_location='cpu')
 gate=CounterfactualGate();gate.load_state_dict(gk['gate_state_dict']);gate.eval()
 source=ROOT/'results/v175_release3';verify_corpus(source,names)
 Xtr,Ytr,Btr=load_split('train',source);letter=CausalLetter(Xtr,Ytr,names)
 bpe=None
 if args.bpe:
  merges=json.loads((ROOT/'checkpoints/v176_v133_style_bpe.json').read_text())['merges']
  bpe=BPEFirst(Xtr,Ytr,names,merges)
 X,Y,B=load_split('val',source)
 lib=ctypes.CDLL(str(ROOT/'bin/libwordfield176.so'));oldlib=ctypes.CDLL(str(ROOT/'bin/libwordfield175.so'))
 for l in [lib,oldlib]:
  l.wf175_now_open.argtypes=[ctypes.c_char_p];l.wf175_now_open.restype=ctypes.c_void_p
  l.wf175_now_close.argtypes=[ctypes.c_void_p]
  l.wf175_now_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];l.wf175_now_predict.restype=ctypes.c_int
 lib.wf176_cross_open.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf176_cross_open.restype=ctypes.c_void_p
 lib.wf176_cross_close.argtypes=[ctypes.c_void_p]
 lib.wf176_cross_add_bpe.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.wf176_cross_add_bpe.restype=ctypes.c_int
 lib.wf176_cross_predict.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];lib.wf176_cross_predict.restype=ctypes.c_int
 side=str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode();calib=str(ROOT/'checkpoints/v176_native_crosslevel.bin').encode()
 h=lib.wf175_now_open(side);h0=oldlib.wf175_now_open(side);assert h and h0
 cross=lib.wf176_cross_open(h,calib);assert cross
 if args.bpe:assert lib.wf176_cross_add_bpe(cross,str(ROOT/'checkpoints/v176_native_bpefirst.bin').encode())==0
 assert not lib.wf176_cross_open(h,b'/no/file')
 arr=np.zeros((12000,),dtype='float32');arr2=arr.copy();oldarr=arr.copy()
 native=arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float));native0=arr2.ctypes.data_as(ctypes.POINTER(ctypes.c_float));ref=oldarr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
 rng=np.random.default_rng(176);ids=rng.choice(len(Y),size=min(args.n,len(Y)),replace=False)
 maxerr=0.;exact=0;rollback=0;word_top1=word_top5=0
 try:
  for j in ids:
   hist=X[j];raw=(' '.join(names[int(w)] if w>=0 else 'zzzzquuxwordunknown' for w in hist)+' ').encode('ascii')
   c=ctypes.c_char_p(raw)
   assert lib.wf176_cross_predict(cross,c,len(raw),native,12000)==0
   assert lib.wf175_now_predict(h,c,len(raw),native0,12000)==0
   assert oldlib.wf175_now_predict(h0,c,len(raw),ref,12000)==0
   assert np.array_equal(arr2,oldarr),'inherited v175 native ABI not bit exact'
   rollback+=1
   assert abs(float(arr.sum())-1.)<1e-5 and np.isfinite(arr).all() and (arr>=0).all()
   with torch.no_grad():
    x=torch.tensor(hist,dtype=torch.long)[None,:];echo,base,_=m(x,True)
    g=gate(features(x,base,echo));z=mix_logprobs(base,echo,g)
    prior=letter.batch(hist[None,:])
    letterbase=torch.zeros((1,26),dtype=torch.float32)
    letterbase.index_add_(1,torch.from_numpy(letter.letters),z.exp())
    ratio=(prior.clamp_min(1e-8).log()-letterbase.clamp_min(1e-8).log())[:,letter.letters]
    out=F.log_softmax(z+.5*ratio,dim=1)
    if bpe is not None:
     orig=bpe.batch(hist[None,:]);dist=torch.zeros((1,bpe.V),dtype=torch.float32)
     dist.index_add_(1,torch.from_numpy(bpe.token),out.exp())
     bias=(orig.clamp_min(1e-8).log()-dist.clamp_min(1e-8).log())[:,bpe.token]
     out=F.log_softmax(out+.35*bias,dim=1)
    p0=out.exp().numpy()[0]
   err=float(np.max(np.abs(p0-arr)));maxerr=max(maxerr,err)
   if err>2e-5:raise AssertionError(('native_full_parity',j,err))
   r0=np.argsort(-p0,kind='stable')[:5];r1=np.argsort(-arr,kind='stable')[:5]
   if not np.array_equal(r0,r1):raise AssertionError(('rank_parity',j,r0,r1))
   exact+=1;word_top1+=int(int(Y[j])==int(r1[0]));word_top5+=int(int(Y[j]) in r1)
 finally:
  lib.wf176_cross_close(cross);lib.wf175_now_close(h);oldlib.wf175_now_close(h0)
 result=dict(status='PASS',bpe_enabled=args.bpe,sampled_val_histories=len(ids),max_native_pytorch_error=maxerr,abs_tolerance=2e-5,
    normalized_12k_vectors=len(ids),exact_top1_top5_rank_vectors=exact,
    legacy_v175_bitexact_12k_rollback=rollback,val_sample_top1=word_top1,val_sample_top5=word_top5,
    signed_parent_fingerprint_match=True,word_dataset_cohort='NaNROUTER Release3 55 val books',test_not_inspected=True)
 (ROOT/('results/v176_crosslevel_native_bpe_parity.json' if args.bpe else 'results/v176_crosslevel_native_parity.json')).write_text(json.dumps(result,indent=2))
 print('V176 NATIVE CROSSLEVEL PARITY',json.dumps(result),flush=True)
if __name__=='__main__':main()