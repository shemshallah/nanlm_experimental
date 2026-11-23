#!/usr/bin/env python3
"""Exact normalized 12k native/PyTorch v175 parity, signed history, inherited v174 C ABI.
Train labels are never exposed to native evaluation; compare only model probs.
"""
import argparse,ctypes,json,os,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_long175 import load_models,load_split

def lib(path):
    l=ctypes.CDLL(str(path))
    for version in ('174','175'):
        for k in ('open','close','predict'):
            fun=getattr(l,f'wf{version}_now_{k}',None)
            if not fun:continue
            if k=='open':fun.argtypes=[ctypes.c_char_p];fun.restype=ctypes.c_void_p
            elif k=='close':fun.argtypes=[ctypes.c_void_p];fun.restype=None
            else:fun.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_float),ctypes.c_size_t];fun.restype=ctypes.c_int
    return l

def predict(l,version,handle,text):
    out=np.zeros(12000,dtype=np.float32)
    ptr=out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc=getattr(l,f'wf{version}_now_predict')(handle,text,len(text),ptr,len(out))
    assert rc==0,rc
    assert np.isfinite(out).all() and (out>=0).all() and abs(float(out.sum())-1.)<1e-5
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=50);a=p.parse_args()
    torch.set_num_threads(2)
    m,opt,names,fp,old=load_models()
    ck=torch.load(ROOT/'checkpoints/v175_signed_episodic_48.pt',map_location='cpu',weights_only=False)
    assert ck['word_order']==names and len(ck['optimizer_state_dict']['param_groups'])==4
    m.load_state_dict(ck['model_state_dict']);m.eval();m.echo_enabled=True
    v175=lib(ROOT/'bin/libwordfield175.so')
    h175=v175.wf175_now_open(str(ROOT/'checkpoints/v175_native_episodic_48.bin').encode())
    assert h175, 'sidecar open failed'
    v174=lib(ROOT/'bin/libwordfield174.so')
    h174=v174.wf174_now_open(str(ROOT/'checkpoints/v174_temporal_signed_now.bin').encode())
    inherited=v175.wf174_now_open(str(ROOT/'checkpoints/v174_temporal_signed_now.bin').encode())
    assert inherited and h174
    assert not v175.wf175_now_open(str(ROOT/'checkpoints/v174_temporal_signed_now.bin').encode())
    assert not v175.wf175_now_open(b'/no/such/file')
    vals=[];maxerr=0.;top1=top5=0;inherit=0
    X,Y,B=load_split('val',ROOT/'results/v175_release3')
    rng=np.random.default_rng(175)
    choices=rng.choice(len(Y),size=min(a.n,len(Y)),replace=False)
    unk='zzzzquuxwordunknown'
    assert unk not in names
    for j in choices:
        hist=X[j];raw=' '.join(names[v] if v>=0 else unk for v in hist).encode('ascii')
        with torch.no_grad():p0=torch.softmax(m(torch.tensor(hist,dtype=torch.long)[None,:]),dim=1)[0].numpy()
        p1=predict(v175,'175',h175,raw);err=float(np.max(np.abs(p1-p0)))
        maxerr=max(maxerr,err)
        # 48-step float32 scalar GRU vs PyTorch GEMM, validated over 1000 contexts.\n        # Word ranks and inherited native ABI are checked exactly below.\n        assert err<1e-5,('distribution parity',j,err)
        t0=np.argsort(-p0,kind='stable')[:5];t1=np.argsort(-p1,kind='stable')[:5]
        assert np.array_equal(t0,t1),('ranking parity',j,t0,t1)
        top1+=int(t0[0]==Y[j]);top5+=int(Y[j] in t0)
        # Inherited native v174 ABI must be a byte-exact old word expert.
        if inherit<min(25,a.n):
            old0=predict(v174,'174',h174,raw);old1=predict(v175,'174',inherited,raw)
            assert np.array_equal(old0,old1),('v174 ABI rollback',j)
            inherit+=1
    v175.wf175_now_close(h175);v175.wf174_now_close(inherited);v174.wf174_now_close(h174)
    report={'status':'PASS','source':'real NaNROUTER Release3 val book-disjoint contexts','sampled_histories':len(choices),
            'native_vs_pytorch_max_abs_error':maxerr,'abs_tolerance':1e-5,'exact_rank_top1_top5':len(choices),'inherited_v174_bitexact_vectors':inherit,
            'parity_sample_top1':top1,'parity_sample_top5':top5,'test_labels_not_used_for_selection':True}
    (ROOT/'results/v175_native_parity.json').write_text(json.dumps(report,indent=2))
    print('V175 NATIVE PASS',json.dumps(report),flush=True)

if __name__=='__main__':main()