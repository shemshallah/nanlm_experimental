#!/usr/bin/env python3
"""Differentiable native signed cloud evaluator for NaNCM v155.
Reproduces cloud137.h forward at a word boundary, with fixed base GeCache
and trainable C(256,2) oriented S+ / S- tensors.
"""
from pathlib import Path
import struct
import numpy as np
import torch
from torch import nn

EDGES=32640;D=8
GE_DTYPE=np.dtype([('nb','<u2'),('nt','<u2'),('b','u1',(48,)),('t','u1',(128,)),('x','<f4',(48,8)),('bf','<f4',(48,)),('gate','<f4',(128,))])

def read_events(path):
    b=Path(path).read_bytes()
    if b[:8]!=b'N155GEO1' or struct.unpack_from('<I',b,8)[0]!=155:raise ValueError('wrong native event signature')
    stride=struct.unpack_from('<I',b,12)[0];ident=struct.unpack_from('<QQQ',b,16);n=struct.unpack_from('<Q',b,40)[0]
    if len(b)!=48+n*GE_DTYPE.itemsize:raise ValueError('native event length mismatch')
    return stride,ident,np.frombuffer(b,offset=48,dtype=GE_DTYPE).copy()

def read_cloud(path):
    b=Path(path).read_bytes()
    if b[:8]!=b'C137D8PM' or struct.unpack_from('<I',b,8)[0]!=137:raise ValueError('bad inherited native cloud')
    pos=36
    arrays={}
    for name,shape in [('plus',(EDGES,D)),('minus',(EDGES,D)),('bscale',(128,D)),('tbias',(128,D)),('head',(128,D,26)),('bias',(26,))]:
        n=int(np.prod(shape));arrays[name]=np.frombuffer(b,offset=pos,count=n,dtype='<f4').reshape(shape).copy();pos+=4*n
    if pos!=len(b):raise ValueError('native cloud truncated')
    return arrays

def read_geom(path):
    b=Path(path).read_bytes()
    if b[:8]!=b'G256PAIR':raise ValueError('bad geometry')
    return np.frombuffer(b,offset=40+8192*8*4,dtype='<f4',count=128*8).copy().reshape(128,8)


def pair(a,b):
    lo=torch.minimum(a,b);hi=torch.maximum(a,b)
    return lo*(511-lo)//2+(hi-lo-1)

class NativeCloudAdjoint(nn.Module):
    def __init__(self,cloud_file,geom_file):
        super().__init__()
        c=read_cloud(cloud_file)
        self.plus=nn.Parameter(torch.tensor(c['plus']))
        self.minus=nn.Parameter(torch.tensor(c['minus']))
        self.bscale=nn.Parameter(torch.tensor(c['bscale']))
        self.tbias=nn.Parameter(torch.tensor(c['tbias']))
        self.register_buffer('intake',torch.tensor(read_geom(geom_file)))
        self.register_buffer('orig_plus',self.plus.detach().clone())
        self.register_buffer('orig_minus',self.minus.detach().clone())
        self.register_buffer('orig_bscale',self.bscale.detach().clone())
        self.register_buffer('orig_tbias',self.tbias.detach().clone())

    def forward(self,ev):
        # Accept a compact structured NumPy slice from the EXACT native GeModel.
        # Native event features never access target/future words.
        device=self.plus.device
        nb=torch.tensor(ev['nb'].astype(np.int64),device=device)
        nt=torch.tensor(ev['nt'].astype(np.int64),device=device)
        N=len(ev);maxb=int(nb.max());maxt=int(nt.max())
        b=torch.tensor(ev['b'][:,:maxb].astype(np.int64),device=device)
        t=torch.tensor(ev['t'][:,:maxt].astype(np.int64),device=device)
        x=torch.tensor(ev['x'][:,:maxb].copy(),device=device)
        bf=torch.tensor(ev['bf'][:,:maxb].copy(),device=device)
        gates=torch.tensor(ev['gate'][:,:maxt].copy(),device=device)
        bm=torch.arange(maxb,device=device)[None,:]<nb[:,None]
        tm=torch.arange(maxt,device=device)[None,:]<nt[:,None]
        nbi=nb.to(torch.float32).clamp_min(1).rsqrt()[:,None,None]
        nti=nt.to(torch.float32).clamp_min(1).rsqrt()[:,None,None]

        h0=(x*self.bscale[b]+.5*bf[:,:,None]*self.intake[b]).tanh()*bm[:,:,None]
        # Native orientation on lower-lower pair a -> b.
        lsrc=128+b[:,None,:]
        ldst=128+b[:,:,None]
        ll=pair(lsrc,ldst)
        llweight=torch.where((lsrc<ldst)[...,None],self.plus[ll],self.minus[ll])
        llmask=bm[:,:,None]&bm[:,None,:]&(lsrc!=ldst)
        h1=(h0+.13*nbi*((h0[:,None,:,:]*llweight)*llmask[:,:,:,None]).sum(2)).tanh()*bm[:,:,None]

        ul=pair(t[:,:,None],128+b[:,None,:])
        ul_plus=self.plus[ul];ul_minus=self.minus[ul]
        u0=(self.tbias[t]+1.2*nbi*(h1[:,None,:,:]*ul_plus*bm[:,None,:,None]).sum(2)).tanh()*tm[:,:,None]

        usrc=t[:,None,:];udst=t[:,:,None]
        uu=pair(usrc,udst)
        uuweight=torch.where((usrc<udst)[...,None],self.plus[uu],self.minus[uu])
        uumask=tm[:,:,None]&tm[:,None,:]&(usrc!=udst)
        u1=(u0+.17*nti*(u0[:,None,:,:]*uuweight*uumask[:,:,:,None]).sum(2)).tanh()*tm[:,:,None]
        r=(h1+.45*nti*(u1[:,:,None,:]*ul_minus*tm[:,:,None,None]).sum(1)).tanh()*bm[:,:,None]
        u2=(u1+.80*nbi*(r[:,None,:,:]*ul_plus*bm[:,None,:,None]).sum(2)).tanh()*tm[:,:,None]

        # Exact live v154 signature recipe: upper, lower, S+, S-.
        gw=torch.maximum(gates,torch.tensor(1e-5,device=device))*tm
        upper=(u2*gw[:,:,None]).sum(1)/gw.sum(1,keepdim=True).clamp_min(1.)
        rm=(r.square().sum(-1)+1e-16).sqrt()+.01
        rm=rm*bm
        lower=(r*rm[:,:,None]).sum(1)/rm.sum(1,keepdim=True).clamp_min(1.)
        ew=torch.maximum(gates,torch.tensor(.01,device=device))*tm
        total=(ew.sum(1,keepdim=True)*nb[:,None]).clamp_min(1.)
        plus=(ew[:,:,None,None]*ul_plus*r[:,None,:,:]*bm[:,None,:,None]).sum((1,2))/total
        minus=(ew[:,:,None,None]*ul_minus*u2[:,:,None,:]*bm[:,None,:,None]).sum((1,2))/total
        z=torch.cat([v/(1e-8+v.square().sum(-1,keepdim=True)).sqrt() for v in [upper,lower,plus,minus]],dim=1)
        return z

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    _,fp,ev=read_events(root/'checkpoints/causal_gecache155_stride8.bin')
    from train_soft_metric154 import load_sidecar
    meta,rows=load_sidecar(root/'checkpoints/highrisk_native/v154_soft_8.bin')
    assert fp==meta
    net=NativeCloudAdjoint(root/'checkpoints/highrisk_native/v153_trained.cloud',root/'parent/geom700k.bin')
    torch.set_num_threads(3)
    for begin in [0,25,1000,10000]:
        end=min(begin+8,len(ev));z=net(ev[begin:end]);goal=torch.from_numpy(rows['z'][begin:end].copy())
        err=(z-goal).abs();print('PARITY',begin,'max_abs',err.max().item(),'mean_abs',err.mean().item())
        if err.max().item()>.0001:raise AssertionError('native cloud exact mapping mismatch')
    print('PASS exact native v154 S+/S- 32D cloud reconstruction')
