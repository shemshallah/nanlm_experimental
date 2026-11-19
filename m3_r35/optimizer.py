"""Verifiable function-preserving 32→34 Adam-group continuation."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from m3_default.predictor import R32_TRAINING_CHECKPOINT_SHA256
from m3_r35.local_feedback import NEW_GROUPS


class R35ContinuationAdam:
    def __init__(self,path:Path):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=R32_TRAINING_CHECKPOINT_SHA256:
            raise ValueError("R35 must inherit exactly the certified R32 checkpoint")
        with np.load(path,allow_pickle=False) as data:
            self.parent={k:data[k].copy() for k in data.files}
        self.immutable={k:self.parent[k].copy() for k in
                        ("theta","adam_m","adam_v","adam_steps")}
        if any(x.shape!=(32,) for x in self.immutable.values()):
            raise ValueError("R32 optimizer history incomplete")
        self.theta=np.zeros(NEW_GROUPS,dtype=np.float64)
        self.m=self.theta.copy()
        self.v=self.theta.copy()
        self.steps=0

    def update(self,gradient:np.ndarray,lr:float=.003)->None:
        g=np.asarray(gradient,dtype=np.float64)
        if g.shape!=(NEW_GROUPS,) or not np.isfinite(g).all() or not 0.<lr<.1:
            raise ValueError("invalid R35 gradient")
        self.steps+=1
        g=np.clip(g,-2.,2.)
        self.m=.9*self.m+.1*g
        self.v=.999*self.v+.001*g*g
        self.theta-=lr*(self.m/(1-.9**self.steps))/(np.sqrt(self.v/(1-.999**self.steps))+1.e-8)
        np.clip(self.theta,-1.5,1.5,out=self.theta)

    def save(self,path:Path,fit:tuple[int,int],gate:tuple[int,int],dev:tuple[int,int])->None:
        for k in self.immutable:
            if not np.array_equal(self.parent[k],self.immutable[k]):
                raise ValueError("inherited Adam history changed")
        data=dict(self.parent)
        for k,v in (("theta",self.theta),("adam_m",self.m),("adam_v",self.v)):
            data[k]=np.concatenate((self.immutable[k],v))
        data["adam_steps"]=np.concatenate((self.immutable["adam_steps"],
                                             np.full(NEW_GROUPS,self.steps,dtype=np.int64)))
        data["r35_parent_checkpoint_sha256"]=R32_TRAINING_CHECKPOINT_SHA256
        for k,v in (("r35_fit_indices",fit),("r35_gate_indices",gate),
                    ("r35_dev_indices",dev),("r34_sealed_test_indices",(572,596))):
            data[k]=np.asarray(v,dtype=np.int64)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**data)
