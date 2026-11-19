"""Low-capacity local feedback for the unified R32 M3 causal predictor.

Retains the frozen TRAIN-only anchor in the same R32 state. Learns only the
local disagreement channels of R34; rejected global-only feedback is removed.
"""
from __future__ import annotations
import numpy as np
from m3_r34.feedback import M3R34Predictor

NEW_GROUPS=2


def local_readout(base:np.ndarray,features:np.ndarray,gains:np.ndarray)->np.ndarray:
    if base.shape!=(256,) or features.shape!=(NEW_GROUPS,256) or gains.shape!=(NEW_GROUPS,):
        raise ValueError("R35 local feedback shape mismatch")
    if not np.isfinite(base).all() or not np.isfinite(features).all() or not np.isfinite(gains).all():
        raise ValueError("nonfinite R35 feedback")
    if not np.any(gains):
        return base
    logits=base+gains@features
    logits-=float(logits.max())
    return logits-np.log(np.exp(logits).sum())


class M3R35Predictor(M3R34Predictor):
    """Same R32 local and frozen-anchor memory, exact R32 at zero R35 gains."""
    def __init__(self,prior,anchor,r32_gains,local_gains=None):
        self.local_gains=(np.zeros(NEW_GROUPS,dtype=np.float64) if local_gains is None
                          else np.asarray(local_gains,dtype=np.float64).copy())
        if self.local_gains.shape!=(NEW_GROUPS,) or not np.isfinite(self.local_gains).all():
            raise ValueError("invalid R35 gains")
        super().__init__(prior,anchor,r32_gains)

    def base_and_local_feedback(self)->tuple[np.ndarray,np.ndarray]:
        base,all_feedback=self.base_and_feedback()
        return base,all_feedback[1:3]

    def predict_logp(self,mode:str|None=None)->np.ndarray:
        if mode not in (None,"r35"):
            return super().predict_logp(mode)
        base,feat=self.base_and_local_feedback()
        return local_readout(base,feat,self.local_gains)
