"""Causal symbiotic feedback on one inherited R32 language predictor.

The document-local lexical state queries a frozen TRAIN-only anchor; anchor
confidence and disagreements then feed a learned corrective readout back to
the same local predictor. The four new gains start at exactly zero.
"""
from __future__ import annotations
import numpy as np
from m3_default.predictor import R29_THETA,R31_THETA
from m3_r29.lexical import WordSuccessorMemory,pack_lexical,lexical_readout
from m3_r30.router import support
from m3_r31.calibration import class_features,calibrated_readout
from m3_r32.anchor import M3R32Predictor,anchor_features,anchored_readout

NEW_GROUPS=4


def feedback_features(r26:np.ndarray,local:np.ndarray,current:WordSuccessorMemory,
                      anchor:WordSuccessorMemory,r31:np.ndarray,r32:np.ndarray)->np.ndarray:
    if any(x.shape!=(256,) for x in (r26,r31,r32)) or local.shape!=(8,256):
        raise ValueError("R34 requires 256-way logits and eight lexical channels")
    if not all(np.isfinite(x).all() for x in (r26,r31,r32,local)):
        raise ValueError("nonfinite R34 features")
    ls=support(current)
    gs=support(anchor,query=current)
    global_lex=pack_lexical(anchor.features(r26,query=current))
    l=R29_THETA@local
    g=R29_THETA@global_lex
    delta=r32-r31
    return np.stack((gs*(1-ls)*delta,
                     gs*ls*np.tanh(g-l)*delta,
                     gs*ls*(l-g),
                     (1-gs)*delta))


def feedback_readout(r32:np.ndarray,features:np.ndarray,gains:np.ndarray)->np.ndarray:
    if r32.shape!=(256,) or features.shape!=(NEW_GROUPS,256) or gains.shape!=(NEW_GROUPS,):
        raise ValueError("R34 feedback shape mismatch")
    if not np.isfinite(features).all() or not np.isfinite(gains).all():
        raise ValueError("nonfinite R34 feedback")
    if not np.any(gains):
        return r32
    logits=r32+gains@features
    logits-=float(logits.max())
    return logits-np.log(np.exp(logits).sum())


class M3R34Predictor(M3R32Predictor):
    """R32 state with 4 extra feedback gains; identical at initialization."""
    def __init__(self,prior:np.ndarray,anchor:WordSuccessorMemory,
                 r32_gains:np.ndarray,feedback_gains:np.ndarray|None=None):
        self.feedback_gains=(np.zeros(NEW_GROUPS,dtype=np.float64) if feedback_gains is None
                             else np.asarray(feedback_gains,dtype=np.float64).copy())
        if self.feedback_gains.shape!=(NEW_GROUPS,) or not np.isfinite(self.feedback_gains).all():
            raise ValueError("invalid R34 gains")
        super().__init__(prior,anchor,r32_gains)

    def base_and_feedback(self)->tuple[np.ndarray,np.ndarray]:
        r26,local=self.base_and_lexical_features()
        r29=lexical_readout(r26,local,R29_THETA)
        r31=calibrated_readout(r29,class_features(local,support(self.lexical)),R31_THETA)
        r32=anchored_readout(r31,anchor_features(r26,local,self.lexical,self.anchor),self.anchor_theta)
        return r32,feedback_features(r26,local,self.lexical,self.anchor,r31,r32)

    def predict_logp(self,mode:str|None=None)->np.ndarray:
        if mode not in (None,"r34"):
            return super().predict_logp(mode)
        base,features=self.base_and_feedback()
        return feedback_readout(base,features,self.feedback_gains)
