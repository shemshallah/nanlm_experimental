"""Event-level Bayesian switching with positive reset prior."""
import numpy as np

class Switch:
    def __init__(self,prior,hazard):
        self.prior=np.asarray(prior,dtype=np.float64).copy()
        assert self.prior.ndim==1 and self.prior.min()>0 and abs(self.prior.sum()-1)<1e-12
        assert 0<hazard<=1
        self.hazard=float(hazard);self.posterior=self.prior.copy()
    def reset(self):self.posterior=self.prior.copy()
    def weights(self):return (1-self.hazard)*self.posterior+self.hazard*self.prior
    def observe(self,likelihood):
        p=np.asarray(likelihood,np.float64);assert p.shape==self.prior.shape and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
        x=self.weights()*p;z=float(x.sum());assert z>0
        self.posterior=x/z
        return z
