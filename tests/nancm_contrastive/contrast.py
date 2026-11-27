"""Contrast observed topic against the reference background, without target access."""
import numpy as np

def donors(h,parent,prior,cls):
    h=np.asarray(h,np.int64);v=len(prior);k=int(cls.max())+1
    if not len(h) or not np.any(h>0):return [parent]*3,np.zeros((3,5),np.float32),np.zeros(3,bool)
    age=np.arange(len(h),0,-1);weight=1/np.sqrt(age);weight[h==0]=0
    counts=np.bincount(h,weights=weight,minlength=v);total=counts.sum();q=counts/total
    cp=np.bincount(cls,weights=prior,minlength=k);cq=np.bincount(cls,weights=counts,minlength=k)
    ratio=(cq+16*cp)/(total+16)/cp
    cr=np.log(ratio).clip(-2,2)[cls]
    wr=np.zeros(v);ok=counts>0;wr[ok]=(counts[ok]/(counts[ok]+1))*np.log1p(q[ok]/prior[ok]).clip(0,4)
    def tilt(x):
        p=parent*np.exp(x);return p/p.sum()
    f=np.array([np.log1p(len(h))/8,np.log1p(ok.sum())/8,float(q.max()),float((q*wr).sum()/4),float((cq*abs(np.log(ratio))).sum()/total/4)],np.float32)
    return [q,tilt(wr),tilt(cr)],np.stack([f]*3),np.ones(3,bool)
