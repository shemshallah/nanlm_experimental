"""Normalized finite-candidate likelihood training, with exact complement."""
import numpy as np
DIM=14

def objective(theta,features,prior,hit,base,scale=1.,reg=0.):
    z=features@theta;lift=4*np.tanh(z/4)*scale
    e=np.exp(lift);mass=prior*e;den=1+np.sum(prior*np.expm1(lift),axis=1)
    target_lift=np.sum(hit*lift,axis=1)
    loss=np.mean(np.log(den)-target_lift-np.log(base))+reg*np.dot(theta,theta)/2
    coeff=(mass/den[:,None]-hit)*scale*(1-np.tanh(z/4)**2)
    grad=np.einsum('bk,bkd->d',coeff,features)/len(base)+reg*theta
    return float(loss),grad

def fit(ds,steps=256,seed=39):
    rng=np.random.default_rng(seed);theta=np.zeros(DIM);m=theta.copy();v=theta.copy();trace=[]
    for i in range(1,steps+1):
        idx=rng.integers(len(ds[-1]),size=32);loss,g=objective(theta,*(a[idx] for a in ds),reg=.005)
        m=.9*m+.1*g;v=.999*v+.001*g*g
        theta-=.02*(m/(1-.9**i))/(np.sqrt(v/(1-.999**i))+1e-8)
        if i%32==0 or i==1:trace.append([i,loss])
    return theta,m,v,trace

def preflight():
    rng=np.random.default_rng(3901);f=rng.normal(size=(12,6,DIM));p=rng.random((12,6));p*=.7/p.sum(1,keepdims=True)
    h=np.zeros_like(p);h[:8,2]=1;b=np.where(h.any(1),p[:,2],.03);t=rng.normal(size=DIM)*.2
    loss,g=objective(t,f,p,h,b,reg=.005);err=[]
    for i in range(DIM):
        a=t.copy();a[i]+=1e-5;v=t.copy();v[i]-=1e-5
        err.append(abs((objective(a,f,p,h,b,reg=.005)[0]-objective(v,f,p,h,b,reg=.005)[0])/2e-5-g[i]))
    assert max(err)<1e-9
    assert abs(objective(t,f,p,h,b,scale=0)[0]+np.log(b).mean())<1e-12
    t,m,v,trace=fit((f,p,h,b));assert np.isfinite(t).all() and objective(t,f,p,h,b)[0]<objective(np.zeros(DIM),f,p,h,b)[0]
    print(dict(max_gradient_error=max(err),zero_scale_parity=True,initial_nll=objective(np.zeros(DIM),f,p,h,b)[0],trained_nll=objective(t,f,p,h,b)[0]))
if __name__=='__main__':preflight()
