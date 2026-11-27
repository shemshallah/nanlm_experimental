"""Signed full-vocabulary transport, exact folded scores and temporal credit."""
import torch
from torch import nn

class Transport(nn.Module):
    def __init__(self,code,rank=24):
        super().__init__();c=torch.as_tensor(code,dtype=torch.float32);self.register_buffer('code',c);d=c.shape[-1]
        self.plus=nn.Parameter(torch.randn(9,d,rank)*.04)
        self.minus=nn.Parameter(torch.randn(9,d,rank)*.04)
        self.route=nn.Parameter(torch.zeros(9));self.read=nn.Parameter(torch.zeros(rank,d))
    def lift(self,x,scale=1.):
        z=(torch.einsum('...ri,rij->...rj',x.relu(),self.plus)-torch.einsum('...ri,rij->...rj',(-x).relu(),self.minus)).tanh()
        h=(z*self.route.softmax(0)[...,None]).sum(-2)@self.read
        return 6*scale*(h@self.code.T).tanh().double()
    def forward(self,x,p,scale=1.):
        if scale==0:return p
        a=p.double()*self.lift(x,scale).exp().unsqueeze(-2)
        return a/a.sum(-1,keepdim=True)

def predict(core,state,prev,features):
    if state is None:return core.prior
    v=prev.clamp_min(1e-300).log();f=torch.cat((features,((v[1:]-v[0])/8).clamp(-1,1)))
    return state@core.matrix(f)

def credit(core,p,event_vector,surface,raw,features,candidates,word_raw,word_coeff,state=None,prev=None,rank_weight=0.):
    ev=p.new_zeros(());rank=p.new_zeros(());word=p.new_zeros(());n=0;hits=0
    for t in range(len(p)):
        w=predict(core,state,prev,features[t]);lik=torch.cat((raw[t:t+1],(p[t]*event_vector[t]).sum(-1)*surface[t]))
        u=w*lik;z=u.sum();ev=ev-z.log();state=u/z;prev=lik
        ids=candidates[t]
        if len(ids):
            scores=w[0]*word_raw[t]+(w[1:,None]*p[t,:,ids]).sum(0)*word_coeff[t]
            logs=scores.clamp_min(1e-300).log();word=word-logs[0];n+=1
            if len(ids)>1:rank=rank+torch.nn.functional.softplus(logs[1:]-logs[0]+.3).mean()
            hits+=int(bool((scores[0]>scores[1:]).all()))
    return ev/len(p)+rank_weight*(rank+.25*word)/max(n,1),state,prev,ev,rank,word,n,hits
