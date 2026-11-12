"""R9 genuinely ordered causal self-attention over observed last 64 words.

This is an additional residual computation on the SAME shared NOW state,
not another vocabulary head, model, or noncausal future branch.
"""
import torch
import torch.nn.functional as F
from torch import nn

class R9Sequence(nn.Module):
    def __init__(self,dim=96,context=64,heads=4):
        super().__init__()
        self.positions=nn.Parameter(torch.randn(context,dim)*.01)
        self.ln=nn.LayerNorm(dim)
        self.block=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=2*dim,
                                               dropout=0.0,activation='gelu',batch_first=True,norm_first=True)
        self.last_query=nn.MultiheadAttention(dim,heads,dropout=0.0,batch_first=True)
        self.proj=nn.Linear(dim,dim,bias=False)
        self.scale=nn.Parameter(torch.zeros(()))
        # Past words may only look at words no later than themselves.
        self.register_buffer('mask',torch.triu(torch.ones(context,context,dtype=torch.bool),diagonal=1),persistent=False)
    def forward(self,h,context,local_pieces,local_lengths,emb,comp):
        seq=F.embedding(context[:,-64:],emb)+comp(local_pieces,local_lengths)+self.positions[None,:,:]
        seq=self.block(self.ln(seq),src_mask=self.mask)
        retrieved,_=self.last_query(h.unsqueeze(1),seq,seq,need_weights=False)
        return torch.tanh(torch.atanh(h.clamp(-.999999,.999999))+self.scale*self.proj(retrieved[:,0,:]))
