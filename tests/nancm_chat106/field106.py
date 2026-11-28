"""Selective source admission: contextual, calibrated, differentiable.

Retain the trained v103 address head and v104 recovery weights. The new
gate sees only pre-observation recurrent state and two predictive laws.
An auxiliary train-only target identifies where admitting source evidence
would improve the likelihood of the next byte. No target access at inference.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from field96 import batch,EOT
from field100 import Chat100
from field103 import Chat103

class Chat106(Chat103):
    def __init__(self):
        super().__init__()
        self.admit106=nn.Linear(518,1)
        nn.init.zeros_(self.admit106.weight)
        nn.init.constant_(self.admit106.bias,-3.)
        self.scale106=1.

    def components106(self,h,previous,consumed,anchor,memory,eligible):
        base=Chat100.decode99(self,h,previous,consumed,anchor,memory[:4],eligible)
        addr=self.address103(h,anchor,memory).softmax(-1)
        b=base.exp()
        copy=torch.zeros_like(b).scatter_add(-1,memory[2][:,None].expand(-1,h.shape[1],-1),addr[...,:-1])
        # Null address returns its mass to the inherited predictor.
        source=copy+addr[...,-1:]*b
        def entropy(q):
            return -(q*q.clamp_min(1e-30).log()).sum(-1,keepdim=True)/math.log(256.)
        feats=torch.cat((h,anchor[:,None].expand(-1,h.shape[1],-1),
                          entropy(b),entropy(source),b.amax(-1,keepdim=True),
                          source.amax(-1,keepdim=True),addr[...,-1:],
                          .5*(b-source).abs().sum(-1,keepdim=True)),-1)
        # The legacy scalar is retained and participates in the new gate's
        # log odds, preserving its optimizer state; no hard-clamp dead zone.
        gate_logit=self.admit106(feats)+self.gain103
        gate=.65*gate_logit.sigmoid()
        return base,b,source,gate,gate_logit

    def decode99(self,h,previous,consumed,anchor,memory,eligible):
        if self.scale106==0. or self.scale103==0.:
            return super().decode99(h,previous,consumed,anchor,memory,eligible)
        base,b,src,g,_=self.components106(h,previous,consumed,anchor,memory,eligible)
        mixed=(1.-g)*b+g*src
        return base+(mixed.clamp_min(1e-30).log()-b.clamp_min(1e-30).log())

    def gate106(self,rows):
        x,y,plen=batch(rows)
        h,_=self.gru(self.embed(x))
        a=h[torch.arange(len(x),device=x.device),plen-1]
        prev=torch.cat((torch.zeros_like(h[:,:1]),h[:,:-1]),1)
        eligible=torch.arange(x.shape[1],device=x.device)[None,:]>=plen[:,None]
        _,b,src,g,logit=self.components106(h,prev,x,a,self.memory(h,x,plen),eligible)
        valid=y!=-100
        target=y.clamp_min(0)[...,None]
        bp=b.gather(-1,target).squeeze(-1)
        sp=src.gather(-1,target).squeeze(-1)
        # Soft target avoids unstable hard decisions around ties. Neutral
        # evidence biases toward the inherited pathway rather than copying.
        oracle=((sp.clamp_min(1e-30).log()-bp.clamp_min(1e-30).log()-.5)*2.).sigmoid().detach()
        loss=F.binary_cross_entropy_with_logits(logit.squeeze(-1)[valid],oracle[valid])
        return loss,g.squeeze(-1)[valid],oracle[valid]

def selftest():
    torch.set_num_threads(2);torch.manual_seed(106)
    parent=Chat103().eval().requires_grad_(False)
    model=Chat106().eval().requires_grad_(False)
    missing=model.load_state_dict(parent.state_dict(),strict=False)
    assert set(missing.missing_keys)=={'admit106.weight','admit106.bias'} and not missing.unexpected_keys
    rows=[(b'User: A=red B=blue. Return B.\nAssistant: ',b'blue'+bytes([EOT]),'',''),
          (b'User: Go.\nAssistant: ',b'yes'+bytes([EOT]),'','')]
    x,y,p=batch(rows)
    with torch.no_grad():
        model.scale106=0.;assert torch.equal(model(x,p),parent(x,p))
        for prompt,answer,*_ in rows:
            s,a,q=model.prefill(prompt);t,b,r=parent.prefill(prompt)
            np.testing.assert_array_equal(q,r)
            for byte in answer:
                s,q=model.advance(s,a,byte);t,r=parent.advance(t,b,byte)
                np.testing.assert_array_equal(q,r)
        model.scale106=1.
        z=model(x,p).double().softmax(-1)
        torch.testing.assert_close(z.sum(-1),torch.ones_like(z[...,0]))
        _,g,_=model.gate106(rows)
        assert (g>0).all() and (g<.65).all()
        for i,(prompt,answer,*_) in enumerate(rows):
            s,a,q=model.prefill(prompt)
            for j,byte in enumerate(answer):
                np.testing.assert_allclose(q,z[i,len(prompt)-1+j].numpy(),atol=3e-7,rtol=2e-5)
                s,q=model.advance(s,a,byte)
        altered=x.clone();cut=int(p[1])+1;altered[1,cut:]=201
        torch.testing.assert_close(model(altered,p)[1,:cut],model(x,p)[1,:cut])
    model.requires_grad_(True)
    loss,g,oracle=model.gate106(rows)
    (loss+F.cross_entropy(model(x,p).transpose(1,2),y,ignore_index=-100)).backward()
    for n,param in model.named_parameters():
        if '106' in n or n=='gain103':
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum()>0,n
    print('v106 exact disabled inheritance, positive differentiable admission, causal isolation, normalized autoregressive law and gate gradients passed',flush=True)

if __name__=='__main__':selftest()
