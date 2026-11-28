"""On-policy repeated-continuation unlikelihood; training only.

A generated history is detached. At each response step, discourage bytes
which would recreate an already completed 12-byte substring in the
response. Reference CE protects correct repetitions and termination.
No target-derived features or changes to the autonomous decoder.
"""
import numpy as np
import torch
from torch.nn import functional as F
from field96 import EOT,batch

def forbidden(prefix,order=12):
    if len(prefix)<order:return ()
    tail=prefix[-(order-1):]
    return tuple(sorted({prefix[i+order-1] for i in range(len(prefix)-order+1)
                         if prefix[i:i+order-1]==tail and prefix[i+order-1]!=EOT}))

@torch.no_grad()
def rollout(model,prompt,rng,limit=64,temperature=0.):
    state,anchor,q=model.prefill(prompt)
    out=bytearray()
    for _ in range(limit):
        if temperature==0.:b=int(q.argmax())
        else:
            probs=np.power(np.maximum(q,1e-300),1./temperature)
            b=int(rng.choice(256,p=probs/probs.sum()))
        if b==EOT:break
        out.append(b)
        state,q=model.advance(state,anchor,b)
    return bytes(out)

def repeat_loss(model,examples,order=12):
    """Compute differentiable probability mass of repeat candidates on
    detached on-policy prefixes. Skip exact gold prefixes when the next
    gold byte is a repeat, and never penalize EOT."""
    rows=[]; masks=[]
    for p,gold,generated,*_ in examples:
        if not generated:continue
        row=(p,generated+bytes([EOT]),'', 'rollout')
        rows.append(row)
        m=[]
        for j in range(len(generated)+1):
            prefix=generated[:j]
            candidates=set(forbidden(prefix,order))
            if j<len(gold) and prefix==gold[:j]:
                candidates.discard(gold[j])
            m.append(tuple(sorted(candidates)))
        masks.append(m)
    if not rows:return next(model.parameters()).sum()*0.,0
    x,y,plen=batch(rows)
    logits=model(x,plen)
    logp=F.log_softmax(logits,dim=-1)
    losses=[]
    for i,ms in enumerate(masks):
        start=int(plen[i])-1
        for j,candidates in enumerate(ms):
            if not candidates:continue
            q=logp[i,start+j,list(candidates)].exp().sum()
            losses.append(-torch.log1p(-q.clamp(max=1.-1e-6)))
    if not losses:return logits.sum()*0.,0
    return torch.stack(losses).mean(),len(losses)

def selftest():
    assert forbidden(b'abcdefghijk')==()
    assert forbidden(b'abcdefghijkl' + b'abcdefghijk')==(ord('l'),)
    assert forbidden(b'abcdefghijkl' + b'abcdefghijkm' + b'abcdefghijk')==(ord('l'),ord('m'))
    assert EOT not in forbidden(b'abcdefghijk'+bytes([EOT])+b'abcdefghijk')
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.zeros(256))
        def forward(self,x,p):return self.bias.expand(*x.shape,256)
    model=Stub();p=b'User: Go.\nAssistant: '
    generated=b'abcdefghijkl'+b'abcdefghijk'
    loss,n=repeat_loss(model,[(p,b'other'+bytes([EOT]),generated)])
    assert n==1 and float(loss.detach())>0
    loss.backward();assert model.bias.grad[ord('l')]>0 and model.bias.grad[EOT]<0
    loss,n=repeat_loss(model,[(p,generated+b'l'+bytes([EOT]),generated)])
    assert n==0 and float(loss.detach())==0.
    print('on-policy unlikelihood: causal 12-byte repeat candidates, protected gold, EOT exclusion, finite differentiable penalty passed',flush=True)
if __name__=='__main__':selftest()
