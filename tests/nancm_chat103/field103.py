"""Prompt-contextual source addressing for v103.

A bidirectional encoder sees only the complete user prompt, never response
bytes. A query from the accepted response history selects a prompt position
or a null address. Unique source spans can supervise the address at training
time; inference uses only the predicted address distribution.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from field96 import batch,EOT
from field100 import Chat100

class Chat103(Chat100):
    def __init__(self):
        super().__init__()
        self.source103=nn.GRU(96,64,batch_first=True,bidirectional=True)
        self.key103=nn.Linear(128,96,bias=False)
        self.query103=nn.Linear(512,96,bias=False)
        self.null103=nn.Linear(512,1)
        self.gain103=nn.Parameter(torch.zeros(()))
        self.scale103=1.

    def memory(self,h,x,plen):
        old=super().memory(h,x,plen)
        if self.scale103==0.:return old
        # Packing prevents short prompts from encoding response/padding bytes.
        n=int(plen.max())
        src=self.embed(x[:,:n])
        packed=pack_padded_sequence(src,plen.detach().cpu(),batch_first=True,enforce_sorted=False)
        enc,_=self.source103(packed)
        enc,_=pad_packed_sequence(enc,batch_first=True,total_length=n)
        return (*old,self.key103(enc))

    def address103(self,h,anchor,memory):
        k,v,ids,valid,keys=memory
        q=torch.cat((h,anchor[:,None].expand(-1,h.shape[1],-1)),-1)
        scores=self.query103(q)@keys.transpose(-1,-2)/math.sqrt(96.)
        scores=scores.masked_fill(~valid[:,None],-torch.inf)
        return torch.cat((scores,self.null103(q)),-1)

    def decode99(self,h,previous,consumed,anchor,memory,eligible):
        if self.scale103==0.:
            return super().decode99(h,previous,consumed,anchor,memory,eligible)
        base=super().decode99(h,previous,consumed,anchor,memory[:4],eligible)
        addr=self.address103(h,anchor,memory).softmax(-1)
        # Null probability goes back to the inherited law. No copying outside
        # the prompt; normalized even with a completely null address.
        copy=torch.zeros_like(base).scatter_add(-1,memory[2][:,None].expand(-1,h.shape[1],-1),addr[...,:-1])
        gain=self.scale103*self.gain103.clamp(0.,.8)
        b=base.exp()
        mixed=b*(1.-gain*(1.-addr[...,-1:]))+gain*copy
        # Subtract the identical baseline transform so gain=0 is bitwise
        # inherited, yet gradients can open the zero-initialized gate.
        return base+(mixed.clamp_min(1e-30).log()-b.clamp_min(1e-30).log())

    def alignment103(self,rows):
        """Train-only exact source targets for uniquely occurring output words.
        Space/EOT targets use null; ambiguous words are ignored, never guessed."""
        x,y,plen=batch(rows)
        h,_=self.gru(self.embed(x))
        a=h[torch.arange(len(x),device=x.device),plen-1]
        mem=self.memory(h,x,plen)
        logits=self.address103(h,a,mem)
        target=torch.full_like(y,-100)
        null=logits.shape[-1]-1
        for i,(p,answer,*_) in enumerate(rows):
            pos=len(p)-1
            assert answer.endswith(bytes([EOT]))
            offset=0
            for word in answer[:-1].split(b' '):
                where=p.find(word)
                unique=where>=0 and p.find(word,where+1)<0
                if unique:target[i,pos+offset:pos+offset+len(word)]=torch.arange(where,where+len(word),device=x.device)
                offset+=len(word)
                if offset<len(answer)-1:
                    target[i,pos+offset]=null;offset+=1
            target[i,pos+len(answer)-1]=null
        return F.cross_entropy(logits.transpose(1,2),target,ignore_index=-100)

def selftest():
    torch.set_num_threads(2);torch.manual_seed(103)
    parent=Chat100().eval().requires_grad_(False)
    model=Chat103().eval().requires_grad_(False)
    missing=model.load_state_dict(parent.state_dict(),strict=False)
    assert not missing.unexpected_keys and all('103' in n for n in missing.missing_keys)
    rows=[(b'User: zzxy belongs to A. qrst belongs to B. Return B then A.\nAssistant: ',b'qrst zzxy'+bytes([EOT]),'',''),
          (b'User: A=mnpo. Return A.\nAssistant: ',b'mnpo'+bytes([EOT]),'','')]
    x,y,p=batch(rows)
    with torch.no_grad():
        model.scale103=0.;assert torch.equal(model(x,p),parent(x,p))
        model.scale103=1.;assert torch.equal(model(x,p),parent(x,p))
        model.gain103.fill_(.4)
        z=model(x,p).double().softmax(-1)
        assert torch.isfinite(z).all();torch.testing.assert_close(z.sum(-1),torch.ones_like(z[...,0]))
        for i,(prompt,answer,*_) in enumerate(rows):
            state,anchor,q=model.prefill(prompt)
            for j,b in enumerate(answer):
                np.testing.assert_allclose(q,z[i,len(prompt)-1+j].numpy(),atol=3e-7,rtol=2e-5)
                old=tuple(t.clone() for t in state)
                a,qa=model.advance(state,anchor,b);bstate,qb=model.advance(state,anchor,b)
                for before,after in zip(old,state):assert torch.equal(before,after)
                np.testing.assert_array_equal(qa,qb)
                state,q=a,qa
        changed=x.clone();cut=int(p[1])+1;changed[1,cut:]=222
        torch.testing.assert_close(model(changed,p)[1,:cut],model(x,p)[1,:cut])
    model.requires_grad_(True)
    loss=F.cross_entropy(model(x,p).transpose(1,2),y,ignore_index=-100)+model.alignment103(rows)
    loss.backward()
    for n,w in model.named_parameters():
        if '103' in n:assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum()>0,n
    print('v103 exact disabled and closed-gate inheritance, packed prompt isolation, normalized null-address law, autoregressive parity, transactional rollback and alignment gradients passed',flush=True)

if __name__=='__main__':selftest()
