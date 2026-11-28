"""v100: persistent signed intention and response-progress state.

The goal is initialized from the inherited prompt anchor. A second signed
recurrent state evolves only on accepted response bytes. Its read is an
additive energy correction to the complete inherited v99 byte law.
Disabled mode preserves the exact v99 public interface and probabilities.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field96 import batch, EOT
from field99 import Chat99


class Chat100(Chat99):
    def __init__(self):
        super().__init__()
        self.goal100 = nn.Linear(256, 256)
        self.progress100 = nn.GRU(256, 256, batch_first=True)
        self.energy100 = nn.Linear(256, 256, bias=False)
        nn.init.zeros_(self.energy100.weight)
        self.scale100 = 1.0

    def initial100(self, anchor):
        return self.goal100(anchor).tanh().unsqueeze(0)

    def correction100(self, progress, anchor):
        goal = self.initial100(anchor)[0][:, None]
        remaining = goal - progress
        p, n = remaining.chunk(2, -1)
        signed = torch.cat((p.tanh() * torch.sigmoid(p-n),
                            n.tanh() * torch.sigmoid(n-p)), -1)
        # Bounded energy adjustment; no changes to the inherited byte alphabet.
        return self.scale100 * 2.0 * self.energy100(signed).tanh()

    def forward(self, x, plen):
        if self.scale100 == 0.0:
            return super().forward(x, plen)
        h, _ = self.gru(self.embed(x))
        anchor = h[torch.arange(len(x), device=x.device), plen-1]
        previous = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), 1)
        pos = torch.arange(x.shape[1], device=x.device)[None, :]
        eligible = pos >= plen[:, None]
        base = self.decode99(h, previous, x, anchor,
                             self.memory(h, x, plen), eligible)
        initial = .25 * self.initial100(anchor)
        # Pack response time relative to each prompt boundary, not absolute
        # sequence time. The initial goal is never advanced by prompt padding.
        length = x.shape[1] - int(plen.min())
        if length:
            indices = (plen[:, None] + torch.arange(length, device=x.device)[None]).clamp_max(x.shape[1]-1)
            response = h.gather(1, indices[..., None].expand(-1,-1,h.shape[-1]))
            progress, _ = self.progress100(response-anchor[:, None], initial)
            trajectory = torch.cat((initial.transpose(0,1),progress),1)
        else:
            trajectory = initial.transpose(0,1)
        relative = (pos-plen[:,None]+1).clamp_min(0)
        progress = trajectory.gather(1, relative[...,None].expand(-1,-1,256))
        delta = self.correction100(progress,anchor)
        return base + delta * (pos >= plen[:,None]-1)[...,None]

    def prefill(self, prompt):
        if self.scale100 == 0.0:
            return super().prefill(prompt)
        if not prompt:
            raise ValueError("empty prompt")
        x = torch.tensor(list(prompt),dtype=torch.long,device=self.embed.weight.device)[None]
        h, state = self.gru(self.embed(x))
        anchor = h[:,-1]
        memory = self.memory(h,x,torch.tensor([len(prompt)],device=x.device))
        initial = .25 * self.initial100(anchor)
        base = self.decode99(h[:,-1:],h[:,-1:],x[:,-1:],anchor,memory,
                             torch.zeros_like(x[:,-1:],dtype=torch.bool))
        logits = base + self.correction100(initial.transpose(0,1),anchor)
        return (state,initial),(anchor,memory),logits[0,0].double().softmax(-1).detach().cpu().numpy()

    def advance(self, state, anchor, b):
        if self.scale100 == 0.0:
            return super().advance(state,anchor,b)
        old, progress = state
        a, memory = anchor
        x = torch.tensor([[b]],dtype=torch.long,device=old.device)
        h, new = self.gru(self.embed(x),old)
        out, next_progress = self.progress100(h-a[:,None],progress)
        base = self.decode99(h,old[-1][:,None],x,a,memory,
                             torch.ones_like(x,dtype=torch.bool))
        logits = base + self.correction100(out,a)
        return (new,next_progress),anchor_q(logits)


def anchor_q(logits):
    return logits[0,0].double().softmax(-1).detach().cpu().numpy()


def selftest():
    torch.manual_seed(10001)
    torch.set_num_threads(2)
    parent = Chat99().eval().requires_grad_(False)
    model = Chat100().eval().requires_grad_(False)
    missing = model.load_state_dict(parent.state_dict(),strict=False)
    assert not missing.unexpected_keys and all('100' in k for k in missing.missing_keys)
    rows = [(b'User: A=red B=blue. Return A then B.\nAssistant: ',b'red blue'+bytes([EOT]),'', ''),
            (b'User: Go.\nAssistant: ',b'yes'+bytes([EOT]),'', '')]
    x,y,plen = batch(rows)
    with torch.no_grad():
        model.scale100 = 0.
        assert torch.equal(model(x,plen),parent(x,plen))
        for p,a,*_ in rows:
            st,an,q = model.prefill(p)
            ps,pa,pq = parent.prefill(p)
            np.testing.assert_array_equal(q,pq)
            for b in a:
                st,q = model.advance(st,an,b)
                ps,pq = parent.advance(ps,pa,b)
                np.testing.assert_array_equal(q,pq)
        model.scale100 = 1.
        assert torch.equal(model(x,plen),parent(x,plen))
        model.energy100.weight.normal_(0,.02)
        z = model(x,plen).double().softmax(-1)
        assert torch.isfinite(z).all()
        torch.testing.assert_close(z.sum(-1),torch.ones_like(z[...,0]))
        for i,(p,a,*_) in enumerate(rows):
            st,an,q = model.prefill(p)
            for j,b in enumerate(a):
                np.testing.assert_allclose(q,z[i,len(p)-1+j].numpy(),rtol=1e-5,atol=2e-7)
                before = tuple(t.clone() for t in st)
                next1,q1 = model.advance(st,an,b)
                next2,q2 = model.advance(st,an,b)
                for prev,now in zip(before,st): assert torch.equal(prev,now)
                for u,v in zip(next1,next2): assert torch.equal(u,v)
                np.testing.assert_array_equal(q1,q2)
                st,q = next1,q1
        changed=x.clone();cut=int(plen[0])+2;changed[0,cut:]=201
        torch.testing.assert_close(model(changed,plen)[0,:cut],model(x,plen)[0,:cut])
    model.requires_grad_(True)
    F.cross_entropy(model(x,plen).transpose(1,2),y,ignore_index=-100).backward()
    for name,p in model.named_parameters():
        if '100' in name:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    print('v100 exact disabled and zero-energy inheritance, response-only recurrence, normalized law, autoregressive parity, causal isolation, rollback and gradients passed',flush=True)


if __name__ == '__main__':
    selftest()
