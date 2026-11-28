"""Causal signed address transport on the actual v97 conversational field.

The previous query is filtered by the byte that was actually consumed, then
transported one source position forward. Fresh signed addressing and transport
share the copy and residual channels; no answer is inserted by the decoder.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from field97 import Chat97
from field96 import batch, EOT


class Chat99(Chat97):
    def __init__(self):
        super().__init__()
        self.transport99 = nn.Linear(256, 1)
        nn.init.zeros_(self.transport99.weight)
        nn.init.constant_(self.transport99.bias, -4.0)
        self.scale99 = 1.0

    def decode99(self, h, previous, consumed, anchor, memory, eligible):
        if self.scale99 == 0.0 or self.scale97 == 0.0:
            return super().decode(h, anchor, memory)
        k, v, ids, valid = memory
        mask = valid[:, None, :]
        def address(q):
            scores = self.query97(q) @ k.transpose(-1, -2) / math.sqrt(96.0)
            return scores.masked_fill(~mask, -torch.inf).softmax(-1)
        fresh = address(h)
        past = address(previous)
        # Bayes-filter previous source address by actual observed byte.
        evidence = past * (ids[:, None, :] == consumed[:, :, None])
        # Shift without wrapping; never move across the prompt boundary.
        transported = F.pad(evidence[..., :-1], (1, 0)) * mask
        mass = transported.sum(-1, keepdim=True)
        normalized = transported / mass.clamp_min(1e-20)
        # Mass-weighted gate makes empty evidence an exact fresh-address fallback.
        gate = self.scale99 * self.transport99(h).sigmoid() * mass * eligible[..., None]
        att = fresh * (1.0 - gate) + normalized * gate
        context = att @ v
        p, n = context.chunk(2, -1)
        signed = torch.cat((p.tanh() * torch.sigmoid(p-n),
                            n.tanh() * torch.sigmoid(n-p)), -1)
        logits = self.read(h, anchor[:, None]) + self.scale97 * self.residual97(signed)
        mix = .65 * self.scale97 * self.copy97(torch.cat((h, signed), -1)).sigmoid()
        copy = torch.zeros_like(logits).scatter_add(
            -1, ids[:, None, :].expand(-1, h.shape[1], -1), att)
        return ((1-mix)*logits.softmax(-1) + mix*copy).clamp_min(1e-30).log()

    def forward(self, x, plen):
        if self.scale99 == 0.0 or self.scale97 == 0.0:
            return super().forward(x, plen)
        h, _ = self.gru(self.embed(x))
        anchor = h[torch.arange(len(x), device=x.device), plen-1]
        prev = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), 1)
        eligible = torch.arange(x.shape[1], device=x.device)[None, :] >= plen[:, None]
        return self.decode99(h, prev, x, anchor, self.memory(h, x, plen), eligible)

    def prefill(self, prompt):
        if not prompt:
            raise ValueError("empty prompt")
        return super().prefill(prompt)

    def advance(self, state, anchor, b):
        if self.scale99 == 0.0 or self.scale97 == 0.0:
            return super().advance(state, anchor, b)
        a, mem = anchor
        x = torch.tensor([[b]], dtype=torch.long, device=state.device)
        h, next_state = self.gru(self.embed(x), state)
        q = self.decode99(h, state[-1][:, None], x, a, mem,
                          torch.ones_like(x, dtype=torch.bool))
        return next_state, q[0, 0].double().softmax(-1).detach().cpu().numpy()


def selftest():
    torch.manual_seed(9900)
    torch.set_num_threads(2)
    parent = Chat97()
    model = Chat99()
    missing = model.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing.missing_keys) == {"transport99.weight", "transport99.bias"}
    assert not missing.unexpected_keys
    p = b"User: red=abcd blue=wxyz. Return red.\nAssistant: "
    a = b"abcd" + bytes([EOT])
    rows = [(p, a, "", ""), (b"User: abc\nAssistant: ", b"abc"+bytes([EOT]), "", "")]
    x, y, plen = batch(rows)
    with torch.no_grad():
        model.scale99 = 0.
        assert torch.equal(model(x, plen), parent(x, plen))
        model.scale99 = 1.
        z = model(x, plen).double().softmax(-1)
        assert torch.isfinite(z).all() and torch.allclose(z.sum(-1), torch.ones_like(z[...,0]))
        for i, (prompt, answer, *_) in enumerate(rows):
            state, anchor, q = model.prefill(prompt)
            for j, b in enumerate(answer):
                np.testing.assert_allclose(q, z[i, len(prompt)-1+j].numpy(), atol=2e-7, rtol=1e-5)
                state, q = model.advance(state, anchor, b)
        changed = x.clone()
        changed[0, len(p)+1:] = 201
        torch.testing.assert_close(model(changed, plen)[0,:len(p)+1], model(x,plen)[0,:len(p)+1])
        # Last source position cannot wrap into the first.
        h = torch.randn(1,1,256); k = torch.randn(1,1,96); v = torch.randn(1,1,256)
        ids = torch.tensor([[97]]); valid = torch.ones_like(ids,dtype=torch.bool)
        z1 = model.decode99(h,h,ids,h[:,0],(k,v,ids,valid),valid)
        z2 = parent.decode(h,h[:,0],(k,v,ids,valid))
        torch.testing.assert_close(z1,z2)
    z = model(x, plen)
    F.cross_entropy(z.transpose(1,2), y, ignore_index=-100).backward()
    assert model.transport99.weight.grad is not None
    assert torch.isfinite(model.transport99.weight.grad).all()
    assert model.transport99.weight.grad.abs().sum() > 0
    print("v99 disabled parity, normalized law, padded recurrent parity, future isolation, no wrap, gradient passed")


if __name__ == "__main__":
    selftest()
