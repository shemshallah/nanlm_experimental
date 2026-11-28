"""Semantic equivalence supervision for v100, independent of decoder heuristics.

Each quadruple is (form A/order 1, form A/order 2, form B/order 1,
form B/order 2). Both forms expose identical concepts, so the only
supervision difference is the requested order. Entire wording families
and lexical values are withheld for the generalization probe.
"""
import hashlib
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from field96 import EOT,batch

TRAIN_FORMS=(
    "{facts}. Give {a} then {b}, with a space.",
    "Data: {facts}. Return the values of {a} and {b} in that order.",
    "First report {a}, next report {b}. Here are the entries: {facts}.",
    "Using {facts}, output {a} followed by {b}. Only the values.",
    "{facts}. Write two values: {a} first, {b} second.",
    "The entries are {facts}. List {a} and then {b}, separated by a space.",
)
HOLDOUT_FORMS=(
    "From these assignments ({facts}), which strings correspond to {a} and subsequently {b}?",
    "I need the contents for {a}, before the contents for {b}. Reference table: {facts}.",
    "Given {facts}, concatenate the entries keyed by {a} and {b}, in sequence, using one space.",
)
TRAIN_FACTS=("{k}={v}","{k}: {v}","{k} contains {v}")
HOLDOUT_FACTS=("the value under {k} is {v}","{v} belongs to {k}")
ALPHABET="abcdefghjkmnpqrstuvwxyz"

def make_groups(n,seed,heldout=False,exclude=()):
    rng=np.random.default_rng(seed)
    used=set(exclude)
    forms=HOLDOUT_FORMS if heldout else TRAIN_FORMS
    facts=HOLDOUT_FACTS if heldout else TRAIN_FACTS
    result=[]
    def value():
        while True:
            s=''.join(rng.choice(list(ALPHABET),6))
            h=hashlib.sha256(s.encode()).hexdigest()
            if (int(h[:8],16)%10==0)==heldout and s not in used:
                used.add(s);return s
    for i in range(n):
        labels=list('ABC' if i%2==0 else 'XYZ')
        vals={k:value() for k in labels}
        a,b=rng.permutation(labels)[:2]
        f0=i%len(forms)
        f1=(f0+int(rng.integers(1,len(forms))))%len(forms)
        rows=[]
        for f in (f0,f1):
            # Render identical facts in independent order and surface syntax.
            source='; '.join(facts[int(rng.integers(len(facts)))].format(k=k,v=vals[k])
                             for k in rng.permutation(labels))
            for first,second in ((a,b),(b,a)):
                p=('User: '+forms[f].format(facts=source,a=first,b=second)+'\nAssistant: ').encode()
                y=(vals[first]+' '+vals[second]).encode()+bytes([EOT])
                key=hashlib.sha256(p).hexdigest()
                rows.append((p,y,key,f"{'holdout' if heldout else 'train'}_form_{f}"))
        result.append(tuple(rows))
    return result,used-set(exclude)

def paired_views(groups):
    return [(g[i],g[i+1]) for g in groups for i in (0,2)]

def equivalence_loss(model,groups):
    """Jensen-Shannon divergence of aligned next-byte laws, not hidden states.
    Both sides condition on their own causal prompt and the same target prefix.
    No prompt token or pad position contributes."""
    rows=[r for g in groups for r in g]
    x,y,p=batch(rows)
    logits=model(x,p)
    logq=F.log_softmax(logits,dim=-1)
    divergences=[]
    for i in range(0,len(rows),4):
        for offset in (0,1):
            a,b=i+offset,i+offset+2
            assert rows[a][1]==rows[b][1]
            left=logq[a,y[a]!=-100];right=logq[b,y[b]!=-100]
            assert left.shape==right.shape
            middle=torch.logaddexp(left,right)-np.log(2.)
            js=.5*((left.exp()*(left-middle)).sum(-1)+(right.exp()*(right-middle)).sum(-1))
            divergences.append(js.mean())
    return torch.stack(divergences).mean()

def selftest():
    train,tv=make_groups(32,10021)
    dev,dv=make_groups(16,10022,True)
    assert not tv&dv
    assert set(TRAIN_FORMS).isdisjoint(HOLDOUT_FORMS)
    for g in train+dev:
        assert len({r[0] for r in g})==4
        assert g[0][1]==g[2][1] and g[1][1]==g[3][1]
        assert g[0][1][:-1].split()==g[1][1][:-1].split()[::-1]
        assert all(r[1].endswith(bytes([EOT])) for r in g)
        assert all(v in r[0] for r in g for v in r[1][:-1].split())
    # JS must be zero for identical predictive distributions and differentiable.
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.randn(256))
        def forward(self,x,p):
            return self.bias.expand(*x.shape,256)
    m=Stub();v=equivalence_loss(m,train[:1]);assert abs(float(v.detach()))<1e-6
    v.backward();assert torch.isfinite(m.bias.grad).all()
    print('v100 semantic equivalence: disjoint values and wording families, paired order reversal, aligned normalized JS and gradients passed',flush=True)

if __name__=='__main__':
    selftest()
