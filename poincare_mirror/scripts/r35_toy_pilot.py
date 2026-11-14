#!/usr/bin/env python3
"""Small, reproducible R34->R35 CPU experiment; NOT a book-corpus benchmark.

Runs a paired-optimizer comparison after exactly logit-preserving migration.
Real FA2 promotion still requires the step-2849 R34 checkpoint and book splits.
"""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, expand_architecture


def documents(seed, count, length):
    """Different documents from the same noisy 8-motif finite grammar."""
    rng = np.random.default_rng(seed)
    motifs = np.array([[((7*j + 5*i + (j//3)*i)%48) for j in range(24)]
                       for i in range(8)], dtype=np.int64)
    docs = []
    for _ in range(count):
        output = []
        while len(output) < length:
            motif = motifs[rng.integers(0,8)].copy()
            if rng.random() < .17:
                motif[rng.integers(0,len(motif))] = rng.integers(0,48)
            output.extend(motif.tolist())
        docs.append(torch.tensor(output[:length]))
    return docs


@torch.no_grad()
def measure(model, eval_windows):
    model.eval()
    loss, correct, top5, tokens = 0.,0,0,0
    for seq in eval_windows:
        logits = model(seq[:,:-3])[0]
        targets = seq[:,1:-2]
        n = targets.numel()
        tokens += n
        loss += F.cross_entropy(logits.flatten(0,1),targets.flatten()).item()*n
        correct += int((logits.argmax(-1)==targets).sum())
        top5 += int((logits.topk(5,dim=-1).indices==targets.unsqueeze(-1)).any(-1).sum())
    ce = loss/tokens
    return {'ce_nats_per_byte':round(ce,6),'bits_per_byte':round(ce/math.log(2),6),
            'top1_accuracy':round(correct/tokens,6),
            'top5_accuracy':round(top5/tokens,6),'predictions':tokens}


def main(updates=24):
    torch.set_num_threads(2)
    torch.manual_seed(350)
    cfg=Config(vocab_size=48,dim=32,heads=4,layers=2,pairs=4,topk_pairs=2,
               expert_hidden=24,context=20,moe_every=2,symbiotic_rank=8,
               micro_meta=True,simplex_transfer=True)
    previous=MirrorLM(cfg)
    prior_opt=torch.optim.AdamW(previous.parameters(),lr=.001)
    train=documents(120,12,320)
    dev=documents(121,4,240)  # disjoint generated documents
    def window(docs,doc,start):
        return docs[doc][start:start+23].unsqueeze(0)
    # Provide inherited nonempty Adam history before enabling new memory.
    for step in range(8):
        previous.train();prior_opt.zero_grad()
        previous.loss_with_future(window(train,step%len(train),(step*31)%296))['loss'].backward()
        prior_opt.step()
    saved={'model':copy.deepcopy(previous.state_dict()),
           'optimizer':copy.deepcopy(prior_opt.state_dict())}
    cfg_cur=copy.deepcopy(cfg);cfg_cur.triangle_curvature=True
    candidate=MirrorLM(cfg_cur)
    candidate_opt=torch.optim.AdamW(candidate.parameters(),lr=.001)
    expand_architecture(candidate,candidate_opt,saved,cfg)
    assert len(prior_opt.state)==len(candidate_opt.state)
    ev=[window(dev,i,j) for i in range(len(dev)) for j in (1,47,113)]
    before_r34=measure(previous,ev)
    before_r35=measure(candidate,ev)
    assert before_r34==before_r35,'Cold R35 migration changed predictions'

    # Matched updates: same initial old tensors, LR, windows, and batch order.
    for step in range(updates):
        seq=window(train,step%len(train),(step*19)%296)
        for model,opt in ((previous,prior_opt),(candidate,candidate_opt)):
            model.train();opt.zero_grad()
            loss=model.loss_with_future(seq)['loss']
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    full=measure(candidate,ev)
    g=candidate.blocks[-1].ff.symbiotic
    g.curvature_enabled=False
    no_curvature=measure(candidate,ev)
    g.curvature_enabled=True
    result={'workload':'TOY synthetic motif grammar; not NaNROUTER/FA2 held-out books',
            'updates_each':updates,'inherited_adam_parameters':len(prior_opt.state),
            'before_equal':before_r34,'r34_control_after':measure(previous,ev),
            'r35_after':full,'r35_no_curvature_after':no_curvature,
            'ce_gain_from_curvature_ablation':round(no_curvature['ce_nats_per_byte']-full['ce_nats_per_byte'],6),
            'curvature_gate':round(float(g.curvature.gate.detach()),7)}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main(int(sys.argv[1]) if len(sys.argv)>1 else 24)
