#!/usr/bin/env python3
"""Synthetic document-local delayed-recall diagnostic for R35 vs R36.

Mappings are randomized independently in every document. Local attention
is limited to eight positions, while geometric memory can read the full
document prefix. Query-target-only CE is reported separately from LM CE.
Not a NaNROUTER score and not a checkpoint promotion criterion alone.
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


def book(rng):
    values = rng.permutation(4)
    tokens = [39]
    for key in rng.permutation(4):
        tokens.extend([40,1+int(key),11+int(values[key]),41])
    tokens += rng.integers(42,48,size=5).tolist()
    queries = []
    for key in rng.permutation(4):
        tokens += [38,1+int(key)]
        queries.append(len(tokens)-1)
        tokens += [11+int(values[key])]
        tokens += rng.integers(42,48,size=3).tolist()
    tokens += rng.integers(42,48,size=4).tolist()
    assert len(tokens) == 50
    # The training caller uses ids[:-3] as causal input and labels ids[1:-2].
    return torch.tensor(tokens,dtype=torch.long).unsqueeze(0), queries


@torch.no_grad()
def evaluate(model,docs):
    model.eval()
    total_ce,top1,top5,n=0.,0,0,0
    for ids,locations in docs:
        logits=model(ids[:,:-3])[0][0,locations]
        targets=ids[0,torch.tensor(locations)+1]
        total_ce+=float(F.cross_entropy(logits,targets,reduction='sum'))
        top1+=int((logits.argmax(-1)==targets).sum())
        top5+=int((logits.topk(5).indices==targets[:,None]).any(-1).sum())
        n+=len(locations)
    ce=total_ce/n
    return {'query_ce':round(ce,6),'query_bits_per_byte':round(ce/math.log(2),6),
            'query_top1':round(top1/n,6),'query_top5':round(top5/n,6),'queries':n}


def update(model,opt,doc):
    ids,locations=doc
    model.train();opt.zero_grad()
    logits,h,metrics=model(ids[:,:-3],ids[:,1:-2])
    p2,p3=model.future_logits(h)
    targets=ids[:,1:-2]
    q_ce=F.cross_entropy(logits[:,locations].reshape(-1,model.cfg.vocab_size),
                         targets[:,locations].reshape(-1))
    future2=F.cross_entropy(p2.flatten(0,1),ids[:,2:-1].flatten())
    future3=F.cross_entropy(p3.flatten(0,1),ids[:,3:].flatten())
    (metrics['loss']+0.1*future2+0.05*future3+3.*q_ce).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()


def main(steps=240):
    torch.set_num_threads(2)
    torch.manual_seed(363)
    cfg=Config(vocab_size=48,dim=32,heads=4,layers=2,pairs=4,
               topk_pairs=2,expert_hidden=24,context=47,moe_every=2,
               attention_window=8,symbiotic_rank=8,micro_meta=True,
               simplex_transfer=True,triangle_curvature=True)
    r35=MirrorLM(cfg)
    optimizer=torch.optim.AdamW(r35.parameters(),lr=.001)
    train_rng=np.random.default_rng(1801)
    # Warm up R35 with actual prior optimizer state.
    for _ in range(32):
        update(r35,optimizer,book(train_rng))
    saved={'model':copy.deepcopy(r35.state_dict()),
           'optimizer':copy.deepcopy(optimizer.state_dict())}
    new_cfg=copy.deepcopy(cfg);new_cfg.holonomy_retrieval=True
    r36=MirrorLM(new_cfg)
    optim36=torch.optim.AdamW(r36.parameters(),lr=.001)
    expand_architecture(r36,optim36,saved,cfg)
    dev_rng=np.random.default_rng(1802)
    dev=[book(dev_rng) for _ in range(32)]
    start=evaluate(r35,dev)
    assert start==evaluate(r36,dev)
    # Compare on precisely matched stochastic training documents.
    for _ in range(steps):
        sample=book(train_rng)
        update(r35,optimizer,sample)
        update(r36,optim36,sample)
    full=evaluate(r36,dev)
    geometry=r36.blocks[-1].ff.symbiotic
    geometry.holonomy_enabled=False
    disabled=evaluate(r36,dev)
    geometry.holonomy_enabled=True
    print(json.dumps({'workload':'synthetic delayed key-value recall; 8-byte local attention',
                      'updates_each':steps,'inherited_adam_states':len(optimizer.state),
                      'cold_equal':start,'r35_control':evaluate(r35,dev),
                      'r36_retrieval':full,'r36_holonomy_disabled':disabled,
                      'holonomy_query_ce_gain':round(disabled['query_ce']-full['query_ce'],6),
                      'holonomy_gate':round(float(geometry.curvature.holonomy.gate.detach()),6)},
                     indent=2))


if __name__=='__main__':
    main(int(sys.argv[1]) if len(sys.argv)>1 else 240)
