#!/usr/bin/env python3
"""Matched R35->R36 optimizer continuation on a reproducible toy grammar.

This is an integration/ablation pilot, not an FA2 checkpoint promotion test.
"""
import copy
import json
import sys

import torch

from r35_toy_pilot import documents, measure
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, expand_architecture


def main(updates=320):
    torch.set_num_threads(2)
    torch.manual_seed(360)
    cfg=Config(vocab_size=48,dim=32,heads=4,layers=2,pairs=4,topk_pairs=2,
               expert_hidden=24,context=20,moe_every=2,symbiotic_rank=8,
               micro_meta=True,simplex_transfer=True,triangle_curvature=True)
    baseline=MirrorLM(cfg)
    old_optimizer=torch.optim.AdamW(baseline.parameters(),lr=.001)
    train=documents(120,12,320)
    dev=documents(121,4,240)
    def window(docs,doc,start):
        return docs[doc][start:start+23].unsqueeze(0)
    for i in range(64):
        baseline.train();old_optimizer.zero_grad()
        seq=window(train,i%len(train),(i*19)%296)
        baseline.loss_with_future(seq)['loss'].backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.)
        old_optimizer.step()
    saved={'model':copy.deepcopy(baseline.state_dict()),
           'optimizer':copy.deepcopy(old_optimizer.state_dict())}
    newcfg=copy.deepcopy(cfg);newcfg.holonomy_retrieval=True
    proposed=MirrorLM(newcfg)
    proposed_optimizer=torch.optim.AdamW(proposed.parameters(),lr=.001)
    expand_architecture(proposed,proposed_optimizer,saved,cfg)
    auxiliary_only=copy.deepcopy(proposed)
    auxiliary_only.blocks[-1].ff.symbiotic.holonomy_feedback_enabled=False
    auxiliary_optimizer=torch.optim.AdamW(auxiliary_only.parameters(),lr=.001)
    auxiliary_optimizer.load_state_dict(copy.deepcopy(proposed_optimizer.state_dict()))
    inherited=len(old_optimizer.state)
    assert inherited==len(proposed_optimizer.state)
    evaluation=[window(dev,i,j) for i in range(4) for j in (1,47,113)]
    before=measure(baseline,evaluation)
    assert before==measure(proposed,evaluation)
    assert before==measure(auxiliary_only,evaluation)
    for i in range(updates):
        seq=window(train,i%len(train),(i*19)%296)
        for model,opt in ((baseline,old_optimizer),(proposed,proposed_optimizer),
                          (auxiliary_only,auxiliary_optimizer)):
            model.train();opt.zero_grad()
            loss=model.loss_with_future(seq)['loss']
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    after=measure(proposed,evaluation)
    g=proposed.blocks[-1].ff.symbiotic
    g.holonomy_enabled=False
    ablation=measure(proposed,evaluation)
    g.holonomy_enabled=True
    outcome={'workload':'TOY synthetic motif grammar (NOT NaNROUTER books)',
             'matched_updates_each':updates,
             'inherited_optimizer_tensors':inherited,
             'zero_drift_baseline':before,
             'r35_continued':measure(baseline,evaluation),
             'r36_fourier_continued':after,
             'r36_auxiliary_only':measure(auxiliary_only,evaluation),
             'r36_holonomy_off':ablation,
             'holonomy_ce_benefit':round(ablation['ce_nats_per_byte']-after['ce_nats_per_byte'],6),
             'holonomy_gate':round(float(g.curvature.holonomy.gate.detach()),6)}
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    main(int(sys.argv[1]) if len(sys.argv)>1 else 320)
