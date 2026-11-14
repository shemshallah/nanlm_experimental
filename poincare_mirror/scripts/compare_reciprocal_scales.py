#!/usr/bin/env python3
"""Small, matched-CPU, held-out synthetic recall sanity check for unified FA2.

This is NOT a substitute for checkpoint-carrying book-disjoint evaluation.
The byte-aligned +1/+2/+3 metrics all target exactly the same held-out query
bytes, with each horizon predicted from its own earlier causal position.
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, expand_architecture


def document(seed):
    generator = torch.Generator().manual_seed(seed)
    mapping = torch.randperm(8,generator=generator)
    # All documents have the same layout; mappings change independently.
    pairs = b''.join(bytes((65+i,61,97+int(mapping[i]),59)) for i in range(8))
    noise = b'...#!' * 5
    queries = b''.join(bytes((63,65+i,61,97+int(mapping[i]),59)) for i in range(8))
    blob = pairs + noise + queries + b'###'  # all three horizons valid for final query
    targets = [len(pairs)+len(noise)+5*i+3 for i in range(8)]
    return torch.tensor(list(blob),dtype=torch.long),targets


@torch.no_grad()
def evaluate(model,seeds):
    model.eval()
    predictions = {1:[],2:[],3:[]}
    targets = []
    for seed in seeds:
        doc,positions = document(seed)
        ids = doc[None,:-3]
        logits,h,_ = model(ids)
        p2,p3 = model.future_logits(h)
        for q in positions:
            if q >= ids.shape[1]+1: continue
            targets.append(int(doc[q]))
            for horizon,head in [(1,logits),(2,p2),(3,p3)]:
                predictions[horizon].append(head[0,q-horizon].float())
    labels = torch.tensor(targets)
    output = {}
    for horizon in [1,2,3]:
        logits = torch.stack(predictions[horizon])
        ce = F.cross_entropy(logits,labels).item()
        ranked = logits.topk(5,-1).indices
        output['+'+str(horizon)] = {
            'positions':len(labels),'ce_nats_per_byte':round(ce,6),
            'bits_per_byte':round(ce/math.log(2),6),
            'top1':round((ranked[:,0]==labels).float().mean().item(),6),
            'top5':round((ranked==labels[:,None]).any(-1).float().mean().item(),6)}
    return output


def run(updates):
    torch.set_num_threads(1)
    torch.manual_seed(398)
    cfg = Config(vocab_size=257,dim=32,heads=4,layers=2,pairs=4,
                 topk_pairs=2,expert_hidden=16,context=128,moe_every=2,
                 hierarchy_rank=8,symbiotic_rank=8,micro_meta=True,
                 simplex_transfer=True,geometry_memory_chunk=8,
                 future_mode='isolated',future_adapter_rank=8)
    control = MirrorLM(cfg)
    control_opt = torch.optim.AdamW(control.parameters(),lr=0.001)
    saved = {'model':copy.deepcopy(control.state_dict()),
             'optimizer':copy.deepcopy(control_opt.state_dict())}
    new_cfg = copy.deepcopy(cfg)
    new_cfg.reciprocal_scales = True
    candidate = MirrorLM(new_cfg)
    candidate_opt = torch.optim.AdamW(candidate.parameters(),lr=0.001)
    expand_architecture(candidate,candidate_opt,saved,cfg)
    before = evaluate(control,range(81000,81016))
    for step in range(updates):
        doc,_ = document(1000+step)
        for model,opt in [(control,control_opt),(candidate,candidate_opt)]:
            model.train();opt.zero_grad(set_to_none=True)
            losses = model.loss_with_future(doc[None])
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
    reference = evaluate(control,range(81000,81016))
    learned = evaluate(candidate,range(81000,81016))
    candidate.reciprocal.enabled = False
    no_bridge = evaluate(candidate,range(81000,81016))
    return {'disclaimer':'synthetic random key-value documents, not real books',
            'seed':398,'updates_per_arm':updates,'documents_per_arm':updates,
            'heldout_docs':16,'heldout_query_bytes':128,
            'initial':before,'r37_control':reference,
            'reciprocal_candidate':learned,
            'candidate_bridge_disabled_at_endpoint':no_bridge}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=64)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1 <= args.updates <= 2000: parser.error('updates outside 1..2000')
    result=run(args.updates)
    result=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(result)
    print(result)


if __name__=='__main__':
    main()
