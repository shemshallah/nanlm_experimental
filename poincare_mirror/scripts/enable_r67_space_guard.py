#!/usr/bin/env python3
"""Install R67 space/EOS rank guard in the SAME FA2 model checkpoint.

This is a configuration-level inference improvement: all trained tensors,
196 inherited Adam states and the R66 joint Adam state remain identical.
It never opens a training, DEV, or TEST corpus.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config,MirrorLM,file_hash,load_v25_compatible


def enable_guard(saved):
    if not saved.get("config",{}).get("pair_successor_joint"):
        raise ValueError("R67 requires the trained joint memory in the same FA2 model")
    updated=dict(saved)
    config=dict(saved['config'])
    if config.get('pair_successor_space_rank_guard'):
        raise ValueError('Space rank guard already enabled')
    config['pair_successor_space_rank_guard']=True
    model=MirrorLM(Config(**config)).eval()
    load_v25_compatible(model,saved['model'])
    for name,tensor in saved['model'].items():
        if not torch.equal(tensor,model.state_dict()[name]):
            raise AssertionError('Inference guard changed model tensor: '+name)
    updated['config']=config
    updated['fa2_r67_guard']={
        'method':'maximum causal probability blend preserving inherited space/EOS top1',
        'new_model_parameters':0,'optimizer_state_changes':0,
        'inherited_model_state_exact':True,
        'test_opened':False,'promoted':False}
    return updated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Refuse to overwrite existing checkpoint')
    if file_hash(args.checkpoint)!=args.expected_sha256:
        raise ValueError('Parent checksum does not match')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    guarded=enable_guard(saved)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(guarded,args.output)
    print(json.dumps({'status':'R67 candidate unpromoted',
                      'step':guarded['step'],
                      'candidate_sha256':file_hash(args.output),
                      'model_weights_identical':True,
                      'optimizer_history_identical':True},indent=2))


if __name__=='__main__':
    main()
