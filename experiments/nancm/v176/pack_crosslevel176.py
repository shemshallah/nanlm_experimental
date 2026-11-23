#!/usr/bin/env python3
"""Deterministic native sidecar, TRAIN-only word-root counts + trained gate.
All parent weights, optimizer groups, and cloud fingerprint hashes retained.
"""
from pathlib import Path
import sys,hashlib,struct,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(ROOT/'experiments/nancm/v175'))
from train_long175 import load_models,load_split,verify_corpus
from eval_crosslevel176 import CausalLetter

def main():
 model,opt,names,fp,old=load_models();source=ROOT/'results/v175_release3';verify_corpus(source,names)
 X,Y,B=load_split('train',source)
 letter=CausalLetter(X,Y,names)
 ck=torch.load(ROOT/'checkpoints/v176_counterfactual_word_gate.pt',weights_only=False,map_location='cpu')
 parent_hash=hashlib.sha256((ROOT/'checkpoints/v175_signed_episodic_48.pt').read_bytes()).hexdigest()
 assert ck['inherited_checkpoint_sha256']==parent_hash
 weight=ck['gate_state_dict']['weight'].numpy().astype('<f4');bias=float(ck['gate_state_dict']['bias'])
 assert len(weight)==14 and np.isfinite(weight).all() and np.isfinite(bias)
 gp=letter.global_count.astype('<u4');c1=letter.c1.astype('<u4');items=sorted(letter.c2.items())
 payload=bytearray(b'N176HYB1')
 payload.extend(struct.pack('<3Q',*map(int,fp)))
 payload.extend(bytes.fromhex(parent_hash))
 payload.extend(struct.pack('<I f I I',12000,24.,14,len(items)))
 payload.extend(weight.tobytes());payload.extend(struct.pack('<f',bias))
 payload.extend(gp.tobytes());payload.extend(c1.tobytes())
 for (a,b),count in items:
  payload.extend(struct.pack('<ii',int(a),int(b)))
  payload.extend(count.astype('<u4').tobytes())
 out=ROOT/'checkpoints/v176_native_crosslevel.bin';out.write_bytes(payload)
 sha=hashlib.sha256(payload).hexdigest()
 info={'path':str(out.relative_to(ROOT)),'sha256':sha,'bytes':len(payload),'magic':'N176HYB1',
       'book_count':396,'train_examples':len(Y),'distinct_last_two_contexts':len(items),'vocab':12000,
       'word_letter_prior':24.0,'root_calibration_beta':.5,'gate_step':ck['step'],
       'gate_optimizer_checkpoint':'checkpoints/v176_counterfactual_word_gate.pt',
       'inherited_v175_sha256':parent_hash,'parent_cloud_fingerprints':list(map(int,fp))}
 (ROOT/'results/v176_native_pack.json').write_text(json.dumps(info,indent=2))
 print('V176 PACK',json.dumps(info),flush=True)
if __name__=='__main__':main()