#!/usr/bin/env python3
"""Prepare 55-book VAL histories and run a safe dedicated native process.
We DO NOT invoke fork() inside Python/Torch/BLAS after threads initialize.
"""
import os,sys,argparse,struct,json,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/nancm/v175'),str(ROOT/'experiments/nancm/v176'),str(ROOT/'experiments/nancm/v178'),str(ROOT/'parent/scripts'),str(ROOT/'scripts')]
from train_gate163 import setup
from test_native_speller156 import MODEL
from train_long175 import load_models,load_split
from test_trained178 import raw_text

def main():
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=24);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 import torch;torch.set_num_threads(2)
 _,_,names,_,_=load_models();X,Y,B=load_split('val',ROOT/'results/v175_release3')
 rng=np.random.default_rng(179);indices=[];seen=set()
 for j in rng.permutation(len(Y)):
  b=int(B[j]);
  if b not in seen:seen.add(b);indices.append(int(j))
  if len(indices)>=a.n:break
 assert len(indices)>=2
 cases=ROOT/'results/v179_book_contexts.bin'
 with cases.open('wb') as out:
  out.write(struct.pack('<I',len(indices)))
  for j in indices:
   raw=raw_text(X[j],names)
   out.write(struct.pack('<I',len(raw)));out.write(raw)
 setup(core=False,domain=True,gate=None)
 os.environ.update(V164_DISTRIBUTION_BUDGET='.6',V164_DISTRIBUTION_INDEX=str(ROOT/'checkpoints/v164_native_distribution.tsv'),V164_DOMAIN_GATE='1',V164_DOMAIN_THRESHOLD='1.5',V166_BIRTH_MODEL=str(ROOT/'checkpoints/v166_lexical_birth.bin'),V166_LEXICAL_BIRTH='.05',V167_GATE_MODE='1',V167_DOMAIN_MIN='1.4',V167_HIGH_ALPHA='.5',V167_TRUST_THRESHOLD='.5',V168_ANN_INDEX=str(ROOT/'checkpoints/v168_signed_ann.bin'),V168_ANN_SIGNED_SHARE='.66',V168_ANN_DOMAIN_MIN='1.4',V168_ANN_STRENGTH='.05',V169_NEURAL_RERANK='.05')
 output=ROOT/f'results/v179_cow_{a.workers}workers_trained.json'
 cmd=[str(ROOT/'bin/native_trained179'),str(ROOT/'checkpoints/v175_native_episodic_48.bin'),*(x.decode() for x in MODEL),str(ROOT/'checkpoints/v169_native_byte_lattice.bin'),str(ROOT/'checkpoints/v176_native_crosslevel.bin'),str(ROOT/'checkpoints/v176_native_bpefirst.bin'),str(cases),str(a.workers),str(output)]
 env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
 result=subprocess.run(cmd,cwd=ROOT,env=env,text=True,capture_output=True,timeout=300)
 print(result.stdout,flush=True);print(result.stderr,file=sys.stderr,flush=True)
 if result.returncode:raise RuntimeError(f'native test exited {result.returncode}')
 data=json.loads(output.read_text());assert data['status']=='PASS'
 print(json.dumps(data,indent=2),flush=True)
if __name__=='__main__':main()
