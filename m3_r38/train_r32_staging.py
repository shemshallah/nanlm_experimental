"""Real-corpus CPU *staging* for R38 with certified R32 as frozen parent.

R37 is NOT on GitHub master. These scores are explicitly not an R37
continuation, nor untouched TEST. Existing FIT/GATE/DEV windows are reused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from m3_default.predictor import R32_CHECKPOINT, R32_TRAINING_CHECKPOINT_SHA256
from m3_r32.anchor import M3R32Predictor, load_r32_checkpoint
from m3_r32.train_dev import selective_books
from m3_r38.optimizer import ContinuationAdam
from m3_r38.experiment import evaluate, fit, strict_gate

FIT=(540,556)
GATE=(556,560)
DEV=(596,608)
SEALED_R34_TEST=(572,596)


def run(root:Path,out:Path):
    parent=load_r32_checkpoint(R32_CHECKPOINT)
    optimizer=ContinuationAdam(R32_CHECKPOINT,R32_TRAINING_CHECKPOINT_SHA256,
                               expected_groups=32)
    factory=lambda: M3R32Predictor(parent.prior,parent.anchor,parent.anchor_theta)
    needed=set(range(*FIT))|set(range(*GATE))|set(range(*DEV))
    paths,contents=selective_books(root,DEV[1],needed)
    if len({hashlib.sha256(contents[i]).hexdigest() for i in needed})!=len(needed):
        raise ValueError('duplicate books among the explicit FIT/GATE/DEV')
    trained=fit([contents[i] for i in range(*FIT)],factory,optimizer,
                warmup=64,batch=64)
    # This development result is exploratory: books were opened in R35.
    gate=evaluate([contents[i] for i in range(*GATE)],factory,optimizer.theta,warmup=64)
    dev=evaluate([contents[i] for i in range(*DEV)],factory,optimizer.theta,warmup=64)
    result={'lineage':'M3 R38 on R32 for staging ONLY; NOT trained R37',
            'parent_sha256':R32_TRAINING_CHECKPOINT_SHA256,
            'fit_indices':FIT,'gate_indices':GATE,'dev_indices':DEV,
            'previously_opened_gate_dev':True,
            'sealed_r34_test_indices':SEALED_R34_TEST,
            'sealed_r34_test_read':False,
            'promoted':False,'production_default_on_remote':'r32',
            'train':trained,'gate':gate,'dev':dev,'gate_decision':strict_gate(gate),
            'dev_decision':strict_gate(dev)}
    out.mkdir(parents=True,exist_ok=True)
    optimizer.save(out/'R38_R32_44GROUP_STAGING.npz',fit_indices=FIT,
                   gate_indices=GATE,dev_indices=DEV)
    (out/'R38_R32_STAGING.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'fit':trained,'gate':gate['r38'],'dev':dev['r38'],
                      'dev_decision':result['dev_decision']},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus-root',required=True,type=Path)
    p.add_argument('--output-dir',default=Path('m3_r38/results'),type=Path)
    args=p.parse_args()
    run(args.corpus_root,args.output_dir)

if __name__=='__main__':main()