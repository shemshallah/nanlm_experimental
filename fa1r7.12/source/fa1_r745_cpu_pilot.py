#!/usr/bin/env python3
"""R7.45 single-lineage geometric word-value attention CPU continuation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from fa1_r723_train import prepare, take, forward
from fa1_r725_controls import different_book_donors
from fa1_r727_checkpoint import load_unified, save_unpromoted, digest
from fa1_r727_eval import validate_book_corpus
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history
from fa1_r745_geometric_value_read import (ROUND, attach_geometric_value_read,
                                            geometric_value_read_disabled)


def foreign_geometry(data, seed):
    donors=different_book_donors(data['book_ids'],seed)
    wrong=dict(data)
    for field in ('vertex_ids','byte_ids','sentence_ids'):
        wrong[field]=data[field][donors]
    return wrong


@torch.no_grad()
def evaluate(model,dev,wrong_hist,wrong_geom,seed):
    model.eval()
    good,ex=forward(model,dev)
    foreign,_=forward(model,wrong_hist)
    foreign_geom,bad_ex=forward(model,wrong_geom)
    no_apex,no_apex_ex=forward(model,dev,'no_apex')
    with geometric_value_read_disabled(model):
        parent,_=forward(model,dev)
        parent_wrong_geom,_=forward(model,wrong_geom)
        parent_no_apex,_=forward(model,dev,'no_apex')
    gain=parent-good
    specificity=gain-(parent_wrong_geom-foreign_geom)
    apex_specificity=gain-(parent_no_apex-no_apex)
    history=foreign-good
    books=dev['book_ids']
    gain_ci=bootstrap_by_book(gain,books,seed)
    specific_ci=bootstrap_by_book(specificity,books,seed+1)
    apex_ci=bootstrap_by_book(apex_specificity,books,seed+2)
    history_ci=bootstrap_by_book(history,books,seed+3)
    result={'r744_parent_word_nll':float(parent.mean()),
            'r745_candidate_word_nll':float(good.mean()),
            'r745_incremental_word_gain':float(gain.mean()),
            'r745_incremental_word_bootstrap':gain_ci,
            'r745_geometric_address_specificity':float(specificity.mean()),
            'r745_geometric_address_bootstrap':specific_ci,
            'r745_apex_specificity':float(apex_specificity.mean()),
            'r745_apex_bootstrap':apex_ci,
            'r745_correct_vs_foreign_history':float(history.mean()),
            'r745_correct_vs_foreign_bootstrap':history_ci,
            'r745_mean_geometry_mixture':float(ex['r745_geometric_beta'].mean()),
            'r745_mean_copy_mixture':float(ex['backoff_alpha'].mean()),
            'independent_books':int(torch.unique(books).numel()),
            'prediction_positions':len(good)}
    result['r745_strict_dev_gate_passed']=bool(
        result['r745_incremental_word_gain']>1.e-4 and
        result['r745_geometric_address_specificity']>0 and
        result['r745_apex_specificity']>0 and
        result['r745_correct_vs_foreign_history']>0 and
        gain_ci['book_bootstrap_ci95'][0]>0 and
        specific_ci['book_bootstrap_ci95'][0]>0 and
        apex_ci['book_bootstrap_ci95'][0]>0 and
        history_ci['book_bootstrap_ci95'][0]>0)
    return result


def run(a):
    if Path(a.output_checkpoint).exists() or Path(a.report).exists():
        raise FileExistsError('R7.45 outputs must be new files')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    t=time.monotonic()
    inventory=validate_book_corpus(a.corpus_dir,'dev')
    source_sha=digest(a.checkpoint)
    model,opt,base,comp,meta=load_unified(a.checkpoint,a.root,a.threads)
    continuing = meta['architecture_round']==ROUND
    if ((not continuing and (meta['architecture_round']!='FA1_R7_44_CAUSAL_WORD_BACKOFF_V1'
                             or len(opt.param_groups)!=25)) or
            (continuing and len(opt.param_groups)!=26) or meta['promoted'] is not False):
        raise ValueError('R7.45 must inherit a complete unpromoted R7.44 or R7.45 checkpoint')
    print('R745 RESTORED',source_sha,flush=True)
    train=prepare(a,model,base,comp,'train',a.train,a.seed+101)
    dev=prepare(a,model,base,comp,'dev',a.dev,a.seed+202)
    if len(torch.unique(dev['book_ids']))<12:
        raise ValueError('need at least 12 development books')
    wrong_hist,_=foreign_word_history(dev,a.seed+303)
    wrong_geom=foreign_geometry(dev,a.seed+404)
    model.eval()
    with torch.no_grad():
        before,_=forward(model,dev)
    if continuing:
        new_params=[model.word_backoff.geometric_gain,
                    model.word_backoff.geometric_query.weight]
        new_ids={id(p) for p in new_params}
        old_params=[p for p in model.parameters() if id(p) not in new_ids]
        for p in old_params:p.requires_grad_(False)
        for p in new_params:p.requires_grad_(True)
        opt._adamw.param_groups[-1]['lr']=a.lr
        with geometric_value_read_disabled(model),torch.no_grad():
            inherited,_=forward(model,dev)
        if torch.equal(before,inherited):
            raise AssertionError('R7.45 continuation lost its trained geometric contribution')
        print('R745 RESUMED original geometric query and Adam moments',flush=True)
    else:
        old_params=list(model.parameters())
        for p in old_params:p.requires_grad_(False)
        attach_geometric_value_read(model,opt,learning_rate=a.lr)
        with torch.no_grad():
            after,_=forward(model,dev)
        if not torch.equal(before,after):
            raise AssertionError('R7.45 zero migration did not recover R7.44 exactly')
    if len(opt.param_groups)!=26:
        raise AssertionError('expected one optimizer and 26 groups')
    print('R745 PARITY/CONTINUITY verified, inherited moments preserved',flush=True)
    if not continuing:
        with torch.no_grad():model.word_backoff.geometric_gain.fill_(a.open_gain)
    new_params=[model.word_backoff.geometric_gain,
                model.word_backoff.geometric_query.weight]
    rng=np.random.default_rng(a.seed+505)
    progress=[]
    best=None
    best_weights=None
    best_moments=None
    evaluation_steps={1,a.steps,*a.eval_steps}
    for step in range(1,a.steps+1):
        ids=torch.as_tensor(rng.choice(a.train,a.minibatch,replace=False),dtype=torch.long)
        sample=take(train,ids)
        model.eval()
        good,ex=forward(model,sample)
        if a.geometry_contrast>0:
            other=foreign_geometry(sample,a.seed+step)
            wrong,wrong_ex=forward(model,other)
            with geometric_value_read_disabled(model):
                inherited,_=forward(model,sample)
                inherited_wrong,_=forward(model,other)
            specificity=((inherited-good)-(inherited_wrong-wrong)).mean()
            geometric_contrast=F.softplus((a.margin-specificity)*10.)/10.
        else:
            specificity=good.new_zeros(())
            geometric_contrast=good.new_zeros(())
        loss=good.mean()+a.geometry_contrast*geometric_contrast
        opt._adamw.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None for p in old_params):
            raise AssertionError('R7.45 inherited gradient appeared')
        torch.nn.utils.clip_grad_norm_(new_params,1.)
        opt._adamw.step()
        if step in evaluation_steps:
            dev_eval=evaluate(model,dev,wrong_hist,wrong_geom,a.seed+step)
            dev_eval.update({'step':step,'train_objective':float(loss.detach()),
                             'train_geometry_specificity':float(specificity.detach())})
            progress.append(dev_eval)
            print('R745 STEP',step,'word gain',round(dev_eval['r745_incremental_word_gain'],7),
                  'address',round(dev_eval['r745_geometric_address_specificity'],7),
                  'apex',round(dev_eval['r745_apex_specificity'],7),
                  'gate',dev_eval['r745_strict_dev_gate_passed'],flush=True)
            rank=(int(dev_eval['r745_strict_dev_gate_passed']),
                  -dev_eval['r745_candidate_word_nll'])
            if best is None or rank>best[0]:
                best=(rank,dev_eval)
                best_weights={n:p.detach().clone() for n,p in
                              model.word_backoff.named_parameters()
                              if n.startswith('geometric_')}
                best_moments={n:copy.deepcopy(opt._adamw.state.get(p,{}))
                              for n,p in model.word_backoff.named_parameters()
                              if n.startswith('geometric_')}
    for name,p in model.word_backoff.named_parameters():
        if name in best_weights:
            with torch.no_grad():p.copy_(best_weights[name])
            opt._adamw.state[p]=copy.deepcopy(best_moments[name])
    if digest(a.checkpoint)!=source_sha:
        raise AssertionError('protected source checkpoint changed')
    checkpoint=save_unpromoted(a.output_checkpoint,model,opt,a.root,a.checkpoint,
                               source_metadata=meta,candidate_trained=True,
                               training_round=ROUND,training_updates=best[1]['step'],
                               development_gate_passed=False,architecture_round=ROUND)
    recovered,rec_opt,_,_,rec_meta=load_unified(a.output_checkpoint,a.root,a.threads)
    if len(rec_opt.param_groups)!=26 or rec_meta['architecture_round']!=ROUND:
        raise AssertionError('R7.45 failed exact optimizer/model recovery')
    for name,weight in model.state_dict().items():
        if not torch.equal(weight,recovered.state_dict()[name]):
            raise AssertionError('R7.45 tensor roundtrip failed: '+name)
    result={'round':ROUND, 'accepted_parent_step':14416, 'branch':'master',
            'source_checkpoint_sha256':source_sha,
            'one_optimizer_groups':26,'source_checkpoint_never_overwritten':True,
            'candidate_checkpoint_promoted':False,
            'independent_test_opened':False,
            'dataset_inventory':inventory,'train_positions':a.train,
            'dev_positions':a.dev,'train_updates':a.steps,
            'continuation_from_saved_optimizer':continuing,
            'source_training_updates':meta.get('candidate_training_updates',0),
            'total_lineage_training_updates':meta.get('candidate_training_updates',0)+best[1]['step'],
            'selected_step':best[1]['step'],
            'selected_strict_dev_pass':best[1]['r745_strict_dev_gate_passed'],
            'promotion':'rejected: full system gate unproven; no independent TEST',
            'candidate_checkpoint':checkpoint,
            'development_history':progress,
            'seconds':time.monotonic()-t}
    out=Path(a.report)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('R745 DONE',json.dumps({'sha256':checkpoint['sha256'],
          'step':best[1]['step'],'word_gain':best[1]['r745_incremental_word_gain'],
          'seconds':round(result['seconds'],1)}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output-checkpoint',required=True)
    p.add_argument('--report',required=True)
    p.add_argument('--train',type=int,default=1024)
    p.add_argument('--dev',type=int,default=384)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--minibatch',type=int,default=12)
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--eval-steps',type=int,nargs='*',default=[16,32,64,128])
    p.add_argument('--seed',type=int,default=745)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--open-gain',type=float,default=.25)
    p.add_argument('--lr',type=float,default=.0004)
    p.add_argument('--geometry-contrast',type=float,default=.3)
    p.add_argument('--margin',type=float,default=.002)
    run(p.parse_args())
