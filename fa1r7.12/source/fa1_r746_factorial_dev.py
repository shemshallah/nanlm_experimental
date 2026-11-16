#!/usr/bin/env python3
"""R7.46 causal geometric-word-read attribution on untouched development books.

The model and optimizer are loaded from the ONE R7.45 checkpoint and never
modified.  Each matched ablation changes pre-target context only, holding the
current target, inherited R7 state, and causal copy candidates constant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from fa1_r723_train import prepare, forward
from fa1_r725_controls import different_book_donors
from fa1_r727_checkpoint import load_unified, digest
from fa1_r727_eval import validate_book_corpus
from fa1_r744_cpu_pilot import bootstrap_by_book, foreign_word_history
from fa1_r745_geometric_value_read import ROUND, geometric_value_read_disabled


def perturbed_context(data, field_group, seed):
    """Different-book donors preserve labels and every other causal tensor."""
    fields={
        'addresses_only': ('vertex_ids',),
        'lexical_towers_only': ('word_ids', 'byte_ids', 'sentence_ids'),
        'byte_sentence_only': ('byte_ids', 'sentence_ids'),
        'all_geometry': ('vertex_ids', 'word_ids', 'byte_ids', 'sentence_ids'),
    }
    if field_group not in fields:
        raise ValueError('unknown factor: '+field_group)
    donors=different_book_donors(data['book_ids'], seed)
    changed=dict(data)
    for field in fields[field_group]:
        changed[field]=data[field][donors]
    return changed


def same_points_new_order(data):
    """A point-order intervention retains every active global head address."""
    changed=dict(data)
    changed['vertex_ids']=data['vertex_ids'].roll(1,dims=1)
    return changed


@torch.no_grad()
def score(model, data, mode='full'):
    model.eval()
    current, extras=forward(model,data,mode)
    with geometric_value_read_disabled(model):
        parent,_=forward(model,data,mode)
    return {'candidate':current.detach(), 'inherited':parent.detach(),
            'incremental':(parent-current).detach(), 'extras':extras}


def _mean(tensor,mask):
    return float(tensor[mask].mean()) if bool(mask.any()) else None


def run(a):
    if Path(a.output).exists():
        raise FileExistsError('R7.46 output exists; do not overwrite prior research')
    started=time.monotonic()
    torch.set_num_threads(a.threads)
    inventory=validate_book_corpus(a.corpus_dir,'dev')
    model,opt,base,comp,meta=load_unified(a.checkpoint,a.root,a.threads)
    if meta['architecture_round']!=ROUND or len(opt.param_groups)!=26 or meta['promoted']:
        raise ValueError('R7.46 requires the exact unpromoted 26-group R7.45 model')
    dev=prepare(a,model,base,comp,'dev',a.targets,a.seed)
    books=dev['book_ids']
    if torch.unique(books).numel()!=15:
        raise ValueError('matched evaluation requires all 15 development books')
    baseline=score(model,dev)
    gain=baseline['incremental']
    candidate_ids=baseline['extras']['backoff_candidate_ids']
    candidate_weights=baseline['extras']['backoff_candidate_weights']
    is_copied=((candidate_ids==dev['target'][:,None]) &
               (candidate_weights>0)).any(dim=1)
    groups={'copied_target':is_copied,'absent_target':~is_copied}
    interventions={
        'addresses_only':perturbed_context(dev,'addresses_only',a.seed+11),
        'lexical_towers_only':perturbed_context(dev,'lexical_towers_only',a.seed+12),
        'byte_sentence_only':perturbed_context(dev,'byte_sentence_only',a.seed+13),
        'all_geometry':perturbed_context(dev,'all_geometry',a.seed+14),
        'same_points_new_order':same_points_new_order(dev),
    }
    controls={}
    for index,(label,modified) in enumerate(interventions.items()):
        # Only the specified causal context fields may change. The original
        # targets and actual copied-word-history IDs are never overridden.
        for immutable in ('target','hidden','book_ids'):
            if not torch.equal(modified[immutable],dev[immutable]):
                raise AssertionError('ablation changed immutable '+immutable)
        result=score(model,modified)
        specific=gain-result['incremental']
        controls[label]={
            'counterfactual_incremental_word_gain':float(result['incremental'].mean()),
            'true_minus_counterfactual':float(specific.mean()),
            'whole_book_bootstrap':bootstrap_by_book(specific,books,a.seed+31+index),
            'copied_target_delta':_mean(specific,is_copied),
            'absent_target_delta':_mean(specific,~is_copied),
        }
        print('R746',label,'true-minus-control',
              round(controls[label]['true_minus_counterfactual'],8),flush=True)
    no_apex=score(model,dev,'no_apex')
    no_pair=score(model,dev,'no_pair')
    for index,(label,result) in enumerate((('no_apex',no_apex),('no_pair',no_pair))):
        specific=gain-result['incremental']
        controls[label]={
            'counterfactual_incremental_word_gain':float(result['incremental'].mean()),
            'true_minus_counterfactual':float(specific.mean()),
            'whole_book_bootstrap':bootstrap_by_book(specific,books,a.seed+46+index),
        }
    wrong_history,_=foreign_word_history(dev,a.seed+49)
    foreign=score(model,wrong_history)
    history_loss=foreign['candidate']-baseline['candidate']
    # Per-book estimates reveal whether an average improvement hides broad
    # regressions. The source's book-disjoint train/dev/test assignment is
    # validated above; no TEST positions are opened in this audit.
    per_book=[]
    for book in torch.unique(books):
        selected=books==book
        per_book.append({'book_id':int(book), 'positions':int(selected.sum()),
                         'mean_incremental_word_gain':float(gain[selected].mean()),
                         'copied_fraction':float(is_copied[selected].float().mean())})
    report={
        'round':'FA1_R7_46_FACTORIAL_ATTRIBUTION_V1',
        'checkpoint_sha256':digest(a.checkpoint),
        'checkpoint_round':meta['architecture_round'],
        'optimizer_groups':len(opt.param_groups),
        'changed_model_parameters':False,
        'independent_test_opened':False,
        'promoted':False,
        'corpus_inventory':inventory,
        'dev_books':int(torch.unique(books).numel()),
        'dev_positions':len(gain),
        'parent_r744_word_nll':float(baseline['inherited'].mean()),
        'r745_word_nll':float(baseline['candidate'].mean()),
        'incremental_word_gain':float(gain.mean()),
        'incremental_book_bootstrap':bootstrap_by_book(gain,books,a.seed+60),
        'correct_minus_foreign_history_advantage':float(history_loss.mean()),
        'correct_history_bootstrap':bootstrap_by_book(history_loss,books,a.seed+61),
        'copy_target_fraction':float(is_copied.float().mean()),
        'by_copy_status':{name:{'positions':int(mask.sum()),
                                 'parent_nll':_mean(baseline['inherited'],mask),
                                 'candidate_nll':_mean(baseline['candidate'],mask),
                                 'incremental_word_gain':_mean(gain,mask)}
                          for name,mask in groups.items()},
        'controls':controls,
        'per_book':per_book,
        'elapsed_seconds':time.monotonic()-started,
    }
    out=Path(a.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--corpus-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--targets',type=int,default=3072)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--seed',type=int,default=746)
    run(p.parse_args())
