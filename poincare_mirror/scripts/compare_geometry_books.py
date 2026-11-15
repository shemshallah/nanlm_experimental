#!/usr/bin/env python3
"""Matched, book-level bootstrap for the inherited FA2 LM and a geometry warm-up.

Only development books may select a candidate. The untouched test split can be
scored after selection, but this script never promotes or mutates a checkpoint.
Each model sees identical causal windows and target bytes from each whole book.
"""
import argparse
import json
import math
import sys
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import Config, MirrorLM, file_hash, load_meta, load_v25_compatible


METRICS = ('ce_1', 'ce_2', 'ce_3', 'acc_1', 'acc_2', 'acc_3',
           'top5_1', 'top5_2', 'top5_3',
           'ce_nonspace_1', 'acc_nonspace_1', 'top5_nonspace_1',
           'ce_space_1', 'acc_space_1', 'top5_space_1')


def book_starts(lengths, context, windows, seed):
    """Deterministic independent starts; no window may cross a book boundary."""
    rng = np.random.default_rng(seed)
    offset = 0
    records = []
    for length in lengths:
        possible = int(length) - context - 2
        if possible < 1:
            raise ValueError('A book is too short for a verified full window')
        starts = rng.choice(possible, size=windows, replace=possible < windows)
        records.append(offset + starts.astype(np.int64))
        offset += int(length)
    return records, offset


def bootstrap_gain(before, after, seed=2026, resamples=2000, larger_better=False):
    """Positive = after improved; resample independent books, never byte positions."""
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    if b.shape != a.shape or b.ndim != 1 or len(b) < 2:
        raise ValueError('Need at least two identically ordered whole books')
    changes = a-b if larger_better else b-a
    indices = np.random.default_rng(seed).integers(0,len(changes),(resamples,len(changes)))
    distribution = changes[indices].mean(1)
    return {'mean_gain':round(float(changes.mean()),6),
            'ci95':[round(float(x),6) for x in np.quantile(distribution,[.025,.975])],
            'books_improved':int((changes>0).sum()),'books':len(changes)}


def passes_content_gate(comparison, feedback, split, content_trust=None):
    """A significant easy-byte gain cannot hide a content-byte regression."""
    return bool(split == 'valid' and feedback is not None and
                comparison['ce_1']['ci95'][0] > 0 and
                comparison['ce_nonspace_1']['ci95'][0] > 0 and
                comparison['acc_nonspace_1']['mean_gain'] >= 0 and
                comparison['top5_nonspace_1']['mean_gain'] >= 0 and
                comparison['top5_1']['ci95'][0] >= -.005 and
                all(comparison[m]['mean_gain'] >= 0 for m in ('ce_2','ce_3')) and
                feedback['ce_1']['ci95'][0] > 0 and
                feedback['ce_nonspace_1']['ci95'][0] > 0 and
                (content_trust is None or
                 content_trust['ce_nonspace_1']['ci95'][0] > 0))


@torch.no_grad()
def measure_books(model, mem, records, context, batch_size=4):
    model.eval()
    rows = []
    device = torch.device('cpu')
    for starts in records:
        sums = np.zeros(len(METRICS),dtype=np.float64)
        count = 0
        nonspace_count = 0
        space_count = 0
        for start in range(0,len(starts),batch_size):
            positions = starts[start:start+batch_size]
            seq = np.stack([mem[int(p):int(p)+context+3] for p in positions])
            ids = torch.from_numpy(seq.astype(np.int64)).to(device)
            logits,h,_ = model(ids[:,:-3])
            p2,p3 = model.future_logits(h)
            for j,(pred,target) in enumerate(zip((logits,p2,p3),
                            (ids[:,1:-2],ids[:,2:-1],ids[:,3:]))):
                sums[j] += F.cross_entropy(pred.flatten(0,1),target.flatten(),
                                            reduction='sum').item()
                sums[3+j] += (pred.argmax(-1)==target).sum().item()
                sums[6+j] += (pred.topk(5,dim=-1).indices ==
                              target.unsqueeze(-1)).any(-1).sum().item()
                if j == 0:
                    mask = target != 32
                    n = int(mask.sum().item())
                    nonspace_count += n
                    space_count += int((~mask).sum().item())
                    ce = F.cross_entropy(pred.flatten(0,1),target.flatten(),
                                         reduction='none').reshape_as(target)
                    correct = pred.argmax(-1) == target
                    top5 = (pred.topk(5,dim=-1).indices ==
                            target.unsqueeze(-1)).any(-1)
                    for offset,subset in ((9,mask),(12,~mask)):
                        sums[offset] += ce[subset].sum().item()
                        sums[offset+1] += correct[subset].sum().item()
                        sums[offset+2] += top5[subset].sum().item()
            count += len(positions)*context
        if not nonspace_count or not space_count:
            raise ValueError('Every development book must contain both space and non-space targets')
        denominators = [count]*9 + [nonspace_count]*3 + [space_count]*3
        rows.append(dict(zip(METRICS,(sums/denominators).tolist())))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',type=Path,required=True,help='R30/latest.pt')
    ap.add_argument('--after',type=Path,required=True,help='candidate/latest.pt')
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--split',choices=('valid','test'),default='valid')
    ap.add_argument('--context',type=int,default=128)
    ap.add_argument('--windows-per-book',type=int,default=8)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if min(args.context,args.windows_per_book,args.batch,args.bootstrap,args.threads) <= 0:
        raise ValueError('Workload quantities must be positive')
    torch.set_num_threads(args.threads)
    before=torch.load(args.before,map_location='cpu',weights_only=False)
    after=torch.load(args.after,map_location='cpu',weights_only=False)
    if before['data']!=after['data'] or before['data']!=load_meta(args.data_dir):
        raise ValueError('Checkpoint/corpus mismatch; matched books not established')
    data=before['data']
    if data.get('split_unit')!='whole_book' or data.get('unit')!='UTF8_bytes_plus_eos':
        raise ValueError('Requires exact whole-book, byte+EOS checkpoint provenance')
    path=args.data_dir/(args.split+'.bin')
    expected=data.get(args.split+'_sha256') or data.get('bin_'+args.split+'_sha256')
    if not expected or file_hash(path)!=expected:
        raise ValueError('Book-split checksum mismatch')
    mem=np.memmap(path,dtype=data['token_dtype'],mode='r')
    records,total=book_starts(data[args.split+'_doc_lengths'],args.context,
                              args.windows_per_book,args.seed)
    if total!=len(mem):
        raise ValueError('Book offsets do not cover complete split')
    if max(args.context,4)>min(before['config']['context'],after['config']['context']):
        raise ValueError('Evaluation context exceeds checkpoint')
    first=MirrorLM(Config(**before['config'])).eval()
    second=MirrorLM(Config(**after['config'])).eval()
    load_v25_compatible(first,before['model'])
    load_v25_compatible(second,after['model'])
    before_start=perf_counter()
    a=measure_books(first,mem,records,args.context,args.batch)
    before_elapsed=perf_counter()-before_start
    after_start=perf_counter()
    b=measure_books(second,mem,records,args.context,args.batch)
    after_elapsed=perf_counter()-after_start
    trust_ablated=None
    if second.reciprocal is not None and second.reciprocal.context_trust is not None:
        with torch.no_grad():
            old_trust = second.reciprocal.context_trust.weight.detach().clone()
            second.reciprocal.context_trust.weight.zero_()
        trust_ablated=measure_books(second,mem,records,args.context,args.batch)
        with torch.no_grad():
            second.reciprocal.context_trust.weight.copy_(old_trust)
    ablated=None
    if second.reciprocal is not None:
        second.reciprocal.enabled=False
        ablated=measure_books(second,mem,records,args.context,args.batch)
    comparison={}
    for m in METRICS:
        higher=m.startswith(('acc_','top5_'))
        comparison[m]={'before':round(float(np.mean([r[m] for r in a])),6),
                       'after':round(float(np.mean([r[m] for r in b])),6),
                       **bootstrap_gain([r[m] for r in a],[r[m] for r in b],
                                        args.seed,args.bootstrap,higher)}
    feedback=None
    if ablated is not None:
        feedback={m:bootstrap_gain([r[m] for r in ablated],[r[m] for r in b],
                                    args.seed,args.bootstrap,m.startswith(('acc_','top5_')))
                  for m in METRICS}
    trust_feedback=None
    if trust_ablated is not None:
        trust_feedback={m:bootstrap_gain([r[m] for r in trust_ablated],
                            [r[m] for r in b],args.seed,args.bootstrap,
                            m.startswith(('acc_','top5_')))
                        for m in METRICS}
    eligible=passes_content_gate(comparison,feedback,args.split,trust_feedback)
    report={'before_step':before['step'],'after_step':after['step'],'split':args.split,
            'split_sha256':expected,'context':args.context,'seed':args.seed,
            'books':len(records),'windows_per_book':args.windows_per_book,
            'target_bytes_per_model':len(records)*args.windows_per_book*args.context,
            'throughput_bytes_per_second':{
                'parent':round(len(records)*args.windows_per_book*args.context/max(before_elapsed,1e-9),1),
                'candidate':round(len(records)*args.windows_per_book*args.context/max(after_elapsed,1e-9),1)},
            'bootstrap_unit':'whole book','bootstrap_replicates':args.bootstrap,
            'metric_direction':'positive gain favors trained candidate',
            'content_gate':'non-space CE book-bootstrap lower bound > 0; '
                           'non-space top1/top5 nondecreasing; enabled-memory '
                           'non-space CE lower bound > 0',
            'comparison':comparison,'reciprocal_feedback_versus_disabled':feedback,
            'content_trust_versus_neutral':trust_feedback,
            'development_gate_pass':bool(eligible),
            'promotion':'not automatic; independent long-context and sealed test gates required'}
    text=json.dumps(report,indent=2)+'\n'
    print(text,end='')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)


if __name__=='__main__':
    main()
