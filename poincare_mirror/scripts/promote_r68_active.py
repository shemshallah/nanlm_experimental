#!/usr/bin/env python3
"""Explicitly promote FA2 R68 as the ONE active model, recording failed gates.

This does NOT claim all pre-registered gates passed; it requires a conscious
--user-override and leaves the candidate and pre-R68 archive untouched.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poincare_frontier import file_hash


def validate_r68(checkpoint, dev_reports):
    expected = 'c377659754cb0d7e2a4583c04d816571ba60547f4e0939124613e7bbb6dd1580'
    if file_hash(checkpoint) != expected:
        raise ValueError('Unverified R68 candidate; refusing promotion')
    summary = {}
    for warmup in (0,128,512):
        report = json.loads(Path(dev_reports[warmup]).read_text())
        if (report.get('parent_sha256') != expected or
            report.get('warmup_positions_per_book') != warmup or
            report.get('dev_books_scored') != 12 or
            report.get('test_opened') is not False):
            raise ValueError('Incompatible or unsealed R68 DEV report')
        summary[str(warmup)] = {
            'strict_dev_gate_pass': report['dev_screen_pass'],
            'neutral_next_bpb': report['summary']['neutral']['bits_per_byte_or_eos_target'],
            'active_next_bpb': report['summary']['exact']['bits_per_byte_or_eos_target'],
            'neutral_top1': report['summary']['neutral']['acc_1'],
            'active_top1': report['summary']['exact']['acc_1'],
            'neutral_top5': report['summary']['neutral']['top5_1'],
            'active_top5': report['summary']['exact']['top5_1']}
    if summary['128']['strict_dev_gate_pass'] is not True or any(
            summary[x]['strict_dev_gate_pass'] is not False for x in ('0','512')):
        raise ValueError('Unexpected R68 validation profile; re-review before promotion')
    return expected, summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dev-0', type=Path, required=True)
    p.add_argument('--dev-128', type=Path, required=True)
    p.add_argument('--dev-512', type=Path, required=True)
    p.add_argument('--active-output', type=Path, required=True)
    p.add_argument('--manifest-output', type=Path, required=True)
    p.add_argument('--user-override', action='store_true',
                   help='Acknowledge 0/512 warm-up failures; override prior gate policy')
    args=p.parse_args()
    if not args.user_override:
        p.error('R68 fails some strict gates; explicit --user-override required')
    if args.active_output.exists():
        p.error('Active target already exists; refusing overwrite')
    sha,summary=validate_r68(args.checkpoint,
                             {0:args.dev_0,128:args.dev_128,512:args.dev_512})
    manifest={
        'architecture':'FA2_UNIFIED','active_round':'R68', 'active_step':4209,
        'checkpoint_filename':args.active_output.name,
        'checkpoint_sha256':sha,'promotion':'explicit_user_override',
        'strict_three_depth_gate_pass':False,'test_opened':False,
        'source_lineage':'R58 -> R66 -> R67 -> R68',
        'single_model':True,'single_optimizer':True,
        'inherited_adam_states':196,'promoted_dev_metrics':summary,
        'limitations':['warmup0 and warmup512 strict gates fail',
                       'small warmup0 and warmup512 top1 regressions',
                       '+2/+3 heads unaffected by the joint-memory specialist']}
    args.active_output.parent.mkdir(parents=True,exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.active_output.parent,
                                     prefix='fa2_active_',delete=False) as fh:
        staged=Path(fh.name)
    try:
        shutil.copyfile(args.checkpoint,staged)
        if file_hash(staged)!=sha:
            raise ValueError('Staged FA2 active checksum mismatch')
        os.replace(staged,args.active_output)
    finally:
        staged.unlink(missing_ok=True)
    args.manifest_output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'promoted':True,'active_checkpoint':str(args.active_output),
                      'sha256':sha,'strict_gate_pass':False,
                      'reason':'explicit_user_override','manifest':str(args.manifest_output)},indent=2))


if __name__=='__main__':
    main()
