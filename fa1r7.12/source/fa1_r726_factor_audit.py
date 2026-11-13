#!/usr/bin/env python3
"""R7.26: read-only factor ablations of the single FA1 geometric Diamond.

No separate model, optimizer reset, weight update, TEST selection, or BPB claim.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path

import torch

from fa1_r723_train import forward, prepare
from fa1_r724_train import boot_books, GEOMETRY_GATES
from fa1_r724_unified import load_geometric
from fa1_r725_controls import unrelated_book_context
from r710_checkpoint_integrity import sha256

FACTORS = (
    'r723_triangle_address', 'face_splus_address', 'face_sminus_address',
    'edge_splus', 'edge_sminus', 'edge_connection', 'edge_odd_joint',
    'tetra_closure', 'tetra_volume',
)


@contextmanager
def _zero_output(module):
    handle = module.register_forward_hook(lambda _mod, _args, value: torch.zeros_like(value))
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _flat_features(module, columns):
    def change(_mod, inputs):
        values = inputs[0].clone()
        for col in columns:
            values[..., col] = 1. if col == 2 else 0.
        return (values,) + tuple(inputs[1:])
    handle = module.register_forward_pre_hook(change)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def mask(model, name):
    """No address tensor or optimizer is edited, and all masks unwind."""
    d = model.diamond
    g = d.geometry
    with ExitStack() as stack:
        if name == 'none':
            pass
        elif name == 'geometry_off':
            saved = [getattr(g, key).detach().clone() for key in GEOMETRY_GATES]
            try:
                with torch.no_grad():
                    for key in GEOMETRY_GATES:
                        getattr(g, key).zero_()
                yield
            finally:
                with torch.no_grad():
                    for key, value in zip(GEOMETRY_GATES, saved):
                        getattr(g, key).copy_(value)
            return
        elif name.startswith('gate_') and name[5:] in GEOMETRY_GATES:
            gate = getattr(g, name[5:])
            saved = gate.detach().clone()
            try:
                with torch.no_grad():
                    gate.zero_()
                yield
            finally:
                with torch.no_grad():
                    gate.copy_(saved)
            return
        elif name == 'r723_triangle_address':
            stack.enter_context(_zero_output(d.triple_code))
        elif name in ('face_splus_address', 'face_sminus_address',
                      'edge_splus', 'edge_sminus', 'edge_connection', 'edge_odd_joint'):
            mapping = {
                'face_splus_address': (g.face_plus,),
                'face_sminus_address': (g.face_minus,),
                'edge_splus': (g.edge_plus,),
                'edge_sminus': (g.edge_minus,),
                'edge_connection': (g.edge_phase,),
                'edge_odd_joint': (g.edge_minus, g.edge_phase),
            }
            for module in mapping[name]:
                stack.enter_context(_zero_output(module))
        elif name == 'tetra_closure':
            # Flat curvature has [c, sin(c), cos(c)]=[0,0,1]; leave volume intact.
            stack.enter_context(_flat_features(g.tetra_net, (0, 1, 2)))
        elif name == 'tetra_volume':
            stack.enter_context(_flat_features(g.tetra_net, (3, 4)))
        else:
            raise ValueError('unknown R7.26 ablation: ' + name)
        yield


@torch.no_grad()
def nll_with(model, data, *factors):
    with ExitStack() as stack:
        for factor in factors:
            stack.enter_context(mask(model, factor))
        nll, _ = forward(model, data)
        return nll.detach().cpu().clone()


@torch.no_grad()
def audit(model, data, seed=726):
    model.eval()
    if 'book_ids' not in data or torch.unique(data['book_ids']).numel() < 2:
        raise ValueError('R7.26 requires multiple book-labeled DEV histories')
    if model.diamond.gate.detach().item() <= 0:
        raise ValueError('R7.23 Diamond gate is closed: ablation would be meaningless')
    books = data['book_ids']
    geometry = model.diamond.geometry
    original = [getattr(geometry, key).detach().clone() for key in GEOMETRY_GATES]
    full = nll_with(model, data)
    inherited = nll_with(model, data, 'geometry_off')
    unrelated, donors = unrelated_book_context(data, seed)
    foreign_full = nll_with(model, unrelated)
    foreign_inherited = nll_with(model, unrelated, 'geometry_off')

    def grade(delta):
        return {'mean_nats_per_vocab_word': float(delta.mean()),
                'book_cluster_ci': boot_books(delta, books, seed)}

    penalties = {}
    for factor in FACTORS + tuple('gate_' + name for name in GEOMETRY_GATES):
        penalties[factor] = grade(nll_with(model, data, factor) - full)
    no_tri = nll_with(model, data, 'r723_triangle_address')
    no_tri_no_geo = nll_with(model, data, 'r723_triangle_address', 'geometry_off')
    geometry_gain = inherited - full
    gain_without_tri = no_tri_no_geo - no_tri
    if not torch.equal(nll_with(model, data), full):
        raise AssertionError('an R7.26 mask changed the original forward')
    if any(not torch.equal(getattr(geometry, key).detach(), value)
           for key, value in zip(GEOMETRY_GATES, original)):
        raise AssertionError('an R7.26 mask modified a geometry gate')
    return {
        'format': 'FA1_R726_READ_ONLY_FACTOR_AUDIT_V1',
        'promoted': False, 'trained': False, 'raw_byte_bpb_measured': False,
        'n_vocab_word_targets': full.numel(),
        'books': int(torch.unique(books).numel()),
        'foreign_donors': int(donors.numel()),
        'full_nll': float(full.mean()), 'geometry_off_nll': float(inherited.mean()),
        'geometry_gain': grade(geometry_gain),
        'correct_history_advantage': grade(foreign_full - full),
        'inherited_history_advantage': grade(foreign_inherited - inherited),
        'r723_address_effect_without_geometry': grade(no_tri_no_geo - inherited),
        'geometry_gain_without_r723_addresses': grade(gain_without_tri),
        'geometry_triangle_address_interaction': grade(geometry_gain - gain_without_tri),
        'removal_penalties': penalties,
        'all_masks_restored': True,
        'geometry_gates': {name: float(value) for name, value in zip(GEOMETRY_GATES, original)},
    }


def run(args):
    if args.split != 'dev':
        raise ValueError('Never select R7.26 diagnostics on sealed TEST')
    torch.set_num_threads(args.threads)
    model, optimizers, base, comp, _ = load_geometric(args.candidate, args.root, threads=args.threads)
    if len(optimizers) != 5:
        raise ValueError('Five inherited optimizer histories required')
    data = prepare(args, model, base, comp, 'dev', args.targets, args.seed)
    result = audit(model, data, args.seed)
    result.update(checkpoint_sha256=sha256(args.candidate), split='dev', seed=args.seed,
                  optimizer_count=len(optimizers))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--corpus-dir', required=True)
    p.add_argument('--split', choices=['dev'], default='dev')
    p.add_argument('--targets', type=int, default=256)
    p.add_argument('--seed', type=int, default=726)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--output', default='')
    run(p.parse_args())
