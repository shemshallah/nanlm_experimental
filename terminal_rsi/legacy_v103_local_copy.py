"""Read-only, target-causal local-copy audit for the same Terminal-RSI GPU model.

The frozen original C reference is used exclusively for *scoring*, never as
an input, a feature, or a source of a candidate byte.  No weights or optimizer
buffers are modified.  The exact-match candidates come from the train-only
v103 index and from bytes already seen inside the current independent book.

This is a diagnostic expert, not a claim of faithful C numerical emulation.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct

import numpy as np


LOCAL_MATCH_WIDTHS = (5, 12)
DEV_SHA256 = "5826929ae3356dda057c1619559afb7e04f534e185d59d45274f401232193465"
C_DEV_SHA256 = "adf21a4490aeec7a1e7f4e204ffffb3a0ede74fca9f50a0fa0ecc73e80167c21"


def read_books(path, *, expected_sha256=DEV_SHA256):
    """Read original books (including their 32-byte observed warm-up)."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Independent DEV book identity changed")
    if len(raw) < 8:
        raise ValueError("Truncated independent-book window header")
    count, skip = struct.unpack_from("<II", raw)
    if not 1 <= count <= 10_000 or skip < 12:
        raise ValueError("Invalid independent-book header")
    offset, books = 8, []
    for _ in range(count):
        if offset + 4 > len(raw):
            raise ValueError("Missing independent-book length")
        length = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if length <= skip or offset + length > len(raw):
            raise ValueError("Invalid independent-book length")
        books.append(raw[offset:offset + length])
        offset += length
    if offset != len(raw) or len({len(book) - skip for book in books}) != 1:
        raise ValueError("Independent-book trailing bytes or unequal scored positions")
    return books, skip, digest


def causal_local_matches(book: bytes, *, widths=LOCAL_MATCH_WIDTHS):
    """Candidate bytes come strictly from earlier positions of *this* book.

    Output columns: candidate5, has5, candidate12, has12.  The candidate at
    target position `i` is book[j], with j < i and identical prior `width`
    bytes.  `book[i]` is read only after computing and saving predictions at i.
    """
    out = np.zeros((len(book), len(widths) * 2), dtype=np.uint16)
    seen = [{} for _ in widths]
    for i in range(len(book)):
        for channel, width in enumerate(widths):
            if i < width:
                continue
            key = book[i - width:i]
            previous = seen[channel].get(key)
            if previous is not None:
                if previous >= i:
                    raise AssertionError("Causal match target is not in the past")
                out[i, 2 * channel] = book[previous]
                out[i, 2 * channel + 1] = width
            # This target is added only AFTER all lookups at its position.
            seen[channel][key] = i
    return out


def book_local_features(path, *, expected_sha256=DEV_SHA256):
    books, skip, digest = read_books(path, expected_sha256=expected_sha256)
    candidates = np.stack([causal_local_matches(book)[skip:] for book in books])
    return candidates, digest


def select_copy_expert(auxiliary, local, *, use_train=True, use_local=True,
                       return_local=False):
    """Fixed, target-free priority: 12-byte exact match before 5-byte match.

    At equal width use local history first, because the C evaluation itself
    allows the currently observed document to extend its causal history.
    """
    if auxiliary.shape != local.shape or auxiliary.shape[-1] != 4:
        raise ValueError("Expected matching [books, positions, 4] match channels")
    shape = auxiliary.shape[:-1]
    candidate = np.zeros(shape, dtype=np.uint8)
    width = np.zeros(shape, dtype=np.uint8)
    from_local = np.zeros(shape, dtype=np.bool_)
    for source, enabled, column, size in (
        (auxiliary, use_train, 0, 5),
        (local, use_local, 0, 5),
        (auxiliary, use_train, 2, 12),
        (local, use_local, 2, 12),
    ):
        if enabled:
            valid = source[..., column + 1] == size
            candidate[valid] = source[..., column][valid].astype(np.uint8)
            width[valid] = size
            from_local[valid] = source is local
    return (candidate, width, from_local) if return_local else (candidate, width)


def copy_gate(widths, is_local, *, train_gate5=0.06, train_gate12=0.12,
              local_gate5=0.10, local_gate12=0.40):
    """Source-specific trust: sparse train matches are less reliable."""
    import torch
    gates = torch.zeros_like(widths, dtype=torch.float32)
    for width, local, gate in ((5, False, train_gate5),
                               (12, False, train_gate12),
                               (5, True, local_gate5),
                               (12, True, local_gate12)):
        if not math.isfinite(gate) or not 0 <= gate < 1:
            raise ValueError("Copy trust must be a fixed probability in [0,1)")
        gates = torch.where((widths == width) & (is_local == local),
                            torch.full_like(gates, gate), gates)
    return gates


def mix_log_target_probability(logp, targets, candidates, widths, is_local,
                               *, train_gate5=0.06, train_gate12=0.12,
                               local_gate5=0.10, local_gate12=0.40):
    """Normalized 256-byte mixture, evaluated on the true target only.

    Candidate/gate selection happens *before* seeing the target; the equality
    below is solely used to score the resulting proper distribution.
    """
    import torch
    gates = copy_gate(widths, is_local, train_gate5=train_gate5,
                      train_gate12=train_gate12, local_gate5=local_gate5,
                      local_gate12=local_gate12).to(logp.dtype)
    match = (candidates == targets)
    ordinary = torch.log1p(-gates) + logp
    injected = torch.where(match & (gates > 0), gates.log(),
                           torch.full_like(logp, float('-inf')))
    return torch.logaddexp(ordinary, injected)


def audit_gpu_copy_full(model, windows, auxiliary, local, c_reference,
                        *, topk_n=512, train_gate5=0.06, train_gate12=0.12,
                        local_gate5=0.10, local_gate12=0.40):
    """Evaluate GPU base, train-only copy, causal local+train and shuffled.

    No C probability is ever fed into the GPU.  Parameters, buffers, online
    memory, optimizer and RNG are not modified by this inference-only audit.
    """
    import torch
    from torch.nn import functional as F

    if windows.ndim != 3 or windows.shape[-1] != 33:
        raise ValueError("Expected 32-byte book-isolated histories")
    books, per, _ = windows.shape
    if auxiliary.shape != (books, per, 6) or local.shape != (books, per, 4):
        raise ValueError("Original causal GPU features do not align")
    records = Path(c_reference).read_bytes()
    if hashlib.sha256(records).hexdigest() != C_DEV_SHA256:
        raise ValueError("Frozen C DEV scoring records changed")
    if len(records) != books * per * 8:
        raise ValueError("C frozen DEV score positions do not align")
    c_records = np.frombuffer(records, dtype=[('nll', '<f4'), ('top1', 'u1'),
                                               ('top5', 'u1'), ('pad', 'V2')])
    c_bpb = float(c_records['nll'].astype(np.float64).mean() / math.log(2))
    all_aux = auxiliary.cpu().numpy().astype(np.uint16, copy=False)
    train = all_aux[..., 2:]
    trials = {
        'gpu_only': (np.zeros((books, per), dtype=np.uint8),
                     np.zeros((books, per), dtype=np.uint8),
                     np.zeros((books, per), dtype=np.bool_)),
        'train_copy_only': select_copy_expert(train, local, use_local=False,
                                              return_local=True),
        'train_plus_causal_book': select_copy_expert(train, local, return_local=True),
        'shuffled_unrelated_book': select_copy_expert(
            train, np.roll(local, shift=1, axis=0), return_local=True),
    }
    flat = windows.reshape(-1, 33)
    flat_aux = auxiliary.reshape(-1, 6)
    labels = flat[:, -1].long()
    device = windows.device
    stored = {}
    for name, (candidate, width, is_local) in trials.items():
        stored[name] = (torch.as_tensor(candidate.reshape(-1), device=device,
                                       dtype=torch.long),
                        torch.as_tensor(width.reshape(-1), device=device,
                                        dtype=torch.long),
                        torch.as_tensor(is_local.reshape(-1), device=device,
                                        dtype=torch.bool))
    nll = {name: [] for name in trials}
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(flat), 512):
                end = min(start + 512, len(flat))
                part = flat[start:end].long()
                logits, bits = model(part[:, :-1], part[:, -1],
                                     flat_aux[start:end].long())
                base_logp = -F.binary_cross_entropy_with_logits(
                    logits, bits, reduction='none').sum(-1)
                for name, (candidate, width, is_local) in stored.items():
                    result = mix_log_target_probability(
                        base_logp, part[:, -1], candidate[start:end],
                        width[start:end], is_local[start:end],
                        train_gate5=train_gate5, train_gate12=train_gate12,
                        local_gate5=local_gate5, local_gate12=local_gate12)
                    nll[name].append(-result.cpu().double())

            size = min(topk_n, len(flat))
            sample = torch.linspace(0, len(flat) - 1, size, device=device).long()
            c_top1 = float(c_records['top1'][sample.cpu().numpy()].mean())
            c_top5 = float(c_records['top5'][sample.cpu().numpy()].mean())
            top1 = {name: 0 for name in trials}
            top5 = {name: 0 for name in trials}
            for start in range(0, size, 16):
                indices = sample[start:start + 16]
                part = flat[indices].long()
                true_byte = part[:, -1]
                guesses = torch.arange(256, device=device).repeat(len(part))
                expanded = part[:, :-1].repeat_interleave(256, 0)
                expanded_aux = flat_aux[indices].long().repeat_interleave(256, 0)
                logits, bits = model(expanded, guesses, expanded_aux)
                logp = -F.binary_cross_entropy_with_logits(
                    logits, bits, reduction='none').sum(-1).reshape(len(part), 256)
                mass_error = float(torch.logsumexp(logp, dim=-1).abs().max())
                if mass_error > 2e-3:
                    raise AssertionError('GPU byte distribution is not normalized')
                vals = torch.arange(256, device=device)[None, :]
                for name, (candidate, width, is_local) in stored.items():
                    # Exact normalized probability mixture for all 256 bytes;
                    # never renormalize selectively around the known label.
                    gates = copy_gate(
                        width[indices], is_local[indices],
                        train_gate5=train_gate5, train_gate12=train_gate12,
                        local_gate5=local_gate5, local_gate12=local_gate12)
                    probs = (1.0 - gates[:, None]) * logp.exp() + (
                        gates[:, None] * (vals == candidate[indices, None]))
                    top = probs.topk(5, dim=-1).indices
                    top1[name] += int((top[:, 0] == true_byte).sum())
                    top5[name] += int((top == true_byte[:, None]).any(dim=-1).sum())
    finally:
        model.train(previous_mode)

    metrics = {}
    for name, chunks in nll.items():
        bits = torch.cat(chunks).reshape(books, per) / math.log(2)
        metrics[name] = {
            'bpb': float(bits.mean()),
            'gpu_minus_c_bpb': float(bits.mean() - c_bpb),
            'book15_bpb': float(bits[14].mean()) if books >= 15 else None,
            'per_book_bpb': bits.mean(-1).tolist(),
            'top1': top1[name] / size,
            'top5': top5[name] / size,
        }
    paired = metrics['train_plus_causal_book']
    unrelated = metrics['shuffled_unrelated_book']
    base = metrics['gpu_only']
    # User's hard gate includes exact matched DEV, honest unrelated-history
    # control, and aligned top1/top5 from the same 512 sampled histories.
    gate = (paired['bpb'] < c_bpb and paired['top1'] >= c_top1 and
            paired['top5'] >= c_top5 and
            paired['bpb'] < unrelated['bpb'])
    # A postprocessing diagnostic cannot authorize training of a DIFFERENT
    # forward path.  The saved neural model is what main() actually trains;
    # until copy is integrated into that model and checkpoint, it must pass
    # its own C comparison before any multi-minute training is approved.
    training_gate = (base['bpb'] < c_bpb and base['top1'] >= c_top1 and
                     base['top5'] >= c_top5)
    return {
        'protocol': 'checkpoint-only-GPU-vs-frozen-C-book-disjoint-DEV',
        'scored_bytes': books * per, 'c_frozen_bpb': c_bpb,
        'c_aligned_top1': c_top1, 'c_aligned_top5': c_top5,
        'topk_bytes': size,
        'train_gate5': train_gate5, 'train_gate12': train_gate12,
        'local_gate5': local_gate5, 'local_gate12': local_gate12,
        'experiments': metrics,
        'gpu_beats_c_quality_gate': bool(gate),
        'gpu_training_model_c_gate': bool(training_gate),
        'numeric_parity_proven': False,
        'weights_or_optimizer_modified': False,
    }
