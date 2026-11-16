"""Original strict C full-256 likelihood oracle, wrapper-aware v103 scoring.

Unlike BPB, this diagnoses numeric C↔RSI mismatch over *all* byte values.
An honest nonzero delta does not invalidate old v103-to-wrapper identity.
"""
import hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
from .legacy_v103_reference import (
    CONTEXT, PARITY_CONTEXT, C_10M_DEV_FULL256_SHA256,
)

def compare_c256_wrapped(model, independent_books, reference_path,
                      *, auxiliary=None, local=None, probe_count=128):
    """Exact C-vs-GPU 256-way distribution gate on aligned DEV histories.

    Unlike historical `Rec` values, this catches different distributions
    even when the correct target has identical NLL. It does not update the
    model, its online state, or the independent-book reference.
    """
    books, per, cols = independent_books.shape
    if cols not in (CONTEXT + 1, PARITY_CONTEXT + 1):
        raise ValueError("Invalid CM distribution DEV history shape")
    raw = Path(reference_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != C_10M_DEV_FULL256_SHA256:
        raise ValueError("Original 10M C full256 numerical oracle digest changed")
    if len(raw) != books * per * 256 * 4:
        raise ValueError("Original C full-distribution records are misaligned")
    original = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(-1, 256)
    if not torch.isfinite(original).all() or (original <= 0).any():
        raise ValueError("Invalid NaNCM C frozen 256-way probability mass")
    sample = torch.linspace(0, books * per - 1,
                            min(probe_count, books * per)).long()
    windows = independent_books.reshape(-1, cols)
    aux = auxiliary.reshape(-1, 6) if auxiliary is not None else None
    local_flat = local.reshape(-1, 4) if local is not None else None
    original = original[sample].to(windows.device)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sample), 8):
            indices = sample[start:start + 8].to(windows.device)
            part = windows[indices].long()
            batch = part[:, :-1].repeat_interleave(256, 0)
            candidates = torch.arange(256, device=batch.device).repeat(len(part))
            features = (aux[indices].long().repeat_interleave(256, 0)
                        if aux is not None else None)
            book_matches = (local_flat[indices].long().repeat_interleave(256, 0)
                            if local_flat is not None else None)
            logits, target_bits = model(batch, candidates, features,
                                        book_matches)
            logp = -F.binary_cross_entropy_with_logits(
                logits, target_bits, reduction="none").sum(-1)
            predictions.append(logp.reshape(-1, 256).exp())
    gpu = torch.cat(predictions)
    absolute = (gpu - original).abs()
    # Numerical comparison is separate from quality on each target. C's
    # float32 and Torch's reduction order need tight-but-not-bitwise tolerance.
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    c_probs = original / original.sum(-1, keepdim=True)
    gpu_probs = gpu / gpu.sum(-1, keepdim=True)
    mean_kl = float((c_probs * (c_probs.log() - gpu_probs.log())).sum(-1).mean())
    return {"protocol": "C-v103-full256-matched-frozen-DEV",
            "c256_sha256": hashlib.sha256(raw).hexdigest(),
            "probe_histories": int(len(sample)), "candidates_per_history": 256,
            "max_abs_probability_delta": max_abs,
            "mean_abs_probability_delta": mean_abs,
            "mean_kl_c_to_gpu_nats": mean_kl,
            "exact_distribution_parity": bool(max_abs <= 1e-6 and mean_kl <= 1e-7)}


