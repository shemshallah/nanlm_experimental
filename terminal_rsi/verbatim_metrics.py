"""Original v103 exact byte-scorer, adapted ONLY to pass the six causal features.

The archived notebook's isinstance(V103ParityGPUFullSystem) check would
incorrectly OMIT aux+local for the new composition wrapper, silently
invalidating BPB/top-5. All byte probability formulas are preserved verbatim.
"""
import math
import torch
from torch.nn import functional as F
from .legacy_v103_reference import CONTEXT, PARITY_CONTEXT

def byte_metrics_wrapped(model, windows, n=256, *, topk_n=64, auxiliary=None,
                 local=None):
    """Exact bit NLL on sampled/all windows and exact 256-way byte top-k.

    Does not compute top-k from eight independent bit guesses. For each
    history, enumerates all 256 causal bit paths and checks their probability
    mass, then measures top-1 and top-5 on stratified histories.
    """
    model.eval()
    with torch.no_grad():
        if windows.ndim == 3:
            books, per, columns = windows.shape
            if columns not in (CONTEXT + 1, PARITY_CONTEXT + 1):
                raise ValueError("Invalid held-out book window shape")
            windows = windows.reshape(books * per, columns)
            if auxiliary is not None:
                auxiliary = auxiliary.reshape(books * per, 6)
            if local is not None:
                local = local.reshape(books * per, 4)
        # Probes may be assembled on CPU while windows live on CUDA. Indexing a
        # CPU tensor with a CUDA index tensor fails; align auxiliary tensors
        # with the window device before applying the shared index.
        if auxiliary is not None:
            auxiliary = auxiliary.to(windows.device)
        if local is not None:
            local = local.to(windows.device)
        if n is None:
            n = windows.size(0)
        ix = torch.linspace(0, windows.size(0) - 1,
                            min(n, windows.size(0)),
                            device=windows.device).long()
        batch = windows[ix].long()
        probe_aux = auxiliary[ix].long() if auxiliary is not None else None
        probe_local = local[ix].long() if local is not None else None
        total_nll = 0.0
        bit_hits = 0
        for offset in range(0, len(batch), 512):
            part = batch[offset:offset + 512]
            part_aux = (probe_aux[offset:offset + 512]
                        if probe_aux is not None else None)
            part_local = (probe_local[offset:offset + 512]
                          if probe_local is not None else None)
            logits, bits = model(part[:, :-1], part[:, -1], part_aux,
                                 part_local)
            total_nll += float(F.binary_cross_entropy_with_logits(
                logits, bits, reduction="sum"))
            bit_hits += int(((logits > 0) == (bits > 0.5)).sum())
        # Enumerate candidate bytes to get exact top-1 and top-5.
        # This is a small, controlled diagnostic, not train/eval leakage.
        top_ix = torch.linspace(0, batch.shape[0] - 1,
                                min(topk_n, batch.shape[0]),
                                device=batch.device).long()
        probe = batch[top_ix]
        top_aux = probe_aux[top_ix] if probe_aux is not None else None
        top_local = probe_local[top_ix] if probe_local is not None else None
        top1_hits, top5_hits = 0, 0
        largest_mass_error = 0.0
        for offset in range(0, len(probe), 16):
            part = probe[offset:offset + 16]
            vals = torch.arange(256, device=part.device).repeat(part.size(0))
            history = part[:, :-1].repeat_interleave(256, dim=0)
            expanded_aux = (top_aux[offset:offset + 16].repeat_interleave(256, dim=0)
                            if top_aux is not None else None)
            expanded_local = (top_local[offset:offset + 16].repeat_interleave(256, dim=0)
                              if top_local is not None else None)
            all_logits, all_bits = model(history, vals, expanded_aux, expanded_local)
            logp = -F.binary_cross_entropy_with_logits(
                all_logits, all_bits, reduction="none").sum(-1)
            logp = logp.reshape(part.shape[0], 256)
            mass_error = float(torch.logsumexp(logp, dim=-1).abs().max())
            largest_mass_error = max(largest_mass_error, mass_error)
            top5 = logp.topk(5, dim=-1).indices
            # The v103 extension consumes 32 observed bytes, whereas the
            # inherited v1 model consumes 8.  The label is always the final
            # column, regardless of history width.  Indexing CONTEXT here
            # silently scored a history byte as the target in v103 mode.
            targets = part[:, -1]
            top1_hits += int((top5[:, 0] == targets).sum())
            top5_hits += int((top5 == targets[:, None]).any(1).sum())
        if largest_mass_error > 2e-3:
            raise AssertionError("256-byte tree probability mass is not normalized")
        return {"bpb": float(total_nll / (len(batch) * math.log(2))),
                "bit_accuracy": bit_hits / (len(batch) * 8),
                "top1_byte": top1_hits / len(probe),
                "top5_byte": top5_hits / len(probe),
                "maximum_log_probability_mass_error": largest_mass_error,
                "evaluated_bytes": int(batch.size(0)),
                "topk_evaluated_bytes": int(probe.size(0))}


