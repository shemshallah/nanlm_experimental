"""Byte-driven generation through the causal hierarchy.

The model still emits exactly one byte per step.  The subword, word and
sentence states are latent; they influence the byte distribution through the
gated top-down and lateral paths and are reported alongside the generated text
so the synthesis can be inspected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .hierarchical_bus import bit_logits_to_byte_log_probs
from .unified import PRESETS
from .verbatim_v103 import VerbatimV103RSI


def _default_aux(model: VerbatimV103RSI, previous: torch.Tensor) -> torch.Tensor:
    original = model.original
    if hasattr(original, "_default_aux"):
        return original._default_aux(previous)
    last = previous.long()[:, -8:]
    bpe = ((last[:, -1] * 257 + last[:, -2]) ^ (last[:, -3] * 31)) & 16383
    word = ((last[:, -1] * 131 + last[:, -4] * 257) ^ last[:, -2]) & 16383
    zeros = torch.zeros_like(bpe)
    return torch.stack((bpe, word, zeros, zeros, zeros, zeros), dim=1)


@torch.no_grad()
def generate_bytes(
    model: VerbatimV103RSI,
    prompt: bytes,
    steps: int,
    *,
    temperature: float = 0.9,
    top_k: int = 0,
    seed: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Generate ``steps`` bytes and expose the latent hierarchical state.

    The target-conditioned v103 bit tree is evaluated for all 256 candidate
    bytes in one batch.  Candidate ``i`` contributes the product of its eight
    conditional bit probabilities, so the sampled distribution is exactly the
    model's byte factorization rather than an approximation from the 100M
    branch alone.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if seed is not None:
        torch.manual_seed(seed)
    device = device or next(model.parameters()).device
    model.to(device).eval()
    prompt = bytes(prompt)
    history = list(prompt)
    if len(history) < 32:
        history = [0] * (32 - len(history)) + history
    generated: list[int] = []
    records: list[dict] = []

    for _ in range(steps):
        context = torch.tensor(
            history[-256:], dtype=torch.long, device=device)[None]
        candidates = torch.arange(256, dtype=torch.long, device=device)
        previous = context.expand(256, -1)
        auxiliary = _default_aux(model, previous)
        old_logits, _ = model.original(previous[:, -32:], candidates, auxiliary)

        if device.type == "cuda" and model.cfg.name == "blackwell":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("Blackwell hierarchy generation requires BF16")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                upgrade_out = model.upgrades(context)
        else:
            upgrade_out = model.upgrades(context)
        current = upgrade_out["logits"][0, :, -1, :]
        proposed = model._byte_to_conditional_bit_logits(current, candidates)

        trust = 1.0 + 0.25 * torch.tanh(
            model.original.feedback_state.detach().float().mean())
        gate = torch.tanh(model.residual_gate) * trust.to(old_logits.dtype)
        final_bits = old_logits + gate * (proposed - old_logits)

        # Row i is the eight-bit factorization for candidate byte i.  The
        # probability of candidate i is its diagonal entry.
        byte_logp = bit_logits_to_byte_log_probs(final_bits)
        candidate_logp = torch.nan_to_num(
            byte_logp.diagonal(), nan=-1e9, posinf=0.0, neginf=-1e9)
        scaled = candidate_logp / float(temperature)
        if top_k and 0 < top_k < 256:
            threshold = torch.topk(scaled, top_k).values[-1]
            scaled = scaled.masked_fill(scaled < threshold, -float("inf"))
        probabilities = scaled.softmax(dim=0)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            probabilities = torch.full_like(probabilities, 1.0 / 256.0)
        next_byte = int(torch.multinomial(probabilities, 1).item())
        generated.append(next_byte)
        history.append(next_byte)

        boundary = torch.sigmoid(
            upgrade_out["boundary_logits"][0, -1].float()).tolist()
        records.append({
            "byte": next_byte,
            "boundary_subword": float(boundary[0]),
            "boundary_word": float(boundary[1]),
            "boundary_sentence": float(boundary[2]),
            "probability": float(probabilities[next_byte]),
        })

    text = bytes(generated).decode("utf-8", errors="replace")
    return {
        "prompt": prompt,
        "bytes": bytes(generated),
        "text": text,
        "records": records,
        "hierarchy_gates": {
            key: value.tolist()
            for key, value in upgrade_out["hierarchy_gates"].items()
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(PRESETS), default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=103)
    args = parser.parse_args(argv)

    model = VerbatimV103RSI(args.mode)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=False)
    result = generate_bytes(
        model, args.prompt.encode("utf-8"), args.steps,
        temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    print(result["text"], end="")
    print()
    gates = result["hierarchy_gates"]
    print("hierarchy gates:", gates)
    if result["records"]:
        sentence_hits = sum(
            1 for record in result["records"]
            if record["boundary_sentence"] > 0.5)
        word_hits = sum(
            1 for record in result["records"]
            if record["boundary_word"] > 0.5)
        print(f"latent boundaries: word>0.5={word_hits} "
              f"sentence>0.5={sentence_hits} of {len(result['records'])} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
