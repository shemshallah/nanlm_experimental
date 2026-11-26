"""Differentiable likelihood for a conserved finite lexical tilt and raw event.
The target can be an exact word, a capped prefix, or an empty nonletter event.
The donor and spelling law are fixed; gradients reach all producers of lifts.
"""
import torch
import numpy as np


def probabilities(lift, prior, member, base, surface, raw_literal, raw_folded, weight):
    """Return literal and folded event masses, with the same normalizer.
    prior/member: [B,K]; base/surface/raw_literal/raw_folded: [B].
    member selects candidates contained in the folded target event.
    """
    lift, prior, member, base, surface, raw_literal, raw_folded = (
        x.double() for x in (lift, prior, member, base, surface, raw_literal, raw_folded))
    delta = prior * torch.expm1(lift)
    den = 1 + delta.sum(-1)
    lexical = (base + (delta * member).sum(-1)) / den
    # Roundoff can only affect zero lexical mass, never the positive raw escape.
    assert bool((den > 0).all()) and bool((lexical >= -1e-14).all())
    lexical = lexical.clamp_min(0)
    literal = (1-weight) * raw_literal + weight * surface * lexical
    folded = (1-weight) * raw_folded + weight * lexical
    assert bool((literal > 0).all()) and bool((folded + 1e-14 >= literal).all())
    return literal, folded


def objective(lift, prior, member, base, surface, raw_literal, raw_folded, length, weight, spelling=1.):
    literal, folded = probabilities(lift, prior, member, base, surface, raw_literal, raw_folded, weight)
    # The two losses partition the same event; spelling=1 is EXACT raw NLL.
    semantic = -folded.log()
    conditional = folded.log() - literal.log()
    return (semantic + spelling * conditional).sum() / length.double().sum()


def factors(event, candidates, law, donor):
    """donor is a folded_event.Event whose weight can be any legal value.
    Derive exact factors, including cap-terminated and empty event mass.
    """
    import folded_event as FE
    assert donor.done(event) and not donor.done(event[:-1])
    raw, _ = donor.literal(event)
    if len(event) == 1 and not FE.IS_LETTER[event[0]]:
        folded_raw = float(donor.donor(donor.suffix(donor.root))[FE.STOP].sum())
        return np.zeros(len(candidates), np.float64), 0., 0., raw, folded_raw, len(event)
    capped = bool(FE.IS_LETTER[event[-1]])
    letters = event if capped else event[:-1]
    word = FE.folded(letters)
    if capped:
        assert len(letters) == donor.cap
        base = law.mass(word)
        members = [w.startswith(word) for w in candidates]
        folded_raw = sum(donor.states(word.encode()).values())
    else:
        base = law.complete(word)
        members = [w == word for w in candidates]
        folded_raw = sum(m * float(donor.donor(s)[FE.STOP].sum()) for s,m in donor.states(word.encode()).items())
    surface = 1.
    for i,c in enumerate(letters):
        surface *= donor.surface.letter(letters[:i], c | 32)[int(c < 97)]
    if not capped: surface *= donor.surface.delimiter(letters)[event[-1]]
    return np.asarray(members,np.float64),float(base),float(surface),float(raw),float(folded_raw),len(event)
