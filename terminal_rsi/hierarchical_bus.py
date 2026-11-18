"""Causal hierarchical bus for the unified Terminal-RSI upgrade branch.

The bus is deliberately a *DAG per timestep*, not a circular chain:

    bottom-up:  byte -> subword -> word -> sentence
    top-down:   sentence -> word -> subword -> byte
    lateral:    subword -> byte logits, word -> byte logits

Every edge has its own zero-initialized gate.  At initialization the bus is an
exact identity on the byte state and contributes exactly zero to the byte
logits, so the inherited v103 path and the v4 gate-0 parity contract are both
preserved.

Generation stays byte-driven: the model samples a byte, appends it to the
causal context, and re-runs the bus.  Word, sentence and paragraph states are
latent synthesized states; they are never sampled as tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .norm import RMSNorm
from .pair_memory import PairConceptMemory, pair_parameter_count
from .unit_heads import (
    SENTENCE_VOCAB_SIZE, TOKEN_VOCAB_SIZE, WORD_VOCAB_SIZE,
    UnitHeads, unit_parameter_count,
)


def hierarchy_bus_width(width: int) -> int:
    """Bottleneck width of the hierarchy bus (cheap enough for 8192-byte batches)."""
    return max(16, width // 4)


def hierarchy_parameter_count(width: int) -> int:
    """Exact parameter count of :class:`CausalHierarchy`."""
    bw = hierarchy_bus_width(width)
    return (
        width * bw + bw * width                 # byte <-> bus projection
        + 3 * bw * bw                           # bottom-up level transforms
        + 3 * bw * bw                           # top-down level transforms
        + 9                                     # nine zero gates
        + 3 * bw * 256                          # lateral byte-logit priors
        + 3 * width + 3                         # boundary heads + biases
        + 3 * 256 + 1                           # boundary -> byte prior
        + width + 3 * bw                        # input + three level RMSNorm
        + pair_parameter_count(bw)              # 65,536 paired concepts
        + unit_parameter_count(bw, width)       # real token/word/sentence heads
        + unit_decoder_parameter_count(width)  # unit+hierarchy -> byte MLP
    )


def unit_decoder_parameter_count(width: int, hidden: int = 256) -> int:
    bw = hierarchy_bus_width(width)
    input_dim = 2 * (TOKEN_VOCAB_SIZE + WORD_VOCAB_SIZE) + SENTENCE_VOCAB_SIZE + width + 3 * bw
    return (input_dim * hidden + hidden + 2 * hidden
            + hidden * 256 + 256 + 1)


def causal_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Causal mean over the last ``window`` positions, vectorized with cumsum.

    At position ``t`` the result only uses ``x[..., max(0, t-window+1):t+1]``.
    No right-hand or future position can enter.
    """
    if window <= 1:
        return x
    b, t, d = x.shape
    if t == 0:
        return x
    cumulative = torch.cumsum(x, dim=1)
    padded = F.pad(cumulative, (0, 0, 1, 0))
    index = torch.arange(t, device=x.device)
    start = (index - window).clamp_min(0)
    sums = padded[:, index + 1] - padded[:, start]
    counts = (index - start + 1).to(x.dtype)[None, :, None]
    return sums / counts


def boundary_targets(byte_ids: torch.Tensor) -> torch.Tensor:
    """Pseudo-boundary labels from observed bytes only.

    Returns ``[batch, time, 3]`` for (subword, word, sentence) boundaries.
    These labels are derived from the *current* observed byte, never a future
    byte, so the auxiliary supervision is causal.
    """
    if byte_ids.ndim != 2:
        raise ValueError("boundary_targets expects [batch,time] byte ids")
    b = byte_ids.long()
    whitespace = (b == 32) | (b == 9) | (b == 10) | (b == 13)
    ascii_punct = (
        ((b >= 33) & (b <= 47))
        | ((b >= 58) & (b <= 64))
        | ((b >= 91) & (b <= 96))
        | ((b >= 123) & (b <= 126))
    )
    word_end = whitespace | ascii_punct
    sentence_end = (b == 46) | (b == 33) | (b == 63)  # . ! ?
    # A subword boundary is a word boundary plus a weak signal at digits and
    # case changes; the head can learn to sharpen this from data.
    previous = F.pad(b[:, :-1], (1, 0), value=0)
    case_or_digit = (
        ((b >= 48) & (b <= 57))
        | ((previous >= 97) & (previous <= 122) & (b >= 65) & (b <= 90))
        | ((previous >= 65) & (previous <= 90) & (b >= 97) & (b <= 122))
    )
    subword_end = word_end | case_or_digit
    return torch.stack(
        (subword_end, word_end, sentence_end), dim=-1).to(torch.float32)


@dataclass
class HierarchyState:
    """Outputs of one causal hierarchy sweep."""

    state: torch.Tensor
    byte_prior: torch.Tensor
    subword: torch.Tensor
    word: torch.Tensor
    sentence: torch.Tensor
    boundary_logits: torch.Tensor
    boundary_probs: torch.Tensor
    unit_logits: dict[str, torch.Tensor]
    pair_entropy: torch.Tensor
    pair_gate: torch.Tensor
    unit_byte_raw: torch.Tensor
    unit_byte_logits: torch.Tensor
    gates: dict[str, torch.Tensor]


class CausalHierarchy(nn.Module):
    """Three-level causal hierarchy with zero-gated feedback edges.

    ``windows=(4, 16, 64)`` are subword, word and sentence horizons in bytes.
    The bus runs at ``width // 4`` to keep the activation footprint small; the
    full-width byte state is restored through a learned up-projection.
    """

    def __init__(self, width: int, windows: tuple[int, int, int] = (4, 16, 64),
                 *, boundary_pooling: bool = True, enable_pair: bool = True,
                 enable_units: bool = True,
                 ablation: dict[str, bool] | None = None):
        super().__init__()
        bw = hierarchy_bus_width(width)
        self.width = width
        self.bus_width = bw
        self.windows = windows
        self.boundary_pooling = boundary_pooling
        self.ablation = dict(ablation or {})
        self.input_norm = RMSNorm(width)
        self.down = nn.Linear(width, bw, bias=False)
        self.up = nn.Linear(bw, width, bias=False)
        self.level_norms = nn.ModuleList([RMSNorm(bw) for _ in windows])
        self.level_up = nn.ModuleList(
            [nn.Linear(bw, bw, bias=False) for _ in windows])
        self.top_down = nn.ModuleList(
            [nn.Linear(bw, bw, bias=False) for _ in windows])
        self.bottom_up_gates = nn.Parameter(torch.zeros(len(windows)))
        self.top_down_gates = nn.Parameter(torch.zeros(len(windows)))
        self.lateral_gates = nn.Parameter(torch.zeros(len(windows)))
        self.byte_priors = nn.ModuleList(
            [nn.Linear(bw, 256, bias=False) for _ in windows])
        self.boundary = nn.Linear(width, 3)
        self.boundary_to_byte = nn.Linear(3, 256, bias=False)
        self.boundary_gate = nn.Parameter(torch.zeros(()))
        self.pair_memory = PairConceptMemory(bw) if enable_pair else None
        self.unit_heads = (UnitHeads(bw, width)
                           if enable_units else None)
        decoder_input_dim = (2 * (TOKEN_VOCAB_SIZE + WORD_VOCAB_SIZE)
                             + SENTENCE_VOCAB_SIZE + width + 3 * bw)
        self.unit_to_byte = nn.Sequential(
            nn.Linear(decoder_input_dim, 256, bias=True),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256, bias=True),
        )
        self.unit_byte_gate = nn.Parameter(torch.zeros(()))

        # Small, nonzero lateral projections so a gate that opens can learn.
        # The gates themselves are exact zero, which is what preserves parity.
        for head in self.byte_priors:
            nn.init.normal_(head.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        for projection in self.top_down:
            nn.init.normal_(projection.weight, std=0.02)
        nn.init.normal_(self.boundary_to_byte.weight, std=0.02)
        for layer in self.unit_to_byte:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.boundary.bias.copy_(torch.tensor([0.0, -1.0, -2.0]))

    def forward(self, state: torch.Tensor) -> HierarchyState:
        if state.ndim != 3 or state.shape[-1] != self.width:
            raise ValueError("Hierarchy expects [batch,time,width]")
        bus = self.down(self.input_norm(state))
        gates_bu = torch.tanh(self.bottom_up_gates)
        gates_td = torch.tanh(self.top_down_gates)
        gates_lat = torch.tanh(self.lateral_gates)
        if not self.ablation.get("bottom_up", True):
            gates_bu = torch.zeros_like(gates_bu)
        if not self.ablation.get("top_down", True):
            gates_td = torch.zeros_like(gates_td)
        if not self.ablation.get("lateral", True):
            gates_lat = torch.zeros_like(gates_lat)
        boundary_logits = self.boundary(state)
        boundary_probs = torch.sigmoid(boundary_logits.float()).to(state.dtype)

        levels: list[torch.Tensor] = []
        current = bus
        for level, window in enumerate(self.windows):
            pooled = causal_mean(current, window)
            pooled = self.level_norms[level](pooled)
            if (self.boundary_pooling
                    and self.ablation.get("boundary_pooling", True)):
                gate = boundary_probs[..., level:level + 1]
                pooled = gate * current + (1 - gate) * pooled
            levels.append(pooled + gates_bu[level] * self.level_up[level](pooled))
            current = levels[-1]
        subword, word, sentence = levels

        pair_entropy = torch.zeros((), device=state.device)
        pair_gate = torch.zeros((), device=state.device)
        if (self.pair_memory is not None
                and self.ablation.get("pair", True)):
            pair_state = self.pair_memory(subword, sentence)
            sentence = pair_state.sentence
            pair_entropy = pair_state.pair_entropy
            pair_gate = pair_state.pair_gate

        # Top-down sweep: sentence -> word -> subword -> byte.  Each edge is
        # a separate gated residual; at gate zero every level is unchanged.
        word_from_sentence = word + gates_td[2] * self.top_down[2](sentence)
        subword_from_word = subword + gates_td[1] * self.top_down[1](word_from_sentence)
        byte_delta = gates_td[0] * self.top_down[0](subword_from_word)
        enhanced = state + self.up(byte_delta)

        # Lateral priors into the byte distribution.  These are independent
        # paths, not a cycle: they never feed back into the pooled states in
        # the same sweep.
        byte_prior = (
            gates_lat[0] * self.byte_priors[0](subword)
            + gates_lat[1] * self.byte_priors[1](word)
            + gates_lat[2] * self.byte_priors[2](sentence)
            + torch.tanh(self.boundary_gate)
            * self.boundary_to_byte(boundary_probs)
        )
        unit_logits = {}
        unit_byte_raw = torch.zeros(
            (state.shape[0], 256), device=state.device, dtype=state.dtype)
        unit_byte_logits = torch.zeros_like(unit_byte_raw)
        if (self.unit_heads is not None
                and self.ablation.get("units", True)):
            unit_logits = self.unit_heads(subword, word, sentence, state)
            if self.ablation.get("unit_decoder", True):
                decoder_input = torch.cat(
                    (unit_logits["current_token"],
                     unit_logits["current_word"],
                     unit_logits["token"], unit_logits["word"],
                     unit_logits["sentence"],
                     state[:, -1], subword[:, -1], word[:, -1],
                     sentence[:, -1]), dim=-1)
                unit_byte_raw = self.unit_to_byte(decoder_input)
                unit_byte_logits = (
                    torch.tanh(self.unit_byte_gate) * unit_byte_raw)
        return HierarchyState(
            state=enhanced,
            byte_prior=byte_prior,
            subword=subword,
            word=word,
            sentence=sentence,
            boundary_logits=boundary_logits,
            boundary_probs=boundary_probs,
            unit_logits=unit_logits,
            pair_entropy=pair_entropy,
            pair_gate=pair_gate,
            unit_byte_raw=unit_byte_raw,
            unit_byte_logits=unit_byte_logits,
            gates={
                "bottom_up": gates_bu.detach(),
                "top_down": gates_td.detach(),
                "lateral": gates_lat.detach(),
                "boundary": torch.tanh(self.boundary_gate).detach(),
                "unit_byte": torch.tanh(self.unit_byte_gate).detach(),
            },
        )


def boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: tuple[float, float, float] = (2.0, 2.0, 20.0),
) -> torch.Tensor:
    """Mean BCE for the three boundary heads with rare-sentence weighting."""
    if logits.shape != targets.shape or logits.shape[-1] != 3:
        raise ValueError("Boundary loss expects matching [batch,time,3] tensors")
    weight = torch.tensor(pos_weight, dtype=torch.float32,
                          device=logits.device).view(1, 1, 3)
    return F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), pos_weight=weight, reduction="mean")


def bit_logits_to_byte_log_probs(bit_logits: torch.Tensor) -> torch.Tensor:
    """Convert eight conditional bit logits into 256 byte log-probabilities.

    ``bit_logits[:, 0]`` is the most-significant bit (bit 7) and
    ``bit_logits[:, 7]`` is the least-significant bit (bit 0), matching
    ``VerbatimV103RSI._byte_to_conditional_bit_logits``.
    """
    if bit_logits.ndim != 2 or bit_logits.shape[-1] != 8:
        raise ValueError("Expected [batch,8] conditional bit logits")
    values = torch.arange(256, device=bit_logits.device)
    logp = torch.zeros((bit_logits.shape[0], 256),
                       dtype=torch.float32, device=bit_logits.device)
    for index, bit in enumerate(range(7, -1, -1)):
        bit_value = ((values >> bit) & 1).to(torch.float32)
        log_one = F.logsigmoid(bit_logits[:, index].float())[:, None]
        log_zero = F.logsigmoid(-bit_logits[:, index].float())[:, None]
        logp = logp + bit_value[None, :] * log_one + (1 - bit_value[None, :]) * log_zero
    return logp
