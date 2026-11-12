#!/usr/bin/env python3
"""Rotary position embeddings with long-context scaling for nanlm v3.

The rotation layout is the half-split / GPT-NeoX layout used by
``z3/v3_model.RoPE``: dimensions ``i`` and ``i + head_dim / 2`` form a 2-D
pair.  It is *not* the interleaved ``(2i, 2i + 1)`` layout used by some LLaMA
implementations.  Keeping the layout compatible means this module can replace
the tiny RoPE in the v3 attention block without changing the checkpoint
format.

Scaling modes
-------------
``"none"``
    Standard RoPE with inverse frequencies
    ``inv_freq[i] = base ** (-2i / head_dim)``.

``"ntk"``
    Static NTK-aware base scaling (bloc97's "NTK-Aware Scaled RoPE").  The
    base is stretched to ``base * factor ** (head_dim / (head_dim - 2))``
    before the inverse-frequency table is built.  Unlike piecewise YaRN, this
    leaves every frequency finite and monotonic while making the low-frequency
    channels extrapolate further.  The ``head_dim / (head_dim - 2)`` exponent
    is the standard NTK formula; it requires ``head_dim > 2``.

``"yarn"``
    YaRN (Peng et al., 2023).  Wavelengths longer than the ``beta_slow``
    cutoff are linearly interpolated by ``factor``; wavelengths shorter than
    the ``beta_fast`` cutoff are left extrapolated; dimensions between the two
    cutoffs are smoothly ramped.  This gives an effective context of
    ``factor * original_context`` without retraining every frequency.

    The rotary embedding is additionally multiplied by ``attention_factor``.
    When not supplied, it defaults to
    ``sqrt(1 + ln(factor) / ln(original_context))`` as specified for v3.
    Note that the YaRN paper applies ``0.1 * ln(factor) + 1`` to attention
    logits; pass an explicit ``attention_factor`` to use that convention
    instead.

The cos/sin tables are precomputed once for ``max_context`` positions, so
memory use is O(max_context * head_dim), never O(max_context ** 2).  The
tables are non-persistent buffers: they move with ``.to(device)`` / ``.cuda()``
but are not written to checkpoints, since they are cheap to rebuild.

Example
-------
>>> rope = build_rope(64, scaling="yarn", factor=8.0)
>>> q = torch.randn(2, 12, 4096, 64)
>>> y = rope(q)                      # (2, 12, 4096, 64)
>>> k = rope(one_token, offset=4096) # optional KV-cache continuation
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

__all__ = ["RoPE", "build_rope"]

_SUPPORTED_SCALINGS = ("none", "ntk", "yarn")


def _yarn_correction_range(
    head_dim: int,
    base: float,
    original_context: int,
    beta_fast: float,
    beta_slow: float,
) -> tuple[float, float]:
    """Return the rotary-dimension range over which YaRN blends.

    This follows the YaRN reference implementation: ``beta_fast`` and
    ``beta_slow`` are numbers of rotations, converted to a dimension index by
    solving ``original_context / (2 * pi * n) = base ** (2 * d / head_dim)``.
    Dimensions below ``low`` extrapolate at the original frequency, dimensions
    from ``high`` onward are fully interpolated, and the range between them is
    a linear ramp.  Both cutoffs are clamped to the valid half-dimension range
    so small head dimensions still work.
    """

    def dim_for_rotations(num_rotations: float) -> float:
        return (
            head_dim
            * math.log(original_context / (num_rotations * 2.0 * math.pi))
        ) / (2.0 * math.log(base))

    half = head_dim // 2
    low = math.floor(dim_for_rotations(beta_fast))
    high = math.ceil(dim_for_rotations(beta_slow))

    low = max(0, min(low, half - 1))
    high = max(0, min(high, half - 1))
    if low == high:
        # Degenerate configuration: keep the ramp well-defined.
        high = low + 0.001
    return float(low), float(high)


def _linear_ramp_factor(low: float, high: float, dim: int) -> torch.Tensor:
    """Linear ramp from 0 at ``low`` to 1 at ``high``, clamped to [0, 1]."""
    if low == high:
        high = low + 0.001
    ramp = (torch.arange(dim, dtype=torch.float64) - low) / (high - low)
    return ramp.clamp_(0.0, 1.0)


class RoPE(nn.Module):
    """Rotary position embedding for ``(B, H, T, D)`` query/key tensors.

    Parameters
    ----------
    head_dim:
        Per-head width.  Must be even.
    base:
        Standard RoPE theta (default 10,000).
    max_context:
        Number of positions cached.  ``T`` may be at most this value.
    original_context:
        Context length the base frequencies were trained on (default 1,024).
        Used by YaRN's wavelength cutoffs and default attention factor.
    scaling:
        ``"none"``, ``"ntk"``, or ``"yarn"``.
    factor:
        Context-extension factor (default 8 -> 8,192 from 1,024).
    beta_fast, beta_slow:
        YaRN wavelength cutoffs in rotations (defaults 32 and 1).
    attention_factor:
        Multiplier applied to the cos/sin tables.  Defaults to 1.0 for
        ``"none"``/``"ntk"`` and ``sqrt(1 + ln(factor)/ln(original_context))``
        for ``"yarn"``.

    Notes
    -----
    The module is stateless apart from its non-persistent ``inv_freq``,
    ``cos_cached``, and ``sin_cached`` buffers.  Calling it is differentiable
    and has the same cost as the standard v3 RoPE.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_context: int = 8192,
        original_context: int = 1024,
        scaling: str = "none",
        factor: float = 8.0,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        attention_factor: Optional[float] = None,
    ) -> None:
        super().__init__()

        if not isinstance(head_dim, int) or isinstance(head_dim, bool):
            raise TypeError("head_dim must be an integer")
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even and >= 2, got {head_dim}")

        if not isinstance(scaling, str) or scaling.lower() not in _SUPPORTED_SCALINGS:
            raise ValueError(
                f"scaling must be one of {_SUPPORTED_SCALINGS}, got {scaling!r}"
            )
        scaling = scaling.lower()

        base = float(base)
        if not math.isfinite(base) or base <= 1.0:
            raise ValueError(f"base must be finite and > 1, got {base}")

        max_context = int(max_context)
        original_context = int(original_context)
        if max_context < 1:
            raise ValueError(f"max_context must be >= 1, got {max_context}")
        if original_context < 2:
            raise ValueError(f"original_context must be >= 2, got {original_context}")

        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"factor must be finite and > 0, got {factor}")
        if scaling in ("ntk", "yarn") and factor < 1.0:
            raise ValueError(
                f"factor must be >= 1 for scaling={scaling!r}, got {factor}"
            )

        if scaling == "ntk" and head_dim <= 2:
            raise ValueError("NTK scaling requires head_dim > 2")

        beta_fast = float(beta_fast)
        beta_slow = float(beta_slow)
        if scaling == "yarn":
            if not (math.isfinite(beta_fast) and math.isfinite(beta_slow)):
                raise ValueError("beta_fast and beta_slow must be finite")
            if beta_slow <= 0.0 or beta_fast <= beta_slow:
                raise ValueError(
                    "YaRN requires 0 < beta_slow < beta_fast, got "
                    f"beta_slow={beta_slow}, beta_fast={beta_fast}"
                )

        if attention_factor is None:
            if scaling == "yarn":
                attention_factor = math.sqrt(
                    1.0 + math.log(factor) / math.log(original_context)
                )
            else:
                attention_factor = 1.0
        else:
            attention_factor = float(attention_factor)
            if not math.isfinite(attention_factor) or attention_factor <= 0.0:
                raise ValueError(
                    "attention_factor must be finite and > 0, got "
                    f"{attention_factor}"
                )

        self.head_dim = head_dim
        self.base = base
        self.max_context = max_context
        self.original_context = original_context
        self.scaling = scaling
        self.factor = factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attention_factor = float(attention_factor)

        # Build frequencies in float64 for accurate angles at 8k positions,
        # then expose a float32 buffer (the usual model dtype) and cache the
        # cos/sin tables once.  Cache construction is O(max_context * head_dim).
        inv_freq = self._build_inv_freq()
        positions = torch.arange(max_context, dtype=torch.float64)
        angles = torch.outer(positions, inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        cos = emb.cos().mul_(self.attention_factor)
        sin = emb.sin().mul_(self.attention_factor)

        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)
        self.register_buffer("cos_cached", cos.to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", sin.to(torch.float32), persistent=False)

    def _build_inv_freq(self) -> torch.Tensor:
        """Return float64 inverse frequencies for the selected scaling mode."""
        exponent = (
            torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim
        )

        if self.scaling == "none":
            return 1.0 / (self.base**exponent)

        if self.scaling == "ntk":
            # Standard NTK-aware scaled base.  The head_dim/(head_dim-2)
            # exponent preserves the Nyquist dimension while stretching the
            # low-frequency wavelengths.
            scaled_base = self.base * self.factor ** (
                self.head_dim / (self.head_dim - 2)
            )
            return 1.0 / (scaled_base**exponent)

        # YaRN: extrapolate high frequencies, interpolate low frequencies, and
        # linearly blend the dimensions between the beta cutoffs.
        pos_freqs = self.base**exponent
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.factor * pos_freqs)
        low, high = _yarn_correction_range(
            self.head_dim,
            self.base,
            self.original_context,
            self.beta_fast,
            self.beta_slow,
        )
        ramp = _linear_ramp_factor(low, high, self.head_dim // 2)
        return inv_freq_extrapolation * (1.0 - ramp) + inv_freq_interpolation * ramp

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Rotate ``x`` of shape ``(B, H, T, D)``.

        ``offset`` is the absolute position of ``x[..., 0, :]`` and is useful
        for KV-cache decoding; the default starts at position zero.
        """
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
        if x.ndim != 4:
            raise ValueError(f"x must have shape (B, H, T, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.head_dim:
            raise ValueError(
                f"x last dimension must be head_dim={self.head_dim}, "
                f"got {x.shape[-1]}"
            )
        if not isinstance(offset, int) or isinstance(offset, bool):
            raise TypeError("offset must be an integer")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")

        t = x.shape[-2]
        end = offset + t
        if end > self.max_context:
            raise ValueError(
                f"positions [{offset}, {end}) exceed max_context="
                f"{self.max_context}; increase max_context or shorten the input"
            )

        # (T, D) -> (1, 1, T, D); cast once per call so fp16/bf16/fp64 inputs
        # get correctly rounded angles without making the cached tables huge.
        cos = self.cos_cached[offset:end].to(dtype=x.dtype)[None, None, :, :]
        sin = self.sin_cached[offset:end].to(dtype=x.dtype)[None, None, :, :]

        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, base={self.base}, "
            f"max_context={self.max_context}, "
            f"original_context={self.original_context}, "
            f"scaling={self.scaling!r}, factor={self.factor}, "
            f"beta_fast={self.beta_fast}, beta_slow={self.beta_slow}, "
            f"attention_factor={self.attention_factor:.6g}"
        )


def build_rope(head_dim: int, **kwargs) -> RoPE:
    """Factory for :class:`RoPE`; forwards all keyword arguments."""
    return RoPE(head_dim, **kwargs)
