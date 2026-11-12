#!/usr/bin/env python3
"""Tests for z3/rope_scaling.py.

Run from the repository root with either:

    python -m unittest z3/test_rope_scaling.py -v
    python z3/test_rope_scaling.py
"""

from __future__ import annotations

import math
import unittest

import torch

try:  # imported as z3.test_rope_scaling
    from .rope_scaling import RoPE, build_rope
except ImportError:  # executed directly as z3/test_rope_scaling.py
    from rope_scaling import RoPE, build_rope


def _randn(*shape: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _direct_rope_reference(
    x: torch.Tensor, base: float = 10000.0, offset: int = 0
) -> torch.Tensor:
    """Straightforward half-split RoPE written independently of the module."""
    dim = x.shape[-1]
    exponent = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    inv_freq = 1.0 / (base**exponent)
    positions = torch.arange(
        offset, offset + x.shape[-2], dtype=torch.float64, device=x.device
    )
    angles = torch.outer(positions, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=x.dtype)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


def _yarn_reference_inv_freq(
    head_dim: int,
    base: float,
    original_context: int,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> tuple[torch.Tensor, float, float]:
    """Independent implementation of the YaRN frequency blend."""

    def dim_for_rotations(num_rotations: float) -> float:
        return (
            head_dim
            * math.log(original_context / (num_rotations * 2.0 * math.pi))
        ) / (2.0 * math.log(base))

    exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
    pos_freqs = base**exponent
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (factor * pos_freqs)

    half = head_dim // 2
    low = max(0, min(math.floor(dim_for_rotations(beta_fast)), half - 1))
    high = max(0, min(math.ceil(dim_for_rotations(beta_slow)), half - 1))
    if low == high:
        high = low + 0.001
    ramp = ((torch.arange(half, dtype=torch.float64) - low) / (high - low)).clamp(0.0, 1.0)
    return extrapolation * (1.0 - ramp) + interpolation * ramp, float(low), float(high)


class RoPETest(unittest.TestCase):
    def test_shape_and_dtype_preserved(self) -> None:
        rope = RoPE(64, max_context=512, original_context=128)
        x = _randn(2, 4, 37, 64)
        y = rope(x)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, x.dtype)
        self.assertTrue(torch.isfinite(y).all())

        x64 = x.to(torch.float64)
        y64 = rope(x64)
        self.assertEqual(y64.dtype, torch.float64)

    def test_standard_matches_direct_reference_at_1024(self) -> None:
        rope = RoPE(32, base=10000.0, max_context=2048, original_context=1024)
        x = _randn(2, 2, 1024, 32, seed=1)
        y = rope(x)
        expected = _direct_rope_reference(x, base=10000.0)
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)

    def test_standard_matches_reference_short_and_offset(self) -> None:
        for head_dim in (16, 32):
            for tokens, offset in ((1, 0), (127, 0), (333, 17)):
                with self.subTest(head_dim=head_dim, tokens=tokens, offset=offset):
                    rope = RoPE(head_dim, max_context=2048, original_context=1024)
                    x = _randn(2, 3, tokens, head_dim, seed=tokens + offset)
                    actual = rope(x, offset=offset)
                    expected = _direct_rope_reference(x, offset=offset)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_half_split_layout_matches_v3_rotation(self) -> None:
        # Dimensions (0, 2) and (1, 3) are rotated pairs at position 1.
        rope = RoPE(4, max_context=2, original_context=2)
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        y = rope(x, offset=1)

        c0, s0 = math.cos(1.0), math.sin(1.0)
        c1, s1 = math.cos(0.01), math.sin(0.01)
        expected = torch.tensor(
            [[[[c0 - 3.0 * s0, 2.0 * c1 - 4.0 * s1,
                3.0 * c0 + 1.0 * s0, 4.0 * c1 + 2.0 * s1]]]]
        )
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)

    def test_offset_matches_sliced_full_forward(self) -> None:
        rope = RoPE(16, max_context=512, original_context=128)
        x = _randn(2, 2, 300, 16, seed=3)
        full = rope(x)
        start, end = 57, 231
        sliced = rope(x[:, :, start:end], offset=start)
        torch.testing.assert_close(sliced, full[:, :, start:end], rtol=0, atol=0)

    def test_norm_is_preserved_up_to_attention_factor(self) -> None:
        x = _randn(2, 3, 33, 32, seed=5)
        for mode in ("none", "ntk", "yarn"):
            with self.subTest(mode=mode):
                rope = RoPE(
                    32,
                    max_context=256,
                    original_context=64,
                    scaling=mode,
                    factor=4.0,
                )
                y = rope(x)
                expected = rope.attention_factor * x.norm(dim=-1)
                torch.testing.assert_close(
                    y.norm(dim=-1), expected, rtol=1e-5, atol=1e-5
                )

    def test_relative_position_property(self) -> None:
        head_dim = 32
        q = _randn(1, 1, 1, head_dim, dtype=torch.float64, seed=7) / math.sqrt(head_dim)
        k = _randn(1, 1, 1, head_dim, dtype=torch.float64, seed=8) / math.sqrt(head_dim)
        # All pairs have the same relative offset of 4.
        pairs = ((1, 5), (33, 37), (64, 68), (128, 132))

        for mode in ("none", "ntk", "yarn"):
            with self.subTest(mode=mode):
                rope = RoPE(
                    head_dim,
                    max_context=256,
                    original_context=64,
                    scaling=mode,
                    factor=4.0,
                )
                scores = []
                for m, n in pairs:
                    qm = rope(q, offset=m)
                    kn = rope(k, offset=n)
                    scores.append((qm * kn).sum())
                for score in scores[1:]:
                    torch.testing.assert_close(
                        score, scores[0], rtol=0, atol=2e-4
                    )

    def test_ntk_base_scaling_formula(self) -> None:
        head_dim, base, factor = 32, 10000.0, 8.0
        rope = RoPE(
            head_dim,
            base=base,
            max_context=8192,
            original_context=1024,
            scaling="ntk",
            factor=factor,
        )
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        scaled_base = base * factor ** (head_dim / (head_dim - 2))
        expected = 1.0 / (scaled_base**exponent)
        standard = 1.0 / (base**exponent)

        torch.testing.assert_close(
            rope.inv_freq.to(torch.float64), expected, rtol=1e-6, atol=1e-12
        )
        self.assertEqual(rope.attention_factor, 1.0)
        # Every channel except the Nyquist one is stretched.
        self.assertTrue((rope.inv_freq[1:] < standard[1:].to(torch.float32)).all())
        self.assertTrue(torch.isfinite(rope.inv_freq).all())

    def test_yarn_frequency_blend_and_attention_factor(self) -> None:
        head_dim, base, original, factor = 64, 10000.0, 1024, 8.0
        beta_fast, beta_slow = 32.0, 1.0
        rope = RoPE(
            head_dim,
            base=base,
            max_context=8192,
            original_context=original,
            scaling="yarn",
            factor=factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )
        expected, low, high = _yarn_reference_inv_freq(
            head_dim, base, original, factor, beta_fast, beta_slow
        )
        torch.testing.assert_close(
            rope.inv_freq.to(torch.float64), expected, rtol=1e-6, atol=1e-12
        )

        expected_attention = math.sqrt(
            1.0 + math.log(factor) / math.log(original)
        )
        self.assertAlmostEqual(
            rope.attention_factor, expected_attention, places=12
        )
        torch.testing.assert_close(
            rope.cos_cached[0],
            torch.full((head_dim,), rope.attention_factor),
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            rope.sin_cached[0],
            torch.zeros(head_dim),
            rtol=0,
            atol=1e-7,
        )

        # Dimensions below `low` extrapolate; dimensions from `high` on are
        # fully interpolated; the middle is a strict blend.
        low_i, high_i = int(low), int(high)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim
        standard = 1.0 / (base**exponent)
        interpolated = 1.0 / (factor * base**exponent)
        torch.testing.assert_close(
            rope.inv_freq[: low_i + 1].to(torch.float64),
            standard[: low_i + 1],
            rtol=1e-6,
            atol=1e-12,
        )
        torch.testing.assert_close(
            rope.inv_freq[high_i:].to(torch.float64),
            interpolated[high_i:],
            rtol=1e-6,
            atol=1e-12,
        )
        middle = rope.inv_freq[low_i + 1 : high_i].to(torch.float64)
        self.assertGreater(middle.numel(), 0)
        self.assertTrue((middle < standard[low_i + 1 : high_i]).all())
        self.assertTrue((middle > interpolated[low_i + 1 : high_i]).all())

    def test_yarn_explicit_attention_factor(self) -> None:
        rope = RoPE(
            32,
            max_context=256,
            original_context=64,
            scaling="yarn",
            factor=4.0,
            attention_factor=1.5,
        )
        self.assertEqual(rope.attention_factor, 1.5)
        torch.testing.assert_close(
            rope.cos_cached[0],
            torch.full((32,), 1.5),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_all_modes_are_finite_at_8192(self) -> None:
        head_dim, max_context = 32, 8192
        x = _randn(1, 2, max_context, head_dim, seed=11)
        for mode in ("none", "ntk", "yarn"):
            with self.subTest(mode=mode):
                rope = RoPE(
                    head_dim,
                    max_context=max_context,
                    original_context=1024,
                    scaling=mode,
                    factor=8.0,
                )
                y = rope(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.isfinite(rope.cos_cached).all())
                self.assertTrue(torch.isfinite(rope.sin_cached).all())
                self.assertTrue(torch.isfinite(rope.inv_freq).all())
                self.assertLessEqual(
                    rope.cos_cached.abs().max().item(),
                    rope.attention_factor + 1e-6,
                )
                self.assertLessEqual(
                    rope.sin_cached.abs().max().item(),
                    rope.attention_factor + 1e-6,
                )

    def test_scaling_modes_differ_at_8192(self) -> None:
        head_dim, max_context = 64, 8192
        ropes = {
            mode: RoPE(
                head_dim,
                max_context=max_context,
                original_context=1024,
                scaling=mode,
                factor=8.0,
            )
            for mode in ("none", "ntk", "yarn")
        }
        for left, right in (("none", "ntk"), ("none", "yarn"), ("ntk", "yarn")):
            with self.subTest(left=left, right=right):
                self.assertFalse(
                    torch.allclose(
                        ropes[left].inv_freq, ropes[right].inv_freq, rtol=1e-4
                    )
                )

                x = _randn(1, 1, max_context, head_dim, seed=13)
                y_left = ropes[left](x)
                y_right = ropes[right](x)
                self.assertTrue(torch.isfinite(y_left).all())
                self.assertTrue(torch.isfinite(y_right).all())
                self.assertGreater(
                    (y_left - y_right).abs().max().item(), 1e-2
                )

    def test_cache_size_is_linear_in_context(self) -> None:
        head_dim, max_context = 64, 8192
        rope = RoPE(head_dim, max_context=max_context, original_context=1024)
        self.assertEqual(tuple(rope.cos_cached.shape), (max_context, head_dim))
        self.assertEqual(tuple(rope.sin_cached.shape), (max_context, head_dim))
        self.assertEqual(rope.cos_cached.numel(), max_context * head_dim)
        self.assertLess(
            rope.cos_cached.numel() + rope.sin_cached.numel(),
            max_context * max_context,
        )

    def test_gradients_are_finite_for_every_mode(self) -> None:
        for mode in ("none", "ntk", "yarn"):
            with self.subTest(mode=mode):
                rope = RoPE(
                    16,
                    max_context=512,
                    original_context=128,
                    scaling=mode,
                    factor=4.0,
                )
                x = _randn(2, 3, 64, 16, seed=17)
                x.requires_grad_(True)
                y = rope(x, offset=128)
                loss = y.pow(2).mean()
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_build_rope_factory(self) -> None:
        rope = build_rope(
            64,
            base=10000.0,
            max_context=8192,
            original_context=1024,
            scaling="yarn",
            factor=8.0,
        )
        self.assertIsInstance(rope, RoPE)
        self.assertEqual(rope.scaling, "yarn")
        self.assertEqual(rope.max_context, 8192)
        y = rope(_randn(1, 2, 8, 64, seed=19))
        self.assertEqual(y.shape, (1, 2, 8, 64))

        standard = build_rope(32, scaling="none")
        self.assertEqual(standard.scaling, "none")
        self.assertEqual(standard.attention_factor, 1.0)

    def test_invalid_configurations(self) -> None:
        invalid = (
            dict(head_dim=0),
            dict(head_dim=3),
            dict(head_dim=8, base=1.0),
            dict(head_dim=8, base=float("nan")),
            dict(head_dim=8, max_context=0),
            dict(head_dim=8, original_context=1),
            dict(head_dim=8, scaling="bogus"),
            dict(head_dim=2, scaling="ntk"),
            dict(head_dim=8, scaling="yarn", factor=0.5),
            dict(head_dim=8, scaling="yarn", beta_fast=1.0, beta_slow=32.0),
            dict(head_dim=8, scaling="yarn", beta_fast=32.0, beta_slow=0.0),
            dict(head_dim=8, scaling="yarn", attention_factor=0.0),
            dict(head_dim=8, scaling="yarn", attention_factor=-1.0),
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    RoPE(**kwargs)

        with self.assertRaises(TypeError):
            RoPE(7.5)  # type: ignore[arg-type]

    def test_invalid_forward_inputs(self) -> None:
        rope = RoPE(8, max_context=16, original_context=8)

        with self.assertRaises(ValueError):
            rope(_randn(1, 1, 8, 8, seed=23)[0])  # 3-D

        with self.assertRaises(ValueError):
            rope(_randn(1, 1, 8, 16, seed=23))  # wrong head_dim

        with self.assertRaises(ValueError):
            rope(_randn(1, 1, 17, 8, seed=23))  # beyond max_context

        with self.assertRaises(ValueError):
            rope(_randn(1, 1, 8, 8, seed=23), offset=9)  # offset + T too long

        with self.assertRaises(ValueError):
            rope(_randn(1, 1, 8, 8, seed=23), offset=-1)

        with self.assertRaises(TypeError):
            rope(torch.zeros(1, 1, 8, 8, dtype=torch.long))

        with self.assertRaises(TypeError):
            rope(_randn(1, 1, 8, 8, seed=23), offset=1.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
