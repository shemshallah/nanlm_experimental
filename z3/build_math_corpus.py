#!/usr/bin/env python3
"""Generate a deterministic, tagged math/translation curriculum.

The corpus is deliberately structured rather than scraped: each line is a
document with source problem, symbolic target, and equation.  The tags are
literal text so the current tokenizer can encode them; a later tokenizer pass
can promote them to structural special IDs without changing the data format.

    python z3/build_math_corpus.py --examples 200000 --out corpora/math_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


def _linear(rng: random.Random) -> str:
    a = rng.choice([i for i in range(-12, 13) if i not in (-1, 0, 1)])
    x = rng.randint(-20, 20)
    b = rng.randint(-30, 30)
    c = a * x + b
    sign = "+" if b >= 0 else "-"
    return (
        f"<math><translate_src>Solve for x: {a}x {sign} {abs(b)} = {c}."
        f"</translate_src><translate_tgt>x = {x}</translate_tgt>"
        f"<eq>{a}x {sign} {abs(b)} = {c}</eq></math>"
    )


def _arithmetic(rng: random.Random) -> str:
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    op = rng.choice(["+", "-", "*", "/"])
    if op == "/":
        b = rng.randint(2, 20)
        a = b * rng.randint(2, 20)
        value = a // b
        expr = f"{a} / {b}"
    else:
        value = {"+": a + b, "-": a - b, "*": a * b}[op]
        expr = f"{a} {op} {b}"
    return (
        f"<math><translate_src>Calculate {expr}.</translate_src>"
        f"<translate_tgt>{value}</translate_tgt>"
        f"<eq>{expr} = {value}</eq></math>"
    )


def _percent(rng: random.Random) -> str:
    p = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80])
    n = rng.randrange(20, 2000, 10)
    value = p * n // 100
    return (
        f"<math><translate_src>What is {p}% of {n}?</translate_src>"
        f"<translate_tgt>{value}</translate_tgt>"
        f"<eq>{p}/100 * {n} = {value}</eq></math>"
    )


def _rectangle(rng: random.Random) -> str:
    w, h = rng.randint(2, 40), rng.randint(2, 40)
    area = w * h
    return (
        f"<math><translate_src>A rectangle has width {w} and height {h}. "
        f"What is its area?</translate_src>"
        f"<translate_tgt>{area} square units</translate_tgt>"
        f"<eq>area = {w} * {h} = {area}</eq></math>"
    )


def _triangle(rng: random.Random) -> str:
    b, h = rng.randint(2, 40), rng.randint(2, 40)
    twice = b * h
    area = twice / 2
    area_text = str(int(area)) if area.is_integer() else f"{area:.1f}"
    return (
        f"<math><translate_src>A triangle has base {b} and height {h}. "
        f"What is its area?</translate_src>"
        f"<translate_tgt>{area_text} square units</translate_tgt>"
        f"<eq>area = ({b} * {h}) / 2 = {area_text}</eq></math>"
    )


def _speed(rng: random.Random) -> str:
    hours = rng.randint(2, 12)
    speed = rng.randint(20, 120)
    distance = hours * speed
    return (
        f"<math><translate_src>A train travels {distance} km in {hours} hours. "
        f"What is its average speed?</translate_src>"
        f"<translate_tgt>{speed} km/h</translate_tgt>"
        f"<eq>speed = {distance} / {hours} = {speed}</eq></math>"
    )


def _expand(rng: random.Random) -> str:
    a, b = rng.randint(-9, 9), rng.randint(-9, 9)
    middle = a + b
    constant = a * b
    middle_text = (
        f"+ {middle}x" if middle > 0 else
        (f"- {abs(middle)}x" if middle < 0 else "")
    )
    constant_text = (
        f"+ {constant}" if constant > 0 else
        (f"- {abs(constant)}" if constant < 0 else "")
    )
    result = f"x^2 {middle_text} {constant_text}".replace("  ", " ").strip()
    return (
        f"<math><translate_src>Expand (x {'+' if a >= 0 else '-'} {abs(a)})"
        f"(x {'+' if b >= 0 else '-'} {abs(b)}).</translate_src>"
        f"<translate_tgt>{result}</translate_tgt>"
        f"<eq>(x {'+' if a >= 0 else '-'} {abs(a)})(x "
        f"{'+' if b >= 0 else '-'} {abs(b)}) = {result}</eq></math>"
    )


def _unit(rng: random.Random) -> str:
    meters = rng.randint(1, 50) * 100
    km = meters / 1000
    km_text = f"{km:g}"
    return (
        f"<math><translate_src>Convert {meters} meters to kilometers."
        f"</translate_src><translate_tgt>{km_text} km</translate_tgt>"
        f"<eq>{meters} / 1000 = {km_text}</eq></math>"
    )


GENERATORS = (_linear, _arithmetic, _percent, _rectangle, _triangle, _speed, _expand, _unit)


def generate(count: int, seed: int = 20261113) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[str] = []
    attempts = 0
    while len(rows) < count and attempts < count * 20:
        attempts += 1
        line = rng.choice(GENERATORS)(rng)
        if line in seen:
            continue
        seen.add(line)
        rows.append(line)
    if len(rows) < count:
        raise RuntimeError(f"generator saturated at {len(rows)} examples")
    return rows


def _bucket(line: str) -> float:
    digest = hashlib.sha256(line.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=200_000)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out", default="corpora/math_v1")
    args = parser.parse_args(argv)
    rows = generate(max(1, args.examples), args.seed)
    train = [row for row in rows if _bucket(row) >= args.valid_fraction]
    valid = [row for row in rows if _bucket(row) < args.valid_fraction]
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "valid.txt").write_text("\n".join(valid) + "\n", encoding="utf-8")
    manifest = {
        "examples": len(rows),
        "train": len(train),
        "valid": len(valid),
        "seed": args.seed,
        "format": "<math><translate_src>...</translate_src>"
                  "<translate_tgt>...</translate_tgt><eq>...</eq></math>",
        "train_bytes": sum(len(r.encode()) + 1 for r in train),
        "valid_bytes": sum(len(r.encode()) + 1 for r in valid),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
