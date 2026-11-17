"""Immutable, dataset-matched RSI champion guard.

RSI internal rewards use TRAIN only. After a complete training round, DEV may
select a separately stored champion; TEST is never touched. Previously promoted
weights AND AdamW histories remain byte-identical unless a strictly better
candidate is proven on the exact same DEV protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

SCHEMA = "terminal-rsi-protected-champion-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identity(receipt: dict) -> dict:
    if not receipt.get("old_v103_logits_identical_at_gate_zero"):
        raise ValueError("Cannot promote without original-v103 zero-gate parity")
    if not receipt.get("new_dev_id_level_train_disjoint_proven"):
        raise ValueError("Cannot promote without book-ID train/DEV disjointness")
    if not receipt.get("test_untouched"):
        raise ValueError("Cannot promote a candidate that opened TEST")
    return {
        "dev_digest": receipt["book_disjoint_dev_digest"],
        "dev_books": receipt["new_dev_books"],
        "dev_scored_positions": receipt["new_dev_scored_positions"],
        "bp_benchmark_positions": receipt["after"]["evaluated_bytes"],
        "context": receipt["upgrade_context"],
    }


def _metric(receipt: dict) -> float:
    score = float(receipt["after"]["bpb"])
    if not math.isfinite(score) or score <= 0:
        raise ValueError("Candidate DEV BPB must be positive and finite")
    return score


def _atomic_write(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=".pending-rsi-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _copy_if_immutable(source: Path, target: Path, digest: str) -> None:
    if target.exists():
        if sha256(target) != digest:
            raise ValueError("Refusing to overwrite an existing immutable champion checkpoint")
        return
    fd, name = tempfile.mkstemp(prefix=".pending-rsi-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out, length=4 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        if sha256(Path(name)) != digest:
            raise ValueError("Candidate checkpoint checksum changed during protected copy")
        if target.exists():
            if sha256(target) != digest:
                raise ValueError("Concurrent champion archive has different contents")
        else:
            # Never replace or truncate a published immutable checkpoint.
            os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)


def consider_candidate(checkpoint: Path, receipt: dict, champion_dir: Path,
                       *, minimum_bpb_improvement: float = 1e-5) -> dict:
    """Archive candidate iff it improves matched DEV, never overwrite champion.

    Keeps all historic accepted checkpoints recoverable, including optimizer
    moments, and updates only the tiny best.json pointer on success.
    """
    if not math.isfinite(minimum_bpb_improvement) or minimum_bpb_improvement <= 0:
        raise ValueError("Minimum promotion improvement must be positive and finite")
    checkpoint, champion_dir = Path(checkpoint), Path(champion_dir)
    if not checkpoint.is_file():
        raise ValueError("Candidate full checkpoint does not exist")
    identity = _identity(receipt)
    score = _metric(receipt)
    champion_dir.mkdir(parents=True, exist_ok=True)
    index = champion_dir / "best.json"
    old = json.loads(index.read_text()) if index.exists() else None
    if old is not None:
        if old.get("schema") != SCHEMA or old.get("benchmark_identity") != identity:
            raise ValueError("Champion uses a DIFFERENT DEV split or metric protocol; do not compare BPB")
        old_file = champion_dir / old["checkpoint"]
        if not old_file.is_file() or sha256(old_file) != old["checkpoint_sha256"]:
            raise ValueError("Champion checkpoint missing or corrupted; refusing promotion")
        if score >= float(old["bpb"]) - minimum_bpb_improvement:
            return {
                "status": "retained", "candidate_bpb": score,
                "champion_bpb": float(old["bpb"]),
                "champion_checkpoint": str(old_file),
                "protected": True,
            }
    digest = sha256(checkpoint)
    name = f"step{int(receipt['final_step'])}_{digest[:16]}.pt"
    target = champion_dir / name
    _copy_if_immutable(checkpoint, target, digest)
    record = {
        "schema": SCHEMA,
        "benchmark_identity": identity,
        "bpb": score,
        "checkpoint": name,
        "checkpoint_sha256": digest,
        "final_step": int(receipt["final_step"]),
        "rsi_self_construction": receipt.get("rsi_self_construction"),
        "previous_champion": old["checkpoint"] if old else None,
    }
    _atomic_write(champion_dir / f"{name}.json",
                  (json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
    _atomic_write(index, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
    return {
        "status": "improved" if old else "initialized",
        "candidate_bpb": score, "champion_bpb": score,
        "champion_checkpoint": str(target), "protected": True,
    }
