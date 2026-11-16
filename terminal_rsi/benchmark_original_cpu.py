"""Read-only, actual-trained-weight CPU benchmark for ORIGINAL Terminal-RSI v103.

Requires original SHA-matched TRAIN stream, BPE ranks and book-disjoint DEV.
Never creates a checkpoint, never tunes weights, never opens sealed TEST.
Produces exact v103 bitwise loss/BPB, all-256-byte top1/top5 and SHA metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from terminal_rsi import legacy_v103_reference as original


def benchmark(*, checkpoint: Path, train_stream: Path, dev_bin: Path,
              bpe_merges: Path, topk_probes: int = 512,
              max_bpb_bytes: int | None = None,
              reference_dev: Path | None = None,
              reference_full256: Path | None = None) -> dict:
    if topk_probes < 1 or (max_bpb_bytes is not None and max_bpb_bytes < 1):
        raise ValueError("Benchmark position counts must be positive")
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("schema") != original.PARITY_SCHEMA:
        raise ValueError("Only the original GPU-v103 lineage is accepted; "
                         "WordNet hybrid_latest.pt and FA1/FA2 are different models")
    state = saved.get("optimizer", {})
    if len(state.get("param_groups", [])) != 2 or not state.get("state"):
        raise ValueError("Missing original trained two-group AdamW optimizer history")
    corpus = train_stream.read_bytes()
    corpus_sha = hashlib.sha256(corpus).hexdigest()
    if saved["source_sha256"] != corpus_sha:
        raise ValueError("Original 10M training stream SHA differs from checkpoint")
    ranks = original.read_bpe_ranks(bpe_merges)
    bpe_sha = hashlib.sha256(bpe_merges.read_bytes()).hexdigest()
    if len(ranks) != 600:
        raise ValueError("Expected exactly the original 600 BPE ranks")
    if bpe_sha != "cff04c727f2d91e3ae3c2f2bdcf0adeca1422c3a8403488d96aaa759440e8ff5":
        raise ValueError("BPE contents differ from original v103 archive")
    # Original native feature builder, original target-free train match index.
    train_features, historic_index = original.build_train_features(corpus, ranks)
    manifest = original.feature_manifest(corpus, ranks, train_features)
    manifest["bpe_merge_sha256"] = bpe_sha
    if manifest != saved.get("v103_feature_manifest"):
        raise ValueError("Original v103 causal-feature SHA or manifest mismatch")
    independent = original.heldout_windows(str(dev_bin), torch.device("cpu"),
                                           history=original.PARITY_CONTEXT)
    windows, dev_sha = independent
    old_dev = (saved.get("holdout_digests") or {}).get("dev")
    if old_dev and old_dev != dev_sha:
        raise ValueError("Original independent-book DEV digest changed")
    aux = original.heldout_auxiliary(str(dev_bin), torch.device("cpu"),
                                    ranks, historic_index)
    copy_enabled = bool((saved.get("configuration") or {}).get("causal_copy_enabled", False))
    local = None
    if copy_enabled:
        from terminal_rsi.gpu_experimental.v103_local_copy import book_local_features
        fields, local_sha = book_local_features(str(dev_bin))
        if local_sha != dev_sha:
            raise AssertionError("Archived local-copy book SHA alignment changed")
        local = torch.from_numpy(fields).to(torch.int32)
    model = original.V103ParityGPUFullSystem().cpu()
    model.load_state_dict(saved["model"], strict=True)
    model.causal_copy_enabled = copy_enabled
    model.eval()
    with torch.inference_mode():
        results = original.byte_metrics(
            model, windows, max_bpb_bytes,
            topk_n=topk_probes, auxiliary=aux, local=local)
    # Independent CPU read-only benchmark must not modify RSI memory, either.
    for key, value in saved["model"].items():
        if not torch.equal(model.state_dict()[key].cpu(), value.cpu()):
            raise AssertionError("DEV scoring unexpectedly modified RSI: " + key)
    report = {
        "protocol": "terminal-rsi-original-v103-native-CPU-book-disjoint-DEV",
        "trained_steps": int(saved["steps"]),
        "historical_adam_groups": 2,
        "original_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "original_train_sha256": corpus_sha,
        "original_dev_sha256": dev_sha,
        "original_bpe_sha256": bpe_sha,
        "historical_feature_sha256": manifest["feature_sha256"],
        "causal_copy_enabled": copy_enabled,
        "train_bytes": len(corpus),
        "device": "cpu",
        **results,
    }
    refs = (saved.get("holdout_digests") or {})
    if reference_dev:
        if refs.get("c_dev_reference") and hashlib.sha256(reference_dev.read_bytes()).hexdigest() != refs["c_dev_reference"]:
            raise ValueError("Wrong frozen C DEV reference SHA")
        report["matched_original_c_dev"] = original.compare_frozen_v103(
            model, windows, str(reference_dev), results, auxiliary=aux, local=local)
    if reference_full256:
        if refs.get("c_full256_reference") and hashlib.sha256(reference_full256.read_bytes()).hexdigest() != refs["c_full256_reference"]:
            raise ValueError("Wrong full-256 C oracle SHA")
        report["matched_original_c_full256"] = original.compare_c_full256(
            model, windows, str(reference_full256), auxiliary=aux, local=local,
            probe_count=min(topk_probes, 128))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-stream", type=Path, required=True)
    p.add_argument("--dev-bin", type=Path, required=True)
    p.add_argument("--bpe-merges", type=Path,
                   default=Path("terminal_rsi/reference/nancm_v103_bpe.txt"))
    p.add_argument("--topk-probes", type=int, default=512)
    p.add_argument("--max-bpb-bytes", type=int, default=None)
    p.add_argument("--reference-dev", type=Path, default=None)
    p.add_argument("--reference-full256", type=Path, default=None)
    args = p.parse_args(argv)
    print(json.dumps(benchmark(
        checkpoint=args.checkpoint, train_stream=args.train_stream,
        dev_bin=args.dev_bin, bpe_merges=args.bpe_merges,
        topk_probes=args.topk_probes, max_bpb_bytes=args.max_bpb_bytes,
        reference_dev=args.reference_dev,
        reference_full256=args.reference_full256), indent=2,
        sort_keys=True))


if __name__ == "__main__":
    main()
