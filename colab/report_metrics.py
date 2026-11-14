#!/usr/bin/env python3
"""Print a compact, human-readable summary of a diamond metrics file."""
import argparse
import json
import math
import os


def bits(r, key):
    return float(r.get(key, float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="path to diamond_metrics.json")
    args = ap.parse_args()
    d = json.load(open(args.metrics))
    print(f"completed_epochs: {d.get('completed_epochs')}")
    for key in ("best", "final"):
        r = d.get(key, {})
        if not r:
            continue
        combined = r.get("combined")
        if not combined:
            combined = (
                bits(r, "byte_bits")
                + 0.5 * bits(r, "word_bits")
                + 0.3 * bits(r, "sent_bits")
            )
        print(f"\n{key.upper()} epoch {r.get('epoch')} combined {combined:.4f}")
        print(
            "  byte  bits %.4f  top1 %.4f  top5 %.4f"
            % (bits(r, "byte_bits"), bits(r, "byte_top1"), bits(r, "byte_top5"))
        )
        print(
            "  word  bits %.4f  top1 %.4f  top5 %.4f"
            % (bits(r, "word_bits"), bits(r, "word_top1"), bits(r, "word_top5"))
        )
        print(
            "  sent  bits %.4f  top1 %.4f  top5 %.4f"
            % (bits(r, "sent_bits"), bits(r, "sent_top1"), bits(r, "sent_top5"))
        )
    hist = os.path.join(os.path.dirname(args.metrics), "diamond_history.jsonl")
    if os.path.exists(hist):
        rows = [json.loads(x) for x in open(hist) if x.strip()]
        epochs = [r for r in rows if r.get("stage", "").startswith("epoch")]
        print("\nlast epochs:")
        for r in epochs[-10:]:
            print(
                "  %3s  sent %.4f  top1 %.3f  top5 %.3f  eval %.4f"
                % (
                    r.get("epoch"),
                    bits(r, "sent_bits"),
                    float(r.get("sent_top1", 0.0)),
                    float(r.get("sent_top5", 0.0)),
                    float(r.get("eval_loss", 0.0)),
                )
            )
    print("\nreference: frontier4 full-data sent 6.7817, top1 0.211, top5 0.380")


if __name__ == "__main__":
    main()
