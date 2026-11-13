#!/usr/bin/env python3
"""Check whether the MTP branch actually uses the supplied t+1 token."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from v3_model import V3Config, V3LM


def load_state(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("model", obj) if isinstance(obj, dict) else obj
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="/kaggle/working/mtp_sensitivity.json")
    args = ap.parse_args()

    obj = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = obj["model_config"]
    cfg = V3Config(**raw_cfg)
    cfg.mtp_block = True
    cfg.mtp_block_experts = 8
    cfg.mtp_block_hidden = 1024
    cfg.mtp_adapter_rank = 0
    model = V3LM(cfg)
    missing, unexpected = model.load_state_dict(load_state(args.checkpoint), strict=False)
    print("missing", missing, "unexpected", unexpected, flush=True)
    model.eval()

    ids = np.fromfile(args.train_bin, dtype=np.uint16)
    starts = np.linspace(args.ctx, len(ids) - args.ctx - 3, args.batch_size, dtype=np.int64)
    x = torch.from_numpy(np.stack([ids[s - args.ctx:s] for s in starts]).astype(np.int64))
    next_ids = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    wrong_ids = torch.roll(next_ids, shifts=1, dims=1)
    result = {}
    with torch.no_grad():
        for name, nxt in (("correct", next_ids), ("wrong", wrong_ids)):
            _, mtp, _ = model(x, mtp_teacher=True, mtp_next_tokens=nxt)
            logits = mtp[0][:, :-2]
            target = x[:, 2:]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            result[name] = float(loss)
            print(name, float(loss), flush=True)
    result["delta_correct_minus_wrong"] = result["correct"] - result["wrong"]
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
