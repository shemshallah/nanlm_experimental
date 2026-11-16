"""One verified-action Terminal-RSI policy with a jointly trained neural RSI core.

The existing terminal actor/critic/verifier is the host. The byte predictor
scores only commands offered in a *trusted manifest*, receives the verified
advantage after execution, and shares one recoverable .pt checkpoint.
No generated command bypasses the existing independent execution verifier.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .terminal_agent import FeedbackPolicy, CommandTask, _sha
from .unified import PRESETS, RSIController, TerminalRSI, device_for, loss_and_stats
from .planner import TerminalPlan, run_plan


class UnifiedTerminalPolicy(FeedbackPolicy):
    """Single live actor + neural specialist + RSI critic + optimizer state."""

    def __init__(self, mode="cpu", seed=2026, *, pretrain=None):
        super().__init__(seed=seed)
        if mode not in PRESETS:
            raise ValueError("Unknown Terminal-RSI compute mode")
        self.cfg = PRESETS[mode]
        self.device = device_for(mode)
        self.neural = TerminalRSI(self.cfg).to(self.device)
        self.action_gate = torch.nn.Parameter(torch.tensor(-0.51, device=self.device))
        self.neural_optimizer = torch.optim.AdamW(list(self.neural.parameters()) + [self.action_gate], lr=self.cfg.lr)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda" and not torch.cuda.is_bf16_supported()))
        self.neural_rsi = RSIController(self.cfg)
        self.neural_updates = 0
        self.neural_history = []
        if pretrain:
            saved = torch.load(pretrain, map_location="cpu", weights_only=True)
            if saved.get("format") != "terminal-rsi-unified-v1" or saved.get("config") != vars(self.cfg):
                raise ValueError("Byte pretraining checkpoint not an exact architecture match")
            self.neural.load_state_dict(saved["model"], strict=True)
            # One extra trainable action gate: migrate all inherited model Adam
            # moments by parameter order; initialize ONLY the new gate state.
            prior = saved["optimizer"]
            current = self.neural_optimizer.state_dict()
            if len(prior["param_groups"]) != 1 or len(current["param_groups"]) != 1:
                raise ValueError("Expected one inherited optimizer group")
            old_ids = prior["param_groups"][0]["params"]
            new_ids = current["param_groups"][0]["params"]
            if len(old_ids) + 1 != len(new_ids):
                raise ValueError("Inherited optimizer parameter count mismatch")
            current["state"] = {new_id: prior["state"][old_id]
                                for old_id, new_id in zip(old_ids, new_ids)
                                if old_id in prior["state"]}
            for key, value in prior["param_groups"][0].items():
                if key != "params":
                    current["param_groups"][0][key] = value
            self.neural_optimizer.load_state_dict(current)
            self.neural_rsi.__dict__.update(saved["controller"])

    def _autocast(self):
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    def _raw_command(self, task: CommandTask, index: int):
        # Verifier's expected stdout MUST NOT enter the neural input.
        prefix = ("task:" + task.family + "\n").encode("utf-8")
        argv = task.choices[index]
        cmd = b" ".join(a.encode("utf-8", "strict") for a in argv)
        full = prefix + cmd + b"\n"
        # Bounded causal context; no byte beyond the current command.
        over = max(0, len(full) - min(self.cfg.context + 1, 256))
        data = full[over:]
        start_target = max(0, len(prefix) - 1 - over)
        return data, start_target

    def _candidate_score(self, task: CommandTask, index: int):
        data, start = self._raw_command(task, index)
        if len(data) < 3:
            raise ValueError("Command lacks sufficient causal prefix")
        raw = torch.tensor(list(data), device=self.device, dtype=torch.long)
        with self._autocast():
            out = self.neural(raw[:-1].unsqueeze(0), passes=self.neural_rsi.passes)
        logits = out["logits"][0, 0].float()
        ce = F.cross_entropy(logits, raw[1:], reduction="none")
        return -ce[start:].mean()

    def _scores(self, task: CommandTask, grad: bool):
        previous_mode = self.neural.training
        self.neural.train(grad)
        if grad:
            scores = torch.stack([self._candidate_score(task, j) for j in range(len(task.choices))])
        else:
            with torch.no_grad():
                scores = torch.stack([self._candidate_score(task, j) for j in range(len(task.choices))])
        self.neural.train(previous_mode)
        return scores

    def probabilities(self, task: CommandTask):
        row = self._row(task.family, len(task.choices))
        logits = np.asarray(row["logits"], dtype=np.float64)
        scores = self._scores(task, grad=False).detach().float().cpu().numpy().astype(np.float64)
        gate = float((2.0 * torch.sigmoid(self.action_gate.detach())).item())
        logits = logits + gate * np.clip(scores - scores.mean(), -6, 6)
        probs = np.exp(logits - logits.max())
        return probs / probs.sum()

    def observe(self, task: CommandTask, choice: int, success: bool, *, train=False, lr=.08):
        if not train:
            return super().observe(task, choice, success, train=False, lr=lr)
        # Capture policy from BEFORE actor/Adam modification (on-policy credit).
        previous_row = np.asarray(self._row(task.family, len(task.choices))["logits"]).copy()
        scores = self._scores(task, grad=True)
        logits = scores * (2.0 * torch.sigmoid(self.action_gate)) + torch.as_tensor(previous_row, device=self.device,
                                                               dtype=scores.dtype)
        advantage = super().observe(task, choice, success, train=True, lr=lr)
        # Verified advantage is the only source of success credit.
        policy_loss = -float(advantage) * F.log_softmax(logits, dim=0)[choice]
        # Self-supervised model training is strictly on *observed* command bytes.
        # The three future lookaheads are masked rather than fabricated.
        data, _ = self._raw_command(task, choice)
        suffix = torch.tensor(list(data) + [0, 0], dtype=torch.long,
                              device=self.device).unsqueeze(0)
        valid = torch.ones_like(suffix, dtype=torch.bool)
        valid[:, -2:] = False
        with self._autocast():
            out = self.neural(suffix[:, :-3], passes=self.neural_rsi.passes)
            language_loss, _ = loss_and_stats(out, suffix, valid)
        total_loss = policy_loss + 0.03 * language_loss
        if not torch.isfinite(total_loss):
            raise RuntimeError("Nonfinite live RSI update; refuse checkpoint promotion")
        self.neural_optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.unscale_(self.neural_optimizer)
        torch.nn.utils.clip_grad_norm_(list(self.neural.parameters()) + [self.action_gate], 1.0)
        self.grad_scaler.step(self.neural_optimizer)
        self.grad_scaler.update()
        self.neural_updates += 1
        self.neural_history.append({
            "step": self.neural_updates, "task": task.task_id,
            "verified_success": bool(success), "advantage": float(advantage),
            "policy_loss": float(policy_loss.detach().item()),
            "command_bpb_objective": float(language_loss.detach().item() / math.log(2)),
        })
        return advantage

    def save(self, path: Path):
        """Single checkpoint for actor/critic, neural core, RSI, and both optimizers."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        actor = super().state()
        state = {
            "format": "terminal-rsi-unified-agent-v1",
            "config": vars(self.cfg),
            "actor": actor,
            "actor_sha256": _sha(actor),
            "neural": self.neural.state_dict(),
            "neural_optimizer": self.neural_optimizer.state_dict(),
            "neural_controller": vars(self.neural_rsi).copy(),
            "action_gate": self.action_gate.detach().cpu(),
            "grad_scaler": self.grad_scaler.state_dict(),
            "neural_updates": self.neural_updates,
            "neural_history": self.neural_history,
        }
        # Atomic write. The existing actor JSON and prior LM .pt are untouched.
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".incomplete", delete=False) as output:
            temp = Path(output.name)
        try:
            torch.save(state, temp)
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

    @classmethod
    def load(cls, path: Path, *, mode="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "terminal-rsi-unified-agent-v1":
            raise ValueError("Wrong unified Terminal-RSI checkpoint format")
        if state.get("config") != vars(PRESETS[mode]):
            raise ValueError("CPU/T4/Blackwell architecture mismatch; refusing silent optimizer reset")
        if _sha(state["actor"]) != state["actor_sha256"]:
            raise ValueError("Verifier/actor receipt integrity failure")
        obj = cls(mode=mode, seed=state["actor"]["seed"])
        inherited = FeedbackPolicy.from_state(state["actor"])
        obj.rows = inherited.rows
        obj.total_updates = inherited.total_updates
        obj.receipt_digest = inherited.receipt_digest
        obj.episode = inherited.episode
        obj.episode_history = inherited.episode_history
        obj.neural.load_state_dict(state["neural"], strict=True)
        obj.neural_optimizer.load_state_dict(state["neural_optimizer"])
        obj.neural_rsi.__dict__.update(state["neural_controller"])
        obj.action_gate.data.copy_(state["action_gate"].to(obj.device))
        obj.grad_scaler.load_state_dict(state.get("grad_scaler", {}))
        obj.neural_updates = int(state["neural_updates"])
        obj.neural_history = list(state["neural_history"])
        return obj


def main():
    p = argparse.ArgumentParser(description="Verified Terminal-RSI with unified CPU/GPU RSI neural policy")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--mode", choices=PRESETS, default="cpu")
    p.add_argument("--byte-pretrain", type=Path,
                   help="Optional same-preset byte model+optimizer for fresh unified actor only")
    p.add_argument("--train", action="store_true")
    p.add_argument("--timeout", type=float, default=5.)
    p.add_argument("--max-commands", type=int)
    args = p.parse_args()
    plan = TerminalPlan.parse(json.loads(args.plan.read_text()))
    if args.checkpoint.exists():
        if args.byte_pretrain:
            p.error("--byte-pretrain cannot overwrite a resumed live actor")
        actor = UnifiedTerminalPolicy.load(args.checkpoint, mode=args.mode)
    else:
        actor = UnifiedTerminalPolicy(args.mode, pretrain=args.byte_pretrain)
    report = run_plan(plan, actor, args.root, args.checkpoint, train=args.train,
                      timeout=args.timeout, max_commands=args.max_commands)
    print(json.dumps({**report, "neural_updates": actor.neural_updates,
                      "model_parameters": actor.neural.parameter_count(),
                      "compute_mode": args.mode}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
