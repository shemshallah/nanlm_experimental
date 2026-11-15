# Archived Poincaré-specific RSI adapter — not Terminal-RSI

This folder records an **optional historical experiment** in which a value
head, supervised rewrite, and on-policy refinement were attached to a small
Poincaré `MirrorLM`. It is neither the independent Terminal-RSI research track
nor a continuation of a trained FA checkpoint.

**The standalone track is [`../terminal_rsi/`](../terminal_rsi/README.md).**
Its actor, critic, optimizer, verifier, metrics, workflow and results do not
import, train, or evaluate FA1/FA2/Poincaré components.

For historical source and regression reproducibility only:

```bash
cd poincare_mirror
python -m unittest discover -s tests -p 'test_poincare_rsi_legacy*.py' -v
python scripts/poincare_rsi_legacy_pilot.py --sft-steps 80 --ppo-steps 32 \
  --out reports/poincare_rsi_legacy_toy_pilot.json
```

The legacy toy evaluation contains **only four held-out next-letter targets**.
It does not measure terminal command execution, Terminal-Bench completion,
independent books, generalization, or real FA checkpoint quality. Its historical
top-1/top-5 and bits/byte values must not be reused as Terminal-RSI metrics.
The model/optimizer in this test is initialized from scratch; any checkpoint
payload is unpromoted. No R38 source or checkpoint is required by this smoke test.

This adapter is preserved for reproducibility, but has no dependency
relationship with the standalone Terminal-RSI track.
