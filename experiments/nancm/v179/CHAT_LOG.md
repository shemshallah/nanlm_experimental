# v179 research-turn log

User: "continue", from v178 shared-hub model, 32-book exact parity and 19.97x warm cached-prefix speed but no single-stream throughput win.

Assistant: Audited model lifetime; implemented source-only verified Linux copy-on-write per-worker process inference with exact P/Q, extended to independent multi-context top-five verification, then full-model parallel word+byte+speculative execution. Confirmed trained 48-book exact values and 3 separate four-worker throughput runs. Source-only and cumulative checkpoints preserved; promotion restricted to opt-in multi-context CPU inference.
