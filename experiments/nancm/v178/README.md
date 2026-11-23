# NaNCM v178 — Clone-free shared verifier hub (experimental)

**2026-11-23. Source implementation, default disabled.** Inherits the full v177 implementation and existing v176 probability law without modifying native weight files, the 128+128 S+/S− cloud, word specialist, BPE head, or optimizer histories.

## What is actually shared

One v176 full model handle owns one v170 byte predictor and one v175 word predictor. A `wf178_hub_open_full(...)` constructs exactly one inherited model. Each `wf178_session_open(hub,past,n)` owns only a history, five-word snapshot, and a bounded 256-node prefix cache. A session may change contexts independently with `wf178_session_begin`. Its `next`, `batch`, `top5`, `prepare_top5` and `sample` exports preserve the original v176 256-way P and Q (and seeded rejection sampling). The hub handle may be closed after session creation because sessions retain the shared model via reference counting. No model instance is copied for each session.

**Crucial limitation:** the inherited WF148→WF170 stack has mutable, request-dependent causal state. All oracle cache misses and model switches are therefore guarded by a single hub mutex, and a switch reinstates the correct session history via `wf176_hybrid_begin`. Only cached hits can execute concurrently without touching the oracle. This version establishes bounded-RAM multi-context correctness, **not** true simultaneous model inference, latency speedup, or a mathematically novel compression result. Serializing oracle misses and reinitializing on context switches can make interleaved cold requests slower. It is a compatibility bridge toward physically separating immutable tables from transient causal state.

## C ABI

`wf178_hub_open_full(teacher,model,geometry,cloud,bytepack,escape,crosslevel,bpe)` / `wf178_hub_close`; `wf178_hub_stats(hub,&model_instances,&live_sessions)`; `wf178_session_open(hub,past,n)` / `wf178_session_close` / `wf178_session_begin`; `wf178_session_next(session,prefix,n,q,p)`; `wf178_session_batch(session,bytes,bytes_len,offsets,lengths,count,outq,outp)`; `wf178_session_prepare_top5(session,scores5,&raw,&unique)`; `wf178_session_sample(session,seed,maxbytes,out,capacity,&written,&accepted,&rejected)`; `wf178_session_top5`; `wf178_session_stats`.

Up to 512 prefixes per batch; up to 40 bytes each; a maximum of 256 cached prefix distributions per session; up to 1,000,000 bytes of history. Caller must not concurrently close a session while its methods are running, or close the hub handle while opening a session. Concurrent methods on live sessions are guarded by per-session locks.

## Promotion gates

1. Source-only synthetic concurrency, reset, bounds, and full P/Q parity.
2. Real inherited sidecars on book-disjoint contexts: byte-for-byte 256-dimensional P/Q parity after alternating histories, top-five rank/probability parity, identical seeded speculative samples, full recovery of original byte ABI.
3. RSS delta (one hub + N sessions versus per-worker full model clones), p50/p95 cold/alternating/hot latency, throughput, and exact acceptance behavior.
4. Default promotion **only** after speed, memory and correctness gates. Do not equate one-model RSS with true parallel speed.

Use `make v178-source-smoke` to run source-only tests; `make v178 && python experiments/nancm/v178/test_trained178.py --n 12` for trained sidecars. GitHub contains source only; no trained sidecar is committed.
