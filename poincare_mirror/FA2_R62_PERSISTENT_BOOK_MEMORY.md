# FA2 R62 — Persistent causal history across book windows

Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA
2026-11-16 · nanlm_experimental/master · FA2 successor-memory lineage

## Solving the remaining disconnected memory path

R54 through R61 operate on `(B,T,E)` geometric address tensors for *one*
model-forward context. R61 indexes the current window efficiently, but
**cannot remember keys from an earlier window**. A 32K lookup horizon does
nothing if only 64 input positions were actually supplied.

R62 introduces `CausalBookSuccessor`, a bounded, persistent memory owned by
one book's stream. It stores (canonical pair address, observed byte and
optional predecessor, observed successor) indexed by exact pair and by
256-point endpoints. It supports all five prior key modes and isolated
query-address, query-byte, and successor-value controls.

At input position t:

1. The newly **observed** x[t] finalizes the pending historical successor for
   u=t-1. No future label or x[t+1] is read.
2. Old entries u<t-horizon expire.
3. The query (geometric pair at t, observed x[t]) consults only finalized
   records from the same book.
4. An exact pair+byte match takes precedence over endpoint+byte fallback.
5. The current record t is left pending until x[t+1] is observed later.

A clear book identifier is required, and explicit `begin_book(book_id)`
resets all previous memory. Cross-book history cannot be silently imported.
Memory size is bounded in the configured horizon and edge count, rather
than in the full document length.

The indexed book stream can be partitioned into arbitrary chunks without
changing its output (floating tolerance 1e-6) and can be restored through
a portable `state_dict` snapshot. The original model's neural and Adam
tensors are NEVER replaced.

## Neural feedback integration

The existing FA2 `PairSuccessorTrust` now accepts an **optional external
joint-memory read** containing histograms, coverage, and hit positions
aligned to B,T. It validates shapes, finite values and probability mass,
then computes the inherited R59/R60 content rail's four features and
mass-preserving non-space correction.

The same override is accepted by `MirrorLM.forward(...,
joint_memory_override=(hist,coverage,hits))`. The R55 and R58 inherited
rails still read their unchanged local history; only the already existing
joint rail consumes external history. No additional trainable parameters
or optimizer groups are introduced.

R62 is opt-in. All old calls and predictions remain unchanged.

## What is and is not demonstrated

New tests cover exact parity with R61's in-window vectorized oracle,
book boundaries, suffix causality, wrong-query/value interventions, snapshot
replay, retrieval beyond a single input chunk, direct nonzero trainable
joint-rail feedback and strict model state identity.

**This is not yet an end-to-end real-book FA2 training result.**
A future checkpoint-backed sequential sampler must feed actual inherited
geometric addresses from the trained FA2 model into the book stream, align
stream histories to the model's predictions, and evaluate DEV by whole
books. Arbitrary shuffled windows may NOT reuse a live cross-book or future
stream. The user-protected step3185 R58 full checkpoint remains unchanged.

No FA1, M3 or Terminal-RSI weights were imported.
