# Terminal-RSI v4: counterfactual self-modification

`terminal_rsi/rsi_v4.py` replaces the v3 fixed `±0.08` gate bandit with a
telemetry-conditioned, counterfactual training controller.  v3 remains
available as a baseline; v4 is opt-in.

**v4 is the unified execution path for the exact same graph as v3.**  CPU, T4
and Blackwell are execution profiles, not model sizes: all three have the same
**6,495,162 parameters** (2,782,515 v103 core + 3,712,646 upgrades + 1 residual
gate).  v4 changes the controller/wiring; gate-zero v103 parity and CPU↔CUDA
functional parity are enforced by tests.  Blackwell adds BF16 autocast only to
the upgrade branch, native SDPA, the GPU window cache and large-batch/v4 branch
search.

## What v4 does differently

| v3 | v4 |
|---|---|
| 10 fixed `±0.08` gate arms | 8 typed primitives with 32 joint choices |
| 16 fixed internal probe positions | 4+ disjoint stratified TRAIN shards |
| Immediate probe BPB reward | Real short-horizon branch outcome |
| No no-op counterfactual | A no-op branch runs from the same snapshot |
| Stateless scalar critic | Telemetry-conditioned ensemble critic + world model |
| No proposal learning | Factorized proposal policy + replay buffer |
| Critic only scores | Critic/world model shortlist and rank proposals |
| Fixed acceptance threshold | Paired BPB gain > max(`min_gain`, `z·SE`) |
| No cost accounting | Branch compute is recorded and reported |

The v4 action space covers:

- global LR multiplier,
- upgrade-group LR multiplier,
- gradient clipping,
- upgrade-group weight decay,
- residual/capacity gate delta,
- recurrent refinement-scale delta,
- router temperature,
- batch size.

These are deliberately high-leverage knobs.  Arbitrary Python source rewriting
is not allowed.

## Branch evaluation

At every v4 cycle:

1. Capture an exact snapshot of model, optimizer, sampler cursor and RNG.
2. Restore the snapshot and run the no-op branch for `horizon` real updates.
3. Restore the same snapshot and run each proposed candidate for the same
   `horizon` updates on the same data sequence.
4. Score every branch on the same disjoint internal TRAIN shards.
5. Accept the best candidate only if its paired BPB gain beats both:
   - the economic minimum `--rsi-v4-min-gain`, and
   - `--rsi-v4-z` paired standard errors.
6. Restore the winning branch and continue training from it.
7. Add every branch outcome to the replay buffer, then train the ensemble
   critic and proposal policy.

The no-op branch is itself real training: if no candidate is accepted, its
trajectory is kept.  Losing branches are discarded.  This is a small beam search
over training trajectories, not a probe-only mutation.

Once the replay buffer has enough outcomes, the policy proposes a larger pool
of actions.  The ensemble critic predicts each action's utility and uncertainty,
the world model predicts its next-step BPB / gradient norm / bit accuracy, and
only the highest-ranked candidates are actually run.  This is how the critic
starts suggesting useful interventions instead of merely scoring random ones.

`branch_updates` in the logs and metrics is the total extra compute spent on
losing/winning branches.  Kept updates and total compute are reported
separately.

## CLI flags

```text
--rsi-version {v3,v4,off}          default: v3
--rsi-v4-cycle-updates N           default: 200
--rsi-v4-horizon N                 default: 10
--rsi-v4-candidates N              default: 2
--rsi-v4-shards N                  default: 4
--rsi-v4-shard-positions N         default: 256
--rsi-v4-min-gain BPB              default: 0.01
--rsi-v4-z N                       default: 2.0
--rsi-v4-meta PATH                 optional replay JSON or full .pt meta prior
--rsi-v4-critic-steps N            default: 32
--rsi-v4-policy-steps N            default: 16
--rsi-v4-entropy N                 default: 0.01
```

For the 1,000-update CPU pilot, a reasonable starting point is:

```text
--rsi-version v4
--rsi-v4-cycle-updates 100
--rsi-v4-horizon 5
--rsi-v4-candidates 2
--rsi-v4-shards 4
--rsi-v4-shard-positions 128
--rsi-v4-min-gain 0.005
--rsi-v4-z 2.0
```

This tests 3 branches (no-op + 2 candidates) for 5 updates every 100 kept
updates.  It costs about 15% extra optimizer updates and answers a real
counterfactual question.  For longer runs, use `cycle-updates 200`,
`horizon 10` and `shard-positions 256`.

The Molab notebook exposes a `v4 / v3 / off` dropdown and defaults to v4.

## A/A noise audit

Before trusting any acceptance threshold, measure the probe noise on an existing
checkpoint:

```bash
python -m terminal_rsi.run_rsi_ab aa \
  --checkpoint path/to/candidate.pt \
  --train-stream path/to/train_cpu_pilot.bin \
  --bpe-merges terminal_rsi/reference/nancm_v103_bpe.txt \
  --shards 4 --positions 256 --repeats 20
```

It reports mean BPB, standard deviation across shards, and the standard
deviation of paired half-shard differences.  If that noise is much larger than
`--rsi-v4-min-gain`, raise the shard count/positions or the minimum gain.

## A/B training comparison

```bash
python -m terminal_rsi.run_rsi_ab ab \
  --base-run path/to/existing_cpu_run \
  --arms off,v3,v4 \
  --seeds 2026,2027,2028 \
  --updates 1000 \
  --output ab_results
```

`--base-run` must contain `train_cpu_pilot.bin`, `win_dev.bin` and
`win_dev.manifest.json`.  `--dry-run` prints the exact commands without running
them.  The report contains final BPB, paired change, updates/s and RSI accepted
counts, plus a Markdown table.

## Meta-learning across runs

Every v4 run writes:

- `rsi_v4_replay.json` — human-readable replay, counters and events.
- `rsi_v4_meta.pt` — full controller state (policy, critic, world model,
  optimizers, replay, telemetry and counters).

Pass the `.pt` file to a later run:

```bash
python -m terminal_rsi.train_verbatim ... \
  --rsi-version v4 --rsi-v4-meta path/to/previous/rsi_v4_meta.pt
```

The new run warm-starts the proposal policy, critic, world model and replay,
then continues collecting counterfactual outcomes.  The A/A/A-B harness remains
the arbiter of whether the transferred prior actually helps.

## Safety and interpretation

- v4 uses only internal TRAIN shards for decisions.  DEV remains a final
  benchmark and TEST remains sealed.
- A branch snapshot restores model, all optimizer state, sampler cursor and RNG.
- `candidate.pt` is unpromoted.  Champion promotion still follows the existing
  `champion_guard` policy.
- The controller is a training tool, not evidence of AGI or arbitrary
  self-rewriting.  It changes bounded training hyperparameters and gate values.
- A/B is mandatory: if v4 does not beat `off` by a statistically meaningful
  margin after accounting for branch compute, turn it off.
