# Experiment branch consolidation on master — 2026-11-13

**Canonical development branch:** `master`. The commit(s) that add this ledger carry **all eleven branch tips as actual Git merge parents**, so `git merge-base --is-ancestor <tip> master` succeeds for every listed tip. No experiment development branch was created or checked out.

**Conflict policy:** The current, tested `master` engine, workflows, checkpoints, and FA1/FA2/Poincare lineage directories remain authoritative. Incompatible variant files are archived verbatim as Git blob references under `experiments/branch_archive/<branch>/...`; they are not executed by default. The complete original branch trees are reachable through the merge-parent history. Remote branch labels may continue to exist; their commits are already merged into `master`.

| Incoming historical branch | Immutable tip | Main retained content |
| --- | --- | --- |
| `fa1-checkpoint-transfer` | `75e04694c0de44552b4c8ec14cf2209da6aa7515` | Original historical 10 binary transfer parts remain reachable from merge parent, *not duplicated into active source* |
| `fa1-hierarchical-r713` | `135cdf034b7f27fbbb4bc86e1e1fdc7646a9c53a` | R7.13 README and historical hierarchical reports archived; ancestor source retained in original commit history |
| `fa2-r23-context64-parallel` | `bc29d2162477bc97c243d06f7f14f4d8818c3ca9` | R23 specialist/contextual source, tests, docs and workflow snapshots archived |
| `fa2-r23-continuous-state` | `61c880629cbe5b917ba2a8532a1b35c87df73ac4` | R23 specialist/contextual source, tests, docs and workflow snapshots archived |
| `sentence-allsent` | `9cf09ae47fbc91efd38138dceb9b7e08bc702b73` | Diamond sentence engine and CPU workflow variants archived |
| `sentence-cache` | `19a6bf8442f46353fe89530355a5f2d7d0ee5135` | Diamond sentence engine and CPU workflow variants archived |
| `sentence-deep` | `fefe48033c7bddd0901a036a7c1e6b7a8cb91e94` | Diamond sentence engine and CPU workflow variants archived |
| `sentence-full` | `7fe8d6bf9de24709c7764d352280117e4a41c5a6` | Diamond sentence engine and CPU workflow variants archived |
| `sentence-isolate` | `e5f013d48105977d81ecc68734249cf04c9567bb` | Diamond sentence engine and CPU workflow variants archived |
| `sentence-lookahead` | `8b0b97f639710dbdb50c06253a76f01ccdc74f2b` | Diamond sentence engine and CPU workflow variants archived |
| `sentence-prefix` | `2b4d65dd1f840229bfdf1510114a042585d732dd` | Diamond sentence engine and CPU workflow variants archived |

## Variant paths

- `experiments/branch_archive/sentence-*/diamond/diamond_engine.py`: alternative Diamond sentence predictors.
- `experiments/branch_archive/sentence-*/.github/workflows/diamond-1000.yml`: matching inactive workflow recipes.
- `experiments/branch_archive/fa2-r23-*/`: historical R23 modules, tests and docs; do not overwrite later FA2 modules.
- `experiments/branch_archive/fa1-hierarchical-r713/`: R7.13 historical reports and README.

## Operational rules

1. All new changes are committed to `master` only; no detached training or model promotion on the archived branches.
2. Keep **FA1 R7**, **FA2**, **Poincare**, **NaNLMind**, and **NaN16** research identities and optimizer histories distinct even though their source trees share this repository.
3. Do not average/mix checkpoints across different architectures. A master-level merger is a *source and history unification*, not automatic checkpoint fusion.
4. Training continues from the audited FA1 R7.27 unified optimizer candidate; the accepted step-14,416 FA1 checkpoint stays immutable until verified independent-book and relevant-memory gates pass.
5. Historical `.fa1_transfer` chunk binaries are retained through ancestry, not copied into the active master tree. This keeps sources compact while preserving their exact recovery commits.
