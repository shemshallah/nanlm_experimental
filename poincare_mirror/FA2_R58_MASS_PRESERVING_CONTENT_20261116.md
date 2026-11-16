# FA2 R58 — Mass-preserving, address-gated content specialist

**2026-11-16 · same unified FA2 model and AdamW lineage · `master` · no promotion**

## Architectural change

R55–R57 established a tradeoff: the global historical successor-memory trust
gate could lower total CE mainly via easy spaces, or improve non-space CE at
the cost of overall prediction. R58 changes the prediction factorization
rather than adding an independent language model.

The inherited R55 general successor gate remains fully attached with its
four trained parameters. R58 appends **one zero-initialized four-parameter
content trust rail**. It observes the same strictly causal, C(256,2)-addressed
successor histogram but learns a separate correction to the conditional
probabilities of 255 literal bytes excluding the literal space (32) and
EOS (256). The existing shared byte head and every other geometric module
remain unchanged.

If `l_v` is the inherited R55 logit and `e_v` is causal successor evidence,
the R58 content candidate is `q_v=l_v+g_content*e_v` for bytes `v != 32,256`.
The final logit is

```
l'_v = q_v - (logsumexp_content(q) - logsumexp_content(l))
```

while `l'_32=l_32` and `l'_256=l_256`. Thus the total unnormalized probability
mass for non-space literal bytes is conserved, so the existing probabilities
of literal space and EOS are retained. Only the distribution *within* the
content-byte class changes. No future byte enters the read: matching history
at `u<t` uses only the already observed successor at `u+1<=t`.

The inherited general gate and the new conditional gate have **separate
query/value negative-control hooks**. R58 wrong-address controls permute
only the conditional query addresses (not historical stored keys), and
wrong-successor controls rotate only observed content. The inherited R55
gate always remains exact during R58 controls.

## Checkpoint lineage and verification

R58 continued the real, unpromoted R55 step-3,089 checkpoint with SHA-256
`b552291ca0a0f583c4d13da91b768d783f0923c1a796d11060052545d76c5c16`.
The newly attached conditional weight was initialized at zero; complete
next-byte logits were **bit-for-bit identical before training**. A single
existing AdamW parameter group was extended by one tensor. All **195**
inherited Adam states, inherited tensors and trained R55 gate parameters
were preserved through 96 TRAIN-only CPU updates, reaching step **3,185**.
The new conditional rail is the only parameter receiving gradient updates.

Local tests validate bit-exact zero migration, strict future masking,
preserved space/EOS logits, preserved conditional content probability mass,
negative controls that do not perturb the inherited R55 gate, gradient flow,
complete optimizer roundtrip, and failure-safe restoration. Source CI is a
separate engineering gate and should not be reported as book prediction gain.

## Measured DEV results — 12 unseen books, 6,144 target positions

All arms use eight 64-position windows per DEV book. Whole-book bootstrap
has 2,000 resamples with within-experiment Bonferroni correction across
three contrasts.

| Condition | Next CE ↓ | Non-space CE ↓ | Bits/raw byte ↓ | Top-1 ↑ | Top-5 ↑ |
|---|---:|---:|---:|---:|---:|
| Inherited R55 (new gate neutral) | **2.351676** | **2.617146** | **3.392752** | 35.742% | 70.785% |
| R58 exact conditional memory | 2.352050 | 2.617591 | 3.393291 | **35.775%** | **70.866%** |
| R58 wrong conditional address | 2.351653 | 2.617118 | 3.392719 | 35.742% | 70.785% |
| R58 wrong conditional values | 2.352159 | 2.617744 | 3.393448 | 35.693% | 70.817% |

R58 exact versus inherited R55: overall CE gain is **−0.000374** nats/target
with adjusted 95% interval [−0.002468, +0.001092]. The non-space CE gain
is **−0.000445** [−0.003052, +0.001400]. Both point estimates are worse and
both intervals include zero. The R58 exact-versus-wrong-address overall
advantage is −0.000396 [−0.002471, +0.001059]. The exact-versus-wrong-value
advantage is +0.000109 [−0.000768, +0.001038]. Neither identifies a
reliable address-specific or content-specific gain.

The R58 content correction changes next-byte ranking only. The inherited
+2 and +3 CE scores remain exactly 2.889248 and 3.070085 on these windows.
The parent space CE of 1.112641 is unchanged to rounding (approximately
2.9e-8 difference), consistent with mass conservation.

## Decision and next test

**No promotion. TEST remains sealed.** R58 solves the structural
space/content interference while retaining the same model and optimizer
history, but the existing exact pair-address retrieval appears too diffuse
to give a robust non-space improvement. A stronger next experiment is a
strictly causal **joint key** combining exact geometric pair identity with
observed current-byte/short-prefix identity, comparing it against both
wrong-address and wrong-successor controls. Add it as another zero-init
capacity to the same checkpoint, never as a replacement model. Any proposed
promotion still requires predeclared, independently book-disjoint TEST and
long-context regression gates.

The full report and reproducible code are in
`results/FA2_R58_CONTENT_DEV_20261116.json` and
`scripts/train_r55_pair_successor.py` (`--conditional-content`).
