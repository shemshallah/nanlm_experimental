# NaNCM v165 — Trainable 128+128 Signed Energy Cloud (research candidate)

Author: Justin Howard-Stanley · Independent Researcher · Santa Cruz, California, USA.

**Status: native implementation and training verified, predictive improvement NOT established; not promoted.** v164 remains the stronger measured experimental whole-word decoder. Keep v158 as active protected development baseline and v148 as fallback. The v164 trained 500-book distribution, its inherited cloud, geometry, network, native models, and original optimizers are unchanged.

## Architecture change

A genuinely **trainable** oriented signed cloud over 128 lower and 128 upper concepts is initialized from the inherited C(256,2) S+/S− edges. Each completed past word addresses a lower point; each candidate next word addresses an upper point (stable FNV-1a indexing). This learns separate excitatory S+ and inhibitory S− directed pair energies plus trainable lower/upper 8D coordinates. The preceding one to three completed words contribute with causal recency weights 0.2/0.3/0.5. Trainable cloud energy reweights the 16-word whole-word distribution of v164 using an exponential normalized energy, and v164's original 27-way conditional prefix/EOS mixture consumes it unchanged. No future target bytes enter `wf148_begin`; rollback is exact with `V165_SIGNED_ENERGY=0`.

`Q165(w|h) = Q164(w|h)*exp(beta E_{+/-}(h,w)) / sum_v Q164(v|h)*exp(beta E_{+/-}(h,v))`

`P165(w|h) = (1-alpha164)P162(w|h) + alpha164 Q165(w|h)`

The learned parameters are stored in `checkpoints/v165_signed_cloud.bin` (~1 MB), separately from v164's immutable core weights. `checkpoints/v165_signed_cloud_train.pt` stores both training weights and their matching optimizer state. This is a genuinely trained additional 128+128 signed-cloud pathway; **the inherited v155 cloud itself has NOT been overwritten or retrained**. Future work would require training new core-cloud weights against native word prediction rather than simply reweighting the proposal.

## TRAIN-only pilot

700 AdamW updates; 180,152 multi-candidate context groups drawn from the v164 500-TRAIN-book index (733,069 original rows), split by context key into 162,136 training and 18,016 validation groups. No DEV/TEST examples were used for training. This context-level split does **not** ensure book-disjointness and is a diagnostic only.

Validation conditional cross entropy: 2.997567 → 2.208525 nats (inherited untrained cloud 2.997316). Both S+ and S− changed. However modal top-one agreement inside this *TRAIN-derived diagnostic* decreased from 44.17% to 34.42%.

## Actual native independent-book DEV result

On 480 deterministic complete words from 24 DEV books with v164 settings frozen, spelling/EOS bits/symbol **worsened** at every tested strength: v164 rollback 2.880214; beta .1 2.880779; .25 2.882804; .5 2.888548; 1.0 2.903277. This is **not** full 256-byte BPB.

On a smaller 120-word / 12-book 96-beam DEV probe: v164 top-1/top-5/top-32 = 14/25/43 hits, trained beta=.5 = 13/23/45. This improvement in top-32 alone is insufficient for promotion. A five-strength 480-word beam sweep was interrupted after book 17/24 and no result is claimed from it.

## Build and test from this complete archive

```sh
make -B v165
OPENBLAS_NUM_THREADS=1 make test-v165
# exact rollback: V165_SIGNED_ENERGY=0
# enable trained research scorer:
export V165_SIGNED_ENERGY=.5
export V165_CLOUD_INDEX="$PWD/checkpoints/v165_signed_cloud.bin"
```

For standalone integration, first configure v164's existing base/model/optimizer and domain settings as in `README_v164.md`. `test-v165` includes 150 exact v164 parity vectors, active signed-cloud loading, normalized native prefix probabilities, and fail-closed invalid sidecar/strength/fingerprint. Also rerun `make test-v164` before deployment. Independent DEV loss is negative, so v165 must remain optional.

## Unresolved gates

- Train signed core-cloud weights using *native target conditional likelihood*, with book-disjoint holdout and full optimizer lineage.
- Improve the quality of the **leading word**; top-32 recovery alone is not the target.
- Test full 256-byte BPB, next/+2/+3, full 24+ book strict word top-k, unrelated-history ablations and Android/Termux.
- Certify near-duplicate book separation and inherited model provenance before final claims.
