#!/usr/bin/env python3
"""Produce current R7.6 release notes and auditable curve from completed 10k run."""
import json,pathlib,sys,hashlib

def main(root):
 root=pathlib.Path(root)
 p=root/'reports/R7p6_10k_results.json'
 r=json.loads(p.read_text()); assert r['completed_updates']==10000 and r['end_joint_step']==14000
 logs=[]
 for row in (root/'reports/R7p6_training_metrics.jsonl').read_text().splitlines():
  try:x=json.loads(row)
  except json.JSONDecodeError: continue
  if x.get('phase')=='EVAL' and 6000<=x.get('joint_step',0)<=14000:logs.append(x)
 steps=sorted({x['joint_step'] for x in logs})
 assert steps==[6000,8000,10000,12000,14000],steps
 curve={'historical_only':True,'screened_new_documents':False,'before_R76':{'joint_step':4000,'dev':r['before_dev'],'test':r['before_test']},
        'comparisons':[{k:x[k] for k in ('joint_step','dev','test')} for x in logs]}
 (root/'reports/R7p6_loss_curve.json').write_text(json.dumps(curve,indent=2)+'\n')
 base=curve['before_R76'];last=logs[-1]
 fmt=lambda n:f'{n:.6f}'
 table=['| Joint step | Dev NLL | Test NLL | Pair-off test NLL | All-retrieval-off test NLL | Pair benefit | Total retrieval benefit |',
        '|---:|---:|---:|---:|---:|---:|---:|']
 for x in [base]+logs:
  t=x['test'];d=x['dev'];table.append('| '+ ' | '.join([str(x['joint_step']),fmt(d['active_nll']),fmt(t['active_nll']),fmt(t['pair_off_nll']),fmt(t['retrieval_off_nll']),fmt(t['incremental_pair_gain']),fmt(t['total_retrieval_gain'])])+' |')
 t=last['test'];d=last['dev'];bt=base['test'];bd=base['dev']
 ckhash=r['checkpoint_sha256']
 readme=f'''# PAL R7.6 — Unified paired associative memory, step 14,000

**One active checkpoint:** `checkpoint/pal_active.pt`. Joint updates: **14,000** total; **10,000 R7.6 updates** after a reproducible R7.5 step-3,500→4,000 replay. **Five optimizer histories** retained and continued. No trained R6/R7 trunk, temporal mirror, recurrent matrix, or existing retrieval weights were reinitialized.

Checkpoint SHA-256: `{ckhash}`

## Architecture

- Original R7.4 trunk + trainable 2,048-word context memory, rank-64 bidirectional mirror rails S+/S− within one shared architecture and 6,144-word older recurrent memory.
- R7.5 exact single-word associative retrieval uses the preceding word as query and candidate following words in the same strictly older 6,144-word FIFO.
- R7.6 adds **ordered two-word exact keys**, retrieves the next word from the *same* strictly older FIFO, and uses the existing NOW-conditioned gate network with a newly learned pair-strength and pair-bias calibration. This is not a separate model or second checkpoint.
- The two new trainable pair parameters are added as a second group within the **existing fifth retrieval AdamW optimizer**. All other optimizer histories, state and RNG were migrated from step 4,000 without reset. The original model's predictions are reproduced exactly with `pair_off=True`.
- Two-word lookup compares `(history[j],history[j+1])` against the previous two observed words, and retrieves `history[j+2]` for `j` in the 6,144-token older prefix. It cannot read the current target or the 2,048-token exact recent window.

## Historical fixed 512-position evaluation (whole-word NLL, nats, lower is better)

{chr(10).join(table)}

Step-4,000 original **R7.5 single-word retrieval** historical test NLL was **5.864036**, without the initially enabled R7.6 pair route. Pair-ready R7.6 before-training statistics in this table have the pair route activated. The observed R7.6 before→after test change was {fmt(bt['active_nll']-t['active_nll'])} nats and development change was {fmt(bd['active_nll']-d['active_nll'])} nats.

Pair opportunities on the 512 historical test positions: {t['pair_positions']} positions had at least one older matching two-word key; {t['pair_target_hits']} positions had the exact target among its retrieved values. These are memory-hit counts, **not** language-model accuracy.

## Run and reproduce

Requirements: Python 3, `numpy`, `torch`, `sentencepiece`. The checkpoint includes all five optimizer states and training RNG. The release also includes tokenizer, the historical corpus binaries, source and reports (no superseded checkpoints).

```sh
python -m pip install numpy torch sentencepiece
export PYTHONPATH=source OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
python source/test_r76.py --root .
python source/r76_pair_training.py --root . --resume checkpoint/pal_active.pt --output checkpoint/pal_active.pt --steps 500 --streams 2 --seq-len 2 --eval 512 --threads 2
```

Logs: `reports/R7p6_10k_run.log`, `reports/R7p6_training_metrics.jsonl`, `reports/R7p6_loss_curve.json`. Preserved original-source R7.5 reference reports can be found in `reports/R7p5*`.

## Important qualifications

The development and test windows were historically consulted, not independent new-book screenings; no new documents were evaluated or trained from GitHub during these 10,000 updates. Historical split document boundary information is incomplete; older context may cross document boundaries. Regression/causal checks support correct **index-level** isolation but are not a distribution-shift test.

The rank-64 older recurrent-memory path remains nearly neutral in the independent `outer_off` ablation; gains primarily demonstrate exact associative retrieval and continued trunk learning. These experiments do not establish fluent generation, standard-benchmark superiority or a frontier trajectory by themselves.
'''
 notes=f'''# PAL R7.6 — 10,000 continuation updates completed

- Original R7.5 step-3,500 archive reloaded and exactly replayed 500 steps. Step-4,000 test retrieval-enabled NLL: 5.864035729; retrieval-disabled 5.952270042.
- Continued the same integrated PAL checkpoint for **10,000 R7.6 updates** from step 4,000 to step 14,000.
- R7.6 uses two-word ordered exact association keys and next-word values with the same causal older FIFO as R7.5, with only two additional parameters and no sixth optimizer.
- Archived all model/optimizer/RNG states, source, tokenizer, train/dev/test corpus and evaluation results into a single complete release. Original evidence and historical reports preserved.
- Historical 512-position test NLL with both retrieval paths: {fmt(bt['active_nll'])} → {fmt(t['active_nll'])}; development {fmt(bd['active_nll'])} → {fmt(d['active_nll'])}.
- Step 14,000 retrieval-off NLL {fmt(t['retrieval_off_nll'])}; one-word-only {fmt(t['pair_off_nll'])}; both retrieval paths {fmt(t['active_nll'])}.
- Retrieval advantage {fmt(t['total_retrieval_gain'])} and incremental pair advantage {fmt(t['incremental_pair_gain'])} nats per token (ablation relative to identical trained weights).
- No independently screened new texts, standard language benchmarks or generation tests. Historical test overlap possible; older recurrent mirror contribution remains unproven.
'''
 claims=f'''# PAL R7.6 claim ledger

| Claim | Evidence | Status |
|---|---|---|
| One continued checkpoint at joint step 14,000 and 5 optimizer histories | checkpoint metadata and exact SHA-256 `{ckhash}` | Verified |
| 10,000 R7.6 updates after reproduced step 4,000 | replay log, training log, loss curve and checkpoint `joint_steps` | Verified |
| Pair retrieval is exact ordered, strictly older and label-causal | `source/test_r76.py` synthetic and live checks | Verified on controlled cases |
| Original predictions recovered when pair retrieval disabled | `pair_off` regression and R7.5 code | Verified |
| Two-word retrieval improves historical 512-position test at final checkpoint | Final ablation +{fmt(t['incremental_pair_gain'])} nats | Historical only |
| Full retrieval improves the historical test | Final ablation +{fmt(t['total_retrieval_gain'])} nats | Historical only |
| New-book generalization or fluent free generation | No independent new-book data / generation audit | Not established |
| Rank-64 outer recurrent matrix yields useful independent gain | `outer_off` effect {fmt(t['outer_off_nll']-t['active_nll'])} | Not established |
| Frontier AI capability | No comparable scale or standard benchmark results | Not established |
'''
 (root/'README.md').write_text(readme)
 (root/'RELEASE_NOTES.md').write_text(notes)
 (root/'CLAIM_LEDGER.md').write_text(claims)
 # Record the wider, matched historical evaluation when available. This is
 # descriptive only: no parameters or hyperparameters are tuned on these splits.
 t4096=root/'reports/R7p6_test_historical_4096.json'
 d4096=root/'reports/R7p6_dev_historical_4096.json'
 if t4096.is_file() and d4096.is_file():
  large={sp:json.loads(fp.read_text())['metrics'] for sp,fp in [('test',t4096),('dev',d4096)]}
  historical="""
## Larger fixed 4,096-position historical audit

| Split | No retrieval | One-word only | Both retrieval routes | Incremental pair gain | Total retrieval gain |
|---|---:|---:|---:|---:|---:|
"""
  for name in ('dev','test'):
   v=large[name]
   historical+=f"| {name} | {fmt(v['retrieval_off_nll'])} | {fmt(v['pair_off_nll'])} | **{fmt(v['active_nll'])}** | {fmt(v['incremental_pair_gain'])} | {fmt(v['total_retrieval_gain'])} |\n"
  historical+='\n**These are historically consulted positions, not an independently screened holdout.** The nonzero pair benefit survives a wider sample, but is not evidence of out-of-distribution generalization.\n'
  with (root/'README.md').open('a') as f:f.write('\n'+historical)
  with (root/'RELEASE_NOTES.md').open('a') as f:f.write('\n'+historical)
  with (root/'CLAIM_LEDGER.md').open('a') as f:f.write('\nWider 4,096-position audit: test incremental pair gain '+fmt(large['test']['incremental_pair_gain'])+', dev gain '+fmt(large['dev']['incremental_pair_gain'])+' nats; both historical only.\n')
 print(json.dumps({'rows':len(table)-2,'joint_step':r['end_joint_step'],'test_nll':t['active_nll'],'test_pair_gain':t['incremental_pair_gain'],'test_total_retrieval_gain':t['total_retrieval_gain'],'dev_nll':d['active_nll']},indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '.')
