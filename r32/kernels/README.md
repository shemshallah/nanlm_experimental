# r32 kernels

## `pilot/` — `shem42/r32-multidomain-pilot`

Latest successful r32 run (2026-11-12):

- 16-domain prebuilt corpus
- Hybrid 151M total / ~21M active on GPU 1
- Plain 21M dense on GPU 0
- 220 steps × batch 32 × context 256 = 1.8M positions per arm
- Per-domain probe on 131,072 held-out positions
- ~8 minutes wall on 2× T4

Kernel: https://www.kaggle.com/code/shem42/r32-multidomain-pilot

The kernel expects both datasets mounted:

- `shem42/r32-code`
- `shem42/r32-multidomain-16domains`

Run:

```sh
kaggle kernels push -p r32/kernels/pilot
kaggle kernels status shem42/r32-multidomain-pilot
```

Change the `id` in `kernel-metadata.json` to run under another account.

## Notes

- Domain runs use `--context 257 --train-context 256`; the extra token is the
  domain prefix.
- The 151M additive MoE uses the default `grouped` dispatch. The `gather` path
  OOMs on a 14.56 GiB T4 at batch 32.
- The legacy `evaluate` command is not domain-token aware; use
  `code/probe_r32.py`.
