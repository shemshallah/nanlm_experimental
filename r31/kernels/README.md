# r31 kernels

Kaggle kernel scripts and metadata. These are reference implementations; the
kernel `id` fields point at the private `shem42/*` kernels. Change the id if
you want to run them under another account.

| Folder | Kaggle kernel | Status |
|---|---|---|
| `bench/` | `shem42/r31-bench` | completed; throughput/VRAM benchmark |
| `10min/` | `shem42/r31-10min` | plain baseline completed; 151M hybrid OOM (gather path) |
| `10min_v2/` | `shem42/r31-10min-v2` | completed; latest both-vs-geo result |
| `50min/` | `shem42/r31-compute-matched` | cancelled/deleted to respect a 10-minute limit |

`10min_v2` is the latest successful experiment. It uses the default `grouped`
MoE dispatch; the `gather` path in `10min/` OOMs at batch 32 for the 151M model.

To run one:

```sh
kaggle kernels push -p r31/kernels/10min_v2
kaggle kernels status shem42/r31-10min-v2
```

The kernels expect two private datasets mounted at `/kaggle/input`: the prepared
books and the r31 code bundle. See `../datasets/README.md`.
