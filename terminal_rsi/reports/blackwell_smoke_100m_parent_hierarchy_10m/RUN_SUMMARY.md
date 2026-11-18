# Blackwell 10-minute parent-warm-start + hierarchy smoke

- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition (94.97 GiB)
- Inherited v103 core: verified 2 BPB parent, SHA256 `d39d27ba...`
- Parent legacy step: 68,452
- Effective batch / microbatch: 2048 / 1024
- Main optimizer updates: **74**
- Updates/s: **0.1231**
- Training loop: 611 s
- Peak allocated / reserved: **66.21 / 93.45 GiB**
- v4: **36 cycles, 72 proposals, 0 accepted, 108 branch updates**
- Parallel replicas: 3 clones (neutral + 2 candidates)
- DEV 32-position probe: **2.2208 -> 2.2406 BPB** (small hard probe)
- Hierarchical DEV (1,024 positions): **1.9253 BPB**
- Byte: top1 **60.25%**, top5 **87.30%**
- Subword: top1 40.40%, top5 78.97%
- Token: top1 40.40%, top5 73.18%
- Word: top1 12.90%, top5 23.66%, exact top5 **45.70%**
- Sentence: mean per-byte top1 60.24%, top5 87.20%; exact 0
- Hierarchy gates: bottom-up `[0.0098, 0.0072, 0.0019]`, lateral `[-0.0130, -0.0118, -0.0115]`, top-down `[0.0164, 0.0098, 0.0002]`
- Generated 32 bytes from `"The "`: **"containe inch of having that all"**
- Latent word boundaries: 6/32; mean word-boundary probability 0.190

## Notes

The previous 10-minute from-scratch run reached 3.897 hierarchical BPB / 60.8% byte top-5.
Warm-starting the inherited v103 core from the verified parent moves the same hierarchy
to **1.925 BPB / 87.3% byte top-5** in 10 minutes. The remaining gap to 95% is training
budget: ~0.3 bits per 10 minutes from this warm start, so another 20-40 minutes should
reach the 1.5 BPB class.
