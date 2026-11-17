# Terminal-RSI Blackwell smoke

- Status: `completed`
- Budget: 600s
- Context / batch: 256 / 8192
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Free VRAM at preflight: 94.4 GiB
- Planned FP32 AdamW state: 1.5 GiB
- Planned activation floor: 16.50 GiB
- Updates: 23
- Training loop: 605.5s
- Updates/s: 0.038
- Peak allocated: 33.46 GiB
- Peak reserved: 91.461 GiB
- DEV before/after BPB: 8.188456547006338 -> 4.279429510911293
- byte: top1=0.2407 top5=0.5132 units=2048 bits/unit=4.9356 loss/unit=3.4211
- subword: top1=0.0562 top5=0.2969 units=1209 bits/unit=8.3607 loss/unit=5.7952
- token: top1=0.0562 top5=0.2969 units=1209 bits/unit=8.3607 loss/unit=5.7952
- word: top1=0.0000 top5=0.1072 units=373 bits/unit=24.0184 loss/unit=16.6483
- sentence: top1=0.0000 top5=0.0000 units=21 bits/unit=476.0913 loss/unit=330.0013
- byte BPB/BPC/loss: 4.9356 / 5.1310 / 3.4211

Console: `/marimo/terminal_rsi_100m_10m_final/blackwell_smoke_console.log`
Metrics: `/marimo/terminal_rsi_100m_10m_final/blackwell_smoke_metrics.json`
