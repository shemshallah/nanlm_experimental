import marimo

__generated_with = "0.9.0"
app = marimo.App(width="full")

REPO_URL = "https://github.com/shemshallah/nanlm_experimental.git"
WIKITEXT_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/{split}.txt"
)


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import time
    import urllib.request
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, os, subprocess, sys, time, urllib


@app.cell
def _(mo):
    mo.md(
        r"""
        # Z4 115M on Blackwell

        The measured `kaggle10m_nomtp` profile, ported from the T4 notebook:

        - 115,238,237 parameters
        - WikiText-2, ctx 512, batch 4
        - 600-second training cap
        - concept loop + feedback bus, MTP compute skipped
        - **bf16 autocast on Blackwell**, fp16 on T4

        Press **Run training**.  The first run also clones the repo,
        downloads WikiText-2, and builds the token bins.  The test evaluation
        runs in the same cell after training.
        """
    )
    return


@app.cell
def _(mo):
    preset = mo.ui.dropdown(
        options=["kaggle10m_nomtp", "kaggle10m_small", "kaggle10m"],
        value="kaggle10m_nomtp",
        label="Preset",
    )
    batch = mo.ui.number(
        value=4, start=1, stop=64, step=1, label="Batch size"
    )
    ctx = mo.ui.number(
        value=512, start=64, stop=4096, step=64, label="Context"
    )
    max_seconds = mo.ui.number(
        value=600, start=60, stop=7200, step=60, label="Training cap (s)"
    )
    amp_dtype = mo.ui.dropdown(
        options=["auto", "bf16", "fp16"], value="auto", label="AMP dtype"
    )
    no_ckpt = mo.ui.checkbox(
        value=True, label="Disable gradient checkpointing"
    )
    run = mo.ui.run_button(label="Run training")

    mo.vstack(
        [
            mo.hstack([preset, batch, ctx, max_seconds, amp_dtype]),
            mo.hstack([no_ckpt, run]),
        ]
    )
    return amp_dtype, batch, ctx, max_seconds, no_ckpt, preset, run


@app.cell
def _(mo, subprocess, sys):
    lines = []
    commands = [
        [sys.executable, "-V"],
        [
            sys.executable,
            "-c",
            "import torch; print('torch', torch.__version__); "
            "print('cuda', torch.version.cuda, torch.cuda.is_available()); "
            "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
            "print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)",
        ],
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    ]
    for command in commands:
        lines.append("$ " + " ".join(command))
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=90
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            lines.append(output.strip() or f"(exit {completed.returncode})")
        except Exception as exc:
            lines.append(f"{type(exc).__name__}: {exc}")
    mo.md("```\n" + "\n".join(lines) + "\n```")
    return


@app.cell
def _(
    Path,
    amp_dtype,
    batch,
    ctx,
    json,
    max_seconds,
    mo,
    no_ckpt,
    preset,
    run,
    subprocess,
    sys,
    urllib,
):
    mo.stop(
        not run.value,
        mo.md("Set the options above and press **Run training**."),
    )

    try:
        base = Path.cwd() / "z4_blackwell"
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = Path.home() / "z4_blackwell"
        base.mkdir(parents=True, exist_ok=True)

    def find_repo():
        try:
            here = Path(__file__).resolve()
            for parent in [here.parent, *here.parents]:
                if (parent / "z3" / "train_v3.py").is_file():
                    return parent
        except NameError:
            pass
        repo = base / "nanlm_experimental"
        if not (repo / "z3" / "train_v3.py").is_file():
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(repo)],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "-C", str(repo), "pull", "--ff-only"], check=True
            )
        return repo

    repo = find_repo()
    for module in ("tokenizers", "pyarrow"):
        try:
            __import__(module)
        except ImportError:
            print(f"installing {module}...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", module],
                check=True,
            )
    data = repo / "data"
    bins = data / "wikitext_bins"
    if not (bins / "train.bin").is_file():
        data.mkdir(parents=True, exist_ok=True)
        parts = []
        for split in ("train", "valid"):
            dest = data / f"wikitext_{split}.txt"
            if not dest.is_file():
                urllib.request.urlretrieve(
                    WIKITEXT_URL.format(split=split), dest
                )
            parts.append(dest.read_text(encoding="utf-8"))
        (data / "wikitext_corpus.txt").write_text(
            "\n\n".join(parts), encoding="utf-8"
        )
        print("building WikiText-2 bins...", flush=True)
        subprocess.run(
            [
                sys.executable,
                "z3/build_domain_bins.py",
                "--corpus",
                "wikitext=data/wikitext_corpus.txt",
                "--out",
                "data/wikitext_bins",
                "--max-bytes-per-domain",
                "12000000",
                "--train-fraction",
                "0.98",
                "--valid-fraction",
                "0.01",
            ],
            cwd=repo,
            check=True,
        )

    out = (repo / "runs" / "z4_115m_blackwell").resolve()
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "z3/train_v3.py",
        "--device",
        "cuda",
        "--amp",
        "--amp-dtype",
        amp_dtype.value,
        "--preset",
        preset.value,
        "--corpus",
        "data/wikitext_bins/train.bin",
        "--val-corpus",
        "data/wikitext_bins/valid.bin",
        "--data-manifest",
        "data/wikitext_bins/manifest.json",
        "--batch-size",
        str(int(batch.value)),
        "--ctx",
        str(int(ctx.value)),
        "--max-seconds",
        str(float(max_seconds.value)),
        "--out",
        str(out),
    ]
    if no_ckpt.value:
        cmd.append("--no-grad-checkpoint")

    print("$ " + " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"training failed with exit code {code}")

    result = json.loads((out / "result.json").read_text())
    best = (result.get("best") or {}).get("metrics") or {}
    final = result.get("final") or {}
    print(
        "\n[done] steps={} tokens={} best_bpb={:.4f} status={}".format(
            result.get("steps_completed"),
            result.get("tokens_seen"),
            float(best.get("bits_per_byte", float("nan"))),
            result.get("status"),
        ),
        flush=True,
    )

    ckpt = out / "best.pt"
    if not ckpt.is_file():
        ckpt = out / "final.pt"
    eval_cmd = [
        sys.executable,
        "z3/eval_domains.py",
        "--checkpoint",
        str(ckpt),
        "--test-dir",
        "data/wikitext_bins/test",
        "--data-manifest",
        "data/wikitext_bins/manifest.json",
        "--tokenizer",
        "layered",
        "--document-framing",
        "--sidecar",
        "--require-sidecar",
        "--ctx",
        str(int(ctx.value)),
        "--batch-size",
        str(int(batch.value)),
        "--device",
        "cuda",
        "--out",
        str(out / "test_domains.json"),
    ]
    print("$ " + " ".join(eval_cmd), flush=True)
    eval_process = subprocess.Popen(
        eval_cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert eval_process.stdout is not None
    for line in eval_process.stdout:
        print(line, end="", flush=True)
    eval_code = eval_process.wait()
    if eval_code != 0:
        raise RuntimeError(f"test evaluation failed with exit code {eval_code}")

    test = json.loads((out / "test_domains.json").read_text())
    rows = {
        "parameters": result.get("params"),
        "steps": result.get("steps_completed"),
        "tokens_seen": result.get("tokens_seen"),
        "wall_seconds": result.get("wall_seconds"),
        "best_step": (result.get("best") or {}).get("step"),
        "best_val_bpb": best.get("bits_per_byte"),
        "final_val_bpb": final.get("bits_per_byte"),
        "test_bpb": test.get("macro_bits_per_byte"),
        "nonfinite_skips": result.get("nonfinite_skips"),
    }
    print("\n" + "\n".join(f"{key}: {value}" for key, value in rows.items()))
    return


if __name__ == "__main__":
    app.run()
