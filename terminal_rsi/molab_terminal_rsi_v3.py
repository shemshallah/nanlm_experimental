"""Molab-ready Terminal-RSI CPU v3 notebook.

Open this Python file in Molab/marimo (not as a PyPI requirement).
Click Run to train exactly 1,000 updates on 1, 2, or 4 CPU threads.
The runner persists its full console log even on a failed training subprocess.
After completion, download the full log, metrics JSON, and a single
SHA-256-indexed checkpoint bundle directly through the notebook.

Only shemshallah/nanlm_experimental master / terminal_rsi is used.
The inherited original v103 checkpoint is NOT fabricated or overwritten.
This pilot explicitly begins from untrained v103-compatible initialization.
"""
import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """# Terminal-RSI v4 — 1,000 CPU updates

        This is a **new** English-only, book-disjoint NaNROUTER experiment.
        It does not recover or overwrite the historical 1.33-BPB weights.

        Choose the controller:

        - **v4** (recommended): telemetry-conditioned counterfactual branches.
          Every N updates it snapshots the model, runs 2-3 real short branches
          plus a no-op branch on the same data, scores 4 disjoint internal TRAIN
          shards, and keeps the best branch only when the paired BPB gain beats
          both a minimum and a paired standard-error threshold.  The v4 action
          space includes LR multipliers, gradient clipping, weight decay,
          residual/refinement gates, router temperature and batch size.
        - **v3**: the previous fixed ±0.08 gate bandit, retained as a baseline.
        - **off**: pure training, no RSI controller.

        Live output prints a compact **PROGRESS** line every 10 byte updates
        (step, percent, ETA, BPB, bit accuracy) plus the active RSI ledger.
        v4 also prints one **RSI-V4 CYCLE/BRANCH/CHOSEN** line per decision.

        Press **Run** once. Training output is saved even if the notebook
        cannot select or copy its printed output.
        """
    )
    return


@app.cell
def _(mo):
    cpu_cores = mo.ui.dropdown(
        options=["1", "2", "4"], value="4", label="CPU math threads"
    )
    rsi_mode = mo.ui.dropdown(
        options=["v4", "v3", "off"], value="v4", label="RSI controller"
    )
    launch = mo.ui.run_button(
        label="Run 1,000 CPU updates and save downloads", kind="success"
    )
    mo.vstack([cpu_cores, rsi_mode, launch])
    return (cpu_cores, rsi_mode, launch)


@app.cell
def _(mo):
    get_run_result, set_run_result = mo.state(None)
    return get_run_result, set_run_result


@app.cell
def _(cpu_cores, launch, mo, rsi_mode, set_run_result):
    mo.stop(not launch.value)

    def run_training():
        import importlib.util
        import shutil
        import subprocess
        import sys
        from datetime import datetime, timezone
        from pathlib import Path

        source_folder = Path(__file__).resolve().parent
        if (source_folder / "terminal_rsi" / "train_verbatim.py").is_file():
            repo = source_folder
        elif (source_folder.parent / "terminal_rsi" / "train_verbatim.py").is_file():
            repo = source_folder.parent
        else:
            repo = Path.cwd() / "nanlm_experimental"

        run_name = datetime.now(timezone.utc).strftime(
            "rsi_" + str(rsi_mode.value) + "_%Y%m%dT%H%M%S_%fZ")
        # Reserve output independently from the repo so an early bootstrap
        # failure still leaves a downloadable diagnostic log.
        output = Path.cwd() / "terminal_rsi_molab_runs" / run_name
        output.mkdir(parents=True, exist_ok=False)
        bootstrap_log = output / "molab_bootstrap.log"

        def run_checked(command, *, cwd=None):
            with bootstrap_log.open("a", encoding="utf-8") as log:
                log.write("RUN: " + " ".join(str(x) for x in command) + "\n")
                log.flush()
                with subprocess.Popen(
                    command, cwd=cwd, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                ) as child:
                    assert child.stdout is not None
                    for line in child.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = child.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, command)

        try:
            if not (repo / "terminal_rsi" / "train_verbatim.py").is_file():
                if repo.exists():
                    raise RuntimeError("Repository folder exists but terminal_rsi source is missing")
                run_checked([
                    "git", "clone", "--branch", "master", "--single-branch",
                    "https://github.com/shemshallah/nanlm_experimental.git", str(repo),
                ])
            elif (repo / ".git").is_dir():
                branch = subprocess.check_output(
                    ["git", "-C", str(repo), "branch", "--show-current"],
                    text=True,
                ).strip()
                if branch and branch != "master":
                    raise RuntimeError("Only master is permitted, not branch " + branch)
                run_checked(["git", "-C", str(repo), "pull", "--ff-only", "origin", "master"])

            if importlib.util.find_spec("torch") is None:
                raise RuntimeError(
                    "PyTorch is missing. Add torch using Molab's package editor; "
                    "do not add terminal-rsi to the PyPI list."
                )
            missing = [name for name in ("numpy", "zstandard")
                       if importlib.util.find_spec(name) is None]
            if missing:
                installer = (
                    ["uv", "pip", "install", "--python", sys.executable, *missing]
                    if shutil.which("uv") else
                    [sys.executable, "-m", "pip", "install", *missing]
                )
                run_checked(installer)
            run_checked([
                sys.executable, "-m", "terminal_rsi.run_molab_cpu_1000",
                "--cpu-threads", str(cpu_cores.value), "--updates", "1000",
                "--output-dir", str(output), "--topk-probes", "32",
                "--progress-every", "10",
                "--rsi-version", str(rsi_mode.value),
                "--rsi-v4-cycle-updates", "100",
                "--rsi-v4-horizon", "5",
                "--rsi-v4-candidates", "2",
                "--rsi-v4-shards", "4",
                "--rsi-v4-shard-positions", "128",
                "--rsi-v4-min-gain", "0.005",
                "--rsi-v4-z", "2.0",
            ], cwd=str(repo))
            return {"path": str(output), "status": "completed"}
        except Exception as error:
            with bootstrap_log.open("a", encoding="utf-8") as log:
                log.write("FAILED: " + repr(error) + "\n")
            return {"path": str(output), "status": f"failed: {error}"}

    set_run_result(run_training())
    return


@app.cell
def _(get_run_result, mo):
    from pathlib import Path

    run_result = get_run_result()
    if run_result is None:
        output_ui = mo.md("Awaiting a training run. Download buttons will appear here.")
    else:
        folder = Path(run_result["path"])
        names = [
            ("molab_bootstrap.log", "Download setup + launch log"),
            ("terminal_rsi_cpu_1000_console.log", "Download complete training log"),
            ("terminal_rsi_cpu_1000_metrics.json", "Download fresh CPU benchmarks"),
            ("terminal_rsi_cpu_1000_bundle.zip", "Download checkpoint + manifest ZIP"),
        ]
        buttons = [
            mo.download(
                data=lambda path=folder / name: path.read_bytes(),
                filename=name,
                mimetype=("application/zip" if name.endswith(".zip") else
                          "application/json" if name.endswith(".json") else "text/plain"),
                label=label,
            )
            for name, label in names if (folder / name).is_file()
        ]
        output_ui = mo.vstack([
            mo.md(f"**Run:** {run_result['status']}"),
            mo.md(f"**Files:** `{folder}`"),
            *buttons,
        ])
    output_ui
    return


if __name__ == "__main__":
    app.run()
