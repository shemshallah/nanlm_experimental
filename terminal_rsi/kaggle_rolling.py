"""Rolling Kaggle backup for a long Terminal-RSI run.

Copies the newest atomic ``last.pt`` plus train.log / launch.json / metrics.json
into a staging folder and versions the Kaggle dataset at most once per 10
minutes.  A final version is written when the trainer writes ``DONE``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--staging", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--title", default="TRSI 100M long run checkpoints")
    p.add_argument("--description", default="Rolling Terminal-RSI weights and logs.")
    p.add_argument("--interval", type=float, default=600.0)
    args = p.parse_args(argv)

    run = args.run_dir
    up = args.staging
    up.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = args.token
    meta = {
        "title": args.title,
        "id": args.dataset,
        "licenses": [{"name": "CC0-1.0"}],
        "description": args.description,
    }
    (up / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (up / "README.md").write_text(
        "Terminal-RSI rolling weights and logs.\n"
        "last.pt is atomically overwritten by the trainer; each Kaggle version\n"
        "contains the newest weights plus train.log, launch.json and metrics.\n")
    log = open(run / "upload.log", "a", buffering=1)

    def note(*values):
        print(time.strftime("%H:%M:%S"), *values, file=log, flush=True)

    def stable(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            s1, m1 = path.stat().st_size, path.stat().st_mtime
            if s1 < 10_000_000:
                return False
            time.sleep(8)
            s2, m2 = path.stat().st_size, path.stat().st_mtime
            return s1 == s2 and m1 == m2
        except FileNotFoundError:
            return False

    def copy_if(src: Path, dst: Path) -> None:
        if not src.is_file():
            return
        tmp = dst.with_name(dst.name + ".part")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    def upload(message: str) -> bool:
        created = (up / ".created").exists()
        for _ in range(2):
            mode = "version" if created else "create"
            cmd = [sys.executable, "-m", "kaggle", "datasets", mode,
                   "-p", str(up), "--dir-mode", "zip",
                   "--ignore-patterns", ".created,*.part"]
            if mode == "version":
                cmd += ["-m", message]
            result = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=3600)
            note("kaggle", mode, "rc", result.returncode,
                 (result.stdout + result.stderr)[-300:].replace("\n", " "))
            if result.returncode == 0:
                (up / ".created").write_text("1")
                return True
            created = True
        return False

    note("uploader start", args.dataset)
    last_key = None
    last_try = 0.0
    final = False
    while True:
        last_pt = run / "last.pt"
        if stable(last_pt):
            st = last_pt.stat()
            key = (st.st_size, int(st.st_mtime))
            if key != last_key and time.time() - last_try >= args.interval:
                copy_if(last_pt, up / "last.pt")
                for name in ("train.log", "launch.json", "metrics.json",
                             "RUN_SUMMARY.md"):
                    copy_if(run / name, up / name)
                if upload(f"TRSI weights/logs {time.strftime('%H:%M')}"):
                    last_key = key
                    last_try = time.time()
        if (run / "DONE").exists():
            if not final and stable(last_pt):
                copy_if(last_pt, up / "final.pt")
                copy_if(last_pt, up / "last.pt")
                for name in ("train.log", "launch.json", "metrics.json",
                             "RUN_SUMMARY.md"):
                    copy_if(run / name, up / name)
                upload("TRSI FINAL weights and logs")
                final = True
            if final:
                note("uploader done")
                break
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
