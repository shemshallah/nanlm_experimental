#!/usr/bin/env python3
"""Download a checkpoint from a Kaggle kernel output and stage it for release.

Uses the direct DownloadKernelOutput endpoint instead of the list endpoint,
which is rate-limited. Requires KAGGLE_API_TOKEN, KERNEL_SLUG, RUN_DIR and
ASSET_PREFIX in the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiDownloadKernelOutputRequest,
)

FILES = ("final.pt", "best.pt", "config.json", "result.json", "eval_ptb.json")


def main() -> int:
    owner, slug = os.environ["KERNEL_SLUG"].split("/", 1)
    run_dir = os.environ["RUN_DIR"]
    prefix = os.environ["ASSET_PREFIX"]
    out = Path("dist")
    out.mkdir(exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        for name in FILES + (f"{slug}.log",):
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug = owner
            request.kernel_slug = slug
            request.file_path = f"runs/{run_dir}/{name}"
            try:
                redirect = kaggle.kernels.kernels_api_client.download_kernel_output(
                    request
                )
            except Exception as exc:  # missing optional file
                print(f"skip {name}: {exc}", flush=True)
                continue
            target_name = "train.log" if name.endswith(".log") else name
            dest = out / f"{prefix}-{target_name}"
            total = 0
            with requests.get(redirect.url, stream=True, timeout=900) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as handle:
                    for chunk in resp.iter_content(1 << 20):
                        if chunk:
                            handle.write(chunk)
                            total += len(chunk)
            print(f"{dest} {total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
