"""Repeat the CPU protocol on actual MPS, preserving all CPU artifacts."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import experiment as e
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", choices=["transformer", "gru"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", choices=["pilot", "full-scale"], default="pilot")
    args = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. On this host, run outside the execution sandbox."
        )
    destination = e.ROOT / "mps"
    if args.worker:
        e.ROOT = destination if args.scale == "pilot" else destination / "full-scale"
        if args.scale == "full-scale":
            e.CONFIG = {
                "hiddenSize": 256,
                "layers": 4,
                "heads": 8,
                "contextSize": e.CONTEXT,
            }
            e.GRU_WIDTH, e.GRU_HIDDEN = 256, 896
        e.run(args.worker, args.seed, "mps")
        return
    for folder in [destination, destination / "full-scale"]:
        folder.mkdir(parents=True, exist_ok=True)
        for pattern in ["tasks-*", "prose-*", "corpora.json"]:
            for source in e.ROOT.glob(pattern):
                shutil.copy2(source, folder / source.name)
        e.dump(
            folder / "environment.json",
            {
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "mps_available": True,
                "mps_built": torch.backends.mps.is_built(),
                "device": "mps",
                "threads": 4,
                "batch": e.BATCH,
                "context": e.CONTEXT,
                "steps": e.STEPS,
                "quantized_engines": torch.backends.quantized.supported_engines,
                "command": ".venv/bin/python research/mps_run.py",
                "execution": "outside sandbox",
                "cpu_corpus_manifest": json.loads(
                    (e.ROOT / "corpora.json").read_text()
                ),
            },
        )
    for scale, seeds in [("pilot", [42, 43, 44]), ("full-scale", [42])]:
        for seed in seeds:
            for kind in ["transformer", "gru"]:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__)),
                        "--worker",
                        kind,
                        "--seed",
                        str(seed),
                        "--scale",
                        scale,
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
