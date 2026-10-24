"""Exploratory one-seed check at approximately Seed's original parameter count."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import experiment as e

if __name__ == "__main__":
    destination = e.ROOT / "full-scale"
    destination.mkdir(exist_ok=True)
    if len(sys.argv) == 1:
        for pattern in ["tasks-*", "prose-*", "corpora.json", "environment.json"]:
            for path in e.ROOT.glob(pattern):
                shutil.copy2(path, destination / path.name)
        metadata = json.loads((destination / "environment.json").read_text())
        metadata["command"] = ".venv/bin/python research/scale_check.py"
        metadata["scope"] = (
            "exploratory one-seed original-parameter-scale check; context 128"
        )
        e.dump(destination / "environment.json", metadata)
        for kind in ["transformer", "gru"]:
            subprocess.run([sys.executable, str(Path(__file__)), kind], check=True)
    else:
        e.ROOT = destination
        e.CONFIG = {
            "hiddenSize": 256,
            "layers": 4,
            "heads": 8,
            "contextSize": e.CONTEXT,
        }
        e.GRU_WIDTH, e.GRU_HIDDEN = 256, 896
        e.run(sys.argv[1], 42, "cpu")
