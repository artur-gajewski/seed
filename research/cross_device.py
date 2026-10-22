"""Check whether the large Transformer's accuracy gap follows weights or inference device."""

import json
from pathlib import Path

import experiment as e
import torch

if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        raise RuntimeError("Run outside the sandbox to access MPS")
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    rows = []
    for trained_on, folder in [
        ("cpu", root / "full-scale"),
        ("mps", root / "mps/full-scale"),
    ]:
        checkpoint = torch.load(
            folder / "checkpoints/transformer-42.pt",
            map_location="cpu",
            weights_only=True,
        )
        for device in ["cpu", "mps"]:
            model = e.Model(checkpoint["config"], dtype=torch.float32, seed=42)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            quality, answers = e.evaluate(model, device)
            rows.append(dict(trained_on=trained_on, evaluated_on=device, **quality))
            e.dump(
                root / "mps" / f"cross-answers-{trained_on}-on-{device}.json", answers
            )
            print(json.dumps(rows[-1]), flush=True)
            del model
            torch.mps.empty_cache()
    e.dump(root / "mps/cross-device.json", rows)
