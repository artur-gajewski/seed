"""Seed interchange plus atomic native PyTorch snapshots."""

import json
import os
import uuid
from pathlib import Path

import torch

from .model import Model


def load_interchange(path):
    saved = json.loads(Path(path).read_text())
    source = saved["model"]
    if (
        source.get("version") != 2
        or source.get("architecture") != "transformer"
        or source.get("tokenizer") != "utf8-bytes"
    ):
        raise ValueError("Only Seed Transformer v2 checkpoints are supported")
    model = Model(source["config"])
    step = source["step"]
    if type(step) is not int or step < 0:
        raise ValueError("Invalid optimizer step")
    optimizer = model.optimizer(0.0003)
    with torch.no_grad():
        for key, p in model.weights.items():
            item = source["parameters"][key.replace("__", ".")]
            if [item["rows"], item["columns"]] != list(p.shape):
                raise ValueError(f"Invalid shape: {key}")
            arrays = {}
            for field in ("values", "firstMoments", "secondMoments"):
                value = torch.tensor(item[field], dtype=p.dtype).reshape_as(p)
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite checkpoint field: {key}.{field}")
                arrays[field] = value
            if (arrays["secondMoments"] < 0).any():
                raise ValueError("Negative second moments")
            p.copy_(arrays["values"])
            optimizer.state[p] = {
                "step": torch.tensor(float(step)),
                "exp_avg": arrays["firstMoments"],
                "exp_avg_sq": arrays["secondMoments"],
            }
    model.step = step
    return model, optimizer


def export_interchange(path, model, optimizer):
    parameters = {}
    for key, p in model.weights.items():
        state = optimizer.state.get(p, {})
        parameters[key.replace("__", ".")] = {
            "rows": p.shape[0],
            "columns": p.shape[1],
            "values": p.detach().cpu().reshape(-1).tolist(),
            "firstMoments": state.get("exp_avg", torch.zeros_like(p)).cpu().reshape(-1).tolist(),
            "secondMoments": state.get("exp_avg_sq", torch.zeros_like(p))
            .cpu()
            .reshape(-1)
            .tolist(),
        }
    payload = {
        "model": {
            "version": 2,
            "architecture": "transformer",
            "tokenizer": "utf8-bytes",
            "config": model.config,
            "step": model.step,
            "parameters": parameters,
        }
    }
    with Path(path).open("x") as file:
        json.dump(payload, file)


def save(path, model, optimizer, session, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(
            {
                "format": "seed-torch-v1",
                "config": model.config,
                "step": model.step,
                "dtype": str(next(model.parameters()).dtype).split(".")[-1],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "session": session,
                "rng": torch.get_rng_state(),
            },
            temporary,
        )
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, device, dtype=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "seed-torch-v1":
        raise ValueError("Unsupported PyTorch checkpoint")
    model = Model(saved["config"], getattr(torch, saved["dtype"]))
    model.load_state_dict(saved["model"])
    model.to(device=device, dtype=dtype)
    model.step = saved["step"]
    optimizer = model.optimizer(0.0003)
    optimizer.load_state_dict(saved["optimizer"])
    return model, optimizer, saved
