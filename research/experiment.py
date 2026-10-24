"""Reproducible, isolated-process Seed architecture and inference pilot."""

import argparse
import copy
import hashlib
import io
import json
import math
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "python"))
from seed_torch.model import Model, encode_training

STEPS, CONTEXT, BATCH = 800, 128, 8
GRU_WIDTH, GRU_HIDDEN = 80, 180
CONFIG = {"hiddenSize": 80, "layers": 2, "heads": 4, "contextSize": CONTEXT}


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(258, GRU_WIDTH)
        self.rnn = nn.GRU(GRU_WIDTH, GRU_HIDDEN, batch_first=True)
        self.output = nn.Linear(GRU_HIDDEN, 258)

    def forward(self, x):
        return self.output(self.rnn(self.embedding(x))[0])

    def next(self, x, state=None):
        y, state = self.rnn(self.embedding(x), state)
        return self.output(y), state


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "white"]
    objects = [
        "car",
        "ball",
        "cup",
        "hat",
        "box",
        "boat",
        "kite",
        "book",
        "bag",
        "chair",
    ]
    sets = {k: [] for k in ["train", "validation", "test"]}
    pairs = {k: [] for k in sets}
    for i, c in enumerate(colors):
        for j, o in enumerate(objects):
            split = (
                "test"
                if (j + i) % 5 == 0
                else "validation"
                if (j + i) % 5 == 1
                else "train"
            )
            pairs[split].append([c, o])
            for other in colors:
                if other == c:
                    continue
                for family, q in [
                    ("direct", f"What color is the {c} {o}?"),
                    (
                        "distractor",
                        f"A {other} {objects[(j + 1) % len(objects)]} is beside a {c} {o}. What color is the {o}?",
                    ),
                    (
                        "update",
                        f"The {o} was {other}. Now the {o} is {c}. What color is it?",
                    ),
                ]:
                    row = {"family": family, "question": q, "answer": c}
                    if row not in sets[split]:
                        sets[split].append(row)
    for split, rows in sets.items():
        random.Random(918).shuffle(rows)
        dump(ROOT / f"tasks-{split}.json", rows)
        (ROOT / f"tasks-{split}.txt").write_text(
            "".join(
                f"User: {r['question']}\nAssistant: {r['answer']}\n\n" for r in rows
            )
        )
    source = ROOT.parent / "data/inputs/short_stories.txt"
    stories = sorted(
        {p.strip() for p in source.read_text().split("<|endofstory|>") if p.strip()}
    )
    story_sets = {k: [] for k in sets}
    for story in stories:
        bucket = int(hashlib.sha256(story.encode()).hexdigest(), 16) % 20
        split = "test" if bucket == 0 else "validation" if bucket == 1 else "train"
        story_sets[split].append(story)
    for split, rows in story_sets.items():
        (ROOT / f"prose-{split}.txt").write_text("\n<|endofstory|>\n".join(rows))
    files = list(ROOT.glob("tasks-*")) + list(ROOT.glob("prose-*"))
    dump(
        ROOT / "corpora.json",
        {
            "source": str(source.relative_to(ROOT.parent)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "pairs": pairs,
            "task_counts": {k: len(v) for k, v in sets.items()},
            "story_counts": {k: len(v) for k, v in story_sets.items()},
            "files": {
                p.name: {
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in files
            },
        },
    )


def tokens(name):
    return torch.tensor(encode_training((ROOT / name).read_text()), dtype=torch.long)


def batches(stream, generator, count=BATCH):
    starts = torch.randint(len(stream) - CONTEXT - 1, (count,), generator=generator)
    x = torch.stack([stream[s : s + CONTEXT] for s in starts])
    y = torch.stack([stream[s + 1 : s + CONTEXT + 1] for s in starts])
    return x, y


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def loss_on(model, stream, device):
    g = torch.Generator().manual_seed(991)
    losses = []
    for _ in range(8):
        x, y = batches(stream, g)
        losses.append(
            F.cross_entropy(
                model(x.to(device)).flatten(0, 1), y.to(device).flatten()
            ).item()
        )
    return statistics.mean(losses)


@torch.no_grad()
def generate(model, prompt, device, count=12):
    ids = [256, *prompt.encode()]
    x = torch.tensor([ids[-CONTEXT:]], device=device)
    state = None
    result = []
    for i in range(count):
        if isinstance(model, GRU):
            logits, state = model.next(x, state)
        else:
            logits = model(x)
        last = logits[0, -1].clone()
        last[256] = -float("inf")
        t = last.argmax().item()
        if t == 257:
            break
        result.append(t)
        ids.append(t)
        x = torch.tensor(
            [[t]] if isinstance(model, GRU) else [ids[-CONTEXT:]], device=device
        )
    return bytes(result).decode(errors="replace"), len(result)


@torch.no_grad()
def evaluate(model, device):
    model.eval()
    rows = json.loads((ROOT / "tasks-test.json").read_text())
    answers = []
    for r in rows:
        text, _ = generate(model, f"User: {r['question']}\nAssistant:", device)
        answer = text.strip().lower().strip(".!?")
        answers.append(dict(**r, generated=text, correct=answer == r["answer"]))
    families = {
        f: statistics.mean(r["correct"] for r in answers if r["family"] == f)
        for f in sorted({r["family"] for r in rows})
    }
    return {
        "task_loss": loss_on(model, tokens("tasks-test.txt"), device),
        "prose_loss": loss_on(model, tokens("prose-test.txt"), device),
        "accuracy": statistics.mean(r["correct"] for r in answers),
        "families": families,
    }, answers


@torch.no_grad()
def benchmark(model, device):
    model.eval()
    x = torch.zeros((1, CONTEXT), dtype=torch.long, device=device)
    for _ in range(5):
        model(x)
    forward = []
    generation = []
    # Fixed 32-step decoding, including special-token outputs, for comparable work.
    for _ in range(7):
        sync(device)
        start = time.perf_counter()
        for _ in range(20):
            model(x)
        sync(device)
        forward.append(20 * CONTEXT / (time.perf_counter() - start))
        sync(device)
        start = time.perf_counter()
        for _ in range(3):
            z = x[:, :64]
            state = None
            for _ in range(32):
                if isinstance(model, GRU):
                    logits, state = model.next(z, state)
                    z = logits[:, -1:].argmax(-1)
                else:
                    z = torch.cat([z, model(z)[:, -1:].argmax(-1)], 1)[:, -CONTEXT:]
        sync(device)
        generation.append(96 / (time.perf_counter() - start))
    return {
        "forward_tokens_s": statistics.median(forward),
        "decode_tokens_s": statistics.median(generation),
        "forward_repeats": forward,
        "decode_repeats": generation,
    }


def prune(model):
    model = copy.deepcopy(model)
    for layer in range(CONFIG["layers"]):
        prefix = f"block{layer}."
        expansion = model.weight(prefix + "feedForwardExpansion")
        projection = model.weight(prefix + "feedForwardProjection")
        score = expansion.square().sum(0) * projection.square().sum(1)
        keep = score.topk(score.numel() // 2).indices.sort().values
        for name, value in [
            ("feedForwardExpansion", expansion[:, keep]),
            (
                "feedForwardExpansionBias",
                model.weight(prefix + "feedForwardExpansionBias")[:, keep],
            ),
            ("feedForwardProjection", projection[keep]),
        ]:
            model.weights[model.key(prefix + name)] = nn.Parameter(
                value.detach().clone()
            )
    return model


def run(kind, seed, device):
    torch.set_num_threads(4)
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    torch.manual_seed(seed)
    start_wall = time.perf_counter()
    model = (
        Model(CONFIG, dtype=torch.float32, seed=seed)
        if kind == "transformer"
        else GRU()
    )
    model.to(device)
    count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    streams = [tokens("tasks-train.txt"), tokens("prose-train.txt")]
    validations = [tokens("tasks-validation.txt"), tokens("prose-validation.txt")]
    g = torch.Generator().manual_seed(seed)
    history = []
    elapsed = 0
    for step in range(1, STEPS + 1):
        x, y = batches(streams[(step - 1) % 2], g)
        x, y = x.to(device), y.to(device)
        sync(device)
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        lr = (
            0.002
            * min(step / 40, 1)
            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        sync(device)
        elapsed += time.perf_counter() - start
        if step == 1 or step % 100 == 0:
            model.eval()
            vals = [loss_on(model, v, device) for v in validations]
            history.append(
                {
                    "step": step,
                    "train_seconds": elapsed,
                    "task_loss": vals[0],
                    "prose_loss": vals[1],
                    "mixed_loss": statistics.mean(vals),
                }
            )
            print(json.dumps(dict(kind=kind, seed=seed, **history[-1])), flush=True)
    quality, answers = evaluate(model, device)
    timing = benchmark(model, device)
    mps_memory = (
        {
            "allocated_bytes_after_benchmark": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes_after_benchmark": torch.mps.driver_allocated_memory(),
        }
        if device == "mps"
        else None
    )
    checkpoint = ROOT / "checkpoints" / f"{kind}-{seed}.pt"
    checkpoint.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "architecture": kind,
            "config": CONFIG
            if kind == "transformer"
            else {"embedding": GRU_WIDTH, "hidden": GRU_HIDDEN},
            "seed": seed,
            "steps": STEPS,
            "state_dict": model.cpu().state_dict(),
        },
        checkpoint,
    )
    result = dict(
        kind=kind,
        seed=seed,
        device=device,
        parameters=count,
        training_seconds=elapsed,
        tokens_seen=STEPS * BATCH * CONTEXT,
        history=history,
        **quality,
        **timing,
    )
    blob = io.BytesIO()
    torch.save(model.state_dict(), blob)
    result["state_bytes"] = blob.tell()
    if mps_memory is not None:
        result["mps_memory"] = mps_memory
    if device == "mps" and seed == 42 and kind == "gru":
        cpu_quality, _ = evaluate(model, "cpu")
        result["cpu_parent"] = dict(
            kind="gru-float32-cpu",
            device="cpu",
            **cpu_quality,
            **benchmark(model, "cpu"),
        )
    dump(ROOT / f"answers-{kind}-{seed}.json", answers)
    variants = []
    if seed == 42:
        if kind == "transformer":
            candidates = [("pruned-50pct-ffn", prune(model))]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidates = [
                    (
                        "int8-gru",
                        torch.ao.quantization.quantize_dynamic(
                            model, {nn.GRU, nn.Linear}, dtype=torch.qint8
                        ),
                    )
                ]
        for label, variant in candidates:
            variant_device = device if kind == "transformer" else "cpu"
            variant.to(variant_device)
            q, a = evaluate(variant, variant_device)
            b = benchmark(variant, variant_device)
            variant.cpu()
            blob = io.BytesIO()
            torch.save(variant.state_dict(), blob)
            torch.save(variant.state_dict(), ROOT / "checkpoints" / f"{label}.pt")
            variants.append(
                dict(
                    kind=label,
                    parent=kind,
                    seed=seed,
                    device=variant_device,
                    state_bytes=blob.tell(),
                    **q,
                    **b,
                )
            )
            dump(ROOT / f"answers-{label}.json", a)
    result["variants"] = variants
    result["wall_seconds"] = time.perf_counter() - start_wall
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    dump(ROOT / f"run-{kind}-{seed}.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", choices=["transformer", "gru"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = p.parse_args()
    if args.worker:
        run(args.worker, args.seed, args.device)
        return
    prepare()
    dump(
        ROOT / "environment.json",
        {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "quantized_engines": torch.backends.quantized.supported_engines,
            "threads": 4,
            "steps": STEPS,
            "batch": BATCH,
            "context": CONTEXT,
            "device": args.device,
            "command": f".venv/bin/python research/experiment.py --device {args.device}",
        },
    )
    for seed in [42, 43, 44]:
        for kind in ["transformer", "gru"]:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--worker",
                    kind,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
