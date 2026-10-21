"""Standalone training and evaluation; the web studio remains on TypeScript."""

import argparse
import hashlib
import json
import math
import random
import re
import signal
import time
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from . import checkpoint
from .model import Model, encode_training
from .progress import Progress


def device_for(name, precision):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; run natively on a supported Mac")
    if name == "mps" and precision == "float64":
        raise ValueError("MPS requires float32")
    return torch.device(name)


def learning_rate(completed, total, peak):
    warmup = max(1, min(100, math.floor(total * 0.05)))
    progress = max(0, (completed - warmup) / max(1, total - warmup))
    multiplier = (
        (completed + 1) / warmup
        if completed < warmup
        else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    )
    return peak * multiplier


@torch.no_grad()
def validation(model, tokens):
    length = min(model.config["contextSize"], len(tokens) - 1)
    count = min(3, max(1, (len(tokens) - 1) // length))
    losses = []
    for index in range(count):
        start = index * (len(tokens) - length - 1) // max(1, count - 1)
        logits = model(tokens[start : start + length].unsqueeze(0))
        losses.append(
            F.cross_entropy(logits.reshape(-1, 258), tokens[start + 1 : start + length + 1]).item()
        )
    return sum(losses) / count


def train(args, device, dtype):
    if args.steps < 1 or not 1 <= args.batch_size <= 8 or not 2 <= args.context_window <= 256:
        raise ValueError("Require positive steps, batch size 1–8, and context 2–256")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not args.resume and Path(args.checkpoint).exists():
        raise ValueError("Checkpoint exists; choose a new path or use --resume")
    raw = Path(args.corpus).read_bytes()
    text = raw.decode("utf-8")
    if len(text) < 64 or len(raw) > 10 * 1024 * 1024:
        raise ValueError("Corpus must contain at least 64 characters and at most 10 MB")
    encoded = encode_training(text)
    split = math.floor(len(encoded) * 0.9)
    training = torch.tensor(encoded[:split], device=device)
    held_out = torch.tensor(encoded[split:], device=device)
    if len(held_out) < 2:
        raise ValueError("Corpus is too short for validation")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = random.Random(args.seed)
    if args.resume:
        model, optimizer, _ = checkpoint.load(args.checkpoint, device, dtype)
        if args.context_window > model.config["contextSize"]:
            raise ValueError("Context exceeds saved model context")
    else:
        model = Model(
            {
                "hiddenSize": args.embedding_width,
                "layers": args.layers,
                "heads": args.attention_heads,
                "contextSize": args.context_window,
            },
            dtype,
        ).to(device)
        optimizer = model.optimizer(args.learning_rate)
    length = min(args.context_window, model.config["contextSize"], len(training) - 1)
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    session = {
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "completed": 0,
        "total": args.steps,
        "peak_learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sequence_length": length,
    }
    started = time.perf_counter()
    progress = Progress()
    durations = deque(maxlen=10)
    last_reported = started
    validation_loss = None
    print(
        json.dumps(
            {
                "device": str(device),
                "precision": str(dtype),
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_step": model.step,
            }
        ),
        flush=True,
    )
    try:
        for completed in range(args.steps):
            if stopped:
                break
            update_started = time.perf_counter()
            starts = [sampler.randrange(len(training) - length) for _ in range(args.batch_size)]
            inputs = torch.stack([training[s : s + length] for s in starts])
            targets = torch.stack([training[s + 1 : s + length + 1] for s in starts])
            lr = learning_rate(completed, args.steps, args.learning_rate)
            loss = model.update(inputs, targets, optimizer, lr)
            durations.append(time.perf_counter() - update_started)
            session["completed"] = completed + 1
            if completed == 0 or (completed + 1) % 100 == 0 or completed + 1 == args.steps:
                validation_loss = validation(model, held_out)
            now = time.perf_counter()
            if (
                completed == 0
                or (completed + 1) % 100 == 0
                or completed + 1 == args.steps
                or stopped
                or now - last_reported >= 5
            ):
                remaining = args.steps - completed - 1
                progress.report(
                    {
                        "step": model.step,
                        "completed": completed + 1,
                        "total": args.steps,
                        "loss": loss,
                        "validationLoss": validation_loss,
                        "learningRate": lr,
                        "elapsedSeconds": now - started,
                        "estimatedRemainingSeconds": (
                            0
                            if remaining == 0
                            else sum(durations) / len(durations) * remaining
                            if len(durations) == 10
                            else None
                        ),
                    }
                )
                last_reported = now
            if model.step % 100 == 0:
                checkpoint.save(args.checkpoint, model, optimizer, session)
        checkpoint.save(args.checkpoint, model, optimizer, session)
    finally:
        progress.finish()
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if stopped:
        raise SystemExit(130)


def chat_prompt(messages):
    def format_message(message):
        role = "User" if message["role"] == "user" else "Assistant"
        return f"{role}: {message['content']}"

    prompt = format_message(messages[-1]) + "\nAssistant:"
    for i in range(len(messages) - 3, -1, -2):
        turn = format_message(messages[i]) + "\n" + format_message(messages[i + 1]) + "\n\n"
        if len(turn + prompt) > 4096:
            break
        prompt = turn + prompt
    return prompt


def evaluate(args, device, dtype):
    if not math.isfinite(args.temperature) or args.temperature <= 0 or not 0 <= args.top_k <= 257:
        raise ValueError("Require positive temperature and top-k 0–257")
    if Path(args.output).resolve() in (
        Path(args.evaluation).resolve(),
        Path(args.checkpoint).resolve(),
    ):
        raise ValueError("Report must differ from evaluation and checkpoint")
    model, _, _ = checkpoint.load(args.checkpoint, device, dtype)
    torch.manual_seed(args.seed)
    results, summary = [], {}
    for example in json.loads(Path(args.evaluation).read_text()):
        messages = example.get("history", []) + [{"role": "user", "content": example["question"]}]
        completion = model.generate(
            chat_prompt(messages), temperature=args.temperature, top_k=args.top_k
        )
        reply = re.split(r"\n(?:User|Assistant):", completion)[0].strip()
        passed = re.sub(r"[.!?]+$", "", reply.strip().lower()) == example["answer"]
        results.append({**example, "reply": reply, "passed": passed})
        group = summary.setdefault(example["family"], {"correct": 0, "total": 0})
        group["correct"] += int(passed)
        group["total"] += 1
    report = {
        "checkpoint": args.checkpoint,
        "evaluation": args.evaluation,
        "steps": model.step,
        "temperature": args.temperature,
        "topK": args.top_k,
        "seed": args.seed,
        "device": str(device),
        "precision": str(dtype),
        "summary": summary,
        "correct": sum(r["passed"] for r in results),
        "total": len(results),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "import", "export"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True)
        if name in ("import", "export"):
            sub.add_argument("--interchange", required=True)
            continue
        sub.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
        sub.add_argument("--precision", choices=["float32", "float64"], default="float32")
        sub.add_argument("--seed", type=int, default=42)
        if name == "train":
            sub.add_argument("--corpus", required=True)
            sub.add_argument("--resume", action="store_true")
            for flag, default in [
                ("embedding-width", 256),
                ("layers", 4),
                ("attention-heads", 8),
                ("context-window", 256),
                ("batch-size", 8),
                ("steps", 30000),
            ]:
                sub.add_argument("--" + flag, type=int, default=default)
            sub.add_argument("--learning-rate", type=float, default=0.0003)
        else:
            sub.add_argument(
                "--evaluation", default="data/references/color_reasoning_01_reference.json"
            )
            sub.add_argument("--output", required=True)
            sub.add_argument("--temperature", type=float, default=0.4)
            sub.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "import":
            if Path(args.checkpoint).exists():
                raise ValueError("Output checkpoint already exists")
            model, optimizer = checkpoint.load_interchange(args.interchange)
            checkpoint.save(args.checkpoint, model, optimizer, {})
        elif args.command == "export":
            model, optimizer, _ = checkpoint.load(args.checkpoint, "cpu")
            checkpoint.export_interchange(args.interchange, model, optimizer)
        else:
            device = device_for(args.device, args.precision)
            dtype = getattr(torch, args.precision)
            (train if args.command == "train" else evaluate)(args, device, dtype)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
