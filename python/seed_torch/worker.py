"""Versioned JSON-lines worker. All model operations run on one owning thread."""

import copy
import hashlib
import json
import logging
import math
import os
import queue
import random
import re
import secrets
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch
from torch.nn import functional as F

from . import checkpoint
from .cli import device_for, learning_rate, validation
from .model import Model, encode_training

logger = logging.getLogger(__name__)


def emit(message):
    print(json.dumps(message, allow_nan=False), flush=True)


class Studio:
    def __init__(self):
        self.device = device_for(os.environ.get("SEED_DEVICE", "auto"), "float32")
        self.model = self.optimizer = self.job = self.settings = self.training = None
        self.history = []
        self.error = None
        self.saved_step = None
        self.checkpoint_path = None
        self.corpus_sha256 = None
        self.lives = {}
        self.durations = deque(maxlen=10)
        self.last_event = 0

    def status(self):
        model = None
        if self.model is not None:
            c = self.model.config
            model = {
                "hiddenSize": c["hiddenSize"],
                "vocabulary": 258,
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "steps": self.model.step,
                "architecture": "transformer",
                "contextSize": c["contextSize"],
                "layers": c["layers"],
                "heads": c["heads"],
                "tokenizer": "utf8-bytes",
            }
            if self.saved_step == self.model.step and self.checkpoint_path:
                model["checkpointFileName"] = Path(self.checkpoint_path).name
        return {
            "running": self.training is not None,
            "model": model,
            "job": self.job,
            "history": self.history,
            "error": self.error,
            **({"trainingSettings": self.settings} if self.settings is not None else {}),
            "backend": {
                "engine": "pytorch",
                "device": str(self.device),
                "precision": "float32",
                "devices": ["cpu"]
                + (["mps"] if torch.backends.mps.is_available() else [])
                + (["cuda"] if torch.cuda.is_available() else []),
            },
        }

    def save(self):
        if self.model is None:
            raise ValueError("No model to save")
        checkpoint.save(
            self.checkpoint_path,
            self.model,
            self.optimizer,
            {
                "studio": True,
                "corpus_sha256": self.corpus_sha256,
                "history": self.history,
                "trainingSettings": self.settings,
                "job": self.job,
            },
        )
        self.saved_step = self.model.step

    def stop(self):
        if self.training is not None:
            self.training = None
            self.job["finishedAt"] = time.time() * 1000
            self.save()

    def start(self, body, path):
        if self.training is not None:
            raise ValueError("Training is already running")
        device = device_for(body.get("device", str(self.device)), "float32")
        if body["resume"] and self.model is None:
            raise ValueError("No model to resume")
        tokens = encode_training(body["text"])
        split = int(len(tokens) * 0.9)
        training = torch.tensor(tokens[:split], device=device)
        held_out = torch.tensor(tokens[split:], device=device)
        if len(held_out) < 2:
            raise ValueError("Corpus is too short for validation")
        if body["resume"]:
            model = self.model
            if device != self.device:
                raise ValueError(
                    "Keep the current device when continuing; save and load to change it"
                )
            optimizer = self.optimizer
        else:
            model = Model(
                {
                    "hiddenSize": body["hiddenSize"],
                    "layers": body["layers"],
                    "heads": body["heads"],
                    "contextSize": body["sequenceLength"],
                },
                torch.float32,
            ).to(device)
            optimizer = model.optimizer(body["learningRate"])
        self.device, self.model, self.optimizer = device, model, optimizer
        self.checkpoint_path = path
        self.saved_step = None
        self.error = None
        if not body["resume"]:
            self.history = []
        self.settings = {
            key: value for key, value in body.items() if key not in ("text", "resume", "device")
        }
        self.settings.update(
            {
                "hiddenSize": model.config["hiddenSize"],
                "layers": model.config["layers"],
                "heads": model.config["heads"],
                "sequenceLength": min(body["sequenceLength"], model.config["contextSize"]),
            }
        )
        self.job = {
            "completed": 0,
            "total": body["steps"],
            "startedAt": time.time() * 1000,
            "loss": None,
            "tokens": len(tokens),
            "trainingTokens": split,
            "validationTokens": len(tokens) - split,
            "batchSize": body["batchSize"],
            "sequenceLength": self.settings["sequenceLength"],
            "learningRate": body["learningRate"],
        }
        self.corpus_sha256 = hashlib.sha256(body["text"].encode("utf-8")).hexdigest()
        self.training = (training, held_out)
        self.durations.clear()

    def update(self):
        started = time.perf_counter()
        training, held_out = self.training
        job = self.job
        length = min(job["sequenceLength"], len(training) - 1)
        starts = [random.randrange(len(training) - length) for _ in range(job["batchSize"])]
        x = torch.stack([training[s : s + length] for s in starts])
        y = torch.stack([training[s + 1 : s + length + 1] for s in starts])
        lr = learning_rate(job["completed"], job["total"], job["learningRate"])
        loss = self.model.update(x, y, self.optimizer, lr)
        self.durations.append(time.perf_counter() - started)
        job["loss"] = loss if job["loss"] is None else 0.95 * job["loss"] + 0.05 * loss
        job["completed"] += 1
        job["effectiveLearningRate"] = lr
        remaining = job["total"] - job["completed"]
        if len(self.durations) == 10:
            job["estimatedRemainingMs"] = sum(self.durations) / 10 * remaining * 1000
        validate = job["completed"] == 1 or job["completed"] % 100 == 0 or remaining == 0
        if validate:
            job["validationLoss"] = validation(self.model, held_out)
        if validate or job["completed"] % 10 == 0:
            point = {"step": self.model.step, "loss": job["loss"]}
            if "validationLoss" in job:
                point["validationLoss"] = job["validationLoss"]
            self.history = (self.history + [point])[-2000:]
        if self.model.step % 100 == 0:
            self.save()
        if remaining == 0:
            self.stop()
        now = time.perf_counter()
        if validate or now - self.last_event >= 0.2:
            emit({"event": "status", "status": self.status()})
            self.last_event = now

    def load(self, path, interchange=None):
        if self.training is not None:
            raise ValueError("Stop training before loading a checkpoint")
        if interchange:
            model, optimizer = checkpoint.load_interchange(interchange)
            model.to(device=self.device, dtype=torch.float32)
            # Rebind optimizer state through load_state_dict to cast moments to the device.
            state = optimizer.state_dict()
            optimizer = model.optimizer(0.0003)
            optimizer.load_state_dict(state)
            source = json.loads(Path(interchange).read_text())
            session = {
                "history": source.get("history", []),
                "trainingSettings": source.get("trainingSettings"),
            }
        else:
            model, optimizer, saved = checkpoint.load(path, self.device, torch.float32)
            session = saved.get("session", {})
        settings = session.get("trainingSettings") or {
            "hiddenSize": model.config["hiddenSize"],
            "layers": model.config["layers"],
            "heads": model.config["heads"],
            "sequenceLength": model.config["contextSize"],
            "steps": session.get("total", 30000),
            "batchSize": session.get("batch_size", 8),
            "learningRate": session.get("peak_learning_rate", 0.0003),
        }
        history = session.get("history", [])
        if not isinstance(history, list):
            raise TypeError("Invalid loss history")
        for point in history:
            if (
                not isinstance(point, dict)
                or type(point.get("step")) is not int
                or point["step"] < 0
                or any(
                    type(point.get(key)) not in (int, float) or not math.isfinite(point[key])
                    for key in ("loss",)
                    + (("validationLoss",) if "validationLoss" in point else ())
                )
            ):
                raise ValueError("Invalid loss history point")
        if interchange:
            checkpoint.save(
                path,
                model,
                optimizer,
                {"studio": True, "history": history[-2000:], "trainingSettings": settings},
            )
        self.model, self.optimizer = model, optimizer
        self.checkpoint_path, self.settings = path, settings
        self.history, self.job, self.error = history[-2000:], None, None
        self.saved_step = model.step
        self.corpus_sha256 = session.get("corpus_sha256")

    def life(self, body):
        size, path, config = body["size"], body["path"], body["config"]
        if size not in self.lives:
            lessons = []
            if Path(path).exists():
                model, optimizer, saved = checkpoint.load(path, self.device, torch.float32)
                lessons = saved.get("session", {}).get("lessons")
                if not isinstance(lessons, list) or len(lessons) > 1000:
                    raise ValueError("Invalid New life checkpoint")
            elif body.get("interchange"):
                model, optimizer = checkpoint.load_interchange(body["interchange"])
                model.to(device=self.device, dtype=torch.float32)
                state = optimizer.state_dict()
                optimizer = model.optimizer(0.001)
                optimizer.load_state_dict(state)
                lessons = body["lessons"]
            else:
                model = optimizer = None
            if model is not None and model.config != config:
                raise ValueError("New life checkpoint does not match the selected model size")
            for lesson in lessons:
                if not isinstance(lesson, dict):
                    raise TypeError("Invalid New life lesson")
                for key in ("message",) + (("answer",) if "answer" in lesson else ()):
                    value = lesson.get(key)
                    if (
                        not isinstance(value, str)
                        or not value.strip()
                        or len(value.encode("utf-8")) > 100
                        or re.search(r"[\r\n]|(?:User|Assistant):", value)
                    ):
                        raise ValueError("Invalid New life lesson")
            if body.get("interchange") and not Path(path).exists():
                checkpoint.save(path, model, optimizer, {"lessons": lessons})
            self.lives[size] = (model, optimizer, lessons)
        model, optimizer, lessons = self.lives[size]
        if model is not None and next(model.parameters()).device.type != self.device.type:
            state = optimizer.state_dict()
            model.to(self.device)
            optimizer = model.optimizer(0.001)
            optimizer.load_state_dict(state)
            self.lives[size] = (model, optimizer, lessons)

        def status():
            return {
                "ready": model is not None,
                "steps": model.step if model else 0,
                "lessons": len(lessons),
                "corrections": sum("answer" in x for x in lessons),
            }

        if body["action"] == "status":
            return status()
        candidate = (
            copy.deepcopy(model)
            if model
            else Model(config, torch.float32, seed=secrets.randbits(32)).to(self.device)
        )
        candidate_optimizer = candidate.optimizer(0.001)
        if optimizer:
            candidate_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for group in candidate_optimizer.param_groups:
            group["lr"] = 0.001
        lesson = body["lesson"]
        previous = next((x for x in lessons if x["message"] == lesson["message"]), None)
        if "answer" not in lesson and previous and "answer" in previous:
            lesson = previous
        updated = [x for x in lessons if x["message"] != lesson["message"]] + [lesson]
        if len(updated) > 1000:
            raise ValueError("New life memory is full (1,000 lessons).")
        for _ in range(12):
            batch = [updated[-1]] + random.choices(updated, k=body["batchSize"] - 1)
            candidate_optimizer.zero_grad(set_to_none=True)
            # Lessons vary in length; each lesson contributes equal mean loss as in Seed.
            losses = []
            for entry in batch:
                text = "User: " + entry["message"]
                if "answer" in entry:
                    text += "\nAssistant: " + entry["answer"]
                tokens = torch.tensor([encode_training(text)], device=self.device)
                losses.append(
                    F.cross_entropy(
                        candidate(tokens[:, :-1]).reshape(-1, 258), tokens[:, 1:].reshape(-1)
                    )
                )
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = sum(p.grad.square().sum() for p in candidate.parameters()).sqrt()
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradients")
            scale = torch.clamp(1 / (norm + 1e-12), max=1.0)
            for p in candidate.parameters():
                p.grad.mul_(scale)
            candidate_optimizer.step()
            candidate.step += 1
        checkpoint.save(path, candidate, candidate_optimizer, {"lessons": updated})
        model, optimizer, lessons = candidate, candidate_optimizer, updated
        self.lives[size] = (model, optimizer, lessons)
        if body["action"] == "teach":
            return status()
        reply = model.generate("User: " + body["lesson"]["message"] + "\nAssistant:", 80, 0.8, 0)
        return {"completion": reply, "status": status()}

    def command(self, action, body):
        if action == "status":
            pass
        elif action == "train":
            self.start(body["settings"], body["path"])
        elif action == "stop":
            self.stop()
        elif action == "save":
            self.save()
        elif action == "load":
            self.load(body["path"], body.get("interchange"))
        elif action == "delete":
            Path(body["path"]).unlink(missing_ok=True)
            self.saved_step = None
        elif action == "generate":
            if self.model is None:
                raise ValueError("Train or load a model first")
            started = time.perf_counter()
            generated = self.model.generate_detailed(
                body["prompt"], body["maxTokens"], body["temperature"], min(body["topK"], 257)
            )
            return {
                **generated,
                "elapsedMs": round((time.perf_counter() - started) * 1000),
                "modelStep": self.model.step,
            }
        elif action == "life":
            return self.life(body)
        elif action == "shutdown":
            self.stop()
        else:
            raise ValueError("Unknown worker command")
        return self.status()


def main():
    torch.set_num_threads(int(os.environ.get("SEED_CPU_THREADS", "4")))
    studio = Studio()
    requests = queue.Queue(maxsize=64)

    def shutdown(_signal, _frame):
        try:
            requests.put_nowait(None)
        except queue.Full:
            pass  # stdin EOF also terminates the worker after queued commands drain.

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    def read():
        try:
            for line in sys.stdin:
                requests.put(json.loads(line))
        finally:
            requests.put(None)

    threading.Thread(target=read, daemon=True).start()
    emit({"event": "ready", "version": 1, "status": studio.status()})
    while True:
        try:
            request = requests.get(block=studio.training is None)
        except queue.Empty:
            request = False
        if request is None:
            studio.stop()
            break
        if request is not False:
            try:
                result = studio.command(request["action"], request.get("body", {}))
                emit({"id": request["id"], "result": result, "status": studio.status()})
                if request["action"] == "shutdown":
                    break
            except Exception as error:
                logger.exception("PyTorch operation failed")
                emit({"id": request["id"], "error": str(error), "status": studio.status()})
        elif studio.training is not None:
            try:
                studio.update()
            except Exception as error:
                logger.exception("PyTorch operation failed")
                studio.training = None
                studio.error = str(error)
                if studio.job:
                    studio.job["finishedAt"] = time.time() * 1000
                emit({"event": "status", "status": studio.status()})


if __name__ == "__main__":
    main()
