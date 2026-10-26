import json
import random
import time
from pathlib import Path

import torch
from seed_torch import qa
from seed_torch.model import Model

torch.set_num_threads(2)
records = [
    (f"What color is the {color} object number {i}?", color + ".")
    for i in range(25)
    for color in ("red", "blue", "green", "yellow")
]
encoded = [qa.encode(r, 128) for r in records]
model = Model(
    {"hiddenSize": 64, "layers": 2, "heads": 4, "contextSize": 128}, torch.float32
)
optimizer = model.optimizer(0.003)
sampler = random.Random(42)
start = time.perf_counter()
for step in range(1, 1501):
    x, y = qa.batch(sampler.choices(encoded, k=16), "cpu")
    loss = model.update(x, y, optimizer, 0.003)
    if step % 100 == 0:
        evaluation_loss, accuracy = qa.evaluate(model, records, encoded, 100)
        result = {
            "step": step,
            "loss": loss,
            "evaluationLoss": evaluation_loss,
            "trainingAccuracy": accuracy,
            "examples": 100,
            "seconds": round(time.perf_counter() - start, 2),
            "device": "cpu",
            "config": model.config,
        }
        print(json.dumps(result), flush=True)
        Path("/tmp/seed-qa-overfit-result.json").write_text(
            json.dumps(result, indent=2)
        )
        if accuracy >= 0.95:
            break
