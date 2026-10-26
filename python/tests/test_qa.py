import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from seed_torch import checkpoint, qa
from seed_torch.model import Model
from seed_torch.worker import Studio


class QATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def model(self):
        return Model({"hiddenSize": 16, "layers": 1, "heads": 2, "contextSize": 128}, torch.float32)

    def test_parser(self):
        self.assertEqual(
            qa.parse("\nUser: blå?\nAssistant: 紫.\n\nUser: two\nlines\nAssistant: a\nb\n"),
            [("blå?", "紫."), ("two\nlines", "a\nb")],
        )
        for text in [
            "",
            "User: a",
            "Assistant: a",
            "junk\nUser: a\nAssistant: b",
            "User:a\nAssistant: b",
            "User: a\nAssistant: ",
            "User: a\nAssistant: b\nAssistant: c",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                qa.parse(text)

    def test_split(self):
        records = [(str(i), "a") for i in range(100)]
        a, b = qa.split(records * 2)
        self.assertEqual((len(a), len(b)), (90, 10))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a, b), qa.split(records * 2))
        self.assertNotEqual((a, b), qa.split(records, 43))
        with self.assertRaises(ValueError):
            qa.split([("a", "b")] * 10)

    def test_encoding_alignment_padding(self):
        record = ("紫?", "blå.")
        x, y = qa.encode(record, 128)
        prefix = [256, *qa.prompt(record[0]).encode()]
        self.assertEqual(x[: len(prefix)], prefix)
        self.assertEqual(y[: len(prefix) - 1], [-100] * (len(prefix) - 1))
        self.assertEqual(y[len(prefix) - 1 :], [*(" " + record[1]).encode(), 257])
        self.assertEqual(y[-1], 257)
        bx, by = qa.batch([(x, y), qa.encode(("a", "b"), 128)], "cpu")
        self.assertTrue((by[1, len(qa.encode(("a", "b"), 128)[0]) :] == -100).all())
        model = self.model()
        torch.testing.assert_close(model(bx)[0, : len(x)], model(torch.tensor([x]))[0])
        with self.assertRaisesRegex(ValueError, "needs"):
            qa.encode(record, len(x) - 1)

    def test_masked_loss_and_prompt_gradients(self):
        model = self.model()
        x, y = qa.batch([qa.encode(("question", "answer"), 128)], "cpu")
        logits = model(x)
        logits.retain_grad()
        loss = F.cross_entropy(logits.reshape(-1, 258), y.reshape(-1), ignore_index=-100)
        changed = logits.detach().clone()
        changed[y == -100] = 1000
        self.assertEqual(
            loss.item(), F.cross_entropy(changed.reshape(-1, 258), y.reshape(-1)).item()
        )
        loss.backward()
        self.assertEqual(logits.grad[y == -100].abs().sum().item(), 0)
        self.assertGreater(model.weight("tokenEmbedding").grad[ord("q")].abs().sum().item(), 0)
        altered = x.clone()
        altered[0, 7] = ord("z")
        self.assertFalse(torch.equal(model(x)[0, -1], model(altered)[0, -1]))

    def test_greedy_and_accuracy(self):
        model = self.model()
        model.train()
        a = model.generate_detailed("a", 5, 0.1, 1)
        torch.manual_seed(912)
        b = model.generate_detailed("a", 5, 2, 1)
        self.assertEqual(a, b)
        self.assertTrue(model.training)
        records = [("a", "yes"), ("b", "no")]
        encoded = [qa.encode(r, 128) for r in records]
        with patch.object(
            model,
            "generate_detailed",
            side_effect=[
                {"completion": " yes", "endedWithEos": True},
                {"completion": " no", "endedWithEos": False},
            ],
        ) as generate:
            _, accuracy = qa.evaluate(model, records, encoded, 2)
            self.assertEqual(accuracy, 0.5)
            self.assertEqual(generate.call_args_list[0].args[0], "User: a\nAssistant:")
        self.assertTrue(model.training)
        with torch.no_grad():
            model.weight("outputBias").fill_(-100)
            model.weight("outputBias")[0, 257] = 100
        self.assertEqual(model.generate_detailed("a")["tokens"], 1)

    def test_worker_checkpoint_switch_and_sampling(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SEED_DEVICE="cpu"):
            studio = Studio()
            model = self.model()
            optimizer = model.optimizer(0.001)
            path = str(Path(directory) / "old.pt")
            checkpoint.save(path, model, optimizer, {})  # Legacy session, no mode metadata.
            studio.load(path)
            previous_optimizer = studio.optimizer
            before = {k: p.clone() for k, p in studio.model.state_dict().items()}
            body = {
                "text": "\n\n".join(f"User: {i}\nAssistant: ok" for i in range(20)),
                "resume": True,
                "hiddenSize": 16,
                "layers": 1,
                "heads": 2,
                "sequenceLength": 128,
                "batchSize": 3,
                "steps": 1,
                "learningRate": 0.001,
                "trainingMode": "supervised-qa",
                "evaluationSamples": 2,
            }
            studio.start(body, path)
            for k, p in studio.model.state_dict().items():
                torch.testing.assert_close(p, before[k])
            self.assertIsNot(studio.optimizer, previous_optimizer)
            with contextlib.redirect_stdout(io.StringIO()):
                studio.update()
            self.assertEqual(studio.model.step, 1)
            self.assertIn("validationAccuracy", studio.job)
            loaded = Studio()
            loaded.load(path)
            self.assertEqual(loaded.settings["trainingMode"], "supervised-qa")
            self.assertEqual(len(loaded.split_metadata["validationIds"]), 2)
            previous_optimizer = loaded.optimizer
            body.update(
                trainingMode="language-modeling",
                restartSchedule=False,
                resetOptimizerOnModeChange=False,
            )
            loaded.start(body, path)
            self.assertIsNone(loaded.qa_data)
            self.assertIs(loaded.optimizer, previous_optimizer)
            self.assertEqual(loaded.schedule_step, 1)
            self.assertEqual(loaded.schedule_total, 1)
            with contextlib.redirect_stdout(io.StringIO()):
                loaded.update()
            self.assertEqual(loaded.model.step, 2)


if __name__ == "__main__":
    unittest.main()
