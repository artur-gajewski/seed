import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from seed_torch import checkpoint
from seed_torch.cli import chat_prompt, device_for, learning_rate
from seed_torch.model import Model, encode_training

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


class PortTests(unittest.TestCase):
    def test_expanded_architectures_and_limits(self):
        # Exercise each expanded dimension without allocating the maximum model.
        for width, layers, heads in [(1024, 1, 16), (16, 12, 16)]:
            config = {"hiddenSize": width, "layers": layers, "heads": heads, "contextSize": 2}
            model = Model(config, torch.float32)
            inputs, targets = torch.tensor([[256, 72]]), torch.tensor([[72, 257]])
            logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (1, 2, 258))
            loss = F.cross_entropy(logits.reshape(-1, 258), targets.reshape(-1))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "expanded.pt"
                optimizer = model.optimizer(0.001)
                checkpoint.save(path, model, optimizer, {})
                restored, _, _ = checkpoint.load(path, "cpu", torch.float32)
                self.assertEqual(restored.config, config)
                torch.testing.assert_close(restored(inputs), logits, atol=0, rtol=0)
        for field, value in [("hiddenSize", 1025), ("layers", 13), ("heads", 17), ("contextSize", 2049)]:
            config = {"hiddenSize": 16, "layers": 1, "heads": 4, "contextSize": 2}
            config[field] = value
            with self.assertRaisesRegex(ValueError, "configuration"):
                Model(config)

    def test_expanded_context_and_checkpoint(self):
        config = {"hiddenSize": 4, "layers": 1, "heads": 1, "contextSize": 2048}
        model = Model(config, torch.float32)
        inputs = torch.tensor([[65] * 257])
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1, 257, 258))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.pt"
            checkpoint.save(path, model, model.optimizer(0.001), {})
            restored, _, _ = checkpoint.load(path, "cpu", torch.float32)
            self.assertEqual(restored.config["contextSize"], 2048)
            torch.testing.assert_close(restored(inputs), logits, atol=0, rtol=0)

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(
            subprocess.check_output(["node", "scripts/torch-parity-fixture.ts"], cwd=ROOT)
        )

    def assert_close(self, actual, expected, tolerance=1e-10):
        torch.testing.assert_close(
            actual.detach().cpu(),
            torch.tensor(expected, dtype=actual.dtype).reshape_as(actual),
            atol=tolerance,
            rtol=tolerance,
        )

    def test_typescript_logits_gradients_adam_and_batches(self):
        f = self.fixture
        model = Model(f["initial"]["config"])
        for key, parameter in model.weights.items():
            self.assert_close(
                parameter, f["initial"]["parameters"][key.replace("__", ".")]["values"]
            )
        x, y = torch.tensor([f["inputs"]]), torch.tensor([f["targets"]])
        self.assert_close(model(x), f["logits"])
        F.cross_entropy(model(x).reshape(-1, 258), y.reshape(-1)).backward()
        for key, parameter in model.weights.items():
            self.assert_close(parameter.grad, f["gradients"][key.replace("__", ".")])
        optimizer = model.optimizer(0.0003)
        self.assertAlmostEqual(model.update(x, y, optimizer, 0.0003), f["firstLoss"], places=11)
        model.update(
            torch.cat((x, torch.tensor([[3, 4, 5, 6]]))),
            torch.cat((y, torch.tensor([[4, 5, 6, 7]]))),
            optimizer,
            0.0002,
        )
        for key, parameter in model.weights.items():
            expected = f["final"]["parameters"][key.replace("__", ".")]
            self.assert_close(parameter, expected["values"])
            self.assert_close(optimizer.state[parameter]["exp_avg"], expected["firstMoments"])
            self.assert_close(optimizer.state[parameter]["exp_avg_sq"], expected["secondMoments"])

    def test_tokenization(self):
        for item in self.fixture["tokenization"]:
            self.assertEqual(encode_training(item["text"]), item["tokens"])

    def test_checkpoint_resume_and_seed_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps({"model": self.fixture["final"]}))
            model, optimizer = checkpoint.load_interchange(source)
            native = Path(directory) / "model.pt"
            checkpoint.save(native, model, optimizer, {"completed": 2})
            restored, restored_optimizer, saved = checkpoint.load(native, "cpu")
            self.assertEqual(saved["session"]["completed"], 2)
            x, y = torch.tensor([[256, 72, 105, 33]]), torch.tensor([[72, 105, 33, 257]])
            model.update(x, y, optimizer, 0.0001)
            restored.update(x, y, restored_optimizer, 0.0001)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            output = Path(directory) / "export.json"
            checkpoint.export_interchange(output, restored, restored_optimizer)
            seed = Path(directory) / "seed.json"
            subprocess.run(
                [
                    "node",
                    "scripts/convert-torch-checkpoint.ts",
                    "--input",
                    str(output),
                    "--output",
                    str(seed),
                    "--to-seed",
                ],
                cwd=ROOT,
                check=True,
            )
            back = Path(directory) / "back.json"
            subprocess.run(
                [
                    "node",
                    "scripts/convert-torch-checkpoint.ts",
                    "--input",
                    str(seed),
                    "--output",
                    str(back),
                ],
                cwd=ROOT,
                check=True,
            )
            self.assertEqual(json.loads(output.read_text()), json.loads(back.read_text()))

    def test_causality_and_learning(self):
        model = Model(self.fixture["initial"]["config"])
        a, b = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 2, 5, 6]])
        torch.testing.assert_close(model(a)[:, :2], model(b)[:, :2], atol=0, rtol=0)
        optimizer = model.optimizer(0.01)
        targets = torch.tensor([[2, 3, 4, 5]])
        first = model.update(a, targets, optimizer, 0.01)
        for _ in range(19):
            last = model.update(a, targets, optimizer, 0.01)
        self.assertLess(last, first)

    def test_schedule_and_chat(self):
        self.assertEqual(learning_rate(0, 1000, 0.001), 0.00002)
        self.assertAlmostEqual(learning_rate(1000, 1000, 0.001), 0.0001)
        self.assertEqual(
            chat_prompt(
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "Bye"},
                ]
            ),
            "User: Hi\nAssistant: Hello\n\nUser: Bye\nAssistant:",
        )
        with self.assertRaises(ValueError):
            device_for("mps", "float64")

    def test_cli_train_resume_evaluate_and_invalid_request(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            corpus = base / "corpus.txt"
            corpus.write_text("User: What color is the sky?\nAssistant: Blue.\n\n" * 8)
            saved = base / "model.pt"
            command = [sys.executable, "-m", "seed_torch.cli"]
            flags = ["--checkpoint", str(saved), "--device", "cpu", "--precision", "float64"]
            training = [
                "train",
                *flags,
                "--corpus",
                str(corpus),
                "--embedding-width",
                "8",
                "--layers",
                "1",
                "--attention-heads",
                "2",
                "--context-window",
                "8",
                "--batch-size",
                "2",
                "--steps",
                "2",
            ]
            subprocess.run(command + training, check=True, capture_output=True)
            before = saved.read_bytes()
            invalid = subprocess.run(
                command + training + ["--learning-rate", "nan"], check=False, capture_output=True
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(before, saved.read_bytes())
            subprocess.run(command + training + ["--resume"], check=True, capture_output=True)
            model, _, _ = checkpoint.load(saved, "cpu")
            self.assertEqual(model.step, 4)
            evaluation = base / "reference.json"
            evaluation.write_text(
                json.dumps(
                    [{"question": "What color is the sky?", "answer": "blue", "family": "smoke"}]
                )
            )
            report = base / "report.json"
            subprocess.run(
                command
                + ["evaluate", *flags, "--evaluation", str(evaluation), "--output", str(report)],
                check=True,
                capture_output=True,
            )
            self.assertEqual(json.loads(report.read_text())["total"], 1)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_update_save_resume(self):
        cpu = Model(self.fixture["initial"]["config"], torch.float32)
        gpu = Model(cpu.config, torch.float32).to("mps")
        x, y = torch.tensor([[256, 72, 105, 33]]), torch.tensor([[72, 105, 33, 257]])
        torch.testing.assert_close(cpu(x), gpu(x.to("mps")).cpu(), atol=1e-5, rtol=1e-4)
        self.assertEqual(cpu.generate("Hi", max_tokens=4), gpu.generate("Hi", max_tokens=4))
        optimizer = gpu.optimizer(0.0003)
        gpu.update(x.to("mps"), y.to("mps"), optimizer, 0.0003)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mps.pt"
            checkpoint.save(path, gpu, optimizer, {})
            restored, resumed, _ = checkpoint.load(path, "mps", torch.float32)
            restored.update(x.to("mps"), y.to("mps"), resumed, 0.0003)
            self.assertEqual(restored.step, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_update_save_resume(self):
        model = Model(self.fixture["initial"]["config"], torch.float32).to("cuda")
        x = torch.tensor([[256, 72, 105, 33]], device="cuda")
        y = torch.tensor([[72, 105, 33, 257]], device="cuda")
        optimizer = model.optimizer(0.0003)
        model.update(x, y, optimizer, 0.0003)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cuda.pt"
            checkpoint.save(path, model, optimizer, {})
            restored, resumed, _ = checkpoint.load(path, "cuda", torch.float32)
            restored.update(x, y, resumed, 0.0003)
            self.assertEqual(restored.step, 2)


if __name__ == "__main__":
    unittest.main()
