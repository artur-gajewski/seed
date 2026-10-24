"""Checks for corpus isolation and recurrent inference correctness."""

import hashlib
import json
import math
import unittest
from html.parser import HTMLParser

import torch
from experiment import GRU, ROOT


class ResearchChecks(unittest.TestCase):
    def test_task_combinations_and_questions_are_disjoint(self):
        manifest = json.loads((ROOT / "corpora.json").read_text())
        pair_sets = [set(map(tuple, values)) for values in manifest["pairs"].values()]
        question_sets = [
            {
                r["question"]
                for r in json.loads((ROOT / f"tasks-{split}.json").read_text())
            }
            for split in ["train", "validation", "test"]
        ]
        for sets in [pair_sets, question_sets]:
            for i in range(3):
                for j in range(i):
                    self.assertFalse(sets[i] & sets[j])

    def test_prose_documents_are_disjoint(self):
        sets = [
            {
                s.strip()
                for s in (ROOT / f"prose-{split}.txt")
                .read_text()
                .split("<|endofstory|>")
            }
            for split in ["train", "validation", "test"]
        ]
        for i in range(3):
            for j in range(i):
                self.assertFalse(sets[i] & sets[j])

    def test_corpus_hashes_match_recorded_artifacts(self):
        manifest = json.loads((ROOT / "corpora.json").read_text())
        for filename, info in manifest["files"].items():
            actual = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, info["sha256"], filename)

    def test_recorded_results_are_finite_and_complete(self):
        for folder, seeds in [
            (ROOT, [42, 43, 44]),
            (ROOT / "full-scale", [42]),
            (ROOT / "mps", [42, 43, 44]),
            (ROOT / "mps/full-scale", [42]),
        ]:
            for kind in ["transformer", "gru"]:
                for seed in seeds:
                    row = json.loads((folder / f"run-{kind}-{seed}.json").read_text())
                    self.assertEqual(row["history"][-1]["step"], 800)
                    self.assertEqual(row["tokens_seen"], 819200)
                    for key in [
                        "task_loss",
                        "prose_loss",
                        "training_seconds",
                        "decode_tokens_s",
                    ]:
                        self.assertTrue(math.isfinite(row[key]) and row[key] > 0)
                    answers = json.loads(
                        (folder / f"answers-{kind}-{seed}.json").read_text()
                    )
                    self.assertAlmostEqual(
                        sum(a["correct"] for a in answers) / len(answers),
                        row["accuracy"],
                    )

    def test_mps_corpora_and_variant_devices(self):
        manifest = json.loads((ROOT / "corpora.json").read_text())
        for folder in [ROOT / "mps", ROOT / "mps/full-scale"]:
            for name, info in manifest["files"].items():
                self.assertEqual(
                    hashlib.sha256((folder / name).read_bytes()).hexdigest(),
                    info["sha256"],
                )
            for kind in ["transformer", "gru"]:
                row = json.loads((folder / f"run-{kind}-42.json").read_text())
                self.assertEqual(row["device"], "mps")
                self.assertEqual(
                    row["variants"][0]["device"],
                    "mps" if kind == "transformer" else "cpu",
                )
                if kind == "gru":
                    self.assertEqual(row["cpu_parent"]["device"], "cpu")
                self.assertGreater(
                    row["mps_memory"]["driver_allocated_bytes_after_benchmark"], 0
                )

    def test_cross_device_results_match_saved_training_scores(self):
        rows = json.loads((ROOT / "mps/cross-device.json").read_text())
        self.assertEqual(len(rows), 4)
        for trained_on, folder in [
            ("cpu", ROOT / "full-scale"),
            ("mps", ROOT / "mps/full-scale"),
        ]:
            recorded = json.loads((folder / "run-transformer-42.json").read_text())
            subset = [r for r in rows if r["trained_on"] == trained_on]
            self.assertEqual({r["evaluated_on"] for r in subset}, {"cpu", "mps"})
            for row in subset:
                self.assertAlmostEqual(row["accuracy"], recorded["accuracy"])
                self.assertAlmostEqual(
                    row["task_loss"], recorded["task_loss"], places=5
                )

    def test_html_has_inline_charts_and_no_external_assets(self):
        class ReportParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.charts = 0
                self.external = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "svg":
                    self.charts += 1
                if tag in {"script", "img", "link", "iframe"}:
                    self.external.append((tag, attrs))

        parser = ReportParser()
        parser.feed((ROOT / "report.html").read_text())
        self.assertGreaterEqual(parser.charts, 4)
        self.assertFalse(parser.external)

    def test_incremental_gru_matches_full_prefix(self):
        torch.manual_seed(5)
        model = GRU().eval()
        x = torch.randint(258, (1, 25))
        with torch.no_grad():
            full = model(x)
            first, state = model.next(x[:, :20])
            rest, _ = model.next(x[:, 20:], state)
        torch.testing.assert_close(
            torch.cat([first, rest], 1), full, atol=1e-6, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
