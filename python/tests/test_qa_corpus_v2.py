"""Independent audits of the generated corpus and its bounded arithmetic answers."""

import hashlib
import importlib.util
import json
import math
import re
import unittest
from collections import Counter
from pathlib import Path

from seed_torch import qa

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "qa_generator", ROOT / "scripts/create-supervised-qa.py"
)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class CorpusV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = generator.build()

    def test_reproducible_balanced_bounded_and_disjoint(self):
        corpus = (ROOT / "data/inputs/supervised_qa_v2.txt").read_text()
        self.assertEqual(generator.render(self.train), corpus)
        self.assertEqual(len(self.train), 44000)
        self.assertEqual(len(self.test), 2200)
        self.assertEqual(set(Counter(r["family"] for r in self.train).values()), {2000})
        self.assertEqual(set(Counter(r["family"] for r in self.test).values()), {100})
        questions = [r["question"] for r in self.train + self.test]
        self.assertEqual(len(questions), len(set(questions)))
        self.assertLess(len(corpus.encode()), 10_000_000)
        for r in self.train + self.test:
            qa.encode((r["question"], r["answer"] + "."), 256)
        self.assertEqual(len(qa.parse(corpus)), 44000)
        unseen = {r["question"] for r in self.test}
        for path in (ROOT / "data/inputs").glob("*.txt"):
            self.assertFalse(
                unseen & set(re.findall(r"^User: (.*)$", path.read_text(), re.MULTILINE))
            )
        for path in (ROOT / "data/references").glob("*.json"):
            if path.name.startswith("supervised_qa_v2"):
                continue
            content = json.loads(path.read_text())
            if isinstance(content, list):
                self.assertFalse(unseen & {r["question"] for r in content})
        manifest = json.loads((ROOT / "data/references/supervised_qa_v2_manifest.json").read_text())
        self.assertEqual(manifest["corpusSha256"], hashlib.sha256(corpus.encode()).hexdigest())
        self.assertEqual(
            generator.render(self.test),
            (ROOT / "data/references/supervised_qa_v2_unseen.txt").read_text(),
        )

    def test_answers_and_boolean_balance(self):
        for family in ("comparison", "parity"):
            self.assertEqual(
                Counter(r["answer"] for r in self.train + self.test if r["family"] == family),
                {"yes": 1050, "no": 1050},
            )
        for r in self.train + self.test:
            q, answer, family = r["question"], r["answer"], r["family"]
            numbers = list(map(int, re.findall(r"\d+", q)))
            expected = None
            if family in ("addition", "counting"):
                expected = str(sum(numbers))
            elif family == "subtraction":
                expected = str(abs(numbers[0] - numbers[1]))
            elif family == "multiplication":
                expected = str(math.prod(numbers))
            elif family == "multi-step-arithmetic":
                expected = str(numbers[0] + numbers[1] - numbers[2])
            elif family == "ordering":
                expected = ", ".join(map(str, sorted(numbers)))
            elif family == "parity":
                expected = "yes" if (numbers[0] % 2 == 0) == ("even" in q) else "no"
            elif family == "comparison":
                a, b = numbers
                correct = a > b if "greater than" in q else a < b if "less than" in q else a == b
                expected = "yes" if correct else "no"
            elif family == "color-distractor":
                pairs = {
                    obj: color
                    for color, obj in re.findall(
                        r"\b(" + "|".join(generator.COLORS) + r") (\w+)", q
                    )
                }
                target = re.search(r"What color is the (\w+)\?", q)[1]
                expected = pairs[target]
            elif family == "object-selection":
                pairs = dict(re.findall(r"\b(" + "|".join(generator.COLORS) + r") (\w+)", q))
                target = re.search(r"Which object is (\w+)\?", q)[1]
                expected = pairs[target]
            if expected is not None:
                self.assertEqual(answer, expected, q)
