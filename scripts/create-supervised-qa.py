"""Deterministic, balanced synthetic Q&A practice with a separate unseen holdout.

Run from the repository root: python3 scripts/create-supervised-qa.py
Existing outputs are never overwritten. No third-party dependencies are required.
"""

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
OBJECTS = [
    "cup",
    "bottle",
    "robot",
    "boat",
    "ball",
    "kite",
    "book",
    "lamp",
    "spoon",
    "pear",
    "box",
    "pencil",
    "hat",
    "coat",
    "banana",
    "tree",
    "bowl",
    "train",
    "star",
    "car",
    "pen",
    "apple",
    "chair",
    "desk",
    "bag",
    "clock",
    "drum",
    "fork",
    "glass",
    "key",
    "leaf",
    "mug",
    "plate",
    "ring",
    "shoe",
    "sock",
    "stone",
    "table",
    "toy",
    "truck",
]
NAMES = [
    "Ada",
    "Ben",
    "Mia",
    "Zoe",
    "Omar",
    "Nora",
    "Sam",
    "Leo",
    "Eva",
    "Max",
    "Ivy",
    "Tom",
]
PLACES = [
    "kitchen",
    "garden",
    "bedroom",
    "hall",
    "garage",
    "library",
    "office",
    "classroom",
]
FAMILIES = [
    "color",
    "object",
    "color-distractor",
    "object-selection",
    "state-update",
    "ownership",
    "transfer",
    "location",
    "carrying",
    "universal-rule",
    "conditional",
    "negation",
    "temporal",
    "addition",
    "subtraction",
    "multiplication",
    "comparison",
    "quantity-comparison",
    "counting",
    "multi-step-arithmetic",
    "ordering",
    "parity",
]


def plural(word):
    return {"box": "boxes", "glass": "glasses", "leaf": "leaves"}.get(word, word + "s")


def example(family, rng):
    c, d, e = rng.sample(COLORS, 3)
    o, p = rng.sample(OBJECTS, 2)
    n, m, z = rng.sample(NAMES, 3)
    a, b = rng.randrange(21), rng.randrange(21)
    first = rng.choice([True, False])
    v = rng.randrange(8)
    if family == "color":
        return [
            f"What color is the {c} {o}?",
            f"The {o} is {c}. What color is it?",
            f'Name the color in "{c} {o}".',
            f"A {o} is {c}. Give its color.",
            f"{n} sees a {c} {o}. What color is the {o}?",
            f"{n} has a {c} {o}. Name its color.",
            f"Describe only the color of the {c} {o}.",
            f"The color of the {o} is {c}. Which color was stated?",
        ][v], c
    if family == "object":
        return [
            f'What is the object in "{c} {o}"?',
            f"A {c} {o} is here. Name the object.",
            f'Name the object in "{c} {o}".',
            f"Which object is described as {c}: a {c} {o}?",
            f"{n} sees a {c} {o}. What is the object?",
            f"{n} has a {c} {o}. Name the object.",
            f'Ignore the color in "{c} {o}". Name the thing.',
            f"A {c} {o} is on the floor. What is the object?",
        ][v], o
    if family in ("color-distractor", "object-selection"):
        target_o, target_c = (o, c) if first else (p, d)
        lead = rng.choice(
            [
                f"A {c} {o} is beside a {d} {p}.",
                f"On a shelf there is a {c} {o} and a {d} {p}.",
                f"{n} sees a {c} {o} next to a {d} {p}.",
                f"There are two things: a {c} {o} and a {d} {p}.",
            ]
        )
        return (
            (f"{lead} What color is the {target_o}?", target_c)
            if family == "color-distractor"
            else (f"{lead} Which object is {target_c}?", target_o)
        )
    if family == "state-update":
        lead = rng.choice(
            [
                f"{n} has a {c} {o}. {n} paints the {o} {d}.",
                f"The {o} was {c}, then {n} painted it {d}.",
            ]
        )
        # Both latest-state and initial-state queries discourage always copying last color.
        return (
            (f"{lead} What color is the {o} now?", d)
            if first
            else (f"{lead} What color was the {o} before painting?", c)
        )
    if family == "ownership":
        return (
            f"{n} has a {c} {o}. {m} has a {d} {p}. Who has the {o if first else p}?",
            n if first else m,
        )
    if family == "transfer":
        return (
            f"{n} gave a {c} {o} to {m}. Who {'received' if first else 'gave'} the {o}?",
            m if first else n,
        )
    if family == "location":
        x, y = rng.sample(PLACES, 2)
        return (
            f"The {c} {o} is in the {x}. The {d} {p} is in the {y}. Where is the {o if first else p}?",
            x if first else y,
        )
    if family == "carrying":
        return (
            f"{n} carries a {c} {o}. {m} carries a {d} {p}. What object is {n if first else m} carrying?",
            o if first else p,
        )
    if family == "universal-rule":
        kind = rng.choice(
            ["robots", "birds", "cats", "dogs", "boats", "cars", "cups", "kites"]
        )
        singular = kind[:-1]
        return rng.choice(
            [
                f"All {kind} are {c}. {n} is a {singular}. What color is {n}?",
                f"Every {singular} is {c}. {n} is a {singular}. Name the color of {n}.",
                f"{n} is a {singular}. All {kind} are {c}. What color is {n}?",
                f"Rule: all {kind} are {c}. Given that {n} is a {singular}, what color is {n}?",
            ]
        ), c
    if family == "conditional":
        actual = c if first else e
        return (
            f"Rule: If the {o} is {c}, choose {d}; otherwise choose {e}. The {o} is {actual}. Which color should be chosen?",
            d if first else e,
        )
    if family == "negation":
        return f"The {o} is {c}, not {d}. {n} asks: what color is the {o}?", c
    if family == "temporal":
        return (
            f"{n} arrived before {m}. {m} arrived before {z}. Who arrived {'first' if first else 'last'}?",
            n if first else z,
        )
    if family == "addition":
        return rng.choice(
            [
                f"What is {a} plus {b}?",
                f"How much is {a} added to {b}?",
                f"Calculate {a} + {b}.",
                f"Find the sum of {a} and {b}.",
                f"{n} asks: what is {a} + {b}?",
                f"Add {a} and {b}.",
            ]
        ), str(a + b)
    if family == "subtraction":
        a, b = max(a, b), min(a, b)
        return rng.choice(
            [
                f"What is {a} minus {b}?",
                f"Find the difference between {a} and {b}.",
                f"How much remains after taking {b} away from {a}?",
                f"Calculate {a} - {b}.",
                f"{n} asks: subtract {b} from {a}.",
                f"{n} had {a} tokens and lost {b}. How many tokens remain?",
            ]
        ), str(a - b)
    if family == "multiplication":
        a, b = a % 11, b % 11
        return rng.choice(
            [
                f"What is {a} times {b}?",
                f"Calculate {a} * {b}.",
                f"{n} asks: what is {a} multiplied by {b}?",
                f"Find the product of {a} and {b}.",
                f"There are {a} groups of {b} tokens. How many tokens in total?",
                f"{n} asks: calculate {a} * {b}.",
            ]
        ), str(a * b)
    if family == "comparison":
        operator = rng.choice(["greater than", "less than", "equal to"])
        truth = {"greater than": a > b, "less than": a < b, "equal to": a == b}[
            operator
        ]
        return (
            f"{n} asks: is {a} {operator} {b}? Answer yes or no.",
            "yes" if truth else "no",
        )
    if family == "quantity-comparison":
        if a == b:
            return None
        return (
            f"{n} has {a} coins. {m} has {b} coins. Who has {'more' if first else 'fewer'} coins?",
            (n if a > b else m) if first else (m if a > b else n),
        )
    if family == "counting":
        return (
            f"{n} has {a} {plural(o)} and {b} {plural(p)}. How many objects does {n} have in total?",
            str(a + b),
        )
    if family == "multi-step-arithmetic":
        spent = rng.randrange(min(a + b, 20) + 1)
        return (
            f"{n} had {a} coins, received {b} more, then spent {spent} coins. How many coins remain?",
            str(a + b - spent),
        )
    if family == "ordering":
        values = rng.sample(range(21), 3)
        return (
            f"{n} asks: put these numbers in ascending order, separated by commas: {', '.join(map(str, values))}.",
            ", ".join(map(str, sorted(values))),
        )
    if family == "parity":
        a = rng.randrange(100)
        return (
            f"{n} asks: is {a} an {'even' if first else 'odd'} number? Answer yes or no.",
            "yes" if (a % 2 == 0) == first else "no",
        )
    raise ValueError(family)


def build(per_family=2000, holdout_per_family=100, seed=20261024):
    rng = random.Random(seed)
    old_questions = set()
    for path in (ROOT / "data/inputs").glob("*.txt"):
        if path.name.startswith("supervised_qa_v2"):
            continue
        old_questions.update(
            re.findall(r"^User: (.*)$", path.read_text(), re.MULTILINE)
        )
    for path in (ROOT / "data/references").glob("*.json"):
        if path.name.startswith("supervised_qa_v2"):
            continue
        content = json.loads(path.read_text())
        if isinstance(content, list):
            old_questions.update(
                r["question"]
                for r in content
                if isinstance(r, dict) and "question" in r
            )
    training, holdout, known = [], [], {}
    for family in FAMILIES:
        train, test = [], []
        # Balance by answer (rather than natural frequency) for boolean tasks.
        boolean = family in ("comparison", "parity")
        counts = Counter()
        for attempt in range(2_000_000):
            value = example(family, rng)
            if value is None:
                continue
            q, a = value
            q = re.sub(
                r"\b([aA]) ([aeiou])",
                lambda m: ("An" if m[1] == "A" else "an") + " " + m[2],
                q,
            )
            if q in known:
                if known[q] != a:
                    raise ValueError(f"Conflicting answers: {q}")
                continue
            bucket = "yes" if a == "yes" else "no"
            if boolean and counts[bucket] >= (per_family + holdout_per_family) // 2:
                continue
            if len(test) < holdout_per_family and q not in old_questions:
                test.append({"question": q, "answer": a, "family": family})
            elif len(train) < per_family:
                train.append({"question": q, "answer": a, "family": family})
            else:
                continue
            known[q] = a
            counts[bucket] += 1
            if len(train) == per_family and len(test) == holdout_per_family:
                break
        else:
            raise ValueError(f"Could not fill {family}")
        training.extend(train)
        holdout.extend(test)
    rng.shuffle(training)
    rng.shuffle(holdout)
    return training, holdout


def render(records):
    return "".join(
        f"User: {r['question']}\nAssistant: {r['answer']}.\n\n" for r in records
    )


def main():
    train, test = build()
    text = render(train)
    maximum = max(
        1 + len(f"User: {r['question']}\nAssistant: {r['answer']}.".encode())
        for r in train + test
    )
    assert len(text.encode()) <= 10_000_000
    assert maximum <= 256
    assert not {r["question"] for r in train} & {r["question"] for r in test}
    outputs = {
        ROOT / "data/inputs/supervised_qa_v2.txt": text,
        ROOT / "data/references/supervised_qa_v2_unseen.json": json.dumps(
            [{**r, "answer": r["answer"].lower()} for r in test],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        ROOT / "data/references/supervised_qa_v2_unseen.txt": render(test),
    }
    manifest = {
        "seed": 20261024,
        "trainingExamples": len(train),
        "unseenExamples": len(test),
        "families": dict(Counter(r["family"] for r in train)),
        "maxInputTokens": maximum,
        "corpusBytes": len(text.encode()),
        "corpusSha256": hashlib.sha256(text.encode()).hexdigest(),
        "scope": "Synthetic bounded tasks; unseen questions share templates and vocabulary. No guarantee of reasoning generalization.",
    }
    outputs[ROOT / "data/references/supervised_qa_v2_manifest.json"] = (
        json.dumps(manifest, indent=2) + "\n"
    )
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    for path, content in outputs.items():
        path.write_text(content)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
