"""Complete-record supervised training, using the existing chat prefix and vocabulary."""

import hashlib
import random

import torch
from torch.nn import functional as F


def prompt(question):
    # Keep identical to server/chat.ts and cli.chat_prompt: no trailing space.
    return f"User: {question}\nAssistant:"


def parse(text):
    records = []
    question = answer = None
    state = None

    def finish(line):
        if question is None or answer is None or not question.strip() or not answer.strip():
            raise ValueError(f"Incomplete Q&A example before line {line}")
        records.append((question, answer))

    lines = text.splitlines()
    # Blank lines between turns are separators; internal multiline content is retained.
    while lines and not lines[-1].strip():
        lines.pop()
    for number, line in enumerate(lines, 1):
        if line.startswith("User: "):
            if state is not None:
                finish(number)
            question, answer, state = line[6:], None, "question"
        elif line.startswith("Assistant: "):
            if state != "question":
                raise ValueError(f"Unexpected Assistant at line {number}")
            answer, state = line[11:], "answer"
        elif line.startswith(("User:", "Assistant:")):
            raise ValueError(f"Expected role label followed by a space at line {number}")
        elif state is None:
            if line.strip():
                raise ValueError(f"Expected User: at line {number}")
        elif state == "question":
            question += "\n" + line
        else:
            answer += "\n" + line
    if state is not None:
        # Strip only separator newlines, never answer spaces.
        answer = answer.rstrip("\n") if answer is not None else None
        finish(len(lines) + 1)
    if not records:
        raise ValueError("No complete Q&A examples")
    records = [(q, a.rstrip("\n")) for q, a in records]
    if any("<|endofstory|>" in q + a for q, a in records):
        raise ValueError("Q&A content cannot contain the reserved <|endofstory|> marker")
    return records


def record_id(record):
    import json

    return hashlib.sha256(json.dumps(record, ensure_ascii=False).encode("utf-8")).hexdigest()


def split(records, seed=42, validation_split=0.1):
    # Deduplicate before shuffling, so exact records cannot leak across the split.
    records = list(dict.fromkeys(records))
    if len(records) < 2:
        raise ValueError("Q&A training needs at least two distinct examples")
    random.Random(seed).shuffle(records)
    count = max(1, min(len(records) - 1, round(len(records) * validation_split)))
    return records[count:], records[:count]


def encode(record, context, assistant_only=True):
    question, answer = record
    prefix = [256, *prompt(question).encode("utf-8")]
    tokens = prefix + list((" " + answer).encode("utf-8")) + [257]
    if len(tokens) - 1 > context:
        raise ValueError(
            f"Q&A example {record_id(record)[:12]} needs {len(tokens) - 1} context tokens; limit is {context}"
        )
    labels = tokens[1:]
    if assistant_only:
        labels[: len(prefix) - 1] = [-100] * (len(prefix) - 1)
    return tokens[:-1], labels


def batch(examples, device):
    length = max(len(x) for x, _ in examples)
    x = torch.zeros((len(examples), length), dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, (inputs, labels) in enumerate(examples):
        x[i, : len(inputs)] = torch.tensor(inputs, device=device)
        y[i, : len(labels)] = torch.tensor(labels, device=device)
    return x, y


@torch.no_grad()
def evaluate(model, records, encoded, sample_count, batch_size=16):
    """Stable first N split members; strict byte-preserving generated answer equality."""
    previous = model.training
    model.eval()
    try:
        count = min(sample_count, len(records))
        device = next(model.parameters()).device
        total_loss = total_labels = correct = 0
        for start in range(0, count, batch_size):
            x, y = batch(encoded[start : min(start + batch_size, count)], device)
            total_loss += F.cross_entropy(
                model(x).reshape(-1, 258), y.reshape(-1), ignore_index=-100, reduction="sum"
            ).item()
            total_labels += (y != -100).sum().item()
        for question, answer in records[:count]:
            # Never slide away the prompt; allow expected answer length plus a margin.
            limit = min(
                model.config["contextSize"] - len(prompt(question).encode("utf-8")),
                len((" " + answer).encode("utf-8")) + 17,
            )
            result = model.generate_detailed(prompt(question), limit, 1.0, 1)
            correct += result["completion"] == " " + answer and result["endedWithEos"]
        return total_loss / total_labels, correct / count
    finally:
        model.train(previous)
