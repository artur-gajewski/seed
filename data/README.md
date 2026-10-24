# Seed data

| Folder        | Contents                        | Git     |
| ------------- | ------------------------------- | ------- |
| `inputs/`     | Training corpora (`.txt`)       | Tracked |
| `references/` | Reference questions and answers | Tracked |
| `reports/`    | Generated evaluation reports    | Ignored |
| `models/`     | Generated checkpoints, New Life | Ignored |

The studio defaults to `data/models/seed_default.pt`; the **Models** panel selects
other named `.pt` models for loading or training. Keep corpora separately;
checkpoints do not contain their source text. Reference datasets contain answers
and should not be imported for training. Reports are generated under `data/reports/`
and must not overwrite the reference datasets.

Evaluate PyTorch models from the repository root with the project's Python environment:

```sh
source .venv/bin/activate
seed-torch evaluate --device auto --checkpoint data/models/seed_default.pt \
  --evaluation data/references/color_reasoning_01_reference.json \
  --output data/reports/color_reasoning_01_report.json
```

Select `color_reasoning_02_reference.json` and a different output path for the
supplemental test. `--output` is required. The retained `npm run evaluate` command
uses the original TypeScript engine and legacy JSON/V8 checkpoints.

## Table of contents

- [Color reasoning](#color-reasoning)
- [Short stories](#short-stories)

## Color reasoning

`color_reasoning_01.txt` contains the original six-color experiment.
It has 12,610 sessions. One empty line separates turns within a conversation;
two empty lines separate independent sessions. These turn boundaries let the
PyTorch tokenizer insert a stop token after each assistant answer. Session
separators organize replay sampling; they do not reset the training context.
`color_reasoning_02.txt` is a supplemental corpus for continuing that model:
31,173 sessions covering 22 color spellings, varied questions, distractors, explicit
color changes, and conversations of up to five turns. It includes a sample of the
earlier lessons to help retain existing behavior. Each session fits the existing
256-byte-token context used by this experiment, including special tokens. This is
a corpus design bound, not the current 2048-token model limit.

Regenerate the original corpus with `node scripts/create-color-reasoning-01.ts`
and the supplement with `node scripts/create-color-reasoning-02.ts`.
The generator preserves the original evaluation holdouts and creates a separate
`color_reasoning_02_reference.json`. **Do not train on either reference JSON.**
Use the [experiment guide](../docs/color-reasoning.md#supplemental-training-add2)
for PyTorch training, resume commands, and before/after evaluation. Suggested
settings are starting points; evaluate both references to check new skills and retention.
The [studio and CLI support larger architectures](../docs/pytorch.md#train-on-an-m4-max),
including width 1024, up to twelve layers, sixteen attention heads, and 2048-token
contexts. The studio defaults to width 512, twelve layers, sixteen heads, and context 2048; the explicit
commands here retain smaller configurations for these corpora. The included
small synthetic corpora do not require the maximum model size; compare held-out
results before increasing capacity.

## Short stories

`short_stories.txt` contains 20,000 unique stories distributed evenly across 32
handwritten templates, with varied names, places, and objects. Topics include sharing,
small mistakes, nature, play, feelings, and simple problems. Locations use phrases such
as “on the beach” and “on the hill.” This is a small synthetic practice corpus inspired
by the TinyStories idea, not the official TinyStories dataset or a reproduction of its results.
Templates still repeat, so fluent output or low validation loss can reflect memorization.

Every story ends with `<|endofstory|>`, which Seed encodes as its learned stop token.
The marker is not printed during generation. Each complete story, including its marker
and beginning-token allowance, fits within 256 byte tokens. Paragraph breaks alone do
not mark story endings. Training still samples random windows, which can cut across stories.

Train a fresh model on the included story corpus:

```sh
source .venv/bin/activate
seed-torch train --device auto \
  --corpus data/inputs/short_stories.txt \
  --checkpoint data/models/short-stories.pt \
  --embedding-width 256 --layers 4 --attention-heads 8 \
  --batch-size 8 --steps 10000 --learning-rate 0.0003 --context-window 256
```

These settings are an experimental starting point. Preserve older checkpoints separately
for comparison. Existing checkpoints remain loadable, but the new marker needs training
before it can reliably stop a story. Increasing the model size or update count alone does
not guarantee better stories.

Use **Text generation**, not Chat, to test plain prose. Start with temperature 0.5,
Top K 10, and 120 generated tokens. Try several beginnings, such as `Nina found a`,
`Ben and Lily`, and `A small bird`. Compare spelling, consistent characters, a coherent
ending, and similarity to training text across multiple completions. The model may still
ignore the stop token or stop too early. The color evaluator does not measure stories.
The studio's short validation sample also does not measure story coherence or novelty.

To load the CLI result, stop studio training, choose `short-stories.pt` under
**Select model** in the **Models** panel, and click **Load model**. No copying to the
default filename is needed. Do not run CLI and studio training against the same
file simultaneously.
