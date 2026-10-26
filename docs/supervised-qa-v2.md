# Supervised Q&A v2 corpus

Import `data/inputs/supervised_qa_v2.txt` into the Seed studio and choose
**Supervised Q&A**. This is a synthetic practice dataset, not a pretrained model.
The existing corpora and trained checkpoints are preserved.

## What changed

The new corpus contains **44,000 unique questions**, 2,000 in each of 22 task
families. This increases practice for answer selection, binding objects to colors,
initial versus final states, ownership, transfer direction, locations, conditions,
negation, temporal order, bounded arithmetic, sorting, and parity. Question wording
varies within each family. Both answer positions are queried in distractor tasks;
comparison and parity contain equal numbers of yes and no answers across the
training corpus and external evaluation set together.

Answers use consistent punctuation and retain proper-name capitalization. Numbers
are deliberately bounded: operands usually 0–20, multiplication 0–10, sorting
0–20, and parity 0–99. Subtraction answers are nonnegative. These are finite
practice domains; this corpus does not teach unrestricted arithmetic or general
reasoning. More examples alone cannot guarantee near-perfect performance.

Every record fits within **256 context tokens**, including the BOS allowance and
EOS prediction alignment. The corpus is under the studio's 10 MB limit. The studio
still creates its deterministic 90/10 internal split; changing corpus changes that
split, so new internal scores are not directly comparable with the previous 72%.

## Separate unseen evaluation

`data/references/supervised_qa_v2_unseen.json` has **2,200 questions**, 100 per
family, excluded by exact question from this corpus, the previous input corpora,
and previous reference JSON datasets. Do **not** import either unseen file for
training. The matching `.txt` preserves canonical answer capitalization and
punctuation for strict answer-plus-EOS evaluation. The JSON uses the existing CLI's
lowercase, punctuation-free reference convention.

These are new surface questions sampled from the same vocabulary and templates,
not a test of entirely unseen task types or templates. Exact overlap checks cannot
detect every semantic near-duplicate. The manifest records seed, size, hash, and
maximum token length. Reproduce the artifacts with:

```sh
python3 scripts/create-supervised-qa.py
```

The generator refuses to replace existing output files. Use a separate checkout
if you want to regenerate and compare them.

## Suggested training for the existing Test checkpoint

1. Stop training and copy `data/models/Test.pt` to a new file such as
   `data/models/Test-qa-v2.pt`. Load the copy in the studio.
2. Import `data/inputs/supervised_qa_v2.txt` and choose **Supervised Q&A**.
3. Use these starting settings:

| Setting                             | Value                                               |
| ----------------------------------- | --------------------------------------------------- |
| Model                               | Continue existing model                             |
| Embedding width / layers / heads    | Loaded checkpoint: 512 / 8 / 4                      |
| Assistant-only loss                 | Enabled                                             |
| Learning rate                       | 0.0001                                              |
| Batch size                          | 16 examples                                         |
| Context window                      | 256                                                 |
| Training steps                      | 8,000 additional updates initially                  |
| Validation split                    | 0.1                                                 |
| Random seed                         | 42                                                  |
| Evaluation interval                 | 500 updates                                         |
| Evaluation samples                  | 200 per split                                       |
| Reset optimizer when changing modes | Enabled; does not reset moments when staying in Q&A |
| Start a new learning-rate schedule  | Enabled                                             |
| Chat decoding                       | Top-K = 1                                           |

The loaded architecture and its 512-position embedding remain unchanged; the
256-token setting limits training sequence length. At batch 16, 8,000 updates
sample roughly 3.2 passes through the 39,600-example internal training split.
This is a starting budget, not a claim that 8,000 updates suffice for mastery.
Save snapshots and compare the external evaluation before and after each run.
If both training and validation accuracy continue improving, continue in 4,000-update
blocks. If training accuracy improves but validation stagnates or declines, inspect
failures rather than blindly adding steps. If MPS runs out of memory, use batch 8
and roughly double steps to maintain the same example budget.

For a brand-new model, use the same architecture if desired, a starting learning
rate of 0.0003, and an initial 12,000-update budget. Reusing the current model is
preferable for a controlled comparison and retains its existing learning.

## Evaluate from the CLI

The existing CLI supports evaluation of the external JSON:

```sh
.venv/bin/seed-torch evaluate \
  --device auto --precision float32 \
  --checkpoint data/models/Test-qa-v2.pt \
  --evaluation data/references/supervised_qa_v2_unseen.json \
  --top-k 1 --temperature 1 --seed 42 \
  --output data/reports/supervised-qa-v2-after.json
```

Run this before training too, using a different output filename. This CLI reports
normalized answer accuracy and per-family results; the studio reports stricter
exact answer accuracy including EOS. Do not equate those metrics. Review family
scores and wrong answers, not only the aggregate. A reasonable target is at least
95% on both a larger training sample and the external set, with no weak task
family hidden by the average. That target must be demonstrated by evaluation.

## Initial baseline

Before training on v2, the existing `Test.pt` checkpoint at step 6,000 scored
**312/2,200 (14.18%) strict exact match** on the new unseen set with greedy CPU
decoding. This is a different, broader test from the old validation sample: it
includes unfamiliar wording, objects, directions, and conditions. It is not a
regression caused by training; no training was performed. The result confirms
that the previous 72% did not carry over to this broader question distribution.
The full baseline and per-family failures are saved in
`data/reports/qa-v2-step6000-baseline.json` locally. The checkpoint hash was checked
before and after evaluation to confirm it was unchanged.
