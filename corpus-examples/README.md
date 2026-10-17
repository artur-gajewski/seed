# Seed corpus examples

## Improved toddler corpus

Use **`toddler_reasoning_improved_30000.txt`** for the revised experiment. The original
`toddler_reasoning_30000.txt` is preserved for comparison.

The improved file contains 30,000 question-and-answer examples:

- 14,000 examples selected from the original everyday reasoning, counting, comparison,
  sharing, and sequencing material, with introductory scene wrappers removed and exact
  duplicate pairs removed before selection.
- 4,000 addition examples, including symbolic questions and stories using “receives,”
  “gets,” and “gives.”
- 4,000 subtraction examples, including both `-` and `−` notation and give-away stories.
- 4,000 give-then-receive examples with two explicit calculation steps.
- 4,000 receive-then-give examples with two explicit calculation steps.

Arithmetic uses initial quantities and first operands from 0 to 12, and second-step
operands from 0 to 8. Results are nonnegative. This is a small synthetic practice
set, not a general mathematics curriculum. Names and objects vary, but the wording
still comes from a limited set of templates.

The revision fixes explanations that incorrectly described people as “longer” instead
of their towers, ribbons, or other objects. It also fixes several recurring sequencing
grammar errors. Every complete example, including role labels and special-token
allowance, fits within a 256-byte context window. Randomly sampled training windows
can still cut through an example.

### Separate evaluation questions

**`toddler_reasoning_evaluation.txt`** contains 528 evaluation questions with reference
answers. **Do not import this file for training.** The matching JSON file contains
the same questions and answers plus arithmetic family identifiers for automated checks.

An arithmetic family identifies an operation and its input numbers. All generated
variants of an evaluation family are excluded from the improved training corpus,
regardless of names, objects, or wording. Addition treats reversed operands as the
same family. The previously tested `4 + 2`, `7 - 3`, and `7 - 3 + 2` families are
explicitly reserved for evaluation.

Arithmetic families are also kept together in the training file to reduce overlap
between the studio's contiguous training/validation partitions. A family at the split
boundary can still straddle both partitions, and the general examples still share
templates. The studio's validation loss is therefore not a substitute for answering
the separate evaluation questions correctly.

For a fair comparison, train a **new model** on the improved file using the same
settings and update budget as your baseline. Continuing an old checkpoint is useful
for experimentation, but its previous training may already include the reserved
evaluation problems. Loading a new corpus does not erase that earlier learning.

The settings below remain experimental starting values. More varied training text
does not guarantee arithmetic reasoning; check unseen answers, not just training loss.

To automatically ask a saved checkpoint a sample of evaluation questions, run
`npm run evaluate -- --limit 20`. Omit `--limit 20` for all 528 questions. The evaluator
uses fresh chats, prints accuracy and timing statistics, and saves full results and
incorrect answers under `data/evaluations/`. It does not train the model. See the
[evaluation instructions](../README.md#evaluate-a-trained-checkpoint) for scoring
limitations and checkpoint comparison options.

Regenerate the files deterministically with:

```sh
node scripts/improve-toddler-corpus.ts
```

The generator checks counts, evaluation-family separation, and example lengths.
`tests/toddler-corpus.test.ts` additionally checks equation results, story operands,
answer subjects, and consistency between the text and JSON evaluation files.

This folder contains different corpus examples for training the Seed model.
Each corpus is a collection of text files that can be used to train the model
on specific tasks or domains. The corpora are designed to be simple and focused,
allowing for effective training without overwhelming the model with unnecessary complexity.

## toddler_reasoning_30000.txt

This corpus emphasizes simple reasoning rather than just memorizing conversations.
It will cover counting, comparisons, spatial relationships, cause and effect, sequencing,
object properties, everyday problem-solving, and basic logic.

### Recommended settings for training with toddler_reasoning_30000.txt

**Embedding width: 256**

Determines how much information each token can represent internally. A width of 256 provides a
reasonable balance between learning capacity, memory consumption, and training speed for a small
language model.

**Layers: 4**

Each transformer layer processes information and learns increasingly complex patterns.
Four layers provide enough depth for experimenting with basic language understanding,
comparisons, and simple reasoning without making the model unnecessarily large.

**Attention heads: 8**

Attention heads allow the model to focus on different relationships between tokens.
With an embedding width of 256 and 8 heads, each head operates on 32 dimensions.

**Batch size: 8**

Controls how many training windows are processed per update. A batch size of 8 generally
produces more stable gradient estimates than a batch size of 2, although it requires more memory.
Reduce it to 4 if memory becomes an issue.

**Training steps: 30,000**

Because the model starts with random weights, it needs substantial training to learn language patterns.
A budget of 30,000 steps provides a useful starting experiment for the 30,000-example corpus.

More steps do not automatically mean better results. Monitor training and evaluation loss to detect
overfitting.

**Learning rate: 0.0003**

Controls how much the model's parameters change during each training update. A learning rate of 0.0003
is a conservative starting point that may provide more stable training than 0.001, particularly with
an Adam-like optimizer.

**Context window: 256 tokens**

Determines how much text the model can process at once. Because this model uses UTF-8 bytes as
tokens, 256 tokens represent approximately 256 English ASCII characters.

A 64-token window may be too short to contain a complete question, answer, and explanation.
Increasing it to 256 allows the model to learn from longer reasoning examples.

### Expected training workload

With the recommended configuration:

- Training steps: 30,000
- Batch size: 8
- Context window: 256 tokens
- Tokens processed per step: 2,048
- Total tokens processed: 61,440,000

This assumes each training step processes 8 complete windows of 256 tokens.

The 61.44 million tokens represent repeated exposure to the training corpus,
not 61.44 million unique tokens.

### Monitoring training

Evaluate the model at approximately:

- 1,000 steps
- 5,000 steps
- 10,000 steps
- 20,000 steps
- 30,000 steps

Monitor both training loss and evaluation loss.

A decreasing training loss indicates that the model is learning to predict its training data.
However, this does not necessarily mean it is developing general reasoning abilities.

If training loss continues decreasing while evaluation loss increases, the model may be overfitting.

**Goal:** Train a small language model from random initialization to recognize basic English
patterns, follow simple conversational structures, and demonstrate elementary reasoning on
previously unseen examples.

**Important:** These settings are experimental starting values, not guaranteed optimal settings.
Actual results depend on the training implementation, corpus quality, hardware, and
optimization algorithm.
