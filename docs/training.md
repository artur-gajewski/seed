# Training and generation

[Documentation](README.md) · [Seed](../README.md)

## Training

The included [color reasoning corpus](../data/inputs/color_reasoning_01.txt) contains
short `User:` / `Assistant:` exchanges. The [supplemental corpus](../data/inputs/color_reasoning_02.txt)
adds colors, varied wording, and longer conversations. A [short-story corpus](../data/inputs/short_stories.txt)
is also available for plain text generation.

See the [corpus examples guide](../data/README.md) for corpus descriptions, suggested training settings, and guidance on monitoring training.

1. Paste or import a UTF-8 `.txt` file: at least 64 Unicode code points, up to 10 MB. The sample is a format example, not enough data for a capable assistant.
2. Choose embedding width, layers, attention heads, context length, batch size, steps, and peak learning rate. Width must be divisible by the number of heads.
3. Click **Start training** when no model is loaded, **Train with new weights** to reset a model, or **Continue existing model** to retain its learned weights. Fresh models start with randomly initialized weights. Batched next-token training uses AdamW, gradient clipping, learning-rate warmup, and cosine decay.
4. Compare training and validation loss. The last 10% of encoded tokens are withheld from training. Validation evaluates up to three fixed windows at the first update, every 100 updates, and the last update. The dashed line shows the most recent validation measurement. Increasing validation loss while training loss drops can indicate overfitting.
5. Test prompts and questions you did not train on. Low loss is not evidence of factual accuracy, reasoning, or reliable instruction following.

Training panel defaults: width 512, twelve layers, sixteen attention heads, a 2048-token context, batch size 8, 30,000 updates, and peak learning rate 0.003. This is about 39.1 million trainable parameters. Training settings and device selection are remembered in this browser between visits; loading a checkpoint explicitly replaces them with that model’s settings.

| Setting                   | Supported values                              |
| ------------------------- | --------------------------------------------- |
| Embedding width in the UI | 16, 32, 48, 64, 128, 256, 384, 512, 768, 1024 |
| Layers                    | 1–12                                          |
| Attention heads in the UI | 1, 2, 4, 8, 16                                |
| Context length            | 8–2048 tokens                                 |
| Batch size                | 1–8                                           |

The context selector offers increasing sizes through 2048 tokens; the default is 2048.
The API and PyTorch CLI also accept intermediate integer widths and head counts within
their limits; width must be divisible by the number of heads. Width 1024 with twelve
layers and context 2048 has 153,782,530 parameters. Larger settings require more
memory and training time. Create a new model to change architecture; continuing training
retains the loaded checkpoint's architecture. See the [PyTorch guide](pytorch.md)
for CLI options.

Choose the training device in the Training panel. See the [PyTorch guide](pytorch.md) for GPU setup. Earlier TypeScript CPU benchmarks are retained in [training performance](training-performance.md).

The split is contiguous and does not deduplicate repeated examples. Repeated or near-identical examples in both partitions can make validation too optimistic. Keep a separate set of genuinely unseen questions for evaluating usefulness. The last partition should be representative of the task you want to measure.

**Train with new weights** creates a fresh model in memory with random weights and starts training it on your text using the selected settings. It replaces the current in-memory model and resets its training step count and loss history. It saves every 100 global updates, at completion, and when stopped. Automatic saves use the currently selected `.pt` file (initially `data/models/seed_default.pt`). This checkpoint is a model snapshot, not a database of your training text.

**Continue existing model** trains further without resetting the learned weights. It retains the loaded architecture, context limit, and optimizer state, and accepts a new corpus. The selected batch size and training length apply to the new session. A resumed session restarts its learning-rate schedule. Architecture controls only affect new models.

Explicitly loading a model prefills Training settings with its architecture and saved
session settings. Refreshing the page restores your last valid browser settings and
device selection, even if they differ from the active model. Regular status polling
preserves these preferences. The session step setting means
additional updates, not the model's lifetime update count.

Architecture, parameter count, vocabulary, and lifetime training steps describe the
active model during training. When idle settings differ from that model, the Training
page labels the architecture, parameter estimate, and vocabulary as **Selected**;
lifetime updates remain labeled **Current model steps**. Batch size changes the work
per update, not the number of parameters. Width 512, four layers, eight heads, and
context 256 give 13,006,082 parameters.

The loss chart fills its card's available width and resizes with the window. The
status badge shows green **Model ready** text when idle with a model loaded and
pulsing yellow **Training in progress** text while training. Only the text pulses;
reduced-motion preferences disable the animation.

There is no pretrained English knowledge. For domain-specific chat, supply many accurate, varied question/answer examples. For broader language ability, a much larger corpus and accelerator-backed training infrastructure would be required. PyTorch runs model operations in a separate process so training does not block HTTP status requests. Model updates, generation, and New life actions are serialized in the worker; a queued generation or learning operation can delay the next update or stop acknowledgment. Distributed training is not implemented.

## Train from the command line

The same PyTorch model and optimizer power command-line experiments and the API:

```sh
source .venv/bin/activate
seed-torch train --device mps \
  --corpus data/inputs/color_reasoning_01.txt \
  --checkpoint data/models/color-reasoning-torch.pt \
  --embedding-width 128 --layers 3 --attention-heads 4 \
  --batch-size 8 --steps 6000 --learning-rate 0.001 --context-window 256
```

These explicit settings reproduce the smaller color experiment rather than the current
studio defaults. The training CLIs retain defaults of width 256, four layers,
eight heads, context 256, batch eight, 30,000 updates, and learning rate 0.0003.

Use `--device cpu` on CPU hosts or `--device cuda` on NVIDIA hosts. Interactive
progress refreshes on one line; redirected output remains JSON lines. Ctrl+C saves
and exits. `--resume` restores weights and optimizer state and runs additional
updates with a new learning-rate schedule. Keep corpus files separately.

CLI `.pt` checkpoints load directly in the studio: select the filename in **Models**
and click **Load model**. Further training saves to that selected file. Avoid
training from CLI and the studio against the same checkpoint path at the same time.

See the [PyTorch guide](pytorch.md) for complete commands and checkpoint
conversion. `npm run train` and `npm run evaluate` retain the original TypeScript
reference CLIs for historical experiments and `.json`/V8 checkpoints.

## Chat and text generation

Use this dialogue format for chat training:

```text
User: Hello!
Assistant: Hello! How can I help?

User: What do plants need?
Assistant: Plants need water, light, and suitable soil.
```

For plain story training, end each story with `<|endofstory|>`. Seed converts that marker
to its learned stop token, so generation can finish a story without printing the marker.
Blank lines remain ordinary paragraph separators. See the [short-story corpus guide](../data/README.md#short-stories)
for a reproducible corpus, training command, and generation settings.

Training inserts a special end-of-response token after assistant answers. Generation can stop when the model predicts it; this is learned behavior, not a canned answer. Ordinary text is encoded as UTF-8 bytes. The tokenizer has 258 tokens: 256 bytes, a beginning token, and an end-of-response token. New characters are accepted without rebuilding a vocabulary. An insufficiently trained byte model may emit malformed UTF-8, which is displayed with replacement characters.

Chat sends recent turns in the `User:` / `Assistant:` format. Requests accept up to 41 alternating messages, starting and ending with a user, with up to 2,000 Unicode code points per message. Chat formatting retains up to 4,096 code points of recent complete turns, but the model actually attends only to its configured context window in **bytes**, which may be much shorter. Text generation accepts prompts up to 4,096 code points.

Chat uses a single-line input that fills the available width. Enter sends a message. Temperature controls randomness. Top K in Text generation limits sampling to the K most likely tokens; 1 chooses the most likely token and 0 uses all candidates. `maxTokens` limits generated tokens, not visible characters. UTF-8 characters may occupy multiple tokens. Chat also removes any subsequent role marker generated after the assistant reply.

Text generation defaults to 1000 new tokens, temperature 2, and Top K 512. Its prompt
and settings are remembered in this browser between visits; saved preferences override
these defaults. Invalid or unreadable saved settings fall back to the defaults.

Chat history and drafts remain in browser memory while switching pages and reset on refresh.
This differs from the saved Text generation prompt. Sending a Chat message does not train the model.

## Evaluate a trained checkpoint

```sh
source .venv/bin/activate
seed-torch evaluate --device mps --checkpoint data/models/seed_default.pt \
  --evaluation data/references/color_reasoning_01_reference.json \
  --output data/reports/color-reasoning.json
```

Use `color_reasoning_02_reference.json` to evaluate supplemental training. Reports
record every reply and scores by question family. Keep both reference datasets
separate from training. Greedy decoding is the default; use `--temperature 0.8
--top-k 0` to check sampled replies. Evaluation does not modify the checkpoint.
See the [PyTorch guide](pytorch.md) and [color experiment guide](color-reasoning.md).
