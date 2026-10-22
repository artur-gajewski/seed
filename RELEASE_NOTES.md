# Seed release notes

A summary of new features, improvements, and changes in Seed.

## Contents

- [Unreleased](#unreleased)
- [2026-10-21](#2026-10-21)
- [2026-10-20](#2026-10-20)
- [2026-10-19](#2026-10-19)
- [2026-10-18](#2026-10-18)
- [2026-10-17](#2026-10-17)

## Unreleased

- **Expanded architectures:** Added embedding widths 768 and 1024 to the studio,
  layers up to twelve, and attention heads up to sixteen. API, CLI, and checkpoint
  validation support the expanded limits. Defaults and the 256-token context cap
  remain unchanged; changing architecture requires a new model.
- **Color corpus boundaries:** Separated conversation turns so every assistant answer
  receives a learned stop token. Generators preserve complete sessions for replay;
  regression tests check answer termination and the 256-token session limit.
- **PyTorch backend:** Studio training, generation, Chat, and New Life now run in a
  persistent Python worker. Automatic device selection prefers CUDA, then Apple
  Silicon MPS, then CPU. The API retains request validation and origin checks;
  status polling remains responsive while model operations run in the worker.
- **GPU experiments:** Added the `seed-torch` training/evaluation CLI, single-line
  terminal progress, CPU Float64 parity tests, and conversion between native `.pt`
  files and the retained TypeScript engine's Transformer checkpoints. Docker now
  includes Python and CPU PyTorch. MPS execution was validated on an M4 Max;
  CUDA hardware validation and matched speed comparisons remain outstanding.
- **Models panel:** Added **Select model**, **Load model**, and **Create new model**.
  Creation asks for a name, always initializes fresh weights and optimizer state
  from the selected settings, resets updates/history, saves without overwriting an
  existing file, and selects the new model. Removed manual Save and Delete buttons;
  automatic saving and the explicit API endpoints remain available.
- **Model storage:** The default is now `data/models/seed_default.pt`. Loading it
  can copy an existing `checkpoint.pt`, preserving the source, or import legacy
  Transformer JSON/V8 checkpoints. CLI-trained `.pt` models load directly. New Life
  uses separate native checkpoints per preset and preserves lessons during migration.
- **Training controls:** Loading or reconnecting to an active model fills in its
  architecture and saved session settings. Idle architecture edits show a labeled
  parameter estimate; live training shows the actual model and separate lifetime
  and session update counts. Renamed the reset action to **Train with new weights**.
- **Studio layout:** The loss plot fills its card width. The ready badge uses green
  text; training uses yellow text with a text-only pulse and reduced-motion support.
  Removed the Model limits panel and adjusted heading/panel spacing.

## 2026-10-21

- **Essay:** Added [How Seed works](docs/how-seed-works.md), explaining
  tokenization, Transformers, training, and generation through Seed's implementation.
  Covers the Training and New Life sections, including corpus validation, teaching
  answers, lesson replay, saved learning, and model limitations.

## 2026-10-20

- **Data layout:** Moved training corpora to `data/inputs`, reference evaluations
  to `data/references`, and generated checkpoints to Git-ignored `data/models`.
  The studio now uses `data/models/checkpoint.json`; New Life models, CLI examples,
  generators, and Docker storage use the same layout. Generated score reports
  are saved in Git-ignored `data/reports` while reference evaluation datasets are tracked.
  Reference files now use `_reference.json` filenames, including
  `color_reasoning_01_reference.json` and `color_reasoning_02_reference.json`.
- **Corpus tools:** Focused corpus creation and evaluation on color reasoning,
  removed the other corpus creation and evaluation tools and their related tests,
  retained `short_stories.txt`, and updated setup and evaluation examples. The
  built-in sample now uses color questions.
- **Supplemental color training:** Added `color_reasoning_02.txt` with 31,173
  sessions spanning 22 color spellings, varied wording, distractors, color changes,
  and conversations of up to five turns. Includes replay of earlier lessons,
  a deterministic generator, 2,088 separate evaluation cases, and documentation
  for continuing and comparing checkpoints. The color evaluator now accepts
  `--evaluation PATH` to select a test dataset.
- **Color reasoning experiment:** Added `data/inputs/color_reasoning_01.txt` with
  12,610 training examples covering six colors, 97 objects, varied question wording,
  distractors, and conversation continuations. Includes a reproducible corpus
  generator, 2,306 separate evaluation examples, and tests for held-out pairs and
  unseen objects.
- **Color model evaluation:** Added `scripts/evaluate-color-reasoning.ts` to evaluate saved
  checkpoints, ask individual questions, and compare greedy decoding with sampled
  replies. Corpus generation is separated into `scripts/create-color-reasoning-01.ts`
  and `scripts/create-color-reasoning-02.ts`. JSON reports include individual
  answers and scores by question family.
  `npm run evaluate` defaults to the studio checkpoint and writes a dataset-specific
  report such as `data/reports/color_reasoning_01_report.json`. Custom output
  directories are created automatically, and the evaluator rejects an output path
  that would overwrite its reference dataset.
  The [experiment guide](docs/color-reasoning.md) documents training, resuming,
  evaluation, loading the model in Seed, and measured results and limitations.
- **Performance documentation:** Removed the raw benchmark results file and obsolete
  checkout instructions. Retained timings are labeled as historical measurements,
  with commands for benchmarking the current engine.
- **Linux and Raspberry Pi acceleration:** Added an optional OpenBLAS native backend
  for Float64 matrix multiplication in training and generation. Linux selects it
  automatically when built; `SEED_TENSOR_BACKEND=openblas` requires it explicitly.
  Build with `npm run native:build` after installing the native dependencies. Model
  architecture and checkpoint formats remain compatible. Raspberry Pi speed gains
  have not yet been measured; the default Docker image still uses JavaScript.
- **New Life model sizes:** Added selectable 38K, 496K, 2.6M, and 13M parameter
  presets, with independent saved learning for each size.
- **New Life checkpoint names:** Checkpoints now use parameter-count filenames such
  as `data/models/new-life-checkpoint-38K.json`. Legacy filenames migrate automatically
  when their preset is first loaded; an existing parameter-count checkpoint takes
  precedence. Files use binary serialization despite their `.json` extension;
  existing JSON contents remain readable.
- Added tables of contents to the main README and corpus examples guide.

## 2026-10-19

- **Headless training:** Added `npm run train -- --corpus <file> --checkpoint <file>`
  to train without the frontend or API. Supports explicit training settings,
  `--resume`, periodic checkpoint saving, and saving on Ctrl+C or SIGTERM.
- **CLI progress:** Interactive terminals refresh progress on one line; redirected
  output keeps separate log lines.
- **Larger models:** Added embedding widths 384 and 512 while retaining the default
  width of 256.
- **Corpus visibility:** The Training page now displays the imported corpus filename.
- **Story training:** Added a short-story corpus with explicit `<|endofstory|>` markers. The tokenizer maps the marker to
  the learned stop token so generation can end a story without printing it.
- Refreshed the README screenshot.

## 2026-10-18

- **Corrections through chat:** New Life recognizes explicit written corrections
  such as `No, the answer is ...` as teaching input. Questions and feedback without
  a replacement answer remain ordinary messages.
- **Input history:** Use Up and Down in the New Life input to recall previous
  messages and teaching commands.
- **Training editor:** Start with an empty corpus editor and disable training until
  text is provided.
- Increased interface font sizes and improved New Life layout, instructions, and
  explanatory text.

## 2026-10-17

- **Initial studio:** Introduced local training from random weights, configurable
  Transformer models, training and validation loss charts, text generation, chat,
  and checkpoint saving and restoration. The TypeScript engine includes automatic
  differentiation and AdamW optimization.
- **Local and container setup:** Included the React/Vite frontend, Node.js 24 API,
  Docker support, optional Apple Accelerate CPU acceleration, benchmarks, and
  numerical and API tests.
- **New Life:** Added a separate experimental model that learns through messages
  and taught answers, automatically saves learning, and reloads it after a server
  restart. Added `!`-prefixed teaching from the chat input.
- Documented the MIT license and refreshed README screenshots.

See the [README](README.md) for current setup and usage instructions and the
[corpus examples guide](data/README.md) for training data recommendations.
