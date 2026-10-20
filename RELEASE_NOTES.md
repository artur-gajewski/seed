# Seed release notes

A summary of new features, improvements, and changes in Seed.

## Contents

- [2026-10-20](#2026-10-20)
- [2026-10-19](#2026-10-19)
- [2026-10-18](#2026-10-18)
- [2026-10-17](#2026-10-17)

## 2026-10-20

- **Linux and Raspberry Pi acceleration:** Added an optional OpenBLAS native backend
  for Float64 matrix multiplication in training and generation. Linux selects it
  automatically when built; `SEED_TENSOR_BACKEND=openblas` requires it explicitly.
  Build with `npm run native:build` after installing the native dependencies. Model
  architecture and checkpoint formats remain compatible. Raspberry Pi speed gains
  have not yet been measured; the default Docker image still uses JavaScript.
- **New Life model sizes:** Added selectable 38K, 496K, 2.6M, and 13M parameter
  presets, with independent saved learning for each size.
- **New Life checkpoint names:** Checkpoints now use parameter-count filenames such
  as `data/new-life-checkpoint-38K.json`. Legacy filenames migrate automatically
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
- **Story training:** Added a short-story corpus, then improved it with a reproducible
  generator and explicit `<|endofstory|>` markers. The tokenizer maps the marker to
  the learned stop token so generation can end a story without printing it.
- **Corpus cleanup:** Consolidated toddler training material into
  `corpus-examples/toddler_reasoning_corpus.txt`, updated examples and guidance, and
  removed the older `toddler_reasoning_30000.txt`,
  `toddler_reasoning_improved_30000.txt`, and `basic_math.txt` files. Update scripts
  that reference those older paths.
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
- **Checkpoint evaluation:** Added improved toddler reasoning training material,
  a held-out evaluation dataset, and `npm run evaluate` for comparing saved models.
  Reports include numerical accuracy, consistency checks, response timings, and
  individual incorrect answers. These scores measure this dataset rather than
  general reasoning ability.
- Added basic math training material and its generator; the standalone corpus file
  was subsequently removed in the October 19 cleanup.
- Documented the MIT license and refreshed README screenshots.

See the [README](README.md) for current setup and usage instructions and the
[corpus examples guide](corpus-examples/README.md) for training data recommendations.
