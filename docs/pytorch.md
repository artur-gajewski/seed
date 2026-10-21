# PyTorch experiments and GPU training

The web studio, API, and New life use a persistent Python/PyTorch worker. Standalone
experiments use the same model and optimizer. Apple Silicon supports MPS/Metal GPU
training, NVIDIA hosts support CUDA, and CPU remains available.

## Studio and API

Install Python dependencies below, then run `npm run dev` or `npm run build && npm start`.
The API discovers `.venv/bin/python` automatically. `SEED_PYTHON` selects another
interpreter; `SEED_DEVICE=auto|mps|cuda|cpu` selects the initial device (default auto,
which prefers CUDA, then MPS). `SEED_CPU_THREADS` defaults to four.

The Training panel offers available devices for new training jobs and shows the
active backend. Continuing training keeps the model's current device. To change the
device for an existing model, save, restart with `SEED_DEVICE`, and load the checkpoint.

In Checkpoints, choose `color-reasoning-torch.pt` and click **Load checkpoint** to use
your CLI-trained model directly for generation, chat, or continued training. Save
and delete apply to the selected file. The default file is `data/models/checkpoint.pt`;
`SEED_CHECKPOINT` overrides its startup path. Keep separate paths for concurrent CLI
and studio runs. CLI and studio native checkpoints are compatible in both directions.

The API preserves its routes, request validation, and same-origin protections.
`POST /api/train` additionally accepts `device`; `GET /api/status` includes backend,
device, precision, available devices, current checkpoint filename, and `.pt` filenames.
`POST /api/checkpoint/load` accepts `{ "fileName": "color-reasoning-torch.pt" }`;
filenames are confined to the configured checkpoint directory.

Node caches progress events so status requests do not wait for GPU updates.
Training, generation, saving/loading, and New life operations are serialized in the
worker. Stop waits for the current worker operation, then saves. Shutdown requests
stop and save before the worker exits, with a 30-second shutdown grace period.
Abrupt termination can lose unsaved updates.
A failed worker reports an error and rejects pending requests; restart the server to
recover from a process failure, then load a checkpoint.

New life uses separate `.pt` files per preset and atomically saves weights, optimizer,
and replay lessons after each action. Existing JSON/V8 Transformer checkpoints are
migrated without modifying the legacy file. New life retains its correction/replay
behavior; generated replies are not added to lessons.

## Install

Use Python 3.11 or later. Python 3.12 is the tested native macOS environment. From the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./python
seed-torch --help
```

`python/requirements-tested.txt` records the dependency versions validated on Apple
Silicon with Python 3.12. To reproduce that environment, install it with
`python -m pip install -r python/requirements-tested.txt` before installing the package.

PyTorch installation depends on the platform. The default package supports native
Apple Silicon MPS; install an appropriate CUDA build using the
[PyTorch installer](https://pytorch.org/get-started/locally/) on NVIDIA Linux hosts.
Run MPS training directly on macOS, rather than inside the Linux Docker container.
The Docker image includes Python and CPU PyTorch. The retained TypeScript reference CLIs do not require Python; the studio/API and their integration tests do.

## Train on an M4 Max

```sh
seed-torch train \
  --device mps --precision float32 \
  --corpus data/inputs/color_reasoning_01.txt \
  --checkpoint data/models/color-reasoning-torch.pt \
  --embedding-width 128 --layers 3 --attention-heads 4 \
  --batch-size 8 --steps 6000 --learning-rate 0.001 --context-window 256
```

`--device auto` chooses CUDA, then MPS, then CPU. An explicit unavailable device
fails. MPS requires Float32; CPU Float64 is available for numerical comparisons.
No mixed precision or distributed training is enabled. Architecture limits remain
width 512, four layers, eight heads, context 256, and batch eight.

Training preserves the UTF-8 byte tokenizer, special tokens, contiguous 90/10 split,
three fixed validation windows, AdamW groups, clipping, and warmup/cosine schedule.
In an interactive terminal, progress refreshes on the same line with session updates,
unsmoothed training loss, the latest validation loss, and estimated time remaining.
It refreshes after the first update, every 100 updates, at least every five seconds
between updates, and at completion. Remaining time uses the last ten update durations
and displays `calculating` until ten updates are available. Redirected or piped output
retains separate JSON lines without terminal escape codes.
Initial weights reproduce Seed's seed-42 initializer. `--seed` controls training
window sampling and generation, rather than initial weights. Floating-point results
can differ across devices, and identical seeds do not reproduce TypeScript's random batches.

Checkpoints are written atomically every 100 global updates and at completion.
Ctrl+C/SIGTERM saves at the next update boundary and exits with status 130. A new
run refuses to overwrite an existing checkpoint. Avoid sharing paths with a running studio.

Continue training with the same architecture:

```sh
seed-torch train --resume --device mps \
  --corpus data/inputs/color_reasoning_02.txt \
  --checkpoint data/models/color-reasoning-torch.pt \
  --context-window 256 --batch-size 8 --steps 4000 --learning-rate 0.0003
```

As in Seed's existing CLI, `--steps` means additional updates and `--resume` starts
a new learning-rate schedule. Weights, Adam moments, and global optimizer step are
restored. The saved architecture is authoritative; architecture flags are ignored
on resume. This is continued training, not exact recovery of a partially completed
session's sampling/schedule. Corpus hashes and session settings are saved for audit;
source text is not embedded. Native snapshots currently save CPU RNG state only.

## Evaluate

```sh
seed-torch evaluate --device mps \
  --checkpoint data/models/color-reasoning-torch.pt \
  --evaluation data/references/color_reasoning_01_reference.json \
  --output data/reports/color-reasoning-torch.json
```

The evaluator preserves chat history formatting, 48-token generation, greedy top-K
one by default, answer normalization, and per-family scores. Use the add2 reference
file to evaluate supplemental training, then run the original reference again to
check forgetting. Reference data must remain separate from training. `--temperature
0.8 --top-k 0` enables sampled evaluation; cross-device samples need not match.

## Convert legacy checkpoints

Seed checkpoint files can contain V8 binary data despite their `.json` suffix.
Use Node 24 to decode them; Python import expects the resulting interchange JSON.
Conversion supports Transformer v2, including Adam state, but not legacy RNN models.
Converters refuse to overwrite their output.

```sh
node scripts/convert-torch-checkpoint.ts \
  --input data/models/checkpoint.json --output /tmp/seed-interchange.json
seed-torch import --interchange /tmp/seed-interchange.json \
  --checkpoint data/models/imported.pt
```

Export to the original TypeScript reference engine when needed (the PyTorch studio loads `.pt` directly):

```sh
seed-torch export --checkpoint data/models/color-reasoning-torch.pt \
  --interchange /tmp/seed-trained.json
node scripts/convert-torch-checkpoint.ts \
  --input /tmp/seed-trained.json --output data/models/torch-export.json --to-seed
```

The legacy export retains learned weights, moments, and model step; it does not
transfer Python's session metadata or loss history. Float32 GPU training cannot
preserve the original Float64 trajectory. Studio **Load checkpoint** automatically
converts a corresponding legacy `.json` file when the default `.pt` is absent,
retaining the source file. Legacy RNN checkpoints are not supported by PyTorch.

## Validation

```sh
.venv/bin/python -m unittest discover -s python/tests -v
npm test
npm run typecheck
npm run format:check
```

For Python formatting and lint checks, install `python -m pip install -e './python[dev]'`
and run `.venv/bin/ruff format --check python` and `.venv/bin/ruff check python`.

Python tests generate a tiny fixed fixture from the TypeScript engine and compare
initial weights, logits, gradients, two AdamW updates including a batched update,
and optimizer moments in CPU Float64 at `1e-10` absolute/relative tolerance. Tests
also cover tokenization, causal masking, loss reduction, native resumed updates,
and binary Seed checkpoint round trips. MPS tests exercise Float32 parity and
training/save/resume when the device is available; otherwise they explicitly skip.
Node integration tests exercise the real worker, API training/stop/generation,
checkpoint save/load/delete, legacy migration, New life replay/persistence, failed
saves, status responsiveness, and startup failures. CUDA hardware validation remains
to be performed. No GPU speedup is claimed by this port.
