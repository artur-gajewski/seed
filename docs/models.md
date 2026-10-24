# Models and checkpoint storage

[Documentation](README.md) · [Seed](../README.md)

The app saves every 100 global training updates, at completion, when stopped, and
on graceful server shutdown. The default is `data/models/seed_default.pt`; select
another `.pt` file in **Models** and click **Load model** to switch the active
model and save destination. `SEED_CHECKPOINT` changes the startup destination.

Click **Create new model**, enter a name, and click **Create** to create and select
a fresh, untrained model using the selected training settings. Current weights, optimizer
state, update count, and loss history are not reused. Existing files are never overwritten
by creation; further training saves to the new checkpoint.

The Models panel has **Load model** and **Create new model** buttons. Manual Save
and Delete buttons are not shown; training progress saves automatically. The save
and delete API endpoints remain available. Checkpoint files are replaced atomically.

Checkpoints contain model configuration, weights, AdamW state, global step, and
studio loss history/settings when saved by the API. They do not contain the corpus,
chat display, or drafts. The Text generation prompt and settings are stored separately
in browser storage, along with Training settings/device and the New Life preset.
Refreshing restores these preferences; explicitly loading a model replaces Training
controls with the checkpoint’s settings. After a restart, choose a file and click **Load model**.

New Life saves learning independently for its eight presets in
`data/models/new-life-checkpoint-<parameters>.pt`, from `38K` to `153M`. See the
[preset table and input limits](new-life.md#continuing-later). Changing the Training
architecture or context requires a fresh model; continuing training retains the
loaded architecture and context capacity. The current maximum context is 2048 tokens.

If `seed_default.pt` is missing, loading the default first copies an existing
`checkpoint.pt`, retaining the original. Otherwise it imports `seed_default.json`
when present, or the original `checkpoint.json` (JSON or V8 binary), without modifying
the source. Float64 values are converted to Float32 for GPU execution. Legacy RNN checkpoints are supported
only by the retained TypeScript reference engine and cannot load into PyTorch.
