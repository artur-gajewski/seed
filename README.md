# Seed

**Seed is a local language-model training studio.** It lets you train a small model from scratch on your own text, watch its learning progress, and experiment with its output.

The interface has four sections:

- **Training:** Paste or import text, configure the model, start or resume training, and track training and validation loss.
- **Text generation:** Give the model a prompt and adjust settings such as temperature and output length.
- **Chat:** Exchange messages with the trained model using a conversation interface.
- **New life:** Grow a separate model from scratch through your messages and taught answers, with automatically saved learning.

The frontend uses **React, TypeScript, and Vite**. A **Node.js 24** backend serves the API and delegates training, generation, and checkpoint storage to a persistent **Python/PyTorch** worker. Apple Silicon uses MPS/Metal GPU acceleration; NVIDIA hosts can use CUDA; CPU execution is also supported. No external inference API is used.

The Training panel’s default model has roughly **3.36 million parameters** and runs on your selected PyTorch device. It starts with random weights, so its output depends entirely on the supplied training data. This makes Seed useful for learning and small experiments; broad assistant capabilities would require much more data and computing power.

You can run it locally or through Docker, and save checkpoints to continue training later.

See the [release notes](RELEASE_NOTES.md) for changes recorded in the Git history.

See the [PyTorch guide](docs/pytorch.md) for GPU setup, command-line experiments, and checkpoint compatibility. The original TypeScript engine remains available for numerical reference tests and legacy experiment scripts.

## Table of contents

- [Screenshot](#screenshot)
- [Run with Docker](#run-with-docker)
- [Run locally](#run-locally)
  - [Run on a MacBook with GPU acceleration](#run-on-a-macbook-with-gpu-acceleration)
- [Training](#training)
- [Train from the command line](#train-from-the-command-line)
- [Chat and text generation](#chat-and-text-generation)
- [New life: teach through chat](#new-life-teach-through-chat)
  - [Try a first lesson](#try-a-first-lesson)
  - [Teach with `!` or a written correction](#teach-with--or-a-written-correction)
  - [How learning works](#how-learning-works)
  - [Continuing later](#continuing-later)
- [Models and storage](#models-and-storage)
- [Source layout](#source-layout)
- [API](#api)
- [Evaluate a trained checkpoint](#evaluate-a-trained-checkpoint)
- [Checks and formatting](#checks-and-formatting)
- [Contributing](#contributing)
- [License](#license)

## Screenshot

This screenshot predates the PyTorch backend and the Models panel described below.

<img width="1896" height="1283" alt="Screenshot 2026-10-19 at 12 03 47" src="https://github.com/user-attachments/assets/df17f4be-9d47-4991-873b-192fbb95d0c7" />

## Run with Docker

```sh
docker compose up --build -d
```

Open **http://localhost:3001**. The container runs as the `node` user and serves the React build and API from one port. The image includes Python and CPU PyTorch. Its build runs Node and Python tests, type checking, and the frontend build. Docker on macOS uses CPU; run natively for MPS GPU training.

```sh
docker compose logs -f studio
docker compose down
```

Checkpoints are stored at `/app/data/models/seed_default.pt` in the Compose `checkpoints` volume, mounted at `/app/data/models`. The volume survives `docker compose down`; `docker compose down -v` deletes it. Host checkpoints are not imported automatically. Included corpora and reference evaluation datasets are available under `/app/data/inputs` and `/app/data/references`.

Without Compose:

```sh
docker build -t seed-studio .
docker run --rm --init -p 127.0.0.1:3001:3001 -v seed-checkpoints:/app/data/models seed-studio
```

## Run locally

Use Node **24** (`nvm use` reads `.nvmrc`) and Python **3.11+** (Python 3.12 recommended on macOS).

```sh
nvm use
npm ci
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ./python
npm run dev
```

Open **http://localhost:5173** or **http://127.0.0.1:5173**. Vite uses port 5173 with `strictPort`, so it fails clearly if that port is occupied rather than switching to an unrecognized origin. The development API explicitly accepts these two page origins even when the Vite proxy rewrites the Host header. Restart `npm run dev` after updating the origin configuration. Other origins remain blocked.

The API defaults to `127.0.0.1:3001`. `PORT` changes its port and the Vite proxy target. `HOST` changes its bind address. Development origin exceptions are only enabled by the development launcher; the production server retains the same-origin check.

For a production build on the host:

```sh
npm run build
npm start
```

Open **http://127.0.0.1:3001**.

### Run on a MacBook with GPU acceleration

The API automatically uses the repository's `.venv/bin/python`. Set `SEED_PYTHON`
to use another interpreter. By default PyTorch selects CUDA, then MPS, then CPU.
On an Apple Silicon Mac, launch natively to use its integrated GPU:

```sh
SEED_DEVICE=mps npm run dev
```

The startup log reports `Training backend: PyTorch (mps, Float32)`. The Training
panel shows the selected device and lets you choose Automatic, CPU, or an available
GPU for a new model. Continuing training keeps the loaded model's current device.
Set `SEED_CPU_THREADS` to change the worker's CPU thread count (default four).
Optional native BLAS addons accelerate only the retained TypeScript reference engine.

## Training

The included [color reasoning corpus](data/inputs/color_reasoning_01.txt) contains
short `User:` / `Assistant:` exchanges. The [supplemental corpus](data/inputs/color_reasoning_02.txt)
adds colors, varied wording, and longer conversations. A [short-story corpus](data/inputs/short_stories.txt)
is also available for plain text generation.

See the [corpus examples guide](data/README.md) for corpus descriptions, suggested training settings, and guidance on monitoring training.

1. Paste or import a UTF-8 `.txt` file: at least 64 Unicode code points, up to 10 MB. The sample is a format example, not enough data for a capable assistant.
2. Choose embedding width, layers, attention heads, context length, batch size, steps, and peak learning rate. Width must be divisible by the number of heads.
3. Click **Start training** when no model is loaded, **Train with new weights** to reset a model, or **Continue existing model** to retain its learned weights. Fresh models start with randomly initialized weights. Batched next-token training uses AdamW, gradient clipping, learning-rate warmup, and cosine decay.
4. Compare training and validation loss. The last 10% of encoded tokens are withheld from training. Validation evaluates up to three fixed windows at the first update, every 100 updates, and the last update. The dashed line shows the most recent validation measurement. Increasing validation loss while training loss drops can indicate overfitting.
5. Test prompts and questions you did not train on. Low loss is not evidence of factual accuracy, reasoning, or reliable instruction following.

Training panel defaults: width 256, four layers, eight attention heads, a 256-token context, batch size 8, 30,000 updates, and peak learning rate 0.0003. This is about 3.36 million trainable parameters. Supported model limits are width up to 512 and context up to 256, up to four layers and eight heads, and batch size up to eight. Even the maximum configuration is only about 13 million parameters, not a large production LLM. Larger settings require more memory and training time.

Choose the training device in the Training panel. See the [PyTorch guide](docs/pytorch.md) for GPU setup. Earlier TypeScript CPU benchmarks are retained in [training performance](docs/training-performance.md).

The split is contiguous and does not deduplicate repeated examples. Repeated or near-identical examples in both partitions can make validation too optimistic. Keep a separate set of genuinely unseen questions for evaluating usefulness. The last partition should be representative of the task you want to measure.

**Train with new weights** creates a fresh model in memory with random weights and starts training it on your text using the selected settings. It replaces the current in-memory model and resets its training step count and loss history. It saves every 100 global updates, at completion, and when stopped. Automatic saves use the currently selected `.pt` file (initially `data/models/seed_default.pt`). This checkpoint is a model snapshot, not a database of your training text.

**Continue existing model** trains further without resetting the learned weights. It retains the loaded architecture, context limit, and optimizer state, and accepts a new corpus. The selected batch size and training length apply to the new session. A resumed session restarts its learning-rate schedule. Architecture controls only affect new models.

Loading a model prefills Training settings with its architecture and saved session
settings. Refreshing the page restores those settings for the active model; regular
status polling preserves subsequent manual edits. The session step setting means
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

Use `--device cpu` on CPU hosts or `--device cuda` on NVIDIA hosts. Interactive
progress refreshes on one line; redirected output remains JSON lines. Ctrl+C saves
and exits. `--resume` restores weights and optimizer state and runs additional
updates with a new learning-rate schedule. Keep corpus files separately.

CLI `.pt` checkpoints load directly in the studio: select the filename in **Models**
and click **Load model**. Further training saves to that selected file. Avoid
training from CLI and the studio against the same checkpoint path at the same time.

See the [PyTorch guide](docs/pytorch.md) for complete commands and checkpoint
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
Blank lines remain ordinary paragraph separators. See the [short-story corpus guide](data/README.md#short-stories)
for a reproducible corpus, training command, and generation settings.

Training inserts a special end-of-response token after assistant answers. Generation can stop when the model predicts it; this is learned behavior, not a canned answer. Ordinary text is encoded as UTF-8 bytes. The tokenizer has 258 tokens: 256 bytes, a beginning token, and an end-of-response token. New characters are accepted without rebuilding a vocabulary. An insufficiently trained byte model may emit malformed UTF-8, which is displayed with replacement characters.

Chat sends recent turns in the `User:` / `Assistant:` format. Requests accept up to 41 alternating messages, starting and ending with a user, with up to 2,000 Unicode code points per message. Chat formatting retains up to 4,096 code points of recent complete turns, but the model actually attends only to its configured context window in **bytes**, which may be much shorter. Text generation accepts prompts up to 4,096 code points.

Enter sends a chat message; Shift + Enter inserts a newline. Temperature controls randomness. Top K in Text generation limits sampling to the K most likely tokens; 1 chooses the most likely token and 0 uses all candidates. `maxTokens` limits generated tokens, not visible characters. UTF-8 characters may occupy multiple tokens. Chat also removes any subsequent role marker generated after the assistant reply.

Chat history and drafts remain in browser memory while switching pages and reset on refresh. Sending a message does not train the model.

## New life: teach through chat

Open **New life** below Chat to grow a separate model from random weights. Send a
short message, then type a correction in the same chat input to supply the response
you wanted. For example, send `What is 1 + 1?` and teach `!1 plus 1 is 2.` Early replies
will be gibberish; a handful of lessons does not produce a fluent assistant.

### Try a first lesson

1. Send `What color is the sky?` in **Say something simple…**.
2. Wait for Seed's reply.
3. Enter `!The sky is blue.` in **Say something simple…** and press Enter to teach and save.
4. Ask the question again, then introduce other short questions and answers.

Enter submits the chat input. After sending a message or correction, focus returns to the message
input, and the conversation follows the latest row automatically. The counters show
remembered lessons, taught answers, and total learning updates.

Use **Up** and **Down** in the input to recall successfully submitted messages and
corrections. Down past the latest entry restores the draft you were writing. Recalled
text can be edited before sending; browsing history does not train the model. Input
history lasts until the page is refreshed.

### Teach with `!` or a written correction

You can also teach directly in **Say something simple…**: prefix your preferred answer
with `!`, for example `!The sky is blue.`. This teaches an answer to the latest message
you sent in the visible conversation, rather than sending a new question. The `!` is
removed before learning and does not count toward the 100-byte limit. Send a message
first; the shortcut needs a previous reply to correct. Both Enter and the send button
support this shortcut, and the input clears and regains focus after saving successfully.

For example, after asking `What color is the sky?`, type `!The sky is blue.` and press
Enter. Seed learns and saves that question-and-answer pair, then acknowledges it in
the chat with a randomly chosen phrase such as “Ok, I will try to learn this.” These
acknowledgments are interface messages, not model-generated answers or training data.
Ask the question again when you want to check its progress. Further corrections still
teach the latest question, rather than the acknowledgment. To teach an earlier
question, ask it again first.

New life also recognizes explicit written corrections, ignoring capitalization:

- `No. The sky is blue.` or `No, the sky is blue.`
- `Nope, the sky is blue.` or `Not quite. The sky is blue.`
- `That's wrong. The sky is blue.` or `Incorrect: The sky is blue.`
- `Actually, the sky is blue.` or `Correction: The sky is blue.`
- `The correct answer is The sky is blue.` or `The right answer: The sky is blue.`
- `It should be The sky is blue.`, `You should say The sky is blue.`, or `I meant The sky is blue.`

These forms use the same teaching action as `!`: the correction prefix is removed,
the supplied answer is paired with your latest message, and learning is saved. The
byte counter measures the extracted answer. The composer shows **Teaching latest
answer** when it recognizes a correction. Combined cues such as `No, the correct
answer is 4.` also work.

Recognition uses explicit text patterns, not the model's understanding of arbitrary
feedback. Bare feedback such as `No.` and questions such as `Actually, what color is
the sky?` remain ordinary messages. Use `!` for an answer that
is itself a question or wording the shortcut does not recognize.

### How learning works

Each message or taught answer triggers 12 small training updates. Training combines
the latest lesson with randomly selected previous lessons to reduce forgetting.
User messages teach language patterns; question-and-answer lessons teach responses.
Generated replies are never automatically added to training. Teaching another answer
to the same question replaces its previous lesson; asking again preserves the taught
answer. Each message and answer must fit in 100 UTF-8 bytes on one line, keeping a
complete lesson within the model's 256-token window. Up to 1,000 distinct questions
are retained; older lessons are not silently removed when this limit is reached.

Every ordinary message trains before generating a reply, including repeated questions. Replies
also use random sampling (temperature 0.8), so the same question can produce different
answers even when saved learning is restored correctly. Refreshing the page does not
itself train the model. Early output may contain `�` replacement characters because
the byte-level model has not yet learned to generate valid UTF-8 consistently.

### Continuing later

New Life offers 38K, 496K, 2.6M, and 13M parameter presets, each with its own learning.
Weights, optimizer state, and replay lessons are saved atomically after each learning
action to `data/models/new-life-checkpoint-38K.pt`, `data/models/new-life-checkpoint-496K.pt`,
`data/models/new-life-checkpoint-2.6M.pt`, or `data/models/new-life-checkpoint-13M.pt`, independently
of the studio's `data/models/seed_default.pt`. Existing Transformer New life JSON/V8
checkpoints are converted to `.pt` when their preset is first loaded, preserving
lessons, weights, and optimizer moments. Legacy files are retained. An existing `.pt`
checkpoint takes precedence. Saved learning reloads automatically
after a server restart.
Conversation display is kept only for the current browser session; saved lessons
include your messages and taught answers. Replies use the current message as context.
This experimental mode can memorize lessons and can still forget earlier skills.

## Models and storage

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
chat display, or drafts. After a restart, choose a file and click **Load model**.

If `seed_default.pt` is missing, loading the default first copies an existing
`checkpoint.pt`, retaining the original. Otherwise it imports `seed_default.json`
when present, or the original `checkpoint.json` (JSON or V8 binary), without modifying
the source. Float64 values are converted to Float32 for GPU execution. Legacy RNN checkpoints are supported
only by the retained TypeScript reference engine and cannot load into PyTorch.

## Source layout

- `data/inputs/`: training corpora, including color reasoning and short stories.
- `data/references/`: reference evaluation datasets.
- `data/reports/`: generated evaluation reports, ignored by Git.
- `data/models/`: generated checkpoints, ignored by Git.
- `server/index.ts`: HTTP routing and startup.
- `server/http.ts`: request bodies, responses, origin checks, static files.
- `server/validation.ts`: input validation and defaults.
- `server/torch-studio.ts`: PyTorch process lifecycle, API validation, cached status, and migration.
- `python/seed_torch/worker.py`: studio training, generation, New life learning, and checkpoint operations.
- `python/seed_torch/model.py`: PyTorch Transformer and optimizer.
- `server/studio.ts`: retained TypeScript reference studio for legacy scripts/tests.
- `server/chat.ts`: conversation formatting and response extraction.
- `server/new-life.ts`: retained TypeScript reference for chat learning and legacy New life tests.
- `server/model.ts`: retained Float64 Transformer reference and version 2 checkpoint conversion.
- `server/tensor.ts`: matrix operations, causal attention, normalization, GELU, and backward derivatives.
- `server/numerics.ts`, `server/accelerated-attention.ts`, and `native/accelerate.cc`: optional native CPU matrix acceleration and attention.
- `server/legacy-model.ts`: original RNN, retained for version 1 checkpoints.
- `shared/api.ts`: request and response types shared with the React app.

Each Transformer block uses pre-layer normalization, masked multi-head self-attention, residual connections, and a GELU feed-forward network with a 4× expansion. Learned token and position embeddings precede the blocks; final normalization and an output projection predict the next byte. The API uses batched Float32 PyTorch tensors and automatic differentiation. The reference engine retains Float64 buffers and manual gradients for parity tests. AdamW uses global gradient clipping at norm 1 and matrix weight decay 0.01.

Architecture reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This implementation uses decoder-only causal attention rather than the paper's complete encoder-decoder architecture. Assistant behavior additionally depends on training objectives and demonstrations; see [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). No RLHF is implemented here.

## API

JSON requests and responses. Errors return `{ "error": "..." }`. No Origin header is needed for command-line clients.

| Method | Path                     | Behavior                                                        |
| ------ | ------------------------ | --------------------------------------------------------------- |
| GET    | `/api/status`            | Model configuration, training job, training and validation loss |
| POST   | `/api/train`             | Start or resume training                                        |
| POST   | `/api/stop`              | Stop after the current worker operation and save                |
| POST   | `/api/generate`          | Generate a text continuation                                    |
| POST   | `/api/chat`              | Generate an assistant reply from recent turns                   |
| GET    | `/api/new-life/status`   | Restore saved New life learning and return lesson/update counts |
| POST   | `/api/new-life/settings` | Select a New Life preset and return its saved status            |
| POST   | `/api/new-life/message`  | Learn from a message, save, and generate a reply                |
| POST   | `/api/new-life/teach`    | Learn a preferred answer and save                               |
| POST   | `/api/checkpoint/save`   | Save weights, optimizer, and chart history                      |
| POST   | `/api/checkpoint/create` | Create and select a fresh model; return HTTP 201                |
| POST   | `/api/checkpoint/load`   | Load the saved model                                            |
| POST   | `/api/checkpoint/delete` | Delete the saved checkpoint while retaining the loaded model    |

```sh
curl http://127.0.0.1:3001/api/train \
  -H 'Content-Type: application/json' \
  -d '{"text":"The sun rises in the morning. The moon is visible at night. Plants need light and water.\n","hiddenSize":64,"layers":2,"heads":4,"sequenceLength":64,"batchSize":2,"steps":1000,"learningRate":0.001,"resume":false}'
```

Create a named, untrained model without a corpus:

```sh
curl http://127.0.0.1:3001/api/checkpoint/create \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-experiment","device":"auto","settings":{"hiddenSize":256,"layers":4,"heads":8,"sequenceLength":256,"batchSize":8,"steps":30000,"learningRate":0.0003}}'
```

Creation returns HTTP 201 and the updated status. The `.pt` extension is appended
when omitted. Names use letters, numbers, dots, hyphens, or underscores, start with a
letter or number, and contain at most 100 characters before the extension. Names
beginning with `new-life-` are reserved. Duplicate names and creation during training
are rejected. New models always start with fresh weights, optimizer state, zero
updates, and empty loss history. Existing model files are preserved.

Training returns HTTP 202; poll `/api/status`. A second training job is rejected while one is running. Add `"device":"mps"`, `"cuda"`, `"cpu"`, or `"auto"` to a new training request. Status includes the active backend/device and available checkpoint filenames. POST `/api/checkpoint/load` with `{ "fileName": "color-reasoning-torch.pt" }` loads a CLI checkpoint from the model directory.

```sh
curl http://127.0.0.1:3001/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"maxTokens":200,"temperature":0.4,"topK":0}'
```

The chat response contains `reply`, `tokens`, `elapsedMs`, and `modelStep`. The generate route takes `prompt` instead of `messages` and returns `completion` instead of `reply`. Token counts include a predicted stop token and are measured before chat trims subsequent role markers. Temperature is 0.1–2; output limits are 1–1,000 tokens.

New life message requests take `{ "message": "What color is the sky?" }` and return
`reply` and `status`. Teaching requests take `{ "message": "What color is the sky?",
"answer": "The sky is blue." }` and return the updated status. Status includes `ready`,
`steps`, `lessons`, and `corrections`. Requests wait for learning and saving to finish;
concurrent New life operations are rejected while it is busy.

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
See the [PyTorch guide](docs/pytorch.md) and [color experiment guide](docs/color-reasoning.md).

## Checks and formatting

```sh
npm test
.venv/bin/python -m unittest discover -s python/tests -v
npm run build
npm run format:check
npm run format
```

The build runs strict TypeScript checking before Vite. Tests cover numerical derivatives for every Transformer parameter group, causal masking, pattern learning from random weights, UTF-8 encoding, exact checkpoint resumption, chat formatting, validation measurement, origin handling, and the API lifecycle. Legacy RNN gradient and checkpoint tests remain. These tests establish implementation behavior, not general assistant capability. There are no browser interaction tests.

Prettier uses two-space indentation, single quotes, semicolons, trailing commas, and a 100-character line width. Keep blank lines between functions and logical sections, one variable declaration per statement, and braces around conditionals and loops.

## Contributing

Contributions are welcome. Open an issue to discuss a bug or idea, or submit a pull request with a focused change and a description of how you tested it. Run `npm test`, the Python suite, `npm run build`, and `npm run format:check` before submitting code changes. See the [PyTorch guide](docs/pytorch.md#validation) for Python formatting and lint checks.

## License

Seed is licensed under the [MIT License](LICENSE). You may use, modify, and redistribute it, including commercially, provided you retain the copyright and license notice.
