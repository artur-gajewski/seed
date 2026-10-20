# Seed

**Seed is a local language-model training studio.** It lets you train a small model from scratch on your own text, watch its learning progress, and experiment with its output.

The interface has four sections:

- **Training:** Paste or import text, configure the model, start or resume training, and track training and validation loss.
- **Text generation:** Give the model a prompt and adjust settings such as temperature and output length.
- **Chat:** Exchange messages with the trained model using a conversation interface.
- **New life:** Grow a separate model from scratch through your messages and taught answers, with automatically saved learning.

The frontend uses **React, TypeScript, and Vite**. A **Node.js 24** backend handles the API, training, generation, and checkpoint storage. The Transformer engine—including attention, gradients, and optimization—is implemented directly in TypeScript, without machine-learning libraries or external inference APIs.

The Training panel’s default model has roughly **3.36 million parameters** and runs on the CPU, with optional native acceleration on macOS. It starts with random weights, so its output depends entirely on the supplied training data. This makes Seed useful for learning and small experiments; broad assistant capabilities would require much more data and computing power.

You can run it locally or through Docker, and save checkpoints to continue training later.

See the [release notes](RELEASE_NOTES.md) for changes recorded in the Git history.

## Table of contents

- [Screenshot](#screenshot)
- [Run with Docker](#run-with-docker)
- [Run locally](#run-locally)
  - [Run on a MacBook with acceleration](#run-on-a-macbook-with-acceleration)
  - [Raspberry Pi and Linux OpenBLAS acceleration](#raspberry-pi-and-linux-openblas-acceleration)
- [Training](#training)
- [Train from the command line](#train-from-the-command-line)
- [Chat and text generation](#chat-and-text-generation)
- [New life: teach through chat](#new-life-teach-through-chat)
  - [Try a first lesson](#try-a-first-lesson)
  - [Teach with `!` or a written correction](#teach-with--or-a-written-correction)
  - [How learning works](#how-learning-works)
  - [Continuing later](#continuing-later)
- [Checkpoints and storage](#checkpoints-and-storage)
- [Source layout](#source-layout)
- [API](#api)
- [Evaluate a trained checkpoint](#evaluate-a-trained-checkpoint)
- [Checks and formatting](#checks-and-formatting)
- [Contributing](#contributing)
- [License](#license)

## Screenshot

<img width="1896" height="1283" alt="Screenshot 2026-10-19 at 12 03 47" src="https://github.com/user-attachments/assets/df17f4be-9d47-4991-873b-192fbb95d0c7" />

## Run with Docker

```sh
docker compose up --build -d
```

Open **http://localhost:3001**. The container runs as the `node` user and serves the React build and API from one port. The image build runs tests, type checking, and the frontend build.

```sh
docker compose logs -f studio
docker compose down
```

Checkpoints are stored at `/app/data/models/checkpoint.json` in the Compose `checkpoints` volume, mounted at `/app/data/models`. The volume survives `docker compose down`; `docker compose down -v` deletes it. Host checkpoints are not imported automatically. Included corpora and reference evaluation datasets are available under `/app/data/inputs` and `/app/data/references`.

Without Compose:

```sh
docker build -t seed-studio .
docker run --rm --init -p 127.0.0.1:3001:3001 -v seed-checkpoints:/app/data/models seed-studio
```

## Run locally

Use Node **24** (`nvm use` reads `.nvmrc`).

```sh
npm ci
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

### Run on a MacBook with acceleration

On an Apple Silicon MacBook (M-series), run Seed directly on macOS to use Apple Accelerate CPU acceleration. The Linux Docker image uses the JavaScript backend.

1. Install Node **24** and Xcode Command Line Tools. If the tools are not installed, run `xcode-select --install` and finish the installation before continuing.
2. From the repository root, install dependencies, build the native addon, and start Seed:

   ```sh
   nvm use
   npm ci
   npm run native:build
   npm run dev
   ```

   If you installed Node 24 without nvm, skip `nvm use`.

3. Open **http://localhost:5173**. Seed automatically detects the built addon. Confirm the terminal shows `Tensor backend: accelerate (CPU, Float64)`.

Build the addon once, then use `npm run dev` for subsequent launches. Restart the server if you build the addon while Seed is running.

For a production build with acceleration:

```sh
npm run build
npm start
```

Open **http://127.0.0.1:3001**. Production launches also detect the addon automatically. Optionally, set `SEED_TENSOR_BACKEND=accelerate` to require acceleration and fail at startup if it is unavailable.

If the native build reports missing Node headers, set `NODE_INCLUDE_DIR` to the directory containing `node_api.h`:

```sh
NODE_INCLUDE_DIR=/path/to/node/include/node npm run native:build
```

Run `npm run benchmark` to measure training speed on your MacBook; its output reports the selected backend. See [training performance](docs/training-performance.md) for benchmark settings and results.

### Raspberry Pi and Linux OpenBLAS acceleration

On Raspberry Pi 5, use a 64-bit OS and Node.js 24. Install the native build dependencies,
then build the optional OpenBLAS addon:

```bash
sudo apt update
sudo apt install g++ pkg-config libopenblas-dev
nvm use
npm ci
npm run native:build
npm run build
OPENBLAS_NUM_THREADS=4 SEED_TENSOR_BACKEND=openblas npm start
```

The build locates Node headers from the active Node installation, node-gyp's cache,
or system include directories. If needed, set `NODE_INCLUDE_DIR` to the matching
Node 24 `include/node` directory. `CXX` can select the C++ compiler and
`PKG_CONFIG_PATH` can locate a custom OpenBLAS installation.

Linux automatically selects OpenBLAS when `native/openblas.node` is present;
otherwise it uses JavaScript. Setting `SEED_TENSOR_BACKEND=openblas` requires the
addon and fails clearly if unavailable. `SEED_TENSOR_BACKEND=javascript` forces
the portable implementation. A broken installed addon surfaces its load error.
The addon accelerates Float64 matrix multiplication in both forward and backward
passes; model architecture and checkpoint formats stay compatible. Other operations
still run in JavaScript. Rebuild native addons on the target machine after copying
source to a different platform. The default Docker image remains JavaScript-only.

Compare the backends on the Pi with the same bounded workload:

```bash
BENCH_WIDTH=128 BENCH_LAYERS=2 BENCH_HEADS=4 BENCH_BATCH=4 BENCH_CONTEXT=64 SEED_TENSOR_BACKEND=javascript npm run benchmark
OPENBLAS_NUM_THREADS=4 BENCH_WIDTH=128 BENCH_LAYERS=2 BENCH_HEADS=4 BENCH_BATCH=4 BENCH_CONTEXT=64 SEED_TENSOR_BACKEND=openblas npm run benchmark
```

Try `OPENBLAS_NUM_THREADS=1`, `2`, and `4`; more threads can add overhead for small
matrices. For OpenMP builds, use `OMP_NUM_THREADS` instead, as described in the
[OpenBLAS runtime documentation](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/).
Run `SEED_TENSOR_BACKEND=openblas npm test` after building to check numerical parity,
gradients, optimizer state, and resumed training against JavaScript. Performance gains
on Raspberry Pi have not yet been measured.

## Training

The included [color reasoning corpus](data/inputs/color_reasoning_01.txt) contains
short `User:` / `Assistant:` exchanges. The [supplemental corpus](data/inputs/color_reasoning_02.txt)
adds colors, varied wording, and longer conversations. A [short-story corpus](data/inputs/short_stories.txt)
is also available for plain text generation.

See the [corpus examples guide](data/README.md) for corpus descriptions, suggested training settings, and guidance on monitoring training.

1. Paste or import a UTF-8 `.txt` file: at least 64 Unicode code points, up to 10 MB. The sample is a format example, not enough data for a capable assistant.
2. Choose embedding width, layers, attention heads, context length, batch size, steps, and peak learning rate. Width must be divisible by the number of heads.
3. Start a new model. All weights are initialized randomly. Batched next-token training uses AdamW, gradient clipping, learning-rate warmup, and cosine decay.
4. Compare training and validation loss. The last 10% of encoded tokens are withheld from training. Validation evaluates up to three fixed windows at the first update, every 100 updates, and the last update. The dashed line shows the most recent validation measurement. Increasing validation loss while training loss drops can indicate overfitting.
5. Test prompts and questions you did not train on. Low loss is not evidence of factual accuracy, reasoning, or reliable instruction following.

Training panel defaults: width 256, four layers, eight attention heads, a 256-token context, batch size 8, 30,000 updates, and peak learning rate 0.0003. This is about 3.36 million trainable parameters. Supported CPU limits are width up to 512 and context up to 256, up to four layers and eight heads, and batch size up to eight. Even the maximum configuration is only about 13 million parameters, not a large production LLM. Larger settings cost substantially more CPU time.

On macOS, run `npm run native:build` and restart the server to enable optional Apple Accelerate CPU acceleration. The portable JavaScript backend remains available. See [training performance](docs/training-performance.md) for setup requirements, benchmarks, correctness checks, and GPU trade-offs.

The split is contiguous and does not deduplicate repeated examples. Repeated or near-identical examples in both partitions can make validation too optimistic. Keep a separate set of genuinely unseen questions for evaluating usefulness. The last partition should be representative of the task you want to measure.

**Train a new model** creates a fresh model in memory with random weights and starts training it on your text using the selected settings. It replaces the current in-memory model and resets its training step count and loss history. It does not immediately write a checkpoint file or overwrite the saved checkpoint. Click **Save checkpoint** to write the current model to `data/models/checkpoint.json`, replacing any previous checkpoint. This checkpoint is a model snapshot, not a database of your training text.

**Continue existing model** trains further without resetting the learned weights. It retains the loaded architecture, context limit, and optimizer state, and accepts a new corpus. The selected batch size and training length apply to the new session. A resumed session restarts its learning-rate schedule. Architecture controls only affect new models.

There is no pretrained English knowledge. For domain-specific chat, supply many accurate, varied question/answer examples. For broader language ability, a much larger corpus and accelerator-backed training infrastructure would be required. The portable engine uses one CPU thread; the optional Accelerate and OpenBLAS backends use native CPU matrix operations. None of these backends uses GPU or distributed training. Individual updates and generation calls run synchronously on the API thread, yielding only between training blocks. Large settings can delay other API requests.

## Train from the command line

Use Node 24 (`nvm use`) and install dependencies with `npm ci`. Run training without
starting the frontend or API:

```sh
npm run train -- --corpus data/inputs/color_reasoning_01.txt --checkpoint data/models/checkpoint.json
```

Both `--corpus` (input text) and `--checkpoint` (output model) are required. Missing
file arguments produce an error. Every omitted training setting uses the Training panel
default, including when resuming. All settings can also be supplied explicitly:

```sh
npm run train -- \
  --corpus data/inputs/color_reasoning_01.txt \
  --embedding-width 256 \
  --layers 4 \
  --attention-heads 8 \
  --batch-size 8 \
  --steps 30000 \
  --learning-rate 0.0003 \
  --context-window 256 \
  --checkpoint data/models/checkpoint.json
```

Embedding width supports up to 512, including 384; it must be divisible by the number
of attention heads. Context is measured in UTF-8 byte tokens. Run `npm run train -- --help`
for all options and supported ranges. Relative paths resolve from your working directory.

The command prints the numerical backend, effective settings, training progress, and losses.
In an interactive terminal, progress refreshes on the same line. Redirected or piped output
uses separate lines for readable logs.
Progress includes estimated time remaining after every 100 updates in the current session,
as well as every five seconds and when training stops. Estimates use the last ten update
durations and display `calculating` until ten updates are available.
It uses the same training engine, validation split, and learning-rate schedule as the studio.
It replaces the selected checkpoint every 100 total model updates and saves again when
training finishes. Ctrl+C or SIGTERM stops training and saves progress before exiting.
An interrupted run exits with code 130; other failures exit with code 1.

To continue a saved model for another 5,000 updates:

```sh
npm run train -- \
  --corpus data/inputs/color_reasoning_01.txt \
  --checkpoint data/models/checkpoint.json \
  --resume --steps 5000
```

Resume restores the learned weights and optimizer state. Omitted settings use the defaults
listed above; `--steps` means additional updates. Architecture settings must match the
checkpoint, so supply them explicitly if the saved model differs from the defaults.
Batch size and learning rate can change. Each session restarts the
learning-rate schedule. Keep the corpus separately; studio checkpoints do not include it.

Use `--checkpoint data/models/checkpoint.json` to save to the studio's checkpoint. Use a different
path when running independent experiments or when the web studio is running, so they do
not overwrite each other's checkpoints. In the studio, use **Load checkpoint** after CLI
training to load the saved model. To evaluate it:

```sh
npm run evaluate -- --checkpoint data/models/checkpoint.json
```

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
action to `data/models/new-life-checkpoint-38K.json`, `data/models/new-life-checkpoint-496K.json`,
`data/models/new-life-checkpoint-2.6M.json`, or `data/models/new-life-checkpoint-13M.json`, independently
of the studio's `data/models/checkpoint.json`. These files use binary serialization despite
the `.json` extension; existing JSON contents remain readable. Legacy New Life
filenames are migrated automatically when their preset is first loaded. An existing
parameter-count checkpoint takes precedence. Saved learning reloads automatically
after a server restart.
Conversation display is kept only for the current browser session; saved lessons
include your messages and taught answers. Replies use the current message as context.
This experimental mode can memorize lessons and can still forget earlier skills.

## Checkpoints and storage

The app **automatically saves a checkpoint every 100 training updates**, based on the model's total update count, including across resumed training sessions. Checkpoints are written to `data/models/checkpoint.json` locally, or `/app/data/models/checkpoint.json` inside Docker. Each save atomically replaces the single checkpoint file.

Click **Save checkpoint** to save immediately, including updates since the last automatic save. After a restart, click **Load checkpoint** to restore the latest saved model. Updates since the last save are lost if the server stops.

Click **Delete checkpoint** and confirm in the modal to permanently remove the saved checkpoint. Cancel or press Escape to keep it. Deleting keeps the currently loaded model and any training in progress available in memory; you can save the model again. Deleting an already missing checkpoint is harmless.

New studio checkpoints use Node’s binary serialization to avoid the JavaScript string-size
limit for larger models. The filename remains `checkpoint.json` for compatibility; existing
JSON checkpoints still load. New life continues to use its separate JSON checkpoint.

Version 2 Transformer checkpoints include architecture, tokenizer identity, configuration, weights, optimizer moments, update count, and chart history. They do **not** include the corpus, validation token partition, chat history, or draft. Keep the corpus file separately for future training. Unsaved weights are lost on server shutdown. Click **Load** after restarting to restore the saved model.

Version 1 RNN checkpoints still load as RNNs, including their original optimizer state and vocabulary restrictions. They are labeled as legacy in the interface. Loading one does not convert it into a Transformer. Start a new model to use the new architecture. Validation evaluation is only available for the Transformer.

## Source layout

- `data/inputs/`: training corpora, including color reasoning and short stories.
- `data/references/`: reference evaluation datasets.
- `data/reports/`: generated evaluation reports, ignored by Git.
- `data/models/`: generated checkpoints, ignored by Git.
- `server/index.ts`: HTTP routing and startup.
- `server/http.ts`: request bodies, responses, origin checks, static files.
- `server/validation.ts`: input validation and defaults.
- `server/studio.ts`: model state, batches, data partitioning, validation, checkpoints.
- `server/chat.ts`: conversation formatting and response extraction.
- `server/new-life.ts`: independent chat learning, lesson replay, and New life checkpoint storage.
- `server/model.ts`: decoder-only Transformer and version 2 checkpoints.
- `server/tensor.ts`: matrix operations, causal attention, normalization, GELU, and backward derivatives.
- `server/numerics.ts`, `server/accelerated-attention.ts`, and `native/accelerate.cc`: optional native CPU matrix acceleration and attention.
- `server/legacy-model.ts`: original RNN, retained for version 1 checkpoints.
- `shared/api.ts`: request and response types shared with the React app.

Each Transformer block uses pre-layer normalization, masked multi-head self-attention, residual connections, and a GELU feed-forward network with a 4× expansion. Learned token and position embeddings precede the blocks; final normalization and an output projection predict the next byte. Weights are row-major `Float64Array` buffers. Reverse-mode derivatives are recorded on a per-example tape and accumulated across a batch. AdamW uses global gradient clipping at norm 1 and matrix weight decay 0.01.

Architecture reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This implementation uses decoder-only causal attention rather than the paper's complete encoder-decoder architecture. Assistant behavior additionally depends on training objectives and demonstrations; see [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). No RLHF is implemented here.

## API

JSON requests and responses. Errors return `{ "error": "..." }`. No Origin header is needed for command-line clients.

| Method | Path                     | Behavior                                                        |
| ------ | ------------------------ | --------------------------------------------------------------- |
| GET    | `/api/status`            | Model configuration, training job, training and validation loss |
| POST   | `/api/train`             | Start or resume training                                        |
| POST   | `/api/stop`              | Stop after the current synchronous training block               |
| POST   | `/api/generate`          | Generate a text continuation                                    |
| POST   | `/api/chat`              | Generate an assistant reply from recent turns                   |
| GET    | `/api/new-life/status`   | Restore saved New life learning and return lesson/update counts |
| POST   | `/api/new-life/message`  | Learn from a message, save, and generate a reply                |
| POST   | `/api/new-life/teach`    | Learn a preferred answer and save                               |
| POST   | `/api/checkpoint/save`   | Save weights, optimizer, and chart history                      |
| POST   | `/api/checkpoint/load`   | Load the saved model                                            |
| POST   | `/api/checkpoint/delete` | Delete the saved checkpoint while retaining the loaded model    |

```sh
curl http://127.0.0.1:3001/api/train \
  -H 'Content-Type: application/json' \
  -d '{"text":"The sun rises in the morning. The moon is visible at night. Plants need light and water.\n","hiddenSize":64,"layers":2,"heads":4,"sequenceLength":64,"batchSize":2,"steps":1000,"learningRate":0.001,"resume":false}'
```

Training returns HTTP 202; poll `/api/status`. A second training job is rejected while one is running.

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

Save the model using **Save checkpoint**, then evaluate it against the original
color reasoning test from the repository root:

```sh
npm run evaluate -- --checkpoint data/models/checkpoint.json \
  --output data/reports/color-reasoning.json
```

The evaluator asks 2,306 questions about held-out color/object combinations, unseen
objects, varied wording, and conversation continuations. It uses greedy decoding
by default, normalizes case and final punctuation, and compares exact answers.
The JSON report records every reply, pass/fail results, and scores by question family.
Without `--output`, reports are written to `data/reports/<dataset>_report.json`.
Existing report files are replaced, so use separate output filenames for comparisons.

For the supplemental color test:

```sh
npm run evaluate -- --checkpoint data/models/checkpoint.json \
  --evaluation data/references/color_reasoning_02_reference.json \
  --output data/reports/color-reasoning-02.json
```

Keep both evaluation files separate from training material. Compare checkpoints
using the same dataset and generation settings. Add `--temperature 0.8 --top-k 0`
to test Chat's default sampling; sampled replies can vary. These controlled tests
do not measure general reasoning or story quality. See the
[color experiment guide](docs/color-reasoning.md) for training, resume commands,
and recorded results. The evaluator loads a checkpoint in a separate process and
does not modify it or require the API server to run.

## Checks and formatting

```sh
npm test
npm run build
npm run format:check
npm run format
```

The build runs strict TypeScript checking before Vite. Tests cover numerical derivatives for every Transformer parameter group, causal masking, pattern learning from random weights, UTF-8 encoding, exact checkpoint resumption, chat formatting, validation measurement, origin handling, and the API lifecycle. Legacy RNN gradient and checkpoint tests remain. These tests establish implementation behavior, not general assistant capability. There are no browser interaction tests.

Prettier uses two-space indentation, single quotes, semicolons, trailing commas, and a 100-character line width. Keep blank lines between functions and logical sections, one variable declaration per statement, and braces around conditionals and loops.

## Contributing

Contributions are welcome. Open an issue to discuss a bug or idea, or submit a pull request with a focused change and a description of how you tested it. Run `npm test`, `npm run build`, and `npm run format:check` before submitting code changes.

## License

Seed is licensed under the [MIT License](LICENSE). You may use, modify, and redistribute it, including commercially, provided you retain the copyright and license notice.
