# Seed

**Seed is a local language-model training studio.** It lets you train a small model from scratch on your own text, watch its learning progress, and experiment with its output.

The interface has four sections:

- **Training:** Paste or import text, configure the model, start or resume training, and track training and validation loss.
- **Text generation:** Give the model a prompt and adjust settings such as temperature and output length.
- **Chat:** Exchange messages with the trained model using a conversation interface.
- **New life:** Grow a separate model from scratch through your messages and taught answers, with automatically saved learning.

The frontend uses **React, TypeScript, and Vite**. A **Node.js 24** backend handles the API, training, generation, and checkpoint storage. The Transformer engine—including attention, gradients, and optimization—is implemented directly in TypeScript, without machine-learning libraries or external inference APIs.

The default model has roughly **137,000 parameters** and runs on the CPU, with optional native acceleration on macOS. It starts with random weights, so its output depends entirely on the supplied training data. This makes Seed useful for learning and small experiments; broad assistant capabilities would require much more data and computing power.

You can run it locally or through Docker, and save checkpoints to continue training later.

## Screenshot

<img width="1517" height="1216" alt="Screenshot 2026-10-17 at 17 27 06" src="https://github.com/user-attachments/assets/085d27c2-2b82-4941-9994-fd60f8fa22c5" />

## Run with Docker

```sh
docker compose up --build -d
```

Open **http://localhost:3001**. The container runs as the `node` user and serves the React build and API from one port. The image build runs tests, type checking, and the frontend build.

```sh
docker compose logs -f studio
docker compose down
```

Checkpoints are stored at `/app/data/checkpoint.json` in the Compose `checkpoints` volume. The volume survives `docker compose down`; `docker compose down -v` deletes it. Host files in `data/` are not imported automatically.

Without Compose:

```sh
docker build -t seed-studio .
docker run --rm --init -p 127.0.0.1:3001:3001 -v seed-checkpoints:/app/data seed-studio
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

## Training

An example English parent–child dialogue corpus is included in [corpus-examples/parent_child_corpus_en.txt](corpus-examples/parent_child_corpus_en.txt). Import this file in the Training page to experiment with short `User:` / `Assistant:` exchanges.

See the [corpus examples guide](corpus-examples/README.md) for corpus descriptions, suggested training settings, and guidance on monitoring training.

1. Paste or import a UTF-8 `.txt` file: at least 64 Unicode code points, up to 10 MB. The sample is a format example, not enough data for a capable assistant.
2. Choose embedding width, layers, attention heads, context length, batch size, steps, and peak learning rate. Width must be divisible by the number of heads.
3. Start a new model. All weights are initialized randomly. Batched next-token training uses AdamW, gradient clipping, learning-rate warmup, and cosine decay.
4. Compare training and validation loss. The last 10% of encoded tokens are withheld from training. Validation evaluates up to three fixed windows at the first update, every 100 updates, and the last update. The dashed line shows the most recent validation measurement. Increasing validation loss while training loss drops can indicate overfitting.
5. Test prompts and questions you did not train on. Low loss is not evidence of factual accuracy, reasoning, or reliable instruction following.

Default configuration: width 64, two layers, four attention heads, a 64-token context, batch size 2, 1,000 updates, and peak learning rate 0.001. This is about 137,000 trainable parameters. Supported CPU limits are width/context up to 256, up to four layers and eight heads, and batch size up to eight. Even the maximum configuration is only about 3.36 million parameters, not a large production LLM. Larger settings cost substantially more CPU time.

On macOS, run `npm run native:build` and restart the server to enable optional Apple Accelerate CPU acceleration. The portable JavaScript backend remains available. See [training performance](docs/training-performance.md) for setup requirements, benchmarks, correctness checks, and GPU trade-offs.

The split is contiguous and does not deduplicate repeated examples. Repeated or near-identical examples in both partitions can make validation too optimistic. Keep a separate set of genuinely unseen questions for evaluating usefulness. The last partition should be representative of the task you want to measure.

**Train a new model** creates a fresh model in memory with random weights and starts training it on your text using the selected settings. It replaces the current in-memory model and resets its training step count and loss history. It does not create a JSON file or overwrite the saved checkpoint automatically. Click **Save checkpoint** to write the current model to `data/checkpoint.json`, replacing any previous checkpoint. This JSON file is a model snapshot, not a database of your training text.

**Continue existing model** trains further without resetting the learned weights. It retains the loaded architecture, context limit, and optimizer state, and accepts a new corpus. The selected batch size and training length apply to the new session. A resumed session restarts its learning-rate schedule. Architecture controls only affect new models.

There is no pretrained English knowledge. For domain-specific chat, supply many accurate, varied question/answer examples. For broader language ability, a much larger corpus and accelerator-backed training infrastructure would be required. The portable engine uses one CPU thread; the optional Accelerate backend uses native CPU matrix operations. Neither backend uses GPU or distributed training. Individual updates and generation calls run synchronously on the API thread, yielding only between training blocks. Large settings can delay other API requests.

## Chat and text generation

Use this dialogue format for chat training:

```text
User: Hello!
Assistant: Hello! How can I help?

User: What do plants need?
Assistant: Plants need water, light, and suitable soil.
```

Training inserts a special end-of-response token after assistant answers. Generation can stop when the model predicts it; this is learned behavior, not a canned answer. Ordinary text is encoded as UTF-8 bytes. The tokenizer has 258 tokens: 256 bytes, a beginning token, and an end-of-response token. New characters are accepted without rebuilding a vocabulary. An insufficiently trained byte model may emit malformed UTF-8, which is displayed with replacement characters.

Chat sends recent turns in the `User:` / `Assistant:` format. Requests accept up to 41 alternating messages, starting and ending with a user, with up to 2,000 Unicode code points per message. Chat formatting retains up to 4,096 code points of recent complete turns, but the model actually attends only to its configured context window in **bytes**, which may be much shorter. Text generation accepts prompts up to 4,096 code points.

Enter sends a chat message; Shift + Enter inserts a newline. Temperature controls randomness. Top K in Text generation limits sampling to the K most likely tokens; 1 chooses the most likely token and 0 uses all candidates. `maxTokens` limits generated tokens, not visible characters. UTF-8 characters may occupy multiple tokens. Chat also removes any subsequent role marker generated after the assistant reply.

Chat history and drafts remain in browser memory while switching pages and reset on refresh. Sending a message does not train the model.

## New life: teach through chat

Open **New life** below Chat to grow a separate model from random weights. Send a
short message, then select **Teach an answer** below its reply to supply the response
you wanted. For example, send `What is 1 + 1?` and teach `1 plus 1 is 2.` Early replies
will be gibberish; a handful of lessons does not produce a fluent assistant.

### Try a first lesson

1. Send `What color is the sky?` in **Say something simple…**.
2. Select **Teach an answer** below Seed's reply.
3. Enter `The sky is blue.` in **What should Seed say?** and press Enter to teach and save.
4. Ask the question again, then introduce other short questions and answers.

Enter submits either input. After sending a message, focus returns to the message
input, and the conversation follows the latest row automatically. The counters show
remembered lessons, taught answers, and total learning updates.

### Teach with the `!` command

You can also teach directly in **Say something simple…**: prefix your preferred answer
with `!`, for example `!The sky is blue.`. This teaches an answer to the latest message
you sent in the visible conversation, rather than sending a new question. The `!` is
removed before learning and does not count toward the 100-byte limit. Send a message
first; the shortcut needs a previous reply to correct. Both Enter and the send button
support this shortcut, and the input clears and regains focus after saving successfully.

For example, after asking `What color is the sky?`, type `!The sky is blue.` and press
Enter. Seed learns and saves that question-and-answer pair without generating another
reply. Ask the question again when you want to check its progress. The **Teach an
answer** button remains available for correcting any earlier reply in the conversation.

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

The model uses width 32, one layer, four attention heads, and learning rate 0.001.
Its weights, optimizer state, and replay lessons are saved atomically after each
learning action to **`data/new-life-checkpoint.json`**, independently of the studio's
`data/checkpoint.json`. Saved learning reloads automatically after a server restart.
**New conversation** clears the visible conversation while preserving learning.
Conversation display is kept only for the current browser session; saved lessons
include your messages and taught answers. Replies use the current message as context.
This experimental mode can memorize lessons and can still forget earlier skills.

## Checkpoints and storage

The app **automatically saves a checkpoint every 100 training updates**, based on the model's total update count, including across resumed training sessions. Checkpoints are written to `data/checkpoint.json` locally, or `/app/data/checkpoint.json` inside Docker. Each save atomically replaces the single checkpoint file.

Click **Save checkpoint** to save immediately, including updates since the last automatic save. After a restart, click **Load checkpoint** to restore the latest saved model. Updates since the last save are lost if the server stops.

Click **Delete checkpoint** and confirm in the modal to permanently remove the saved checkpoint. Cancel or press Escape to keep it. Deleting keeps the currently loaded model and any training in progress available in memory; you can save the model again. Deleting an already missing checkpoint is harmless.

Version 2 Transformer checkpoints include architecture, tokenizer identity, configuration, weights, optimizer moments, update count, and chart history. They do **not** include the corpus, validation token partition, chat history, or draft. Keep the corpus file separately for future training. Unsaved weights are lost on server shutdown. Click **Load** after restarting to restore the saved model.

Version 1 RNN checkpoints still load as RNNs, including their original optimizer state and vocabulary restrictions. They are labeled as legacy in the interface. Loading one does not convert it into a Transformer. Start a new model to use the new architecture. Validation evaluation is only available for the Transformer.

## Source layout

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

Use Node 24 (`nvm use`) and install dependencies with `npm ci`. Save the checkpoint
you want to test using **Save checkpoint** in the app, then run from the repository root:

```sh
npm run evaluate
```

This asks the saved model all **528 questions** from
`corpus-examples/toddler_reasoning_evaluation.json`. Do not import either evaluation
file as training material. The `.txt` version contains the same questions and reference
answers for manual testing.

For a quicker check, run a smaller sample:

```sh
npm run evaluate -- --limit 20
```

This runs an evenly spaced 20-question sample against `data/checkpoint.json`.
The evaluator loads the saved model in a separate process; it does not train, modify
the checkpoint, or require the API to be running. It evaluates the saved update count,
which may be behind the model currently in memory.

To evaluate a copied checkpoint:

```sh
npm run evaluate -- --checkpoint /path/to/checkpoint.json
```

Each question starts a fresh chat. Defaults are temperature 0.4, top K 1, and a maximum
of 160 generated byte tokens per answer. Top K 1 selects the most likely next token
for repeatable comparisons. To approximate the chat page's temperature sampling,
use `--temperature 0.4 --top-k 0`; those runs can vary. Keep settings and the evaluation
subset unchanged when comparing checkpoints. Run `npm run evaluate -- --help` for options.

The terminal reports numerical accuracy, a pass rate with additional name and equation
checks, and mean, median, and 95th-percentile response times. A JSON report in
`data/evaluations/` includes every question, reference answer, generated reply, scoring
details, incorrect answers, scores by operation, model configuration, and generation
settings. Use `--output /path/to/report.json` to choose a report location; existing
files are not overwritten.

Example terminal output:

```text
Numerical accuracy: 136/528 (25.8%)
Pass with consistency checks: 112/528 (21.2%)
Unrecognized answers: 0 (counted as incorrect)
Response time: mean 703 ms · median 420.5 ms · p95 1314 ms
Report with incorrect answers: .../data/evaluations/step-14900-<unique-id>.json
```

These are example results from a checkpoint at 14,900 updates, not a target score.
Numerical accuracy measures the extracted final number. The consistency pass also
requires no detected name or equation mismatch. An unrecognized answer is one the
scorer cannot extract a number from; zero unrecognized answers does not mean zero
wrong answers. The p95 response time is the time within which approximately 95% of
answers completed.

To compare saved checkpoints, keep the dataset and generation settings the same and
save separate reports:

```sh
npm run evaluate -- --checkpoint /path/to/earlier-checkpoint.json --output data/evaluations/earlier.json
npm run evaluate -- --checkpoint /path/to/later-checkpoint.json --output data/evaluations/later.json
```

Compare the overall scores and `summary.byOperation`, then inspect `incorrectAnswers`
for recurring mistakes. Reports also record model update counts and settings.
`npm test` checks application and corpus correctness; `npm run evaluate` measures the
saved model's answers.

Scoring uses rules tailored to this dataset. It recognizes numeric answers, common
quantity statements, and explicit addition/subtraction equations. Unrecognized answers
count as incorrect and remain in the report for manual review. Numerical accuracy
can credit a correct number even when a name or explanation is wrong; the additional
checks reject detected contradictions. Neither metric proves reasoning, and the
checks cannot recognize every possible phrasing or error. The evaluator does not
test whether this checkpoint previously trained on evaluation material.

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
