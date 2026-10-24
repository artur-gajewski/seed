# HTTP API

[Documentation](README.md) · [Seed](../README.md)

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

## Training limits and defaults

The browser’s maximum defaults and saved preferences are separate from the API’s
fallbacks for omitted fields. Explicit request values take precedence.

| Field            | API range                     | API fallback | Browser default |
| ---------------- | ----------------------------- | ------------ | --------------- |
| `hiddenSize`     | 16–1024, divisible by `heads` | 64           | 1024            |
| `layers`         | 1–12                          | 2            | 12              |
| `heads`          | 1–16                          | 4            | 16              |
| `sequenceLength` | 8–2048 UTF-8 byte tokens      | 64           | 2048            |
| `batchSize`      | 1–8                           | 2            | 8               |
| `steps`          | 1–1,000,000                   | 1000         | 1,000,000       |
| `learningRate`   | 0.00001–0.05                  | 0.001        | 0.05            |

Integer fields must contain integers. Training text must contain at least 64 Unicode
code points and at most 10,000,000 UTF-8 bytes. `resume` retains the loaded model’s
architecture and context capacity. Larger context settings require a fresh model.

Create a named, untrained model without a corpus (this example deliberately uses a
smaller configuration than the browser defaults):

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

The chat response contains `reply`, `tokens`, `elapsedMs`, and `modelStep`. The generate route takes `prompt` instead of `messages` and returns `completion` instead of `reply`. Token counts include a predicted stop token and are measured before chat trims subsequent role markers. Temperature is 0.1–2; output limits are 1–1,000 tokens, and Top K is 0–512
(0 uses all candidates). Text generation accepts prompts up to 4096 Unicode code
points. Omitted generation fields default to 200 tokens, temperature 0.8, and Top K 0.
The Text generation UI instead defaults to 1000 tokens, temperature 2, and Top K 512,
and remembers its settings and prompt in browser storage. Chat uses a full-width,
single-line input; its request limits remain 41 alternating messages and 2000 code
points per message.

## New Life

New Life message requests take `{ "size": "x-large", "message": "What color is the sky?" }`
and return `reply` and `status`. Teaching requests take
`{ "size": "x-large", "message": "What color is the sky?", "answer": "The sky is blue." }`
and return the updated status. Status includes `ready`, `steps`, `lessons`, and
`corrections`. Requests wait for learning and saving to finish; concurrent operations
for the same preset are rejected while it is busy.

Valid `size` values are `small`, `small-plus`, `medium`, `medium-plus`, `large`,
`large-plus`, `huge`, and `x-large`. See [New Life presets](new-life.md#continuing-later)
for their architectures and parameter labels. The browser defaults to `x-large` and
remembers its selection. API calls without `size`, including `GET /api/new-life/status`,
use `small`; use `POST /api/new-life/settings` with `{ "size": "x-large" }` to retrieve
that preset’s status. Selection does not create weights or train a model.

Messages allow `contextSize - 48` UTF-8 bytes: 208 for `small` through 2000 for
`x-large`. Answers allow the smaller of that limit and
`contextSize - 20 - messageByteLength`. Byte lengths exclude surrounding whitespace.
For example, a 2000-byte message leaves 28 answer bytes in `x-large`. Empty values,
newlines, and embedded `User:` or `Assistant:` labels are rejected. Teaching prefixes
such as `!` are handled by the UI; API clients send the extracted answer directly.
Each preset retains up to 1000 lessons in its own checkpoint.
