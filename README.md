# Seed

**A local language-model training studio.** Train a small model from scratch on your own
text, watch its learning progress, and experiment with its output.

- **Training:** Import text, configure a model, and track training and validation loss.
- **Text generation and Chat:** Try prompts and conversations with your trained model.
- **New life:** Teach a separate model through short messages and corrected answers.
- **Models:** Create checkpoints and switch between experiments.

Seed runs locally with a React/TypeScript frontend, a Node.js 24 API, and a Python/PyTorch
worker. It supports Apple Silicon GPUs (MPS), NVIDIA GPUs (CUDA), and CPUs. No external
inference API is used.

The default model has about **3.36 million parameters** and starts with random weights.
It learns from the text you supply; it has no pretrained English knowledge. Seed is built
for learning and small experiments, rather than broad assistant capabilities.

## Get started

### Run locally

Use Node **24** and Python **3.11+**. Python 3.12 is recommended on macOS; substitute your
installed Python version below if needed.

```sh
nvm use
npm ci
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ./python
npm run dev
```

Open **http://localhost:5173**. PyTorch automatically selects CUDA, then MPS, then CPU.
See [setup and configuration](docs/setup.md) for device selection, production builds,
ports, and origin configuration.

### Run with Docker

```sh
docker compose up --build -d
```

Open **http://localhost:3001**. Docker uses CPU PyTorch; run natively on macOS for MPS
GPU acceleration. Checkpoints persist in a Docker volume. See the
[Docker guide](docs/setup.md#run-with-docker) for logs, storage, and standalone commands.

## Try your first model

1. Open **Training** and import an [included corpus](data/README.md), or paste your own text.
2. Choose your settings and start training. Compare training and validation loss.
3. Try an unseen prompt in **Text generation** or **Chat**. Low loss alone does not establish
   reliable answers.

Training saves checkpoints automatically. Keep your corpus files separately: checkpoints
store learned weights and training state, not the source text. Sending a message in Chat
does not train the model; use **New life** to experiment with teaching through conversation.

## Documentation

| Guide                                       | What you’ll find                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| [Setup and configuration](docs/setup.md)    | Local and Docker setup, GPU selection, ports, and environment settings   |
| [Training and generation](docs/training.md) | Training controls, data formats, chat, CLI training, and evaluation      |
| [New life](docs/new-life.md)                | First lessons, corrections, learning behavior, and saved learning        |
| [Models and storage](docs/models.md)        | Creating and loading models, autosaves, and legacy checkpoints           |
| [PyTorch guide](docs/pytorch.md)            | Backend setup, CLI options, checkpoint conversion, and Python validation |
| [HTTP API](docs/api.md)                     | Routes, request examples, and limits                                     |
| [Efficiency research](research/RESULTS.md)  | Transformer/GRU results, CPU/MPS comparisons, and recommendations        |
| [Development](docs/development.md)          | Source layout, architecture, checks, and contributing                    |

Explore the [full documentation index](docs/README.md) for explanations, corpora, and
experiment guides. See the [release notes](RELEASE_NOTES.md) for recorded changes.

## Screenshot

This screenshot predates the PyTorch backend and the Models panel.

<img width="1896" height="1283" alt="Seed training studio" src="https://github.com/user-attachments/assets/df17f4be-9d47-4991-873b-192fbb95d0c7" />

## License

Seed is licensed under the [MIT License](LICENSE).
