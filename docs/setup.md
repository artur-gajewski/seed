# Setup and configuration

[Documentation](README.md) · [Seed](../README.md)

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

### Initial settings and saved preferences

Training starts with the supported maximum settings: width 1024, twelve layers,
sixteen heads, batch eight, context 2048, 1,000,000 updates, and learning rate 0.05.
Text generation starts at 1000 new tokens, temperature 2, and Top K 512. New Life
starts at 153M when no preset selection is saved. Choose smaller settings in the UI
for smaller experiments; see [Training](training.md) and [New Life](new-life.md).

The browser remembers Training settings/device, Text generation settings/prompt, and
the New Life preset. Preferences belong to that browser origin: development on port
5173 and production on port 3001 have separate storage. Clearing site storage restores
the initial defaults. Checkpoints are stored separately on the server. Explicitly
loading a checkpoint updates Training controls; polling does not overwrite preferences.

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
