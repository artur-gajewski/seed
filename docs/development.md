# Development and contributing

[Documentation](README.md) · [Seed](../README.md)

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
- `shared/api.ts`: shared request/response types, eight New Life presets, and context-aware byte budgets.
- `src/lib/trainingSettings.ts`: maximum browser training defaults and checkpoint-prefill behavior.
- `src/lib/studioPreferences.ts`: generation defaults, preference validation, and browser storage access.
- `src/hooks/useStoredState.ts`: restore and persist valid Training/generation settings.
- `src/lib/newLifeInput.ts`: correction parsing and model-aware message/answer validation.

Each Transformer block uses pre-layer normalization, masked multi-head self-attention, residual connections, and a GELU feed-forward network with a 4× expansion. Learned token and position embeddings precede the blocks; final normalization and an output projection predict the next byte. The API uses batched Float32 PyTorch tensors and automatic differentiation. The reference engine retains Float64 buffers and manual gradients for parity tests. AdamW uses global gradient clipping at norm 1 and matrix weight decay 0.01.

Architecture reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This implementation uses decoder-only causal attention rather than the paper's complete encoder-decoder architecture. Assistant behavior additionally depends on training objectives and demonstrations; see [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155). No RLHF is implemented here.

## Checks and formatting

```sh
npm test
.venv/bin/python -m unittest discover -s python/tests -v
npm run build
npm run format:check
npm run format
```

The build runs strict TypeScript checking before Vite. Tests cover numerical derivatives for every Transformer parameter group, causal masking, pattern learning from random weights, UTF-8 encoding, exact checkpoint resumption, chat formatting, validation measurement, origin handling, and the API lifecycle. Tests also cover 2048-token model/checkpoint limits, preference persistence and corrupt
storage, and New Life message/answer budgets and restoration. Legacy RNN gradient
and checkpoint tests remain. These tests establish implementation behavior, not general assistant capability. There are no browser interaction tests.

Prettier uses two-space indentation, single quotes, semicolons, trailing commas, and a 100-character line width. Keep blank lines between functions and logical sections, one variable declaration per statement, and braces around conditionals and loops.

## Contributing

Contributions are welcome. Open an issue to discuss a bug or idea, or submit a pull request with a focused change and a description of how you tested it. Run `npm test`, the Python suite, `npm run build`, and `npm run format:check` before submitting code changes. See the [PyTorch guide](pytorch.md#validation) for Python formatting and lint checks.
