# Validation record

Completed on 2026-10-22 using the repository's existing Node 24 and Python virtual environment.

- `npm run typecheck`: passed.
- `npm test`: 72 passed, zero failures.
- `.venv/bin/python -m unittest discover -s python/tests`: 13 tests run, two MPS tests skipped, zero failures.
- `.venv/bin/python -m unittest discover -s research -p 'test_*.py'`: six passed. Checks task-combination/question separation, whole-story separation, corpus hashes, incremental GRU/full-prefix equivalence, complete finite metrics and accuracy consistency, and inline HTML charts without external assets.
- `.venv/bin/ruff check research` and `.venv/bin/ruff format --check research`: passed.
- `npm run format:check`: passed after formatting generated reports.
- Both experiment orchestrators exited successfully. Eight retained training runs and four post-training inference variants completed. The raw stderr files retain the QNNPACK reduce_range warning.
- Checkpoints are local, under Git-ignored research checkpoint directories. No studio/API code was changed.

The HTML was checked structurally, not with a browser screenshot. Energy measurement and MPS execution were unavailable and are not claimed. Timing repeats measure local variability, not hardware-independent performance. Known concurrent repository tests were completed before the retained scale-check timing rerun.

## MPS replication validation

The earlier availability restriction applies only to the sandbox. Actual MPS experiments ran outside it and completed successfully: eight additional training runs and four additional inference variants, with original CPU artifacts preserved.

- Research suite: eight tests passed, including identical CPU/MPS corpus hashes, all sixteen retained training outputs, variant device labels, CPU float32 controls for int8, MPS memory snapshots, and the cross-device accuracy check.
- Python regression suite outside the sandbox: 13 tests run, one CUDA test skipped, zero failures. The MPS parity/update/save/resume test ran successfully. See `mps-python-tests.txt`.
- Node suite: 72 passed, zero failures. See `node-tests.txt`.
- Type checking and Ruff lint/format checks passed; generated reports were formatted with Prettier.
- Four cross-device evaluations confirmed that large-Transformer accuracy follows checkpoint weights rather than the inference device.
- No energy measurement was performed. CPU fallback was not enabled. MPS allocation measurements are post-benchmark snapshots, not peak GPU memory.
