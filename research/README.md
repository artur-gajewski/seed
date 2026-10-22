# Seed learning-efficiency research

Open `report.html` directly in a browser. It contains inline charts, measured results, methodology, limitations, and embedded raw run data; it requires no server or network connection.

- `PLAN.md`: experiment protocol and the explicitly exploratory scale-check addition.
- `experiment.py`: generate corpora and run six CPU training jobs plus two inference interventions.
- `mps_run.py`: repeat all eight training jobs on MPS outside the sandbox, storing results under `mps/`.
- `cross_device.py`: evaluate the two large Transformer checkpoints on CPU and MPS to validate the accuracy gap.
- `mps_report.py`: render CPU/MPS comparisons in the standalone report.
- `scale_check.py`: run the additional two original-size training jobs and two interventions.
- `report.py`: regenerate the single-file report from recorded results.
- `corpora.json`: source hashes, split pairs, corpus counts and file hashes.
- `tasks-*`, `prose-*`: actual training, validation and test corpora.
- `run-*.json`, `summary.json`: small-model metrics, learning curves and timing repeats.
- `answers-*.json`: all greedy test answers, including failures.
- `full-scale/`: separate original-size experiment artifacts.
- `checkpoints/`: local trained inference states (Git-ignored); optimizer state is not retained.

From the repository root, using the existing PyTorch environment:

```sh
.venv/bin/python research/experiment.py
.venv/bin/python research/scale_check.py
.venv/bin/python research/mps_run.py
.venv/bin/python research/cross_device.py
.venv/bin/python research/report.py
.venv/bin/python -m unittest discover -s research -p 'test_*.py'
```

Runs overwrite their own artifacts. The study includes CPU and MPS runs with four PyTorch CPU threads on macOS ARM64. MPS is available outside the execution sandbox; the initial false availability flag was caused by sandbox restrictions. The MPS orchestrator must run outside that restriction. CPU outputs are preserved; GPU outputs use `mps/`. Actual energy consumption was not measured. The scale check holds context at 128 bytes, rather than the original 256, and uses only one seed.

Initial smoke-run outputs were discarded after correcting an ambiguous distractor template and selecting the available QNNPACK quantization backend. The retained pilot was rerun from scratch with the corrected corpus. The scale-check timing run was repeated after repository tests finished to avoid that known competing workload. The script does not control unrelated host activity or thermals.

The prose source is the repository's `data/inputs/short_stories.txt`; generated tasks use no external data. Source text is separate from model checkpoints. The work changes no studio or API behavior.
